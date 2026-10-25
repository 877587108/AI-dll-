// 模型在 N 的对角笔画上为什么是 0？
//
// 反查结果（out/trace_line.txt）：
//   S=1938 那条线在 t=56..128 有连续 73px 落在"signal 0.2~1.5、freq 全 0"上。
//   放大原图（out/trace/zoom_leakseg.png）看，这段正压在字母 N 的对角笔画上
//   —— 那是真实字形，不是划痕。
//
// 所以问题变成：真实存在的笔画，为什么在 freq 里是 0？
//
// 训练（model.cpp:157-163）：
//   逐张 mask = signal > strokeHi(0.35)，count += mask，freq = count/used*255
// 若某张图上 N 的对角笔画 signal < 0.35，它在这张的 mask 里就是 0。
// 若 84 张里都低于 0.35，freq 就整体为 0。
//
// 本工具量三件事：
//   1) N 对角笔画沿线（从一条实际的高亮线取样）在各张 OK 图上的 signal 值分布
//   2) 同一位置 strokeHi 的命中率 —— 也就是 freq 的理论值
//   3) 剖面：绕笔画中轴取垂线，看 signal 峰值与 background 噪声的比
//
// 用法: markloc/strokefreq <modelDir> <imageDir> <x1> <y1> <x2> <y2>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

namespace fs = std::filesystem;

static bool readGray(const std::string& path, cv::Mat& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::vector<unsigned char> buf;
    unsigned char tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof(tmp), fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(fp);
    if (buf.empty()) return false;
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::printf("用法: markloc_strokefreq <modelDir> <imageDir> <x1> <y1> <x2> <y2>\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    const cv::Point p1(std::atoi(argv[3]), std::atoi(argv[4]));
    const cv::Point p2(std::atoi(argv[5]), std::atoi(argv[6]));

    std::vector<std::string> files;
    const std::string dir = argv[2];
    for (const auto& e : fs::directory_iterator(
             fs::path(std::u8string(reinterpret_cast<const char8_t*>(dir.data()), dir.size())))) {
        if (!e.is_regular_file()) continue;
        const std::string stem = e.path().stem().string();
        if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
        files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    std::printf("笔画段 (%d,%d)-(%d,%d)\n", p1.x, p1.y, p2.x, p2.y);
    std::printf("strokeHi=%.2f\n\n", cfg.strokeHi);

    // 逐图统计该线段上方 strokeHi 的采样点比例
    std::printf("%-24s %10s %10s %8s\n", "文件", "过阈值点", "占比", "沿线均值");
    std::vector<double> ratios;
    for (const auto& path : files) {
        cv::Mat gray;
        if (!readGray(path, gray) || gray.size() != model.freq.size()) continue;
        const cv::Mat sig =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        const double len = cv::norm(p2 - p1);
        const int steps = std::max(2, static_cast<int>(len));
        int over = 0, n = 0;
        double sum = 0;
        for (int t = 0; t <= steps; ++t) {
            const double rr = static_cast<double>(t) / steps;
            const int x = static_cast<int>(p1.x + (p2.x - p1.x) * rr);
            const int y = static_cast<int>(p1.y + (p2.y - p1.y) * rr);
            if (x < 0 || y < 0 || x >= sig.cols || y >= sig.rows) continue;
            // 取垂线 ±3px 内的最大值，抵消亚像素抖动
            const double ux = (p2.x - p1.x) / len, uy = (p2.y - p1.y) / len;
            float mx = 0;
            for (int k = -3; k <= 3; ++k) {
                const int xi = static_cast<int>(std::lround(x - uy * k));
                const int yi = static_cast<int>(std::lround(y + ux * k));
                if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
                mx = std::max(mx, sig.at<float>(yi, xi));
            }
            if (mx > cfg.strokeHi) ++over;
            sum += mx;
            ++n;
        }
        const double r = n > 0 ? 100.0 * over / n : 0.0;
        ratios.push_back(r);
        std::string bare = fs::path(path).stem().string();
        if (bare.size() > 22) bare = bare.substr(0, 22);
        std::printf("%-24s %10d %9.1f%% %8.3f\n", bare.c_str(), over, r, n > 0 ? sum / n : 0.0);
    }
    std::sort(ratios.begin(), ratios.end());
    std::printf("\n=== 汇总（%zu 张）===\n", ratios.size());
    if (!ratios.empty()) {
        std::printf("过阈值占比: 最小=%.1f%% 中位=%.1f%% 最大=%.1f%%\n", ratios.front(),
                    ratios[ratios.size() / 2], ratios.back());
        int above50 = 0;
        for (double r : ratios)
            if (r > 50) ++above50;
        std::printf("占比>50%% 的图: %d/%zu  → 预期 freq 约 %d/%zu\n", above50, ratios.size(),
                    above50, ratios.size());
    }

    // 沿线的 freq 实测值
    std::printf("\n=== 模型 freq 沿该线段 ===\n");
    const double len = cv::norm(p2 - p1);
    const int steps = std::max(2, static_cast<int>(len));
    int zero = 0;
    for (int t = 0; t <= steps; t += 4) {
        const double rr = static_cast<double>(t) / steps;
        const int x = static_cast<int>(p1.x + (p2.x - p1.x) * rr);
        const int y = static_cast<int>(p1.y + (p2.y - p1.y) * rr);
        if (x < 0 || y < 0 || x >= model.freq.cols || y >= model.freq.rows) continue;
        const int f = model.freq.at<uchar>(y, x);
        if (f == 0) ++zero;
        std::printf("  t=%-4d (%d,%d) freq=%d\n", t, x, y, f);
    }
    return 0;
}
