// 圆圈标记（圆点）为什么不在模型里？
//
// 现象（out/trace/edgeleak_compare.png）：0109_RT 上，圆圈标记整圈是红色
// ——信号里有、模型里没有。而其他字形是黄色（两边都有）。
// 它不是划痕，是印在芯片上的正常标记（`7` 与 `HU` 的圆环）。
//
// 训练（model.cpp:157-171）：
//   mask = signal > strokeHi(0.35)     逐张二值化
//   count += mask ; freq = count/used * 255
// 于是 freq 的含义是"该像素在多少比例的样本里亮过"。
// 推断：圆圈标记只在部分样本上可见（刻印深浅/方向/打标位置差异），
// 出现率低于 blankFreq=0.15 → 被 findScratches 当成"模型里没有墨"。
//
// 本工具直接回答两个问题：
//   1) 圆圈标记在各张 OK 图上到底亮不亮？出现率是多少？
//   2) 模型里这块区域的 freq 实际值是多少？是否真的低于阈值？
//
// 用法: markloc/dotstat <modelDir> <imageDir> [x y w h]
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
    if (argc < 3) {
        std::printf("用法: markloc_dotstat <modelDir> <imageDir> [x y w h]\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;

    // 圆圈标记区域：默认按 0109 上看到的范围给一个估计框
    cv::Rect roi(474, 500, 160, 170);
    if (argc >= 7) roi = cv::Rect(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
                                  std::atoi(argv[5]));

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(fs::path(std::u8string(
             reinterpret_cast<const char8_t*>(argv[2]), std::strlen(argv[2]))))) {
        if (!e.is_regular_file()) continue;
        const std::string stem = e.path().stem().string();
        if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
        files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    std::printf("圆圈标记区域 (%d,%d %dx%d)\n", roi.x, roi.y, roi.width, roi.height);
    std::printf("strokeHi=%.2f  blankFreq=%.3f (阈值 %d/255)\n", cfg.strokeHi, cfg.blankFreq,
                static_cast<int>(cfg.blankFreq * 255));

    // 模型在该区的 freq 分布
    const cv::Rect rr = roi & cv::Rect(0, 0, model.freq.cols, model.freq.rows);
    {
        const cv::Mat f = model.freq(rr);
        int mx = 0, nAbove = 0;
        double sum = 0;
        for (int y = 0; y < f.rows; ++y)
            for (int x = 0; x < f.cols; ++x) {
                const int v = f.at<uchar>(y, x);
                mx = std::max(mx, v);
                sum += v;
                if (v > cfg.blankFreq * 255) ++nAbove;
            }
        std::printf("\n模型 freq: 最大=%d 均值=%.2f 超过阈值像素=%d/%d (%.2f%%)\n", mx,
                    sum / rr.area(), nAbove, rr.area(), 100.0 * nAbove / rr.area());
        std::printf("  → 若最大只有 %d，说明这块在模型里**整体不存在**\n", mx);
    }

    // 逐图统计：该区信号 > strokeHi 的像素数
    std::printf("\n%-24s %10s %10s %8s\n", "文件", "亮像素", "占区域", "p99");
    std::vector<double> ratios;
    int nBright = 0;
    for (const auto& path : files) {
        cv::Mat gray;
        if (!readGray(path, gray) || gray.size() != model.freq.size()) continue;
        const cv::Mat sig =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        const cv::Mat s = sig(rr);
        int n = 0;
        std::vector<float> vals;
        for (int y = 0; y < s.rows; ++y)
            for (int x = 0; x < s.cols; ++x) {
                const float v = s.at<float>(y, x);
                if (v > cfg.strokeHi) ++n;
                vals.push_back(v);
            }
        std::sort(vals.begin(), vals.end());
        const double p99 = vals[static_cast<size_t>(vals.size() * 0.99)];
        const double ratio = 100.0 * n / rr.area();
        ratios.push_back(ratio);
        if (n > 100) ++nBright;
        std::string bare = fs::path(path).stem().string();
        if (bare.size() > 22) bare = bare.substr(0, 22);
        std::printf("%-24s %10d %9.2f%% %8.3f\n", bare.c_str(), n, ratio, p99);
    }
    std::sort(ratios.begin(), ratios.end());
    std::printf("\n=== 汇总 ===\n");
    if (!ratios.empty()) {
        std::printf("亮像素占比: 最小=%.2f%% 中位=%.2f%% 最大=%.2f%%\n", ratios.front(),
                    ratios[ratios.size() / 2], ratios.back());
        std::printf("其中亮像素>100 的图: %d / %zu\n", nBright, ratios.size());
        std::printf("→ 这就是 freq 里该区的上限：%d/%zu ≈ %.0f%%\n", nBright, ratios.size(),
                    100.0 * nBright / ratios.size());
    }
    return 0;
}
