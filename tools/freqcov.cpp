// 模型里各区域的真实墨量：把"扣模型"这一步的前提量出来
//
// 背景：findScratches 的扣模型步骤是
//   modelInk = freq > blankFreq*255 (38)   → 也就是"41 张以上都亮的像素"
//   weak &= ~dilate(modelInk, 9x9)
// 也就是说：freq 不到 38 的像素，被当成"模型里没有墨"。
// 如果某个真实字形/logo 在 freq 里常年低于 38，那块区域就永远不被扣除，
// 信号里真实存在的笔画会 100% 变成 leak。这不是调参问题，是掩膜覆盖面问题。
//
// 本工具对每个 region 直接统计 freq 的分布，并给出"按 blankFreq 阈值
// 算进模型墨的比例"，以及"如果放宽到 X 才能覆盖 90%/95% 的墨"。
//
// 用法: markloc/freqcov <modelDir> [imageDir]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

namespace fs = std::filesystem;

static fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

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
    if (argc < 2) {
        std::printf("用法: markloc_freqcov <modelDir> [imageDir]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    const int inkThr = static_cast<int>(cfg.blankFreq * 255);
    std::printf("模型: freq %dx%d   blankFreq=%.3f → 墨阈值 %d\n", model.freq.cols, model.freq.rows,
                cfg.blankFreq, inkThr);

    // 若有图片目录，顺便统计信号侧的真实笔画分布，做对照
    std::vector<std::string> files;
    if (argc >= 3) {
        for (const auto& e : fs::directory_iterator(u8p(argv[2]))) {
            if (!e.is_regular_file()) continue;
            const std::string stem = e.path().stem().string();
            if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
            files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    }

    // 每个 region 一个统计块
    for (size_t i = 0; i < model.regions.size(); ++i) {
        const auto& rg = model.regions[i];
        const cv::Rect r = rg.rect & cv::Rect(0, 0, model.freq.cols, model.freq.rows);
        if (r.area() <= 0) continue;
        const cv::Mat f = model.freq(r);

        // freq 值直方图
        int hist[256] = {0};
        for (int y = 0; y < f.rows; ++y)
            for (int x = 0; x < f.cols; ++x) ++hist[f.at<uchar>(y, x)];
        const int total = f.rows * f.cols;

        double p50 = 0, p90 = 0, p95 = 0, p99 = 0;
        int acc = 0;
        for (int v = 0; v < 256; ++v) {
            acc += hist[v];
            if (!p50 && acc >= total * 0.50) p50 = v;
            if (!p90 && acc >= total * 0.90) p90 = v;
            if (!p95 && acc >= total * 0.95) p95 = v;
            if (!p99 && acc >= total * 0.99) p99 = v;
        }
        int covered = 0;
        for (int v = inkThr + 1; v < 256; ++v) covered += hist[v];

        std::printf("\n=== region %zu  框=(%d,%d,%dx%d)  字符数=%zu ===\n", i, r.x, r.y, r.width,
                    r.height, rg.chars.size());
        std::printf("freq 分位: p50=%.0f p90=%.0f p95=%.0f p99=%.0f\n", p50, p90, p95, p99);
        std::printf("被算进模型墨(>%d): %d / %d = %.1f%%\n", inkThr, covered, total,
                    100.0 * covered / total);

        // 若某处 freq 只有 20~37，它达不到阈值，但那确实是墨
        int band[4] = {0};   // [0,12) [12,25) [25,38) [38,255]
        for (int v = 0; v < 256; ++v) {
            if (v < 12) band[0] += hist[v];
            else if (v < 25) band[1] += hist[v];
            else if (v < inkThr + 1) band[2] += hist[v];
            else band[3] += hist[v];
        }
        std::printf("频段分布: [0,12)=%.1f%%  [12,25)=%.1f%%  [25,%d)=%.1f%%  [%d,255]=%.1f%%\n",
                    100.0 * band[0] / total, 100.0 * band[1] / total, inkThr + 1,
                    100.0 * band[2] / total, inkThr + 1, 100.0 * band[3] / total);

        // 信号侧：这个框里真实笔画有多亮（取盒子区域，避免外圈背景）
        if (!files.empty()) {
            std::vector<double> sigP90;
            for (const auto& path : files) {
                cv::Mat gray;
                if (!readGray(path, gray) || gray.size() != model.freq.size()) continue;
                const cv::Mat sig =
                    chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
                // 框内取 p90：真实笔画应该远高于 strokeHi=0.35
                const cv::Mat sb = sig(r);
                std::vector<float> vals;
                vals.assign(reinterpret_cast<const float*>(sb.data),
                            reinterpret_cast<const float*>(sb.data) + sb.total());
                std::sort(vals.begin(), vals.end());
                sigP90.push_back(vals[static_cast<size_t>(vals.size() * 0.90)]);
            }
            if (!sigP90.empty()) {
                std::sort(sigP90.begin(), sigP90.end());
                std::printf("信号 p90 (%zu 张): 中位=%.3f 最小=%.3f 最大=%.3f   [strokeHi=%.2f]\n",
                            sigP90.size(), sigP90[sigP90.size() / 2], sigP90.front(),
                            sigP90.back(), cfg.strokeHi);
            }
        }
    }

    // 另报：region3 里的字（bbox=(512,345,279,177)）在 freq 里的实际轮廓
    std::printf("\n=== 逐字符框 freq 覆盖率 ===\n");
    for (size_t i = 0; i < model.regions.size(); ++i) {
        for (size_t c = 0; c < model.regions[i].chars.size(); ++c) {
            const cv::Rect b =
                model.regions[i].chars[c].bbox & cv::Rect(0, 0, model.freq.cols, model.freq.rows);
            if (b.area() <= 0) continue;
            const cv::Mat f = model.freq(b);
            const int cov = cv::countNonZero(f > inkThr);
            std::printf("  r%zu c%-2zu (%4d,%4d %3dx%3d)  模型墨占比 %5.1f%%\n", i, c, b.x, b.y,
                        b.width, b.height, 100.0 * cov / b.area());
        }
    }
    return 0;
}
