// 归一化刻度的数据推导
//
// 背景：computeSignal 用 scale = max(8, q995*0.55 + q90*0.45) 做归一化。
// q995 是该图最亮的 0.5% 像素（反光/管脚/芯片边缘），用它当刻度会把
// 笔画压到 0.3 量级，反光推到 1.0，导致固定 strokeHi=0.35 等于"全部算墨"。
//
// 本工具直接复刻 computeSignal 的残差 d，然后对比不同 scale 公式下：
//   - 字符框内笔画的核心亮度（p99）
//   - 背景的亮度水平（p50）
// 目标是让"笔画 / 背景"的对比度尽量大，这样 strokeHi 才有绝对意义。
//
// 用法: markloc/scalenorm <modelDir> <imageDir>
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

static double pctOf(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t k = static_cast<size_t>(q * (v.size() - 1));
    return v[k];
}

// 复刻 computeSignal 的残差（不含归一化）
static cv::Mat rawResidual(const cv::Mat& gray, double sigma) {
    cv::Mat p;
    gray.convertTo(p, CV_32F);
    cv::Mat bg;
    cv::GaussianBlur(p, bg, cv::Size(0, 0), sigma);
    cv::Mat d;
    cv::subtract(p, bg, d);
    cv::threshold(d, d, 0.0, 0.0, cv::THRESH_TOZERO);
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_scalenorm <modelDir> <imageDir>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    cv::Mat boxMask = cv::Mat::zeros(model.freq.size(), CV_8U);
    for (const auto& u : base) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, boxMask.cols, boxMask.rows);
        if (b.width > 0 && b.height > 0) boxMask(b).setTo(255);
    }

    std::vector<std::string> files;
    {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(u8p(argv[2]), ec)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext != ".tif" && ext != ".tiff" && ext != ".png" && ext != ".bmp") continue;
            const std::string stem = e.path().stem().string();
            if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
            files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    }

    struct Formula {
        const char* name;
        double q995w;  // q995 的权重
        double q90w;   // q90 的权重
    };
    const std::vector<Formula> formulas = {
        {"当前: q995*0.55+q90*0.45", 0.55, 0.45},
        {"q90 单独",                   0.0,  1.0},
        {"q95 单独",                   0.0,  1.0},  // 占位，下面单独算
        {"q995*0.30+q90*0.70",         0.30, 0.70},
        {"q995*0.20+q90*0.80",         0.20, 0.80},
    };

    // 每个公式下，跨 84 张图统计：box 内 p99（笔画）和 p50（背景）的 scale 归一化后值
    std::vector<std::vector<double>> sigP99(5), sigP50(5), contrast(5);

    int used = 0;
    for (const auto& f : files) {
        cv::Mat g;
        if (!readGray(f, g) || g.size() != model.freq.size()) continue;
        const cv::Mat d = rawResidual(g, cfg.sigma);

        // 残差直方图求分位（与 computeSignal 同口径）
        std::vector<int> hist(512, 0);
        for (int y = 0; y < d.rows; ++y) {
            const float* row = d.ptr<float>(y);
            for (int x = 0; x < d.cols; ++x) {
                int b = std::clamp(static_cast<int>(std::lround(row[x])) + 256, 0, 511);
                ++hist[static_cast<size_t>(b)];
            }
        }
        const long total = static_cast<long>(d.rows) * d.cols;
        auto quantile = [&](double q) {
            const long target = static_cast<long>(total * q);
            long a = 0;
            for (int b = 0; b < 512; ++b) {
                a += hist[static_cast<size_t>(b)];
                if (a >= target) return static_cast<double>(b - 256);
            }
            return 255.0;
        };
        const double q995 = quantile(0.995);
        const double q95 = quantile(0.95);
        const double q90 = quantile(0.90);

        // 字符框内残差
        std::vector<float> boxVals;
        for (int y = 0; y < d.rows; ++y) {
            const float* dp = d.ptr<float>(y);
            const uchar* mp = boxMask.ptr<uchar>(y);
            for (int x = 0; x < d.cols; ++x)
                if (mp[x]) boxVals.push_back(dp[x]);
        }
        if (boxVals.empty()) continue;

        auto boxPct = [&](double q) {
            const size_t k = static_cast<size_t>(q * (boxVals.size() - 1));
            std::vector<float> t = boxVals;
            std::nth_element(t.begin(), t.begin() + k, t.end());
            return static_cast<double>(t[k]);
        };
        const double boxP99 = boxPct(0.99);
        const double boxP50 = boxPct(0.50);

        const double scales[5] = {
            std::max(8.0, q995 * 0.55 + q90 * 0.45),
            std::max(8.0, q90),
            std::max(8.0, q95),
            std::max(8.0, q995 * 0.30 + q90 * 0.70),
            std::max(8.0, q995 * 0.20 + q90 * 0.80),
        };
        for (int i = 0; i < 5; ++i) {
            const double sp99 = boxP99 / scales[i];
            const double sp50 = boxP50 / scales[i];
            sigP99[i].push_back(sp99);
            sigP50[i].push_back(sp50);
            contrast[i].push_back(sp99 / std::max(1e-6, sp50));
        }
        ++used;
    }

    std::printf("样本 %d 张（仅 v00 视角）\n\n", used);
    std::printf("%-26s %8s %8s %8s   %8s\n", "归一化公式", "笔画p99", "背景p50", "对比度",
                "p99/8阈值");
    std::printf("%-26s %8s %8s %8s   %8s\n", "", "中位数", "中位数", "中位数", "倍数");
    for (int i = 0; i < 5; ++i) {
        const double m99 = pctOf(sigP99[i], 0.5);
        const double m50 = pctOf(sigP50[i], 0.5);
        const double mc = pctOf(contrast[i], 0.5);
        std::printf("%-26s %8.3f %8.3f %8.2f   %8.2f\n", formulas[i].name, m99, m50, mc,
                    m99 / 0.35);
    }

    std::printf("\n=== 关键判据 ===\n");
    std::printf("目标：笔画 p99 要明显大于 0.35（现在的 strokeHi），背景 p50 要明显小于 0.35。\n");
    std::printf("当前公式下笔画 p99 中位数只有 0.3 左右，与 0.35 几乎重合 → 阈值形同虚设。\n");
    for (int i = 0; i < 5; ++i) {
        const double m99min = *std::min_element(sigP99[i].begin(), sigP99[i].end());
        const double m99max = *std::max_element(sigP99[i].begin(), sigP99[i].end());
        std::printf("  %-26s 笔画p99 全距 [%.3f, %.3f]\n", formulas[i].name, m99min, m99max);
    }
    return 0;
}
