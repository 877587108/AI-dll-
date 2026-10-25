// 上下界的数据推导
//
// 目的：不预设 strokeHi 的下界/上界公式，让数据自己给出安全区间。
//
// 核心约束：自适应阈值 trf = clamp(p90, LO, HI) 必须满足
//   (1) 不把字切碎  → trf 不能高于"笔画核心亮度"的下限
//   (2) 不能把噪声当墨 → trf 不能低于"背景噪声"的上限
//
//   "笔画核心亮度" = p99（该图最亮的那批像素 = 笔画中心）
//   "背景噪声上限" = p90 在当前这批图上就是安全的（因为现在的 0.35 判出的模型是糊的，
//                   说明 0.35 太低；真实笔画亮度是 1.1~1.4）
//
// 本工具输出：
//   每张图的 p90 / p95 / p99 / 笔画占比，以及把这些量按全量统计后的
//   各个候选下界/上界，并报告"如果取这个界，多少张图会被夹住"。
//
// 用法: markloc_boundmgr <modelDir> <imageDir>
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

static double pct(const std::vector<float>& v, double q) {
    if (v.empty()) return 0.0;
    const size_t k = static_cast<size_t>(q * (v.size() - 1));
    std::vector<float> t = v;
    std::nth_element(t.begin(), t.begin() + k, t.end());
    return t[k];
}

static double pctOf(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t k = static_cast<size_t>(q * (v.size() - 1));
    return v[k];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_boundmgr <modelDir> <imageDir>\n");
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
    const int boxPixels = cv::countNonZero(boxMask);

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

    std::vector<double> allP90, allP95, allP99, allInkRatio;
    for (const auto& f : files) {
        cv::Mat g;
        if (!readGray(f, g) || g.size() != model.freq.size()) continue;
        const cv::Mat s = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
        std::vector<float> v;
        v.reserve(static_cast<size_t>(boxPixels));
        for (int y = 0; y < s.rows; ++y) {
            const float* sp = s.ptr<float>(y);
            const uchar* mp = boxMask.ptr<uchar>(y);
            for (int x = 0; x < s.cols; ++x)
                if (mp[x]) v.push_back(sp[x]);
        }
        if (v.empty()) continue;
        const double p90 = pct(v, 0.90);
        allP90.push_back(p90);
        allP95.push_back(pct(v, 0.95));
        allP99.push_back(pct(v, 0.99));
        // 笔画像素占比：信号超过当前 p90 的像素比例
        allInkRatio.push_back(0.10);  // p90 定义即 10%
    }

    if (allP90.empty()) {
        std::printf("没有可用样本\n");
        return 1;
    }

    std::sort(allP90.begin(), allP90.end());
    std::sort(allP95.begin(), allP95.end());
    std::sort(allP99.begin(), allP99.end());
    const double p90min = allP90.front(), p90max = allP90.back();
    const double p99min = allP99.front(), p99max = allP99.back();

    std::printf("样本 %zu 张（仅 v00 视角）\n\n", allP90.size());
    std::printf("%-12s %8s %8s %8s %8s\n", "统计量", "min", "p50", "p99", "max");
    std::printf("%-12s %8.4f %8.4f %8.4f %8.4f\n", "p90", p90min, pctOf(allP90, 0.5),
                pctOf(allP90, 0.99), p90max);
    std::printf("%-12s %8.4f %8.4f %8.4f %8.4f\n", "p95", allP95.front(), pctOf(allP95, 0.5),
                pctOf(allP95, 0.99), allP95.back());
    std::printf("%-12s %8.4f %8.4f %8.4f %8.4f\n", "p99", p99min, pctOf(allP99, 0.5),
                pctOf(allP99, 0.99), p99max);

    // ---- 候选下界：必须低于所有图的 p90，否则会夹住正常图 ----
    std::printf("\n=== 候选下界（要求 < 最小 p90 = %.4f，否则夹住正常图）===\n", p90min);
    struct Cand {
        const char* name;
        double value;
        const char* basis;
    };
    std::vector<Cand> lo = {
        {"p99_min * 0.50", p99min * 0.50, "最暗图笔画核心的一半"},
        {"p99_min * 0.60", p99min * 0.60, ""},
        {"p99_min * 0.70", p99min * 0.70, ""},
        {"p99_p50 * 0.50", pctOf(allP99, 0.5) * 0.50, "典型图笔画核心的一半"},
        {"p90_min * 0.70", p90min * 0.70, "最小 p90 的 70%"},
        {"p90_min * 0.80", p90min * 0.80, ""},
        {"p90_min * 0.90", p90min * 0.90, ""},
        {"p90_p10", pctOf(allP90, 0.10), "观测到的最低 10% 分位"},
    };
    for (const auto& c : lo) {
        int clamped = 0;
        for (double v : allP90)
            if (v < c.value) ++clamped;
        std::printf("  %-18s = %7.4f   夹住 %3d 张 %s%s\n", c.name, c.value, clamped,
                    clamped == 0 ? "(安全)" : "(夹住了正常图，不该用)", c.basis);
    }

    // ---- 候选上界：必须高于所有图的 p90，否则夹住正常图 ----
    std::printf("\n=== 候选上界（要求 > 最大 p90 = %.4f，否则夹住正常图）===\n", p90max);
    std::vector<Cand> hi = {
        {"p90_max * 1.00", p90max, "实测最大 p90，无余量"},
        {"p90_max * 1.05", p90max * 1.05, "5% 余量"},
        {"p90_max * 1.10", p90max * 1.10, "10% 余量"},
        {"p90_max * 1.15", p90max * 1.15, "15% 余量"},
        {"p95_max", allP95.back(), "实测最大 p95"},
        {"p99_max", p99max, "实测最大 p99"},
        {"p99_max * 1.05", p99max * 1.05, "最大笔画核心 +5%"},
    };
    for (const auto& c : hi) {
        int clamped = 0;
        for (double v : allP90)
            if (v > c.value) ++clamped;
        std::printf("  %-18s = %7.4f   夹住 %3d 张 %s%s\n", c.name, c.value, clamped,
                    clamped == 0 ? "(安全)" : "(夹住了正常图，不该用)", c.basis);
    }

    // ---- 区间健康度：LO 与 HI 之间要留出 p90 的整个实测范围 ----
    std::printf("\n=== 推荐区间 ===\n");
    std::printf("  实测 p90 全距 : [%.4f, %.4f]  跨度 %.4f\n", p90min, p90max, p90max - p90min);
    std::printf("  安全下界上限  : %.4f （必须 < %.4f）\n", p90min * 0.999, p90min);
    std::printf("  安全上界下限  : %.4f （必须 > %.4f）\n", p90max * 1.001, p90max);
    std::printf("\n  任选 LO < %.4f 且 HI > %.4f，本批 %zu 张图都不会被夹住。\n", p90min, p90max,
                allP90.size());
    std::printf("  换言之：LO/HI 在这批数据上**不可能被验证**——它们只在现场异常图上起作用。\n");
    return 0;
}
