// 阈值错配的定量验证：weak 的 0.105 和掩膜的 0.149 差在哪
//
// 因果链（到此为止都是推测，本工具负责证实或推翻）：
//   训练：freq = %(signal > strokeHi=0.35)，把每个像素二值化再平均
//   推理：modelInk = freq > blankFreq=0.15  → 即"15% 的样本里这个像素 signal>0.35"
//   推理：weak     = signal > missStrokeLo*0.7 = 0.105
//
// 关键推论：一个像素落在字形边缘时，它大约有 50% 的样本 signal>0.35
//   → freq ≈ 128 ≫ 38 → 被当模型墨扣掉。所以边缘**本该**扣干净。
// 真正漏出来的是 freq ∈ (0, 38] 的那条窄带，即出现率 0~15% 的像素。
//
// 那么问题就变成：这条窄带有多宽？9x9 膨胀够不够盖住？
// 若信号与模型之间存在**亚像素/1~2px 的相对位移**，字形边缘会出现
// "信号在这侧、模型在那侧"的错位，错位宽度 = 位移量。
// 若位移 > 4.5px（9x9 核的半径），膨胀盖不住 → 露一圈边。
//
// 本工具直接量：
//   1) 边缘带宽度分布（leak 像素离 modelInk 的最短距离）
//   2) 同一字，模型 vs 信号 的一维行/列剖面互相关，求亚像素错位
//
// 用法: markloc/thrmatch <modelDir> <image>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

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

/// 距离变换：每个前景像素到最近背景的距离
static cv::Mat distTo(const cv::Mat& fg) {
    cv::Mat inv;
    cv::bitwise_not(fg, inv);
    cv::Mat dt;
    cv::distanceTransform(inv, dt, cv::DIST_L2, 3);
    return dt;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_thrmatch <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("读取失败\n");
        return 1;
    }
    std::vector<chardet::CharUnit> units;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) units.push_back(std::move(x));
    }
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat r32;
    model.ref.convertTo(r32, CV_32F);
    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.area() > 0) inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
    cv::Mat sC, rC;
    sig(inkArea).copyTo(sC);
    r32(inkArea).copyTo(rC);
    const cv::Point2d gsh = cv::phaseCorrelate(sC, rC, cv::noArray());
    if (std::abs(gsh.x) < 60 && std::abs(gsh.y) < 60 &&
        (std::abs(gsh.x) > 0.01 || std::abs(gsh.y) > 0.01)) {
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -gsh.x, 0, 1, -gsh.y);
        cv::Mat al;
        cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
        sig = al;
    }
    std::printf("全局配准 (ref): (%+.2f, %+.2f)\n", gsh.x, gsh.y);

    const double weakThr = cfg.missStrokeLo * 0.7;
    const int inkThr = static_cast<int>(cfg.blankFreq * 255);
    cv::Mat weak, modelInk;
    cv::compare(sig, weakThr, weak, cv::CMP_GT);
    cv::compare(model.freq, inkThr, modelInk, cv::CMP_GT);
    cv::Mat d9, nm9, leak;
    cv::dilate(modelInk, d9, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(d9, nm9);
    cv::bitwise_and(weak, nm9, leak);

    // ---- 1) leak 像素离 modelInk 的距离分布
    const cv::Mat dt = distTo(modelInk);
    std::vector<double> ds;
    for (int y = 0; y < leak.rows; ++y)
        for (int x = 0; x < leak.cols; ++x)
            if (leak.at<uchar>(y, x)) ds.push_back(dt.at<float>(y, x));
    std::sort(ds.begin(), ds.end());
    std::printf("\n=== leak 像素到最近模型墨的距离（全图 %zu 个）===\n", ds.size());
    if (!ds.empty()) {
        const auto q = [&](double f) { return ds[std::min(ds.size() - 1, size_t(ds.size() * f))]; };
        std::printf("  p50=%.2f p75=%.2f p90=%.2f p95=%.2f p99=%.2f max=%.2f\n", q(0.50),
                    q(0.75), q(0.90), q(0.95), q(0.99), ds.back());
        int over45 = 0;
        for (double v : ds)
            if (v > 4.5) ++over45;
        std::printf("  距离 > 4.5px（9x9 核半径）的比例: %.1f%%\n", 100.0 * over45 / ds.size());
    }

    // ---- 2) 逐字：模型 vs 信号 的行/列剖面互相关，求亚像素错位
    std::printf("\n=== 逐字剖面错位（正=信号比模型偏右下）===\n");
    std::printf("%-6s %-22s %8s %8s %8s\n", "index", "框", "行向错位", "列向错位", "合成");
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width <= 8 || b.height <= 8) continue;
        // 行剖面（沿 y 的墨量）与列剖面
        const cv::Mat s = sig(b), f = model.freq(b);
        std::vector<double> sRow(b.height, 0.0), fRow(b.height, 0.0);
        std::vector<double> sCol(b.width, 0.0), fCol(b.width, 0.0);
        for (int y = 0; y < b.height; ++y)
            for (int x = 0; x < b.width; ++x) {
                sRow[y] += s.at<float>(y, x);
                sCol[x] += s.at<float>(y, x);
                fRow[y] += f.at<uchar>(y, x) / 255.0;
                fCol[x] += f.at<uchar>(y, x) / 255.0;
            }
        // 一维互相关求平移，再做抛物线亚像素
        const auto shift1d = [](const std::vector<double>& a, const std::vector<double>& bb) {
            const int n = static_cast<int>(std::min(a.size(), bb.size()));
            const int maxLag = 6;
            int best = 0;
            double bestV = -1e30;
            std::vector<double> corr(2 * maxLag + 1, 0.0);
            for (int lag = -maxLag; lag <= maxLag; ++lag) {
                double num = 0, da = 0, db = 0;
                for (int i = 0; i < n; ++i) {
                    const int j = i + lag;
                    if (j < 0 || j >= n) continue;
                    num += a[i] * bb[j];
                    da += a[i] * a[i];
                    db += bb[j] * bb[j];
                }
                const double v = num / std::sqrt(std::max(1e-12, da * db));
                corr[lag + maxLag] = v;
                if (v > bestV) { bestV = v; best = lag; }
            }
            // 抛物线插值：峰在 best，取左右
            const int idx = best + maxLag;
            double sub = 0;
            if (idx > 0 && idx + 1 < 2 * maxLag + 1) {
                const double l = corr[idx - 1], c = corr[idx], r = corr[idx + 1];
                const double den = 2 * c - l - r;
                if (std::abs(den) > 1e-12) sub = std::clamp((r - l) / den, -1.0, 1.0);
            }
            return static_cast<double>(best) + sub;
        };
        const double dy = shift1d(sRow, fRow);
        const double dx = shift1d(sCol, fCol);
        std::printf("%-6d %-22s %+8.2f %+8.2f %8.2f\n", u.index,
                    cv::format("(%d,%d %dx%d)", b.x, b.y, b.width, b.height).c_str(), dy, dx,
                    std::hypot(dx, dy));
    }
    return 0;
}
