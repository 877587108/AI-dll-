#include "chardet.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace chardet {

namespace {

struct Seg {
    int x0, x1;
};

}  // namespace

// 从频率图切分字符（用于训练时的自动切分）
std::vector<CharUnit> split(const cv::Mat& freq8, const Config& cfg) {
    std::vector<CharUnit> units;
    if (freq8.empty()) return units;

    cv::Mat bin;
    cv::compare(freq8, static_cast<int>(255 * std::clamp(cfg.binFreq, 0.02, 0.95)), bin,
                cv::CMP_GT);

    const cv::Mat kOpen = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kOpen);

    cv::Mat proj;
    cv::reduce(bin, proj, 0, cv::REDUCE_SUM, CV_32S);
    const int cols = bin.cols;
    std::vector<int> colInk(static_cast<size_t>(cols));
    for (int x = 0; x < cols; ++x) {
        colInk[static_cast<size_t>(x)] = proj.at<int>(0, x) / 255;
    }

    int maxCol = 0;
    for (int v : colInk) maxCol = std::max(maxCol, v);
    if (maxCol <= 0) return units;

    const double thrRatio = std::clamp(cfg.colThrRatio, 0.01, 0.90);
    const int colThr = std::max(1, static_cast<int>(maxCol * thrRatio));

    std::vector<Seg> segs;
    int start = -1;
    for (int x = 0; x < cols; ++x) {
        const bool on = colInk[static_cast<size_t>(x)] >= colThr;
        if (on && start < 0) start = x;
        if (!on && start >= 0) {
            segs.push_back({start, x - 1});
            start = -1;
        }
    }
    if (start >= 0) segs.push_back({start, cols - 1});
    if (segs.empty()) return units;

    // 迭代合并过窄间隙
    for (int iter = 0; iter < 8; ++iter) {
        std::vector<int> widths;
        for (const auto& s : segs) widths.push_back(s.x1 - s.x0 + 1);
        std::sort(widths.begin(), widths.end());
        const int medianW = widths[widths.size() / 2];
        const double gr = std::clamp(cfg.gapMergeRatio, 0.0, 0.60);
        const int gapThr = std::max(2, static_cast<int>(medianW * gr));

        std::vector<Seg> merged;
        bool changed = false;
        for (const auto& s : segs) {
            if (!merged.empty()) {
                const int gap = s.x0 - merged.back().x1 - 1;
                if (gap >= 0 && gap < gapThr) {
                    merged.back().x1 = s.x1;
                    changed = true;
                    continue;
                }
            }
            merged.push_back(s);
        }
        segs.swap(merged);
        if (!changed) break;
    }

    const int minH = std::max(8, static_cast<int>(freq8.rows * cfg.minHeightRatio));
    int idx = 0;
    for (const auto& s : segs) {
        const int w = s.x1 - s.x0 + 1;
        if (w <= 0) continue;

        const cv::Mat sub = bin(cv::Rect(s.x0, 0, w, bin.rows));
        cv::Mat rowsInk;
        cv::reduce(sub, rowsInk, 1, cv::REDUCE_SUM, CV_32S);
        int y0 = -1, y1 = -1;
        for (int y = 0; y < bin.rows; ++y) {
            if (rowsInk.at<int>(y, 0) > 0) {
                if (y0 < 0) y0 = y;
                y1 = y;
            }
        }
        if (y0 < 0 || y1 < y0) continue;
        const int h = y1 - y0 + 1;
        if (h < minH) continue;

        const int ink = cv::countNonZero(sub);
        if (ink < minH * 2) continue;

        const cv::Rect base(s.x0, y0, w, h);
        constexpr int padUp = 2;
        constexpr int padOut = 6;
        cv::Rect ext;
        ext.x = std::max(0, base.x - padUp);
        ext.y = std::max(0, base.y - padOut);
        const int ex2 = std::min(freq8.cols, base.x + base.width + padUp);
        const int ey2 = std::min(freq8.rows, base.y + base.height + padOut);
        ext.width = ex2 - ext.x;
        ext.height = ey2 - ext.y;

        CharUnit u;
        u.bbox = base;
        u.extend = ext;
        u.index = idx++;
        u.freq = cv::Mat::zeros(freq8.size(), CV_8U);
        freq8(base).copyTo(u.freq(base));
        units.push_back(std::move(u));
    }
    return units;
}

}  // namespace chardet
