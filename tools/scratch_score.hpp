// 划痕评分（与 src/core/detect_scratch.cpp 同一算法）
//
// 放在工具里而不是依赖 markloc_core 的 det:: 命名空间，
// 是因为渲染工具只需要"给定模型+图片，算出划痕分"这一件事。
//
// 算法：HoughLinesP 找直线段 → 沿线段采样 → 统计覆盖率与能量
// 能区分真实划痕与笔画边缘：笔画边缘不构成长直线。

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace render_detail {

struct ScratchScore {
    double best = 0.0;          // 最强划痕能量
    double severe = 0.0;        // 严重划痕能量（长度达标者）
    double coverage = 0.0;
    double length = 0.0;
    double meanS = 0.0;
    int count = 0;              // 达标的划痕条数
    int severeCount = 0;        // 严重划痕条数

    // 【笔画相关】只有压在字符笔画上的划痕才影响可读性
    double onInkBest = 0.0;     // 压在笔画上的划痕：最强能量
    double onInkLength = 0.0;   // 压在笔画上的最长划痕
    double onInkRatio = 0.0;    // 压笔画部分占该划痕长度的比例
    int onInkCount = 0;         // 压在笔画上的划痕条数
};

/// aligned: 笔画信号 CV_32F；weak: 弱笔画掩码 CV_8U（0/255）
/// minLen: 最短划痕长度；minCoverage: 最短覆盖率
/// inkMask: 字符笔画区域 CV_8U（0/255）；传入时同时统计"压笔画"的划痕
inline ScratchScore scoreScratch(const cv::Mat& aligned, const cv::Mat& weak, double strokeLo,
                                 double minLen, double minCoverage,
                                 const cv::Mat* inkMask = nullptr) {
    ScratchScore out;
    if (aligned.empty() || weak.empty()) return out;
    if (aligned.type() != CV_32F || weak.type() != CV_8U) return out;
    if (aligned.size() != weak.size()) return out;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(weak, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);

    constexpr int BAND = 9;

    for (const auto& ln : lines) {
        const cv::Point p1(ln[0], ln[1]);
        const cv::Point p2(ln[2], ln[3]);
        const double len = cv::norm(p2 - p1);
        if (len < 1.0) continue;

        const int steps = std::max(2, static_cast<int>(len));
        double energy = 0.0;
        int covered = 0;

        for (int t = 0; t <= steps; ++t) {
            const double r = static_cast<double>(t) / steps;
            const int x = static_cast<int>(p1.x + (p2.x - p1.x) * r);
            const int y = static_cast<int>(p1.y + (p2.y - p1.y) * r);
            if (y < 0 || y >= weak.rows) continue;
            bool hit = false;
            for (int dy = -BAND / 2; dy <= BAND / 2; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= weak.rows) continue;
                const int xx = std::clamp(x, 0, weak.cols - 1);
                if (weak.at<uchar>(yy, xx) == 0) continue;
                energy += std::max(aligned.at<float>(yy, xx) - strokeLo, 0.0);
                if (dy >= -1 && dy <= 1) hit = true;
            }
            if (hit) ++covered;
        }

        const double coverage = static_cast<double>(covered) / (steps + 1);
        if (coverage < minCoverage) continue;   // 噪点连线，不算划痕

        const double score = energy * len;
        ++out.count;

        // 统计该线段有多少点落在字符笔画上（或笔画邻域内）
        int onInk = 0;
        if (inkMask != nullptr && inkMask->size() == weak.size()) {
            for (int t = 0; t <= steps; ++t) {
                const double r = static_cast<double>(t) / steps;
                const int x = static_cast<int>(p1.x + (p2.x - p1.x) * r);
                const int y = static_cast<int>(p1.y + (p2.y - p1.y) * r);
                if (y < 0 || y >= inkMask->rows) continue;
                bool hitInk = false;
                for (int dy = -3; dy <= 3; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= inkMask->rows) continue;
                    const int xx = std::clamp(x, 0, inkMask->cols - 1);
                    if (inkMask->at<uchar>(yy, xx)) { hitInk = true; break; }
                }
                if (hitInk) ++onInk;
            }
        }
        const double ratio = static_cast<double>(onInk) / (steps + 1);
        if (ratio >= 0.30) {   // 30% 以上压在笔画上，视为影响可读性
            ++out.onInkCount;
            if (score > out.onInkBest) {
                out.onInkBest = score;
                out.onInkLength = len;
                out.onInkRatio = ratio;
            }
        }
        if (len >= minLen) {
            out.severe = std::max(out.severe, score);
            ++out.severeCount;
        }
        if (score > out.best) {
            out.best = score;
            out.coverage = coverage;
            out.length = len;
            out.meanS = energy / std::max(1.0, static_cast<double>(covered));
        }
    }
    return out;
}

}  // namespace render_detail
