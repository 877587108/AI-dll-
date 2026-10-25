// 连续划痕检测（区域内）
// 判据：Hough 候选线段 → 密度（±1px 窄带内有残差的采样点占比）→ 长度门槛
//   密度不足：噪点连线，丢弃
//   长度不足：轻微划痕，放行（不计入严重划痕分）

#include "core/detectors.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace markloc::charinsp::det {

void scoreScratch(const cv::Mat& aligned, const cv::Mat& weak, const Params& p, RoiResult& rr) {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(weak, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);

    constexpr int BAND = 9;
    double best = 0.0;

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
            bool hit = false;
            for (int dy = -BAND / 2; dy <= BAND / 2; ++dy) {
                const int yy = y + dy;
                if (yy < 0 || yy >= weak.rows) continue;
                const int xx = std::clamp(x, 0, weak.cols - 1);
                if (weak.at<uchar>(yy, xx) == 0) continue;
                energy += std::max(aligned.at<float>(yy, xx) - p.strokeLo, 0.0);
                if (dy >= -1 && dy <= 1) hit = true;   // 密度只统计窄带
            }
            if (hit) ++covered;
        }

        const double coverage = static_cast<double>(covered) / (steps + 1);
        if (coverage < p.scratchMinCoverage) continue;   // 噪点连线

        const double score = energy * len;
        if (len >= p.scratchMinLen) rr.scratchSevere = std::max(rr.scratchSevere, score);
        if (score > best) {
            best = score;
            rr.scratchCoverage = coverage;
            rr.scratchLen = len;
            rr.scratchMeanS = energy / std::max(1.0, static_cast<double>(covered));
        }
    }
    rr.scratch = best;
}

}  // namespace markloc::charinsp::det
