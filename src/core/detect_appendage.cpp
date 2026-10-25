// 多余笔画检测（仅 logo 区启用）
// 判据：容差区外的强残差连通域，若主方向伸长比 > 2.2（细长）则可能是多余笔画

#include "core/detectors.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace markloc::charinsp::det {

double scoreAppendage(const cv::Mat& aligned, const cv::Mat& extra) {
    double best = 0.0;

    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(extra, labels, stats, cents, 8, CV_32S);
    for (int k = 1; k < n; ++k) {
        const int area = stats.at<int>(k, cv::CC_STAT_AREA);
        if (area < 15) continue;

        // 收集该连通域的像素用于 PCA
        std::vector<cv::Point> pts;
        pts.reserve(static_cast<size_t>(area));
        for (int y = 0; y < labels.rows; ++y) {
            const int* row = labels.ptr<int>(y);
            for (int x = 0; x < labels.cols; ++x) {
                if (row[x] == k) pts.emplace_back(x, y);
            }
        }
        if (pts.size() < 15) continue;

        cv::Mat data(static_cast<int>(pts.size()), 2, CV_32F);
        for (size_t j = 0; j < pts.size(); ++j) {
            data.at<float>(static_cast<int>(j), 0) = static_cast<float>(pts[j].x);
            data.at<float>(static_cast<int>(j), 1) = static_cast<float>(pts[j].y);
        }
        cv::Mat mean, covar;
        cv::calcCovarMatrix(data, covar, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS);
        cv::Mat evals;
        cv::eigen(covar, evals);
        if (evals.rows < 2) continue;

        const double l1 = std::max(evals.at<double>(0), 1e-6);
        const double l2 = std::max(evals.at<double>(1), 1e-6);
        if (std::sqrt(l1 / l2) <= 2.2) continue;   // 不够细长，判为非笔画

        double sum = 0.0;
        for (const auto& pt : pts) sum += aligned.at<float>(pt);
        best = std::max(best, static_cast<double>(area) * sum / pts.size());
    }
    return best;
}

}  // namespace markloc::charinsp::det
