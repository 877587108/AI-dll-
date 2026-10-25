#include "core/char_core.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace markloc::charinsp {

cv::Mat strokeSignal(const cv::Mat& gray, const cv::Rect& roi, const Params& p, double* q94out) {
    const cv::Rect r = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width <= 0 || r.height <= 0) return {};

    cv::Mat patch = gray(r).clone();
    patch.convertTo(patch, CV_32F);

    // 去背景：减去平滑背景，只留"比周围突出"的部分
    cv::Mat bg;
    cv::GaussianBlur(patch, bg, cv::Size(0, 0), p.sigma);
    cv::Mat d;
    cv::subtract(patch, bg, d);

    // 暗刻印：翻转使笔画为正
    if (p.invert) cv::multiply(d, -1.0, d);

    // 取 94 分位作为归一化刻度（自适应曝光）
    std::vector<int> hist(512, 0);
    for (int y = 0; y < d.rows; ++y) {
        const float* row = d.ptr<float>(y);
        for (int x = 0; x < d.cols; ++x) {
            int b = static_cast<int>(std::lround(row[x])) + 256;
            b = std::clamp(b, 0, 511);
            ++hist[static_cast<size_t>(b)];
        }
    }
    const long total = static_cast<long>(d.rows) * d.cols;
    long acc = 0;
    double q94 = 8.0;
    for (int b = 0; b < 512; ++b) {
        acc += hist[static_cast<size_t>(b)];
        if (acc >= static_cast<long>(total * 0.94)) {
            q94 = static_cast<double>(b - 256);
            break;
        }
    }
    if (q94out) *q94out = q94;

    const double scale = std::max(8.0, q94);
    cv::Mat s;
    d.convertTo(s, CV_32F, 1.0 / scale);
    cv::threshold(s, s, 0.0, 0.0, cv::THRESH_TOZERO);   // 丢弃负残差
    return s;
}

}  // namespace markloc::charinsp
