#include "core/warp.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace markloc::charinsp::detail {

bool estimateWarp(const cv::Mat& sImg, const cv::Mat& sRef, cv::Mat& warp, double* eccOut) {
    if (sImg.size() != sRef.size()) return false;

    cv::Mat a, b;
    cv::resize(sImg, a, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(sRef, b, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    auto to8u = [](const cv::Mat& src) {
        cv::Mat out;
        src.convertTo(out, CV_8U, 80.0, 0.0);
        return out;
    };
    cv::Mat a8 = to8u(a), b8 = to8u(b);

    cv::Mat m = cv::Mat::eye(2, 3, CV_32F);
    const cv::TermCriteria term(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-5);
    double ecc = -1;
    try {
        ecc = cv::findTransformECC(b8, a8, m, cv::MOTION_EUCLIDEAN, term, cv::noArray(), 5);
    } catch (const cv::Exception&) {
        ecc = -1;
    }
    if (eccOut) *eccOut = ecc;
    if (ecc < 0.0) {
        warp = cv::Mat::eye(2, 3, CV_32F);
        return false;
    }
    warp = m.clone();
    warp.at<float>(0, 2) = m.at<float>(0, 2) * 2.0f;   // 半分辨率 -> 全分辨率
    warp.at<float>(1, 2) = m.at<float>(1, 2) * 2.0f;
    return true;
}

cv::Mat applyWarp(const cv::Mat& s, const cv::Mat& warp) {
    cv::Mat out;
    cv::warpAffine(s, out, warp, s.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
    return out;
}

}  // namespace markloc::charinsp::detail
