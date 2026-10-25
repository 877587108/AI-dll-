#pragma once

// ECC 配准（半分辨率估计，全分辨率应用）

#include <opencv2/core.hpp>

namespace markloc::charinsp::detail {

/// 估计把 sImg 对齐到 sRef 的变换；返回是否成功（失败时 warp 为单位阵）
bool estimateWarp(const cv::Mat& sImg, const cv::Mat& sRef, cv::Mat& warp,
                  double* eccOut = nullptr);

/// 按 warp 反变换重采样（内部使用 WARP_INVERSE_MAP）
cv::Mat applyWarp(const cv::Mat& s, const cv::Mat& warp);

}  // namespace markloc::charinsp::detail
