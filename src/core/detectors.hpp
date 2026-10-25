#pragma once

// 单项检测器：每个检测分支一个独立实现文件。
#include <opencv2/core.hpp>

#include "core/char_core.hpp"

namespace markloc::charinsp::det {

// 稳定的检测器输入契约：算法只读取输入并写入自己负责的 RoiResult 字段。
struct DetectorInput {
    const cv::Mat& aligned;        // CV_32FC1，已配准的笔画信号
    const cv::Mat& extra;          // CV_8UC1，强残差
    const cv::Mat& weak;           // CV_8UC1，弱残差
    const cv::Mat& frequency;      // CV_8UC1，0~255 表示频率
    const Params& params;          // 当前视角的只读参数
    bool allowAppendage = false;
};

// 执行当前检测器集合并返回归一化区域分数。该编排位于 detector_suite.cpp，
// 具体分支只通过下面三个函数连接；替换一个分支不会改其它分支。
double runDetectors(const DetectorInput& input, RoiResult& result);

/// 多余笔画：连通域 + PCA 伸长比，返回 面积×平均S 的最大值。
double scoreAppendage(const cv::Mat& aligned, const cv::Mat& extra);
/// 连续划痕：Hough + 密度 + 长度门槛，结果写入 rr。
void scoreScratch(const cv::Mat& aligned, const cv::Mat& weak, const Params& p, RoiResult& rr);
/// 缺墨断线：正常样本高频像素处当前信号过弱，返回区域分。
double scoreMissing(const cv::Mat& aligned, const cv::Mat& freq, const Params& p, RoiResult& rr);

}  // namespace markloc::charinsp::det
