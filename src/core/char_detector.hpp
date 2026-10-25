#pragma once

// 字符级检测适配层。analyze.cpp 只负责流程编排，字符算法与 chardet 的耦合集中在
// 这里，替换字符切分/匹配实现时不需要修改 ROI 检测流程。

#include <opencv2/core.hpp>

#include "core/char_core.hpp"

namespace markloc::charinsp::chars {

void appendResults(const cv::Mat& gray, const ViewModel& model, const Params& params,
                   Result& result);

} // namespace markloc::charinsp::chars
