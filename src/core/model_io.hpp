#pragma once

// 模型存取：params.txt / rois.txt / 模板图像
//
// 目录布局（每个视角一个子目录）：
//   <dir>/vNN/
//   ├── logo _ref|freq|tol.png     四个区域的模板
//   ├── line1_ref|freq|tol.png
//   ├── line2_ref|freq|tol.png
//   ├── code _ref|freq|tol.png
//   ├── full_freq.png              整图频率（供字符级检测）
//   ├── full_ref.png               整图配准基准
//   ├── params.txt                 标量参数 + 4 行 ROI
//   ├── rois.txt                   用户可编辑的区域配置（存在则覆盖 params.txt）
//   └── code_stats.txt             历史保留文件，无实际语义

#include <string>

#include "core/char_core.hpp"

namespace markloc::charinsp {

/// 视角子目录路径拼装：<dir>/vNN
std::string viewDir(const std::string& dir, int view);

/// 保存/读取单个视角参数（params.txt）
bool saveParams(const std::string& dir, int view, const Params& p);
bool loadParams(const std::string& dir, int view, Params& p);

/// 把当前区域定义导出成用户可编辑的 rois.txt（带注释）
bool exportRoisFile(const std::string& dir, int view, const Params& p);

/// 保存单个视角的模板图像
bool saveViewModel(const std::string& dir, int view, const ViewModel& vm, std::string* err);

}  // namespace markloc::charinsp
