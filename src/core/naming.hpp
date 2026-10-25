#pragma once

// 图片文件名解析与同产品多视角查找
// 命名约定：{料号}_{FT|RT}_..._{视角}.tif
// 只依赖标准库，核心层不引入 Qt

#include <array>
#include <string>
#include <vector>

namespace markloc::charinsp::naming {

/// 从文件名末尾解析视角编号（_00.._05），失败返回 -1
int viewOf(const std::string& path);

/// 料号（文件名前 4 位，如 "0026"）
std::string productOf(const std::string& path);

/// 视角前缀（去掉末尾 _NN 与扩展名），用于拼出同产品其它视角的文件名
std::string prefixOf(const std::string& path);

/// 拍摄批次：FT / RT / 空
std::string shotOf(const std::string& path);

/// 拼接指定视角的文件路径（同目录、同后缀）
std::string siblingPath(const std::string& anyView, int view);

/// 查找同产品的六个视角路径（不存在的为空串），present 标记存在性
std::vector<std::string> findSiblings(const std::string& anyView, std::array<bool, 6>& present);

/// 从目录批量收集图片（优先 ok/ng 子目录）
std::vector<std::string> collectImages(const std::string& root);

}  // namespace markloc::charinsp::naming
