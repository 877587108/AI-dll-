#pragma once

// 仅 markloc_core 内部使用。应用层不得包含此文件。

#include "core/algorithm_types.hpp"
#include "core/char_core.hpp"

namespace markloc::algorithm::detail {

/// 公共 ModelSnapshot 的私有实现。模型对象只在 engine/application 适配层出现。
struct ModelData {
    charinsp::Model model;
};

const ModelData* modelData(const ModelSnapshot& snapshot) noexcept;

} // namespace markloc::algorithm::detail
