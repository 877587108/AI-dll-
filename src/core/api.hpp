#pragma once

// markloc 对外使用的稳定入口。应用层只应依赖本文件和 algorithm_types.hpp，
// 不要直接包含 analyze.cpp 或具体 detector 头文件。
#include "core/algorithm_types.hpp"
#include "core/detection_algorithm.hpp"
#include "core/model_repository.hpp"
#include "core/pipeline.hpp"
