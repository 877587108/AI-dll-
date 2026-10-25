# 项目结构与接口规范

## 目标

应用层只依赖稳定接口，检测算法按独立模块替换。模型文件、图像索引和算法实现通过明确的数据结构传递，避免 GUI 或命令行工具直接访问算法内部变量。字段约束的完整定义见 [data-contracts.md](data-contracts.md)。

## 目录职责

```text
src/core/
  api.hpp                          应用层唯一推荐入口
  algorithm_types.hpp/.cpp         稳定 DTO、模型只读句柄、状态枚举
  detection_algorithm.hpp          可替换的算法接口
  default_detection_algorithm.cpp  旧 charinsp 算法的适配器
  model_repository.*               图片读取、数据集索引、模型加载
  pipeline.*                       单视角/产品级流程编排与输入校验
  algorithm_internal.hpp           仅实现层可见的模型句柄定义
  char_core.hpp                    旧算法实现与兼容入口
  analyze.cpp                      默认检测流程
  detectors.hpp                    ROI 检测器内部输入契约
  detect_*.cpp                     一个缺陷分支一个实现
  signal.* / warp.*                笔画信号与 ECC 配准
  model_io.*                       模型参数文件读写
  naming.*                         文件名和视角解析
modules/chardet/                   字符切分与字符匹配模块
src/gui/                           Qt 展示层
tools/                             诊断和批处理程序
tests/                             公共接口契约测试
```

依赖方向固定为：`gui/tools -> markloc_application -> markloc_engine -> markloc_chardet`。
`markloc_core` 是保留给旧工具的兼容 INTERFACE target。检测实现不得依赖 Qt 或 GUI 类型。

CMake 目标与代码边界一一对应：

| CMake target | 内容 | 可替换点 |
|---|---|---|
| `markloc_chardet` | 字符切分、字符匹配和图案检测 | 字符算法模块 |
| `markloc_engine` | 信号、配准、ROI 检测和模型格式 | 具体检测分支、内部算法 |
| `markloc_application` | 公共 DTO、模型仓库、流程和默认适配器 | `IDetectionAlgorithm` 实现 |
| `markloc_core` | 兼容聚合接口 | 旧代码迁移过渡 |

## 推荐调用方式

```cpp
#include "core/api.hpp"

markloc::algorithm::ModelRepository repo;
std::string error;
auto model = repo.load(modelDir, &error);
markloc::algorithm::DetectionPipeline pipeline;
auto result = pipeline.runView(gray, view, model);
```

`DetectionPipeline` 负责模型有效性、视角范围、图像类型、图像尺寸和错误信息；
`charinsp::analyze` 保留为兼容的底层算法入口。默认算法通过
`makeDefaultDetectionAlgorithm()` 注入，调用方也可以传入自己的 `IDetectionAlgorithm`。

## 规范数据结构

| 类型 | 用途 | 约束 |
|---|---|---|
| `algorithm::Sample` | 一张输入图片 | `view` 为 0~5，无法解析为 -1；`label` 只能为 `Unknown/Ok/Ng` |
| `algorithm::DatasetIndex` | 数据集扫描结果 | `samples` 只放可解析图片；跳过原因写入 `warnings` |
| `algorithm::ImageSet` | 一个产品的六视角图片 | `images[v]` 与 `present[v]` 一一对应；缺失视角不得伪造空图 |
| `algorithm::ModelSnapshot` | 已加载模型只读句柄 | `valid=true` 才可运行；内部模型不出公共头 |
| `algorithm::ViewRun` | 单视角运行结果 | `status=Ok` 时 `valid=true`，失败时 `error` 非空 |
| `algorithm::ProductRun` | 产品级运行结果 | 可部分缺视角；跳过原因写入 `warnings` |
| `algorithm::RoiResult` | 公共单区域结果 | 原始量和归一化分同时保留，`reliable=false` 时不参与判定 |
| `algorithm::CharResult` | 逐字结果 | `rect` 必须配合 `CoordinateSpace` 使用，默认是整图坐标 |

图像约定：输入灰度图为 `CV_8UC1`；`strokeSignal` 和检测器中间图为 `CV_32FC1`；
证据图为 `CV_8UC1`。坐标统一使用 OpenCV 左上角原点。六个视角和四个 ROI 的下标
含义固定，不能复用。更详细的字段定义见 [data-contracts.md](data-contracts.md)。

## 更换算法的边界

新增或替换整套算法时，实现 `IDetectionAlgorithm` 并注入 `DetectionPipeline`；新增或替换
某个 ROI 分支时，实现 `detectors.hpp` 中约定的输入消费方式，并只改对应 `detect_*.cpp`。
参数、结果结构和 GUI 不变时，算法可以单独编译和回归。

```cpp
class NewAlgorithm final : public markloc::algorithm::IDetectionAlgorithm {
public:
    ViewRun runView(const cv::Mat&, int, const ModelSnapshot&, bool) const override;
    ProductRun runProduct(const ImageSet&, const ModelSnapshot&, bool) const override;
};

auto pipeline = markloc::algorithm::DetectionPipeline({},
    std::make_shared<const NewAlgorithm>());
```

## 兼容规则

公共接口新增字段必须提供默认值；枚举和数组下标含义不能复用；模型读写格式变化时更新
`model_io` 和版本说明。跨模块错误使用 `error/warnings` 返回，不用异常表达正常的检测失败。
`char_core.hpp`、`detectors.hpp` 和 `chardet.hpp` 继续为内部/兼容入口，新业务代码不应新增
对它们的直接依赖。
