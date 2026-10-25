# 公共数据结构规范

本文定义 `src/core/api.hpp` 暴露的应用层数据。GUI、命令行和算法实现只通过这些类型交换数据；`charinsp::*` 与 `chardet::*` 类型属于实现层，不能出现在应用层头文件或 GUI 成员变量中。

## 图像类型

| 名称 | C++ 类型 | 约束 |
|---|---|---|
| 输入灰度图 | `cv::Mat` | `CV_8UC1`、非空、整图坐标，左上角为原点 |
| 笔画信号 | `cv::Mat` | `CV_32FC1`，算法内部使用，不跨公共接口传递 |
| 频率/掩膜 | `cv::Mat` | `CV_8UC1`，0~255 对应 0.0~1.0 或二值掩膜 |
| 证据图 | `cv::Mat` | `CV_8UC1`，仅用于显示和导出 |

`ImageSet::present[v]` 与 `ImageSet::images[v]` 必须成对：`present[v] == true` 时图像必须非空且满足输入类型。流程会再次校验并把无效视角写入 `ProductRun::warnings`，不能用空图冒充缺失视角。

## 标识与索引

- 视角编号固定为 `0..5`，对应 `kViewCount = 6`。`-1` 只表示“未解析/未知”，不能作为数组下标。
- 区域编号固定为 `0..3`，顺序为 logo、line1、line2、code，对应 `kRegionCount = 4`。
- `SampleLabel` 只有 `Unknown`、`Ok`、`Ng` 三个值。没有真值时使用 `Unknown`，不能把缺失真值当作 `Ok`。
- `CoordinateSpace::FullImage` 表示整图坐标；`CoordinateSpace::Roi` 表示对应 ROI 内的相对坐标。公共 `CharResult` 默认使用整图坐标，若算法返回 ROI 坐标必须显式设置枚举。

## 模型句柄

`ModelSnapshot` 是只读、可复制的模型句柄：

```cpp
markloc::algorithm::ModelRepository repository;
std::string error;
auto model = repository.load(modelDir, &error);
if (!model.valid) { /* 处理 error */ }
```

模型文件格式、参数数组、模板图和 `charinsp::Model` 都隐藏在句柄内部。应用层只调用 `valid`、`hasView(view)` 和 `imageSize(view)`。如果模型格式升级，只修改 `model_repository.cpp` 与适配器，不修改 GUI 的数据结构。

## 运行结果

`ViewRun` 和 `ProductRun` 使用 `RunStatus` 表达状态：

- `Ok`：算法完成，`valid == true`，结果字段可读取。
- `InvalidInput`：输入图像、视角或尺寸不符合契约。
- `ModelUnavailable`：模型无效或视角没有模型。
- `NoUsableView`：产品输入没有任何可运行视角。
- `AlgorithmError`：算法实现未配置或执行失败。

产品可以缺少视角。`ProductResult::present` 只标记真正参与计算的视角，`ProductRun::warnings` 保存被跳过视角及原因。

## 检测器结果

`RoiResult` 中的原始量和归一化量同时保留：

- `scratch`、`appendage`、`missing` 是算法原始量，便于调试和回归；
- `scratchSevere / scratchT`、`appendage / appendageT`、`missing / missT` 组成区域分；
- `score` 是区域最终分，`ViewResult::score` 是视角最高分；
- `reliable == false` 的区域不能参与最终判定；
- `ecc` 是配准质量，范围由具体 ECC 实现决定，失败使用负值。

新增字段必须提供默认值，已有枚举值和数组下标不能复用。模型读写格式变更时要同步更新版本号和 `model_io`，不能在 GUI 中解析参数文件。

## 算法替换规则

实现新算法时继承 `IDetectionAlgorithm`：

```cpp
class NewAlgorithm final : public markloc::algorithm::IDetectionAlgorithm {
public:
    ViewRun runView(const cv::Mat&, int, const ModelSnapshot&, bool) const override;
    ProductRun runProduct(const ImageSet&, const ModelSnapshot&, bool) const override;
};
```

然后把实现注入 `DetectionPipeline`。流程校验、GUI 和结果 DTO 不需要改动。若只替换某一个 ROI 分支，继续使用 `detectors.hpp` 的内部输入契约，并只修改对应的 `detect_*.cpp`；不要把阈值、Qt 类型或文件路径写入检测器。
