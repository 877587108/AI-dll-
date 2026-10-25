#pragma once

#include <memory>

#include "core/algorithm_types.hpp"

namespace markloc::algorithm {

/// 可替换的检测算法实现。
///
/// Pipeline 只负责输入校验、模型句柄和结果状态；具体算法通过此接口注入。
/// 新算法可以在独立源文件/库中实现该接口，不需要修改 GUI、CLI 或流程编排。
class IDetectionAlgorithm {
public:
    virtual ~IDetectionAlgorithm() = default;

    virtual ViewRun runView(const cv::Mat& gray, int view,
                            const ModelSnapshot& model, bool wantEvidence) const = 0;

    virtual ProductRun runProduct(const ImageSet& images,
                                  const ModelSnapshot& model,
                                  bool wantEvidence) const = 0;
};

/// 当前内置 charinsp 算法的适配器。调用方通常只需要默认构造 DetectionPipeline。
std::shared_ptr<const IDetectionAlgorithm> makeDefaultDetectionAlgorithm();

} // namespace markloc::algorithm
