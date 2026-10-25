#include "core/pipeline.hpp"

namespace markloc::algorithm {

DetectionPipeline::DetectionPipeline(PipelineOptions options,
                                     std::shared_ptr<const IDetectionAlgorithm> algorithm)
    : options_(options),
      algorithm_(algorithm ? std::move(algorithm) : makeDefaultDetectionAlgorithm()) {}

ViewRun DetectionPipeline::runView(const cv::Mat& gray, int view,
                                   const ModelSnapshot& snapshot) const {
    ViewRun out;
    out.view = view;
    if (view < 0 || view >= kViewCount) {
        out.status = RunStatus::InvalidInput;
        out.error = "视角编号无效";
        return out;
    }
    if (gray.empty() || (options_.requireGray8 && gray.type() != CV_8UC1)) {
        out.status = RunStatus::InvalidInput;
        out.error = options_.requireGray8 ? "输入必须是非空 CV_8UC1 灰度图" : "输入图像为空";
        return out;
    }
    if (options_.requireValidModel && (!snapshot.valid || !snapshot.hasView(view))) {
        out.status = RunStatus::ModelUnavailable;
        out.error = "模型或视角无效";
        return out;
    }
    const cv::Size expected = snapshot.imageSize(view);
    if (options_.rejectSizeMismatch && !expected.empty() && gray.size() != expected) {
        out.status = RunStatus::InvalidInput;
        out.error = "输入尺寸与模型不一致";
        return out;
    }
    if (!algorithm_) {
        out.status = RunStatus::AlgorithmError;
        out.error = "未配置检测算法";
        return out;
    }
    return algorithm_->runView(gray, view, snapshot, options_.evidence);
}

ProductRun DetectionPipeline::runProduct(const ImageSet& images,
                                         const ModelSnapshot& snapshot) const {
    ProductRun out;
    if (options_.requireValidModel && !snapshot.valid) {
        out.status = RunStatus::ModelUnavailable;
        out.error = "模型无效";
        return out;
    }

    ImageSet usable = images;
    usable.present.fill(false);
    for (int v = 0; v < kViewCount; ++v) {
        const auto index = static_cast<size_t>(v);
        if (!images.present[index]) continue;
        const cv::Mat& image = images.images[index];
        if (image.empty()) {
            out.warnings.push_back("视角 v" + std::to_string(v) + " 图像为空，已跳过");
            continue;
        }
        if (options_.requireGray8 && image.type() != CV_8UC1) {
            out.warnings.push_back("视角 v" + std::to_string(v) +
                                   " 不是 CV_8UC1 灰度图，已跳过");
            continue;
        }
        if (options_.requireValidModel && !snapshot.hasView(v)) {
            out.warnings.push_back("视角 v" + std::to_string(v) + " 没有可用模型，已跳过");
            continue;
        }
        const cv::Size expected = snapshot.imageSize(v);
        if (options_.rejectSizeMismatch && !expected.empty() && image.size() != expected) {
            out.warnings.push_back("视角 v" + std::to_string(v) + " 尺寸不一致，已跳过");
            continue;
        }
        usable.present[index] = true;
    }

    bool any = false;
    for (bool present : usable.present) any = any || present;
    if (!any) {
        out.status = RunStatus::NoUsableView;
        out.error = "没有可运行的视角";
        return out;
    }
    if (!algorithm_) {
        out.status = RunStatus::AlgorithmError;
        out.error = "未配置检测算法";
        return out;
    }

    ProductRun result = algorithm_->runProduct(usable, snapshot, options_.evidence);
    result.warnings.insert(result.warnings.begin(), out.warnings.begin(), out.warnings.end());
    return result;
}
} // namespace markloc::algorithm
