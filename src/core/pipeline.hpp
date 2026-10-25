#pragma once

#include "core/algorithm_types.hpp"
#include "core/detection_algorithm.hpp"
#include "core/model_repository.hpp"

namespace markloc::algorithm {

class DetectionPipeline {
public:
    explicit DetectionPipeline(
        PipelineOptions options = {},
        std::shared_ptr<const IDetectionAlgorithm> algorithm = {});

    ViewRun runView(const cv::Mat& gray, int view, const ModelSnapshot& snapshot) const;
    ProductRun runProduct(const ImageSet& images, const ModelSnapshot& snapshot) const;

private:
    PipelineOptions options_;
    std::shared_ptr<const IDetectionAlgorithm> algorithm_;
};

} // namespace markloc::algorithm
