#include "core/api.hpp"

#include <cstdio>
#include <memory>

#include <opencv2/core.hpp>

namespace {

class FakeAlgorithm final : public markloc::algorithm::IDetectionAlgorithm {
public:
    mutable int viewCalls = 0;
    mutable int productCalls = 0;
    mutable std::array<bool, markloc::algorithm::kViewCount> productViews{};

    markloc::algorithm::ViewRun runView(
        const cv::Mat&, int view, const markloc::algorithm::ModelSnapshot&, bool) const override {
        ++viewCalls;
        markloc::algorithm::ViewRun out;
        out.view = view;
        out.status = markloc::algorithm::RunStatus::Ok;
        out.valid = true;
        out.result.score = 0.25;
        return out;
    }

    markloc::algorithm::ProductRun runProduct(
        const markloc::algorithm::ImageSet& images,
        const markloc::algorithm::ModelSnapshot&, bool) const override {
        ++productCalls;
        productViews = images.present;
        markloc::algorithm::ProductRun out;
        out.status = markloc::algorithm::RunStatus::Ok;
        out.valid = true;
        return out;
    }
};

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "contract test failed: %s\n", message);
    return condition;
}

} // namespace

int main() {
    using namespace markloc::algorithm;

    auto fake = std::make_shared<FakeAlgorithm>();
    PipelineOptions options;
    options.requireValidModel = false;
    DetectionPipeline pipeline(options, fake);
    ModelSnapshot opaqueModel;

    const cv::Mat gray(8, 8, CV_8UC1, cv::Scalar(0));
    const auto view = pipeline.runView(gray, 2, opaqueModel);
    if (!check(view.valid && view.status == RunStatus::Ok && fake->viewCalls == 1,
               "injected view algorithm must be called through the public pipeline")) return 1;

    const cv::Mat color(8, 8, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto invalid = pipeline.runView(color, 2, opaqueModel);
    if (!check(!invalid.valid && invalid.status == RunStatus::InvalidInput,
               "non-gray input must be rejected by the pipeline")) return 1;

    ImageSet images;
    images.present[0] = true;
    images.images[0] = gray;
    images.present[1] = true;              // declared but empty: must be filtered
    images.present[2] = true;
    images.images[2] = color;              // wrong type: must be filtered
    const auto product = pipeline.runProduct(images, opaqueModel);
    if (!check(product.valid && product.status == RunStatus::Ok && fake->productCalls == 1,
               "injected product algorithm must be called")) return 1;
    if (!check(fake->productViews[0] && !fake->productViews[1] && !fake->productViews[2],
               "product input must contain only validated views")) return 1;
    if (!check(product.warnings.size() == 2,
               "invalid product views must be reported as warnings")) return 1;

    if (!check(!images.hasView(1) && images.hasView(0),
               "ImageSet::hasView must follow the present/non-empty contract")) return 1;
    return 0;
}
