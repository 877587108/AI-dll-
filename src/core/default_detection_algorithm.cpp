#include "core/detection_algorithm.hpp"

#include <algorithm>
#include <memory>

#include "core/algorithm_internal.hpp"

namespace markloc::algorithm {
namespace {

RoiResult convertRoi(const charinsp::RoiResult& in) {
    RoiResult out;
    out.ecc = in.ecc;
    out.scratch = in.scratch;
    out.scratchCoverage = in.scratchCoverage;
    out.scratchLen = in.scratchLen;
    out.scratchMeanS = in.scratchMeanS;
    out.scratchSevere = in.scratchSevere;
    out.appendage = in.appendage;
    out.missing = in.missing;
    out.missingArea = in.missingArea;
    out.worstChar = in.worstChar;
    out.worstCharIndex = in.worstCharIndex;
    out.score = in.score;
    out.reliable = in.reliable;
    return out;
}

RoiEvidence convertEvidence(const charinsp::RoiEvidence& in) {
    RoiEvidence out;
    out.aligned8 = in.aligned8;
    out.signal8 = in.s8;
    out.tolerance8 = in.tol8;
    out.extra8 = in.extra8;
    out.weak8 = in.weak8;
    out.valid = in.valid;
    return out;
}

ViewResult convertView(const charinsp::Result& in) {
    ViewResult out;
    for (int i = 0; i < kRegionCount; ++i) {
        out.regions[static_cast<size_t>(i)] = convertRoi(in.regions[static_cast<size_t>(i)]);
        out.evidence[static_cast<size_t>(i)] =
            convertEvidence(in.evidence[static_cast<size_t>(i)]);
    }
    out.regionMax = in.regionMax;
    out.worstRegion = in.worstRegion;
    out.score = in.score;
    out.ng = in.ng;
    out.blockExtra = in.blockExtra;
    out.blockMissing = in.blockMissing;
    out.chars.reserve(in.chars.size());
    for (const auto& c : in.chars) {
        CharResult item;
        item.roiIndex = c.roiIndex;
        item.charIndex = c.charIndex;
        item.rect = c.rect;
        item.coordinateSpace = CoordinateSpace::FullImage;
        item.score = c.score;
        item.missScore = c.missScore;
        item.extraScore = c.extraScore;
        item.shapeScore = c.shapeScore;
        item.missRatio = c.missRatio;
        item.extraRatio = c.extraRatio;
        item.label = c.label;
        out.chars.push_back(std::move(item));
    }
    return out;
}

ProductResult convertProduct(const charinsp::ProductResult& in) {
    ProductResult out;
    for (int v = 0; v < kViewCount; ++v) {
        const auto index = static_cast<size_t>(v);
        out.views[index] = convertView(in.views[index]);
        out.present[index] = in.present[index];
        out.viewScore[index] = in.viewScore[index];
    }
    out.score = in.score;
    out.regionScore = in.regionScore;
    out.worstView = in.worstView;
    out.ng = in.ng;
    return out;
}

class DefaultDetectionAlgorithm final : public IDetectionAlgorithm {
public:
    ViewRun runView(const cv::Mat& gray, int view, const ModelSnapshot& snapshot,
                    bool wantEvidence) const override {
        ViewRun out;
        out.view = view;
        const auto* data = detail::modelData(snapshot);
        if (!data || !snapshot.valid || view < 0 || view >= charinsp::kViewCount ||
            !data->model.valid(view)) {
            out.status = RunStatus::ModelUnavailable;
            out.error = "模型或视角无效";
            return out;
        }
        const auto index = static_cast<size_t>(view);
        const auto& vm = data->model.views[index];
        out.result = convertView(charinsp::analyze(
            gray, vm, data->model.params[index], wantEvidence));
        out.status = RunStatus::Ok;
        out.valid = true;
        return out;
    }

    ProductRun runProduct(const ImageSet& images, const ModelSnapshot& snapshot,
                          bool wantEvidence) const override {
        ProductRun out;
        const auto* data = detail::modelData(snapshot);
        if (!data || !snapshot.valid) {
            out.status = RunStatus::ModelUnavailable;
            out.error = "模型无效";
            return out;
        }

        std::array<cv::Mat, charinsp::kViewCount> mats{};
        std::array<bool, charinsp::kViewCount> present{};
        for (int v = 0; v < charinsp::kViewCount; ++v) {
            const auto index = static_cast<size_t>(v);
            mats[index] = images.images[index];
            present[index] = images.present[index];
        }
        const auto result = charinsp::analyzeProduct(
            mats, present, data->model, data->model.params, wantEvidence);
        out.result = convertProduct(result);
        out.status = RunStatus::Ok;
        out.valid = true;
        return out;
    }
};

} // namespace

std::shared_ptr<const IDetectionAlgorithm> makeDefaultDetectionAlgorithm() {
    static const std::shared_ptr<const IDetectionAlgorithm> instance =
        std::make_shared<DefaultDetectionAlgorithm>();
    return instance;
}

} // namespace markloc::algorithm
