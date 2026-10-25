#include "core/detectors.hpp"

#include <algorithm>

namespace markloc::charinsp::det {

double runDetectors(const DetectorInput& input, RoiResult& result) {
    if (input.aligned.empty() || input.aligned.type() != CV_32F ||
        input.extra.empty() || input.weak.empty() || input.extra.size() != input.aligned.size() ||
        input.weak.size() != input.aligned.size()) {
        return 0.0;
    }

    result.appendage = input.allowAppendage ? scoreAppendage(input.aligned, input.extra) : 0.0;
    scoreScratch(input.aligned, input.weak, input.params, result);

    double score = result.scratchSevere / std::max(1.0, input.params.scratchT);
    if (input.allowAppendage) {
        score = std::max(score, result.appendage / std::max(1.0, input.params.appendageT));
    }
    if (input.params.checkMissing) {
        score = std::max(score, scoreMissing(input.aligned, input.frequency, input.params, result));
    }
    return score;
}

} // namespace markloc::charinsp::det
