#include "core/algorithm_internal.hpp"

namespace markloc::algorithm {

bool ModelSnapshot::hasView(int view) const noexcept {
    return data_ && view >= 0 && view < charinsp::kViewCount && data_->model.valid(view);
}

cv::Size ModelSnapshot::imageSize(int view) const noexcept {
    if (!data_ || view < 0 || view >= charinsp::kViewCount) return {};
    const auto& vm = data_->model.views[static_cast<size_t>(view)];
    // fullFreq 是唯一记录整图尺寸的公共模型信息。旧模型没有它时不做尺寸猜测，
    // 让算法适配器自行处理兼容格式。
    return vm.fullFreq.empty() ? cv::Size{} : vm.fullFreq.size();
}

namespace detail {
const ModelData* modelData(const ModelSnapshot& snapshot) noexcept {
    return snapshot.data_.get();
}
} // namespace detail

} // namespace markloc::algorithm
