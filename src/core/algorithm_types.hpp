#pragma once

// 应用层稳定数据契约。
//
// 这里故意不包含 char_core.hpp，也不把 charinsp::Model/Result 暴露给调用方。
// charinsp 是当前算法实现；本文件中的类型是 GUI、CLI 和后续算法实现共同依赖的
// 数据格式。修改算法内部结构时，只需要更新适配器，不需要改应用层。

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace markloc::algorithm {

inline constexpr int kViewCount = 6;
inline constexpr int kRegionCount = 4;

using Path = std::filesystem::path;

enum class SampleLabel { Unknown, Ok, Ng };

/// 一次运行的结果状态。正常的“跳过某个视角”不使用异常表达。
enum class RunStatus {
    Ok,
    InvalidInput,
    ModelUnavailable,
    NoUsableView,
    AlgorithmError,
};

/// 字符框坐标系。公共结果中的字符框默认是整图坐标。
enum class CoordinateSpace { FullImage, Roi };

struct Sample {
    Path path;
    int view = -1;                 // 0..5；无法解析时为 -1
    std::string product;
    SampleLabel label = SampleLabel::Unknown;
};

struct DatasetIndex {
    std::vector<Sample> samples;   // 只放可解析视角的图片
    std::vector<std::string> warnings;
};

/// 一个产品的多视角输入。present[v] 为真时 images[v] 必须是非空灰度图。
struct ImageSet {
    std::array<cv::Mat, kViewCount> images;
    std::array<bool, kViewCount> present{};
    std::string product;

    bool hasView(int view) const noexcept {
        return view >= 0 && view < kViewCount && present[static_cast<size_t>(view)] &&
               !images[static_cast<size_t>(view)].empty();
    }
};

class ModelSnapshot;
namespace detail {
struct ModelData;
const ModelData* modelData(const ModelSnapshot&) noexcept;
}

/// 已加载模型的只读句柄。
///
/// 模型文件格式和当前 charinsp::Model 都属于实现细节，应用层只能询问模型是否
/// 有效、某视角是否可用以及期望图像尺寸。句柄可廉价复制，底层数据共享且只读。
class ModelSnapshot {
public:
    Path root;
    std::string id;
    bool valid = false;

    bool hasView(int view) const noexcept;
    cv::Size imageSize(int view) const noexcept;

private:
    std::shared_ptr<const detail::ModelData> data_;

    friend class ModelRepository;
    friend const detail::ModelData* detail::modelData(const ModelSnapshot&) noexcept;
};

/// 单区域结果 DTO。字段含义和单位在 docs/data-contracts.md 中固定。
struct RoiResult {
    double ecc = -1.0;
    double scratch = 0.0;
    double scratchCoverage = 0.0;
    double scratchLen = 0.0;
    double scratchMeanS = 0.0;
    double scratchSevere = 0.0;
    double appendage = 0.0;
    double missing = 0.0;
    double missingArea = 0.0;
    double worstChar = 0.0;
    int worstCharIndex = -1;
    double score = 0.0;
    bool reliable = false;
};

struct RoiEvidence {
    cv::Mat aligned8;              // CV_8UC1
    cv::Mat signal8;               // CV_8UC1，可视化用
    cv::Mat tolerance8;            // CV_8UC1
    cv::Mat extra8;                // CV_8UC1
    cv::Mat weak8;                 // CV_8UC1
    bool valid = false;
};

struct CharResult {
    int roiIndex = -1;
    int charIndex = -1;
    cv::Rect rect;
    CoordinateSpace coordinateSpace = CoordinateSpace::FullImage;
    double score = 0.0;
    double missScore = 0.0;
    double extraScore = 0.0;
    double shapeScore = 0.0;
    double missRatio = 0.0;
    double extraRatio = 0.0;
    std::string label;
};

struct ViewResult {
    std::array<RoiResult, kRegionCount> regions;
    double regionMax = 0.0;
    int worstRegion = -1;
    double score = 0.0;
    bool ng = false;
    int blockExtra = 0;
    int blockMissing = 0;
    std::array<RoiEvidence, kRegionCount> evidence;
    std::vector<CharResult> chars;
};

struct ProductResult {
    std::array<ViewResult, kViewCount> views;
    std::array<bool, kViewCount> present{};
    std::array<double, kViewCount> viewScore{};
    double score = 0.0;
    double regionScore = 0.0;
    int worstView = -1;
    bool ng = false;
};

struct PipelineOptions {
    bool evidence = true;
    bool rejectSizeMismatch = true;
    bool requireGray8 = true;
    bool requireValidModel = true;
};

struct ViewRun {
    int view = -1;
    RunStatus status = RunStatus::InvalidInput;
    bool valid = false;
    ViewResult result;
    std::string error;
};

struct ProductRun {
    RunStatus status = RunStatus::NoUsableView;
    bool valid = false;
    ProductResult result;
    std::vector<std::string> warnings;
    std::string error;
};

} // namespace markloc::algorithm
