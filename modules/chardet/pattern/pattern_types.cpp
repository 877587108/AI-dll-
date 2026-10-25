// pattern_types.cpp —— 规范化数据结构的成员实现
//
// 这里只放**纯计算**：坐标换算、索引折回、参数转换。
// 任何涉及图像处理的代码都不该出现在这里。
#include "pattern_types.hpp"

#include <algorithm>
#include <cmath>

namespace chardet {
namespace pattern {

// ---------------------------------------------------------------- Params

Params Params::fromLegacy(const PatternBreakParams& L) {
    Params p;
    p.angles = L.angles;
    p.inkThr = L.inkThr;
    p.innerBlockRatio = L.innerBlockRatio;
    p.bandInnerRatio = L.bandInnerRatio;
    p.bandOuterRatio = L.bandOuterRatio;
    p.preGrayThr = L.preGrayThr;
    // 汇总门槛：老逻辑是 score >= minScore 即报，score = 角度×100。
    // 新逻辑用 severity，换算成同一物理含义：severity = score / 1000。
    // minScore=400 → minSeverity=0.4，口径一致。
    p.minSeverity = L.minScore / 1000.0;
    return p;
}

// ---------------------------------------------------------------- Geometry

cv::Point2f Geometry::toStraight(cv::Point2f p) const {
    if (affine.empty()) return p;
    const double x = affine.at<double>(0, 0) * p.x + affine.at<double>(0, 1) * p.y +
                     affine.at<double>(0, 2);
    const double y = affine.at<double>(1, 0) * p.x + affine.at<double>(1, 1) * p.y +
                     affine.at<double>(1, 2);
    return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}

cv::Point2f Geometry::toOriginal(cv::Point2f p) const {
    if (affine.empty()) return p;
    cv::Mat inv;
    try {
        inv = affine(cv::Rect(0, 0, 3, 2)).inv(cv::DECOMP_LU);
    } catch (const cv::Exception&) {
        return p;
    }
    const double x = inv.at<double>(0, 0) * p.x + inv.at<double>(0, 1) * p.y + inv.at<double>(0, 2);
    const double y = inv.at<double>(1, 0) * p.x + inv.at<double>(1, 1) * p.y + inv.at<double>(1, 2);
    return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}

// ---------------------------------------------------------------- PolarMap

int PolarMap::idxOf(double deg) const {
    if (angles <= 0) return 0;
    double d = std::fmod(deg, 360.0);
    if (d < 0) d += 360.0;
    int i = static_cast<int>(std::lround(d * angles / 360.0));
    // 359.9 度可能四舍五入到 angles，折回 0
    if (i >= angles) i %= angles;
    if (i < 0) i = 0;
    return i;
}

float PolarMap::at(int r, int aIdx) const {
    if (!valid()) return 0.0f;
    if (r < 0 || r >= radii) return 0.0f;
    int c = aIdx % angles;
    if (c < 0) c += angles;
    return img.at<float>(r, c);
}

// ---------------------------------------------------------------- Profile

double Profile::relAt(int i) const {
    if (rel.empty()) return 0.0;
    const int n = static_cast<int>(rel.size());
    int k = i % n;
    if (k < 0) k += n;
    return rel[static_cast<size_t>(k)];
}

// ---------------------------------------------------------------- Finding

Finding Finding::atSpan(double fromDeg, double toDeg, double sev, double sc,
                        const std::string& d) {
    Finding f;
    f.triggered = sev > 0.0;
    f.severity = sev;
    f.score = sc;
    f.detail = d;
    const int lo = static_cast<int>(std::lround(std::min(fromDeg, toDeg)));
    const int hi = static_cast<int>(std::lround(std::max(fromDeg, toDeg)));
    f.spansDeg.emplace_back(lo, std::max(lo, hi));
    return f;
}

// ---------------------------------------------------------------- Result

const Finding* Result::find(const std::string& name) const {
    for (const auto& kv : findings) {
        if (kv.first == name) return &kv.second;
    }
    return nullptr;
}

}  // namespace pattern
}  // namespace chardet
