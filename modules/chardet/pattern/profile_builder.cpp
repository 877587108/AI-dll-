// profile_builder.cpp —— 环带剖面：把环上的墨量量化成一条曲线
//
// ================================================================
// 这个阶段做什么
// ================================================================
//
// 输入：极坐标图（行=半径、列=角度）
// 输出：每个角度在环带上能取到的最大墨量，即 Profile::ink
//
// 这一步是纯量化，不做任何判断。
// 所有"沿周长看"的判据（缺口游程、定点窗口、凹陷深度、相关……）
// 都建立在同一条曲线上，于是它们之间是可比的。
//
// ================================================================
// 为什么取"半径带内的最大值"而不是"跟踪最亮的半径"
// ================================================================
//
// 老做法是每个角度找半径方向最亮的位置 r(θ)，再沿这条曲线判有无墨。
// 它假设"环在任何一个角度都只对应一个半径"。但实测的环是一段粗一段细的
// 闭合带，"最亮处"会在带的两条边缘之间来回跳，跳动幅度就等于起伏量
// （83 张 OK 上中位 28px）。所以那个起伏量测到的是环的粗细变化，
// 不是对准误差——0109_RT 起伏 48px，但覆盖率 1.000、断口 0。
//
// 取带内最大值就绕开了这个问题：环粗一段细一段完全不影响结果。
// 代价是"环整体变淡"和"环局部断开"都表现为这条曲线变低，
// 所以下游检测器要自己区分这两件事（这正是各算法之间的差别所在）。
#include <algorithm>
#include <cmath>
#include <vector>

#include "pattern_api.hpp"

namespace chardet {
namespace pattern {

namespace {

/// 把一组数排序后取中位数（不修改输入）
double medianCopy(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    return v[mid];
}

}  // namespace

/// 标准剖面：半径带内逐角度取最大墨量
class BandMaxProfileBuilder final : public IProfileBuilder {
public:
    const char* name() const override { return "band_max"; }

    bool build(const PolarMap& pm, const Geometry& geom, const Params& p,
               Profile& out) override {
        out = Profile{};
        if (!pm.valid() || !geom.ok) return false;

        const double a = static_cast<double>(geom.radiusA);
        if (a < 2.0) return false;

        // 半径带：以长半轴为单位，换算成像素索引
        int lo = static_cast<int>(std::lround(a * p.bandInnerRatio));
        int hi = static_cast<int>(std::lround(a * p.bandOuterRatio));
        lo = std::max(0, std::min(lo, pm.radii - 1));
        hi = std::max(lo, std::min(hi, pm.radii - 1));
        if (hi - lo < 1) return false;

        const int N = pm.angles;
        out.ink.assign(static_cast<size_t>(N), 0.0);
        for (int x = 0; x < N; ++x) {
            double peak = 0.0;
            for (int r = lo; r <= hi; ++r) {
                peak = std::max(peak, static_cast<double>(pm.img.at<float>(r, x)));
            }
            out.ink[static_cast<size_t>(x)] = peak;
        }
        out.bandLo = lo;
        out.bandHi = hi;

        // 归一化基准：全环墨量中位数
        //
        // 用中位数而不是均值：缺口会把均值拉低，中位数不受影响，
        // 于是 rel 反映的是"相对这枚料自身环的浓淡"，天然抵消曝光差异。
        out.ringMedian = medianCopy(out.ink);
        if (out.ringMedian < 1e-6) {
            // 整圈几乎没墨：不是缺口而是整条没印出来。
            // 下游会看到 rel 全为 0，各检测器自行处理这种情况。
            out.ringMedian = 1e-6;
        }
        out.rel.assign(static_cast<size_t>(N), 0.0);
        for (int x = 0; x < N; ++x) {
            out.rel[static_cast<size_t>(x)] =
                out.ink[static_cast<size_t>(x)] / out.ringMedian;
        }
        return true;
    }
};

std::unique_ptr<IProfileBuilder> makeProfileBuilder() {
    return std::make_unique<BandMaxProfileBuilder>();
}

}  // namespace pattern
}  // namespace chardet
