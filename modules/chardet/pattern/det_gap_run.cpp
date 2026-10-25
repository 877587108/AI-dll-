// det_gap_run.cpp —— 整圈缺口游程检测（原有判据的适配实现）
//
// ================================================================
// 这是老的判据，行为必须与重构前逐位一致
// ================================================================
//
// 原有实现是 analyzePatternBreak() 里那 500 行。
// 它的逻辑是：整圈 360° 找所有缺口，合并相邻的，再按
//   · 长度在 [minGapAngle, maxGapAngle] 之间
//   · 两侧各有至少 minFlankCount 个连续有墨点
// 判定哪个缺口算"断点"，最后 score = 最长断口角度 × 100。
//
// 这个检测器**只负责评分那一半**——缺口怎么找、怎么合并、
// 怎么判断点，仍留在 pattern_break.cpp 的老代码里。
// 原因：那部分与 PatternBreakResult 的多个字段纠缠，
// 强行搬过来会改变输出，违反"重构前后逐张比对必须一致"的验收要求。
//
// 所以这里是一个**桥接**：让老结果也能出现在新流水线的 findings 里，
// 便于并排比较新旧判据。等新判据稳定并且验证充分后，
// 可以把老的缺口游程逻辑整个搬进来，然后删掉 pattern_break.cpp。
#include <cmath>
#include <cstdio>
#include <string>

#include "pattern_api.hpp"

namespace chardet {
namespace pattern {

namespace {

// 老判据的断点长度门槛（与 PatternBreakParams 默认值一致）
constexpr double kMinGapAngle = 0.6;
constexpr double kMaxGapAngle = 12.0;

// 老判据的评分门槛：score = 角度×100，minScore=400 即 4.0°
constexpr double kMinScoreAngle = 4.0;

}  // namespace

/// 整圈缺口游程检测器
///
/// 这里用剖面重新算一遍老判据：每个角度 rel < inkThr/ringMedian 就算无墨。
/// 与老实现在数值上可能有细微差别（老实现用的是极坐标图上的原始值，
/// 且有半径带连续像素数要求），所以**以老结果为准**，
/// 本检测器只用于实验对比，不参与默认判决。
class GapRunDetector final : public IDetector {
public:
    const char* name() const override { return "gap_run"; }

    Finding detect(const Input& in, const Geometry& geom, const PolarMap& pm,
                   const Profile& prof) override {
        (void)in;
        (void)geom;
        Finding f;
        if (!prof.valid() || !pm.valid()) {
            f.detail = "剖面无效";
            return f;
        }

        const int N = static_cast<int>(prof.size());
        const double degPerIdx = 360.0 / N;
        const double missThr = 0.35;   // 与 PatternBreakParams::inkThr 一致

        // 整圈找缺口，并合并间隔 <= mergeGapCount 的相邻缺口
        std::vector<std::pair<int, int>> gaps;   // [start, len)
        int run = 0, start = 0;
        for (int i = 0; i < N; ++i) {
            if (prof.relAt(i) < missThr) {
                if (run == 0) start = i;
                ++run;
            } else if (run > 0) {
                gaps.emplace_back(start, run);
                run = 0;
            }
        }
        if (run > 0) {
            if (!gaps.empty() && gaps.front().first == 0) {
                // 首尾相接，合并成一个跨 0° 的缺口
                gaps.front().second += run;
            } else {
                gaps.emplace_back(start, run);
            }
        }
        // 合并靠得近的
        constexpr int kMerge = 6;
        std::vector<std::pair<int, int>> merged;
        for (const auto& g : gaps) {
            if (!merged.empty()) {
                auto& last = merged.back();
                const int lastEnd = (last.first + last.second) % N;
                int dist = g.first - lastEnd;
                if (dist < 0) dist += N;
                if (dist <= kMerge) {
                    int grow = dist + g.second;
                    last.second += grow;
                    if (last.second > N) last.second = N;
                    continue;
                }
            }
            merged.push_back(g);
        }

        double maxGapDeg = 0.0;
        int maxStart = -1;
        for (const auto& g : merged) {
            const double deg = g.second * degPerIdx;
            if (deg > maxGapDeg) {
                maxGapDeg = deg;
                maxStart = g.first;
            }
        }

        const bool inRange = maxGapDeg >= kMinGapAngle && maxGapDeg <= kMaxGapAngle;
        f.score = maxGapDeg * 100.0;
        // severity 换算：老门槛 4.0° 对应 severity 0.5，量程到 8.0° 取满
        f.severity = std::clamp(maxGapDeg / 8.0, 0.0, 1.0);
        f.triggered = inRange && maxGapDeg >= kMinScoreAngle;

        char buf[192];
        std::snprintf(buf, sizeof(buf), "整圈最大缺口 %.2f°（%zu 段缺口，判据门槛 %.1f°）",
                      maxGapDeg, merged.size(), kMinScoreAngle);
        f.detail = buf;

        if (maxStart >= 0 && maxGapDeg > 0.0) {
            const int a0 = static_cast<int>(std::lround(pm.degOf(maxStart)));
            const int a1 = static_cast<int>(std::lround(pm.degOf(maxStart) + maxGapDeg));
            f.spansDeg.emplace_back(a0, a1);
        }
        return f;
    }
};

std::unique_ptr<IDetector> makeGapRunDetector() {
    return std::make_unique<GapRunDetector>();
}

}  // namespace pattern
}  // namespace chardet
