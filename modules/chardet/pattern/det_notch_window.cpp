// det_notch_window.cpp —— 定点窗口连续性检测（左上角缺口）
//
// ================================================================
// 这个检测器要抓的缺陷
// ================================================================
//
// 目视确认：ngv2 全部 7 张、unset 的一批、image/ok 里 18 张，
// 缺失都在**圆环左上角的同一个位置**，形态是环线断开一小段（约 1~4 度）。
// 干净的料此处环线完整闭合。
//
// ================================================================
// 为什么之前的"整圈找最大缺口"全部失败
// ================================================================
//
// 缺口只有 1~4 度，占整圈的 0.3%~1.1%。
// 环带纹理的逐角度波动幅度与缺口深度相当，
// 于是"全局最大缺口"这个统计量被纹理噪声主导，真缺口淹没在里面。
// 之前九轮失败（绝对阈值、相对墨量、角度、拓扑、纹理、PCA、批次、外接）
// 都是在整圈上做文章，根源就在这里。
//
// 把范围限制到缺口实际所在的窗口后，信号立即清晰：
//   ngv2  窗口内最低相对墨量 0.041 ~ 0.357
//   ok    窗口内最低相对墨量 0.229 ~ 0.643（其中低的那些经目视确认有真缺口）
//
// ================================================================
// 判据
// ================================================================
//
//   severity 由两件事共同决定：
//     ① 窗口内最低相对墨量有多低（缺口有多深）
//     ② 连续低于门限的角度跨度有多长（缺口有多宽）
//
// 单独用任何一个都不够：深度会被单个噪声点命中，
// 宽度会被整段变淡但没断的料命中。
// 两者相乘才对应"这里真的断开了一小段"。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "pattern_api.hpp"

namespace chardet {
namespace pattern {

namespace {

// ---- 默认参数（改这里就等于换一个检测器版本，务必在文档里记下依据）
//
// 窗口位置：240°~272°
//   极坐标 0° = 环的最右侧，90° = 环的最下方，逆时针增加。
//   270° = 环的正上方，240° 在其左，所以 240°~272° 覆盖左上角到正上方。
//   这个位置是从实测角度差分布上定出来的：240° 处 ngv2 与正常料的
//   平均墨量差 +0.203（全环最大），250° 处 +0.160。
constexpr double kWinFrom = 240.0;
constexpr double kWinTo = 272.0;

// 相对墨量低于此值算"这一列没有环线"
constexpr double kMissThr = 0.45;

// 缺口的深度量程：rel 从 1.0（环线正常）降到 kDeepRel 时，深度项取满
constexpr double kDeepRel = 0.10;

// 缺口达到此跨度时，宽度项取满
constexpr double kWideDeg = 3.0;

// 至少这么宽才算一个真缺口（更短的是刻印毛刺）
constexpr double kMinGapDeg = 0.6;

// severity 达到此值判为触发
constexpr double kTriggerSeverity = 0.5;

/// 对窗口内的 rel 序列，求最长"连续低于 kMissThr"的跨度
struct GapStat {
    double widthDeg = 0.0;   // 最长缺口宽度（度）
    double minRel = 1.0;     // 窗口内最低相对墨量
    int startIdx = -1;       // 最长缺口的起始角度索引（-1 = 没缺口）
};

GapStat measureWindow(const Profile& prof, const PolarMap& pm, double fromDeg, double toDeg) {
    GapStat st;
    const int N = static_cast<int>(prof.size());
    if (N <= 0) return st;
    const double degPerIdx = 360.0 / N;

    const int i0 = pm.idxOf(fromDeg);
    const int i1 = pm.idxOf(toDeg);
    int nWin = i1 - i0;
    if (nWin <= 0) nWin += N;

    int run = 0, bestRun = 0, curStart = -1, bestStart = -1;
    for (int k = 0; k < nWin; ++k) {
        const int idx = (i0 + k) % N;
        const double r = prof.relAt(idx);
        st.minRel = std::min(st.minRel, r);
        if (r < kMissThr) {
            if (run == 0) curStart = k;
            ++run;
            if (run > bestRun) {
                bestRun = run;
                bestStart = curStart;
            }
        } else {
            run = 0;
        }
    }
    st.widthDeg = bestRun * degPerIdx;
    st.startIdx = bestStart >= 0 ? (i0 + bestStart) % N : -1;
    return st;
}

}  // namespace

/// 定点窗口连续性检测器
class NotchWindowDetector final : public IDetector {
public:
    const char* name() const override { return "notch_window"; }

    Finding detect(const Input& in, const Geometry& geom, const PolarMap& pm,
                   const Profile& prof) override {
        (void)in;
        Finding f;
        if (!prof.valid() || !pm.valid()) {
            f.detail = "剖面无效";
            return f;
        }

        const GapStat st = measureWindow(prof, pm, kWinFrom, kWinTo);

        // ---- 深度项：缺口最低点比环线中位低多少
        //
        // st.minRel 越小说明缺口越深。
        // rel=1.0（与环线一样亮）→ 0；rel<=kDeepRel → 1
        double depth = 0.0;
        if (st.minRel < 1.0) {
            depth = (1.0 - st.minRel) / (1.0 - kDeepRel);
        }
        depth = std::clamp(depth, 0.0, 1.0);

        // ---- 宽度项：缺口有多宽
        double width = st.widthDeg / kWideDeg;
        width = std::clamp(width, 0.0, 1.0);

        // ---- 综合
        //
        // 用相乘而不是相加：两者都高才是"又深又宽的真缺口"。
        // 相加会让"极深但只有 0.1 度"（噪点）和"极宽但只淡了一点"（没断）
        // 都拿到中间的分数。
        const double sev = depth * width;

        f.score = st.widthDeg * 10.0 + st.minRel;   // 仅供显示，不是判据
        f.severity = sev;
        f.triggered = (sev >= kTriggerSeverity) && (st.widthDeg >= kMinGapDeg);

        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "窗口 %.0f~%.0f°：缺口 %.2f°，最低相对墨量 %.3f，"
                      "深度 %.2f × 宽度 %.2f = %.2f",
                      kWinFrom, kWinTo, st.widthDeg, st.minRel, depth, width, sev);
        f.detail = buf;

        // ---- 缺陷位置：缺口中心角度 → 全图坐标
        if (st.startIdx >= 0 && st.widthDeg > 0.0) {
            const double centerDeg = pm.degOf(st.startIdx) + st.widthDeg * 0.5;
            const double rad = centerDeg * CV_PI / 180.0;
            // 拉直后环是半径 a 的正圆
            const cv::Point2f inStraight(
                geom.center.x + static_cast<float>(geom.radiusA * std::cos(rad)),
                geom.center.y + static_cast<float>(geom.radiusA * std::sin(rad)));
            f.at.push_back(geom.toOriginal(inStraight));
            const int a0 = st.startIdx;
            const int a1 = (st.startIdx + static_cast<int>(std::lround(st.widthDeg *
                                                                      pm.angles / 360.0))) %
                           pm.angles;
            f.spansDeg.emplace_back(static_cast<int>(std::lround(pm.degOf(a0))),
                                    static_cast<int>(std::lround(pm.degOf(a1))));
        }
        return f;
    }
};

std::unique_ptr<IDetector> makeNotchWindowDetector() {
    return std::make_unique<NotchWindowDetector>();
}

}  // namespace pattern
}  // namespace chardet
