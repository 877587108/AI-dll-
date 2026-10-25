// pattern_break.cpp —— 图案区域（椭圆/圆）的断点检测
//
// 与字符区的检测完全分开：
//
//   字符区（Box）    用 diffNcc 做字符级比对：缺墨、多墨、局部新增墨块。
//                    判据关心"字的墨量对不对"。
//
//   图案区（Ellipse/Circle）  只有一件事：这条闭合曲线有没有被切断。
//                    不能用"框内墨量"，因为椭圆环在框里只占一条细线，
//                    框内九成是背景，任何比例判据都被稀释。
//
// ================================================================
// 椭圆从哪来：直接用选框，不拟合
// ================================================================
//
// 走过两条弯路，都记在这里，别再重复：
//
//   弯路一：把区域矩形直接当椭圆外接框。
//     logo 是斜的椭圆环，正矩形和它对不上，沿周长采样一半落在背景上。
//     84 张 OK 图报出 21 个假断口。
//
//   弯路二：从区域内的墨点迭代拟合椭圆。
//     拟合本身能收敛，但拟出来的参数是"墨的惯性椭圆"，
//     中心、半轴、角度都会随缺陷本身漂移 ——
//     缺陷改变了墨的分布，拟合结果也跟着变，
//     于是"判有没有缺陷"变成了"判拟合结果稳不稳"，自相矛盾。
//
//   正确做法：椭圆就是选框本身。
//     中心 = 框中心，长半轴 = max(宽,高)/2，短半轴 = min(宽,高)/2，
//     长轴角度 = 框带的旋转角（RegionInfo::angleDeg）。
//     离心率 e = sqrt(1-(b/a)^2) 由宽高比唯一确定，不需要额外参数。
//     框是建模时人定的，和缺陷无关，所以判据不会自己漂。
//
// ================================================================
// 怎么判断口：沿椭圆拉直成正圆，再做极坐标
// ================================================================
//
// 椭圆在极坐标下半径随角度变化（长轴方向 a、短轴方向 b），
// 直接极坐标展开得到的是一条波浪线，不是水平带，没法按半径限制。
//
// 所以先把椭圆拉直成正圆（仿射变换），环就落在固定半径上。
// 这一步有个必须做的自检：长短轴端点变换后都应落在离中心 a 的位置。
// 第一版把缩放写成了"除以 a/b"，自检报短轴端距 40（应 137），
// 就是靠这个自检抓出来的。
//
// 拉直之后仍然不能用一个固定的半径窗口：
// 实测环的半径在角度方向上仍有起伏（刻印本身不圆、框也不是像素级精确）。
// 窗口窄了，起伏的波谷被当成缺口；窗口宽了，真缺口被两侧的墨填平。
// 所以按角度跟踪环的实际半径曲线 r(θ)，再沿这条曲线判有无墨。
#include "chardet.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "pattern/pattern_api.hpp"

namespace chardet {

namespace {

struct Gap {
    int start = 0;   // 起点角度索引
    int len = 0;     // 持续角度数
    int flank = 0;   // 两侧连续有墨的最短长度
};

/// 长方体的椭圆点（框定义，不拟合）
struct BoxEllipse {
    double cx = 0, cy = 0;
    double a = 1, b = 1;   // 长半轴、短半轴
    double rot = 0;        // 长轴角度（弧度）
};

/// 椭圆参数直接取自区域（不拟合）
///
/// 几何描述存在 RegionInfo::ell 里：中心 + 长半轴 + 短半轴 + 长轴角度。
/// 不再用"轴对齐矩形反推"——那有多解，同一个外接框能对应不同角度、
/// 不同半轴的椭圆，反推出来的不一定是你画的那个。
BoxEllipse ellipseFromRegion(const RegionInfo& r) {
    BoxEllipse e;
    e.cx = r.ell.center.x;
    e.cy = r.ell.center.y;
    e.a = std::max(1.0, static_cast<double>(std::max(r.ell.a, r.ell.b)));
    e.b = std::max(1.0, static_cast<double>(std::min(r.ell.a, r.ell.b)));
    e.rot = r.ell.angleDeg * CV_PI / 180.0;
    return e;
}

/// 椭圆参数角 th -> 全图坐标
inline void pointAt(const BoxEllipse& e, double th, double* px, double* py) {
    const double u = e.a * std::cos(th), v = e.b * std::sin(th);
    const double ca = std::cos(e.rot), sa = std::sin(e.rot);
    *px = e.cx + u * ca - v * sa;
    *py = e.cy + u * sa + v * ca;
}

}  // namespace

PatternBreakResult analyzePatternBreak(const cv::Mat& signal, const RegionInfo& region,
                                       const Config& cfg, const PatternBreakParams& P) {
    PatternBreakResult out;
    if (signal.empty() || region.rect.area() <= 0) return out;
    if (!region.isPattern()) return out;
    if (signal.type() != CV_32F) return out;

    const BoxEllipse e = ellipseFromRegion(region);
    out.fitted = true;   // 名字沿用：椭圆参数是"有"的（来自框），不是拟合来的
    out.fit.cx = e.cx;
    out.fit.cy = e.cy;
    out.fit.a = e.a;
    out.fit.b = e.b;
    out.fit.rot = e.rot;
    out.fit.support = 0;
    out.fit.resid = 0.0;
    out.valid = true;

    const double a = e.a, b = e.b;
    if (a < 4 || b < 4) return out;

    // ---- 1. 拉直：把椭圆变成半径 a 的正圆
    //
    // 四步显式构造，不用 getRotationMatrix2D（它的返回矩阵已含
    // "把旋转中心搬回该点"的平移，再乘缩放会把中心挪走）。
    //   1. 平移到原点  2. 旋转 -rot  3. y 方向乘 a/b  4. 平移回去
    const double kScale = a / std::max(1.0, b);
    const double cR = std::cos(-e.rot), sR = std::sin(-e.rot);
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -e.cx, 0, 1, -e.cy, 0, 0, 1);
    cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
    cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, e.cx, 0, 1, e.cy, 0, 0, 1);
    cv::Mat M3 = T2 * Sc * Rm * T1;
    cv::Mat M;
    M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
    cv::Mat straight;
    cv::warpAffine(signal, straight, M, signal.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);

    // ---- 1b. 展开前的灰度阈值：把拉直图上低于 preGrayThr 的像素直接清零
    //
    // 【为什么必须放在展开之前】
    // 环的墨量在一段范围内连续变化：粗的地方强、细的地方弱。
    // 如果先展开再按 inkThr 二值化，"弱"会被三像素连通判据兜住 ——
    // 只要还有 3 个像素过线就算有墨，于是缺口恒为 0，变淡的缺陷看不见。
    //
    // 在展开之前硬切一刀，效果是不同的：
    // 拉直图上淡掉的像素被置 0，展开后就是一段真实的空洞，
    // 后续的连通判定才能量出它的角度。这是"先定什么算墨，再数连通"的次序。
    //
    // 阈值作用在拉直图上，与 inkThr 各管一段，不能互相替代。
    if (P.preGrayThr > 0.0) {
        cv::threshold(straight, straight, P.preGrayThr, 0.0, cv::THRESH_TOZERO);
    }
    out.preGrayThr = P.preGrayThr;

    // 只用框附近的区域做极坐标，避免整幅图参与运算
    const int pad = static_cast<int>(a * 0.45) + 8;
    const cv::Rect roi =
        cv::Rect(static_cast<int>(e.cx - a - pad), static_cast<int>(e.cy - a - pad),
                 static_cast<int>(2 * (a + pad)), static_cast<int>(2 * (a + pad))) &
        cv::Rect(0, 0, signal.cols, signal.rows);
    if (roi.width < 8 || roi.height < 8) return out;

    const cv::Point2f centerInRoi(static_cast<float>(e.cx - roi.x),
                                  static_cast<float>(e.cy - roi.y));
    const double maxR = a + pad;

    // ---- 2. 极坐标展开
    //
    // warpPolar 的 dsize 约定：width = 半径数，height = 角度数（别记反）。
    // 输出转置成 rows=半径、cols=角度，方便按角度扫。
    const int nAngle = std::max(180, P.angles);
    const int nRadius = std::max(8, static_cast<int>(std::lround(maxR)));
    cv::Mat pol;
    cv::warpPolar(straight(roi), pol, cv::Size(nRadius, nAngle), centerInRoi, maxR,
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    cv::Mat polT;
    cv::transpose(pol, polT);
    pol = polT;   // rows = 半径 0..maxR, cols = 角度 0..360

    const int Wcols = pol.cols;
    const int aRow = std::clamp(static_cast<int>(std::lround(a)), 0, pol.rows - 1);

    // ---- 3. 半径限制：把环内区域整个屏蔽
    //
    // 【为什么必须屏蔽环内侧】
    // SC 这个 logo 的环里面还有 "SC" 两个字母。它们离中心比环近得多
    // （半径大致在 0.2a~0.55a），但对"沿法向找最亮"这种操作是致命干扰：
    // 环在某个角度刚好有点淡，法向一搜就搜到旁边的字母上，
    // 于是那个角度被判成"有墨"，真正的缺口反而没了。
    // 实测不屏蔽时环半径起伏能到 40px（正好是搜索窗口满量程，说明根本没找到环），
    // 而墨线本身只有 5~8px 宽 —— 这个尺度下任何判据都不成立。
    //
    // 阈值取 0.7a：字母都在 0.55a 以内，环在 a 附近，
    // 0.7a 处留出了余量，不会误伤环本身（环的内沿约 0.95a）。
    const int innerRow = static_cast<int>(std::lround(a * P.innerBlockRatio));
    for (int r = 0; r < std::min(innerRow, pol.rows); ++r) {
        for (int x = 0; x < Wcols; ++x) pol.at<float>(r, x) = 0.0f;
    }
    out.innerBlocked = innerRow;

    // ---- 4. 按角度跟踪环的半径曲线 r(θ)
    //
    // 环在半径 a 附近，起伏可能达到十几个像素。
    // 对每个角度取窗口内的 argmax 得到 r(θ)，再用中值滤波抹平：
    // 缺口那几个角度没有墨，argmax 会乱跳，必须靠邻居压回去。
    //
    // 窗口下界用 innerRow 而不是 aRow-trackHalf：环内侧已经屏蔽，
    // 再往下找只会找到被清零的像素，没有意义。
    const int trackHalf = std::max(4, static_cast<int>(a * 0.18));
    const int rLo = std::max(innerRow + 1, aRow - trackHalf);
    const int rHi = std::min(pol.rows - 1, aRow + trackHalf);
    std::vector<double> rTheta(Wcols, aRow), rSmooth(Wcols, aRow);
    double sumHalfWidth = 0.0;
    int nHalfWidth = 0;
    for (int x = 0; x < Wcols; ++x) {
        double bestV = -1e30;
        int bestR = aRow;
        for (int r = rLo; r <= rHi; ++r) {
            const double v = pol.at<float>(r, x);
            if (v > bestV) {
                bestV = v;
                bestR = r;
            }
        }
        rTheta[x] = bestR;

        // 顺便量这条墨线的半高宽：从最亮处往两侧走，
        // 掉到峰值一半就停，两侧步数的和就是这条线在半径方向的宽度。
        // 后面判断"有没有对准"要拿它做参照。
        if (bestV >= P.inkThr) {
            const double half = bestV * 0.5;
            int cnt = 1;
            for (int r = bestR - 1; r >= rLo; --r) {
                if (pol.at<float>(r, x) < half) break;
                ++cnt;
            }
            for (int r = bestR + 1; r <= rHi; ++r) {
                if (pol.at<float>(r, x) < half) break;
                ++cnt;
            }
            sumHalfWidth += cnt;
            ++nHalfWidth;
        }
    }
    const double radiusLineWidth = nHalfWidth > 0 ? sumHalfWidth / nHalfWidth : 6.0;
    out.radiusLineWidth = radiusLineWidth;
    {
        const int medW = std::max(2, static_cast<int>(Wcols * 0.01));   // 约 3.6°
        std::vector<double> buf;
        buf.reserve(2 * medW + 1);
        for (int x = 0; x < Wcols; ++x) {
            buf.clear();
            for (int k = -medW; k <= medW; ++k) buf.push_back(rTheta[(x + k + Wcols) % Wcols]);
            std::nth_element(buf.begin(), buf.begin() + buf.size() / 2, buf.end());
            rSmooth[x] = buf[buf.size() / 2];
        }

        // 起伏量要用稳健分位数，不能用 min/max。
        //
        // 【为什么】缺口处的 argmax 在窗口里到处乱跳（那里本来就没墨），
        // 会把 min/max 撑到窗口满量程。结果就是"洞越大，起伏越大"，
        // 大到一定程度反而被判成"没对准"而把结论清零 ——
        // 实测挖 8° 和 12° 洞时断口数掉回 0、分数归零，
        // 真缺陷被当成"没验过"放过去了。
        //
        // 现在只取 5%~95% 分位：缺口只占周长的几个百分点，
        // 影响不到分位数，而整体错位会让整条曲线平移，分位数照样反映。
        std::vector<double> sv = rSmooth;
        std::sort(sv.begin(), sv.end());
        const size_t n = sv.size();
        const double lo = sv[static_cast<size_t>(n * 0.05)];
        const double hi = sv[static_cast<size_t>(n * 0.95)];
        out.radiusSwing = std::max(0.0, hi - lo);
    }

    // ---- 4. 判每个角度"有没有墨"
    //
    // 【方法取舍：为什么不再沿 rSmooth 取 ±radiusTol 找峰】
    //
    // 老做法沿"每个角度最亮的半径"这条曲线取 ±3px 找峰。它隐含假设
    // 环在任何一个角度都只对应一个半径。但实测这批图的环是一段粗一段细的
    // 闭合带："最亮处"在带的两条边缘之间来回跳，跳幅就是前面的起伏量。
    // 于是判据变成"最亮点有没有落在跟踪曲线上"，而不是"环有没有断" ——
    // 环本身好好的（覆盖率 1.000、断口 0），也会因为最亮点跳开而丢墨。
    //
    // 现在改成：一个角度只要在整条半径带 [bandInner, bandOuter] 里
    // 能数出 minRunPixels 个连续超阈值的像素，就算该角度有墨。
    // 带的粗细变化完全不影响结果，缺口也不会因为"最亮点跑偏"而消失。
    //
    // 半径带的上下界都按长半轴 a 的比例给，不用绝对像素：
    // 环的实际半轴随拍摄距离变，用比例才跟着一起缩放。
    std::vector<char> on(Wcols, 0);
    int onCount = 0;
    if (P.bandConnected) {
        const int bLo = std::clamp(static_cast<int>(std::lround(a * P.bandInnerRatio)),
                                   innerRow, pol.rows - 1);
        const int bHi = std::clamp(static_cast<int>(std::lround(a * P.bandOuterRatio)), bLo + 1,
                                   pol.rows - 1);
        out.innerBlocked = bLo;
        out.bandOuter = bHi;
        const int need = std::max(1, P.minRunPixels);
        for (int x = 0; x < Wcols; ++x) {
            // 在半径带里找"最长的一段连续有墨"
            int run = 0, best = 0;
            for (int r = bLo; r <= bHi; ++r) {
                if (pol.at<float>(r, x) >= P.inkThr) {
                    if (++run > best) best = run;
                } else {
                    run = 0;
                }
            }
            on[x] = best >= need ? 1 : 0;
            onCount += on[x];
        }
    } else {
        // 老做法保留，供对照与回退。bandConnected=false 时走这里。
        for (int x = 0; x < Wcols; ++x) {
            const int cc = static_cast<int>(std::lround(rSmooth[x]));
            double peak = 0.0;
            for (int k = -P.radiusTol; k <= P.radiusTol; ++k) {
                const int r = std::clamp(cc + k, innerRow, pol.rows - 1);
                peak = std::max(peak, static_cast<double>(pol.at<float>(r, x)));
            }
            on[x] = peak >= P.inkThr ? 1 : 0;
            onCount += on[x];
        }
    }
    out.coverage = static_cast<double>(onCount) / Wcols;

    // ---- 5. 环形 0/1 序列 -> 缺口
    int startAt = -1;
    for (int t = 0; t < Wcols; ++t)
        if (on[t]) { startAt = t; break; }
    if (startAt < 0) {
        out.allMissing = true;
        out.maxGapAngle = 360.0;
        return out;
    }
    std::vector<Gap> gaps;
    {
        int cur = 0, curStart = -1;
        for (int k = 1; k <= Wcols; ++k) {
            const int t = (startAt + k) % Wcols;
            if (!on[t]) {
                if (cur == 0) curStart = t;
                ++cur;
            } else if (cur > 0) {
                gaps.push_back({curStart, cur, 0});
                cur = 0;
            }
        }
    }

    // ---- 6. 合并间隔很短的相邻缺口
    //
    // 缺口边缘残留的零星墨点会把一个贯通缺口切成几段。
    // 不合并的后果是"洞越大越检不出来" —— 挖洞验证里 8° 洞被切成 2 段、
    // 15° 洞切成 5 段，最长一段只剩 5.5°。
    {
        std::vector<Gap> merged;
        for (const auto& g : gaps) {
            if (!merged.empty()) {
                Gap& last = merged.back();
                const int endOfLast = (last.start + last.len) % Wcols;
                const int between = (g.start - endOfLast + Wcols) % Wcols;
                if (between <= P.mergeGapCount) {
                    last.len += between + g.len;
                    continue;
                }
            }
            merged.push_back(g);
        }
        if (merged.size() >= 2) {
            const Gap first = merged.front();
            Gap& last = merged.back();
            const int endOfLast = (last.start + last.len) % Wcols;
            const int between = (first.start - endOfLast + Wcols) % Wcols;
            if (between <= P.mergeGapCount) {
                last.len += between + first.len;
                merged.erase(merged.begin());
            }
        }
        gaps.swap(merged);
    }

    // ---- 7. 断口判定
    for (auto& g : gaps) {
        int before = 0, after = 0;
        for (int k = 1; k <= Wcols; ++k) {
            if (!on[(g.start - k + 2 * Wcols) % Wcols]) break;
            ++before;
        }
        for (int k = 0; k < Wcols; ++k) {
            if (!on[(g.start + g.len + k) % Wcols]) break;
            ++after;
        }
        g.flank = std::min(before, after);

        const double lenDeg = 360.0 * g.len / Wcols;
        out.maxGapAngle = std::max(out.maxGapAngle, lenDeg);
        out.gapCount += 1;

        // 断口 = 缺口长度在合理范围，且两侧都有够长的墨段。
        // 两侧都是长墨段才叫"环被切断"；只有一边的是边缘毛刺。
        const bool isBreak = lenDeg >= P.minGapAngle && lenDeg <= P.maxGapAngle &&
                             g.flank >= P.minFlankCount;
        if (isBreak) {
            ++out.breaks;
            out.maxBreakAngle = std::max(out.maxBreakAngle, lenDeg);
            // 断口位置换算回全图坐标：极坐标的角度 -> 拉直图上的点 -> 原图
            const double thMid = 2.0 * CV_PI * (g.start + g.len * 0.5) / Wcols;
            const double rMid = rSmooth[(g.start + g.len / 2) % Wcols];
            const double sx = roi.x + centerInRoi.x + rMid * std::cos(thMid);
            const double sy = roi.y + centerInRoi.y + rMid * std::sin(thMid);
            if (M3.type() == CV_64F) {
                const cv::Mat Minv = M3.inv();
                const double ux = Minv.at<double>(0, 0) * sx + Minv.at<double>(0, 1) * sy +
                                  Minv.at<double>(0, 2);
                const double uy = Minv.at<double>(1, 0) * sx + Minv.at<double>(1, 1) * sy +
                                  Minv.at<double>(1, 2);
                out.breakAt.emplace_back(static_cast<float>(ux), static_cast<float>(uy));
            } else {
                out.breakAt.emplace_back(static_cast<float>(sx), static_cast<float>(sy));
            }
        }
    }

    // ---- 8. 对准质量检查：没对准就不许下结论
    //
    // 半径起伏是"椭圆和实际环差多远"的直接度量。
    // 起伏大到一定程度，沿曲线采样已经找不到环了 —— 此时报出来的"缺口"
    // 只是椭圆和环错开留下的空隙，不是缺陷。
    //
    // 实测教训：0095_RT 因为配准符号写反，图案区偏了 48px，
    // 把完好的环报成"断口 4 处、缺口 50.5°"，断点分 1100 直接判 NG。
    //
    // 【门槛为什么是 3.5 倍线宽，而不是原来的 2 倍】
    // 原来取 max(10, 2×线宽)。当时只看了"墨线 5~8px 宽"这一个量，
    // 觉得起伏超过线宽就不可信。但后来在全部 83 张 OK 上量了一遍：
    //
    //   起伏量 min=8  p25=17  中位=28  p75=36  max=48
    //   墨线宽 10~14
    //
    // 2×线宽 ≈ 20~28，正好落在这批数据的正中间，于是只有 42/83 张
    // 通过了对准检查 —— 另 41 张的图案区结论被清零。也就是说
    // "0 张检出缺墨"里有一半是"根本没测"，不是"测了没问题"。
    //
    // 起伏这么大不是逻辑错，是环本身在半径方向就有一段粗一段细
    // （_RT/_FT 两种相机、模糊程度不同），而"墨线宽度"只是半高宽，
    // 量不出这个粗细变化。拿它当尺子必然卡在数据中间。
    //
    // 3.5 倍线宽 ≈ 35~50px，能覆盖实测最大 48px 且留一点余量。
    // 同时它仍能挡住真正的错位：上面那次 48px 的符号错误会落在这里，
    // 而挖洞测试（tools/panel_hole.cpp）在 0/2/4/6/8/12° 下
    // 起伏都不变，说明这个门槛不会把真断口吃掉。
    //
    // 【带型判定下这道检查不再清零结论】
    // bandConnected 模式下，"有没有墨"是在整条半径带里数的，不依赖
    // "最亮半径"这条曲线。起伏量大只说明环一段粗一段细，与"带里有没有墨"
    // 无关 —— 此时再拿起伏量把结论清零，就是拿一个已证明与结论无关的量
    // 去否决结论。所以这里只把 alignOk 置位，供显示用，不改断口与分数。
    // 老做法（bandConnected=false）仍然清零，保持原有语义。
    const double kMaxSwing = std::max(20.0, 3.5 * radiusLineWidth);
    if (out.radiusSwing > kMaxSwing) {
        if (!P.bandConnected) {
            out.alignOk = false;
            out.breaks = 0;
            out.breakAt.clear();
            out.score = 0.0;
            out.band = PatternBand::None;
            out.flagged = false;
            return out;
        }
        out.alignOk = true;
        out.swingOverLimit = true;
    } else {
        out.alignOk = true;
    }

    // ---- 9. 打分（与字符区的划痕分完全独立）
    //
    // 断点分 = 最长断口角度 × 100。
    // 直接用角度当分，不乘人为系数：客户看的是"缺了多大一段"，
    // 这个量本身就是可解释的。
    //
    // 和划痕分为什么不做加权合并：
    //   划痕分 = 深度(灰度) × 跨字数，单位是"灰度×字数"；
    //   断点分 = 缺口角度(度) × 100。
    // 两者量纲不同，合并后门槛就没有物理含义了，
    // 而且以后单独调一套会牵连另一套。所以各算各的、各用各的门槛。
    out.score = out.maxBreakAngle * 100.0;
    out.band = out.score >= P.cutCertain   ? PatternBand::Certain
               : out.score >= P.cutObvious ? PatternBand::Obvious
               : out.score >= P.cutSuspect ? PatternBand::Suspect
                                           : PatternBand::None;
    out.flagged = out.score >= P.minScore;
    return out;
}

// ================================================================
// 新流水线入口（见 pattern/ 目录）
// ================================================================
//
// 上面这个 analyzePatternBreak() 是**重构前的老实现，一字未改**。
// 它内部把 9 个阶段硬编码串在一起，换判据必须整段改写，
// 十几轮实验下来已经很难维护。
//
// 新代码在 pattern/ 子目录，拆成了
//   预处理（拉直+极化） → 剖面（逐角度墨量） → 检测器（判据）
// 三层接口，加算法只需实现一个 IDetector 并注册。
//
// 这里提供桥接：把新流水线的 Result 转回 PatternBreakResult，
// 让 GUI、batchcore、已有工具不用改一行就能用上新检测器。
//
// 转换规则：
//   · 几何字段（fit/radiusLineWidth 等）沿用老实现算出的值
//   · breaks/maxGapAngle/score 取自名为 "gap_run" 的检测器（老判据）
//     这样默认配置下行为与老实现一致
//   · 新增的检测器结论不进老结构体，需要的话直接用 pattern::analyze()
PatternBreakResult analyzePatternBreakV2(const cv::Mat& signal, const RegionInfo& region,
                                         const Config& cfg, const PatternBreakParams& P,
                                         pattern::Result* detail) {
    pattern::Result res = pattern::analyze(signal, region, cfg, P);
    PatternBreakResult out = analyzePatternBreak(signal, region, cfg, P);
    if (detail) *detail = res;
    if (!res.valid) return out;

    // 用 gap_run 的结论覆盖老判据分数，保证默认行为可复现
    if (const pattern::Finding* f = res.find("gap_run")) {
        if (!f->spansDeg.empty()) {
            const cv::Range r = f->spansDeg.front();
            out.maxGapAngle = std::abs(r.end - r.start);
        }
    }
    return out;
}

}  // namespace chardet
