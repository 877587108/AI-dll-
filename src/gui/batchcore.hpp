#pragma once

// 批量检测的单张核心（无 Qt 依赖的部分）
//
// 为什么单独抽出来：
//   1. 原来这段流程（笔画信号 → 整图配准 → 逐字符对齐 → 三条判据 → 划痕）
//      只写在 runBatch() 的循环体里，既没法单独跑，也没法验证。
//      抽出来之后可以用 --batch 命令行模式跑同一份代码，拿目录真值当基准。
//   2. 判定依据只有划痕一条（客户确认的唯一缺陷类型），
//      缺字/缺墨/多墨退成旁证写进结果，不再参与判定。
//
// 这里只做"一张图 → 一行结果"，不碰界面、不写文件。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "chardet.hpp"
#include "core/char_alignment.hpp"
#include "gui/scratch.hpp"

namespace markloc {

/// 逐字符对齐位移超过这个值(px)且同时出现旁证墨量差异 → 归"对齐异常"
///
/// 保留原来的大位移提示门槛；当前定位搜索半径为 ±24px。
/// 它结合墨量旁证作保守提示，不等同于配准失败，不参与 NG 判定。
inline constexpr int kAlignSuspiciousPx = 10;

/// 一张图的检测结果
struct BatchRow {
    std::string name;   // 文件名（UTF-8）
    std::string path;   // 绝对路径（UTF-8）
    std::string cls;    // 分类：确定 / 明显 / 可疑 / 旁证 / OK
    bool ng = false;

    // 划痕：判定依据
    int scratchCount = 0;
    int scratchBand = -1;   // -1 = 无
    double scratchS = 0.0;  // 最高严重度
    int scratchChars = 0;
    double scratchDepth = 0.0;
    double scratchLen = 0.0;

    // 逐字符对齐实际改了哪个字、改了多少。
    // 存在的意义：对齐失败时整条结果都不可信（框没对准字形，任何墨量判据都是错的），
    // 这类结果必须和"真的没有划痕"分开呈现，不能混进 OK。
    int alignMoved = 0;     // 找到更好位置的字数
    int alignMaxShift = 0;  // 最大位移(px)
    int alignWarped = 0;    // 实际把像素搬过去、对齐到位的字数
    std::string alignEvidence;  // 位移最大的那个字：型号 + 方向

    // 没报划痕时，说清是哪道门槛拦的。为空 = 报出来了。
    int candTotal = 0;         // Hough 直线候选数
    double bestRejectedS = 0;  // 落选候选里最高的严重度
    std::string rejectReason;

    // 旁证：只记录，不参与判定
    int rowMissing = 0;
    int miss = 0;
    int extra = 0;

    // 图案区（椭圆 / 圆）断点检测。
    // 和字符区是两套判据，所以单独记，不混进 scratchS。
    int patternRegions = 0;      // 本次检测的图案区个数
    int patternBreaks = 0;       // 判出的断口总数
    int patternMissing = 0;      // 整圈无墨的图案区个数（不是断点）
    double patternMaxGap = 0;    // 周上最长缺口角度(度)
    // 环半径沿角度的起伏(px)。这个数要是超过墨线宽度，
    // 说明椭圆框和实际的环对不上，断点检测的结论不可信。
    double patternRadiusSwing = 0;
    // 图案区断点分（与划痕分独立，各自比各自的门槛）
    double patternScore = 0.0;
    int patternBand = 0;          // 0=无 1=可疑 2=明显 3=确定
    bool patternFlagged = false;  // 达到图案区的 minScore
    // 没对上的图案区个数。这些区域"没验过"，和"验过没问题"不是一回事，
    // 必须单独计数，不能混进 OK。
    int patternUnaligned = 0;
    std::vector<cv::Point2f> patternAt;  // 断口坐标

    double elapsedMs = 0.0;
};

/// 检测设置
struct DetectSettings {
    chardet::Config cfg;
    bool perCharAlign = true;

    // 将信号一次性映射到模型坐标；关闭时只移动框，供诊断对照。
    bool perCharWarp = true;
    bool checkRowCount = true;
    bool checkCharLevel = true;
    bool checkScratch = true;
    ScratchParams scratch;

    // 图案区（椭圆 / 圆）断点检测。走单独一套判据，见 analyzePatternBreak。
    bool checkPattern = true;
    chardet::PatternBreakParams pattern;

    // 上一次整图配准施加的平移量（像素）。
    //
    // 存在的理由：配准作用在 sig 上，字符框和图案区都还在原坐标里。
    // 字符由逐字符精对齐各自修正，图案区没有字可对，只能整体跟着挪，
    // 否则环的采样点会整体偏掉 5~9px —— 环只有 5~8px 宽，偏这么多就采到背景上，
    // 整圈都会被判成"无墨"。
    mutable int lastAlignDx = 0;
    mutable int lastAlignDy = 0;
};

/// 单张检测。gray 必须与 model.freq 同尺寸，否则返回 valid=false。
///
/// 传入 baseUnits（模型里的字符框）而不是从 model 里现算，
/// 是因为调用方已经算好一次，循环里重复算没有意义。
inline BatchRow detectOne(const cv::Mat& gray, const chardet::Model& model,
                          const std::vector<chardet::CharUnit>& baseUnits,
                          // 用非 const 引用：要把这次的配准位移写回去，给图案区用。
    DetectSettings& st, bool* valid = nullptr) {
    BatchRow row;
    if (valid) *valid = false;
    if (gray.empty() || gray.size() != model.freq.size()) return row;
    if (valid) *valid = true;

    const chardet::Config& cfg = st.cfg;

    // 每张图重新算配准位移，必须先清零。
    //
    // 不清零的话，本张若没触发相位相关（位移 < 0.01，或 > 60 被拒），
    // lastAlignDx/Dy 会留着上一张图的值，图案区就被挪到了别处 ——
    // 实测 0095_RT 因此把完好的环报成"断口 2 处、8°、800 分"。
    st.lastAlignDx = 0;
    st.lastAlignDy = 0;

    // ---- 笔画信号
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // ---- 整图配准：只用字符区域做相位相关
    //
    // 整幅图里引脚、封装边缘的高对比结构远强于刻印字符，相位相关会被它们主导，
    // 给出错误的平移量。实测三张误报件上两种方式的差异达 8~16px，
    // 足以让所有字符框整体错位、全部误报缺墨。
    if (!model.ref.empty()) {
        cv::Mat r32, s32;
        model.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : baseUnits) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
        }
        if (inkArea.area() > 0) {
            inkArea.x = std::max(0, inkArea.x - 20);
            inkArea.y = std::max(0, inkArea.y - 20);
            inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
            inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            s32(inkArea).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            // 60px 是实测的失效边界，与 DefectViewWindow 保持一致：
            // 位移 <= 56px 零误报，60px 起相位相关放弃修正，立即崩成
            // 几十条误报；放宽门限也只多撑到 68px。投影对齐经实测更差
            // （84 张上相位胜 81、投影胜 0），不要替换成投影法。
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                // 源与目标不能是同一个 Mat（未定义行为），先算到临时块再换手
                cv::Mat aligned;
                cv::warpAffine(sig, aligned, M, sig.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                sig = aligned;
                // 记下这次配准实际挪了多少。
                //
                // 【符号别弄反】warpAffine 用的矩阵是 M = [1,0,-sh.x; 0,1,-sh.y]，
                // 即把图像内容平移了 **-sh**。所以模型坐标里的点要加 -sh
                // 才落在图像的同一处。第一版写成 +sh，0095_RT 上图案区偏了 48px，
                // 把完好的环报成"断口 4 处、缺口 50.5°"，是一次硬误报。
                st.lastAlignDx = -static_cast<int>(std::lround(sh.x));
                st.lastAlignDy = -static_cast<int>(std::lround(sh.y));
            }
        }
    }

    // 一次定位，以模型框为目标。信号重采样后不再移动 bbox / extend。
    std::vector<chardet::CharUnit> units = baseUnits;
    if (st.perCharAlign) {
        const auto locations = locateCharacters(sig, model.freq, baseUnits);
        for (size_t i = 0; i < locations.size(); ++i) {
            const auto& a = locations[i];
            if (!a.accepted) continue;
            const int dx = static_cast<int>(std::lround(a.sourceOffset.x));
            const int dy = static_cast<int>(std::lround(a.sourceOffset.y));
            ++row.alignMoved;
            const int shift = std::max(std::abs(dx), std::abs(dy));
            if (shift > row.alignMaxShift) {
                row.alignMaxShift = shift;
                char buf[96];
                std::snprintf(buf, sizeof(buf), "#%d dx%+d dy%+d", units[i].index, dx, dy);
                row.alignEvidence = buf;
            }
            if (!st.perCharWarp) {
                units[i].bbox += cv::Point(dx, dy);
                units[i].extend += cv::Point(dx, dy);
            } else if (shift > 0) {
                ++row.alignWarped;
            }
        }
        if (st.perCharWarp) sig = alignCharacterSignal(sig, locations);
    }

    // ---- 旁证：整字数量 / 字符级缺墨多墨
    if (st.checkRowCount) {
        const auto bd = chardet::diffBlocks(sig, model.freq, units, cfg, 8);
        if (bd.valid) {
            for (const auto& rc : bd.rows) row.rowMissing += rc.missingChars;
        }
    }
    if (st.checkCharLevel) {
        for (const auto& u : units) {
            const auto nc = chardet::diffNcc(sig, model.freq, u, cfg);
            if (!nc.valid) continue;
            if (nc.missScore > 0.0) ++row.miss;
            if (nc.extraScore > 0.0) ++row.extra;
        }
    }

    // ---- 图案区断点检测（椭圆 / 圆）
    //
    // 与字符区是两套互不相干的判据：
    //
    //   字符区  问"墨量对不对" —— 缺墨、多墨、划痕切断了字
    //   图案区  问"这条闭合曲线有没有被切断"
    //
    // 为什么不把椭圆塞进划痕检测：椭圆环在一个 322x225 的框里只占
    // 一条 5~8px 宽的线，框内九成是背景。划痕判据按"线扫过多少个字符框"
    // 计分，而椭圆被当成了一个巨型字符框 —— 那条 279x177 的假框就是这么来的，
    // 它既不是一个字，也量不出环的断口。
    //
    // 依赖：图案区必须已在 model.regions 里标成 Ellipse/Circle。
    if (st.checkPattern && !model.regions.empty()) {
        for (const auto& rg : model.regions) {
            if (!rg.isPattern()) continue;
            // 【椭圆不需要跟着平移 —— 这一步曾经写错，造成 0095_RT 硬误报】
            //
            // 配准的矩阵是 M = [1,0,-sh.x; 0,1,-sh.y] 配 WARP_INVERSE_MAP，
            // 语义是 dst(x,y) = src(x-sh.x, y-sh.y)，即图像内容移动了 +sh，
            // 目的正是把这张图挪回模型的基准位置。
            //
            // 挪完之后模型坐标就重新对上了，椭圆保持原位即可。
            // 之前在椭圆上加 -sh，等于把已校正的偏移反着又加一遍：
            // 实测 0095_RT 覆盖率 0.9944 -> 0.9444，完好的环被报成
            // "断口 4 处、缺口 8.5°、断点分 850"，是一次硬误报。
            //
            // 三个位置实测（同一张图、同一套判据）：
            //   原位 0.9944  |  +sh 0.9264  |  -sh 0.9444
            const chardet::RegionInfo& moved = rg;
            const auto pb = chardet::analyzePatternBreak(sig, moved, cfg, st.pattern);
            if (!pb.valid) continue;
            row.patternRegions += 1;
            row.patternBreaks += pb.breaks;
            row.patternMaxGap = std::max(row.patternMaxGap, pb.maxGapAngle);
            row.patternRadiusSwing = std::max(row.patternRadiusSwing, pb.radiusSwing);
            if (pb.allMissing) ++row.patternMissing;
            for (const auto& pt : pb.breakAt) row.patternAt.push_back(pt);
            // 图案区的分数单独记，不并进划痕分。
            // 两个量纲不同的东西合在一起，门槛就失去物理含义了。
            if (pb.score > row.patternScore) {
                row.patternScore = pb.score;
                row.patternBand = static_cast<int>(pb.band);
            }
            if (pb.flagged) row.patternFlagged = true;
            if (!pb.alignOk) ++row.patternUnaligned;
        }
    }

    // ---- 划痕：唯一的判定依据
    if (st.checkScratch) {
        ScratchStats ss;
        const auto scratches = findScratches(sig, model.freq, units, cfg, st.scratch, &ss);
        row.candTotal = ss.candidates;
        row.bestRejectedS = ss.bestComposite;
        // 一句话说清"为什么没报"：哪道门槛拦的、最大的严重度是多少。
        // 没有这句，"OK"和"卡在门槛上"看起来一模一样。
        if (scratches.empty()) {
            char buf[96];
            if (ss.candidates == 0) {
                std::snprintf(buf, sizeof(buf), "无直线候选");
            } else if (ss.bestComposite <= 0.0) {
                // 有直线、也过了长度/覆盖/能量，但严重度算出来是 0：
                // 说明这条线扫过的字符框是 0 个，即痕迹整条落在字外。
                // 客户的标准里字外的不影响识别，这种"未报"是对的，要能和
                // "有痕迹但不够重"区分开。
                std::snprintf(buf, sizeof(buf), "候选%d条均落在字外", ss.passed + ss.belowComposite);
            } else if (ss.belowComposite > 0) {
                std::snprintf(buf, sizeof(buf), "严重度 %d < 门槛 %d",
                              static_cast<int>(ss.bestComposite),
                              static_cast<int>(st.scratch.minComposite));
            } else {
                std::snprintf(buf, sizeof(buf), "长度%d 覆盖%d 能量%d",
                              ss.belowLen, ss.belowCov, ss.belowEnergy);
            }
            row.rejectReason = buf;
        }
        row.scratchCount = static_cast<int>(scratches.size());
        for (const auto& sh : scratches) {
            if (sh.composite <= row.scratchS) continue;
            row.scratchS = sh.composite;
            row.scratchBand = sh.band;
            row.scratchChars = sh.charsHit;
            row.scratchDepth = sh.depth;
            row.scratchLen = sh.len;
        }
    }
    // NG = 任一判据达到自己的门槛。
    //
    // 两套判据各有各的量纲和门槛，这里只做"或"，
    // 绝不做加权求和 —— 划痕分是"深度×跨字数"，断点分是"缺口角度×100"，
    // 加在一起会得到一个新单位不明的数，门槛就没法解释了。
    row.ng = (row.scratchCount > 0) || row.patternFlagged;

    // 分类：划痕四档优先，其次图案区断口，没报的但有旁证的单独归"旁证"。
    //
    // "旁证"里其实混着两种完全不同的东西，必须拆开看：
    //   · 真有别的毛病（缺字/缺墨多墨是实症）
    //   · 对齐失败导致的虚假墨量差异 —— 框没对准字形，框里装的是
    //     "半个字形 + 一块空白"，任何墨量判据都会给出假的缺墨。
    // 第二种是"这一条没被证明"，不是"查过了没问题"。混进 OK 就是在
    // 拿一个没跑通的检测当合格判据，所以单独标成"对齐异常"。
    const bool evidence = (row.rowMissing + row.miss + row.extra) > 0;
    const bool alignBroken = evidence && row.alignMaxShift >= kAlignSuspiciousPx;
    row.cls = (row.scratchBand >= 1) ? std::string(bandName(row.scratchBand))
              : row.patternFlagged  ? std::string("图案断")   // 环被切断，与划痕并列
              : alignBroken         ? std::string("对齐异常")
              : evidence            ? std::string("旁证")
                                    : std::string("OK");
    return row;
}

}  // namespace markloc
