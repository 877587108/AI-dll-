#pragma once

// 划痕判定（单张检测与批量检测共用同一份实现）
//
// 为什么单独抽出来：
//   runDetect() 里原有一整套划痕判据（Hough 直线 + 覆盖率 + 深度 × 跨字数），
//   但 runBatch() 完全没有调它 —— 批量只跑了「整字数量 / 缺墨 / 多墨」三条，
//   而客户确认的唯一缺陷类型就是划痕。于是批量页把真 NG 全判成 OK，
//   与单张页自相矛盾。抽成独立函数后两条路径共用同一份代码、同一组参数，
//   不可能再出现"单张报、批量不报"。
//
// 判据链与原实现逐行一致，只做搬运，不改任何数值：
//   笔画信号 → 字符并集外扩 60px ∩ 塑封体 → 弱阈值二值化
//   → 扣除膨胀后的模型笔画 → 闭运算重连断口 → 中值去噪
//   → HoughLinesP 取直线 → 长度/覆盖率/能量门槛
//   → 垂直剖面求深度、扫字符框求跨字数
//   → 严重度 = 深度 × 跨字数 → 四档切点

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/annotate.hpp"

namespace markloc {

/// 划痕参数
///
/// 默认值与 DefectViewWindow::Sensitivity 中的同名字段保持一致；
/// GUI 每次调用前按当前界面设置填一遍，不要在这里偷偷改数值。
struct ScratchParams {
    bool   enabled      = true;
    double minLen       = 60.0;    // 最短长度(px)
    double minCov       = 0.25;    // 最低覆盖率
    int    bridge       = 25;      // 跨字符重连核(px)，0 = 不重连
    double minEnergy    = 0.0;     // 能量门槛（只作去噪）
    double minComposite = 1000.0;  // 严重度报告门槛
    double cutCertain   = 1900.0;  // 四档切点
    double cutObvious   = 1300.0;
    double cutSuspect   = 1000.0;
    int    depthReach   = 12;      // 深度剖面半长(px)
    int    growPx       = 60;      // 字符并集外扩(px)

    // ---- 灰度阈值
    //
    // 已经算出过这个量，但一直没接进判据。这里把它接上：
    // 划痕的平均灰度低于 minGrayOfSignal 的一律不收，记进
    // stats.belowGray，界面上按"灰度"档显示落选。
    // 关闭时置 0（0 = 不筛）。
    bool   useGly     = false;
    double minGrayOfSignal = 0.0;  // 0~1，相对 strokeHi 归一

    // ---- 差异阈值：深度门槛
    //
    // 深度 = 垂直剖面对比度，单位与「严重度」里的深度一致。
    // 这是"比旁边亮多少"的直接门槛，比严重度更贴近人眼判深浅。
    // 关闭时置 0（0 = 不筛）。
    bool   useDiff    = false;
    double minDiff    = 0.0;

    // ---- 离字形距离门槛
    //
    // 用来剔除"字形漏出被连成假线"。
    //
    // 依据（全是实测，见 docs/0109rt-root-cause.md）：
    //   0109_RT 那条 S=1938 的线在原图上根本不存在，放大 7 倍看到的是
    //   字形缝里的空白；它的原料是字母 N 的对角笔画，那段亮像素的 signal
    //   高达 1.5，但模型 freq 只有 0~3——真实字形，模型没学到。
    //
    // 但这条门槛单独不够：0109_RT 有两条落在 6.7，贴着 6.5 的线过去。
    // 现在由下面的笔画方向判据接管，这里默认关闭，保留能力供对照。
    bool   useInkDist      = false;
    double minInkDist      = 6.5;   // px

    // 跨越字数的最低重叠比例：线段在某字符框内实际走过的长度
    // 不足该框宽的这个比例时，不算跨越该字。0 = 退回"有交就算"的旧行为。
    //
    // 实测结论：这个维度不能动。minCharCover 从 0 抬到 0.10，
    // 0026_RT 的严重度就从 2287 掉到 0、0112 两张也从 1288/1160 掉到 0。
    // "跨越字数收紧"会先把真缺陷打死，因为真划痕本来就靠一条线跨多个字。
    // 保留参数供对照，默认关闭。
    double minCharCover    = 0.0;

    // ---- 笔画方向一致度门槛（核心判据）
    //
    // 问法变了：不问"这条线像不像划痕"，问"它是不是顺着字形笔画走"。
    //
    // 依据：真划痕是外来物，走向与字形无关，必然横穿笔画；
    //       字形漏出是字形的一部分，只会沿着笔画走。
    //
    // 量法：沿线取亮像素，取模型墨迹的局部梯度（= 笔画法向），
    //       其垂直方向就是笔画切向；算线段方向与笔画的 |cos|。
    //       另统计"横穿笔画"（|cos| < 0.35）的采样点占比。
    //
    // 全库实测（各取最严重那条）：
    //   真缺陷 0026_FT 0.595/垂直0.08  0026_RT 0.601/0.08
    //          0083_FT 0.542/0.19  0083_RT 0.566/0.16
    //          0112_FT 0.673/0.37  0112_RT 0.648/0.37
    //   0109_RT（无划痕，用户确认） 0.905 / 垂直 0.00
    //   平行中位 0.90 已经把两类分开，垂直占比 0.00 vs 0.08 更是没有重叠。
    //
    // 门槛取 0.78：比真缺陷最高的 0.673 高 0.107，比 0109_RT 的 0.905 低 0.125，
    // 落在两者中间。同时要求垂直占比 >= 0.03：一条完全顺着笔画走的线，
    // 一次都不横穿，那就不是划痕。
    //
    // 关闭时置 useDir = false。
    bool   useDir          = true;
    double maxParCos       = 0.78;  // 平行中位高于此值 -> 判为字形漏出
    double minVertFrac     = 0.03;  // 横穿占比低于此值 -> 判为字形漏出
};

/// 落选统计：每条候选被哪一道门槛拦下的计数
///
/// 只统计"有痕迹但报不出来"的原因，用来区分三种情况：
///   · 根本没有候选（Hough 没找到直线）—— 痕迹不够长/不够直
///   · 有候选但长度、覆盖率、能量、严重度某一项不过
/// 阈值该不该动是客户的决定，但"卡在哪一道"必须看得见。
struct ScratchStats {
    int candidates = 0;        // HoughLinesP 给出的直线数
    int belowLen = 0;          // 长度 < minLen
    int belowCov = 0;          // 覆盖率 < minCov
    int belowEnergy = 0;       // 能量 < minEnergy
    int belowComposite = 0;    // 严重度 < minComposite
    int belowGray = 0;         // 平均灰度 < minGrayOfSignal
    int belowDiff = 0;         // 深度(差异) < minDiff
    int belowInkDist = 0;      // 到字形墨迹的距离 < minInkDist（字形漏出，不是划痕）
    int belowDir = 0;          // 顺着笔画走（方向一致），判为字形漏出
    int passed = 0;            // 过全部门槛
    double bestComposite = 0;  // 落选候选里最大的严重度（过长度门槛的才计入）
    // 落选候选里最高分那条的线段（全图坐标）。
    // 只记分数不够：0101 的 796 分落在一根竖线上，而真划痕那条斜线只有 609，
    // 光看"最高分"会把排查引到错误的位置。
    //
    // 【0101 的排查结论，别重复走】
    //   796 那条竖线 (667,936)-(671,815) 贴着数字 4 的右竖笔。
    //   沿法向取剖面，一侧均值 0.91（笔画）、另一侧 0.17（背景），
    //   两侧不对称度 0.81 —— 是台阶，不是山峰，即字形边缘，不是划痕。
    //   真划痕那条斜线 (587,917)-(670,834) 两侧均值 0.50 / 0.47，
    //   不对称度 0.06 —— 是山峰，独立痕迹。
    //   所以"台阶/山峰"能分开这两类（实测 0.15~0.60 vs 0.81）。
    //   但去掉假线之后 0101 最高只剩 609，仍在门槛 1000 以下，
    //   0101 依然是漏检。要不要降门槛是待定事项，不要擅自改。
    double bestRejectedS = 0;
    std::array<cv::Point, 2> bestRejectedLine{cv::Point(0, 0), cv::Point(0, 0)};
    // 各门槛的实测值分布，供「阈值边缘」图用：
    // 每过一道前面门槛就把当时的实测值记下来，界面上按升序画
    // 「实测值 vs 门槛线」，越过线的那批就是报出来的。
    std::vector<double> lenVals, covVals, energyVals, grayVals, diffVals, compVals;
    const char* blockedBy = "";
};

/// 一条已通过全部门槛的划痕
struct ScratchHit {
    cv::Point a1, a2;         // 全图坐标，线段端点
    cv::Rect  box;            // 标注框（已外扩）
    double len       = 0.0;
    double depth     = 0.0;   // 垂直剖面对比度
    double widthPx   = 0.0;   // 半高宽
    double energy    = 0.0;
    double cov       = 0.0;   // 覆盖率
    double gray      = 0.0;   // 平均灰度（在线段 ±1px 内，按 strokeHi 归一）
    double composite = 0.0;   // 严重度 = 深度 × 跨越字数
    int    charsHit  = 0;
    int    band      = 0;     // 0 轻微 1 可疑 2 明显 3 确定
};

inline const char* bandTag(int b) {
    static const char* t[4] = {"FAINT", "SUSPECT", "OBVIOUS", "CERTAIN"};
    return t[std::clamp(b, 0, 3)];
}

inline const char* bandName(int b) {
    static const char* t[4] = {"轻微", "可疑", "明显", "确定"};
    return t[std::clamp(b, 0, 3)];
}

inline cv::Scalar bandColor(int b) {
    static const cv::Scalar c[4] = {
        cv::Scalar(170, 170, 170),   // 轻微
        cv::Scalar(0, 215, 255),     // 可疑
        cv::Scalar(0, 140, 255),     // 明显
        cv::Scalar(0, 0, 255)};      // 确定
    return c[std::clamp(b, 0, 3)];
}

inline int bandOf(const ScratchParams& P, double composite) {
    return (composite >= P.cutCertain)   ? 3
           : (composite >= P.cutObvious) ? 2
           : (composite >= P.cutSuspect) ? 1
                                         : 0;
}

/// 塑封体（黑色矩形本体）的近似范围
///
/// 用途：限定划痕这类"框外检测"的边界——塑封体上的痕迹才与刻印质量相关，
/// 引脚、基板上的高亮属于另一回事，纳入会带来大量误报。
/// 数值按 1688×1460 的成像标定，其他分辨率按比例缩放。
inline cv::Rect moldingArea(const cv::Size& sz) {
    const double sx = static_cast<double>(sz.width) / 1688.0;
    const double sy = static_cast<double>(sz.height) / 1460.0;
    const int x0 = static_cast<int>(300 * sx);
    const int y0 = static_cast<int>(225 * sy);
    const int x1 = static_cast<int>(1365 * sx);
    const int y1 = static_cast<int>(1140 * sy);
    return cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
}

/// 划痕检测范围：字符框并集 → 外扩 growPx → 与塑封体取交集
inline cv::Rect scratchBand(const cv::Size& sz, const std::vector<chardet::CharUnit>& units,
                            int growPx) {
    cv::Rect band;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sz.width, sz.height);
        if (b.width <= 0 || b.height <= 0) continue;
        band = band.area() > 0 ? (band | b) : b;
    }
    if (band.area() <= 0) return {};
    const cv::Rect grown(band.x - growPx, band.y - growPx, band.width + 2 * growPx,
                         band.height + 2 * growPx);
    return grown & moldingArea(sz) & cv::Rect(0, 0, sz.width, sz.height);
}

/// 跑一遍划痕检测，返回所有过门槛的划痕（未排序）
///
/// 末尾带一个可选的落选统计：每条候选究竟被哪一道门槛拦下的。
/// 存在的意义是让"这一张为什么没报"变成可以查的事实，
/// 而不是对着结果猜是参数太紧还是根本没有痕迹。outStats 为空时不计。
inline std::vector<ScratchHit> findScratches(const cv::Mat& signal, const cv::Mat& freq,
                                             const std::vector<chardet::CharUnit>& units,
                                             const chardet::Config& cfg, const ScratchParams& P,
                                             ScratchStats* outStats = nullptr) {
    std::vector<ScratchHit> out;
    ScratchStats stats;
    const auto never = [&](const char* why) {
        if (outStats) {
            stats.blockedBy = why;
            // 提前返回时也要给出确定的初值。之前 bestComposite 未赋值，
            // 导致"为什么没报"那句提示里印出 -152872458 这种溢出垃圾。
            stats.bestComposite = 0.0;
            *outStats = stats;
        }
        return out;
    };

    if (!P.enabled) return never("");
    if (signal.empty() || freq.empty() || freq.size() != signal.size()) return never("");

    const cv::Rect band = scratchBand(signal.size(), units, P.growPx);
    if (band.area() <= 0) return never("");

    const cv::Mat sBand = signal(band);

    // 模型墨迹：freq 高过空白判定线的地方算"这里本来就有笔画"。
    // 后面三处都要用（做掩膜、算距离图、数真实跨越字数），先算一次。
    cv::Mat modelInk;
    cv::compare(freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);

    // 到"字形墨迹"的距离图（在 band 坐标系下，与线段坐标同系）
    //
    // 划痕是外来物，会离开字形；字形漏出只沿字形边缘走。
    // 实测两类在这上面不重叠，详见 ScratchParams::minInkDist 的注释。
    cv::Mat inkDist;
    if (P.useInkDist) {
        cv::Mat notInk;
        cv::bitwise_not(modelInk, notInk);
        cv::distanceTransform(notInk, inkDist, cv::DIST_L2, 3);
    }

    // 模型墨迹的梯度场（band 坐标系，与线段坐标同系）：
    // 梯度方向是笔画的法向，它的垂直方向就是笔画走向。
    // 5×5 高斯模糊一下再求导，避免 freq 的量化台阶把方向弄脏。
    cv::Mat gxModel, gyModel;
    if (P.useDir) {
        cv::Mat inkF;
        modelInk.convertTo(inkF, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(inkF, inkF, cv::Size(5, 5), 0);
        cv::Sobel(inkF, gxModel, CV_32F, 1, 0, 3);
        cv::Sobel(inkF, gyModel, CV_32F, 0, 1, 3);
    }

    // 弱阈值二值化：只留"像墨"的像素，再扣掉模型里的笔画
    cv::Mat weak;
    cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    weak.convertTo(weak, CV_8U, 255.0);

    {
        const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
        cv::Mat modelDil, notModel;
        cv::dilate(modelInk, modelDil, kd);
        cv::bitwise_not(modelDil, notModel);
        cv::bitwise_and(weak, notModel, weak);
    }

    // 跨字符重连：减去模型笔画会把一条划痕剁成几截，压在字上的部分全被删掉，
    // 碎片各自都过不了长度门槛，整条划痕就消失了。闭运算沿划痕方向接回断口，
    // 核要大于笔画宽度才能跨过一根竖笔。
    if (P.bridge > 0) {
        const int bk = std::max(3, P.bridge | 1);
        const cv::Mat kb = cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk});
        cv::morphologyEx(weak, weak, cv::MORPH_CLOSE, kb);
    }
    // 只去孤立单像素噪点，不用 3×3 开运算削细划痕本身
    cv::medianBlur(weak, weak, 3);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(weak, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);

    const double grayScale = 255.0 / std::max(1e-6, cfg.strokeHi);
    const int reach = std::max(6, P.depthReach);

    for (const auto& ln : lines) {
        const cv::Point p1(ln[0], ln[1]);
        const cv::Point p2(ln[2], ln[3]);
        ++stats.candidates;
        const double len = cv::norm(p2 - p1);
        if (len < P.minLen) {
            ++stats.belowLen;
            continue;
        }

        const int steps = std::max(2, static_cast<int>(len));
        double energy = 0.0;
        int covered = 0;
        for (int t = 0; t <= steps; ++t) {
            const double rr = static_cast<double>(t) / steps;
            const int x = static_cast<int>(p1.x + (p2.x - p1.x) * rr);
            const int y = static_cast<int>(p1.y + (p2.y - p1.y) * rr);
            if (y < 0 || y >= weak.rows || x < 0 || x >= weak.cols) continue;
            bool hit = false;
            for (int dy = -4; dy <= 4; ++dy) {
                const int yy = std::clamp(y + dy, 0, weak.rows - 1);
                if (weak.at<uchar>(yy, x)) {
                    energy += std::max(static_cast<double>(sBand.at<float>(yy, x)) - cfg.missStrokeLo,
                                       0.0);
                    if (std::abs(dy) <= 1) hit = true;
                }
            }
            if (hit) ++covered;
        }
        const double cov = static_cast<double>(covered) / (steps + 1);
        if (cov < P.minCov) {
            ++stats.belowCov;
            continue;
        }

        // ---- 离字形距离：剔除"字形漏出连成的假线"
        //
        // 只统计亮像素的距离中位数，不用整条线的平均值：
        // 划痕穿过字形那几段距离必然很小，平均值会被拉低；
        // 中位数反映的是"这条线整体上离字形有多远"。
        if (P.useInkDist) {
            std::vector<float> dsv;
            dsv.reserve(steps + 1);
            for (int t = 0; t <= steps; ++t) {
                const double rr = static_cast<double>(t) / steps;
                const int x = static_cast<int>(p1.x + (p2.x - p1.x) * rr);
                const int y = static_cast<int>(p1.y + (p2.y - p1.y) * rr);
                if (y < 0 || y >= weak.rows || x < 0 || x >= weak.cols) continue;
                if (weak.at<uchar>(y, x)) dsv.push_back(inkDist.at<float>(y, x));
            }
            if (!dsv.empty()) {
                std::nth_element(dsv.begin(), dsv.begin() + dsv.size() / 2, dsv.end());
                if (dsv[dsv.size() / 2] < P.minInkDist) {
                    ++stats.belowInkDist;
                    continue;
                }
            }
        }

        const double score = energy * len;
        if (score < P.minEnergy) {
            ++stats.belowEnergy;
            continue;
        }

        // ---- 笔画方向一致度：这条线是不是顺着字形笔画走的
        //
        // 真划痕是外来物，走向与字形无关，必然横穿笔画；
        // 字形漏出是字形的一部分，只会沿笔画走。
        // 实测两类在"平行中位"和"横穿占比"上都不重叠，详见 ScratchParams::useDir。
        if (P.useDir && !gxModel.empty() && !gyModel.empty()) {
            const double dl = std::max(1e-6, len);
            const double ux0 = (p2.x - p1.x) / dl, uy0 = (p2.y - p1.y) / dl;
            std::vector<double> cosv;
            int vert = 0;
            // 每 2px 取一个点，每个点取 ±2px 垂直邻域并统计整块局部窗口，
            // 而不是只取轴线那一列。短线（64~173px）按原来的单点采样
            // 只能拿到三五个有效样本，判定等于没做；扩到窗口后样本量x10以上。
            const int nD = std::max(2, static_cast<int>(len / 2.0));
            for (int t = 0; t <= nD; ++t) {
                const double rr = static_cast<double>(t) / nD;
                const int bx = static_cast<int>(std::lround(p1.x + (p2.x - p1.x) * rr));
                const int by = static_cast<int>(std::lround(p1.y + (p2.y - p1.y) * rr));
                for (int k = -2; k <= 2; ++k) {
                    const int x = static_cast<int>(std::lround(bx - uy0 * k));
                    const int y = static_cast<int>(std::lround(by + ux0 * k));
                    if (x < 0 || y < 0 || x >= gxModel.cols - 1 || y >= gxModel.rows - 1) continue;
                    const double gx = gxModel.at<float>(y, x), gy = gyModel.at<float>(y, x);
                    const double gm = std::hypot(gx, gy);
                    if (gm < 0.02) continue;  // 这里没有笔画，不计入
                    const double tx = -gy / gm, ty = gx / gm;  // 笔画切向
                    const double c = std::abs(ux0 * tx + uy0 * ty);
                    cosv.push_back(c);
                    if (c < 0.35) ++vert;
                }
            }
            // 样本门槛按"至少要有一半的采样点能取到笔画信息"来定，
            // 取不到就说明这条线压根不在字形附近，那反而不像字形漏出，放行。
            if (cosv.size() >= 12) {
                std::nth_element(cosv.begin(), cosv.begin() + cosv.size() / 2, cosv.end());
                const double parP50 = cosv[cosv.size() / 2];
                const double vertFrac = static_cast<double>(vert) / cosv.size();
                if (parP50 > P.maxParCos && vertFrac < P.minVertFrac) {
                    ++stats.belowDir;
                    continue;
                }
            }
        }

        // ---- 深度：垂直剖面的对比度
        //
        // 不用绝对亮度（曝光一变就变），取垂直剖面中间最大值减去两端背景中位数，
        // 得到"比旁边亮多少"，再数半高宽。这两样才是人眼判深浅时用的。
        const double lenN = std::max(1e-6, len);
        const double ux = (p2.x - p1.x) / lenN, uy = (p2.y - p1.y) / lenN;
        const double nx = -uy, ny = ux;
        double sumDepth = 0.0, sumWidth = 0.0;
        int nSamp = 0;
        const int nStep = std::max(4, static_cast<int>(len / 6.0));
        for (int t = 1; t < nStep; ++t) {
            const double f = static_cast<double>(t) / nStep;
            const double qx = p1.x + (p2.x - p1.x) * f;
            const double qy = p1.y + (p2.y - p1.y) * f;
            std::vector<double> prof(static_cast<size_t>(2 * reach + 1), 0.0);
            for (int k = -reach; k <= reach; ++k) {
                const int xi = static_cast<int>(std::lround(qx + nx * k));
                const int yi = static_cast<int>(std::lround(qy + ny * k));
                if (xi < 0 || yi < 0 || xi >= sBand.cols || yi >= sBand.rows) continue;
                prof[static_cast<size_t>(k + reach)] = sBand.at<float>(yi, xi) * grayScale;
            }
            std::vector<double> ends;
            for (int k = 0; k < reach - 4; ++k) {
                ends.push_back(prof[static_cast<size_t>(k)]);
                ends.push_back(prof[prof.size() - 1 - static_cast<size_t>(k)]);
            }
            std::sort(ends.begin(), ends.end());
            const double bg = ends.empty() ? 0.0 : ends[ends.size() / 2];
            double peak = 0.0;
            for (int k = reach - 5; k <= reach + 5; ++k)
                peak = std::max(peak, prof[static_cast<size_t>(k)]);
            const double contrast = peak - bg;
            const double half = bg + contrast * 0.5;
            int wide = 0;
            for (int k = reach - 6; k <= reach + 6; ++k)
                if (prof[static_cast<size_t>(k)] >= half) ++wide;
            sumDepth += contrast;
            sumWidth += wide;
            ++nSamp;
        }
        const double depth = nSamp > 0 ? sumDepth / nSamp : 0.0;
        const double widthPx = nSamp > 0 ? sumWidth / nSamp : 0.0;

        // ---- 跨越字数
        //
        // 试过的两种算法，结论写在这里，不要再走回头路：
        //
        // A. 轴对齐 bbox 与字符框求交（原实现）
        //    错在轴对齐：斜线的外接矩形面积是线本身的几十倍，会把不沾边的框
        //    也框进来。但它在当前数据上是**能工作的**，因为深度项做了主要区分。
        //
        // B. 沿线段数"连续压墨区段"（本轮试过，已废弃）
        //    想法是绕开框直接数真实穿过的字形（本模型 22 个框宽度
        //    16 21 31 … 279，i/l 被切成窄条、区域3 是个 logo 环，框确实不可靠）。
        //    实测：ok 误报 4→4 不变，ng 漏检 1→4（0026_FT/0026_RT/0101_FT 全被击穿）。
        //    原因：压低跨越字数对两类样本是同向的，它不是区分维度。
        //    所以退回 A，但把"有没有交"换成"交得够不够深"：
        //    线段压过某个字符框的长度不足框宽的 minCharCover 时，不算跨越该字。
        //    理由是线段的外接矩形远大于线段本身，一个框和矩形相交、
        //    但和真正的线段几乎不沾，这种"擦边"不该计入跨越字数。
        //    这一步只修正轴对齐带来的虚增，不改变"数框"这个前提。
        const cv::Point a1 = p1 + cv::Point(band.x, band.y);
        const cv::Point a2 = p2 + cv::Point(band.x, band.y);
        const cv::Rect seg(std::min(a1.x, a2.x), std::min(a1.y, a2.y), std::abs(a2.x - a1.x) + 1,
                           std::abs(a2.y - a1.y) + 1);
        int charsHit = 0;
        if (P.minCharCover <= 0.0) {
            for (const auto& u : units) {
                if ((u.bbox & seg).area() > 0) ++charsHit;
            }
        } else {            // 沿线段采样，看每个字符框里落了多少个采样点
            const int nS = std::max(2, static_cast<int>(len));
            for (const auto& u : units) {
                const cv::Rect ub = u.bbox;
                if (ub.width <= 0 || ub.height <= 0) continue;
                if ((ub & seg).area() <= 0) continue;
                int hits = 0;
                for (int t = 0; t <= nS; ++t) {
                    const double rr = static_cast<double>(t) / nS;
                    const int x = static_cast<int>(std::lround(a1.x + (a2.x - a1.x) * rr));
                    const int y = static_cast<int>(std::lround(a1.y + (a2.y - a1.y) * rr));
                    if (ub.contains(cv::Point(x, y))) ++hits;
                }
                // 换算成"线段在该框横向跨度内占的比例"：框宽对应的采样点数
                const double spanInBox =
                    static_cast<double>(nS) * std::min<double>(1.0, (double)ub.width / std::max(1.0, (double)std::abs(a2.x - a1.x)));
                const double cover = spanInBox > 1.0 ? hits / spanInBox : (hits > 0 ? 1.0 : 0.0);
                if (cover >= P.minCharCover) ++charsHit;
            }
        }

        // ---- 严重度 = 深度 × 跨越字数
        double composite = depth * charsHit;
        stats.bestComposite = std::max(stats.bestComposite, composite);
        if (composite < P.minComposite) {
            ++stats.belowComposite;
            // 落选的也要能把明细带出去。
            // 只记"最高分"不够：0101 的落选最高分 796 落在一根竖线上，
            // 而真划痕那条斜线只有 609。看"最高分"会把注意力引到错误的位置。
            if (composite > stats.bestRejectedS) {
                stats.bestRejectedS = composite;
                stats.bestRejectedLine = {p1 + cv::Point(band.x, band.y),
                                          p2 + cv::Point(band.x, band.y)};
            }
            continue;
        }

        ScratchHit h;
        h.a1 = a1;
        h.a2 = a2;
        h.box = cv::Rect(std::min(a1.x, a2.x), std::min(a1.y, a2.y), std::abs(a2.x - a1.x) + 1,
                         std::abs(a2.y - a1.y) + 1);
        h.box.x -= 4;
        h.box.y -= 8;
        h.box.width += 8;
        h.box.height += 16;
        h.len = len;
        h.depth = depth;
        h.widthPx = widthPx;
        h.energy = score;
        h.composite = composite;
        h.cov = cov;
        h.charsHit = charsHit;
        h.band = bandOf(P, composite);
        out.push_back(h);
    }
    stats.passed = static_cast<int>(out.size());
    if (outStats) *outStats = stats;

    // ---- 合并重复候选（NMS）
    //
    // HoughLinesP 对同一条划痕会吐出多条近乎重合的直线：只要有两三像素的
    // 角度差，或者中间断一截，就会被当成两条独立的线。实测 0026_FT 一条
    // 划痕报到 8 处、0026_RT 报 4 处——数量差一倍，但图上画出来的红线
    // 完全重叠，人眼只能看到一条。不合并的话：
    //   1. 状态栏和标题的"划痕 N 处"是假的，客户会以为有 8 条划痕；
    //   2. 每一份说明文字都画在同一位置，糊成一团。
    // 合并规则：按严重度从高到低保留，若与已保留者的框 IoU 明显重叠，
    // 或者两条线段的端点都挨得很近，就认为是同一条。
    if (out.size() > 1) {
        std::sort(out.begin(), out.end(), [](const ScratchHit& a, const ScratchHit& b) {
            return a.composite > b.composite;
        });
        std::vector<ScratchHit> kept;
        kept.reserve(out.size());
        auto nearPt = [](const cv::Point& a, const cv::Point& b, int tol) {
            return std::abs(a.x - b.x) <= tol && std::abs(a.y - b.y) <= tol;
        };
        for (const auto& h : out) {
            bool dup = false;
            for (const auto& k : kept) {
                const double iou =
                    static_cast<double>((h.box & k.box).area()) /
                    std::max(1.0, static_cast<double>((h.box | k.box).area()));
                // 端点顺序可能相反，两个方向都比一遍
                const bool sameEnds = (nearPt(h.a1, k.a1, 12) && nearPt(h.a2, k.a2, 12)) ||
                                      (nearPt(h.a1, k.a2, 12) && nearPt(h.a2, k.a1, 12));
                if (iou > 0.5 || sameEnds) {
                    dup = true;
                    break;
                }
            }
            if (!dup) kept.push_back(h);
        }
        out.swap(kept);
        // 去重后再刷新统计。之前 stats.passed / bestComposite 在去重前就写死了，
        // 与调用方实际拿到的 out.size() 不一致："划痕 N 处"会对不上图上的红线数。
        stats.passed = static_cast<int>(out.size());
        stats.bestComposite = 0.0;
        for (const auto& h : out)
            stats.bestComposite = std::max(stats.bestComposite, h.composite);
    }
    return out;
}

/// 一条划痕的"升降档边界"：要让它的档位变高/变低，切点必须调到多少
///
/// 这是界面上最该让客户看见的两个数。原来只画在标注图里，字小、挤在框旁边，
/// 换成列表才能一眼扫完。
///
/// 语义（以严重度 S 为分值，切点都是"低于它就落档"的下界）：
///   upCut  = 把当前档的切点压到这个值，本条就升一档；-1 表示已是最高档
///   dnCut  = 把下一档的切点抬到这个值，本条就降一档；-1 表示已是最低档
/// 注意 upCut 用的是"升档"所需的切点，dnCut 用的是"降档"所需的切点，
/// 两者都是可直接填进参数界面的具体数字，不需要反推。
struct ScratchEdges {
    double upCut = -1.0;    // 升档需要的切点
    double dnCut = -1.0;    // 降档需要的切点
    const char* upTo = "";  // 升到哪一档（英文标签）
    const char* dnTo = "";  // 降到哪一档
};

inline ScratchEdges edgesOf(const ScratchParams& P, const ScratchHit& h) {
    ScratchEdges e;
    const int b = std::clamp(h.band, 0, 3);
    if (b < 3) {
        // 升一档：把当前档位的下界压到本条严重度之下
        e.upCut = (b == 2) ? P.cutCertain : (b == 1) ? P.cutObvious : P.cutSuspect;
        e.upTo = bandTag(b + 1);
    }
    if (b > 0) {
        // 降一档：把下一档的下界抬到本条严重度之上
        e.dnCut = (b == 3) ? P.cutCertain : (b == 2) ? P.cutObvious : P.cutSuspect;
        e.dnTo = bandTag(b - 1);
    }
    return e;
}

/// 在图上标注一条划痕：线 + 框 + 右侧多行说明
///
/// 说明里直接写出「要让这一条升级/降级，切点需要调到多少」：
///   升级 = 把上一档的切点压到不超过本条严重度，本条于是落进上档；
///   降级 = 把本档的切点抬过本条严重度，本条于是掉回下档。
/// 两个数都是可直接填进参数界面的具体值，省掉反推。
inline void drawScratch(cv::Mat& vis, const ScratchHit& h, const ScratchParams& P,
                        int noteSlot = -1) {
    if (vis.empty()) return;
    const int b = std::clamp(h.band, 0, 3);
    const cv::Scalar col = bandColor(b);

    if (b >= 1) cv::line(vis, h.a1, h.a2, col, b >= 2 ? 3 : 2, cv::LINE_AA);
    const cv::Rect clip = h.box & cv::Rect(0, 0, vis.cols, vis.rows);
    if (clip.width > 0 && clip.height > 0) cv::rectangle(vis, clip, col, 2);

    // 框上不写任何文字，只保留线条与矩形。
    //
    // 这里是"证据区"：框圈住的就是那条划痕，任何压在上面的文字都会
    // 干扰判断。试过三种编号位置——左上角（两条划痕首尾相接时编号叠成
    // 一团）、右下角（落在 Silan 的字母上）、框内（直接盖住划痕）——
    // 三种都比不画更差，干脆不画。
    // 完整说明（含编号 1/2/3）由调用方合并成一块图例，画在图片空白处，
    // 见 annotate.hpp 的 drawLegendPanel。
    (void)noteSlot;
}

/// 把一条划痕的说明整理成图例里的几行（第一行带编号）
///
/// 与 drawScratch 分开，是因为图例要在**所有**划痕都检测完之后统一画，
/// 否则每画一条就占一块位置，又回到面板打架的老路。
inline std::vector<std::string> scratchLegendLines(const ScratchHit& h, const ScratchParams& P,
                                                   int index) {
    std::vector<std::string> lines;
    char head[160];
    // 只用 ASCII：putText 的 Hershey 字体没有中文字形，中文会画成一串问号。
    // 这段文字是**画进图像里**的，所以必须用 bandTag（英文），
    // 不能用 bandName（中文）——中文版给界面/表格用。
    std::snprintf(head, sizeof(head), "%d) S=%d %s", index + 1, static_cast<int>(h.composite),
                  bandTag(std::clamp(h.band, 0, 3)));
    lines.push_back(head);

    const ScratchEdges e = edgesOf(P, h);
    if (e.upCut >= 0.0) {
        char up[128];
        std::snprintf(up, sizeof(up), "   UP %s: %d->%d", e.upTo, static_cast<int>(e.upCut),
                      static_cast<int>(h.composite));
        lines.push_back(up);
    }
    if (e.dnCut >= 0.0) {
        char dn[128];
        std::snprintf(dn, sizeof(dn), "   DN %s: %d->%d", e.dnTo, static_cast<int>(e.dnCut),
                      static_cast<int>(h.composite) + 1);
        lines.push_back(dn);
    }
    return lines;
}

}  // namespace markloc
