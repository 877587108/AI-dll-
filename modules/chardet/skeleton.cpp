// 字符级缺陷判据：归一化互相关（NCC）+ 骨架拓扑
//
// 背景：为什么之前的墨量类判据全部失败
//
//   已系统尝试 8 种判据：二值化投影差分、尺度归一、连续信号投影、
//   直方图匹配、亚像素对齐投影、笔画粗细归一、残差连通域、骨架拓扑。
//   全部无法区分 OK 与 NG，最难的样本始终是"笔画偏细的 OK 件"
//   （实测 OK-0012 在多数判据上分数最高，甚至高于真实 NG）。
//
//   根本原因：模型是 84 张平均的频率图，`freq > 0.5` 得到的是**交集区域**，
//   比任何单张图都窄。于是任意 OK 样本与模型比对时，都会同时出现
//   "系统性多墨"（单张更宽）与"系统性少墨"（交集外部分），
//   两者相互抵消，真实缺陷被淹没。
//
// 本文件的两个判据都绕开了"比绝对墨量"：
//
//   ① NCC —— 对整体粗细/亮度是线性的，只反映形态分布是否一致
//   ② 骨架拓扑 —— 对笔画粗细完全不敏感，只反映笔画结构

#include "chardet.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace chardet {

namespace {

/// 细化：把笔画压成 1px 骨架
cv::Mat skeletonize(const cv::Mat& bin) {
    cv::Mat skel = cv::Mat::zeros(bin.size(), CV_8U);
    cv::Mat img = bin.clone();
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_CROSS, {3, 3});
    cv::Mat temp, eroded;
    for (int i = 0; i < 200; ++i) {
        cv::erode(img, eroded, k);
        cv::dilate(eroded, temp, k);
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);
        if (cv::countNonZero(img) == 0) break;
    }
    return skel;
}

/// 骨架拓扑特征
struct Topo {
    int endpoints = 0;   // 端点（1 个 8-邻域）
    int branches = 0;    // 分支点（>=3 个 8-邻域）
    int length = 0;      // 骨架像素数
};

Topo analyzeSkeleton(const cv::Mat& skel) {
    Topo t;
    if (skel.empty()) return t;
    for (int y = 0; y < skel.rows; ++y) {
        const uchar* row = skel.ptr<uchar>(y);
        for (int x = 0; x < skel.cols; ++x) {
            if (!row[x]) continue;
            ++t.length;
            int n = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    const int yy = y + dy, xx = x + dx;
                    if (yy < 0 || yy >= skel.rows || xx < 0 || xx >= skel.cols) continue;
                    if (skel.at<uchar>(yy, xx)) ++n;
                }
            }
            if (n == 1) ++t.endpoints;
            else if (n >= 3) ++t.branches;
        }
    }
    return t;
}

}  // namespace

NccResult diffNcc(const cv::Mat& signal, const cv::Mat& modelFreq, const CharUnit& unit,
                  const Config& cfg) {
    NccResult out;
    const cv::Rect img(0, 0, signal.cols, signal.rows);
    const cv::Rect box = unit.bbox & img;
    if (box.width <= 3 || box.height <= 3) return out;
    // 跳过超大框（logo 区）：形状复杂、模板频率低、背景划痕多。
    // 实测它是 OK 件误报的主要来源——OK-0014/0028/0030 的最差字符
    // 都是 #21 这个 279×177 的 logo 框。字符级判据对它不适用。
    if (box.width > 200 || box.height > 160) return out;
    if (signal.type() != CV_32F || modelFreq.size() != signal.size()) return out;

    // 用连续值而非二值，保留"墨有多强"的信息
    cv::Mat mf, tf;
    modelFreq(box).convertTo(mf, CV_32F, 1.0 / 255.0);
    signal(box).convertTo(tf, CV_32F);

    // 用模型频率把像素分成三组，检查待检是否符合预期
    //
    // 为什么不用逐点 NCC：
    //   模型是"有墨的概率"，单张图是"确定有无墨"。即使完全正常的样本，
    //   与概率图的逐点相关也只有 0.80 左右（模型自比才是 0.97），
    //   这个基线偏差会淹没缺陷信号。
    //
    //   改成分组判断，只看"是否违反预期"，与概率值本身的高低无关：
    //     freq > 0.70 → 必然有墨，待检无墨 = 缺墨
    //     freq < 0.15 → 不应有墨，待检有墨 = 多墨
    //   中间地带不参与判定，避开交界处的模糊性。
    // 阈值按该字符模板的最大频率缩放，而不是固定 0.70。
    //
    // 有些字符在训练集中位置不够稳定，平均后峰值频率偏低（实测 #14 墨量 2298，
    // 远低于其他字符的 3000+），固定 0.70 会让它们的"必然有墨"区几乎为空，
    // 判据随之失效。按峰值自适应可保证每个字符都有足够的分析区域。
    double mMax = 0;
    cv::minMaxLoc(mf, nullptr, &mMax);
    const double inkThr = std::max(0.50, mMax * 0.75);

    cv::Mat sureInk, sureBlank;
    cv::compare(mf, inkThr, sureInk, cv::CMP_GT);
    cv::compare(mf, cfg.blankFreq, sureBlank, cv::CMP_LT);
    const int nInk = cv::countNonZero(sureInk);
    const int nBlank = cv::countNonZero(sureBlank);
    if (nInk < 20) return out;

    // 待检的二值墨 —— 阈值按该字符的实际墨强度自适应。
    //
    // 固定阈值 0.35 的问题：刻印整体偏暗的图里，所有字符都会落到阈值以下，
    // 导致"全部字符都缺墨"的误报。实测 0007_RT 所有 21 个字符缺墨率 18~57%，
    // 而正常的 0001_RT 全部 <2% —— 这不是缺笔画，是整张图信号偏弱。
    //
    // 做法：在"必然有墨"的像素上取待检信号的中位数，作为该字符的参考强度，
    // 阈值取它的一半。这样刻印偏暗时阈值自动降低，缺一笔时仍能检出。
    double inkRef = 0.0;
    {
        std::vector<float> vals;
        for (int yy = 0; yy < box.height; ++yy) {
            const float* ps = signal(box).ptr<float>(yy);
            const uchar* pm = sureInk.ptr<uchar>(yy);
            for (int xx = 0; xx < box.width; ++xx) {
                if (pm[xx]) vals.push_back(ps[xx]);
            }
        }
        if (!vals.empty()) {
            std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
            inkRef = vals[vals.size() / 2];
        }
    }
    // 参考强度过低（整字偏淡）或未取到，退回配置值
    const double inkThr2 = inkRef > cfg.missStrokeLo * 1.5
                               ? std::clamp(inkRef * 0.5, cfg.missStrokeLo, cfg.strokeHi * 1.2)
                               : cfg.strokeHi;

    cv::Mat tBin;
    cv::compare(signal(box), inkThr2, tBin, cv::CMP_GT);

    // 缺墨：必然有墨处，待检无墨
    cv::Mat notInk;
    cv::bitwise_not(tBin, notInk);
    cv::Mat missPx;
    cv::bitwise_and(sureInk, notInk, missPx);
    const int missN = cv::countNonZero(missPx);

    // 多墨：应为空白处，待检有墨
    //
    // 只统计"紧贴笔画的空白"，不统计远离笔画的背景。
    // 背景里的划痕、脏污会大面积落在 freq<0.15 的区域，
    // 若全部算作多墨，实测有 1/3 的 OK 件被判 NG（它们确实有划痕）。
    // 划痕应由专门的划痕检测处理，这里只关心"字长胖了没有"。
    cv::Mat inkNear, nearBlank;
    {
        const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {11, 11});
        cv::dilate(sureInk, inkNear, kd);   // 必然有墨区外扩 5px
        cv::Mat farFromInk;
        cv::bitwise_not(inkNear, farFromInk);
        cv::bitwise_and(sureBlank, farFromInk, nearBlank);   // 空白且贴近笔画
    }

    cv::Mat extraPx;
    cv::bitwise_and(nearBlank, tBin, extraPx);
    const int extraN = cv::countNonZero(extraPx);

    // 去掉沿笔画边缘的细碎差异（形态学开运算）
    // 边缘 1~2px 的差异由刻印抖动造成，不是缺陷
    const cv::Mat kOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
    cv::Mat missOpen, extraOpen;
    cv::morphologyEx(missPx, missOpen, cv::MORPH_OPEN, kOpen);
    cv::morphologyEx(extraPx, extraOpen, cv::MORPH_OPEN, kOpen);
    const int missOpenN = cv::countNonZero(missOpen);
    const int extraOpenN = cv::countNonZero(extraOpen);

    // 背景干扰检测：若"应为空白处"大面积出现墨，说明该框有划痕或被污染，
    // 此时字节级判定不可信（实测 84 张 OK 中有 33 张因背景划痕被误判）。
    // 这类框直接跳过，交给专门的划痕检测处理。
    {
        const double outArea = static_cast<double>(nBlank);
        if (outArea > 0) {
            // 用未开运算的原始多余墨量衡量背景污染程度
            const double rawExtra = static_cast<double>(extraN);
            if (rawExtra / std::max(1.0, outArea) > 0.25) {
                // 背景污染过重：本字符不参与判定
                out.box = box;
                out.valid = false;
                return out;
            }
        }
    }

    out.box = box;
    out.ncc = 0.0;   // 字段保留但不再使用
    out.valid = true;

    // 归一化：相对"必然有墨"的总面积
    out.missRatio = static_cast<double>(missOpenN) / nInk;
    out.extraRatio = nBlank > 0 ? static_cast<double>(extraOpenN) / nInk : 0.0;


    // 局部新增墨：最大连通块面积 / 模型墨量
    //
    // 为什么需要这个额外指标：
    //   改笔画（0->Q、S->6）新增的墨只占该字符的 2%~12%，
    //   用"总墨量比"衡量时，0->Q 只有 0.023，与 OK 的上限 0.027 几乎重叠。
    //   但这一小笔是"成块的、集中在某处"的，而 OK 的多余墨是散在边缘的碎片。
    //   取最大连通块面积，就能把"集中的一笔"和"分散的碎屑"区分开。
    {
        cv::Mat lb, st, ce;
        const int n = cv::connectedComponentsWithStats(extraOpen, lb, st, ce, 8, CV_32S);
        int maxBlob = 0;
        for (int i = 1; i < n; ++i) {
            maxBlob = std::max(maxBlob, st.at<int>(i, cv::CC_STAT_AREA));
        }
        out.localExtra = static_cast<double>(maxBlob) / nInk;
    }
    // 定位：缺陷像素在框内的列/行范围
    {
        const auto spansOf = [](const cv::Mat& m) {
            std::vector<cv::Range> sp;
            cv::Mat px, py;
            cv::reduce(m, px, 0, cv::REDUCE_SUM, CV_32S);
            cv::reduce(m, py, 1, cv::REDUCE_SUM, CV_32S);
            const auto scan = [](const cv::Mat& p, bool rowWise, std::vector<cv::Range>& dst) {
                const int n = rowWise ? p.rows : p.cols;
                int start = -1;
                for (int i = 0; i < n; ++i) {
                    const int v = rowWise ? p.at<int>(i, 0) : p.at<int>(0, i);
                    const bool on = v > 0;
                    if (on && start < 0) start = i;
                    if (!on && start >= 0) {
                        if (i - start >= 2) dst.emplace_back(start, i);
                        start = -1;
                    }
                }
                if (start >= 0 && n - start >= 2) dst.emplace_back(start, n);
            };
            scan(px, false, sp);
            scan(py, true, sp);
            return sp;
        };
        out.missSpanX = spansOf(missOpen);
        out.extraSpanX = spansOf(extraOpen);
    }

    // 判据：超过门槛才算缺陷
    //
    // 门槛取 5%：实测正常件的边缘抖动经开运算后残留 < 2%，
    // 而"缺一笔"通常占该字符墨量的 15% 以上。
    constexpr double kThr = 0.15;

    // ---- 断连判定：缺墨是否把笔画切断
    //
    // 只看缺墨面积不够。边缘磨掉 15% 和横断一笔 15%，对人的意义完全不同：
    // 前者可能还认得出，后者直接破坏识别。原来的打分对两者一视同仁。
    //
    // 判法：把"模型该有的墨"减去"实际缺掉的墨"，数剩下的墨被切成了几块。
    //   边缘磨损：剩余墨仍是 1 大块（块数与模型一致）
    //   笔画断连：剩余墨碎成 ≥2 块，且出现明显的小碎块
    // 模型侧同样处理一次作为基准，避免把模型本身的碎块算进来。
    //
    // 输出：broken = 断出的新块数（相对模型），断得越多分越高。
    int brokenParts = 0;
    {
        cv::Mat modelInkBin;
        cv::compare(mf, cfg.missFreq, modelInkBin, cv::CMP_GT);
        if (cv::countNonZero(modelInkBin) > 0) {
            // 模型墨减去实际缺墨 = 断裂后残留的墨
            cv::Mat remain;
            cv::bitwise_and(modelInkBin, tBin, remain);
            // 只保留成规模的块，边缘碎屑不算断连
            cv::Mat kkeep = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
            cv::morphologyEx(remain, remain, cv::MORPH_OPEN, kkeep);

            auto bigBlocks = [](const cv::Mat& m) {
                cv::Mat lb, st, ce;
                const int n = cv::connectedComponentsWithStats(m, lb, st, ce, 8, CV_32S);
                int cnt = 0;
                for (int i = 1; i < n; ++i)
                    if (st.at<int>(i, cv::CC_STAT_AREA) >= 25) ++cnt;   // 至少 5x5 的面积
                return cnt;
            };
            cv::Mat modelKeep = modelInkBin.clone();
            cv::morphologyEx(modelKeep, modelKeep, cv::MORPH_OPEN, kkeep);
            const int nModel = bigBlocks(modelKeep);
            const int nRemain = bigBlocks(remain);
            brokenParts = std::max(0, nRemain - nModel);
        }
    }

    // 缺墨判据
    if (out.missRatio > kThr) {
        out.missScore = std::min(3.0, (out.missRatio - kThr) / 0.15);
        // 笔画被切断：这是"断字"，破坏识别，给更高权重
        //
        // 权重用乘法而不是加法：断开 1 处 x1.6，2 处 x2.2，3 处及以上 x2.8。
        // 乘法保证"面积越大分越高"这个基本单调性不变，
        // 只是在同等面积下，断连的比掉边的更重。
        if (brokenParts >= 1) {
            const double w = brokenParts == 1 ? 1.6 : (brokenParts == 2 ? 2.2 : 2.8);
            out.missScore = std::min(3.0, out.missScore * w);
        }
        out.score = std::max(out.score, out.missScore);
    }

    // 多墨判据（整框比例）
    if (out.extraRatio > kThr) {
        out.extraScore = std::min(3.0, (out.extraRatio - kThr) / 0.15);
        out.score = std::max(out.score, out.extraScore);
    }

    // 改笔画判据：集中的新增墨（抗稀释）
    //
    // 实测：S->6（新增 12% 墨）localExtra = 0.108；OK 样本上限 0.027。
    // 取 0.05 作门槛，是 OK 上限的 1.9 倍，留出安全余量。
    //
    // 局限：0->Q 只增 2% 的极细改动检不出（localExtra 0.018），
    // 其变化量与 OK 的刻印波动同量级，属于当前模型表示的能力上限。
    constexpr double kLocalThr = 0.05;
    if (out.localExtra > kLocalThr) {
        const double s = std::min(3.0, (out.localExtra - kLocalThr) / 0.10);
        out.extraScore = std::max(out.extraScore, s);
        out.score = std::max(out.score, out.extraScore);
    }

    return out;
}


}  // namespace chardet
