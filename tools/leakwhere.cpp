// 扣模型这一步：为什么 logo 区永远扣不掉？
//
// 已量的前提（markloc_freqcov）：
//   region3 (474,315,322x225, Silan 椭圆 + 圆圈标记) 信号 p90 = 0.069，
//   而文字区是 0.342~0.774。也就是说 logo 在 strokeSignal 里几乎等于背景。
//   同时模型在该区有 25.1% 被算作墨。
//
// leak 的定义是 weak(信号>0.105) & ~dilate(模型墨)。若某处信号只有 0.069，
// 它根本进不了 weak，那它就不该产生 leak——所以 logo"弱"本身不足以解释 11.7% 的残留。
// 真正要问的是：那 11.7% 残留在哪？是什么形状？
//
// 本工具把 leak 按区域拆开，并对每个连通块报几何量，
// 找出"闭运算之后能撑到 60px 以上"的块到底是哪些。
//
// 用法: markloc/leakwhere <modelDir> <image>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

namespace fs = std::filesystem;

static bool readGray(const std::string& path, cv::Mat& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::vector<unsigned char> buf;
    unsigned char tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof(tmp), fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(fp);
    if (buf.empty()) return false;
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

/// 用 phaseCorrelate 把信号对齐到参照
static cv::Mat alignTo(const cv::Mat& sig, const cv::Mat& ref8, const cv::Rect& area,
                       cv::Point2d* sh) {
    cv::Mat r32;
    ref8.convertTo(r32, CV_32F);
    cv::Mat sC, rC;
    sig(area).copyTo(sC);
    r32(area).copyTo(rC);
    *sh = cv::phaseCorrelate(sC, rC, cv::noArray());
    if (std::abs(sh->x) >= 60 || std::abs(sh->y) >= 60) return sig;
    if (std::abs(sh->x) <= 0.01 && std::abs(sh->y) <= 0.01) return sig;
    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh->x, 0, 1, -sh->y);
    cv::Mat out;
    cv::warpAffine(sig, out, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_leakwhere <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("读取失败或尺寸不符\n");
        return 1;
    }

    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);
    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, gray.cols, gray.rows);
        if (b.width <= 0 || b.height <= 0) continue;
        inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(gray.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(gray.rows - inkArea.y, inkArea.height + 40);

    const cv::Mat sigRaw =
        chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Point2d sh;
    const cv::Mat sig = alignTo(sigRaw, model.ref, inkArea, &sh);
    std::printf("全局配准 (ref 为参照): (%.2f, %.2f)\n", sh.x, sh.y);

    markloc::ScratchParams P;
    const cv::Rect band = markloc::scratchBand(sig.size(), units, P.growPx);
    const cv::Mat sBand = sig(band);

    cv::Mat weak;
    cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    const cv::Mat weak0 = weak.clone();

    cv::Mat modelInk;
    cv::compare(model.freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);
    cv::Mat d, nm;
    cv::dilate(modelInk, d, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(d, nm);
    cv::bitwise_and(weak, nm, weak);

    // ---- 按 region 拆分 leak
    std::printf("\n=== leak 按区域（扣模型后、闭运算前）===\n");
    std::printf("%-10s %-22s %9s %9s %8s\n", "区域", "框", "亮像素", "残留", "残留率");
    for (size_t i = 0; i < model.regions.size(); ++i) {
        const cv::Rect rr = (model.regions[i].rect & band);
        if (rr.area() <= 0) continue;
        const cv::Rect lr(rr.x - band.x, rr.y - band.y, rr.width, rr.height);
        const int w = cv::countNonZero(weak0(lr));
        const int k = cv::countNonZero(weak(lr));
        std::printf("region%-3zu %-22s %9d %9d %7.1f%%\n", i,
                    cv::format("(%d,%d %dx%d)", rr.x, rr.y, rr.width, rr.height).c_str(), w, k,
                    w > 0 ? 100.0 * k / w : 0.0);
    }

    // ---- 离字最远的那块：框外残留
    cv::Mat inAny = cv::Mat::zeros(weak.size(), CV_8U);
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & band;
        if (b.area() > 0) inAny(cv::Rect(b.x - band.x, b.y - band.y, b.width, b.height)) = 255;
    }
    int kIn = 0, kOut = 0;
    for (int y = 0; y < weak.rows; ++y)
        for (int x = 0; x < weak.cols; ++x) {
            if (!weak.at<uchar>(y, x)) continue;
            if (inAny.at<uchar>(y, x)) ++kIn;
            else ++kOut;
        }
    std::printf("\nleak: 字框内 %d  字框外 %d  (框外占 %.1f%%)\n", kIn, kOut,
                kIn + kOut > 0 ? 100.0 * kOut / (kIn + kOut) : 0.0);

    // ---- 闭运算前后的连通块几何
    const int bk = std::max(3, P.bridge | 1);
    cv::Mat closed;
    cv::morphologyEx(weak, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk}));
    cv::medianBlur(closed, closed, 3);

    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(closed, labels, stats, cents, 8);
    struct Blk {
        int area, w, h, x, y;
        double elong;
    };
    std::vector<Blk> blks;
    for (int i = 1; i < n; ++i) {
        Blk b;
        b.x = stats.at<int>(i, cv::CC_STAT_LEFT);
        b.y = stats.at<int>(i, cv::CC_STAT_TOP);
        b.w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        b.h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        b.area = stats.at<int>(i, cv::CC_STAT_AREA);
        b.elong = std::max(b.w, b.h) / static_cast<double>(std::max(1, std::min(b.w, b.h)));
        blks.push_back(b);
    }
    std::sort(blks.begin(), blks.end(), [](const Blk& a, const Blk& c) {
        return std::max(a.w, a.h) > std::max(c.w, c.h);
    });
    std::printf("\n=== 闭运算后连通块 %d 个，按长边排序前 12 ===\n", n - 1);
    std::printf("%-26s %7s %5s %5s %6s %6s\n", "框(整图坐标)", "面积", "宽", "高", "长宽比", "够60px");
    for (size_t i = 0; i < std::min<size_t>(12, blks.size()); ++i) {
        const Blk& b = blks[i];
        std::printf("%-26s %7d %5d %5d %6.1f %6s\n",
                    cv::format("(%d,%d %dx%d)", b.x + band.x, b.y + band.y, b.w, b.h).c_str(),
                    b.area, b.w, b.h, b.elong,
                    std::max(b.w, b.h) >= static_cast<int>(P.minLen) ? "是" : "");
    }

    // ---- 落选的候选里，"最好得分"那条长什么样
    markloc::ScratchStats st;
    const auto real = markloc::findScratches(sig, model.freq, units, cfg, P, &st);
    std::printf("\n=== 真实判定路径 (ref 配准) ===\n");
    std::printf("候选=%d 长度挡=%d 覆盖挡=%d 严重度挡=%d 最好=%.0f 命中=%zu\n", st.candidates,
                st.belowLen, st.belowCov, st.belowComposite, st.bestComposite, real.size());
    for (const auto& h : real) {
        std::printf("  S=%.0f 跨%d 深%.0f 长%.0f  (%d,%d)-(%d,%d)\n", h.composite, h.charsHit,
                    h.depth, h.len, h.a1.x, h.a1.y, h.a2.x, h.a2.y);
    }

    // 视觉证据
    cv::Mat vis;
    cv::cvtColor(closed, vis, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < std::min<size_t>(12, blks.size()); ++i) {
        const Blk& b = blks[i];
        if (std::max(b.w, b.h) < static_cast<int>(P.minLen)) continue;
        cv::rectangle(vis, cv::Rect(b.x, b.y, b.w, b.h), cv::Scalar(0, 0, 255), 2);
        cv::putText(vis, cv::format("%dx%d", b.w, b.h), cv::Point(b.x, std::max(12, b.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
    std::filesystem::create_directories("out/trace");
    cv::imwrite("out/trace/leakwhere_closed.png", vis);
    cv::Mat leakVis;
    cv::cvtColor(weak, leakVis, cv::COLOR_GRAY2BGR);
    cv::imwrite("out/trace/leakwhere_leak.png", leakVis);
    return 0;
}
