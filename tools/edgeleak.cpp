// 字形边缘漏出：弱阈值 vs 模型二值阈值的错配有多大？
//
// 推理链（已有证据）：
//   · scratch.hpp:190   weak = signal > missStrokeLo*0.7 = 0.15*0.7 = 0.105
//   · scratch.hpp:194   modelInk = freq > blankFreq*255 = 38/255 = 0.149
//   · 训练（model.cpp:158） freq 却是用 **strokeHi=0.35** 二值化累加的平均，
//     freq = (该像素在多少比例的样本里 signal>0.35) * 255
//   · 所以 freq=38 的意思是"38% 的样本里 signal>0.35"，不是"signal≈0.15"
//
// 后果：freq 在 [38,255] 区间的像素，实际 signal 可能只有 0.15~0.35，
// 它们照样被 weak 选中（>0.105），又被当作"模型墨"扣掉——扣得过头，
// 反而没事。真正的漏洞在另一侧：freq 恰好掉到 38 以下、
// 但 signal 仍 >0.105 的字形边缘一圈。9x9 膨胀本意是吸收这圈，
// 但如果这圈比 9px 宽，就会被当 leak，且它天然是圆弧——正是观测到的形状。
//
// 本工具直接把这两张图（判定阈值图 vs 掩膜图）叠出来，量：
//   1) freq 在 (0,38] 但 signal>0.105 的像素有多少、分布在字框边缘多宽
//   2) 这圈里有多少能活过 9x9 膨胀
//   3) 逐字统计"该字框内未被模型覆盖的亮像素"，直方图
//
// 用法: markloc/edgeleak <modelDir> <image>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_edgeleak <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("读取失败或尺寸不符\n");
        return 1;
    }
    std::vector<chardet::CharUnit> units;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) units.push_back(std::move(x));
    }
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat r32;
    model.ref.convertTo(r32, CV_32F);
    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.area() > 0) inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
    cv::Mat sC, rC;
    sig(inkArea).copyTo(sC);
    r32(inkArea).copyTo(rC);
    const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
    if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
        (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
        cv::Mat al;
        cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
        sig = al;
    }
    std::printf("配准 (ref): (%+.2f, %+.2f)\n", sh.x, sh.y);

    const double weakThr = cfg.missStrokeLo * 0.7;
    const int inkThr = static_cast<int>(cfg.blankFreq * 255);
    std::printf("weak 阈值 = %.3f   模型墨阈值 = %d/255 = %.3f   strokeHi = %.2f\n", weakThr,
                inkThr, inkThr / 255.0, cfg.strokeHi);

    // ---- 全图三张掩膜
    cv::Mat weak, modelInk;
    cv::compare(sig, weakThr, weak, cv::CMP_GT);
    cv::compare(model.freq, inkThr, modelInk, cv::CMP_GT);

    // 现状：9x9 膨胀后扣除
    cv::Mat d9, nm9, leak9;
    cv::dilate(modelInk, d9, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(d9, nm9);
    cv::bitwise_and(weak, nm9, leak9);

    // 假想：完全不扣模型
    const int nWeak = cv::countNonZero(weak);
    const int nInk = cv::countNonZero(modelInk);

    std::printf("\n全图: weak=%d  模型墨=%d\n", nWeak, nInk);

    // 膨胀半径扫描：核越大扣得越多，看曲线在哪饱和
    std::printf("\n=== 膨胀核扫描（全图）===\n%6s %10s %10s %8s\n", "核", "扣后剩余", "相对不扣", "降幅");
    for (int k : {1, 3, 5, 7, 9, 11, 13, 15, 19, 25, 31, 41}) {
        cv::Mat dk, nmk, lk;
        cv::dilate(modelInk, dk, cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k}));
        cv::bitwise_not(dk, nmk);
        cv::bitwise_and(weak, nmk, lk);
        const int r = cv::countNonZero(lk);
        std::printf("%6d %10d %9.1f%% %7.1f%%\n", k, r, 100.0 * r / std::max(1, nWeak),
                    100.0 - 100.0 * r / std::max(1, nWeak));
    }

    // ---- 逐字：框内未被覆盖的亮像素，以及"这些亮像素在模型里的 freq 是多少"
    std::printf("\n=== 逐字：残留亮像素的模型 freq 分布 ===\n");
    std::printf("%-6s %-22s %8s %8s %8s %8s %8s\n", "index", "框", "亮像素", "残留", "残留率",
                "freq均值", "freq中位");
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.area() <= 0) continue;
        const cv::Mat w = weak(b), l = leak9(b), f = model.freq(b);
        const int nw = cv::countNonZero(w);
        const int nl = cv::countNonZero(l);
        std::vector<int> fv;
        for (int y = 0; y < l.rows; ++y)
            for (int x = 0; x < l.cols; ++x)
                if (l.at<uchar>(y, x)) fv.push_back(f.at<uchar>(y, x));
        std::sort(fv.begin(), fv.end());
        const double fm = fv.empty() ? 0
                                     : std::accumulate(fv.begin(), fv.end(), 0.0) / fv.size();
        const int fmed = fv.empty() ? 0 : fv[fv.size() / 2];
        std::printf("%-6d %-22s %8d %8d %7.1f%% %8.1f %8d\n", u.index,
                    cv::format("(%d,%d %dx%d)", b.x, b.y, b.width, b.height).c_str(), nw, nl,
                    nw > 0 ? 100.0 * nl / nw : 0.0, fm, fmed);
    }

    // ---- 视觉：weak 红 / 模型墨 绿 / 交集 黄
    cv::Mat vis(sig.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < vis.rows; ++y) {
        for (int x = 0; x < vis.cols; ++x) {
            const bool w = weak.at<uchar>(y, x), m = modelInk.at<uchar>(y, x);
            cv::Vec3b& px = vis.at<cv::Vec3b>(y, x);
            if (w && m) px = cv::Vec3b(0, 255, 255);        // 黄：两边都有
            else if (w) px = cv::Vec3b(0, 0, 255);          // 红：只有信号有
            else if (m) px = cv::Vec3b(0, 255, 0);          // 绿：只有模型有
        }
    }
    for (const auto& u : units) cv::rectangle(vis, u.bbox, cv::Scalar(120, 120, 120), 1);
    fs::create_directories("out/trace");
    cv::imwrite("out/trace/edgeleak_compare.png", vis);
    std::printf("\n视觉证据 -> out/trace/edgeleak_compare.png （红=仅信号 绿=仅模型 黄=两者）\n");
    return 0;
}
