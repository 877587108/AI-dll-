// 逐字符对齐验证
//
// 背景（要验证的假设）：
//   DefectViewWindow.cpp:1131-1196 的逐字符对齐只改了 u.bbox（框坐标），
//   没有改 signal_（信号图）也没有改 model_.freq（频率图）。
//   而 findScratches 里 signal(band) 和 freq(band) 都按整图坐标取样，
//   这意味着"框对齐了但像素没对齐"，两者差一个 (dx,dy)，
//   会在字形边缘留下一圈"信号有墨、模型空白"的漏出。
//
// 本工具输出：
//   1. 全局相位相关位移
//   2. 每个字符框 matchTemplate 得到的 (dx,dy) 与匹配置信度
//   3. 逐字符 (dx,dy) 相对全局位移的残余量 —— 这就是没被修正的错位
//   4. 按残余量重采样 signal 后，重算掩膜漏出率，对比"修之前/修之后"
//
// 用法: markloc/alignprobe <modelDir> <image>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

namespace fs = std::filesystem;

static fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

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

static double leakRatio(const cv::Mat& sig, const cv::Mat& freq, const std::vector<chardet::CharUnit>& units,
                        const chardet::Config& cfg, double* sigCntOut) {
    markloc::ScratchParams P;
    const cv::Rect band = markloc::scratchBand(sig.size(), units, P.growPx);
    if (band.area() <= 0) return -1;
    const cv::Mat sB = sig(band);
    const cv::Mat fB = freq(band);
    cv::Mat weak;
    cv::compare(sB, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    const int sigCnt = cv::countNonZero(weak);
    cv::Mat modelInk;
    cv::compare(fB, static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);
    const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
    cv::Mat modelDil, notModel, left;
    cv::dilate(modelInk, modelDil, kd);
    cv::bitwise_not(modelDil, notModel);
    cv::bitwise_and(weak, notModel, left);
    if (sigCntOut) *sigCntOut = sigCnt;
    return 100.0 * cv::countNonZero(left) / std::max(1, sigCnt);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_alignprobe <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("图片读取失败或尺寸不符\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // ---- 1. 全局相位相关（复刻 :1079-1118）----
    cv::Mat r32, s32;
    model.freq.convertTo(r32, CV_32F);
    sig.convertTo(s32, CV_32F);
    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width <= 0 || b.height <= 0) continue;
        inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    cv::Point2d gsh(0, 0);
    cv::Mat sigAligned = sig.clone();
    if (inkArea.area() > 0) {
        inkArea.x = std::max(0, inkArea.x - 20);
        inkArea.y = std::max(0, inkArea.y - 20);
        inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
        inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
        cv::Mat rC, sC;
        r32(inkArea).copyTo(rC);
        s32(inkArea).copyTo(sC);
        gsh = cv::phaseCorrelate(sC, rC, cv::noArray());
        std::printf("全局相位相关: (%.2f, %.2f)\n", gsh.x, gsh.y);
        if (std::abs(gsh.x) < 60 && std::abs(gsh.y) < 60 &&
            (std::abs(gsh.x) > 0.01 || std::abs(gsh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -gsh.x, 0, 1, -gsh.y);
            cv::warpAffine(sig, sigAligned, M, sig.size(),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
            std::printf("已应用全局配准\n");
        }
    }

    double sigCnt = 0;
    const double leakBefore = leakRatio(sigAligned, model.freq, units, cfg, &sigCnt);
    std::printf("\n【修之前】全局配准后，掩膜漏出率 = %.1f%%  (信号亮像素 %d)\n", leakBefore,
                static_cast<int>(sigCnt));

    // ---- 2. 逐字符 matchTemplate，测真实 (dx,dy) ----
    std::printf("\n%-6s %-22s %8s %8s %7s %8s\n", "char", "bbox", "dx", "dy", "mVal",
                "残余偏移");
    struct Perc {
        int idx;
        int dx, dy;
        double mVal;
        cv::Rect box;
    };
    std::vector<Perc> percs;
    constexpr int kSearch = 12;
    for (auto& u : units) {
        const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (box.width <= 4 || box.height <= 4) continue;
        cv::Rect search(box.x - kSearch, box.y - kSearch, box.width + 2 * kSearch,
                        box.height + 2 * kSearch);
        search &= cv::Rect(0, 0, sig.cols, sig.rows);
        if (search.width < box.width || search.height < box.height) continue;

        cv::Mat mS, tS;
        model.freq(search).convertTo(mS, CV_32F, 1.0 / 255.0);
        sigAligned(search).convertTo(tS, CV_32F);
        const cv::Rect templRect(box.x - search.x, box.y - search.y, box.width, box.height);
        cv::Mat res;
        cv::matchTemplate(tS, mS(templRect), res, cv::TM_CCOEFF_NORMED);
        cv::Point mLoc;
        double mVal = 0;
        cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
        if (mVal < 0.30) continue;
        const int dx = mLoc.x - kSearch;
        const int dy = mLoc.y - kSearch;
        percs.push_back({u.index, dx, dy, mVal, box});

        // 残余 = 逐字符要求的位移 - 全局已经做的位移
        const double resX = dx - gsh.x;
        const double resY = dy - gsh.y;
        std::printf("%-6d (%4d,%4d %3dx%3d) %8d %8d %7.3f  (%+.1f,%+.1f)\n", u.index, box.x,
                    box.y, box.width, box.height, dx, dy, mVal, resX, resY);
    }

    if (percs.empty()) {
        std::printf("没有可对齐的字符\n");
        return 1;
    }

    // ---- 3. 残余偏移统计 ----
    std::vector<double> rx, ry;
    for (const auto& p : percs) {
        rx.push_back(std::abs(p.dx - gsh.x));
        ry.push_back(std::abs(p.dy - gsh.y));
    }
    std::sort(rx.begin(), rx.end());
    std::sort(ry.begin(), ry.end());
    double sumX = 0, sumY = 0;
    for (double v : rx) sumX += v;
    for (double v : ry) sumY += v;
    std::printf("\n残余偏移 |dx-全局|: 中位 %.2f 最大 %.2f\n",
                rx[rx.size() / 2], rx.back());
    std::printf("残余偏移 |dy-全局|: 中位 %.2f 最大 %.2f\n",
                ry[ry.size() / 2], ry.back());
    std::printf("残余偏移 平均: (%.2f, %.2f) px\n", sumX / rx.size(), sumY / ry.size());

    // ---- 4. 按逐字符残余量重采样 signal，重算漏出率 ----
    //
    // 做法：造一张新图 sigFixed，对每个字符区，
    //   把 sigAligned 里 (x - resX, y - resY) 的像素搬到 (x, y)，
    //   使每个字符都对齐到它自己的框。框外保持原样。
    cv::Mat sigFixed = sigAligned.clone();
    for (const auto& p : percs) {
        const double resX = p.dx - gsh.x;
        const double resY = p.dy - gsh.y;
        if (std::abs(resX) < 0.01 && std::abs(resY) < 0.01) continue;
        const cv::Rect& box = p.box;
        // 用 warpAffine 把这块区域按 -resX,-resY 平移后贴回
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, resX, 0, 1, resY);
        cv::Mat patch;
        cv::warpAffine(sigAligned(box), patch, M, box.size(),
                       cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
        patch.copyTo(sigFixed(box));
    }

    // 框也要跟着走（模拟 :1192 的 u.bbox = moved）
    std::vector<chardet::CharUnit> unitsMoved = units;
    for (auto& u : unitsMoved) {
        for (const auto& p : percs) {
            if (p.idx == u.index) {
                u.bbox = cv::Rect(p.box.x + p.dx, p.box.y + p.dy, p.box.width, p.box.height);
            }
        }
    }

    const double leakAfter = leakRatio(sigFixed, model.freq, unitsMoved, cfg, &sigCnt);
    std::printf("\n【修之后】逐字符重采样后，掩膜漏出率 = %.1f%%\n", leakAfter);
    std::printf("改善: %.1f 个百分点\n", leakBefore - leakAfter);

    std::printf("\n=== 判据 ===\n");
    std::printf("若残余偏移明显非零（中位 > 1px），说明逐字符对齐确实没有修正像素，假设成立。\n");
    std::printf("若修之后漏出率显著下降，说明这就是误报的机制。\n");
    return 0;
}
