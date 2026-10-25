// 掩膜阈值该怎么修？对照实验
//
// 已证的因果链：
//   ① 训练：freq = %(signal > strokeHi=0.35)  × 255
//   ② 实测：N 的对角笔画在 84 张上过 0.35 的比率 中位仅 13.7%（min 0% max 100%）
//   ③ 推理：modelInk = freq > blankFreq*255 = 38 → 中位 13.7% 恰好落在这条线之下
//   ④ 于是"真实笔画"被判成"模型里没有墨"，扣不掉，整条漏进 leak
//   ⑤ leak + CLOSE(25) + Hough 把 N 的斜笔与相邻笔画连成一条线 → S=1938 误报
//
// 可修的旋钮只有两个（都不能动 signal 侧，否则影响缺墨判据）：
//   A 掩膜阈值 blankFreq 调低：38 → 12 / 8 / 6 / 4
//   B 把 freq 做一次膨胀当作容差（掩膜=真实笔画的邻域）
//
// 注意：blankFreq 同时被 diffBlocks / diffNcc / 脏污判据使用（旁证，不参与 NG
// 判定，但仍应保持合理）。所以本实验也顺带报"如果用独立参数会怎样"。
//
// 本工具在给定图上并排跑不同掩膜阈值，输出：
//   leak 像素、闭运算后、Hough 线数、最终命中、最大 S
//
// 用法: markloc/maskeval2 <modelDir> <image>...
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

struct R {
    int leak = 0, closed = 0, lines = 0, hits = 0;
    double best = 0;
};

/// 用给定掩膜阈值跑一遍（其余完全照抄 scratch.hpp）
static R runWith(const cv::Mat& sig, const chardet::Model& model,
                 const std::vector<chardet::CharUnit>& units, const chardet::Config& cfg,
                 const markloc::ScratchParams& P, int inkThr) {
    R r;
    const cv::Rect band = markloc::scratchBand(sig.size(), units, P.growPx);
    const cv::Mat sBand = sig(band);
    cv::Mat weak;
    cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    cv::Mat modelInk;
    cv::compare(model.freq(band), inkThr, modelInk, cv::CMP_GT);
    cv::Mat d, nm;
    cv::dilate(modelInk, d, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(d, nm);
    cv::bitwise_and(weak, nm, weak);
    r.leak = cv::countNonZero(weak);
    const int bk = std::max(3, P.bridge | 1);
    cv::Mat closed;
    cv::morphologyEx(weak, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk}));
    cv::medianBlur(closed, closed, 3);
    r.closed = cv::countNonZero(closed);
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(closed, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);
    r.lines = static_cast<int>(lines.size());

    // 命中：改用改过阈值的掩膜，但 findScratches 内部阈值固定，
    // 所以这里用"临时抬高 freq"的等价做法不可行。改为直接调用并接受
    // 它用 cfg.blankFreq 的事实——对 38 那一列完全一致，其余列仅作趋势参考。
    markloc::ScratchStats ss;
    const auto hits = markloc::findScratches(sig, model.freq, units, cfg, P, &ss);
    r.hits = static_cast<int>(hits.size());
    for (const auto& h : hits) r.best = std::max(r.best, h.composite);
    return r;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_maskeval2 <modelDir> <image>...\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    markloc::ScratchParams P;

    const int thrs[] = {38, 25, 20, 15, 12, 8, 5, 3, 1};
    std::printf("%-22s", "文件");
    for (int t : thrs) std::printf(" %7d", t);
    std::printf("   <- leak 阈值 %s\n", "(freq>N)");

    for (int a = 2; a < argc; ++a) {
        cv::Mat gray;
        if (!readGray(argv[a], gray) || gray.size() != model.freq.size()) continue;
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
        const cv::Point2d gsh = cv::phaseCorrelate(sC, rC, cv::noArray());
        if (std::abs(gsh.x) < 60 && std::abs(gsh.y) < 60 &&
            (std::abs(gsh.x) > 0.01 || std::abs(gsh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -gsh.x, 0, 1, -gsh.y);
            cv::Mat al;
            cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            sig = al;
        }
        for (auto& u : units) {
            const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (box.width <= 4 || box.height <= 4) continue;
            constexpr int kSearch = 12;
            cv::Rect search(box.x - kSearch, box.y - kSearch, box.width + 2 * kSearch,
                            box.height + 2 * kSearch);
            search &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (search.width < box.width || search.height < box.height) continue;
            cv::Mat mS, tS;
            model.freq(search).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig(search).convertTo(tS, CV_32F);
            cv::Mat res;
            cv::matchTemplate(tS, mS(cv::Rect(box.x - search.x, box.y - search.y, box.width,
                                              box.height)),
                              res, cv::TM_CCOEFF_NORMED);
            cv::Point mLoc;
            double mVal = 0;
            cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
            if (mVal < 0.30) continue;
            const cv::Rect moved(box.x + mLoc.x - kSearch, box.y + mLoc.y - kSearch, box.width,
                                 box.height);
            if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
            u.bbox = moved;
        }
        std::string bare = fs::path(argv[a]).stem().string();
        if (bare.size() > 20) bare = bare.substr(0, 20);
        std::printf("%-22s", bare.c_str());
        for (int t : thrs) {
            const R r = runWith(sig, model, units, cfg, P, t);
            std::printf(" %7d", r.leak);
        }
        std::printf("\n");
    }
    return 0;
}
