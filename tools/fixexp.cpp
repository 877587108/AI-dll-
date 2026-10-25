// 修复实验：把逐字符求出的位移真正作用到 signal 上，看误报是否消失
//
// 已证实的因果（全部来自测量，非推理）：
//   1. 0109_RT 上 S=1938 那条线落点 (970,808)-(1065,1021) 在 "8N"/"Z1" 的字形缝里，
//      原图那里干净（out/trace/zoom_0109_hit.png）——它是链路产物，不是被测物。
//   2. 逐字符剖面错位：region0 全部 +6.00、region1 #5~#8 全部 +6.00（搜索窗上限，饱和），
//      而干净件 0001_RT / 0104_RT 都在 ±2px 内（markloc_thrmatch）。
//   3. batchcore.hpp:163-176 求出 (dx,dy) 后**只改 u.bbox，不改 signal 像素**，
//      于是框对齐了、扣模型用的像素还错开约 10px。
//   4. 错开 10px 后，字形边缘整条落在 freq 低的区域 → 扣不掉 → leak 一夜之间成片，
//      再被 CLOSE(25) 连起来、Hough 出线。
//
// 本工具做对照：同一张图，分别用
//   A 现状（只移框）
//   B 移框 + 把 signal 按 (dx,dy) 重采样到框内原位
// 跑完整判定，比较命中条数与最大严重度。
//
// 用法: markloc/fixexp <modelDir> <image>...
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

struct Result {
    int hits = 0;
    double best = 0;
    int chars = 0;
    double depth = 0;
    double len = 0;
    int leakPx = 0;
};

/// 跑一遍，fixPixels 决定是否把位移作用到信号上
static Result run(const cv::Mat& sigIn, const chardet::Model& model,
                  const std::vector<chardet::CharUnit>& base, const chardet::Config& cfg,
                  const markloc::ScratchParams& P, bool fixPixels, const cv::Rect& inkArea,
                  const cv::Point2d& gsh) {
    cv::Mat sig = sigIn.clone();
    std::vector<chardet::CharUnit> units = base;
    for (auto& u : units) {
        cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
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
        cv::matchTemplate(
            tS, mS(cv::Rect(box.x - search.x, box.y - search.y, box.width, box.height)), res,
            cv::TM_CCOEFF_NORMED);
        cv::Point mLoc;
        double mVal = 0;
        cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
        if (mVal < 0.30) continue;
        const int dx = mLoc.x - kSearch, dy = mLoc.y - kSearch;
        const cv::Rect moved(box.x + dx, box.y + dy, box.width, box.height);
        if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;

        if (fixPixels) {
            // 把该字符区域内的信号按 (-dx,-dy) 重采样，使字形落回框内原位。
            // 取样区取框外扩 kSearch，保证平移后不引入黑边。
            const cv::Rect pull = (moved & cv::Rect(0, 0, sig.cols, sig.rows));
            cv::Mat patch;
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -dx, 0, 1, -dy);
            cv::warpAffine(sig(pull), patch, M, pull.size(),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
            patch.copyTo(sig(pull));
        }
        u.bbox = moved;
    }
    (void)inkArea;
    (void)gsh;

    markloc::ScratchStats ss;
    const auto hits = markloc::findScratches(sig, model.freq, units, cfg, P, &ss);
    Result r;
    r.hits = static_cast<int>(hits.size());
    for (const auto& h : hits) {
        if (h.composite > r.best) {
            r.best = h.composite;
            r.chars = h.charsHit;
            r.depth = h.depth;
            r.len = h.len;
        }
    }
    // leak 量
    return r;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_fixexp <modelDir> <image>...\n");
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

    std::printf("%-26s %-22s %-22s\n", "文件", "A 现状(只移框)", "B 移框+移像素");
    std::printf("%-26s %7s %8s %7s %8s\n", "", "命中", "最大S", "命中", "最大S");
    int aNg = 0, bNg = 0, total = 0;
    for (int a = 2; a < argc; ++a) {
        cv::Mat gray;
        if (!readGray(argv[a], gray) || gray.size() != model.freq.size()) {
            continue;
        }
        ++total;
        std::vector<chardet::CharUnit> base;
        for (size_t ri = 0; ri < model.regions.size(); ++ri) {
            auto u = chardet::toUnits(model, static_cast<int>(ri));
            for (auto& x : u) base.push_back(std::move(x));
        }
        cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        cv::Mat r32;
        model.ref.convertTo(r32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : base) {
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
        const Result ra = run(sig, model, base, cfg, P, false, inkArea, gsh);
        const Result rb = run(sig, model, base, cfg, P, true, inkArea, gsh);
        std::string bare = fs::path(argv[a]).stem().string();
        if (bare.size() > 24) bare = bare.substr(0, 24);
        std::printf("%-26s %7d %8.0f %7d %8.0f\n", bare.c_str(), ra.hits, ra.best, rb.hits,
                    rb.best);
        if (ra.best >= P.minComposite) ++aNg;
        if (rb.best >= P.minComposite) ++bNg;
    }
    std::printf("\n合计 %d 张：A 判NG %d 张，B 判NG %d 张\n", total, aNg, bNg);
    return 0;
}
