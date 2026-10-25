// inkdist.cpp —— inkDistP50 在全库上稳不稳定
//
// 上一轮 split0109 在 7 张图上发现：清白件的命中线紧贴字形
// （inkDistP50 4.6~5.2），真划痕明显更远（8.2~16.4）。
// 7 张图上的分界不能信——必须先在 84 张清洁件上看它是否稳定，
// 否则就是在拟合那 7 个样本。
//
// 输出每张图"最严重那条命中"的 inkDistP50，按大小排序。
// 若清洁件里有相当数量超过真缺陷的最小值（8.2），该指标作废。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

namespace fs = std::filesystem;

static cv::Mat readTif(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::vector<uchar> buf;
    uchar tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(f);
    if (buf.empty()) return {};
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_inkdist <modelDir> <imageDir> [--top N]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型加载失败: %s\n", argv[1]);
        return 1;
    }
    int top = 200;
    for (int i = 3; i + 1 < argc; i += 2)
        if (std::strcmp(argv[i], "--top") == 0) top = std::atoi(argv[i + 1]);

    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);
    const int inkThr = static_cast<int>(cfg.blankFreq * 255.0);
    const float kWeak = static_cast<float>(cfg.missStrokeLo * 0.7);

    cv::Mat inkMask = (model.freq > inkThr);
    cv::Mat inv, dist;
    cv::bitwise_not(inkMask, inv);
    cv::distanceTransform(inv, dist, cv::DIST_L2, 3);

    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--top") { ++i; continue; }
        const fs::path p(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
        if (fs::is_directory(p)) {
            for (const auto& e : fs::directory_iterator(p)) {
                if (!e.is_regular_file()) continue;
                const std::string stem = e.path().stem().string();
                if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
                files.push_back(e.path().string());
            }
        } else {
            files.push_back(s);
        }
    }
    std::sort(files.begin(), files.end());

    struct Rec {
        std::string name;
        double p50 = 0;
        int outsideRuns = 0;
        int chars = 0;
        double inFrac = 0;
        int passed = 0;
    };
    std::vector<Rec> recs;

    for (const auto& p8 : files) {
        cv::Mat gray = readTif(p8);
        if (gray.empty() || gray.size() != model.freq.size()) continue;

        cv::Mat sig =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        std::vector<chardet::CharUnit> units = baseUnits;
        if (!model.ref.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : units) {
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
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat aligned;
                    cv::warpAffine(sig, aligned, M, sig.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE);
                    sig = aligned;
                }
            }
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

        markloc::ScratchParams sp;
        markloc::ScratchStats st;
        auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp, &st);
        if (hits.empty()) continue;
        std::sort(hits.begin(), hits.end(),
                  [](const markloc::ScratchHit& a, const markloc::ScratchHit& b) {
                      return a.composite > b.composite;
                  });
        const auto& h = hits.front();

        const int n = std::max(2, static_cast<int>(h.len));
        std::vector<float> ds;
        int inBox = 0, counted = 0, outsideRuns = 0;
        bool prevOut = false;
        for (int t = 0; t <= n; ++t) {
            const double u = static_cast<double>(t) / n;
            const cv::Point p(static_cast<int>(std::lround(h.a1.x + (h.a2.x - h.a1.x) * u)),
                              static_cast<int>(std::lround(h.a1.y + (h.a2.y - h.a1.y) * u)));
            if (p.x < 0 || p.y < 0 || p.x >= sig.cols || p.y >= sig.rows) continue;
            ++counted;
            bool inside = false;
            for (const auto& cu : units)
                if (cu.bbox.contains(p)) { inside = true; break; }
            if (inside) { ++inBox; prevOut = false; }
            else { if (!prevOut && t > 0) ++outsideRuns; prevOut = true; }
            if (sig.at<float>(p) > kWeak) ds.push_back(dist.at<float>(p));
        }
        float p50 = -1.f;
        if (!ds.empty()) {
            std::sort(ds.begin(), ds.end());
            p50 = ds[ds.size() / 2];
        }
        Rec r;
        r.name = fs::path(p8).filename().string();
        r.p50 = p50;
        r.outsideRuns = outsideRuns;
        r.chars = h.charsHit;
        r.inFrac = counted ? static_cast<double>(inBox) / counted : 0.0;
        r.passed = static_cast<int>(hits.size());
        recs.push_back(std::move(r));
    }

    std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) { return a.p50 > b.p50; });
    std::printf("%-4s %8s %6s %5s %6s %5s  %s\n", "序", "inkDist", "框外段", "跨字", "框内", "通过",
                "文件");
    const int n = std::min(static_cast<int>(recs.size()), top);
    for (int i = 0; i < n; ++i)
        std::printf("%-4d %8.1f %6d %5d %6.2f %5d  %s\n", i, recs[i].p50, recs[i].outsideRuns,
                    recs[i].chars, recs[i].inFrac, recs[i].passed, recs[i].name.c_str());
    return 0;
}
