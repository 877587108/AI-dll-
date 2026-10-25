// sweep_cover.cpp —— 一次扫描，跑遍 minCharCover 的全部取值
//
// 目的：不要再"改一个值 → 重编 → 跑一次"地猜。
// 直接对全库（ok 84 + ng 7）一次性算完，打印每条命中在各取值下的 charsHit，
// 看跨字覆盖门槛到底有没有区分力。
//
// 判读标准（先写死，再看结果）：
//   若同一个 minCharCover 下，清白件的命中能被压到 < 门槛、
//   而真缺陷件仍 >= 门槛，则这个维度有效。
//   若两类同时被压、或同时保住，则无效，立即放弃，不再扫第二遍。
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
        std::printf("用法: markloc_sweepcover <modelDir> <imageDir>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(fs::path(argv[2]))) {
        if (!e.is_regular_file()) continue;
        const std::string stem = e.path().stem().string();
        if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
        files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

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

        const double covers[] = {0.0, 0.10, 0.20, 0.30, 0.40, 0.50};
        std::string line;
        char head[512];
        std::snprintf(head, sizeof(head), "%s |", fs::path(p8).filename().string().c_str());
        line = head;
        for (double cv_ : covers) {
            markloc::ScratchParams sp;
            sp.minCharCover = cv_;
            markloc::ScratchStats st;
            auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp, &st);
            int best = 0;
            for (const auto& h : hits) best = std::max(best, (int)h.composite);
            char buf[64];
            std::snprintf(buf, sizeof(buf), " %6d", best);
            line += buf;
        }
        std::printf("%s\n", line.c_str());
    }
    std::printf("\n列: minCharCover = 0 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50 下的最高严重度\n");
    return 0;
}
