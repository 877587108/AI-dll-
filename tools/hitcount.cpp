// hitcount.cpp —— 命中条数能不能区分清白件与真缺陷件
//
// 假设 H10（本轮唯一的假设）：
//   清白件的漏出是零碎多发的，真划痕是孤立的。
//   若成立，则"通过命中的条数"在两类之间有量级差。
//
// 预测（写在这里，跑完对照）：
//   - 0109_RT 命中数 >> NG 侧真缺陷件的命中数
//   - NG 侧那 4 张（0026_FT / 0026_RT / 0083_FT / 0083_RT）的命中数应当在个位数
//
// 若预测不成立 -> H10 作废，不得据此改代码。
#include <algorithm>
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
        std::printf("用法: markloc_hitcount <modelDir> <imageDir> [--top N]\n");
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

    struct Rec {
        std::string name;
        int candidates = 0, passed = 0;
        int bestS = 0;
    };
    std::vector<Rec> recs;

    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--top") { ++i; continue; }
        const fs::path p(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
        if (fs::is_directory(p)) {
            for (const auto& e : fs::directory_iterator(p)) {
                if (!e.is_regular_file()) continue;
                const std::string stem = e.path().stem().string();
                // 只取 _00 视角（模型只按该视角训练）
                if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
                files.push_back(e.path().string());
            }
        } else {
            files.push_back(s);
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& p8 : files) {
        cv::Mat gray = readTif(p8);
        if (gray.empty() || gray.size() != model.freq.size()) continue;

        cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg,
                                            false);
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
        const auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp, &st);
        Rec r;
        r.name = fs::path(p8).filename().string();
        r.candidates = st.candidates;
        r.passed = static_cast<int>(hits.size());
        for (const auto& h : hits)
            r.bestS = std::max(r.bestS, static_cast<int>(h.composite));
        recs.push_back(std::move(r));
    }

    std::sort(recs.begin(), recs.end(),
              [](const Rec& a, const Rec& b) { return a.passed > b.passed; });
    std::printf("%-4s %6s %6s %6s  %s\n", "序", "候选", "通过", "最高S", "文件");
    int n = std::min(static_cast<int>(recs.size()), top);
    for (int i = 0; i < n; ++i)
        std::printf("%-4d %6d %6d %6d  %s\n", i, recs[i].candidates, recs[i].passed, recs[i].bestS,
                    recs[i].name.c_str());
    return 0;
}
