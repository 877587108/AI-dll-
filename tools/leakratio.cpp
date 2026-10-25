// 判决用指标：leak 的"相对量"而不是绝对量
//
// 上面 maskeval2 显示，降低掩膜阈值对所有图都是一起降——这不是判别证据。
// 判别要看的是：同一张图内部，leak 相对于该图自身亮像素的比例，
// 以及 **leak 里有多少落在 freq=0 的"真实笔画"上**。
//
// 关键量化（对应已证的因果）：
//   漏报笔画像素 = 满足 signal > weakThr 且 freq ∈ (0, 阈值) 的像素
//   这些像素是"信号说这里有东西、模型说这里从来没有"。
//   对真实字形而言，这个量应当很小；对 0109 这类暗照明件，它很大。
//
// 本工具输出每张图：
//   leak 密度 = leak / weak
//   笔画漏出密度 = (signal>weakThr & freq<=thr) / (signal>weakThr)
//   并用后者作为"该图照明是否偏暗"的客观标尺
//
// 用法: markloc/leakratio <modelDir> <image|dir>...
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
        std::printf("用法: markloc_leakratio <modelDir> <image|dir>...\n");
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

    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string s = argv[i];
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

    const int inkThr = static_cast<int>(cfg.blankFreq * 255);
    std::printf("%-22s %8s %8s %8s %8s %8s\n", "文件", "亮像素", "leak", "密度", "笔画漏出",
                "最大S");
    for (const auto& path : files) {
        cv::Mat gray;
        if (!readGray(path, gray) || gray.size() != model.freq.size()) continue;
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

        const cv::Rect band = markloc::scratchBand(sig.size(), units, P.growPx);
        const cv::Mat sBand = sig(band), fBand = model.freq(band);
        cv::Mat weak, modelInk;
        cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
        cv::compare(fBand, inkThr, modelInk, cv::CMP_GT);
        const int nWeak = cv::countNonZero(weak);
        cv::Mat d, nm, leakM;
        cv::dilate(modelInk, d, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
        cv::bitwise_not(d, nm);
        cv::bitwise_and(weak, nm, leakM);
        const int nLeak = cv::countNonZero(leakM);

        // 笔画漏出：亮、但模型里 freq 低于阈值
        cv::Mat bright, lowFreq;
        cv::compare(sBand, cfg.missStrokeLo * 0.7, bright, cv::CMP_GT);
        cv::compare(fBand, inkThr, lowFreq, cv::CMP_LE);
        cv::Mat strokeLeak;
        cv::bitwise_and(bright, lowFreq, strokeLeak);
        const int nSL = cv::countNonZero(strokeLeak);

        markloc::ScratchStats ss;
        const auto hits = markloc::findScratches(sig, model.freq, units, cfg, P, &ss);

        std::string bare = fs::path(path).stem().string();
        if (bare.size() > 20) bare = bare.substr(0, 20);
        std::printf("%-22s %8d %8d %7.1f%% %7.1f%% %8.0f\n", bare.c_str(), nWeak, nLeak,
                    100.0 * nLeak / std::max(1, nWeak), 100.0 * nSL / std::max(1, nWeak),
                    ss.bestComposite);
    }
    return 0;
}
