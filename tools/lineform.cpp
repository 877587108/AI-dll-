// lineform.cpp —— 量"这条线是连成的，还是本来就有的"
//
// L4 换约束：跨越字数不能动（一动真缺陷先死），距离门槛已经到边界。
// 剩下的唯一角度是"线本身的物质性"：
//   假线 = 由零碎漏出经 CLOSE(25) 桥接而成，本身不连续，亮度参差
//   真划痕 = 连续的物质，沿线亮度平稳
//
// 对每条命中沿轴线取样，量四个形态量：
//   gapMax   沿线最长的"暗断口"长度（px）：连成的假线断口多而长
//   onInkFrac 线段上落在模型墨迹上的比例
//   stdRel   沿线亮度的相对标准差：连成的线亮度参差
//   zigzag   方向变化次数：桥接出来的线会拐
//
// 判读：若清白件与真缺陷件在这四个量上不重叠，则有效；
// 若重叠，立即作废，不再扫第二遍。
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
        std::printf("用法: markloc_lineform <modelDir> <imageDir>\n");
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

    std::printf("%-46s %6s %6s %7s %7s %6s\n", "文件(最严重那条)", "S", "断口", "墨上比例",
                "亮度std", "拐点");
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

        const cv::Mat modelInk =
            (model.freq > static_cast<int>(cfg.blankFreq * 255));
        const int n = std::max(2, static_cast<int>(h.len));
        // 沿线取样：亮度序列 + 是否落墨
        std::vector<double> vals;
        int onInk = 0, total = 0, curGap = 0, maxGap = 0, zig = 0;
        double prevHalf = -999, prevSlope = 0;
        for (int t = 0; t <= n; ++t) {
            const double rr = static_cast<double>(t) / n;
            const int x = static_cast<int>(std::lround(h.a1.x + (h.a2.x - h.a1.x) * rr));
            const int y = static_cast<int>(std::lround(h.a1.y + (h.a2.y - h.a1.y) * rr));
            if (x < 0 || y < 0 || x >= sig.cols || y >= sig.rows) continue;
            // 轴线垂直方向 ±3 取最大，避开采样落在线外的抖动
            const double ux = (h.a2.x - h.a1.x) / std::max(1e-6, h.len);
            const double uy = (h.a2.y - h.a1.y) / std::max(1e-6, h.len);
            double mx = 0;
            for (int k = -3; k <= 3; ++k) {
                const int xi = static_cast<int>(std::lround(x - uy * k));
                const int yi = static_cast<int>(std::lround(y + ux * k));
                if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
                mx = std::max(mx, (double)sig.at<float>(yi, xi));
            }
            vals.push_back(mx);
            ++total;
            const bool ink = modelInk.at<uchar>(y, x) != 0;
            if (ink) {
                ++onInk;
                curGap = 0;
            } else {
                ++curGap;
                maxGap = std::max(maxGap, curGap);
            }
            // 方向变化：亮度序列的符号翻转
            const double slope = mx - prevHalf;
            if (prevHalf > -998 && std::abs(slope) > 0.15 && prevSlope != 0 &&
                ((slope > 0) != (prevSlope > 0)))
                ++zig;
            if (std::abs(slope) > 0.15) prevSlope = slope;
            prevHalf = mx;
        }
        double mean = 0, sd = 0;
        for (double v : vals) mean += v;
        if (!vals.empty()) mean /= vals.size();
        for (double v : vals) sd += (v - mean) * (v - mean);
        if (vals.size() > 1) sd = std::sqrt(sd / (vals.size() - 1));
        const double stdRel = mean > 1e-6 ? sd / mean : 0.0;

        char name[64];
        std::snprintf(name, sizeof(name), "%.40s", fs::path(p8).filename().string().c_str());
        std::printf("%-46s %6d %6d %7.2f %7.3f %6d\n", name, (int)h.composite, maxGap,
                    total ? (double)onInk / total : 0.0, stdRel, zig);
    }
    return 0;
}
