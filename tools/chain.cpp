// 逐步骤把 scratch.hpp 的中间量画出来，定位"干净的缝"为什么变成了一条线
//
// 已经确认的事实（见 out/trace/zoom_0109_hit.png）：
//   S=1938 那条线 (970,808)-(1065,1021) 落在 "8N" 与 "Z1" 之间的字形缝里，
//   原图上那里干净，没有任何划痕。所以它是判定链路的产物，不是被测物。
//
// scratch.hpp 的步骤：
//   ① weak = signal > 0.105
//   ② weak &= ~dilate(freq>38, 9x9)
//   ③ CLOSE(bridge=25)          ← 关键嫌疑：25px 圆核闭运算
//   ④ medianBlur(3)
//   ⑤ HoughLinesP
// 输出每一阶段在**同一裁剪窗**内的图，人眼直接看那条线在第几步成形。
//
// 用法: markloc/chain <modelDir> <image> <x> <y> <w> <h> <scale> <out.png>
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

int main(int argc, char** argv) {
    if (argc < 9) {
        std::printf("用法: markloc_chain <modelDir> <image> <x> <y> <w> <h> <scale> <out.png>\n");
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
        std::printf("读取失败\n");
        return 1;
    }
    const cv::Rect win(std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                       std::atoi(argv[6]));
    const int sc = std::max(1, std::atoi(argv[7]));

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
        cv::matchTemplate(
            tS, mS(cv::Rect(box.x - search.x, box.y - search.y, box.width, box.height)), res,
            cv::TM_CCOEFF_NORMED);
        cv::Point mLoc;
        double mVal = 0;
        cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
        if (mVal < 0.30) continue;
        const cv::Rect moved(box.x + mLoc.x - kSearch, box.y + mLoc.y - kSearch, box.width,
                             box.height);
        if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
        u.bbox = moved;
    }

    // 复刻 scratch.hpp 的每一步（全图，不分带，便于裁剪）
    const double weakThr = cfg.missStrokeLo * 0.7;
    const int inkThr = static_cast<int>(cfg.blankFreq * 255);

    cv::Mat weak;
    cv::compare(sig, weakThr, weak, cv::CMP_GT);
    cv::Mat modelInk;
    cv::compare(model.freq, inkThr, modelInk, cv::CMP_GT);
    cv::Mat d9, nm9, leak;
    cv::dilate(modelInk, d9, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(d9, nm9);
    cv::bitwise_and(weak, nm9, leak);

    markloc::ScratchParams P;
    const int bk = std::max(3, P.bridge | 1);
    cv::Mat closed, med;
    cv::morphologyEx(leak, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk}));
    cv::medianBlur(closed, med, 3);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(med, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);

    // 六联：原图 / 信号 / weak / leak / closed / Hough
    const auto prep = [&](const cv::Mat& m) {
        cv::Mat c;
        if (m.type() == CV_8UC1) c = m;
        else m.convertTo(c, CV_8U);
        cv::Mat crop = c(win).clone(), big;
        cv::resize(crop, big, cv::Size(), sc, sc, cv::INTER_NEAREST);
        cv::Mat bgr;
        cv::cvtColor(big, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    };
    cv::Mat rawCrop = gray(win).clone(), rawBig;
    cv::resize(rawCrop, rawBig, cv::Size(), sc, sc, cv::INTER_NEAREST);
    cv::Mat rawBgr;
    cv::cvtColor(rawBig, rawBgr, cv::COLOR_GRAY2BGR);

    cv::Mat sig8;
    sig.convertTo(sig8, CV_8U, 255.0 / 1.6);

    const auto label = [&](cv::Mat& img, const char* t) {
        cv::putText(img, t, cv::Point(6, 26), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255),
                    2);
    };
    std::vector<cv::Mat> panes;
    panes.push_back(rawBgr);
    label(panes.back(), "1 raw");
    panes.push_back(prep(sig8));
    label(panes.back(), "2 signal");
    panes.push_back(prep(weak));
    label(panes.back(), "3 weak>0.105");
    panes.push_back(prep(leak));
    label(panes.back(), "4 -model");
    panes.push_back(prep(closed));
    char cb[64];
    std::snprintf(cb, sizeof(cb), "5 CLOSE %d", bk);
    label(panes.back(), cb);
    cv::Mat hou = prep(med);
    for (const auto& ln : lines) {
        cv::line(hou,
                 cv::Point((ln[0] - win.x) * sc, (ln[1] - win.y) * sc),
                 cv::Point((ln[2] - win.x) * sc, (ln[3] - win.y) * sc), cv::Scalar(0, 255, 0), 1);
    }
    char hb[64];
    std::snprintf(hb, sizeof(hb), "6 Hough %zu", lines.size());
    label(hou, hb);
    panes.push_back(hou);

    cv::Mat out;
    cv::hconcat(panes, out);
    fs::path p(argv[8]);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    cv::imwrite(argv[8], out);
    std::printf("-> %s\n", argv[8]);

    // 在窗内统计每步面积，量化"放大倍数"
    std::printf("\n窗内各步非零像素（整图坐标 %d,%d %dx%d）：\n", win.x, win.y, win.width,
                win.height);
    std::printf("  weak   %6d\n", cv::countNonZero(weak(win)));
    std::printf("  leak   %6d\n", cv::countNonZero(leak(win)));
    std::printf("  closed %6d  (放大 %.2f 倍)\n", cv::countNonZero(closed(win)),
                static_cast<double>(cv::countNonZero(closed(win))) /
                    std::max(1, cv::countNonZero(leak(win))));
    std::printf("  med    %6d\n", cv::countNonZero(med(win)));
    return 0;
}
