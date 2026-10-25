// 反查：那条线是靠哪些像素活下来的？
//
// 已经知道的事实：
//   · S=1938 的线段 (970,808)-(1065,1021) 在原图上跨过的是"8N"/"Z1"的字形缝，
//     那里干净（out/trace/zoom_0109_hit.png）。
//   · 但 Hough 只能在 med 图（扣模型+闭运算+中值）的非零像素上出线。
//     所以这条线的每一个采样点，在 med 里都必须是亮的。
//   · 那么"亮"是谁贡献的？三种可能：
//       (a) 字形的真实边缘（signal 有、freq 低 → 扣不掉）
//       (b) 闭运算把远处的东西桥接过来（CLOSE 25 的圆核半径 12px）
//       (c) 闭运算后中值滤波保留下来的噪点
//
// 本工具沿线逐采样点回查四张图（signal / weak / leak / med）的值，
// 并标注该点最近的 modelInk 距离，直接看出"亮"的来源属于哪一类。
// 这是把"猜"换成"查"的一步。
//
// 用法: markloc/trace <modelDir> <image> <x1> <y1> <x2> <y2>
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
    if (argc < 7) {
        std::printf("用法: markloc_trace <modelDir> <image> <x1> <y1> <x2> <y2>\n");
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
    const cv::Point p1(std::atoi(argv[3]), std::atoi(argv[4]));
    const cv::Point p2(std::atoi(argv[5]), std::atoi(argv[6]));

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
    // 逐字符平移框（与 detectOne 一致，不动像素）
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

    const double weakThr = cfg.missStrokeLo * 0.7;
    const int inkThr = static_cast<int>(cfg.blankFreq * 255);
    cv::Mat weak, modelInk, d9, nm9, leak, dt;
    cv::compare(sig, weakThr, weak, cv::CMP_GT);
    cv::compare(model.freq, inkThr, modelInk, cv::CMP_GT);
    cv::dilate(modelInk, d9, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(d9, nm9);
    cv::bitwise_and(weak, nm9, leak);
    {
        cv::Mat inv;
        cv::bitwise_not(modelInk, inv);
        cv::distanceTransform(inv, dt, cv::DIST_L2, 3);
    }
    markloc::ScratchParams P;
    const int bk = std::max(3, P.bridge | 1);
    cv::Mat closed, med;
    cv::morphologyEx(leak, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk}));
    cv::medianBlur(closed, med, 3);

    const double len = cv::norm(p2 - p1);
    const int steps = std::max(2, static_cast<int>(len));
    std::printf("线段 (%d,%d)-(%d,%d) 长 %.0f   weak阈值=%.3f 模型墨阈值=%d  核=%d\n", p1.x,
                p1.y, p2.x, p2.y, len, weakThr, inkThr, bk);
    std::printf("\n%6s %-14s %8s %8s %6s %6s %7s %s\n", "t", "坐标", "signal", "freq", "weak",
                "leak", "med", "离模型墨");
    int nW = 0, nL = 0, nM = 0, nOnlyClosed = 0;
    for (int t = 0; t <= steps; ++t) {
        const double rr = static_cast<double>(t) / steps;
        const int x = static_cast<int>(p1.x + (p2.x - p1.x) * rr);
        const int y = static_cast<int>(p1.y + (p2.y - p1.y) * rr);
        if (x < 0 || y < 0 || x >= sig.cols || y >= sig.rows) continue;
        const float sv = sig.at<float>(y, x);
        const int fv = model.freq.at<uchar>(y, x);
        const bool w = weak.at<uchar>(y, x) != 0;
        const bool l = leak.at<uchar>(y, x) != 0;
        const bool m = med.at<uchar>(y, x) != 0;
        if (w) ++nW;
        if (l) ++nL;
        if (m) ++nM;
        if (m && !l) ++nOnlyClosed;
        if (t % 8 == 0 || (m && !l)) {
            std::printf("%6d %-14s %8.3f %8d %6s %6s %7s %7.1f\n", t,
                        cv::format("(%d,%d)", x, y).c_str(), sv, fv, w ? "Y" : ".", l ? "Y" : ".",
                        m ? "Y" : ".", dt.at<float>(y, x));
        }
    }
    const int n = steps + 1;
    std::printf("\n沿线 %d 个采样点：weak %d (%.0f%%)  leak %d (%.0f%%)  med %d (%.0f%%)\n", n, nW,
                100.0 * nW / n, nL, 100.0 * nL / n, nM, 100.0 * nM / n);
    std::printf("med 亮但 leak 不亮的点 %d 个 —— 这些是闭运算/中值凭空造出来的\n",
                nOnlyClosed);
    return 0;
}
