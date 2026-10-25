// aligndir.cpp —— 线的走向与字形笔画走向是否一致
//
// 深层换框：前面所有判据都在问"这条线像不像划痕"，都失败。
// 看 0026_FT 的标注图发现关键差异：
//   真划痕（0026 S=2224）是一条水平线，横穿 Silan 的字高，
//     它的走向跟字形毫无关系——它就是一条外来物。
//   0109_RT 那五条是顺着字形走向的斜线，走的是 8NLP 的下沿和字缝。
//
// 判据：对线上每个亮像素，取该处模型墨迹的局部梯度方向（=字形笔画法向），
// 再取线本身的方向。两者夹角接近于 0（平行）说明线在"沿着笔画走"，
// 即这条线是字形的一部分；夹角接近 90（垂直）说明线横穿字形，是外来物。
//
// 输出 cosPar = |cos(线方向, 笔画方向)| 的中位数：
//   高 -> 线顺着笔画走（疑似字形漏出）
//   低 -> 线横穿笔画（疑似真划痕）
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
        std::printf("用法: markloc_aligndir <modelDir> <imageDir>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    // 模型墨迹的模糊图，用于取平滑梯度
    cv::Mat inkF;
    model.freq.convertTo(inkF, CV_32F, 1.0 / 255.0);
    cv::GaussianBlur(inkF, inkF, cv::Size(0, 0), 2.0);
    cv::Mat gx, gy;
    cv::Sobel(inkF, gx, CV_32F, 1, 0, 3);
    cv::Sobel(inkF, gy, CV_32F, 0, 1, 3);

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(fs::path(argv[2]))) {
        if (!e.is_regular_file()) continue;
        const std::string stem = e.path().stem().string();
        if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
        files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    std::printf("%-44s %6s %8s %8s %8s\n", "文件(最严重那条)", "S", "平行中位", "垂直占比", "样本数");
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

        const double dx = h.a2.x - h.a1.x, dy = h.a2.y - h.a1.y;
        const double dl = std::max(1e-6, std::hypot(dx, dy));
        const double ux = dx / dl, uy = dy / dl;

        std::vector<double> cosv;
        int vert = 0, n = 0;
        const int steps = std::max(2, (int)h.len);
        for (int t = 0; t <= steps; ++t) {
            const double rr = static_cast<double>(t) / steps;
            const int x = static_cast<int>(std::lround(h.a1.x + dx * rr));
            const int y = static_cast<int>(std::lround(h.a1.y + dy * rr));
            if (x < 1 || y < 1 || x >= inkF.cols - 1 || y >= inkF.rows - 1) continue;
            // 笔画方向 = 梯度方向的垂直；梯度要够强才算"这里有笔画"
            const double gxv = gx.at<float>(y, x), gyv = gy.at<float>(y, x);
            const double gm = std::hypot(gxv, gyv);
            if (gm < 0.02) continue;
            // 笔画切向
            const double tx = -gyv / gm, ty = gxv / gm;
            const double c = std::abs(ux * tx + uy * ty);
            cosv.push_back(c);
            if (c < 0.35) ++vert;  // 线横穿笔画
            ++n;
        }
        double p50 = -1;
        if (!cosv.empty()) {
            std::nth_element(cosv.begin(), cosv.begin() + cosv.size() / 2, cosv.end());
            p50 = cosv[cosv.size() / 2];
        }
        char name[64];
        std::snprintf(name, sizeof(name), "%.40s", fs::path(p8).filename().string().c_str());
        std::printf("%-44s %6d %8.3f %8.2f %8d\n", name, (int)h.composite, p50,
                    n ? (double)vert / n : 0.0, n);
    }
    return 0;
}
