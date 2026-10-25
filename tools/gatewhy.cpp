// gatewhy.cpp —— 每条命中到底卡在哪道门槛
//
// 上一轮的坑：我以为 S=1938 那条线贴字形，加距离门槛就该没。
// 实测只从 1938 降到 1927——说明它根本没被拦。
// 光看最终 S 看不出原因，必须逐条打印"实测值 vs 门槛"。
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
        std::printf("用法: markloc_gatewhy <modelDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    for (int ai = 2; ai < argc; ++ai) {
        const std::string p8 = argv[ai];
        cv::Mat gray = readTif(p8);
        if (gray.empty() || gray.size() != model.freq.size()) continue;
        std::printf("\n===== %s =====\n", fs::path(p8).filename().string().c_str());

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

        // 复刻 findScratches 内部状态，逐条候选量距离
        markloc::ScratchParams P;
        const cv::Rect band = markloc::scratchBand(sig.size(), units, P.growPx);
        const cv::Mat sBand = sig(band);
        cv::Mat modelInk0;
        cv::compare(model.freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk0, cv::CMP_GT);
        cv::Mat notInk, inkDist;
        cv::bitwise_not(modelInk0, notInk);
        cv::distanceTransform(notInk, inkDist, cv::DIST_L2, 3);

        cv::Mat weak;
        cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
        weak.convertTo(weak, CV_8U, 255.0);
        {
            const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
            cv::Mat modelDil, notModel;
            cv::dilate(modelInk0, modelDil, kd);
            cv::bitwise_not(modelDil, notModel);
            cv::bitwise_and(weak, notModel, weak);
        }
        if (P.bridge > 0) {
            const int bk = std::max(3, P.bridge | 1);
            const cv::Mat kb = cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk});
            cv::morphologyEx(weak, weak, cv::MORPH_CLOSE, kb);
        }
        cv::medianBlur(weak, weak, 3);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(weak, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);

        struct Row {
            double dP50, len, cov;
            int chars;
            double comp;
        };
        std::vector<Row> rows;
        for (const auto& ln : lines) {
            const cv::Point p1(ln[0], ln[1]), p2(ln[2], ln[3]);
            const double len = cv::norm(p2 - p1);
            if (len < P.minLen) continue;
            const int steps = std::max(2, static_cast<int>(len));
            std::vector<float> dsv;
            int covered = 0;
            double energy = 0;
            for (int t = 0; t <= steps; ++t) {
                const double rr = static_cast<double>(t) / steps;
                const int x = static_cast<int>(p1.x + (p2.x - p1.x) * rr);
                const int y = static_cast<int>(p1.y + (p2.y - p1.y) * rr);
                if (y < 0 || y >= weak.rows || x < 0 || x >= weak.cols) continue;
                bool hit = false;
                for (int dy = -4; dy <= 4; ++dy) {
                    const int yy = std::clamp(y + dy, 0, weak.rows - 1);
                    if (weak.at<uchar>(yy, x)) {
                        energy += std::max(static_cast<double>(sBand.at<float>(yy, x)) -
                                               cfg.missStrokeLo, 0.0);
                        if (std::abs(dy) <= 1) hit = true;
                    }
                }
                if (hit) ++covered;
                if (weak.at<uchar>(y, x)) dsv.push_back(inkDist.at<float>(y, x));
            }
            const double cov = static_cast<double>(covered) / (steps + 1);
            if (cov < P.minCov) continue;
            if (dsv.empty()) continue;
            std::sort(dsv.begin(), dsv.end());
            const double dP50 = dsv[dsv.size() / 2];
            // 只报最严重的那些：粗算 composite = 深度近似用 energy/len，这里只看相对
            const cv::Point a1 = p1 + cv::Point(band.x, band.y);
            const cv::Point a2 = p2 + cv::Point(band.x, band.y);
            const cv::Rect seg(std::min(a1.x, a2.x), std::min(a1.y, a2.y),
                               std::abs(a2.x - a1.x) + 1, std::abs(a2.y - a1.y) + 1);
            int ch = 0;
            for (const auto& u : units)
                if ((u.bbox & seg).area() > 0) ++ch;
            rows.push_back({dP50, len, cov, ch, energy});
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row& a, const Row& b) { return a.comp > b.comp; });
        std::printf("%5s %7s %6s %6s %5s\n", "len", "inkP50", "cov", "跨字", "能量");
        for (size_t i = 0; i < rows.size() && i < 14; ++i)
            std::printf("%5.0f %7.1f %6.2f %6d %5.0f\n", rows[i].len, rows[i].dP50, rows[i].cov,
                        rows[i].chars, rows[i].comp);
    }
    return 0;
}
