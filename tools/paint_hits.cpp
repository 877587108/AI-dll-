// paint_hits.cpp —— 把每条命中的线画到原图上，按颜色区分
//
// 为什么要做这个：我一直在数值里打转（严重度、覆盖率、漏出比例），
// 但"这是不是划痕"是视觉判断。用户已经确认 0109 没有划痕，
// 那 0109 上画出来的东西就是"漏出长什么样"的样板；
// 拿它去对照 NG 侧的真缺陷，才知道两类在图像上差在哪。
//
// 输出：out/trace/paint_<编号>.png
//   红线 = 命中（按严重度排序，只画前 6 条，避免糊成一团）
//   黄框 = 字符框（对齐后）
//   左上角写文件名与 S 值
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
    if (argc < 4) {
        std::printf("用法: markloc_painthits <modelDir> <outDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型加载失败: %s\n", argv[1]);
        return 1;
    }
    const fs::path outDir(argv[2]);
    fs::create_directories(outDir);
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    for (int ai = 3; ai < argc; ++ai) {
        const std::string p8 = argv[ai];
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

        cv::Mat vis;
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        // 只画字符区，省得整幅图缩得看不清
        cv::Rect roi = markloc::scratchBand(sig.size(), units, sp.growPx);
        roi &= cv::Rect(0, 0, vis.cols, vis.rows);
        if (roi.area() <= 0) continue;
        cv::Mat crop = vis(roi).clone();

        std::sort(hits.begin(), hits.end(),
                  [](const markloc::ScratchHit& a, const markloc::ScratchHit& b) {
                      return a.composite > b.composite;
                  });
        const int show = std::min<int>(6, static_cast<int>(hits.size()));
        for (int i = 0; i < show; ++i) {
            const auto& h = hits[i];
            // 颜色随严重度：前 3 红，其余橙
            const cv::Scalar col = i < 3 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 165, 255);
            cv::line(crop, h.a1 - roi.tl(), h.a2 - roi.tl(), col, 2, cv::LINE_AA);
            char buf[48];
            std::snprintf(buf, sizeof(buf), "#%d S=%.0f", i, h.composite);
            cv::putText(crop, buf, h.a1 - roi.tl() + cv::Point(3, -3), cv::FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv::LINE_AA);
        }
        for (const auto& u : units) {
            cv::Rect b = u.bbox - roi.tl();
            cv::rectangle(crop, b, cv::Scalar(0, 220, 220), 1);
        }

        const std::string base = fs::path(p8).filename().string();
        // 取"编号_视角"做标签，例如 0026_FT / 0109_RT。
        // 只取前 4 位会让 0026_FT 和 0026_RT 撞名，后画的盖掉先画的。
        std::string tag = base.substr(0, std::min<size_t>(4, base.size()));
        {
            const size_t us = base.find("_FT");
            const size_t ur = base.find("_RT");
            const size_t up = (us != std::string::npos) ? us : ur;
            if (up != std::string::npos && up + 3 <= base.size())
                tag = base.substr(0, up + 3);
        }
        const fs::path op = outDir / ("paint_" + tag + ".png");
        cv::imwrite(op.string(), crop);
        std::printf("%-8s 命中 %2d 条，前 %d 条已画 -> %s\n", tag.c_str(),
                    static_cast<int>(hits.size()), show, op.string().c_str());
    }
    return 0;
}
