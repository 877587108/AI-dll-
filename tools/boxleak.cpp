// 假跨界诊断
//
// 背景：0109_RT 的头号命中 box=(966,800,104,230) charsHit=4 S=1938，
// 但它是一条从左上到右下的斜线。scratch.hpp:307-312 用线段的包围盒做交集，
// 斜线的包围盒会覆盖一大片区域，把根本没被扫到的字也算进 charsHit。
//
// 本工具复刻 findScratches 的命中筛选，对每条命中同时算出：
//   charsHitBox  = 包围盒与字符框求交（现行实现）
//   charsHitLine = 沿线段逐点取样，落进哪个框就算哪个（真实跨越）
// 并打印假跨界的命中。
//
// 用法: markloc_boxleak <modelDir> <image>
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

static fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

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
        std::printf("用法: markloc_boxleak <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("图片读取失败或尺寸不符\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    auto units = chardet::toAllUnits(model);

    // 复刻推理侧配准
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat r32, s32;
    model.freq.convertTo(r32, CV_32F);
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
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
            sig = aligned;
        }
        std::printf("配准: (%.2f, %.2f)\n", sh.x, sh.y);
    }

    markloc::ScratchParams P;
    const std::vector<markloc::ScratchHit> hits =
        markloc::findScratches(sig, model.freq, units, cfg, P);
    std::printf("findScratches 命中 %zu 条（现实现，包围盒判跨界）\n\n", hits.size());

    std::printf("%-4s %-22s %8s %8s %6s %6s %10s %10s\n", "#", "box", "len", "depth", "box跨",
                "线跨", "S(box)", "S(line)");
    int faked = 0;
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        const cv::Point a1(h.a1), a2(h.a2);

        // 现行实现：包围盒求交
        const cv::Rect seg(std::min(a1.x, a2.x), std::min(a1.y, a2.y),
                           std::abs(a2.x - a1.x) + 1, std::abs(a2.y - a1.y) + 1);
        int byBox = 0;
        for (const auto& u : units)
            if ((u.bbox & seg).area() > 0) ++byBox;

        // 真实跨越：沿线段逐点取样
        std::vector<int> touched;
        const double len = cv::norm(a2 - a1);
        const int steps = std::max(2, static_cast<int>(len));
        for (int t = 0; t <= steps; ++t) {
            const double rr = static_cast<double>(t) / steps;
            const cv::Point q(static_cast<int>(a1.x + (a2.x - a1.x) * rr),
                              static_cast<int>(a1.y + (a2.y - a1.y) * rr));
            for (const auto& u : units) {
                if (u.bbox.contains(q) &&
                    std::find(touched.begin(), touched.end(), u.index) == touched.end()) {
                    touched.push_back(u.index);
                }
            }
        }
        const int byLine = static_cast<int>(touched.size());
        const double sBox = h.depth * byBox;
        const double sLine = h.depth * byLine;
        std::printf("%-4zu (%4d,%4d %3dx%3d) %8.0f %8.0f %6d %6d %10.0f %10.0f", i, h.box.x,
                    h.box.y, h.box.width, h.box.height, len, h.depth, byBox, byLine, sBox, sLine);

        if (byBox != byLine) {
            ++faked;
            std::printf("  <== 跨界不符");
            if (byBox >= 2 && byLine <= 1) std::printf("  ★假跨界");
        }
        std::printf("\n");
    }

    std::printf("\n共 %zu 条，其中 %d 条 box跨 != 线跨\n", hits.size(), faked);
    int falseAlarm = 0;
    for (const auto& h : hits) {
        const cv::Point a1(h.a1), a2(h.a2);
        const double len = cv::norm(a2 - a1);
        const int steps = std::max(2, static_cast<int>(len));
        int n = 0;
        std::vector<int> touched;
        for (int t = 0; t <= steps; ++t) {
            const double rr = static_cast<double>(t) / steps;
            const cv::Point q(static_cast<int>(a1.x + (a2.x - a1.x) * rr),
                              static_cast<int>(a1.y + (a2.y - a1.y) * rr));
            for (const auto& u : units)
                if (u.bbox.contains(q) &&
                    std::find(touched.begin(), touched.end(), u.index) == touched.end())
                    touched.push_back(u.index);
        }
        n = static_cast<int>(touched.size());
        if (h.charsHit >= 2 && n <= 1) ++falseAlarm;
    }
    std::printf("其中「包围盒跨>=2 但真实线只落进<=1 个字」的：%d 条\n", falseAlarm);
    return 0;
}
