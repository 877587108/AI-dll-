// 单图快速迭代回路
//
// 目的：改判定逻辑时，不需要每次都跑 84 张批量。
// 这个工具对单张图跑 findScratches，并逐个阶段打印剩余量，
// 用来回答"到底是哪一步把候选放大成了误报"。
//
// 输出每个阶段的数字，从原始信号到最终命中：
//   1. 信号亮像素（weak 之前）
//   2. 扣模型后剩余（leak）
//   3. 闭运算后
//   4. Hough 候选
//   5. 过长度门槛
//   6. 过覆盖率门槛
//   7. 过严重度门槛（最终命中）
//
// 用法: markloc/quick <modelDir> <image>
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
        std::printf("用法: markloc_quick <modelDir> <image>\n");
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
    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // 全局配准
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
        std::printf("全局配准: (%.2f, %.2f)\n", sh.x, sh.y);
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat al;
            cv::warpAffine(sig, al, M, sig.size(),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
            sig = al;
        }
    }

    markloc::ScratchParams P;
    const cv::Rect band = markloc::scratchBand(sig.size(), units, P.growPx);
    std::printf("搜索带: (%d,%d,%d,%d)\n", band.x, band.y, band.width, band.height);
    const cv::Mat sBand = sig(band);

    // 逐阶段复刻 findScratches
    cv::Mat weak;
    cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    std::printf("\n%-24s %8d\n", "① 信号亮像素", cv::countNonZero(weak));

    cv::Mat modelInk;
    cv::compare(model.freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);
    const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
    cv::Mat modelDil, notModel;
    cv::dilate(modelInk, modelDil, kd);
    cv::bitwise_not(modelDil, notModel);
    cv::bitwise_and(weak, notModel, weak);
    const int afterSub = cv::countNonZero(weak);
    std::printf("%-24s %8d  (%.1f%% 残留)\n", "② 扣模型笔画后", afterSub,
                100.0 * afterSub / std::max(1, cv::countNonZero(
                    [&] {
                        cv::Mat t;
                        cv::compare(sBand, cfg.missStrokeLo * 0.7, t, cv::CMP_GT);
                        return t;
                    }())));

    const int bk = std::max(3, P.bridge | 1);
    const cv::Mat kb = cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk});
    cv::morphologyEx(weak, weak, cv::MORPH_CLOSE, kb);
    std::printf("%-24s %8d  (放大 %.1f 倍)\n", "③ 闭运算后", cv::countNonZero(weak),
                static_cast<double>(cv::countNonZero(weak)) / std::max(1, afterSub));
    cv::medianBlur(weak, weak, 3);
    std::printf("%-24s %8d\n", "④ 中值滤波后", cv::countNonZero(weak));

    // 连通块分析：这是关键——闭运算造出多大的块
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(weak, labels, stats, centroids, 8);
    std::vector<int> areas;
    for (int i = 1; i < n; ++i) areas.push_back(stats.at<int>(i, cv::CC_STAT_AREA));
    std::sort(areas.begin(), areas.end(), std::greater<int>());
    std::printf("\n连通块: %d 个\n", n - 1);
    if (!areas.empty()) {
        std::printf("  面积前 5: ");
        for (size_t i = 0; i < std::min<size_t>(5, areas.size()); ++i)
            std::printf("%d ", areas[i]);
        std::printf("\n  面积中位: %d\n", areas[areas.size() / 2]);
    }

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(weak, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);
    std::printf("%-24s %8zu\n", "⑤ Hough 候选", lines.size());

    // 过长度门槛
    int passLen = 0;
    std::vector<double> lens;
    for (const auto& ln : lines) {
        const double L = cv::norm(cv::Point(ln[0], ln[1]) - cv::Point(ln[2], ln[3]));
        lens.push_back(L);
        if (L >= P.minLen) ++passLen;
    }
    std::printf("%-24s %8d\n", "⑥ 过 minLen 后", passLen);
    std::sort(lens.begin(), lens.end(), std::greater<double>());
    if (!lens.empty()) {
        std::printf("  长度前 5: ");
        for (size_t i = 0; i < std::min<size_t>(5, lens.size()); ++i)
            std::printf("%.0f ", lens[i]);
        std::printf("\n");
    }

    // 最终
    const std::vector<markloc::ScratchHit> hits =
        markloc::findScratches(sig, model.freq, units, cfg, P);
    std::printf("%-24s %8zu\n", "⑦ 最终命中", hits.size());
    std::fflush(stdout);
    // 逐门槛拦截统计：看清是哪一道在挡
    markloc::ScratchStats st;
    markloc::findScratches(sig, model.freq, units, cfg, P, &st);
    std::printf("   候选=%d 被长度挡=%d 被覆盖挡=%d 被严重度挡=%d 最好得分=%.0f\n", st.candidates,
                st.belowLen, st.belowCov, st.belowComposite, st.bestComposite);
    for (const auto& h : hits) {
        std::printf("   S=%-6.0f 跨%d 深%-5.0f 长%-4.0f 框=(%d,%d %dx%d)\n", h.composite,
                    h.charsHit, h.depth, h.len, h.box.x, h.box.y, h.box.width, h.box.height);
    }
    return 0;
}
