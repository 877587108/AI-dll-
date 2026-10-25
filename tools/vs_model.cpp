// vsmodel.cpp —— 出与模板的对比图
//
// 要求：一张图里同时看到 模板 / 待检图 / 命中的线 / 假线的原料，
// 而不是把红线画在原图上让人自己猜。
//
// 四个面板（横向拼接）：
//   A 模板 freq        模型认为字形在哪、有多亮
//   B 待检信号 sig     这张图实际亮在哪（已做全局对齐）
//   C 差异 sig-freq    正数是"图里亮但模型没有"，负数是"模型有但图里暗"
//   D 原图+命中线      A/B/C 都叠不了位置感，最后落到真实照片上
//
// 三个面板的坐标系完全一致（都是模型尺寸 1688x1460），
// 所以同一个像素在 A/B/C 里指的是同一个位置。D 用同一坐标系裁剪。
//
// 搜索余量：为了让逐字符 matchTemplate 能兜住比全局对齐更大的残余位移，
// 搜索区在字符框左右各多留 --padX 像素（默认 8）。
// 全局对齐之后实测每个字符还要往左挪约 9px，而推理侧固定半径 12 是按
// 训练侧 clamp(短边*0.08,6,30) 放大来的，左右各 +8 正好把余量补够。
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
    if (argc < 4) {
        std::printf("用法: markloc_vsmodel <modelDir> <outPng> <image>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型加载失败: %s\n", argv[1]);
        return 1;
    }
    const std::string outPng = argv[2];
    const std::string p8 = argv[3];
    cv::Mat gray = readTif(p8);
    if (gray.empty() || gray.size() != model.freq.size()) {
        std::printf("读图/尺寸不符: %s\n", p8.c_str());
        return 1;
    }
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    // 左右各多留多少像素给逐字符搜索
    int padX = 8, padY = 0;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--padX") == 0) padX = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--padY") == 0) padY = std::atoi(argv[i + 1]);
    }
    std::printf("逐字符搜索余量: 左右各 %d px，上下各 %d px\n", padX, padY);

    // ---- 与 batchcore::detectOne 完全同路
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    std::vector<chardet::CharUnit> units = baseUnits;
    cv::Point2d sh(0, 0);
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
        inkArea.x = std::max(0, inkArea.x - 20);
        inkArea.y = std::max(0, inkArea.y - 20);
        inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
        inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
        cv::Mat rC, sC;
        r32(inkArea).copyTo(rC);
        s32(inkArea).copyTo(sC);
        sh = cv::phaseCorrelate(sC, rC, cv::noArray());
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat aligned;
            cv::warpAffine(sig, aligned, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            sig = aligned;
        }
    }
    for (auto& u : units) {
        const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (box.width <= 4 || box.height <= 4) continue;
        // 左右各多 padX、上下各多 padY，让搜索有余量兜住全局对齐之后
        // 每个字符还要再挪动的那 9px。kSearch 只作为下限不足时的兜底。
        constexpr int kSearch = 12;
        const int sx = std::max(kSearch, padX);
        const int sy = std::max(kSearch, padY);
        cv::Rect search(box.x - sx, box.y - sy, box.width + 2 * sx, box.height + 2 * sy);
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
        const cv::Rect moved(box.x + mLoc.x - sx, box.y + mLoc.y - sy, box.width, box.height);
        if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
        u.bbox = moved;
    }

    markloc::ScratchParams sp;
    markloc::ScratchStats st;
    auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp, &st);
    std::sort(hits.begin(), hits.end(),
              [](const markloc::ScratchHit& a, const markloc::ScratchHit& b) {
                  return a.composite > b.composite;
              });

    // ---- 裁剪区域：包裹全部字符 + 外扩，保证同一坐标系
    cv::Rect roi = markloc::scratchBand(sig.size(), units, 40);
    roi &= cv::Rect(0, 0, sig.cols, sig.rows);

    // ---- A 模板
    cv::Mat pA;
    cv::normalize(model.freq(roi).clone(), pA, 0, 255, cv::NORM_MINMAX);
    pA.convertTo(pA, CV_8U);

    // ---- B 待检信号（对齐后）
    cv::Mat sNorm;
    cv::normalize(sig(roi).clone(), sNorm, 0, 255, cv::NORM_MINMAX);
    cv::Mat pB;
    sNorm.convertTo(pB, CV_8U);

    // ---- C 差异：图 - 模型（正=图里多出来的）
    //
    // 先算在单通道上，再上色。早先直接 subtract(sNorm, pA) 崩了，
    // 因为那时 pA 已转成 3 通道，尺寸对得上但通道数对不上。
    cv::Mat diffF;
    cv::subtract(sNorm, pA, diffF, cv::noArray(), CV_32F);
    cv::Mat pC(diffF.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < diffF.rows; ++y) {
        const float* d = diffF.ptr<float>(y);
        cv::Vec3b* o = pC.ptr<cv::Vec3b>(y);
        for (int x = 0; x < diffF.cols; ++x) {
            const float v = d[x];
            const int a = (int)std::min(255.0f, std::abs(v) * 1.6f);
            // 正：图里亮而模型没有 -> 红；负：模型亮而图里暗 -> 蓝
            o[x] = v > 0 ? cv::Vec3b(0, 0, (uchar)a) : cv::Vec3b((uchar)a, 0, 0);
        }
    }
    // 上色版本给显示用
    cv::Mat pAc, pBc;
    cv::cvtColor(pA, pAc, cv::COLOR_GRAY2BGR);
    cv::cvtColor(pB, pBc, cv::COLOR_GRAY2BGR);

    // ---- D 原图 + 命中线（同一 roi 坐标系）
    cv::Mat pD;
    cv::cvtColor(gray(roi).clone(), pD, cv::COLOR_GRAY2BGR);
    {
        // 对齐是把 sig 平移过来的，原图也要平移同样的量才对得上
        if (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat gw;
            cv::warpAffine(gray, gw, M, gray.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            cv::cvtColor(gw(roi).clone(), pD, cv::COLOR_GRAY2BGR);
        }
        const int show = std::min<int>(6, (int)hits.size());
        for (int i = 0; i < show; ++i) {
            const auto& h = hits[i];
            const cv::Scalar col = i == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 255);
            cv::line(pD, h.a1 - roi.tl(), h.a2 - roi.tl(), col, 2, cv::LINE_AA);
            char buf[48];
            std::snprintf(buf, sizeof(buf), "#%d S=%.0f", i, h.composite);
            cv::putText(pD, buf, h.a1 - roi.tl() + cv::Point(2, -4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 1, cv::LINE_AA);
        }
        for (const auto& u : units) cv::rectangle(pD, u.bbox - roi.tl(), cv::Scalar(0, 220, 220), 1);
    }

    // ---- 拼成一张，上方标注
    const int labH = 26;
    auto addLabel = [&](cv::Mat& m, const char* t) {
        cv::Mat o(m.rows + labH, m.cols, CV_8UC3, cv::Scalar(30, 30, 30));
        m.copyTo(o(cv::Rect(0, labH, m.cols, m.rows)));
        cv::putText(o, t, cv::Point(6, 19), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
        m = o;
    };
    addLabel(pAc, "A  model freq (what the model thinks is ink)");
    addLabel(pBc, "B  image signal (aligned)");
    addLabel(pC, "C  diff: red = bright but no model ink");
    addLabel(pD, "D  raw image + hits");

    cv::Mat out;
    cv::hconcat(std::vector<cv::Mat>{pAc, pBc, pC, pD}, out);

    char head[256];
    std::snprintf(head, sizeof(head), "%.60s   align dx=%+.1f dy=%+.1f   passed=%d",
                  fs::path(p8).filename().string().c_str(), sh.x, sh.y, (int)hits.size());
    cv::Mat fin(out.rows + 30, out.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    out.copyTo(fin(cv::Rect(0, 30, out.cols, out.rows)));
    cv::putText(fin, head, cv::Point(8, 21), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 120),
                1, cv::LINE_AA);

    cv::imwrite(outPng, fin);
    std::printf("命中 %d 条，已写 %s  (%dx%d)\n", (int)hits.size(), outPng.c_str(), fin.cols,
                fin.rows);
    for (size_t i = 0; i < hits.size() && i < 6; ++i)
        std::printf("  #%zu (%d,%d)-(%d,%d) S=%.0f 跨%d 深%.0f\n", i, hits[i].a1.x, hits[i].a1.y,
                    hits[i].a2.x, hits[i].a2.y, hits[i].composite, hits[i].charsHit,
                    hits[i].depth);
    return 0;
}
