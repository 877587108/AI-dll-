// lboxmatch.cpp —— 逐字符匹配体检：模板大小、搜索半径、峰值、位移
//
// 用户问：定位时是不是真的在"字符框的大小"下对模板字符框做匹配。
//
// 代码事实（已核对源码）：
//   训练 model.cpp:306   模板 = coarseF(box)，尺寸 = charBox 原尺寸
//   训练 model.cpp:302   搜索半径 = clamp(短边*0.08, 6, searchRadius)
//   推理 batchcore:170   模板 = freq 里 event box 区域，尺寸 = charBox 原尺寸
//   推理 batchcore:164   搜索半径 = 固定 12
// 尺寸一致，但半径策略不一致。本工具量这件事的实际影响。
//
// 输出每个字符：框尺寸、训练半径、推理半径、匹配峰值、位移，
// 以及"模板内部有效结构"（标准差）——TM_CCOEFF_NORMED 在模板几乎平坦时无意义。
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
        std::printf("用法: markloc_boxmatch <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;
    cv::Mat gray = readTif(argv[2]);
    if (gray.empty() || gray.size() != model.freq.size()) {
        std::printf("读图/尺寸不符: %s\n", argv[2]);
        return 1;
    }
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // 全局对齐（与 batchcore 一致）
    if (!model.ref.empty()) {
        cv::Mat r32, s32;
        model.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : baseUnits) {
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
        const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat aligned;
            cv::warpAffine(sig, aligned, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            sig = aligned;
        }
        std::printf("全局对齐 dx=%+.2f dy=%+.2f\n", sh.x, sh.y);
    }

    std::printf("\n%4s %14s %6s %6s %6s %8s %10s %8s\n", "序", "框 w x h", "训练r", "推理r",
                "距边", "模板std", "峰值", "位移");

    int lowPeak = 0, bigShift = 0, flatTempl = 0, nearEdge = 0;
    for (size_t ui = 0; ui < baseUnits.size(); ++ui) {
        const auto& u = baseUnits[ui];
        const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (box.width <= 4 || box.height <= 4) continue;

        // 训练侧的半径策略（model.cpp:301-303）
        const int shortSide = std::min(box.width, box.height);
        const int rTrain = std::clamp(static_cast<int>(shortSide * 0.08), 6, 30);
        // 推理侧的半径（batchcore:164）
        const int rInfer = 12;

        // 模板内部结构强度：TM_CCOEFF_NORMED 在模板平坦时无意义
        cv::Mat tmpl = model.freq(box);
        cv::Scalar tmean, tstd;
        cv::meanStdDev(tmpl, tmean, tstd);

        cv::Rect search(box.x - rInfer, box.y - rInfer, box.width + 2 * rInfer,
                        box.height + 2 * rInfer);
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
        const int dx = mLoc.x - rInfer, dy = mLoc.y - rInfer;

        // 搜索区是否碰到了图像边界（碰到说明"外扩"被裁掉了，实际可选位置变少）
        const bool clipped = (box.x - rInfer < 0) || (box.y - rInfer < 0) ||
                             (box.x + box.width + rInfer > sig.cols) ||
                             (box.y + box.height + rInfer > sig.rows);

        char b[32];
        std::snprintf(b, sizeof(b), "%dx%d", box.width, box.height);
        std::printf("%4d %14s %6d %6d %6s %8.2f %10.3f %+4d,%+d%s%s\n", u.index, b, rTrain,
                    rInfer, clipped ? "是" : "-", tstd[0], mVal, dx, dy,
                    mVal < 0.60 ? "  <-峰值低" : "", (std::abs(dx) > 6 || std::abs(dy) > 6) ? "  <-位移大" : "");
        if (mVal < 0.60) ++lowPeak;
        if (std::abs(dx) > 6 || std::abs(dy) > 6) ++bigShift;
        if (tstd[0] < 5.0) ++flatTempl;
        if (clipped) ++nearEdge;
    }
    std::printf("\n汇总: 峰值<0.60 的字符 %d 个，位移>6px 的 %d 个，模板几乎平坦(std<5) 的 %d 个，搜索区被边界裁掉 %d 个\n",
                lowPeak, bigShift, flatTempl, nearEdge);
    return 0;
}
