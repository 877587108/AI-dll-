// 信号 vs 模型 叠加诊断
//
// 目的：判定"模型没覆盖的信号亮区"是什么东西。
//   - 如果是一片完整的笔画形状 → 模型缺这个字/笔画，需补样本重训
//   - 如果是一条细长条纹            → 是真划痕，本来就该判 NG
//   - 如果整体错开一个位移           → 配准方向错了
//
// 输出三栏 + 一栏叠加：
//   1. 配准后的信号（SIGNAL）
//   2. 模型频率图（FREQ）
//   3. 两者并排，同坐标
//   4. 叠加：红=信号亮模型空白(漏出) 绿=模型有墨信号空 黄=重合
//
// 用法: markloc/sigoverlay <modelDir> <image> <outPrefix> [x y w h]
#include <algorithm>
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

static void saveU8(const std::string& path, const cv::Mat& m) {
    std::vector<uchar> buf;
    if (!cv::imencode(".png", m, buf)) return;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    std::fwrite(buf.data(), 1, buf.size(), fp);
    std::fclose(fp);
}

// 灰度 → 伪彩色（热力），便于看清亮区形状
static cv::Mat heat(const cv::Mat& g) {
    cv::Mat c;
    cv::applyColorMap(g, c, cv::COLORMAP_INFERNO);
    return c;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: markloc_sigoverlay <modelDir> <image> <outPrefix> [x y w h]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    cv::Mat gray;
    if (!readGray(argv[2], gray)) {
        std::printf("图片读取失败\n");
        return 1;
    }
    if (gray.size() != model.freq.size()) {
        std::printf("尺寸不符: 图 %dx%d 模型 %dx%d\n", gray.cols, gray.rows, model.freq.cols,
                    model.freq.rows);
        return 1;
    }
    const chardet::Config cfg = model.config;
    const auto units = chardet::toAllUnits(model);

    // ---- 复刻推理侧配准（DefectViewWindow.cpp:1079-1118）----
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat r32;
    model.freq.convertTo(r32, CV_32F);
    cv::Mat s32;
    sig.convertTo(s32, CV_32F);

    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width <= 0 || b.height <= 0) continue;
        inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    cv::Point2d sh(0, 0);
    if (inkArea.area() > 0) {
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
            cv::warpAffine(sig, aligned, M, sig.size(),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
            sig = aligned;
        }
    }
    std::printf("配准: (%.2f, %.2f)\n", sh.x, sh.y);

    // ---- 裁剪区域：默认整个划痕搜索带，也可命令行指定 ----
    markloc::ScratchParams P;
    cv::Rect band;
    if (argc >= 8) {
        band = cv::Rect(std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]),
                        std::atoi(argv[7]));
    } else {
        band = markloc::scratchBand(sig.size(), units, P.growPx);
    }
    band &= cv::Rect(0, 0, sig.cols, sig.rows);
    std::printf("区域: (%d,%d,%d,%d)\n", band.x, band.y, band.width, band.height);

    const cv::Mat sRoi = sig(band);
    const cv::Mat fRoi = model.freq(band);

    // ---- 掩膜：与 findScratches 同口径 ----
    cv::Mat weak, modelInk, modelDil, notModel, leak;
    cv::compare(sRoi, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    cv::compare(fRoi, static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);
    const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
    cv::dilate(modelInk, modelDil, kd);
    cv::bitwise_not(modelDil, notModel);
    cv::bitwise_and(weak, notModel, leak);
    std::printf("漏出(闭运算前): %d / %d = %.1f%%\n", cv::countNonZero(leak),
                cv::countNonZero(weak), 100.0 * cv::countNonZero(leak) / std::max(1, cv::countNonZero(weak)));

    // 复刻 findScratches 的 bridge 闭运算 + medianBlur，看闭运算后还剩什么
    cv::Mat closed = leak.clone();
    if (P.bridge > 0) {
        const int bk = std::max(3, P.bridge | 1);
        const cv::Mat kb = cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk});
        cv::morphologyEx(closed, closed, cv::MORPH_CLOSE, kb);
    }
    cv::medianBlur(closed, closed, 3);
    std::printf("闭运算后(bridge=%d): %d px  (闭运算前 %d px, 放大 %.1f 倍)\n", P.bridge,
                cv::countNonZero(closed), cv::countNonZero(leak),
                static_cast<double>(cv::countNonZero(closed)) / std::max(1, cv::countNonZero(leak)));

    // Hough 找线，打印真实命中
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(closed, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);
    std::printf("Hough 检出 %zu 条\n", lines.size());

    // ---- 三栏 + 叠加，统一放大 2x ----
    cv::Mat sig8, sNorm;
    cv::normalize(sRoi, sNorm, 0, 255, cv::NORM_MINMAX);
    sNorm.convertTo(sig8, CV_8U);
    const cv::Mat sigColor = heat(sig8);
    const cv::Mat freqColor = heat(fRoi);

    // 叠加：B=信号(蓝) G=模型(绿) R=漏出(红)
    cv::Mat over = cv::Mat::zeros(sRoi.size(), CV_8UC3);
    for (int y = 0; y < sRoi.rows; ++y) {
        for (int x = 0; x < sRoi.cols; ++x) {
            const unsigned int sv = sig8.at<uchar>(y, x);
            const unsigned int fv = fRoi.at<uchar>(y, x);
            const uchar lv = leak.at<uchar>(y, x);
            cv::Vec3b& p = over.at<cv::Vec3b>(y, x);
            p[0] = static_cast<uchar>(sv > 255u ? 255u : sv);  // B: 信号
            p[1] = static_cast<uchar>(fv > 255u ? 255u : fv);  // G: 模型
            p[2] = lv ? 255 : 0;                               // R: 漏出
        }
    }

    auto up = [](const cv::Mat& m) {
        cv::Mat o;
        cv::resize(m, o, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
        return o;
    };
    const cv::Mat a1 = up(sigColor), a2 = up(freqColor), a3 = up(over);
    cv::Mat combo;
    cv::hconcat(std::vector<cv::Mat>{a1, a2, a3}, combo);
    cv::putText(combo, "SIGNAL", {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {255, 255, 255}, 2);
    cv::putText(combo, "MODEL FREQ", {a1.cols + 10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                {255, 255, 255}, 2);
    cv::putText(combo, "LEAK=B1 G1 R1", {a1.cols + a2.cols + 10, 28}, cv::FONT_HERSHEY_SIMPLEX,
                0.9, {255, 255, 255}, 2);

    const std::string out = std::string(argv[3]) + "_overlay.png";
    saveU8(out, combo);
    std::printf("已写出 %s\n", out.c_str());

    // 单独再存一张纯漏出掩膜，方便看形状
    cv::Mat leakVis;
    cv::resize(leak, leakVis, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
    const std::string outLeak = std::string(argv[3]) + "_leak.png";
    saveU8(outLeak, leakVis);
    std::printf("已写出 %s\n", outLeak.c_str());

    // 闭运算后的掩膜：这才是真正进 Hough 的输入
    cv::Mat closedVis;
    cv::resize(closed, closedVis, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
    const std::string outClosed = std::string(argv[3]) + "_closed.png";
    saveU8(outClosed, closedVis);
    std::printf("已写出 %s\n", outClosed.c_str());

    // 闭运算掩膜叠加到信号上：看线条是不是沿着笔画走
    cv::Mat onSig;
    const cv::Mat sigGray2 = up(sig8);
    cv::cvtColor(sigGray2, onSig, cv::COLOR_GRAY2BGR);
    for (const auto& ln : lines) {
        cv::line(onSig, cv::Point(ln[0] * 2, ln[1] * 2), cv::Point(ln[2] * 2, ln[3] * 2),
                 {0, 0, 255}, 2);
    }
    cv::Mat closedUp = up(closed);
    cv::Mat closedBgr;
    cv::cvtColor(closedUp, closedBgr, cv::COLOR_GRAY2BGR);
    for (const auto& ln : lines) {
        cv::line(closedBgr, cv::Point(ln[0] * 2, ln[1] * 2), cv::Point(ln[2] * 2, ln[3] * 2),
                 {0, 0, 255}, 2);
    }
    cv::Mat linesCombo;
    cv::hconcat(onSig, closedBgr, linesCombo);
    cv::putText(linesCombo, "SIGNAL + HOUGH", {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                {255, 255, 255}, 2);
    cv::putText(linesCombo, "CLOSED MASK + HOUGH", {onSig.cols + 10, 28},
                cv::FONT_HERSHEY_SIMPLEX, 0.9, {255, 255, 255}, 2);
    const std::string outLines = std::string(argv[3]) + "_hough.png";
    saveU8(outLines, linesCombo);
    std::printf("已写出 %s\n", outLines.c_str());
    return 0;
}
