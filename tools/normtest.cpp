// 归一化假设的直接实验：同一条流水线，只换信号预处理，看 leak 怎么变
//
// 立论：strokeSignal 的归一化刻度由**整幅图**估计（q995/q90 直方图，
// model.cpp:56-59）。整幅图里有引脚、封装边缘这些远强于刻印字符的结构，
// 刻度被它们抬高，刻印字符的相对幅度被压低。实测 region0~3 的信号 p90
// 只有 0.069~0.774，而 strokeHi 是 0.35 —— logo 区几乎整块落在阈值以下。
//
// 如果这个立论成立，那么"完全不归一化"（原残差）应当让漏出的残留大幅下降。
// 本工具就是拿这条对照来判它成不成立，而不是靠推理。
//
// 用法: markloc/normtest <modelDir> <image>
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

/// 复刻 computeSignal，但把归一化刻度强制成给定值（1.0 = 不归一化，保留原始残差）
static cv::Mat signalWithScale(const cv::Mat& gray, const chardet::Config& cfg, double forceScale) {
    cv::Mat p;
    gray.convertTo(p, CV_32F);
    cv::Mat bg;
    cv::GaussianBlur(p, bg, cv::Size(0, 0), cfg.sigma);
    cv::Mat d;
    cv::subtract(p, bg, d);
    if (forceScale > 0) d.convertTo(d, CV_32F, 1.0 / forceScale);
    cv::threshold(d, d, 0.0, 0.0, cv::THRESH_TOZERO);
    return d;
}

struct Stage {
    int bright = 0, leak = 0, closed = 0, lines = 0, passLen = 0;
    int hits = 0;
    double best = 0;
};

static Stage runPipeline(const cv::Mat& sig, const chardet::Model& model,
                         const std::vector<chardet::CharUnit>& units,
                         const chardet::Config& cfg, const markloc::ScratchParams& P,
                         int* leakIn = nullptr, int* leakOut = nullptr) {
    Stage st;
    const cv::Rect band = markloc::scratchBand(sig.size(), units, P.growPx);
    const cv::Mat sBand = sig(band);

    cv::Mat weak;
    cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    st.bright = cv::countNonZero(weak);
    const cv::Mat weak0 = weak.clone();

    cv::Mat modelInk;
    cv::compare(model.freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);
    cv::Mat d, nm;
    cv::dilate(modelInk, d, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(d, nm);
    cv::bitwise_and(weak, nm, weak);
    st.leak = cv::countNonZero(weak);

    if (leakIn || leakOut) {
        cv::Mat inAny = cv::Mat::zeros(weak.size(), CV_8U);
        for (const auto& u : units) {
            const cv::Rect b = u.bbox & band;
            if (b.area() > 0) inAny(cv::Rect(b.x - band.x, b.y - band.y, b.width, b.height)) = 255;
        }
        int ki = 0, ko = 0;
        for (int y = 0; y < weak.rows; ++y)
            for (int x = 0; x < weak.cols; ++x) {
                if (!weak.at<uchar>(y, x)) continue;
                if (inAny.at<uchar>(y, x)) ++ki;
                else ++ko;
            }
        if (leakIn) *leakIn = ki;
        if (leakOut) *leakOut = ko;
    }

    cv::Mat closed;
    cv::morphologyEx(weak, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               {std::max(3, P.bridge | 1), std::max(3, P.bridge | 1)}));
    cv::medianBlur(closed, closed, 3);
    st.closed = cv::countNonZero(closed);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(closed, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);
    st.lines = static_cast<int>(lines.size());
    for (const auto& ln : lines) {
        if (cv::norm(cv::Point(ln[2], ln[3]) - cv::Point(ln[0], ln[1])) >= P.minLen) ++st.passLen;
    }

    markloc::ScratchStats s2;
    const auto hits = markloc::findScratches(sig, model.freq, units, cfg, P, &s2);
    st.hits = static_cast<int>(hits.size());
    st.best = s2.bestComposite;
    return st;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_normtest <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("读取失败或尺寸不符\n");
        return 1;
    }
    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);
    markloc::ScratchParams P;

    // 先量出当前实际用的刻度
    {
        cv::Mat p;
        gray.convertTo(p, CV_32F);
        cv::Mat bg;
        cv::GaussianBlur(p, bg, cv::Size(0, 0), cfg.sigma);
        cv::Mat d;
        cv::subtract(p, bg, d);
        std::vector<float> v;
        v.assign(reinterpret_cast<const float*>(d.data),
                 reinterpret_cast<const float*>(d.data) + d.total());
        std::vector<float> pos;
        for (float x : v)
            if (x > 0) pos.push_back(x);
        std::sort(pos.begin(), pos.end());
        const auto q = [&](double f) {
            return pos.empty() ? 0.0 : static_cast<double>(pos[static_cast<size_t>(pos.size() * f)]);
        };
        const double q995 = q(0.995), q90 = q(0.90);
        const double scale = std::max(8.0, q995 * 0.55 + q90 * 0.45);
        std::printf("当前刻度: q995=%.1f q90=%.1f → scale=%.1f （残差最大 %.1f）\n", q995, q90,
                    scale, pos.empty() ? 0.0 : pos.back());
    }

    const double scales[] = {0.0, 50.0, 100.0, 150.0, 200.0, 400.0};
    std::printf("\n%-24s %8s %8s %8s %8s %6s %6s %6s\n", "信号预处理", "亮像素", "leak",
                "框外leak", "闭后", "线", "过长", "命中");
    for (double sc : scales) {
        cv::Mat sig = signalWithScale(gray, cfg, sc);
        // 对齐：ref 参照，与 runDetect 一致
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

        int li = 0, lo = 0;
        const Stage st = runPipeline(sig, model, units, cfg, P, &li, &lo);
        char name[64];
        if (sc <= 0) std::snprintf(name, sizeof(name), "原始残差(不归一化)");
        else std::snprintf(name, sizeof(name), "刻度=%.0f", sc);
        std::printf("%-24s %8d %8d %8d %8d %6d %6d %6d\n", name, st.bright, st.leak, lo, st.closed,
                    st.lines, st.passLen, st.hits);
    }
    std::printf("\n另有 1 行对照：当前 strokeSignal（整图刻度）\n");
    {
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
        int li = 0, lo = 0;
        const Stage st = runPipeline(sig, model, units, cfg, P, &li, &lo);
        std::printf("%-24s %8d %8d %8d %8d %6d %6d %6d\n", "strokeSignal(现状)", st.bright, st.leak,
                    lo, st.closed, st.lines, st.passLen, st.hits);
    }
    return 0;
}
