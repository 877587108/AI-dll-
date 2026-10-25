// regionalign.cpp —— 新规格图（1496x1496）上重新定位区域 1/2/logo
//
// 背景：同工位换了拍摄参数，新图 1496x1496，旧模型基准 1688x1460。
// 宽高缩放比不同（宽 0.886、高 1.025），不是等比缩放，
// 所以不能把新图直接 resize 到旧尺寸——那样会把字形拉变形。
//
// 做法：对每个区域分别定位，允许区域之间有独立的平移和尺度。
//   1. 用旧模型的区域外框作为初始猜测，按尺寸比例映射到新图
//   2. 在该猜测附近做多尺度模板匹配，求每个区域自己的 (sx, sy, dx, dy)
//   3. 输出每个区域的对齐结果与置信度，供人工核对
//
// 区域编号沿用 regions.txt：
//   1 = Q046R8NLP 那一行（9 个字符）
//   2 = Silan 那一行（5 个字符）
//   3 = logo 椭圆（1 个"字符"框，实为整个椭圆）
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

static cv::Mat readAny(const std::string& p) {
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

struct RegionFit {
    int idx = -1;
    bool ok = false;
    double scale = 1.0;   // 统一缩放（先按等比找，再放开）
    int dx = 0, dy = 0;   // 相对初始猜测的平移
    double peak = 0.0;
    cv::Rect box;         // 在新图上的位置
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_regionalign <modelDir> <image> [--out prefix]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型加载失败\n");
        return 1;
    }
    const std::string imgPath = argv[2];
    std::string outPrefix = "out/trace/regionalign";
    for (int i = 3; i + 1 < argc; i += 2)
        if (std::strcmp(argv[i], "--out") == 0) outPrefix = argv[i + 1];

    cv::Mat gray = readAny(imgPath);
    if (gray.empty()) {
        std::printf("读图失败: %s\n", imgPath.c_str());
        return 1;
    }
    std::printf("图像 %dx%d，模型 %dx%d\n", gray.cols, gray.rows, model.freq.cols,
                model.freq.rows);

    const chardet::Config& cfg = model.config;
    // 新图的笔画信号。strokeSignal 的缩放是按整幅图统计的，
    // 分辨率变了要重新算，不能沿用旧图的比例
    const cv::Mat sig =
        chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // 只定位这三类区域：1 = 字符行、2 = Silan 行、3 = logo
    const int wantRegions[] = {1, 2, 3};

    std::vector<RegionFit> fits;
    for (int ri : wantRegions) {
        if (ri < 0 || ri >= (int)model.regions.size()) continue;
        const auto& rg = model.regions[ri];
        RegionFit fit;
        fit.idx = ri;

        // 初始猜测：按宽高比把旧区域框映射到新图
        const double fx = (double)gray.cols / model.freq.cols;
        const double fy = (double)gray.rows / model.freq.rows;
        const int gx = (int)std::lround(rg.rect.x * fx);
        const int gy = (int)std::lround(rg.rect.y * fy);
        const int gw = (int)std::lround(rg.rect.width * fx);
        const int gh = (int)std::lround(rg.rect.height * fy);

        // 模型侧的参考图：该区域的信号（用 ref 作为强度参考更贴近原始信号）
        cv::Mat ref32;
        model.ref.convertTo(ref32, CV_32F);
        cv::Mat refRegion = ref32(rg.rect & cv::Rect(0, 0, ref32.cols, ref32.rows)).clone();
        if (refRegion.empty()) continue;

        // 多尺度搜索：围绕 fx/fy 附近找最佳尺度，再在最佳尺度附近找平移
        double bestPeak = -1, bestScale = 1.0;
        int bestDx = 0, bestDy = 0;
        cv::Rect bestBox;
        const double sc0 = std::min(fx, fy);
        for (double s = sc0 * 0.85; s <= sc0 * 1.15; s += 0.005) {
            cv::Mat tpl;
            cv::resize(refRegion, tpl, cv::Size(), s, s, cv::INTER_LINEAR);
            if (tpl.cols < 8 || tpl.rows < 8) continue;
            if (tpl.cols > gray.cols || tpl.rows > gray.rows) continue;
            // 在初始猜测附近 ±60px 搜索
            const int pad = 60;
            cv::Rect sr(gx - pad, gy - pad, tpl.cols + 2 * pad, tpl.rows + 2 * pad);
            sr &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (sr.width < tpl.cols || sr.height < tpl.rows) continue;
            cv::Mat res;
            cv::matchTemplate(sig(sr), tpl, res, cv::TM_CCOEFF_NORMED);
            cv::Point ml;
            double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
            if (mv > bestPeak) {
                bestPeak = mv;
                bestScale = s;
                bestDx = sr.x + ml.x;
                bestDy = sr.y + ml.y;
                bestBox = cv::Rect(bestDx, bestDy, tpl.cols, tpl.rows);
            }
        }
        fit.ok = bestPeak > 0.40;
        fit.scale = bestScale;
        fit.dx = bestDx - gx;
        fit.dy = bestDy - gy;
        fit.peak = bestPeak;
        fit.box = bestBox;
        fits.push_back(fit);

        std::printf("区域 %d  猜测(%d,%d,%d,%d) -> 定位(%d,%d,%d,%d)  尺度%.3f 平移(%+d,%+d) 峰值%.3f %s\n",
                    ri, gx, gy, gw, gh, bestBox.x, bestBox.y, bestBox.width, bestBox.height,
                    bestScale, fit.dx, fit.dy, bestPeak, fit.ok ? "" : "  <- 置信度低");
    }

    // ---- 出对比图：左=旧模型参考，右=新图 + 定位框
    // 左边按区域裁剪后缩放成统一高度拼接，右边同理，方便并排看
    const int cellH = 200;
    std::vector<cv::Mat> leftCells, rightCells;
    for (const auto& f : fits) {
        if (f.idx < 0 || f.idx >= (int)model.regions.size()) continue;
        const auto& rg = model.regions[f.idx];
        cv::Mat ref;
        model.ref.convertTo(ref, CV_8U);
        cv::Mat lc = ref(rg.rect & cv::Rect(0, 0, ref.cols, ref.rows)).clone();
        cv::Mat rc = gray(f.box & cv::Rect(0, 0, gray.cols, gray.rows)).clone();
        if (lc.empty() || rc.empty()) continue;
        const double ls = (double)cellH / lc.rows;
        const double rs = (double)cellH / rc.rows;
        cv::resize(lc, lc, cv::Size((int)(lc.cols * ls), cellH));
        cv::resize(rc, rc, cv::Size((int)(rc.cols * rs), cellH));
        cv::Mat l3, r3;
        cv::cvtColor(lc, l3, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rc, r3, cv::COLOR_GRAY2BGR);
        cv::rectangle(r3, cv::Rect(0, 0, r3.cols - 1, r3.rows - 1), cv::Scalar(0, 220, 220), 1);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "region %d  scale %.3f  peak %.2f", f.idx, f.scale,
                      f.peak);
        cv::putText(l3, buf, cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 255, 120), 1, cv::LINE_AA);
        cv::putText(r3, buf, cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 255, 120), 1, cv::LINE_AA);
        leftCells.push_back(l3);
        rightCells.push_back(r3);
    }
    if (!leftCells.empty()) {
        const int wmax = 700;
        auto pad = [&](const cv::Mat& m) {
            cv::Mat o(cellH, wmax, CV_8UC3, cv::Scalar(20, 20, 20));
            m(cv::Rect(0, 0, std::min(m.cols, wmax), cellH)).copyTo(o(cv::Rect(0, 0, std::min(m.cols, wmax), cellH)));
            return o;
        };
        std::vector<cv::Mat> L, R;
        for (auto& m : leftCells) L.push_back(pad(m));
        for (auto& m : rightCells) R.push_back(pad(m));
        cv::Mat lo, ro;
        cv::vconcat(L, lo);
        cv::vconcat(R, ro);
        cv::Mat both, hdr(26, lo.cols + ro.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::hconcat(lo, ro, both);
        cv::Mat fin(both.rows + 26, both.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        both.copyTo(fin(cv::Rect(0, 26, both.cols, both.rows)));
        cv::putText(fin, "LEFT = model reference  |  RIGHT = new-spec image (1496x1496)",
                    cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
        const std::string op = outPrefix + ".png";
        cv::imwrite(op, fin);
        std::printf("对比图已写 %s (%dx%d)\n", op.c_str(), fin.cols, fin.rows);
    }

    // 同时把新图上的定位框画到大图上，便于看整体位置
    {
        cv::Mat vis;
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        const char* names[] = {"", "region1 Q046R8NLP", "region2 Silan", "region3 logo"};
        for (const auto& f : fits) {
            if (!f.ok) continue;
            const cv::Scalar col = f.idx == 3 ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255);
            cv::rectangle(vis, f.box, col, 2);
            const char* nm = (f.idx >= 0 && f.idx <= 3) ? names[f.idx] : "?";
            cv::putText(vis, nm, cv::Point(f.box.x + 4, f.box.y - 6), cv::FONT_HERSHEY_SIMPLEX,
                        0.6, col, 2, cv::LINE_AA);
        }
        const std::string op2 = outPrefix + "_full.png";
        cv::imwrite(op2, vis);
        std::printf("整体定位图已写 %s (%dx%d)\n", op2.c_str(), vis.cols, vis.rows);
    }
    return 0;
}
