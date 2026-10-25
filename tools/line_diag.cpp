// linediag.cpp —— 一条候选线到底是划痕还是字形边缘
//
// 【要回答的问题】
// 0101 配准后最高分 796 落在线段 (667,936)-(671,815) 上。
// 从放大图看，这条线紧贴数字 4 的竖笔右侧，不是独立的痕迹。
// 但"看起来像"不算证据。真正的区别在这里：
//
//   真划痕   线两侧都是背景，线本身是亮的
//             -> 垂直剖面上是"山峰"形状：两侧低、中间高
//
//   字形边缘 线一侧是笔画（亮），另一侧是背景（暗）
//             -> 垂直剖面上是"台阶"形状：一侧一直亮、另一侧一直暗
//
// 所以只要沿垂直线方向取剖面，看它是山峰还是台阶，就能分开。
//
// 判据：台阶 = 一侧均值明显高于另一侧，且线本身不在两者中间
//       山峰 = 两侧均值接近，且都明显低于线本身
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <QFile>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

static bool readGrayU8(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray b = f.readAll();
    f.close();
    if (b.isEmpty()) return false;
    const std::vector<uchar> buf(b.begin(), b.end());
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::printf("usage: markloc_linediag <modelDir> <image> x1 y1 x2 y2 [reach]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g) || g.size() != model.freq.size()) {
        std::printf("read failed\n");
        return 1;
    }
    const cv::Point p1(std::atoi(argv[3]), std::atoi(argv[4]));
    const cv::Point p2(std::atoi(argv[5]), std::atoi(argv[6]));
    const int reach = argc >= 8 ? std::atoi(argv[7]) : 10;

    const cv::Mat sigRaw =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    // 必须配准：locate 给出的坐标是配准后的，
    // 拿配准前的信号去采样会整片偏掉（0101 上差 9.6px）。
    cv::Mat sig = sigRaw.clone();
    {
        std::vector<chardet::CharUnit> base;
        for (size_t ri = 0; ri < model.regions.size(); ++ri) {
            if (model.regions[ri].isPattern()) continue;
            auto u = chardet::toUnits(model, static_cast<int>(ri));
            for (auto& x : u) base.push_back(std::move(x));
        }
        if (!model.ref.empty() && !base.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sigRaw.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : base) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sigRaw.cols, sigRaw.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            if (inkArea.area() > 0) {
                inkArea.x = std::max(0, inkArea.x - 20);
                inkArea.y = std::max(0, inkArea.y - 20);
                inkArea.width = std::min(sigRaw.cols - inkArea.x, inkArea.width + 40);
                inkArea.height = std::min(sigRaw.rows - inkArea.y, inkArea.height + 40);
                cv::Mat rC, sC;
                r32(inkArea).copyTo(rC);
                s32(inkArea).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                std::printf("(配准位移 %.2f, %.2f)\n", sh.x, sh.y);
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat al;
                    cv::warpAffine(sigRaw, al, M, sigRaw.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                    sig = al;
                }
            }
        }
    }

    // 线段法向
    cv::Point2d d(p2.x - p1.x, p2.y - p1.y);
    const double nl = std::hypot(d.x, d.y);
    if (nl < 1e-6) {
        std::printf("线段长度为 0\n");
        return 1;
    }
    d.x /= nl;
    d.y /= nl;
    const cv::Point2d nrm(-d.y, d.x);

    std::printf("线段 (%d,%d)-(%d,%d) 长 %.1f  法向 (%.3f, %.3f)  reach=%d\n", p1.x, p1.y, p2.x,
                p2.y, nl, nrm.x, nrm.y, reach);

    int nSamp = 0;
    double sumSideN = 0, sumSideP = 0, sumPeak = 0;
    int peakIsMiddle = 0, flankHigher = 0;
    // 采样点跳过两个端点附近，端点处远离字形，会把统计拉稀
    const int margin = std::max(2, static_cast<int>(nl * 0.1));
    for (int t = margin; t <= static_cast<int>(nl) - margin; ++t) {
        const double f = static_cast<double>(t) / nl;
        const double qx = p1.x + d.x * nl * f;
        const double qy = p1.y + d.y * nl * f;
        // 取法向剖面
        std::vector<double> prof;
        prof.reserve(2 * reach + 1);
        for (int k = -reach; k <= reach; ++k) {
            const int xi = static_cast<int>(std::lround(qx + nrm.x * k));
            const int yi = static_cast<int>(std::lround(qy + nrm.y * k));
            if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) {
                prof.push_back(0.0);
                continue;
            }
            prof.push_back(static_cast<double>(sig.at<float>(yi, xi)));
        }
        // 剖面中心（线本身）与两侧
        const double mid = prof[reach];
        double sideN = 0, sideP = 0;
        int cN = 0, cP = 0;
        for (int k = 1; k <= reach; ++k) {
            if (reach - k >= 0) {
                sideN += prof[reach - k];
                ++cN;
            }
            if (reach + k < static_cast<int>(prof.size())) {
                sideP += prof[reach + k];
                ++cP;
            }
        }
        sideN /= std::max(1, cN);
        sideP /= std::max(1, cP);
        sumSideN += sideN;
        sumSideP += sideP;
        sumPeak += mid;
        // 中心是不是"峰"：比两侧都高
        if (mid > sideN && mid > sideP) ++peakIsMiddle;
        // 有没有一侧明显更亮（台阶的标志）
        if (std::max(sideN, sideP) > 0.5 && std::min(sideN, sideP) < 0.15) ++flankHigher;
        ++nSamp;
    }
    if (nSamp == 0) {
        std::printf("采样数为 0\n");
        return 1;
    }
    const double avgN = sumSideN / nSamp;
    const double avgP = sumSideP / nSamp;
    const double avgMid = sumPeak / nSamp;

    std::printf("\n=== 剖面统计（%d 个采样点）===\n", nSamp);
    std::printf("线本身均值   %.4f\n", avgMid);
    std::printf("一侧均值     %.4f\n", avgN);
    std::printf("另一侧均值   %.4f\n", avgP);
    std::printf("中心为峰的采样点 %d / %d = %.2f\n", peakIsMiddle, nSamp,
                static_cast<double>(peakIsMiddle) / nSamp);
    std::printf("一侧亮一侧暗的采样点 %d / %d = %.2f\n", flankHigher, nSamp,
                static_cast<double>(flankHigher) / nSamp);

    // 判读
    const double asym = std::abs(avgN - avgP) / std::max(1e-6, std::max(avgN, avgP));
    std::printf("\n两侧不对称度 %.2f\n", asym);
    std::printf("判读：");
    if (asym > 0.7) {
        std::printf("【台阶】一侧明显亮、一侧明显暗 —— 这是字形边缘，不是划痕\n");
    } else if (avgMid > std::max(avgN, avgP) * 1.3 && asym < 0.4) {
        std::printf("【山峰】两侧接近且都低于线本身 —— 这是独立痕迹（划痕）\n");
    } else {
        std::printf("【不明确】两侧不对称度%.2f，线本身/两侧比%.2f —— 需要人看图\n", asym,
                    avgMid / std::max(1e-6, (avgN + avgP) * 0.5));
    }
    return 0;
}
