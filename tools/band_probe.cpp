// band_probe.cpp —— 环在极坐标展开图上到底落在哪一行
//
// 【为什么需要】
// 带型判定按 [0.75a, 1.35a] 取半径带，认为环就在这附近。
// 但 0095_RT 上环完好却被报成断口 4 处，怀疑环实际不在这个带里。
// 这里把每个角度"有墨的半径范围"直接打出来，看它落在哪：
//   · 若环稳定在 r≈a，说明带宽没问题，问题在别处
//   · 若环实际在 0.9a 或 1.1a 附近浮动，说明拉直/极坐标的尺度对不上
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <QDir>
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
    if (argc < 3) {
        std::printf("usage: markloc_bandprobe <modelRoot> <image>\n");
        return 2;
    }
    chardet::Model model;
    const QString vdir =
        QDir(QString::fromLocal8Bit(argv[1])).absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    chardet::RegionInfo pat;
    for (const auto& r : model.regions)
        if (r.isPattern()) {
            pat = r;
            break;
        }
    if (!pat.isPattern()) {
        std::printf("模型里没有图案区\n");
        return 1;
    }
    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g) || g.size() != model.freq.size()) {
        std::printf("读图失败或尺寸不符\n");
        return 1;
    }
    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

    const auto pb = chardet::analyzePatternBreak(sig, pat, model.config,
                                                 chardet::PatternBreakParams{});
    std::printf("a=%.1f b=%.1f  innerBlocked=%d bandOuter=%d\n", pat.ell.a, pat.ell.b,
                pb.innerBlocked, pb.bandOuter);
    std::printf("覆盖%.4f 断口%d 最长缺口%.2f° 断点分%.0f 判定%s\n", pb.coverage, pb.breaks,
                pb.maxGapAngle, pb.score, pb.flagged ? "NG" : "OK");
    std::printf("断口坐标（全图）:\n");
    for (const auto& p : pb.breakAt) std::printf("  (%.1f, %.1f)\n", p.x, p.y);

    // 复算拉直 + 极坐标，把环的半径剖面打出来
    const double a = pat.ell.a, b = pat.ell.b;
    const double kScale = a / std::max(1.0, b);
    const double rot = pat.ell.angleDeg * CV_PI / 180.0;
    const double cR = std::cos(-rot), sR = std::sin(-rot);
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -pat.ell.center.x, 0, 1, -pat.ell.center.y, 0, 0, 1);
    cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
    cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, pat.ell.center.x, 0, 1, pat.ell.center.y, 0, 0, 1);
    cv::Mat M3 = T2 * Sc * Rm * T1;
    cv::Mat M;
    M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
    cv::Mat straight;
    cv::warpAffine(sig, straight, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);

    const int pad = static_cast<int>(a * 0.45) + 8;
    const cv::Rect roi =
        cv::Rect(static_cast<int>(pat.ell.center.x - a - pad),
                 static_cast<int>(pat.ell.center.y - a - pad), static_cast<int>(2 * (a + pad)),
                 static_cast<int>(2 * (a + pad))) &
        cv::Rect(0, 0, straight.cols, straight.rows);
    cv::Mat roiM = straight(roi);
    const cv::Point2f cInRoi(static_cast<float>(pat.ell.center.x - roi.x),
                             static_cast<float>(pat.ell.center.y - roi.y));
    const double maxR = a + pad;
    const int nAngle = 720, nRadius = static_cast<int>(std::lround(maxR));
    cv::Mat pol;
    cv::warpPolar(roiM, pol, cv::Size(nRadius, nAngle), cInRoi, maxR,
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    cv::Mat polT;
    cv::transpose(pol, polT);
    pol = polT;  // rows=半径, cols=角度

    // 对每个角度，找半径方向上超过 0.35 的最长连续段，打印它的起点/终点
    //
    // 注意：不能只取"最长的一段"。环内部还有字母（SC），字母在某些角度
    // 比环本身在半径方向更厚，取最长会挑中字母。这里只统计落在
    // [0.75a, 1.35a] 这个带内的段 —— 那才是环该在的位置。
    const int bLo = static_cast<int>(std::lround(a * 0.75));
    const int bHi = static_cast<int>(std::lround(a * 1.35));
    std::printf("\n(只看带内 r=%d..%d 的连续段)\n", bLo, bHi);
    std::printf("\n角度  带内有墨半径段(r)     段中心   a\n");
    for (int x = 0; x < pol.cols; x += 45) {
        int bestRun = 0, bestStart = -1, run = 0, start = -1;
        for (int r = bLo; r <= std::min(bHi, pol.rows - 1); ++r) {
            if (pol.at<float>(r, x) >= 0.35f) {
                if (run == 0) start = r;
                ++run;
                if (run > bestRun) {
                    bestRun = run;
                    bestStart = start;
                }
            } else {
                run = 0;
            }
        }
        const double deg = 360.0 * x / pol.cols;
        if (bestRun > 0) {
            std::printf("%5.0f°  r=%4d..%-4d(长%2d)  %6.1f  %.1f\n", deg, bestStart,
                        bestStart + bestRun - 1, bestRun,
                        (bestStart + bestStart + bestRun - 1) / 2.0, a);
        } else {
            std::printf("%5.0f°  带内无墨  <-- 这里会被判成缺口\n", deg);
        }
    }
    return 0;
}
