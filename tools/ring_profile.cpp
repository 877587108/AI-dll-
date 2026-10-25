// ring_profile.cpp —— 单张图环上每个角度的墨况全表
//
// 【为什么需要】
// "缺口 0.50°" 这种汇总数字看不出环到底在哪断开、断得多深。
// 而老板明确说"不能单纯用角度去看" —— 意味着断开的形态
// （断续程度、边缘锐利度、断口两侧的墨量）比角度本身更关键。
// 这里把 720 个角度的原始信息全打出来，供判断该换什么判据。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <QDir>
#include <QFile>
#include <QFileInfo>
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
    if (argc < 4) {
        std::printf("usage: markloc_ringprofile <modelRoot> <tmpl.png> <image...>\n");
        return 2;
    }
    chardet::Model model;
    const QString vdir =
        QDir(QString::fromLocal8Bit(argv[1])).absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat tmpl;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), tmpl)) {
        std::printf("模板读取失败\n");
        return 1;
    }
    const double tplCx = 127.8, tplCy = 116.0, tplA = 132.1, tplB = 76.75, tplRot = -13.78;

    for (int a = 3; a < argc; ++a) {
        cv::Mat g;
        if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g)) continue;

        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        const double cx = mloc.x + tplCx, cy = mloc.y + tplCy;
        const double A = tplA, B = tplB;

        // 在整幅上算信号（与正式判据一致）
        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

        // 用与 analyzePatternBreak 相同的方式：拉直 + 极坐标
        const double kScale = A / std::max(1.0, B);
        const double rot = tplRot * CV_PI / 180.0;
        const double cR = std::cos(-rot), sR = std::sin(-rot);
        cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -cx, 0, 1, -cy, 0, 0, 1);
        cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
        cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
        cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
        cv::Mat M3 = T2 * Sc * Rm * T1;
        cv::Mat M;
        M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
        cv::Mat straight;
        cv::warpAffine(sig, straight, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);

        const int pad = static_cast<int>(A * 0.45) + 8;
        cv::Rect roi(static_cast<int>(cx - A - pad), static_cast<int>(cy - A - pad),
                     static_cast<int>(2 * (A + pad)), static_cast<int>(2 * (A + pad)));
        roi &= cv::Rect(0, 0, straight.cols, straight.rows);
        const cv::Point2f cInRoi(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
        const double maxR = A + pad;
        const int nRadius = static_cast<int>(std::lround(maxR));
        cv::Mat pol;
        cv::warpPolar(straight(roi), pol, cv::Size(nRadius, 720), cInRoi, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        cv::Mat polT;
        cv::transpose(pol, polT);
        pol = polT;

        const int bLo = static_cast<int>(std::lround(A * 0.75));
        const int bHi = std::min(static_cast<int>(std::lround(A * 1.35)), pol.rows - 1);
        const int Wcols = pol.cols;

        std::printf("\n=== %s  峰值%.3f  带 r=%d..%d ===\n",
                    QFileInfo(QString::fromLocal8Bit(argv[a])).fileName().toUtf8().constData(),
                    mval, bLo, bHi);
        std::printf("角度  最长连续段  峰值   均值   （每 5 度一采）\n");

        std::vector<int> runLen(Wcols, 0);
        std::vector<double> peak(Wcols, 0.0);
        for (int x = 0; x < Wcols; ++x) {
            int run = 0, best = 0;
            double pk = 0;
            for (int r = bLo; r <= bHi; ++r) {
                const double v = pol.at<float>(r, x);
                pk = std::max(pk, v);
                if (v >= 0.35f) {
                    if (++run > best) best = run;
                } else {
                    run = 0;
                }
            }
            runLen[x] = best;
            peak[x] = pk;
        }
        for (int deg = 0; deg < 360; deg += 5) {
            const int x = deg * Wcols / 360;
            // 该角度 ±2 个采样内的最小段长（看连续性是否稳）
            int mn = 99;
            for (int k = -2; k <= 2; ++k) mn = std::min(mn, runLen[(x + k + Wcols) % Wcols]);
            std::printf("%4d  %2d  %6.3f  %s\n", deg, mn, peak[x],
                        mn == 0 ? "  <<< 完全无墨" : (mn <= 2 ? "  <-- 很薄" : ""));
        }
    }
    return 0;
}
