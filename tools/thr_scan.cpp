// thr_scan.cpp —— 扫描 inkThr 对缺口的敏感性
//
// 【要解决的问题】
// 老板明确："环变淡但没断，不能用角度去看"。
// 现有 inkThr=0.35 太高 —— 环上淡到 40% 强度的地方仍算"有墨"，
// 于是缺口为 0，缺陷被抹平。
//
// 这个工具把 inkThr 从 0.35 一路降到 0.05，看每个样本的缺口怎么变。
// 如果"变淡"是真的缺陷，那么降低阈值后它的缺口应该明显大于正常样本。
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
        std::printf("usage: markloc_thrscan <modelRoot> <tmpl.png> <image...>\n");
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

    const double thrs[] = {0.35, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05};

    // 展开前的灰度阈值，不填则 0（不启用）
    double pre = 0.0;
    for (int i = 3; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--pre")) pre = std::atof(argv[i + 1]);

    std::printf("%-14s", "file");
    for (double t : thrs) std::printf(" %6.2f", t);
    std::printf("   <- inkThr 下的 maxGapAngle(度)\n");

    for (int a = 3; a < argc; ++a) {
        // 跳过 --pre 及它的值，否则会被当成图片路径
        if (!std::strcmp(argv[a], "--pre")) {
            ++a;
            continue;
        }
        cv::Mat g;
        if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g)) continue;

        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        const double cx = mloc.x + tplCx, cy = mloc.y + tplCy;
        const double A = tplA, B = tplB;

        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

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
        // 展开前的灰度阈值（命令行可用 --pre X 指定）
        if (pre > 0.0) cv::threshold(straight, straight, pre, 0.0, cv::THRESH_TOZERO);

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
        const int need = 3;

        // 每个阈值下算一遍：有墨序列 -> 最长连续无墨段
        auto gapAt = [&](double thr) {
            std::vector<char> on(Wcols, 0);
            for (int x = 0; x < Wcols; ++x) {
                int run = 0, best = 0;
                for (int r = bLo; r <= bHi; ++r) {
                    if (pol.at<float>(r, x) >= thr) {
                        if (++run > best) best = run;
                    } else {
                        run = 0;
                    }
                }
                on[x] = best >= need ? 1 : 0;
            }
            int startAt = -1;
            for (int t = 0; t < Wcols; ++t)
                if (on[t]) { startAt = t; break; }
            if (startAt < 0) return 360.0;
            int bestRun = 0, cur = 0;
            for (int k = 1; k <= Wcols; ++k) {
                const int t = (startAt + k) % Wcols;
                if (!on[t]) {
                    if (++cur > bestRun) bestRun = cur;
                } else {
                    cur = 0;
                }
            }
            return 360.0 * bestRun / Wcols;
        };

        std::printf("%-14s", QFileInfo(QString::fromLocal8Bit(argv[a])).fileName().toStdString().c_str());
        for (double t : thrs) std::printf(" %6.2f", gapAt(t));
        std::printf("\n");
    }
    return 0;
}
