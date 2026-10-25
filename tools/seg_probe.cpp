// seg_probe.cpp —— 只测环的一个扇形区段
//
// 【为什么这样测】
// 前九轮全部用"整圈平均"，缺陷只占环的一小段时会被稀释掉。
// 这次限定在指定角度区间内测量，信噪比直接提高一个数量级。
//
// 实测的角度对应（把展开图的角度标回原图确认过）：
//   极坐标 0 度   = 环的最右端
//   极坐标 90 度  = 环的底部
//   极坐标 150~180= 环的左下角
//   极坐标 240~270= 环的左上角   <-- 缺陷在这里
//
// 用法：markloc_segprobe <modelRoot> <tmpl.png> <角度起> <角度止> <image...>
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
    if (argc < 6) {
        std::printf("usage: markloc_segprobe <modelRoot> <tmpl.png> <fromDeg> <toDeg> <image...>\n");
        return 2;
    }
    const double fromDeg = std::atof(argv[3]);
    const double toDeg = std::atof(argv[4]);

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
    const int N = 720;

    std::printf("区段 %.0f~%.0f 度\n", fromDeg, toDeg);
    std::printf("%-18s %8s %8s %9s %8s %8s\n", "file", "peakMed", "peakMin", "ratio", "runMed",
                "runMin");
    for (int a = 5; a < argc; ++a) {
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

        const int pad = static_cast<int>(A * 0.45) + 8;
        cv::Rect roi(static_cast<int>(cx - A - pad), static_cast<int>(cy - A - pad),
                     static_cast<int>(2 * (A + pad)), static_cast<int>(2 * (A + pad)));
        roi &= cv::Rect(0, 0, straight.cols, straight.rows);
        const cv::Point2f cInRoi(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
        const double maxR = A + pad;
        const int nRadius = static_cast<int>(std::lround(maxR));
        cv::Mat pol;
        cv::warpPolar(straight(roi), pol, cv::Size(nRadius, N), cInRoi, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        cv::Mat polT;
        cv::transpose(pol, polT);
        pol = polT;

        const int bLo = static_cast<int>(std::lround(A * 0.72));
        const int bHi = std::min(static_cast<int>(std::lround(A * 1.38)), pol.rows - 1);

        // 只在这个角度区段内统计
        const int t0 = static_cast<int>(std::lround(fromDeg * N / 360.0));
        const int t1 = static_cast<int>(std::lround(toDeg * N / 360.0));
        std::vector<double> peaks;
        std::vector<double> runs;
        for (int t = t0; t < t1; ++t) {
            const int x = ((t % N) + N) % N;
            double pk = 0;
            int run = 0, best = 0;
            for (int r = bLo; r <= bHi; ++r) {
                const double v = pol.at<float>(r, x);
                pk = std::max(pk, v);
                if (v >= 0.35) {
                    if (++run > best) best = run;
                } else {
                    run = 0;
                }
            }
            peaks.push_back(pk);
            runs.push_back((double)best);
        }
        std::vector<double> sp = peaks, sr = runs;
        std::sort(sp.begin(), sp.end());
        std::sort(sr.begin(), sr.end());
        const double pm = sp[sp.size() / 2];
        const double pmin = sp.front();
        const double rm = sr[sr.size() / 2];
        const double rmin = sr.front();

        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::printf("%-18s %8.3f %8.3f %9.3f %8.1f %8.1f\n", fn.toStdString().substr(0, 18).c_str(),
                    pm, pmin, pmin / std::max(1e-6, pm), rm, rmin);
    }
    return 0;
}
