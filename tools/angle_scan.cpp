// angle_scan.cpp —— 逐角度扫描，精确定位断点位置
//
// 【背景】
// 用户指出缺陷在环的左上角（极坐标约 240~270 度）。
// 但把整个 240~270 区段拿来统计，正常料在该区段也偏低，
// 说明区间取得太宽，混入了正常波动。
//
// 这个工具把每个角度单独输出，看 ngv2 的谷底究竟落在哪几个角度，
// 再据此收窄区间。
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

// 取某张图的极坐标展开（rows=半径 cols=角度）
static bool polarOf(const chardet::Model& model, const cv::Mat& tmpl, const cv::Mat& g,
                    cv::Mat& pol, int& bLo, int& bHi) {
    const double tplCx = 127.8, tplCy = 116.0, tplA = 132.1, tplB = 76.75, tplRot = -13.78;
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
    if (roi.area() <= 0) return false;
    const cv::Point2f cInRoi(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
    const double maxR = A + pad;
    const int nRadius = static_cast<int>(std::lround(maxR));
    cv::Mat p;
    cv::warpPolar(straight(roi), p, cv::Size(nRadius, 720), cInRoi, maxR,
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    cv::Mat pt;
    cv::transpose(p, pt);
    pol = pt;
    bLo = static_cast<int>(std::lround(A * 0.72));
    bHi = std::min(static_cast<int>(std::lround(A * 1.38)), pol.rows - 1);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("usage: markloc_anglescan <modelRoot> <tmpl.png> <ngv2Dir> <normalDir>\n");
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
    const int N = 720;

    auto profileOf = [&](const QString& p, std::vector<double>& prof) {
        cv::Mat g, pol;
        int bLo = 0, bHi = 0;
        if (!readGrayU8(p, g)) return false;
        if (!polarOf(model, tmpl, g, pol, bLo, bHi)) return false;
        prof.assign(N, 0.0);
        for (int x = 0; x < N; ++x) {
            double pk = 0;
            for (int r = bLo; r <= bHi; ++r) pk = std::max(pk, (double)pol.at<float>(r, x));
            prof[x] = pk;
        }
        return true;
    };

    // 读 ngv2 的 _00
    std::vector<std::vector<double>> ng;
    QDir bd(QString::fromLocal8Bit(argv[3]));
    for (const QFileInfo& fi : bd.entryInfoList({QStringLiteral("*_00.bmp"), QStringLiteral("*_00.tif")},
                                                QDir::Files, QDir::Name)) {
        std::vector<double> p;
        if (profileOf(fi.absoluteFilePath(), p)) ng.push_back(p);
    }
    std::vector<std::vector<double>> no;
    QDir nd(QString::fromLocal8Bit(argv[4]));
    for (const QFileInfo& fi : nd.entryInfoList({QStringLiteral("*_00.tif")}, QDir::Files, QDir::Name)) {
        std::vector<double> p;
        if (profileOf(fi.absoluteFilePath(), p)) no.push_back(p);
    }
    std::printf("ngv2 %d 张，正常 %d 张\n\n", (int)ng.size(), (int)no.size());

    // 逐角度比较：ngv2 的均值曲线 vs 正常料的均值曲线
    std::printf("角度  ngv2均值  ngv2最小  正常均值  正常p05  差值(正常-ngv2)\n");
    for (int deg = 0; deg < 360; deg += 10) {
        double sN = 0, mN = 1e9;
        for (const auto& p : ng) {
            double s = 0;
            for (int k = 0; k < N / 360 * 10; ++k) s += p[(deg * N / 360 + k) % N];
            const double v = s / (N / 360 * 10);
            sN += v;
            mN = std::min(mN, v);
        }
        const double aN = sN / ng.size();
        std::vector<double> nv;
        for (const auto& p : no) {
            double s = 0;
            for (int k = 0; k < N / 360 * 10; ++k) s += p[(deg * N / 360 + k) % N];
            nv.push_back(s / (N / 360 * 10));
        }
        std::sort(nv.begin(), nv.end());
        const double aO = [&] {
            double s = 0;
            for (double v : nv) s += v;
            return s / nv.size();
        }();
        const double p05 = nv[static_cast<size_t>(nv.size() * 0.05)];
        std::printf("%4d  %8.3f  %8.3f  %8.3f  %8.3f   %+8.3f\n", deg, aN, mN, aO, p05, aO - aN);
    }
    return 0;
}
