// ring_panel.cpp —— 7 个样本的环展开 + 墨量曲线 拼图
//
// 【用途】
// 交给人工圈注：把每个样本的环按极坐标拉直成一条横带（左端 0°，
// 右端 360°），下面配一条墨量曲线。这样"环在哪里变淡/断开"
// 在图上是一目了然的一条竖直缺口，而不是看椭圆环时的错觉。
//
// 用法：markloc_ringpanel <modelRoot> <tmpl.png> <out.png> <img...>
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
    if (argc < 5) {
        std::printf("usage: markloc_ringpanel <modelRoot> <tmpl.png> <out.png> <img...>\n");
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

    const int STRIP_W = 1200;   // 拉直带显示宽度（对应 360 度）
    const int STRIP_H = 90;     // 拉直带显示高度（对应半径方向）
    const int CURVE_H = 90;     // 墨量曲线高度
    const int PAD_X = 150;

    std::vector<cv::Mat> rows;
    for (int a = 4; a < argc; ++a) {
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
        cv::warpPolar(straight(roi), pol, cv::Size(nRadius, 720), cInRoi, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        cv::Mat polT;
        cv::transpose(pol, polT);
        pol = polT;   // rows=半径 cols=角度

        const int bLo = static_cast<int>(std::lround(A * 0.72));
        const int bHi = std::min(static_cast<int>(std::lround(A * 1.38)), pol.rows - 1);

        // 裁出半径带，横向重采样到 STRIP_W
        cv::Mat band = pol(cv::Rect(0, bLo, pol.cols, bHi - bLo + 1));
        cv::Mat strip;
        cv::resize(band, strip, cv::Size(STRIP_W, STRIP_H), 0, 0, cv::INTER_LINEAR);
        cv::Mat strip8;
        cv::normalize(strip, strip8, 0, 255, cv::NORM_MINMAX);
        strip8.convertTo(strip8, CV_8U);
        cv::Mat stripC;
        cv::cvtColor(strip8, stripC, cv::COLOR_GRAY2BGR);

        // 行画布
        cv::Mat row(STRIP_H + CURVE_H + 34, PAD_X + STRIP_W + 20, CV_8UC3,
                    cv::Scalar(18, 20, 24));
        stripC.copyTo(row(cv::Rect(PAD_X, 8, STRIP_W, STRIP_H)));

        // 墨量曲线
        const int cy0 = 8 + STRIP_H + 4;
        cv::line(row, cv::Point(PAD_X, cy0 + CURVE_H - 1),
                 cv::Point(PAD_X + STRIP_W, cy0 + CURVE_H - 1), cv::Scalar(60, 60, 60), 1);
        std::vector<double> prof(STRIP_W, 0.0);
        for (int x = 0; x < STRIP_W; ++x) {
            double mx = 0;
            for (int r = 0; r < strip.rows; ++r) mx = std::max(mx, (double)strip.at<float>(r, x));
            prof[x] = mx;
        }
        double pmax = 1e-6;
        for (double v : prof) pmax = std::max(pmax, v);
        for (int x = 0; x + 1 < STRIP_W; ++x) {
            const int y0 = cy0 + CURVE_H - 2 - static_cast<int>(prof[x] / pmax * (CURVE_H - 4));
            const int y1 = cy0 + CURVE_H - 2 - static_cast<int>(prof[x + 1] / pmax * (CURVE_H - 4));
            cv::line(row, cv::Point(PAD_X + x, y0), cv::Point(PAD_X + x + 1, y1),
                     cv::Scalar(120, 220, 120), 1);
        }
        // 阈值参考线（0.35 / 归一后）
        {
            // prof 用的是未归一化的 strip，阈值 0.35 相对 pmax 的比例
            const int y = cy0 + CURVE_H - 2 - static_cast<int>(0.35 / pmax * (CURVE_H - 4));
            if (y > cy0 && y < cy0 + CURVE_H)
                cv::line(row, cv::Point(PAD_X, y), cv::Point(PAD_X + STRIP_W, y),
                         cv::Scalar(70, 70, 180), 1);
        }
        // 角度刻度
        for (int t = 0; t <= 360; t += 45) {
            const int x = PAD_X + t * STRIP_W / 360;
            cv::line(row, cv::Point(x, 8), cv::Point(x, 8 + STRIP_H), cv::Scalar(60, 90, 60), 1);
            char b[16];
            std::snprintf(b, sizeof(b), "%d", t);
            cv::putText(row, b, cv::Point(x - 10, 8 + STRIP_H + 16), cv::FONT_HERSHEY_SIMPLEX,
                        0.38, cv::Scalar(120, 140, 160), 1);
        }
        // 左侧标签
        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::string nm = fn.toStdString();
        cv::putText(row, nm, cv::Point(6, 34), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 220, 255), 2);
        char pk[64];
        std::snprintf(pk, sizeof(pk), "peak %.2f", mval);
        cv::putText(row, pk, cv::Point(6, 58), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(160, 160, 160), 1);
        cv::putText(row, "band 0.72a-1.38a", cv::Point(6, 80), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    cv::Scalar(120, 120, 120), 1);

        rows.push_back(row);
        std::printf("%s  峰值%.3f\n", nm.c_str(), mval);
    }

    if (rows.empty()) {
        std::printf("没有可处理的图\n");
        return 1;
    }
    cv::Mat out;
    cv::vconcat(rows, out);
    // 顶部标题条
    cv::Mat title(46, out.cols, CV_8UC3, cv::Scalar(18, 20, 24));
    cv::putText(title,
                "ring unwrapped: left=0deg right=360deg   top strip=ink band   bottom curve=peak ink per angle   "
                "blue line=inkThr0.35",
                cv::Point(8, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 210, 220), 1);
    cv::vconcat(title, out, out);
    cv::imwrite(argv[3], out);
    std::printf("出图 %s  (%dx%d)\n", argv[3], out.cols, out.rows);
    return 0;
}
