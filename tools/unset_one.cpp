// unset_one.cpp —— 单张 unset 图的 logo 检测过程图
//
// 【为什么需要】
// "缺口 2.0°" 这个数字看不出它在环上哪个位置、长什么样。
// 把它画出来：椭圆框、断口位置、放大后的局部，才能判断
// 这是真的断墨，还是采样抖动/外物遮挡。
//
// 用法：markloc_unsetone <modelRoot> <image> <tmpl.png> <out.png>
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
        std::printf("usage: markloc_unsetone <modelRoot> <image> <tmpl.png> <out.png>\n");
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
    if (!readGrayU8(QString::fromLocal8Bit(argv[3]), tmpl)) {
        std::printf("模板读取失败\n");
        return 1;
    }
    const double tplCx = 127.8, tplCy = 116.0;
    const double tplA = 132.1, tplB = 76.75;
    const double tplRotDeg = -13.78;

    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g)) {
        std::printf("读图失败\n");
        return 1;
    }

    cv::Mat res;
    cv::matchTemplate(g, tmpl, res, cv::TM_CCOEFF_NORMED);
    cv::Point mloc;
    double mval = 0;
    cv::minMaxLoc(res, nullptr, &mval, nullptr, &mloc);

    const double cx = mloc.x + tplCx;
    const double cy = mloc.y + tplCy;
    const double a = tplA, b = tplB;

    const int pad = static_cast<int>(a * 0.45) + 8;
    cv::Rect roi(static_cast<int>(cx - a - pad), static_cast<int>(cy - a - pad),
                 static_cast<int>(2 * (a + pad)), static_cast<int>(2 * (a + pad)));
    roi &= cv::Rect(0, 0, g.cols, g.rows);

    const cv::Mat fullSig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    const cv::Mat sig = fullSig(roi);

    chardet::RegionInfo rg;
    rg.shape = chardet::RegionShape::Ellipse;
    rg.ell.center = cv::Point2f(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
    rg.ell.a = static_cast<float>(a);
    rg.ell.b = static_cast<float>(b);
    rg.ell.angleDeg = static_cast<float>(tplRotDeg);
    rg.syncRectFromEll();

    chardet::PatternBreakParams P;
    const auto pb = chardet::analyzePatternBreak(sig, rg, model.config, P);

    std::printf("=== %s ===\n", QFileInfo(QString::fromLocal8Bit(argv[2])).fileName().toUtf8().constData());
    std::printf("定位 (%d,%d) 峰值%.4f  椭圆中心(%.1f,%.1f) a=%.1f b=%.1f 角%.2f°\n", mloc.x,
                mloc.y, mval, cx, cy, a, b, tplRotDeg);
    std::printf("覆盖%.4f 断口%d 缺口%.2f° 断点分%.0f 档位%s 判定%s\n", pb.coverage, pb.breaks,
                pb.maxGapAngle, pb.score, chardet::patternBandName(pb.band),
                pb.flagged ? "NG" : "OK");
    std::printf("断口位置（全图）:\n");
    for (const auto& pt : pb.breakAt)
        std::printf("  (%.1f, %.1f)\n", pt.x + roi.x, pt.y + roi.y);

    // 出图：原图灰度 + 椭圆 + 断口圈
    cv::Mat view;
    cv::cvtColor(g, view, cv::COLOR_GRAY2BGR);
    cv::ellipse(view,
                cv::RotatedRect(cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
                                cv::Size2f(static_cast<float>(a * 2), static_cast<float>(b * 2)),
                                static_cast<float>(tplRotDeg)),
                cv::Scalar(0, 220, 0), 2, cv::LINE_AA);
    for (const auto& pt : pb.breakAt) {
        const cv::Point c(cvRound(pt.x + roi.x), cvRound(pt.y + roi.y));
        cv::circle(view, c, 14, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        // 局部放大：断口周围 60x60
        cv::Rect zm(c.x - 30, c.y - 30, 60, 60);
        zm &= cv::Rect(0, 0, g.cols, g.rows);
        if (zm.area() > 0) {
            cv::Mat z;
            cv::resize(g(zm), z, cv::Size(240, 240), 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(z, z, cv::COLOR_GRAY2BGR);
            cv::circle(z, cv::Point(120, 120), 56, cv::Scalar(0, 0, 255), 2);
            const int zx = view.cols - 250, zy = 10;
            if (zx > 0 && zy + 240 < view.rows) z.copyTo(view(cv::Rect(zx, zy, 240, 240)));
        }
    }
    // 裁到椭圆附近
    cv::RotatedRect rr(cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
                       cv::Size2f(static_cast<float>(a * 2), static_cast<float>(b * 2)),
                       static_cast<float>(tplRotDeg));
    cv::Rect crop = rr.boundingRect();
    crop = cv::Rect(crop.x - 40, crop.y - 40, crop.width + 80, crop.height + 80);
    crop &= cv::Rect(0, 0, view.cols, view.rows);
    if (crop.area() > 0) view = view(crop);
    cv::imwrite(argv[4], view);
    std::printf("出图 %s\n", argv[4]);
    return 0;
}
