// align_panel.cpp —— 单张图的对齐对比拼图
//
// 【为什么需要】
// 判 NG 之后第一个要回答的问题是"到底是环断了，还是框没对上"。
// 数字（缺口几度、覆盖率多少）分不出这两种情况，必须把对齐过程摊开看。
//
// 拼四块，都对准同一张料：
//   ① 原始图 + 三个椭圆：
//        红 = 模型里的原位（652.6,433.1，只对旧规格 1688x1460 有意义）
//        黄 = 模板匹配定位后的椭圆（这批 1496x1496 实际用的）
//        绿 = 判据真正采样的椭圆
//   ② 拉直后的环：椭圆被拉成正圆，环应该是完美的圆
//   ③ 极坐标展开：环应该是一条水平带，断口就是带上的缺口
//   ④ 周向墨量曲线：直接看"哪些角度没墨"
//
// 用法：markloc_alignpanel <modelRoot> <image> <tmpl.png> <out.png> [ngDeg]
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

static void label(cv::Mat& img, const char* text, cv::Point at, cv::Scalar col = cv::Scalar(0, 255, 255)) {
    cv::putText(img, text, at, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, text, at, cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv::LINE_AA);
}

// 把任意尺寸的图贴进固定画布，保持长宽比
static cv::Mat fitInto(const cv::Mat& src, int W, int H) {
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(25, 25, 25));
    if (src.empty()) return canvas;
    const double s = std::min(static_cast<double>(W) / src.cols, static_cast<double>(H) / src.rows);
    const int w = std::max(1, static_cast<int>(src.cols * s));
    const int h = std::max(1, static_cast<int>(src.rows * s));
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(w, h), 0, 0, s < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
    tmp.copyTo(canvas(cv::Rect((W - w) / 2, (H - h) / 2, w, h)));
    return canvas;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("usage: markloc_alignpanel <modelRoot> <image> <tmpl.png> <out.png> [ngDeg]\n");
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
    const double tplCx = 127.8, tplCy = 116.0, tplA = 132.1, tplB = 76.75, tplRot = -13.78;
    const double ngDeg = argc > 5 ? std::atof(argv[5]) : 1.0;

    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g)) {
        std::printf("读图失败\n");
        return 1;
    }

    // ---- 定位
    cv::Mat res;
    cv::matchTemplate(g, tmpl, res, cv::TM_CCOEFF_NORMED);
    cv::Point mloc;
    double mval = 0;
    cv::minMaxLoc(res, nullptr, &mval, nullptr, &mloc);
    const double cx = mloc.x + tplCx, cy = mloc.y + tplCy;
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
    rg.ell.angleDeg = static_cast<float>(tplRot);
    rg.syncRectFromEll();

    chardet::PatternBreakParams P;
    P.minGapAngle = ngDeg;
    P.minScore = ngDeg * 100.0;
    const auto pb = chardet::analyzePatternBreak(sig, rg, model.config, P);

    std::printf("=== %s ===\n", QFileInfo(QString::fromLocal8Bit(argv[2])).fileName().toUtf8().constData());
    std::printf("模板匹配 (%d,%d) 峰值%.4f\n", mloc.x, mloc.y, mval);
    std::printf("椭圆 中心(%.1f,%.1f) a=%.1f b=%.1f 角%.2f° 离心率%.3f\n", cx, cy, a, b, tplRot,
                std::sqrt(1.0 - (b * b) / (a * a)));
    std::printf("覆盖%.4f 断口%d 缺口%.2f° 断点分%.0f 判定%s 阈值%.1f°\n", pb.coverage, pb.breaks,
                pb.maxGapAngle, pb.score, pb.flagged ? "NG" : "OK", ngDeg);
    std::printf("断口位置: ");
    for (const auto& pt : pb.breakAt) std::printf("(%.1f,%.1f) ", pt.x + roi.x, pt.y + roi.y);
    std::printf("\n");

    // ---- ① 原图 + 三个椭圆
    cv::Mat p1;
    cv::cvtColor(g, p1, cv::COLOR_GRAY2BGR);
    // 红：模型原位。模型是 1688x1460 的坐标系，这里只作为"没有重新定位会采哪"的参照。
    if (!model.regions.empty()) {
        for (const auto& r : model.regions) {
            if (!r.isPattern()) continue;
            cv::ellipse(p1,
                        cv::RotatedRect(cv::Point2f(r.ell.center.x, r.ell.center.y),
                                        cv::Size2f(r.ell.a * 2, r.ell.b * 2), r.ell.angleDeg),
                        cv::Scalar(60, 60, 230), 2, cv::LINE_AA);
            break;
        }
    }
    // 黄：模板匹配定位后的椭圆（本图真正用的位置）
    cv::ellipse(p1, cv::RotatedRect(cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
                                    cv::Size2f(static_cast<float>(a * 2), static_cast<float>(b * 2)),
                                    static_cast<float>(tplRot)),
                cv::Scalar(0, 230, 230), 2, cv::LINE_AA);
    // 红圈：断口
    for (const auto& pt : pb.breakAt)
        cv::circle(p1, cv::Point(cvRound(pt.x + roi.x), cvRound(pt.y + roi.y)), 12,
                   cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    // ---- ② 拉直后的环（自己复算，与判据同一套矩阵）
    cv::Mat straight;
    {
        const double kScale = a / std::max(1.0, b);
        const double rot = tplRot * CV_PI / 180.0;
        const double cR = std::cos(-rot), sR = std::sin(-rot);
        cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -rg.ell.center.x, 0, 1, -rg.ell.center.y, 0, 0, 1);
        cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
        cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
        cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, rg.ell.center.x, 0, 1, rg.ell.center.y, 0, 0, 1);
        cv::Mat M3 = T2 * Sc * Rm * T1;
        cv::Mat M;
        M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
        cv::warpAffine(sig, straight, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);
    }
    cv::Mat p2;
    {
        // 裁环附近并归一化显示
        const int r = static_cast<int>(a) + 30;
        cv::Rect cr(static_cast<int>(rg.ell.center.x) - r, static_cast<int>(rg.ell.center.y) - r,
                    2 * r, 2 * r);
        cr &= cv::Rect(0, 0, straight.cols, straight.rows);
        cv::Mat v;
        cv::normalize(straight(cr), v, 0, 255, cv::NORM_MINMAX);
        v.convertTo(v, CV_8U);
        cv::cvtColor(v, p2, cv::COLOR_GRAY2BGR);
        // 画上应该重合的理想圆
        cv::circle(p2, cv::Point(rg.ell.center.x - cr.x, rg.ell.center.y - cr.y),
                   static_cast<int>(a), cv::Scalar(0, 220, 0), 1, cv::LINE_AA);
    }

    // ---- ③ 极坐标展开
    cv::Mat pol;
    {
        const double rot = tplRot * CV_PI / 180.0;
        const double cR = std::cos(-rot), sR = std::sin(-rot);
        const double kScale = a / std::max(1.0, b);
        cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -rg.ell.center.x, 0, 1, -rg.ell.center.y, 0, 0, 1);
        cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
        cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
        cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, rg.ell.center.x, 0, 1, rg.ell.center.y, 0, 0, 1);
        cv::Mat M3 = T2 * Sc * Rm * T1;
        cv::Mat M;
        M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
        cv::Mat straight2;
        cv::warpAffine(sig, straight2, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);

        const int r = static_cast<int>(a) + pad;
        cv::Rect cr(static_cast<int>(rg.ell.center.x) - r, static_cast<int>(rg.ell.center.y) - r,
                    2 * r, 2 * r);
        cr &= cv::Rect(0, 0, straight2.cols, straight2.rows);
        const cv::Point2f cInRoi(static_cast<float>(rg.ell.center.x - cr.x),
                                 static_cast<float>(rg.ell.center.y - cr.y));
        const double maxR = a + pad;
        const int nRadius = static_cast<int>(std::lround(maxR));
        cv::warpPolar(straight2(cr), pol, cv::Size(nRadius, 720), cInRoi, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        cv::Mat polT;
        cv::transpose(pol, polT);
        pol = polT;  // rows=半径 cols=角度
    }
    cv::Mat p3;
    {
        cv::Mat v;
        cv::normalize(pol, v, 0, 255, cv::NORM_MINMAX);
        v.convertTo(v, CV_8U);
        cv::cvtColor(v, p3, cv::COLOR_GRAY2BGR);
        // 标出半径带上下界
        const int bLo = static_cast<int>(std::lround(a * 0.75));
        const int bHi = std::min(static_cast<int>(std::lround(a * 1.35)), pol.rows - 1);
        cv::line(p3, cv::Point(0, bLo), cv::Point(p3.cols - 1, bLo), cv::Scalar(0, 200, 255), 1);
        cv::line(p3, cv::Point(0, bHi), cv::Point(p3.cols - 1, bHi), cv::Scalar(0, 200, 255), 1);
    }

    // ---- ④ 周向墨量曲线
    cv::Mat p4(400, 900, CV_8UC3, cv::Scalar(25, 25, 25));
    {
        std::vector<double> on(pol.cols, 0.0);
        const int bLo = static_cast<int>(std::lround(a * 0.75));
        const int bHi = std::min(static_cast<int>(std::lround(a * 1.35)), pol.rows - 1);
        for (int x = 0; x < pol.cols; ++x) {
            double mx = 0;
            for (int r = bLo; r <= bHi; ++r) mx = std::max(mx, static_cast<double>(pol.at<float>(r, x)));
            on[x] = mx;
        }
        cv::line(p4, cv::Point(40, 340), cv::Point(880, 340), cv::Scalar(80, 80, 80), 1);
        for (int x = 0; x + 1 < static_cast<int>(on.size()); ++x) {
            const int x0 = 40 + x * 840 / static_cast<int>(on.size());
            const int x1 = 40 + (x + 1) * 840 / static_cast<int>(on.size());
            const int y0 = 340 - static_cast<int>(std::min(1.0, on[x]) * 280);
            const int y1 = 340 - static_cast<int>(std::min(1.0, on[x + 1]) * 280);
            const bool ok0 = on[x] >= 0.35;
            cv::line(p4, cv::Point(x0, y0), cv::Point(x1, y1),
                     ok0 ? cv::Scalar(120, 220, 120) : cv::Scalar(0, 0, 255), 1);
        }
        label(p4, "ink vs angle (green=on, red=gap, gray=thr0.35)", cv::Point(40, 26));
        label(p4, "0deg", cv::Point(36, 370), cv::Scalar(160, 160, 160));
        label(p4, "180deg", cv::Point(440, 370), cv::Scalar(160, 160, 160));
        label(p4, "360deg", cv::Point(840, 370), cv::Scalar(160, 160, 160));
        char buf[160];
        std::snprintf(buf, sizeof(buf), "cover=%.4f  maxGap=%.2fdeg  score=%.0f  brk=%d  %s",
                      pb.coverage, pb.maxGapAngle, pb.score, pb.breaks, pb.flagged ? "NG" : "OK");
        label(p4, buf, cv::Point(40, 396), pb.flagged ? cv::Scalar(0, 0, 255) : cv::Scalar(120, 220, 120));
    }

    // ---- 拼图 2x2
    const int TW = 900, TH = 600;
    cv::Mat canvas(TH * 2 + 10, TW * 2 + 10, CV_8UC3, cv::Scalar(0, 0, 0));
    fitInto(p1, TW, TH).copyTo(canvas(cv::Rect(0, 0, TW, TH)));
    fitInto(p2, TW, TH).copyTo(canvas(cv::Rect(TW + 10, 0, TW, TH)));
    fitInto(p3, TW, TH).copyTo(canvas(cv::Rect(0, TH + 10, TW, TH)));
    fitInto(p4, TW, TH).copyTo(canvas(cv::Rect(TW + 10, TH + 10, TW, TH)));

    label(canvas, "1. raw + ellipses (red=model, yellow=located, ring=break)", cv::Point(10, 28));
    label(canvas, "2. straightened (green=ideal circle)", cv::Point(TW + 20, 28));
    label(canvas, "3. polar unwrap (yellow=band edges)", cv::Point(10, TH + 38));
    label(canvas, "4. ink vs angle", cv::Point(TW + 20, TH + 38));

    cv::imwrite(argv[4], canvas);
    std::printf("出图 %s\n", argv[4]);
    return 0;
}
