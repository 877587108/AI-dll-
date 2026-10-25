// panel_hole.cpp —— 生成"挖了洞"的对比图，用来看断口长什么样
//
// 直接复用 logo_panel 的绘制逻辑，只是在画图之前先在环上挖一个洞。
// 目的：给人一个"NG 长什么样"的参照，和完好的那张并排看。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    if (argc < 5) {
        std::printf("usage: markloc_panelhole <modelDir> <image> <out.png> <holeDeg>\n");
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
        std::printf("读图失败\n");
        return 1;
    }
    const double holeDeg = std::atof(argv[4]);

    cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

    // 挖洞：用与检测内部完全相同的拉直变换，按极角与半径挑像素清零
    const double aA = std::max(pat.ell.a, pat.ell.b), bB = std::min(pat.ell.a, pat.ell.b);
    const double kScale = aA / std::max(1.0, bB);
    const double rot = pat.ell.angleDeg * CV_PI / 180.0;
    const double cR = std::cos(-rot), sR = std::sin(-rot);
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -pat.ell.center.x, 0, 1, -pat.ell.center.y, 0, 0, 1);
    cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
    cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, pat.ell.center.x, 0, 1, pat.ell.center.y, 0, 0, 1);
    const cv::Mat M3 = T2 * Sc * Rm * T1;

    {
        const double halfW = holeDeg / 2.0;
        for (int y = 0; y < sig.rows; ++y)
            for (int x = 0; x < sig.cols; ++x) {
                const double rx = M3.at<double>(0, 0) * x + M3.at<double>(0, 1) * y +
                                  M3.at<double>(0, 2);
                const double ry = M3.at<double>(1, 0) * x + M3.at<double>(1, 1) * y +
                                  M3.at<double>(1, 2);
                const double rr = std::hypot(rx - pat.ell.center.x, ry - pat.ell.center.y);
                if (std::abs(rr - aA) > 20) continue;
                const double angDeg =
                    std::atan2(ry - pat.ell.center.y, rx - pat.ell.center.x) * 180.0 / CV_PI;
                double dd = std::fmod(std::abs(angDeg), 360.0);
                if (dd > 180) dd = 360 - dd;
                if (dd <= halfW) sig.at<float>(y, x) = 0.0f;
            }
    }

    chardet::PatternBreakParams P;
    const auto pb = chardet::analyzePatternBreak(sig, pat, model.config, P);
    std::printf("挖 %.1f° 后：覆盖%.4f 断口%d 最长缺口%.2f° 断点分%.0f 档位%s 判定%s\n", holeDeg,
                pb.coverage, pb.breaks, pb.maxGapAngle, pb.score,
                chardet::patternBandName(pb.band), pb.flagged ? "NG" : "OK");

    // ---- 拼图（与 logo_panel 同样的四块）
    const int pad = 60;
    cv::Rect crop(static_cast<int>(pat.ell.center.x - aA - pad),
                  static_cast<int>(pat.ell.center.y - aA - pad),
                  static_cast<int>(2 * (aA + pad)), static_cast<int>(2 * (aA + pad)));
    crop &= cv::Rect(0, 0, g.cols, g.rows);

    cv::Mat p1;
    sig(crop).convertTo(p1, CV_8U, 255.0 / std::max(0.05, model.config.strokeHi * 2.2));
    cv::cvtColor(p1, p1, cv::COLOR_GRAY2BGR);
    {
        const double ca = std::cos(rot), sa = std::sin(rot);
        std::vector<cv::Point> pts;
        for (int t = 0; t <= 180; ++t) {
            const double th = 2.0 * CV_PI * t / 180.0;
            const double u = pat.ell.a * std::cos(th), v = pat.ell.b * std::sin(th);
            pts.emplace_back(
                static_cast<int>(std::lround(pat.ell.center.x + u * ca - v * sa)) - crop.x,
                static_cast<int>(std::lround(pat.ell.center.y + u * sa + v * ca)) - crop.y);
        }
        cv::polylines(p1, pts, true, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    cv::putText(p1, "1 signal + ellipse (hole punched)", cv::Point(6, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    cv::Mat M;
    M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
    cv::Mat straight;
    cv::warpAffine(sig, straight, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);
    cv::Mat p2;
    straight(crop).convertTo(p2, CV_8U, 255.0 / std::max(0.05, model.config.strokeHi * 2.2));
    cv::cvtColor(p2, p2, cv::COLOR_GRAY2BGR);

    const int polPad = static_cast<int>(aA * 0.45) + 8;
    const cv::Rect roi =
        cv::Rect(static_cast<int>(pat.ell.center.x - aA - polPad),
                 static_cast<int>(pat.ell.center.y - aA - polPad),
                 static_cast<int>(2 * (aA + polPad)), static_cast<int>(2 * (aA + polPad))) &
        cv::Rect(0, 0, sig.cols, sig.rows);
    const cv::Point2f cInRoi(static_cast<float>(pat.ell.center.x - roi.x),
                             static_cast<float>(pat.ell.center.y - roi.y));
    const double maxR = aA + polPad;
    const int nAngle = 720;
    const int nRadius = static_cast<int>(std::lround(maxR));
    cv::Mat pol;
    cv::warpPolar(straight(roi), pol, cv::Size(nRadius, nAngle), cInRoi, maxR,
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    cv::Mat polT;
    cv::transpose(pol, polT);
    cv::Mat p3;
    polT.convertTo(p3, CV_8U, 255.0 / std::max(0.05, model.config.strokeHi * 2.2));
    cv::cvtColor(p3, p3, cv::COLOR_GRAY2BGR);
    {
        // 画两样东西：半径 a 的位置（环应在的半径）、检测出的断口区间
        const int aRow = static_cast<int>(std::lround(aA));
        cv::line(p3, cv::Point(0, aRow), cv::Point(p3.cols, aRow), cv::Scalar(0, 0, 255), 1);
        for (const auto& pt : pb.breakAt) {
            const double ang = std::atan2(pt.y - pat.ell.center.y, pt.x - pat.ell.center.x);
            int xAng = static_cast<int>(std::lround((ang < 0 ? ang + 2 * CV_PI : ang) /
                                                     (2 * CV_PI) * p3.cols));
            xAng = std::clamp(xAng, 0, p3.cols - 1);
            cv::line(p3, cv::Point(xAng, 0), cv::Point(xAng, p3.rows), cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::putText(p3, "3 polar unwrap", cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    const int H = 180, W = p3.cols;
    cv::Mat p4(H, W, CV_8UC3, cv::Scalar(20, 20, 20));
    std::vector<double> curvePk(W, 0.0), curveOn(W, 0.0);
    const int aRow2 = std::clamp(static_cast<int>(std::lround(aA)), 0, polT.rows - 1);
    const int tol = std::max(4, static_cast<int>(aA * 0.10));
    for (int x = 0; x < W; ++x) {
        double pk = 0;
        for (int r = std::max(0, aRow2 - tol); r <= std::min(polT.rows - 1, aRow2 + tol); ++r)
            pk = std::max(pk, static_cast<double>(polT.at<float>(r, x)));
        curvePk[x] = pk;
        curveOn[x] = pk >= model.config.strokeHi ? 1.0 : 0.0;
    }
    const double pkMax = std::max(0.01, *std::max_element(curvePk.begin(), curvePk.end()));
    auto toY = [&](double v) { return H - 1 - static_cast<int>(std::lround(v / pkMax * (H - 30))); };
    for (int x = 1; x < W; ++x) {
        cv::line(p4, cv::Point(x - 1, toY(curvePk[x - 1])), cv::Point(x, toY(curvePk[x])),
                 cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
        if (curveOn[x] > 0.5)
            cv::line(p4, cv::Point(x, H - 8), cv::Point(x, H - 1), cv::Scalar(0, 255, 0), 1);
    }
    cv::line(p4, cv::Point(0, toY(model.config.strokeHi)), cv::Point(W, toY(model.config.strokeHi)),
             cv::Scalar(120, 120, 120), 1);
    cv::putText(p4, "4 ink vs angle  (green=on-ink, gap shows as rise)", cv::Point(6, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    auto fitW = [](cv::Mat& m, int w) {
        const double s = static_cast<double>(w) / m.cols;
        cv::resize(m, m, cv::Size(w, static_cast<int>(m.rows * s)));
    };
    const int colW = 520;
    fitW(p1, colW);
    fitW(p2, colW);
    fitW(p3, colW);
    fitW(p4, colW);

    const int rightH = p2.rows + p3.rows + p4.rows + 8;
    const int totalH = std::max(p1.rows, rightH);
    if (p1.rows != totalH) {
        const double s = static_cast<double>(totalH) / p1.rows;
        cv::resize(p1, p1, cv::Size(static_cast<int>(p1.cols * s), totalH));
    }
    cv::Mat right(totalH, colW, CV_8UC3, cv::Scalar(0, 0, 0));
    int y = 0;
    p2.copyTo(right(cv::Rect(0, y, p2.cols, p2.rows)));
    y += p2.rows + 4;
    p3.copyTo(right(cv::Rect(0, y, p3.cols, p3.rows)));
    y += p3.rows + 4;
    p4.copyTo(right(cv::Rect(0, y, p4.cols, p4.rows)));

    cv::Mat panel(totalH, p1.cols + 8 + colW, CV_8UC3, cv::Scalar(0, 0, 0));
    p1.copyTo(panel(cv::Rect(0, 0, p1.cols, totalH)));
    right.copyTo(panel(cv::Rect(p1.cols + 8, 0, colW, totalH)));

    cv::Mat bar(34, panel.cols, CV_8UC3, cv::Scalar(30, 30, 30));
    char head[256];
    std::snprintf(head, sizeof(head),
                  "hole=%.1fdeg  cover=%.4f  maxGap=%.2f  score=%.0f  verdict=%s", holeDeg,
                  pb.coverage, pb.maxGapAngle, pb.score, pb.flagged ? "NG" : "OK");
    cv::putText(bar, head, cv::Point(6, 23), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                pb.flagged ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    cv::Mat out;
    cv::vconcat(bar, panel, out);
    cv::imwrite(argv[3], out);
    std::printf("已写出 %s\n", argv[3]);
    return 0;
}
