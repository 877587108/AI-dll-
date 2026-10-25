// logo_panel.cpp —— 单张 logo 的检测过程对比图
//
// 【为什么需要这张图】
// 前面所有结论都是数字（覆盖 0.999、缺口 0°、断点分 0），
// 但数字看不出"检测器到底在看哪里、看到了什么"。
// 把中间状态摊开成一张拼图，人一眼就能判断流程对不对：
//
//   ① 原图 + 椭圆框     —— 框有没有套住环
//   ② 拉直后的环        —— 椭圆有没有被拉成正圆
//   ③ 极坐标展开        —— 环是不是一条水平的带
//   ④ 周向墨量曲线      —— 缺口在曲线上长什么样
//
// 四张图对准同一张料，看它们互相印证。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
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

// 在图上写中文很麻烦（Hershey 没有中文字形），所以标签全用 ASCII 编号。
static void label(cv::Mat& img, const char* text, cv::Point at) {
    cv::putText(img, text, at, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1,
                cv::LINE_AA);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: markloc_logopanel <modelDir> <image> <out.png>\n");
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
        std::printf("读图失败或尺寸不符（模型 %dx%d）\n", model.freq.cols, model.freq.rows);
        return 1;
    }

    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

    // 跑真正的检测
    chardet::PatternBreakParams P;
    const auto pb = chardet::analyzePatternBreak(sig, pat, model.config, P);

    std::printf("=== %s ===\n", argv[2]);
    std::printf("椭圆 中心(%.1f,%.1f) 半轴%.1f×%.1f 角%.2f° 离心率%.3f\n", pat.ell.center.x,
                pat.ell.center.y, pat.ell.a, pat.ell.b, pat.ell.angleDeg, pat.ell.eccentricity());
    std::printf("结果 覆盖%.4f 断口%d 最长缺口%.2f° 断点分%.0f 档位%s 判定%s\n", pb.coverage,
                pb.breaks, pb.maxGapAngle, pb.score, chardet::patternBandName(pb.band),
                pb.flagged ? "NG" : "OK");
    std::printf("对准 半径起伏%.1fpx 墨线宽%.1fpx %s\n", pb.radiusSwing, pb.radiusLineWidth,
                pb.alignOk ? "OK" : "未对准（结论作废）");

    // ---- ① 原图 + 椭圆
    const int pad = 60;
    cv::Rect crop(static_cast<int>(pat.ell.center.x - pat.ell.a - pad),
                  static_cast<int>(pat.ell.center.y - pat.ell.a - pad),
                  static_cast<int>(2 * (pat.ell.a + pad)), static_cast<int>(2 * (pat.ell.a + pad)));
    crop &= cv::Rect(0, 0, g.cols, g.rows);
    cv::Mat p1;
    cv::cvtColor(g(crop).clone(), p1, cv::COLOR_GRAY2BGR);
    {
        const double ca = std::cos(pat.ell.angleDeg * CV_PI / 180.0);
        const double sa = std::sin(pat.ell.angleDeg * CV_PI / 180.0);
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
    label(p1, "1 original + ellipse", cv::Point(6, 22));

    // ---- ② 拉直成正圆
    const double aA = std::max(pat.ell.a, pat.ell.b), bB = std::min(pat.ell.a, pat.ell.b);
    const double kScale = aA / std::max(1.0, bB);
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

    cv::Mat straightV;
    straight.convertTo(straightV, CV_8U, 255.0 / std::max(0.05, model.config.strokeHi * 2.2));
    cv::Mat p2;
    cv::cvtColor(straightV(crop).clone(), p2, cv::COLOR_GRAY2BGR);

    // ---- ③ 极坐标展开
    // warpPolar 的 dsize：width=半径数，height=角度数。转置后 rows=半径、cols=角度。
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
    cv::Mat polV;
    polT.convertTo(polV, CV_8U, 255.0 / std::max(0.05, model.config.strokeHi * 2.2));
    cv::Mat p3;
    cv::cvtColor(polV, p3, cv::COLOR_GRAY2BGR);
    // 标出半径 a 那条线（环应该在这里）
    {
        const int aRow = static_cast<int>(std::lround(aA));
        if (aRow >= 0 && aRow < p3.rows) cv::line(p3, cv::Point(0, aRow), cv::Point(p3.cols, aRow),
                                                  cv::Scalar(0, 0, 255), 1);
    }

    // ---- ④ 周向墨量曲线
    const int H = 180, W = p3.cols;
    cv::Mat p4(H, W, CV_8UC3, cv::Scalar(20, 20, 20));
    // 该角度在半径 a 附近的峰值强度
    std::vector<double> curvePk(W, 0.0);
    std::vector<double> curveOn(W, 0.0);
    {
        const int aRow = std::clamp(static_cast<int>(std::lround(aA)), 0, polT.rows - 1);
        const int tol = std::max(4, static_cast<int>(aA * 0.10));
        for (int x = 0; x < W; ++x) {
            double pk = 0;
            for (int r = std::max(0, aRow - tol); r <= std::min(polT.rows - 1, aRow + tol); ++r)
                pk = std::max(pk, static_cast<double>(polT.at<float>(r, x)));
            curvePk[x] = pk;
            curveOn[x] = pk >= model.config.strokeHi ? 1.0 : 0.0;
        }
    }
    const double pkMax = std::max(0.01, *std::max_element(curvePk.begin(), curvePk.end()));
    for (int x = 1; x < W; ++x) {
        auto toY = [&](double v) {
            return H - 1 - static_cast<int>(std::lround(v / pkMax * (H - 30)));
        };
        cv::line(p4, cv::Point(x - 1, toY(curvePk[x - 1])), cv::Point(x, toY(curvePk[x])),
                 cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
        // 沟底线：判为有墨的部分画绿
        if (curveOn[x] > 0.5)
            cv::line(p4, cv::Point(x, H - 8), cv::Point(x, H - 1), cv::Scalar(0, 255, 0), 1);
    }
    // 门槛线
    cv::line(p4, cv::Point(0, H - 1 - static_cast<int>(model.config.strokeHi / pkMax * (H - 30))),
             cv::Point(W, H - 1 - static_cast<int>(model.config.strokeHi / pkMax * (H - 30))),
             cv::Scalar(120, 120, 120), 1);
    label(p4, "4 ring ink vs angle  (green=on-ink)", cv::Point(6, 20));

    // ---- 拼图：左右两列
    auto fitH = [](cv::Mat& m, int targetH) {
        if (m.rows == targetH) return;
        const double s = static_cast<double>(targetH) / m.rows;
        cv::resize(m, m, cv::Size(std::max(1, static_cast<int>(m.cols * s)), targetH));
    };
    const int colW = 520;
    auto fitW = [&](cv::Mat& m) {
        const double s = static_cast<double>(colW) / m.cols;
        cv::resize(m, m, cv::Size(colW, static_cast<int>(m.rows * s)));
    };
    fitW(p1);
    fitW(p2);
    fitW(p3);
    fitW(p4);

    // 右列三张叠起来，左列一张，总高度对齐
    const int rightH = p2.rows + p3.rows + p4.rows + 8;
    const int totalH = std::max(p1.rows, rightH);
    if (p1.rows != totalH) fitH(p1, totalH);
    cv::Mat right(totalH, colW, CV_8UC3, cv::Scalar(0, 0, 0));
    int y = 0;
    p2.copyTo(right(cv::Rect(0, y, p2.cols, p2.rows)));
    y += p2.rows + 4;
    p3.copyTo(right(cv::Rect(0, y, p3.cols, p3.rows)));
    y += p3.rows + 4;
    p4.copyTo(right(cv::Rect(0, y, p4.cols, p4.rows)));

    cv::Mat panel(totalH, colW * 2 + 8, CV_8UC3, cv::Scalar(0, 0, 0));
    p1.copyTo(panel(cv::Rect(0, 0, p1.cols, p1.rows)));
    right.copyTo(panel(cv::Rect(colW + 8, 0, colW, totalH)));

    // 顶部状态条
    cv::Mat bar(34, panel.cols, CV_8UC3, cv::Scalar(30, 30, 30));
    char head[256];
    std::snprintf(head, sizeof(head),
                  "cover=%.4f  maxGap=%.2f deg  score=%.0f  band=%s  swing=%.1fpx  verdict=%s",
                  pb.coverage, pb.maxGapAngle, pb.score, chardet::patternBandName(pb.band),
                  pb.radiusSwing, pb.flagged ? "NG" : "OK");
    cv::putText(bar, head, cv::Point(6, 23), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                pb.flagged ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    cv::Mat out;
    cv::vconcat(bar, panel, out);
    cv::imwrite(argv[3], out);
    std::printf("已写出 %s (%dx%d)\n", argv[3], out.cols, out.rows);
    return 0;
}
