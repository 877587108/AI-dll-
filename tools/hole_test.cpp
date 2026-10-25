// hole_test.cpp —— 在真实的 RegionInfo 椭圆上挖洞，验证断点检测还能动
//
// 【为什么必须做这个】
// 现在 83 张 OK 图全部报"断口 0"。这既可能是对的（OK 图确实没断），
// 也可能是检测器已经死了、什么都报不出来。
// 两者在没有 NG 样本时长得一模一样。
//
// 做法：直接调用生产的 analyzePatternBreak，在椭圆环上人为挖一个角度的洞，
// 看断口数是否从 0 变成 1，并且随洞的大小单调变化。
// 如果挖了洞还报 0，那这个检测在当前配置下就是死的，前面所有"0 断口"都无意义。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "batchcore.hpp"
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
        std::printf("usage: markloc_hole <modelDir> <image>\n");
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
    std::printf("图案区 中心(%.1f,%.1f) 半轴%.2fx%.2f 角%.2f° e=%.3f 内屏%.0f%%\n",
                pat.ell.center.x, pat.ell.center.y, pat.ell.a, pat.ell.b, pat.ell.angleDeg,
                pat.ell.eccentricity(), 70.0);

    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g) || g.size() != model.freq.size()) {
        std::printf("读图失败\n");
        return 1;
    }
    const cv::Mat sigBase =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

    chardet::PatternBreakParams P;
    const auto base = chardet::analyzePatternBreak(sigBase, pat, model.config, P);
    std::printf("\n基线（不挖洞）：有墨%.4f 断口%d 缺口数%d 最长缺口%.2f° 半径起伏%.0fpx\n",
                base.coverage, base.breaks, base.gapCount, base.maxGapAngle, base.radiusSwing);

    std::printf("\n挖洞测试（在椭圆环上整圈清空）：\n");
    std::printf("%8s %10s %10s %12s %8s %8s %6s %6s\n", "洞角(°)", "有墨比例", "断口数", "最长缺口°", "清px", "断点分", "档位", "判定");
    // 【第一版这里挖错了角度】
    // 原来按椭圆"参数角"挖洞，结果挖 12° 时有墨比例只从 1.0000 掉到 0.9917，
    // 而 12° 应当对应 3.3% —— 说明洞根本没挖在环上。
    // 原因：analyzePatternBreak 内部先把椭圆拉直成正圆再做极坐标，
    // 拉直变换会改变点的角度位置，参数角 θ 与"极坐标图里的角度"不是一回事。
    //
    // 正确挖法：对全图每个像素，先用与内部完全相同的拉直变换算它在拉直后的坐标，
    // 再按"离中心的距离 ≈ a"和"极角落在洞里"两个条件清零。
    // 这样绕开参数角/极角的换算，一定能挖到环上。
    const double aA = std::max(pat.ell.a, pat.ell.b), bB = std::min(pat.ell.a, pat.ell.b);
    const double kScale = aA / std::max(1.0, bB);
    const double rot = pat.ell.angleDeg * CV_PI / 180.0;
    const double cR = std::cos(-rot), sR = std::sin(-rot);
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -pat.ell.center.x, 0, 1, -pat.ell.center.y, 0, 0,
                  1);
    cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
    cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, pat.ell.center.x, 0, 1, pat.ell.center.y, 0, 0,
                  1);
    const cv::Mat M3 = T2 * Sc * Rm * T1;

    for (double holeDeg : {1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0}) {
        cv::Mat sig = sigBase.clone();
        const double halfW = holeDeg / 2.0;
        int cleared = 0;
        for (int y = 0; y < sig.rows; ++y) {
            for (int x = 0; x < sig.cols; ++x) {
                const double rx = M3.at<double>(0, 0) * x + M3.at<double>(0, 1) * y +
                                  M3.at<double>(0, 2);
                const double ry = M3.at<double>(1, 0) * x + M3.at<double>(1, 1) * y +
                                  M3.at<double>(1, 2);
                const double rr = std::hypot(rx - pat.ell.center.x, ry - pat.ell.center.y);
                if (std::abs(rr - aA) > 20) continue;
                const double angDeg = std::atan2(ry - pat.ell.center.y, rx - pat.ell.center.x) *
                                      180.0 / CV_PI;
                double dd = std::fmod(std::abs(angDeg), 360.0);   // 洞中心在 0°
                if (dd > 180) dd = 360 - dd;
                if (dd <= halfW) {
                    sig.at<float>(y, x) = 0.0f;
                    ++cleared;
                }
            }
        }
        const auto r = chardet::analyzePatternBreak(sig, pat, model.config, P);
        std::printf("%8.1f %10.4f %10d %12.2f %8d %8.0f %6s %6s\n", holeDeg, r.coverage, r.breaks,
                    r.maxGapAngle, cleared, r.score, chardet::patternBandName(r.band),
                    r.flagged ? "NG" : "");
    }
    return 0;
}
