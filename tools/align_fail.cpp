// alignfail.cpp —— 没通过"对准检查"的那些图，环到底有没有问题
//
// 【要回答的问题】
// 门槛放宽到 3.5 倍线宽后，83 张里还有 14 张被判"未对准"。
// 如果这 14 张的覆盖率其实都是 1.000、断口 0，那说明
// "半径起伏大"并不等于"没对准"，只是环有一段粗一段细。
// 这时候该改的是判据，不是继续抬门槛。
//
// 所以这里把"未对准"的图和它们的覆盖率/缺口/起伏一起列出来：
//   覆盖率 1.000 且缺口 0  → 环是完整的，起伏大只是形态
//   覆盖率明显 < 1 或缺口大 → 真的没对上，结论作废是对的
#include <algorithm>
#include <cstdio>
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
    if (argc < 3) {
        std::printf("usage: markloc_alignfail <modelRoot> <imageDir>\n");
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

    QDir dir(QString::fromLocal8Bit(argv[2]));
    const QStringList files = dir.entryList({QStringLiteral("*_00.tif")}, QDir::Files, QDir::Name);

    std::printf("%-42s %6s %6s %6s %6s %s\n", "file", "swing", "lineW", "limit", "cover", "状态");
    int nAlignFail = 0, failButClean = 0;
    for (const QString& fn : files) {
        cv::Mat g;
        if (!readGrayU8(dir.absoluteFilePath(fn), g) || g.size() != model.freq.size()) continue;
        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
        chardet::PatternBreakParams P;
        const auto pb = chardet::analyzePatternBreak(sig, pat, model.config, P);
        const double limit = std::max(20.0, 3.5 * pb.radiusLineWidth);
        if (pb.alignOk) continue;
        ++nAlignFail;
        // 环是否本身上好的：覆盖率接近满且没有缺口
        const bool clean = pb.coverage >= 0.99 && pb.maxGapAngle < 3.0;
        if (clean) ++failButClean;
        std::printf("%-42s %6.1f %6.1f %6.1f %6.3f %s\n", fn.left(42).toStdString().c_str(),
                    pb.radiusSwing, pb.radiusLineWidth, limit, pb.coverage,
                    clean ? "未对准但环完好" : "未对准且环有缺口");
    }
    std::printf("\n未对准 %d 张，其中环本身完好 %d 张。\n", nAlignFail, failButClean);
    return 0;
}
