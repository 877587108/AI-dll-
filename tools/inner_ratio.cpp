// inner_ratio.cpp —— innerBlockRatio 从 0.7 提到 0.8，会不会误伤环
//
// 【要回答的问题】
// pattern_break 先把半径 < innerBlockRatio*a 的区域清零，避免字母的墨
// 被当成环的一部分。实测 0.7 时字母仍有残留，半径跟踪被拉到 0.71a，
// 起伏量被撑到 40+px，于是"对准检查"判没对准、结论清零 ——
// 图案区检测在大多数图上其实根本没跑，只是碰巧都判 OK。
//
// 但提高这个比例有个风险：环的内沿约在 0.95a，一旦提到 0.95 以上
// 就会把环本身也切掉。所以这里对一批图分别用 0.7/0.8/0.85 跑，
// 看覆盖率和断口数会不会变 —— 变了就说明切到环了。
#include <algorithm>
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
    if (argc < 3) {
        std::printf("usage: markloc_innerratio <modelRoot> <imageDir>\n");
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
    const double ratios[] = {0.70, 0.75, 0.80, 0.85, 0.90};

    std::printf("%-40s", "file");
    for (double r : ratios) std::printf("  r%.2f:swing/cover/brk", r);
    std::printf("\n");

    int aligned[5] = {0, 0, 0, 0, 0};
    std::vector<double> swings[5];
    int total = 0;
    for (const QString& fn : files) {
        cv::Mat g;
        if (!readGrayU8(dir.absoluteFilePath(fn), g) || g.size() != model.freq.size()) continue;
        ++total;
        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

        std::printf("%-40s", fn.left(40).toStdString().c_str());
        for (int i = 0; i < 5; ++i) {
            chardet::PatternBreakParams P;
            P.innerBlockRatio = static_cast<float>(ratios[i]);
            const auto pb = chardet::analyzePatternBreak(sig, pat, model.config, P);
            if (pb.alignOk) ++aligned[i];
            swings[i].push_back(pb.radiusSwing);
            std::printf("  %5.1f/%5.3f/%2d", pb.radiusSwing, pb.coverage, pb.breaks);
        }
        std::printf("\n");
    }
    std::printf("\n共 %d 张。各比例下对准成功的张数：", total);
    for (int i = 0; i < 5; ++i) std::printf(" r%.2f=%d", ratios[i], aligned[i]);
    std::printf("\n");

    // 起伏量直方图：判断门槛该放哪、放开了会不会失去意义。
    //
    // 只报"成功 42 张"看不出分布长什么样 —— 是"大多数很小、少数偶发偏大"
    // 还是"整体就在门槛附近晃"，两种情况的结论完全不同。
    std::printf("\n起伏量(px) 分布：\n");
    for (int i = 0; i < 5; ++i) {
        auto v = swings[i];
        std::sort(v.begin(), v.end());
        if (v.empty()) continue;
        std::printf("  r%.2f  min=%5.1f p25=%5.1f 中位=%5.1f p75=%5.1f max=%5.1f\n", ratios[i],
                    v.front(), v[v.size() / 4], v[v.size() / 2], v[v.size() * 3 / 4], v.back());
    }
    return 0;
}
