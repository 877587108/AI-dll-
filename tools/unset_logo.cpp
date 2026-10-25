// unset_logo.cpp —— 用 A 方案（带型连通判定）测 unset 里的 logo 断连
//
// 【为什么要专门写一个】
// image/unset 是 1496x1496 的新规格图，模型是 1688x1460，尺寸不符。
// 整幅的 strokeSignal 尺度不可比，字符区完全没法用。
// 但 logo 环是**同一个部件、同一块区域**，用模板匹配能稳定定位
// （实测峰值 0.93~0.95），所以只在这一小块上做断点检测是成立的。
//
// 流程：
//   ① 模板匹配找到 logo 在原图上的位置（整块，不是逐像素）
//   ② 由模板里已知的椭圆参数换算出原图坐标下的椭圆
//   ③ 裁一小块 ROI，在 ROI 内算 strokeSignal（尺度只看这一块，可比）
//   ④ 调用 analyzePatternBreak（带型判定），看有没有断口
//
// 【关键：信号尺度】
// strokeSignal 的 scale 是在整幅上估的，跨图不可比。这里只在 ROI 内估，
// 并且用同一套 cfg，保证同批图之间可比。跨规格（旧 1688 / 新 1496）
// 比较绝对值没有意义，所以这里只报"断没断"，不比分数大小。
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
    if (argc < 4) {
        std::printf("usage: markloc_unsetlogo <modelRoot> <imageDir> <tmpl.png> [--csv out.csv]\n");
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
    // 与 logo_track 一致的实测模板内椭圆参数
    const double tplCx = 127.8, tplCy = 116.0;
    const double tplA = 132.1, tplB = 76.75;
    const double tplRotDeg = -13.78;

    QString csvPath;
    bool probe = false;
    bool allFrames = false;
    // 判 NG 的门槛由外部指定，不动库里的默认值。
    //
    // 默认是 minGapAngle=0.6° / minScore=400（即 4.0°）。
    // 客户要求"缺口 >1.0° 就判 NG"时，用 --ngdeg 1.0 覆盖这两个数：
    //   minGapAngle = 1.0（缺口必须够 1.0° 才算断口）
    //   minScore    = 100（断点分 = 角度×100，1.0° 对应 100 分）
    // 两个一起改才自洽：只改 minScore 的话，0.6~1.0° 的缺口仍会被
    // 算成"断口"并给出分数，只是不报 NG —— 那是另一套语义。
    double ngDeg = -1.0;
    // 展开前的灰度阈值，作用在拉直图上（见 PatternBreakParams::preGrayThr）
    double preGray = 0.0;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--probe")) {
            probe = true;
        } else if (!std::strcmp(argv[i], "--pre") && i + 1 < argc) {
            preGray = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--all")) {
            // 连拍的全部帧都测。同一块料的 5 帧应该给出一致结论，
            // 不一致就说明判据本身不稳定，那比漏检更值得注意。
            allFrames = true;
        } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
            csvPath = QString::fromLocal8Bit(argv[++i]);
        } else if (!std::strcmp(argv[i], "--ngdeg") && i + 1 < argc) {
            ngDeg = std::atof(argv[++i]);
        }
    }

    QStringList files;
    for (const QFileInfo& fi : QDir(QString::fromLocal8Bit(argv[2]))
                                   .entryInfoList({QStringLiteral("*.tif"), QStringLiteral("*.png"),
                                                   QStringLiteral("*.bmp")},
                                                  QDir::Files, QDir::Name)) {
        if (!allFrames && !fi.completeBaseName().endsWith(QStringLiteral("_00"))) continue;
        files << fi.absoluteFilePath();
    }
    std::printf("模板 %dx%d，共 %d 张 _00 图\n", tmpl.cols, tmpl.rows,
                static_cast<int>(files.size()));

    std::FILE* csv = nullptr;
    if (!csvPath.isEmpty()) {
        csv = std::fopen(csvPath.toLocal8Bit().constData(), "w");
        if (csv) std::fprintf(csv, "file,mx,my,peak,valid,cover,maxGapDeg,breaks,score,band,flagged\n");
    }
    std::printf("%-52s %5s %5s %6s %5s %6s %7s %3s %5s %s\n", "file", "mx", "my", "cover",
                "align", "maxGap", "score", "brk", "判定", "档位");

    int nDetected = 0, nFlagged = 0, nUnaligned = 0;
    for (const QString& path : files) {
        cv::Mat g;
        if (!readGrayU8(path, g)) continue;
        if (tmpl.cols > g.cols || tmpl.rows > g.rows) continue;

        cv::Mat res;
        cv::matchTemplate(g, tmpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(res, nullptr, &mval, nullptr, &mloc);

        // 模板内椭圆 -> 原图坐标
        const double cx = mloc.x + tplCx;
        const double cy = mloc.y + tplCy;
        const double a = tplA, b = tplB;

        // 裁 ROI：椭圆外接多一点，保证环完整落在里面
        const int pad = static_cast<int>(a * 0.45) + 8;
        cv::Rect roi(static_cast<int>(cx - a - pad), static_cast<int>(cy - a - pad),
                     static_cast<int>(2 * (a + pad)), static_cast<int>(2 * (a + pad)));
        roi &= cv::Rect(0, 0, g.cols, g.rows);
        if (roi.width < 16 || roi.height < 16) continue;

        // 【关键：信号必须在整幅上算，不能在 ROI 里算】
        //
        // strokeSignal 的归一化 scale 由 99.5 分位估计。ROI 里环占了很大面积，
        // q995 会被环本身主导，scale 被抬高，环的强度反而压到 inkThr 以下，
        // 于是大量角度被判"无墨"——实测挖 0° 洞就报 12.5° 缺口，基线全错。
        //
        // 改成整幅算、再取 ROI：scale 由整幅统计决定，
        // 与旧规格（1688x1460）上的做法一致，环的绝对强度才可比。
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
        // 展开前的灰度阈值：把拉直图上不够亮的墨先切掉，
        // 这样"环变淡"在展开图上是一段真实的空洞，而不是被连通判据兜住。
        P.preGrayThr = preGray;
        if (ngDeg > 0.0) {
            P.minGapAngle = ngDeg;
            P.minScore = ngDeg * 100.0;
        }
        const auto pb = chardet::analyzePatternBreak(sig, rg, model.config, P);
        if (!pb.valid) continue;

        // --probe：只处理第一张，把它内部的几何全打出来。
        // "全批 0 张"这种结论必须能解释，否则分不清"环都好"和"没测对"。
        static int probed = 0;
        if (probe && probed++ == 0) {
            std::printf("\n=== PROBE %s ===\n", QFileInfo(path).fileName().toUtf8().constData());
            std::printf("模板匹配 (%d,%d) 峰值%.4f\n", mloc.x, mloc.y, mval);
            std::printf("椭圆(全图) 中心(%.1f,%.1f) a=%.1f b=%.1f 角%.2f°\n", cx, cy, a, b,
                        tplRotDeg);
            std::printf("ROI (%d,%d) %dx%d  椭圆在ROI内 (%d,%d)\n", roi.x, roi.y, roi.width,
                        roi.height, static_cast<int>(cx - roi.x), static_cast<int>(cy - roi.y));
            std::printf("信号 min/max/mean: %.3f / %.3f / %.3f\n", 0.0, 0.0, 0.0);
            double mn = 1e9, mx = -1e9, sm = 0;
            for (int yy = 0; yy < sig.rows; ++yy)
                for (int xx = 0; xx < sig.cols; ++xx) {
                    const double v = sig.at<float>(yy, xx);
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                    sm += v;
                }
            std::printf("   实为 %.3f / %.3f / %.3f\n", mn, mx, sm / (sig.rows * sig.cols));
            std::printf("innerBlocked=%d bandOuter=%d 覆盖%.4f 断口%d 缺口%.2f° 分%.0f\n",
                        pb.innerBlocked, pb.bandOuter, pb.coverage, pb.breaks, pb.maxGapAngle,
                        pb.score);
            for (const auto& pt : pb.breakAt)
                std::printf("  断口@ (%.1f,%.1f) 全图 (%.1f,%.1f)\n", pt.x, pt.y, pt.x + roi.x,
                            pt.y + roi.y);
        }

        ++nDetected;
        if (pb.flagged) ++nFlagged;
        if (!pb.alignOk) ++nUnaligned;

        const std::string nm = QFileInfo(path).fileName().toStdString().substr(0, 52);
        std::printf("%-52s %5d %5d %6.3f %5s %6.2f %5.0f %3d %-5s %s\n", nm.c_str(), mloc.x,
                    mloc.y, pb.coverage, pb.alignOk ? "OK" : "未对", pb.maxGapAngle, pb.score,
                    pb.breaks, pb.flagged ? "NG" : "OK", chardet::patternBandName(pb.band));
        if (csv)
            std::fprintf(csv, "%s,%d,%d,%.4f,%d,%.4f,%.2f,%.0f,%d,%s,%d\n",
                         QFileInfo(path).fileName().toUtf8().constData(), mloc.x, mloc.y, mval,
                         pb.valid ? 1 : 0, pb.coverage, pb.maxGapAngle, pb.score, pb.breaks,
                         chardet::patternBandName(pb.band), pb.flagged ? 1 : 0);
    }
    if (csv) std::fclose(csv);
    std::printf("\n=== 汇总 ===\n");
    std::printf("检测 %d 张，判 NG %d 张，未对准 %d 张\n", nDetected, nFlagged, nUnaligned);
    return 0;
}
