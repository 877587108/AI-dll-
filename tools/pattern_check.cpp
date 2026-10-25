// patterncheck.cpp —— 图案区（椭圆/圆）断点检测的验证工具
//
// 【存在的理由】
// 我这一轮反复犯同一个错：把一个量当成"有区分力"就收工，
// 事后发现它要么恒定不动、要么 OK 和 NG 完全重叠。
// 这个工具把断点检测的每一个中间量都摊开，先证明它"能动"，
// 再谈"能不能分"。
//
// 用法：
//   markloc_patterncheck <modelDir> <imageDir> [--angles N] [--ink N]
//                                          [--minGap N] [--maxGap N] [--flank N]
//
// 输出的每一行：一张图 × 一个图案区
//   cov   周上有墨比例
//   gaps  缺口个数 / 最长缺口角度
//   brk   判为断口的个数
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "batchcore.hpp"
#include "chardet.hpp"

namespace fs = std::filesystem;

static bool readGrayU8(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray bytes = f.readAll();
    f.close();
    if (bytes.isEmpty()) return false;
    const std::vector<uchar> buf(bytes.begin(), bytes.end());
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: markloc_patterncheck <modelDir> <imageDir> [--angles N] [--ink N]\n");
        return 2;
    }
    const QString modelDir = QString::fromLocal8Bit(argv[1]);
    const QString imgDir = QString::fromLocal8Bit(argv[2]);

    chardet::PatternBreakParams P;
    for (int i = 3; i + 1 < argc; i += 2) {
        const double v = std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "--angles")) P.angles = static_cast<int>(v);
        else if (!std::strcmp(argv[i], "--ink")) P.inkThr = v;
        else if (!std::strcmp(argv[i], "--minGap")) P.minGapAngle = v;
        else if (!std::strcmp(argv[i], "--maxGap")) P.maxGapAngle = v;
        else if (!std::strcmp(argv[i], "--flank")) P.minFlankCount = static_cast<int>(v);
    }

    const QString vdir = QDir(modelDir).absoluteFilePath(QStringLiteral("v00"));
    chardet::Model model;
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed: %s\n", vdir.toStdString().c_str());
        return 1;
    }

    // 确认模型里到底认出了几个图案区 —— 这是"有没有跑"的前提
    int nPattern = 0;
    std::printf("模型图案区：\n");
    for (const auto& r : model.regions) {
        const char* name = r.shape == chardet::RegionShape::Ellipse
                               ? "椭圆"
                               : (r.shape == chardet::RegionShape::Circle ? "圆" : "框");
        if (r.isPattern()) ++nPattern;
        std::printf("  r%d %-4s %dx%d @(%d,%d) 字符%d\n", r.index, name, r.rect.width,
                    r.rect.height, r.rect.x, r.rect.y, static_cast<int>(r.chars.size()));
    }
    if (nPattern == 0) {
        std::printf("\n【失败】模型里没有任何图案区，断点检测不会执行。\n");
        std::printf("       请把 regions.txt 里对应区域标成 ellipse 或 circle。\n");
        return 1;
    }

    QStringList files;
    for (const QFileInfo& fi : QDir(imgDir).entryInfoList(
             {QStringLiteral("*.tif"), QStringLiteral("*.tiff"), QStringLiteral("*.png"),
              QStringLiteral("*.bmp")},
             QDir::Files, QDir::Name)) {
        if (!fi.completeBaseName().endsWith(QStringLiteral("_00"))) continue;
        files << fi.absoluteFilePath();
    }
    if (files.isEmpty()) {
        std::printf("no v00 images in %s\n", imgDir.toStdString().c_str());
        return 1;
    }

    std::printf("\n参数：angles=%d ink=%.2f minGap=%.2f maxGap=%.2f flank=%d\n", P.angles,
                P.inkThr, P.minGapAngle, P.maxGapAngle, P.minFlankCount);
    std::printf("%-14s %6s %7s %7s %8s %8s %5s\n", "file", "area", "align", "cov", "gaps",
                "maxGap", "brk");

    int totalBreaks = 0;
    std::vector<double> gapHist;
    int covLow = 0;
    for (const QString& path : files) {
        cv::Mat g;
        if (!readGrayU8(path, g)) {
            std::printf("%-14s 读图失败\n", QFileInfo(path).completeBaseName().toStdString().c_str());
            continue;
        }
        if (g.size() != model.freq.size()) {
            std::printf("%-14s 尺寸不符 %dx%d vs 模型 %dx%d\n",
                        QFileInfo(path).completeBaseName().toStdString().c_str(), g.cols, g.rows,
                        model.freq.cols, model.freq.rows);
            continue;
        }
        const chardet::Config& cfg = model.config;
        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);

        // 整图配准，和正式链路一致：图案区必须跟着同样平移
        int dx = 0, dy = 0;
        std::vector<chardet::CharUnit> base;
        for (size_t ri = 0; ri < model.regions.size(); ++ri) {
            auto u = chardet::toUnits(model, static_cast<int>(ri));
            for (auto& x : u) base.push_back(std::move(x));
        }
        if (!model.ref.empty() && !base.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : base) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            if (inkArea.area() > 0) {
                inkArea.x = std::max(0, inkArea.x - 20);
                inkArea.y = std::max(0, inkArea.y - 20);
                inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
                inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
                cv::Mat rC, sC;
                r32(inkArea).copyTo(rC);
                s32(inkArea).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat aligned;
                    cv::warpAffine(sig, aligned, M, sig.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                    sig = aligned;
                    dx = static_cast<int>(std::lround(sh.x));
                    dy = static_cast<int>(std::lround(sh.y));
                }
            }
        }

        const std::string stem = QFileInfo(path).completeBaseName().toStdString();
        for (const auto& r : model.regions) {
            if (!r.isPattern()) continue;
            chardet::RegionInfo mv = r;
            mv.rect.x += dx;
            mv.rect.y += dy;
            const auto pb = chardet::analyzePatternBreak(sig, mv, cfg, P);
            totalBreaks += pb.breaks;
            gapHist.push_back(pb.maxGapAngle);
            if (pb.coverage < 0.9) ++covLow;
            std::printf("%-14s %6d %3d,%3d %7.3f %8s %8.2f %5d\n", stem.c_str(), r.index, dx, dy,
                        pb.coverage, pb.allMissing ? "全缺失" : "-", pb.maxGapAngle, pb.breaks);
            for (const auto& pt : pb.breakAt) {
                std::printf("                 断口 @(%.0f,%.0f)\n", pt.x, pt.y);
            }
        }
    }

    // 汇总：断口总数必须能随参数变化，否则说明这个指标是死的
    std::printf("\n合计断口 %d\n", totalBreaks);
    if (totalBreaks == 0) {
        std::printf("【注意】一张图都没判出断口。这不等于没问题 ——\n");
        std::printf("       先用 --ink 调高/调低看 cov 和 gaps 是否变化，确认指标是活的。\n");
    }

    // 缺口角度的分布。
    //
    // 【为什么要看分布】
    // 只看"有几个断口"分不清两件事：
    //   · 检测器太灵，把刻印的正常粗糙判成断口（缺口都很小）
    //   · 真的有一批件环是断的（缺口明显偏大）
    // 分布的形状直接回答这个。如果 OK 图的缺口集中在 1~2°、而 NG 在 6° 以上，
    // 门槛就该落在两者之间；如果两者重叠，这套判据在这个尺度上就不成立。
    std::printf("\n--- 最长缺口角度分布 ---\n");
    std::printf("区间(°)   张数\n");
    const double buckets[] = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 361.0};
    for (int b = 0; b + 1 < 8; ++b) {
        int cnt = 0;
        for (double v : gapHist) {
            if (v >= buckets[b] && v < buckets[b + 1]) ++cnt;
        }
        if (cnt > 0) {
            std::printf("%5.1f-%-5.1f %3d\n", buckets[b], buckets[b + 1], cnt);
        }
    }
    std::printf("覆盖度 <0.9 的张数 %d\n", covLow);
    return 0;
}
