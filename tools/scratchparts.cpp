// scratchparts.cpp —— 把 findScratches 的分数拆开看
//
// 【为什么需要】
// 0101 用户判定是划痕，但严重度只有 883/831，卡在门槛 1000 下。
// 直接降门槛是最省事的错解：ok 侧 84 张会跟着乱报。
//
// 严重度 = depth × charsHit。所以要么 depth 算低了，要么 charsHit 数少了。
// 得先看清是哪一项，再谈改什么。
//
// 这个工具直接调 findScratches，把每条候选的
//   长度 / 覆盖 / 落差(depth) / 宽度 / 跨字数 / 总分
// 全部打出来，OK 和 NG 并排看。
#include <algorithm>
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
        std::printf("usage: markloc_scratchparts <modelDir> <imageDir> [--grep S]\n");
        return 2;
    }
    const QString modelDir = QString::fromLocal8Bit(argv[1]);
    const QString imgDir = QString::fromLocal8Bit(argv[2]);
    QString filter;
    for (int i = 3; i + 1 < argc; i += 2)
        if (!std::strcmp(argv[i], "--grep")) filter = QString::fromLocal8Bit(argv[i + 1]);

    chardet::Model model;
    const QString vdir = QDir(modelDir).absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    std::vector<chardet::CharUnit> base;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        if (model.regions[ri].isPattern()) continue;   // 图案区不参与划痕检测
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) base.push_back(std::move(x));
    }

    QStringList files;
    for (const QFileInfo& fi : QDir(imgDir).entryInfoList(
             {QStringLiteral("*.tif"), QStringLiteral("*.tiff"), QStringLiteral("*.png"),
              QStringLiteral("*.bmp")},
             QDir::Files, QDir::Name)) {
        if (!fi.completeBaseName().endsWith(QStringLiteral("_00"))) continue;
        if (!filter.isEmpty() && !fi.fileName().contains(filter)) continue;
        files << fi.absoluteFilePath();
    }

    std::printf("%-20s %5s %5s %6s %6s %6s %7s %5s\n", "file", "len", "cov%", "depth", "width",
                "chars", "S", "band");
    for (const QString& path : files) {
        cv::Mat g;
        if (!readGrayU8(path, g)) continue;
        if (g.size() != model.freq.size()) continue;

        markloc::DetectSettings det;
        det.cfg = model.config;
        bool valid = false;
        auto row = markloc::detectOne(g, model, base, det, &valid);

        // 三个对照，用来定位"分数在哪一步掉的"：
        //   A 裸信号             —— 不做任何平移
        //   B 只做全局配准        —— 和 detectOne 的第二步一致
        //   C detectOne 全流程    —— 配准 + 逐字符像素重对齐
        // 如果 A 高、B 低，问题在配准；如果 B 高、C 低，问题在逐字符重对齐。
        const chardet::Config& cfg = model.config;

        auto report = [&](const char* tag, const cv::Mat& s) {
            markloc::ScratchStats ss;
            const auto hits = markloc::findScratches(s, model.freq, base, cfg, det.scratch, &ss);
            double best = 0;
            for (const auto& h : hits) best = std::max(best, h.composite);
            double bestRej = ss.bestComposite;
            std::printf("  %-8s 命中%d 最高S=%.0f 落选最高S=%.0f\n", tag,
                        static_cast<int>(hits.size()), best, bestRej);
            for (const auto& h : hits) {
                std::printf("      len%5.0f cov%5.2f depth%6.1f width%5.1f chars%2d S%6.0f b%d\n",
                            h.len, h.cov, h.depth, h.widthPx, h.charsHit, h.composite, h.band);
            }
        };

        const std::string stem = QFileInfo(path).completeBaseName().toStdString();
        std::printf("=== %s\n", stem.c_str());
        std::printf("  detectOne  S=%d 候选%d 未报[%s]\n", static_cast<int>(row.scratchS),
                    row.candTotal, row.rejectReason.c_str());

        // A：裸信号
        const cv::Mat sigRaw =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
        report("A裸信号", sigRaw);

        // B：只做全局配准
        cv::Mat sigB = sigRaw.clone();
        double alignDx = 0, alignDy = 0;
        if (!model.ref.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sigRaw.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : base) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sigRaw.cols, sigRaw.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            if (inkArea.area() > 0) {
                inkArea.x = std::max(0, inkArea.x - 20);
                inkArea.y = std::max(0, inkArea.y - 20);
                inkArea.width = std::min(sigRaw.cols - inkArea.x, inkArea.width + 40);
                inkArea.height = std::min(sigRaw.rows - inkArea.y, inkArea.height + 40);
                cv::Mat rC, sC;
                r32(inkArea).copyTo(rC);
                s32(inkArea).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                std::printf("  配准位移 (%.2f, %.2f)\n", sh.x, sh.y);
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    alignDx = sh.x;
                    alignDy = sh.y;
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::warpAffine(sigRaw, sigB, M, sigRaw.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                }
            }
        }
        report("B仅配准", sigB);

        // C：配准 + 把字符框按配准量一起挪（这是"物理上正确"的做法）
        //
        // B 的做法是"像素移动、框不动"，这会让划痕和字符区的相对关系错位。
        // 正确做法是框跟着一起挪，这样框里装的还是同一个字形。
        std::vector<chardet::CharUnit> moved = base;
        if (std::abs(alignDx) > 0.5 || std::abs(alignDy) > 0.5) {
            for (auto& u : moved) {
                u.bbox.x += static_cast<int>(std::lround(alignDx));
                u.bbox.y += static_cast<int>(std::lround(alignDy));
                u.extend.x += static_cast<int>(std::lround(alignDx));
                u.extend.y += static_cast<int>(std::lround(alignDy));
            }
        }
        {
            markloc::ScratchStats ss;
            const auto hits = markloc::findScratches(sigB, model.freq, moved, cfg, det.scratch, &ss);
            double best = 0;
            for (const auto& h : hits) best = std::max(best, h.composite);
            std::printf("  %-8s 命中%d 最高S=%.0f 落选最高S=%.0f\n", "C框随移",
                        static_cast<int>(hits.size()), best, ss.bestComposite);
            for (const auto& h : hits) {
                std::printf("      len%5.0f cov%5.2f depth%6.1f width%5.1f chars%2d S%6.0f b%d\n",
                            h.len, h.cov, h.depth, h.widthPx, h.charsHit, h.composite, h.band);
            }
        }
    }
    return 0;
}
