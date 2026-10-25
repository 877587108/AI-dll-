// 用途：确认"对齐后剩下的候选"是不是用户看到的那道划痕
//
// 【为什么必须查这个】
// 0101 的分数经历了 1347 -> 796。
// 如果 796 那条候选就是用户看到的那道划痕，那问题是"判据太严"，
// 应该改判据。
// 如果 796 那条候选是别的东西，而真划痕在对齐后根本没被检出，
// 那问题是"对齐把真划痕的证据抹掉了"，改判据没用，得改对齐。
//
// 这两种结论对应的修法完全相反，不能靠猜。
//
// 做法：把候选线段画到对齐后的图上，同时画上字符框，输出成图给人看。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
    if (argc < 4) {
        std::printf("usage: markloc_hitviz <modelDir> <image> <out.png>\n");
        return 2;
    }
    const QString modelDir = QString::fromLocal8Bit(argv[1]);
    const QString imgPath = QString::fromLocal8Bit(argv[2]);
    const QString outPath = QString::fromLocal8Bit(argv[3]);

    chardet::Model model;
    if (!chardet::loadModel(modelDir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat g;
    if (!readGrayU8(imgPath, g) || g.size() != model.freq.size()) {
        std::printf("read failed\n");
        return 1;
    }

    std::vector<chardet::CharUnit> base;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        if (model.regions[ri].isPattern()) continue;
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) base.push_back(std::move(x));
    }

    const chardet::Config& cfg = model.config;

    // 复刻 detectOne 的配准（和 batchcore.hpp 一致）
    cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
    double sx = 0, sy = 0;
    if (!model.ref.empty()) {
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
                sx = sh.x;
                sy = sh.y;
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::Mat aligned;
                cv::warpAffine(sig, aligned, M, sig.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                sig = aligned;
            }
        }
    }
    std::printf("配准位移 (%.2f, %.2f)\n", sx, sy);

    markloc::ScratchParams sp;
    markloc::ScratchStats ss;
    const auto hits = markloc::findScratches(sig, model.freq, base, cfg, sp, &ss);
    std::printf("命中 %d 条，落选最高 S=%.0f\n", static_cast<int>(hits.size()), ss.bestComposite);

    // ---- 把门槛临时降到 0，把"落选的那条"也拿出来画
    //
    // 目的：确认 796 分那条候选究竟落在哪。
    // 如果它落在用户看到的那道划痕上，说明判据只是差一点；
    // 如果它落在别处（比如字形边缘），那门槛不能动，得从别处想办法。
    markloc::ScratchParams spAll = sp;
    spAll.minComposite = 0.0;
    markloc::ScratchStats ssAll;
    const auto all = markloc::findScratches(sig, model.freq, base, cfg, spAll, &ssAll);
    std::printf("降门槛后全部候选 %d 条\n", static_cast<int>(all.size()));

    // 画：原图（配准后）+ 字符框（绿）+ 命中（红）
    cv::Mat vis;
    cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);
    for (const auto& u : base) {
        cv::rectangle(vis, u.bbox, cv::Scalar(0, 200, 0), 1);
    }
    for (const auto& h : hits) {
        cv::line(vis, h.a1, h.a2, cv::Scalar(0, 0, 255), 2);
        cv::rectangle(vis, h.box, cv::Scalar(0, 128, 255), 1);
        std::printf("  命中 S=%.0f len=%.0f depth=%.1f chars=%d 线 (%.0f,%.0f)-(%.0f,%.0f)\n",
                    h.composite, h.len, h.depth, h.charsHit, h.a1.x, h.a1.y, h.a2.x, h.a2.y);
    }
    // 落选的也画：品红细线
    for (const auto& h : all) {
        if (h.composite >= sp.minComposite) continue;
        if (h.a1 == cv::Point(0, 0) && h.a2 == cv::Point(0, 0)) continue;   // 坐标无效，跳过
        cv::line(vis, h.a1, h.a2, cv::Scalar(255, 0, 255), 1);
        std::printf("  落选 S=%.0f len=%.0f depth=%.1f chars=%d 线 (%.0f,%.0f)-(%.0f,%.0f)\n",
                    h.composite, h.len, h.depth, h.charsHit, h.a1.x, h.a1.y, h.a2.x, h.a2.y);
    }
    cv::imwrite(outPath.toStdString(), vis);
    std::printf("已写出 %s\n", outPath.toStdString().c_str());
    return 0;
}
