// 0101 定位工具：把"落选最高分"的线段找出来画在图上
//
// 【为什么要专门写这个】
// hitviz 打印出来端点全是 (0,0)，说明我在打印时取错了字段。
// 与其继续猜结构，不如把 ScratchHit 的每个字段都打一遍，看清到底哪个是坐标。
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
        std::printf("usage: markloc_locate <modelDir> <image> <out.png>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g) || g.size() != model.freq.size()) {
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

    cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
    // 配准
    {
        cv::Mat r32, s32;
        model.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : base) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
        }
        inkArea.x = std::max(0, inkArea.x - 20);
        inkArea.y = std::max(0, inkArea.y - 20);
        inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
        inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
        cv::Mat rC, sC;
        r32(inkArea).copyTo(rC);
        s32(inkArea).copyTo(sC);
        const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat al;
            cv::warpAffine(sig, al, M, sig.size(),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
            sig = al;
        }
    }

    markloc::ScratchParams spAll;
    spAll.minComposite = 0.0;
    markloc::ScratchStats ss;
    const auto all = markloc::findScratches(sig, model.freq, base, cfg, spAll, &ss);
    std::printf("候选 %d 条（minComposite=0），最高 S=%.0f\n", static_cast<int>(all.size()),
                ss.bestComposite);

    // 按分数排序，打印前 5 条的全部字段
    auto sorted = all;
    std::sort(sorted.begin(), sorted.end(), [](const markloc::ScratchHit& a,
                                               const markloc::ScratchHit& b) {
        return a.composite > b.composite;
    });
    std::printf("\n%-8s %-22s %-22s %-26s %6s %6s %4s\n", "S", "a1(x,y)", "a2(x,y)", "box(x,y,w,h)",
                "len", "depth", "ch");
    for (size_t i = 0; i < sorted.size() && i < 6; ++i) {
        const auto& h = sorted[i];
        char bs[64], b1[32], b2[32];
        std::snprintf(b1, sizeof b1, "(%d,%d)", h.a1.x, h.a1.y);
        std::snprintf(b2, sizeof b2, "(%d,%d)", h.a2.x, h.a2.y);
        std::snprintf(bs, sizeof bs, "(%d,%d,%d,%d)", h.box.x, h.box.y, h.box.width, h.box.height);
        std::printf("%-8.0f %-22s %-22s %-26s %6.0f %6.1f %4d\n", h.composite, b1, b2, bs, h.len,
                    h.depth, h.charsHit);
    }

    // 画：绿框=字符框，橙=最高分候选的 box，红=最高分候选的线
    cv::Mat vis;
    cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);
    for (const auto& u : base) cv::rectangle(vis, u.bbox, cv::Scalar(0, 200, 0), 1);
    for (size_t i = 0; i < sorted.size() && i < 6; ++i) {
        const auto& h = sorted[i];
        const cv::Scalar col = (i == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 255);
        cv::rectangle(vis, h.box, cv::Scalar(0, 140, 255), 1);
        if (h.a1 != h.a2) cv::line(vis, h.a1, h.a2, col, 2);
    }
    // 只裁出上半部分的字符区以便看清
    cv::Rect roi(400, 760, 900, 260);
    roi &= cv::Rect(0, 0, vis.cols, vis.rows);
    cv::Mat crop;
    cv::resize(vis(roi), crop, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
    cv::imwrite(argv[3], crop);
    std::printf("\n已写出 %s\n", argv[3]);
    return 0;
}
