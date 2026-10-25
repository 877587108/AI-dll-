// gui_pattern_path.cpp —— 复现 GUI 图案区那条完整调用链
//
// 【为什么要单独做一个工具】
// 图案区在批量入口（batchcore）里已经验证过，但 GUI 是另一条代码路径：
// 它自己算相位相关配准、自己把椭圆跟着平移、自己判 NG。
// "批量对了"推不出"GUI 对了" —— 两边各写了一份平移，符号写反一边
// 就是一次硬误报（0095_RT 上 +sh 报出过"断口 4 处、缺口 50.5°"）。
//
// 所以这里逐字复刻 DefectViewWindow::runDetect() 的顺序：
//   ① strokeSignal
//   ② 只用字符区算 phaseCorrelate → warpAffine(-sh)
//   ③ 椭圆 center += -sh   ← 就是这一步的符号要验
//   ④ analyzePatternBreak
// 再把椭圆画到图上，让人能看见它到底压没压在环上。
//
// 用法：markloc_guipattern <modelRoot> <image> <out.png>
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
    if (argc < 4) {
        std::printf("usage: markloc_guipattern <modelRoot> <image> <out.png>\n");
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
        std::printf("读图失败或尺寸不符（模型 %dx%d，图 %dx%d）\n", model.freq.cols,
                    model.freq.rows, g.cols, g.rows);
        return 1;
    }

    // ① 信号
    cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

    // ② 相位相关配准：只用字符区，和 GUI 完全一致
    const std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);
    double alignDx = 0.0, alignDy = 0.0;
    {
        cv::Mat r32, s32;
        model.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : units) {
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
                alignDx = sh.x;
                alignDy = sh.y;
            }
        }
    }

    // ③ 椭圆保持原位。
    //
    // 配准矩阵 M=[1,0,-sh;0,1,-sh] 配 WARP_INVERSE_MAP 的语义是
    // dst(x,y)=src(x-sh,y-sh)，图像内容移动了 +sh，正是"挪回模型基准"。
    // 挪完之后模型坐标重新对上，椭圆不需要再平移。
    // 曾在这里加 -sh，0095_RT 覆盖率 0.9944 -> 0.9444 造成硬误报。
    chardet::RegionInfo moved = pat;

    // ④ 检测
    chardet::PatternBreakParams P;
    const auto pb = chardet::analyzePatternBreak(sig, moved, model.config, P);

    std::printf("=== %s ===\n", argv[2]);
    std::printf("配准 相位相关 dx=%.2f dy=%.2f  → 椭圆加 (-dx,-dy)\n", alignDx, alignDy);
    std::printf("椭圆 模型(%.1f,%.1f) → 移动后(%.1f,%.1f) 半轴%.2f×%.2f 角%.2f° 离心率%.3f\n",
                pat.ell.center.x, pat.ell.center.y, moved.ell.center.x, moved.ell.center.y,
                moved.ell.a, moved.ell.b, moved.ell.angleDeg, moved.ell.eccentricity());
    std::printf("结果 valid=%d 对准=%d 覆盖%.4f 断口%d 最长缺口%.2f° 断点分%.0f 档位%s\n",
                pb.valid ? 1 : 0, pb.alignOk ? 1 : 0, pb.coverage, pb.breaks, pb.maxGapAngle,
                pb.score, chardet::patternBandName(pb.band));
    std::printf("半径起伏 %.1fpx 墨线宽 %.1fpx  判定 %s\n", pb.radiusSwing, pb.radiusLineWidth,
                pb.flagged ? "NG" : "OK");

    // 出图：把信号转灰度，画椭圆与断口位置，裁到椭圆附近
    cv::Mat view;
    cv::normalize(sig, view, 0, 255, cv::NORM_MINMAX);
    view.convertTo(view, CV_8U);
    cv::cvtColor(view, view, cv::COLOR_GRAY2BGR);

    // 量一下两种符号各自的"环上墨量"（在原图灰度上量，信号图被高通拉平了）。
    // 这个数直接回答"哪个符号真的压在环上"，不用靠看图。
    auto ringMean = [&](double dx, double dy) {
        const double th = pat.ell.angleDeg * CV_PI / 180.0;
        const double ca = std::cos(th), sa = std::sin(th);
        double sum = 0.0;
        int cnt = 0;
        for (int i = 0; i < 720; ++i) {
            const double t = 2.0 * CV_PI * i / 720.0;
            const double u = pat.ell.a * std::cos(t), v = pat.ell.b * std::sin(t);
            const int x = cvRound(pat.ell.center.x + dx + u * ca - v * sa);
            const int y = cvRound(pat.ell.center.y + dy + u * sa + v * ca);
            if (x < 0 || y < 0 || x >= g.cols || y >= g.rows) continue;
            sum += g.at<uchar>(y, x);
            ++cnt;
        }
        return cnt > 0 ? sum / cnt : 0.0;
    };
    std::printf("环上墨量 不平移 %.1f | 加+sh %.1f | 加-sh %.1f（取最高者为正确符号）\n",
                ringMean(0, 0), ringMean(alignDx, alignDy), ringMean(-alignDx, -alignDy));

    // 模型原位（红，虚线看不出，用细线）—— 对照"没平移就画哪"
    cv::ellipse(view,
                cv::RotatedRect(cv::Point2f(pat.ell.center.x, pat.ell.center.y),
                                cv::Size2f(pat.ell.a * 2, pat.ell.b * 2),
                                static_cast<float>(pat.ell.angleDeg)),
                cv::Scalar(0, 0, 180), 1, cv::LINE_AA);
    // 平移后（绿）—— 真正用于判据的那个
    cv::ellipse(view,
                cv::RotatedRect(cv::Point2f(moved.ell.center.x, moved.ell.center.y),
                                cv::Size2f(moved.ell.a * 2, moved.ell.b * 2),
                                static_cast<float>(moved.ell.angleDeg)),
                cv::Scalar(0, 220, 0), 2, cv::LINE_AA);
    for (const auto& pt : pb.breakAt) {
        cv::circle(view, cv::Point(cvRound(pt.x), cvRound(pt.y)), 7, cv::Scalar(0, 0, 255), 2,
                   cv::LINE_AA);
    }
    cv::putText(view, "green=shifted  red=model  circle=break", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    const int pad = 40;
    // 裁切框取椭圆真正的轴对齐外接框。
    // 椭圆是斜的，竖直方向的跨度不等于长半轴 a；用手算容易偏，
    // 直接让 OpenCV 给出旋转矩形的 boundingRect 最稳。
    const cv::RotatedRect rrMoved(
        cv::Point2f(moved.ell.center.x, moved.ell.center.y),
        cv::Size2f(moved.ell.a * 2, moved.ell.b * 2), static_cast<float>(moved.ell.angleDeg));
    cv::Rect crop = rrMoved.boundingRect();
    crop = cv::Rect(crop.x - pad, crop.y - pad, crop.width + pad * 2, crop.height + pad * 2);
    crop &= cv::Rect(0, 0, view.cols, view.rows);
    if (crop.area() > 0) view = view(crop);
    cv::imwrite(argv[3], view);
    std::printf("出图 %s\n", argv[3]);
    return 0;
}
