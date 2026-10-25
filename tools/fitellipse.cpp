// fitellipse.cpp —— 从实际墨像素直接拟合椭圆，不再拿框当椭圆
//
// 【这一版纠正的错误】
// 我第一版把区域框（474,315,322,225）直接当椭圆的外接框去采样，
// 结果周上有墨比例只有 2%~23%、缺口 160°~330°。
// 看图才明白：logo 是一个**倾斜的椭圆环**，还带着 SL 两个字。
// 框是正的，椭圆是斜的 —— 拿框当椭圆，采样点一大半落在背景上。
//
// 所以椭圆必须从墨自己拟合出来。
// 但直接用 connectedComponents 的最大块拟合有陷阱：环和字是连在一起的
// （面积 8754 那块框 276x178 覆盖整圈），拟合出来是"整块墨的惯性椭圆"，
// 不是环的几何轴。
//
// 做法：环是"离外接框边界最近的墨"。取该区域所有墨像素，
// 用最小二乘拟合椭圆，再按残差筛选迭代 —— 环上的点残差小，字上的点残差大。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
        std::printf("usage: markloc_fitellipse <modelDir> <image> [x y w h]\n");
        return 2;
    }
    const QString modelDir = QString::fromLocal8Bit(argv[1]);
    const QString imgPath = QString::fromLocal8Bit(argv[2]);

    chardet::Model model;
    const QString vdir = modelDir.endsWith(QStringLiteral("v00")) ? modelDir
                                                                 : modelDir + QStringLiteral("/v00");
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }

    cv::Rect roi(474, 315, 322, 225);
    if (argc >= 7) {
        roi = cv::Rect(std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                       std::atoi(argv[6]));
    }
    cv::Mat g;
    if (!readGrayU8(imgPath, g)) {
        std::printf("read failed\n");
        return 1;
    }
    if (g.size() != model.freq.size()) {
        std::printf("size mismatch\n");
        return 1;
    }
    roi &= cv::Rect(0, 0, g.cols, g.rows);
    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    cv::Mat sub = sig(roi).clone();
    cv::Mat bin;
    cv::threshold(sub, bin, model.config.strokeHi, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255);

    // 所有墨点 -> 全图坐标
    std::vector<cv::Point2f> pts;
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x)
            if (bin.at<uchar>(y, x)) pts.emplace_back(static_cast<float>(roi.x + x),
                                                      static_cast<float>(roi.y + y));
    std::printf("区域 %dx%d @(%d,%d) 墨点 %d\n", roi.width, roi.height, roi.x, roi.y,
                static_cast<int>(pts.size()));
    if (pts.size() < 20) {
        std::printf("墨点太少，无法拟合\n");
        return 1;
    }

    // 迭代拟合：cv::fitEllipse 用最小二乘，字上的点会把椭圆拉偏。
    // 每轮只保留"离椭圆最近"的那部分点 —— 环上的点排成一条细带，
    // 字上的点散布在内部，残差大，会被筛掉。
    std::vector<cv::Point2f> cur = pts;
    cv::RotatedRect best;
    double bestResid = 1e9;
    for (int iter = 0; iter < 8; ++iter) {
        if (cur.size() < 30) break;
        const cv::RotatedRect rr = cv::fitEllipse(cur);
        // 点到椭圆的近似残差：把点变换到单位圆坐标，看 |r-1|
        const double ca = std::cos(rr.angle * CV_PI / 180.0);
        const double sa = std::sin(rr.angle * CV_PI / 180.0);
        const double a = std::max(1.0, rr.size.width * 0.5);
        const double b = std::max(1.0, rr.size.height * 0.5);
        std::vector<double> res(cur.size());
        for (size_t i = 0; i < cur.size(); ++i) {
            const double dx = cur[i].x - rr.center.x;
            const double dy = cur[i].y - rr.center.y;
            const double u = (dx * ca + dy * sa) / a;
            const double v = (-dx * sa + dy * ca) / b;
            res[i] = std::abs(std::hypot(u, v) - 1.0);   // 0 = 恰在椭圆上
        }
        std::vector<double> sorted = res;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const double med = sorted[sorted.size() / 2];
        if (med < bestResid) {
            bestResid = med;
            best = rr;
        }
        std::printf("  迭代%d: 点%5d 轴 %.1fx%.1f 角%.1f 中位残差%.4f\n", iter,
                    static_cast<int>(cur.size()), rr.size.width, rr.size.height, rr.angle, med);
        // 只留残差最小的 60%
        std::vector<double> s2 = res;
        std::sort(s2.begin(), s2.end());
        const double thr = s2[static_cast<size_t>(s2.size() * 0.60)];
        std::vector<cv::Point2f> next;
        next.reserve(cur.size());
        for (size_t i = 0; i < cur.size(); ++i)
            if (res[i] <= thr) next.push_back(cur[i]);
        if (next.size() == cur.size()) break;
        cur.swap(next);
    }

    std::printf("\n=== 拟合椭圆（全图坐标）===\n");
    std::printf("中心 (%.1f, %.1f)  轴长 %.1f x %.1f  旋转 %.2f°\n", best.center.x, best.center.y,
                best.size.width, best.size.height, best.angle);

    // 输出可直接写进 regions.txt 的正外接框 + 供代码用的椭圆参数
    const cv::Rect bb = cv::RotatedRect(best.center, best.size, best.angle).boundingRect();
    std::printf("外接框 %d %d %d %d\n", bb.x, bb.y, bb.width, bb.height);

    // 验证：沿拟合椭圆采样，看有墨比例能不能到 0.9 以上
    // 这是"拟合对没对"的判据 —— 拟合对了，环上处处有墨
    const int N = 720;
    int on = 0;
    std::vector<uchar> onv(N, 0);
    const double ca = std::cos(best.angle * CV_PI / 180.0);
    const double sa = std::sin(best.angle * CV_PI / 180.0);
    const double a = best.size.width * 0.5, b = best.size.height * 0.5;
    for (int t = 0; t < N; ++t) {
        const double th = 2.0 * CV_PI * t / N;
        double bestPeak = 0.0;
        for (int k = -6; k <= 6; ++k) {
            const double aa = a + k, bb2 = b + k;
            const double u = aa * std::cos(th), v = bb2 * std::sin(th);
            const double px = best.center.x + u * ca - v * sa;
            const double py = best.center.y + u * sa + v * ca;
            const int xi = static_cast<int>(std::lround(px));
            const int yi = static_cast<int>(std::lround(py));
            if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
            bestPeak = std::max(bestPeak, static_cast<double>(sig.at<float>(yi, xi)));
        }
        if (bestPeak >= model.config.strokeHi) {
            ++on;
            onv[t] = 1;
        }
    }
    std::printf("沿拟合椭圆一周的有墨比例 %.3f\n", static_cast<double>(on) / N);

    // 最长的连续缺口 —— 这才是断点检测真正要用的量
    int mgl = 0, cur0 = 0, s = -1;
    for (int i = 0; i <= 2 * N; ++i) {
        const int t = i % N;
        if (i >= N && t == 0) continue;
        if (!onv[t]) {
            if (cur0 == 0) s = t;
            ++cur0;
            if (cur0 > mgl && cur0 <= N) mgl = cur0;
        } else {
            cur0 = 0;
        }
    }
    std::printf("最长连续缺口 %.1f°\n", 360.0 * mgl / N);

    // 画出拟合结果供肉眼确认
    cv::Mat vis;
    cv::cvtColor(g(roi).clone(), vis, cv::COLOR_GRAY2BGR);
    cv::ellipse(vis, best, cv::Scalar(0, 0, 255), 1);
    cv::imwrite("out/fitellipse.png", vis);
    std::printf("已写出 out/fitellipse.png\n");
    return 0;
}
