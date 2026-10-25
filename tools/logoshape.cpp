// logoshape.cpp —— 看清 logo 到底在哪、是什么形状
//
// 【为什么先做这个】
// 我按 regions.txt 里的 r3（474,315,322,225）当椭圆去采样，
// 结果周上有墨比例只有 2%~23%，缺口 160°~330°。
// 这说明这个框不是"包住一个椭圆环"的框 —— 我之前把它当成
// "整个 logo 椭圆的外接框"是猜的，没验证。
//
// 这个工具不看任何假设，直接把 r3 范围内的墨真实地摊开：
//   · 前景像素的行、列投影（墨到底在哪几行哪几列）
//   · 前景像素的质心与二阶矩（判断是环、是块、还是几段线）
//   · 按连通域列出每一块墨的外接框与大小
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
        std::printf("usage: markloc_logoshape <modelDir> <image> [x y w h]\n");
        return 2;
    }
    const QString modelDir = QString::fromLocal8Bit(argv[1]);
    const QString imgPath = QString::fromLocal8Bit(argv[2]);

    chardet::Model model;
    const QString vdir = modelDir.endsWith(QStringLiteral("v00"))
                             ? modelDir
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
        std::printf("size mismatch %dx%d vs model %dx%d\n", g.cols, g.rows, model.freq.cols,
                    model.freq.rows);
        return 1;
    }

    roi &= cv::Rect(0, 0, g.cols, g.rows);
    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    cv::Mat sub = sig(roi).clone();

    std::printf("=== 区域 %dx%d @(%d,%d) ===\n", roi.width, roi.height, roi.x, roi.y);

    double mn = 0, mx = 0;
    cv::minMaxLoc(sub, &mn, &mx);
    cv::Scalar mean, sd;
    cv::meanStdDev(sub, mean, sd);
    std::printf("强度 min=%.4f max=%.4f mean=%.4f sd=%.4f\n", mn, mx, mean[0], sd[0]);

    cv::Mat bin;
    cv::threshold(sub, bin, model.config.strokeHi, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255);
    const int nInk = cv::countNonZero(bin);
    std::printf("墨像素 %d / %d = %.3f\n", nInk, roi.area(),
                static_cast<double>(nInk) / roi.area());
    if (nInk == 0) {
        std::printf("【该区域没有墨】\n");
        return 0;
    }

    std::vector<int> colInk(roi.width, 0), rowInk(roi.height, 0);
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x)
            if (bin.at<uchar>(y, x)) {
                ++colInk[x];
                ++rowInk[y];
            }
    auto dumpProj = [](const char* name, const std::vector<int>& v, int origin) {
        std::printf("\n%s投影（有墨的段）:\n", name);
        int s = -1;
        for (size_t i = 0; i <= v.size(); ++i) {
            const bool on = i < v.size() && v[i] > 0;
            if (on && s < 0) s = static_cast<int>(i);
            if (!on && s >= 0) {
                int peak = 0;
                for (int k = s; k < static_cast<int>(i); ++k) peak = std::max(peak, v[k]);
                std::printf("  %d..%d (长%d, 峰%d)\n", origin + s, origin + static_cast<int>(i) - 1,
                            static_cast<int>(i) - s, peak);
                s = -1;
            }
        }
    };
    dumpProj("列", colInk, roi.x);
    dumpProj("行", rowInk, roi.y);

    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(bin, labels, stats, cents, 8);
    std::printf("\n连通域 %d 个（去背景）：\n", n - 1);
    std::vector<int> order;
    for (int i = 1; i < n; ++i) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return stats.at<int>(a, cv::CC_STAT_AREA) > stats.at<int>(b, cv::CC_STAT_AREA);
    });
    for (size_t k = 0; k < order.size() && k < 20; ++k) {
        const int i = order[k];
        std::printf("  #%d 面积%6d 框 %dx%d @(%d,%d) 质心(%.0f,%.0f)\n", i,
                    stats.at<int>(i, cv::CC_STAT_AREA), stats.at<int>(i, cv::CC_STAT_WIDTH),
                    stats.at<int>(i, cv::CC_STAT_HEIGHT), roi.x + stats.at<int>(i, cv::CC_STAT_LEFT),
                    roi.y + stats.at<int>(i, cv::CC_STAT_TOP), cents.at<double>(i, 0) + roi.x,
                    cents.at<double>(i, 1) + roi.y);
    }

    int sx = 0, sy = 0;
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x)
            if (bin.at<uchar>(y, x)) {
                sx += x;
                sy += y;
            }
    const double cx = static_cast<double>(sx) / nInk;
    const double cy = static_cast<double>(sy) / nInk;
    double mxx = 0, myy = 0, mxy = 0;
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x)
            if (bin.at<uchar>(y, x)) {
                const double dx = x - cx, dy = y - cy;
                mxx += dx * dx;
                myy += dy * dy;
                mxy += dx * dy;
            }
    mxx /= nInk;
    myy /= nInk;
    mxy /= nInk;
    const double tr = mxx + myy;
    const double det = mxx * myy - mxy * mxy;
    const double l1 = tr * 0.5 + std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
    const double l2 = tr * 0.5 - std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
    std::printf("\n墨质心 局部(%.1f, %.1f) 全图(%.0f, %.0f)\n", cx, cy, roi.x + cx, roi.y + cy);
    std::printf("二阶矩主/次轴比 %.2f（环≈1~2，单条直线≈10+）\n",
                l2 > 1e-9 ? std::sqrt(l1 / l2) : 999.0);

    cv::Mat crop = g(roi).clone();
    cv::Mat cropBig;
    cv::resize(crop, cropBig, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
    cv::imwrite("out/logoshape_crop.png", cropBig);
    cv::imwrite("out/logoshape_ink.png", bin);
    std::printf("\n已写出 out/logoshape_crop.png 与 out/logoshape_ink.png\n");
    return 0;
}
