// logo_viz.cpp —— 把定位到的椭圆画出来，肉眼确认定位对不对
//
// 【为什么必须有这一步】
// logo_locate 报"长半轴 224~402"，而这些都是同一个零件的图。
// 这个数不可能是真的——除非拟合根本没找到环。
// 迭代拟合有个通病：点集越筛越少，最后几个残渣点也能拟合出一个"椭圆"，
// 残差还特别小。必须画出来看。
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
        std::printf("usage: markloc_logoviz <modelDir> <image> <out.png> [roiX roiY roiW roiH]\n");
        return 2;
    }
    chardet::Model model;
    const QString vdir =
        QDir(QString::fromLocal8Bit(argv[1])).absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g)) {
        std::printf("read failed\n");
        return 1;
    }

    cv::Mat blurred, sig;
    cv::GaussianBlur(g, blurred, cv::Size(0, 0), model.config.sigma);
    cv::subtract(g, blurred, sig);
    sig.convertTo(sig, CV_32F, 1.0 / 255.0);
    cv::threshold(sig, sig, model.config.strokeHi, 1.0, cv::THRESH_BINARY);
    cv::Mat bin;
    sig.convertTo(bin, CV_8U, 255);

    cv::Rect roi(static_cast<int>(g.cols * 0.10), static_cast<int>(g.rows * 0.15),
                 static_cast<int>(g.cols * 0.55), static_cast<int>(g.rows * 0.30));
    if (argc >= 8)
        roi = cv::Rect(std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]),
                       std::atoi(argv[7]));
    roi &= cv::Rect(0, 0, g.cols, g.rows);

    // 直接看这个 ROI 里墨的实际分布：连通域 + 外接框
    cv::Mat sub = bin(roi);
    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(sub, labels, stats, cents, 8);
    std::printf("ROI %dx%d @(%d,%d)  连通域 %d\n", roi.width, roi.height, roi.x, roi.y, n - 1);

    cv::Mat vis;
    cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);
    cv::rectangle(vis, roi, cv::Scalar(0, 200, 0), 1);

    std::vector<int> order;
    for (int i = 1; i < n; ++i) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return stats.at<int>(a, cv::CC_STAT_AREA) > stats.at<int>(b, cv::CC_STAT_AREA);
    });
    for (size_t k = 0; k < order.size() && k < 6; ++k) {
        const int i = order[k];
        const cv::Rect bb(roi.x + stats.at<int>(i, cv::CC_STAT_LEFT),
                          roi.y + stats.at<int>(i, cv::CC_STAT_TOP),
                          stats.at<int>(i, cv::CC_STAT_WIDTH),
                          stats.at<int>(i, cv::CC_STAT_HEIGHT));
        std::printf("  #%zu 面积%6d 框 %dx%d @(%d,%d)\n", k, stats.at<int>(i, cv::CC_STAT_AREA),
                    bb.width, bb.height, bb.x, bb.y);
        cv::rectangle(vis, bb, cv::Scalar(0, 140, 255), 1);
    }

    // 把最大连通域的所有点拿去拟合，画出结果
    std::vector<cv::Point2f> pts;
    if (!order.empty()) {
        const int big = order[0];
        for (int y = 0; y < roi.height; ++y)
            for (int x = 0; x < roi.width; ++x)
                if (labels.at<int>(y, x) == big)
                    pts.emplace_back(static_cast<float>(roi.x + x), static_cast<float>(roi.y + y));
    }
    if (pts.size() >= 5) {
        const cv::RotatedRect rr = cv::fitEllipse(pts);
        std::printf("  最大连通域 %d 点 -> 椭圆 中心(%.1f,%.1f) 轴 %.1fx%.1f 角%.2f\n",
                    static_cast<int>(pts.size()), rr.center.x, rr.center.y, rr.size.width,
                    rr.size.height, rr.angle);
        cv::ellipse(vis, rr, cv::Scalar(0, 0, 255), 2);
    }

    cv::Rect crop = roi;
    crop.x = std::max(0, crop.x - 40);
    crop.y = std::max(0, crop.y - 40);
    crop.width = std::min(g.cols - crop.x, crop.width + 80);
    crop.height = std::min(g.rows - crop.y, crop.height + 80);
    cv::Mat outv;
    cv::resize(vis(crop), outv, cv::Size(), 1.5, 1.5, cv::INTER_LINEAR);
    cv::imwrite(argv[3], outv);
    std::printf("已写出 %s\n", argv[3]);
    return 0;
}
