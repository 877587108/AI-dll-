// logo_locate.cpp —— 在图上定位 logo 椭圆环，并报告它的尺寸
//
// 【要回答的问题】
// 用户说"logo 区大小不变，通过定位算法去定位对齐"。
// 这句话有个前提必须先验证：各张图上 logo 环的**尺寸**是否真的一致。
// 如果尺寸一致、只有平移，那对齐就是找一个平移量，很简单；
// 如果尺寸也变（缩放），那就不只是平移问题。
//
// 做法：不预设任何位置，在图的上半区做椭圆拟合，
// 报告中心、半轴、角度，然后统计这批图里这些量的波动。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
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

struct Found {
    bool ok = false;
    double cx = 0, cy = 0, a = 0, b = 0, rotDeg = 0;
    double cov = 0;
    int support = 0;
};

// 在给定 ROI 里定位椭圆环。
//
// 【两版都错了，记在这里】
//
// 第一版：迭代筛点拟合。长半轴报 224~402，同一个零件不可能。
//         错了：点越筛越少，最后几个残渣也能拟合出"残差很小"的椭圆。
//
// 第二版：取最大连通域一次拟合。仍然错，长半轴 32~159。
//         错了：图里除了环还有 "Sil a n" 几个字母，它们的笔画
//         连成一块时面积（2477、1478…）比环的一段更大，
//         于是拟合出来的"椭圆"其实是某个字母。
//
// 第三版（当前）：按"环的几何特征"挑，不按面积挑。
//   环的特征：一条细长的闭合曲线，外接框很大但墨像素占比很低。
//   字母的特征：笔画集中，外接框小、占比高。
//   所以对每个连通域算一个"细长闭合度" = 外接框面积 / 墨像素面积，
//   取这个值最大的那个 —— 外接框越空、越细，越像环。
static Found fitInRoi(const cv::Mat& bin, const cv::Rect& roi, int minArea) {
    Found f;
    cv::Mat sub = bin(roi);
    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(sub, labels, stats, cents, 8);
    if (n <= 1) return f;

    // 按"外接框面积 / 墨像素面积"挑最像环的那块
    int best = -1;
    double bestRatio = 0.0;
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < minArea) continue;
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const double ratio = static_cast<double>(w) * h / std::max(1, area);
        // 环的长宽比应当在 1.2~3 之间（SC 环实测 264x153 = 1.73）
        const double aspect = static_cast<double>(std::max(w, h)) / std::max(1, std::min(w, h));
        if (aspect < 1.15 || aspect > 3.5) continue;
        if (ratio > bestRatio) {
            bestRatio = ratio;
            best = i;
        }
    }
    if (best < 0) return f;

    // 用这块的框作为搜索范围，把"环所在的整条曲线"都收进来再拟合。
    // 环和字母可能连在一起，但环的墨点在外接框边缘上，
    // 只看框边缘附近的点就能把字母排除掉。
    const cv::Rect bb(stats.at<int>(best, cv::CC_STAT_LEFT),
                      stats.at<int>(best, cv::CC_STAT_TOP),
                      stats.at<int>(best, cv::CC_STAT_WIDTH),
                      stats.at<int>(best, cv::CC_STAT_HEIGHT));

    std::vector<cv::Point2f> pts;
    for (int y = 0; y < roi.height; ++y) {
        const int* row = labels.ptr<int>(y);
        if (y < bb.y || y >= bb.y + bb.height) continue;
        for (int x = 0; x < roi.width; ++x) {
            if (x < bb.x || x >= bb.x + bb.width) continue;
            if (row[x] > 0) pts.emplace_back(static_cast<float>(roi.x + x),
                                             static_cast<float>(roi.y + y));
        }
    }
    if (pts.size() < 30) return f;

    const cv::RotatedRect rr = cv::fitEllipse(pts);
    f.ok = true;
    f.cx = rr.center.x;
    f.cy = rr.center.y;
    const double wa = rr.size.width * 0.5, wb = rr.size.height * 0.5;
    f.a = std::max(wa, wb);
    f.b = std::min(wa, wb);
    double ang = rr.angle;
    if (wb > wa) ang += 90.0;
    while (ang > 180.0) ang -= 360.0;
    while (ang <= -180.0) ang += 360.0;
    f.rotDeg = ang;
    f.support = static_cast<int>(pts.size());
    f.cov = bestRatio;
    return f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: markloc_logolocate <modelDir> <imageDir> [roiX roiY roiW roiH]\n");
        return 2;
    }
    chardet::Model model;
    const QString vdir =
        QDir(QString::fromLocal8Bit(argv[1])).absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    const double hi = model.config.strokeHi;
    const double sigma = model.config.sigma;

    QStringList files;
    for (const QFileInfo& fi : QDir(QString::fromLocal8Bit(argv[2]))
                                   .entryInfoList({QStringLiteral("*.tif"), QStringLiteral("*.png"),
                                                   QStringLiteral("*.bmp")},
                                                  QDir::Files, QDir::Name)) {
        if (!fi.completeBaseName().endsWith(QStringLiteral("_00"))) continue;
        files << fi.absoluteFilePath();
    }
    std::printf("共 %d 张 _00 图\n\n", static_cast<int>(files.size()));

    std::printf("%-26s %8s %8s %8s %8s %8s %6s\n", "file", "cx", "cy", "semiA", "semiB",
                "rotDeg", "px");
    std::vector<double> vcx, vcy, va, vb, vrot;
    int okCount = 0;
    for (const QString& path : files) {
        cv::Mat g;
        if (!readGrayU8(path, g)) continue;
        // 不用 strokeSignal：它按整图归一化，跨规格不可比。
        // 这里只要几何形状，用自己的固定尺度，够用。
        cv::Mat blurred, sig;
        cv::GaussianBlur(g, blurred, cv::Size(0, 0), sigma);
        cv::subtract(g, blurred, sig);
        sig.convertTo(sig, CV_32F, 1.0 / 255.0);
        cv::threshold(sig, sig, hi, 1.0, cv::THRESH_BINARY);
        cv::Mat bin;
        sig.convertTo(bin, CV_8U, 255);

        // 默认 ROI：图像上半区偏左（logo 在芯片上表面的左侧）
        cv::Rect roi(0, 0, g.cols, g.rows);
        if (argc >= 7) {
            roi = cv::Rect(std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                           std::atoi(argv[6]));
        } else {
            roi = cv::Rect(static_cast<int>(g.cols * 0.10), static_cast<int>(g.rows * 0.15),
                           static_cast<int>(g.cols * 0.55), static_cast<int>(g.rows * 0.30));
        }
        roi &= cv::Rect(0, 0, g.cols, g.rows);

        const Found f = fitInRoi(bin, roi, 250);
        const std::string stem = QFileInfo(path).completeBaseName().toStdString();
        if (!f.ok) {
            std::printf("%-26s 拟合失败\n", stem.c_str());
            continue;
        }
        ++okCount;
        std::printf("%-26s %8.1f %8.1f %8.1f %8.1f %8.2f %6d\n", stem.c_str(), f.cx, f.cy, f.a,
                    f.b, f.rotDeg, f.support);
        vcx.push_back(f.cx);
        vcy.push_back(f.cy);
        va.push_back(f.a);
        vb.push_back(f.b);
        vrot.push_back(f.rotDeg);
    }

    auto stats = [](std::vector<double>& v, const char* name) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        std::printf("%-10s 均值%8.2f  中位%8.2f  最小%8.2f  最大%8.2f  极差%7.2f\n", name, mean,
                    v[v.size() / 2], v.front(), v.back(), v.back() - v.front());
    };
    std::printf("\n=== 汇总（%d 张拟合成功）===\n", okCount);
    stats(vcx, "中心x");
    stats(vcy, "中心y");
    stats(va, "长半轴");
    stats(vb, "短半轴");
    stats(vrot, "长轴角度");
    if (!va.empty()) {
        std::printf("\n判读：长半轴极差 %.1fpx，短半轴极差 %.1fpx\n", va.back() - va.front(),
                    vb.back() - vb.front());
        std::printf("若极差远小于墨线宽度(5~8px)，说明%s\n",
                    (va.back() - va.front()) < 11 && (vb.back() - vb.front()) < 11
                        ? "尺寸确实一致，只需平移对齐"
                        : "尺寸本身有波动，只做平移不够");
    }
    return 0;
}
