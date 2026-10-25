// color_probe.cpp —— 用彩色信息检测 logo 环异常
//
// 【第 10 轮的动机】
// 前九轮全部在**灰度**上工作：所有判据都先 strokeSignal(gray) 再做处理。
// 但这批图是 3 通道彩色（实测 B/G/R 均值 83/75/77，|R-G| 均值约 7），
// 也就是说颜色本身带有信息，转灰度时被丢掉了。
//
// 刻印是激光烧蚀，烧蚀深度不同会改变表面的反射/吸收特性，
// 可能在色度（而不是亮度）上留下痕迹 —— 灰度看不到，色度能看到。
//
// 做法：对环带做"色度"展开，而不是亮度。
//   色度定义：每个像素的 (R-G) 与 (B-G) 两个通道差，与整体亮度无关，
//   因此不受曝光影响（曝光变化会同时抬高三个通道，差值基本不变）。
//
// 输出环带上每个角度的色度统计，供与正常料对比。
#include <algorithm>
#include <cmath>
#include <cstdio>
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

// 读彩色图（BGR）
static bool readColor(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray b = f.readAll();
    f.close();
    if (b.isEmpty()) return false;
    const std::vector<uchar> buf(b.begin(), b.end());
    out = cv::imdecode(buf, cv::IMREAD_COLOR);
    return !out.empty();
}

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
        std::printf("usage: markloc_colorprobe <modelRoot> <tmpl.png> <image...>\n");
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
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), tmpl)) {
        std::printf("模板读取失败\n");
        return 1;
    }
    const double tplCx = 127.8, tplCy = 116.0, tplA = 132.1, tplB = 76.75, tplRot = -13.78;

    std::printf("%-18s %9s %9s %9s %9s %9s\n", "file", "hueRing", "hueBg", "hueDelta",
                "chrStd", "satRatio");
    for (int a = 3; a < argc; ++a) {
        cv::Mat g, bgr;
        if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g)) continue;
        if (!readColor(QString::fromLocal8Bit(argv[a]), bgr)) continue;
        if (bgr.size() != g.size()) continue;

        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        const double cx = mloc.x + tplCx, cy = mloc.y + tplCy;
        const double A = tplA, B = tplB;

        // 色度图：R-G 与 B-G（对曝光不敏感）
        std::vector<cv::Mat> ch;
        cv::split(bgr, ch);   // B,G,R
        cv::Mat rg, bg;
        cv::subtract(ch[2], ch[1], rg, cv::noArray(), CV_32F);   // R-G
        cv::subtract(ch[0], ch[1], bg, cv::noArray(), CV_32F);   // B-G
        // 色度幅值（不区分方向）：饱和度一样的东西，方向可以不同
        cv::Mat chr;
        cv::magnitude(rg, bg, chr);

        // 沿椭圆环采样：环上 vs 环外背景
        const double th = tplRot * CV_PI / 180.0;
        const double ca = std::cos(th), sa = std::sin(th);
        double ringChr = 0, bgChr = 0, ringRg = 0;
        int nRing = 0, nBg = 0;
        std::vector<double> chrRing;
        for (int t = 0; t < 720; ++t) {
            const double ang = 2.0 * CV_PI * t / 720.0;
            // 环上（半径 1.0a）
            const double u = A * std::cos(ang), v = B * std::sin(ang);
            const int x = cvRound(cx + u * ca - v * sa);
            const int y = cvRound(cy + u * sa + v * ca);
            if (x >= 1 && y >= 1 && x < chr.cols - 1 && y < chr.rows - 1) {
                const double c = chr.at<float>(y, x);
                chrRing.push_back(c);
                ringChr += c;
                ringRg += rg.at<float>(y, x);
                ++nRing;
            }
            // 环外（半径 1.55a，肯定在背景上）
            const double uo = A * 1.55 * std::cos(ang), vo = B * 1.55 * std::sin(ang);
            const int xo = cvRound(cx + uo * ca - vo * sa);
            const int yo = cvRound(cy + uo * sa + vo * ca);
            if (xo >= 1 && yo >= 1 && xo < chr.cols - 1 && yo < chr.rows - 1) {
                bgChr += chr.at<float>(yo, xo);
                ++nBg;
            }
        }
        const double mRing = nRing ? ringChr / nRing : 0;
        const double mBg = nBg ? bgChr / nBg : 0;
        const double mRg = nRing ? ringRg / nRing : 0;
        // 环上色度的波动：局部色斑会让这个值变大
        double var = 0;
        for (double v : chrRing) var += (v - mRing) * (v - mRing);
        const double sd = chrRing.empty() ? 0 : std::sqrt(var / chrRing.size());

        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::printf("%-18s %9.3f %9.3f %9.3f %9.3f %9.4f\n", fn.toStdString().substr(0, 18).c_str(),
                    mRing, mBg, mRing - mBg, sd, mRing / std::max(1e-6, mBg));
        (void)mRg;
    }
    return 0;
}
