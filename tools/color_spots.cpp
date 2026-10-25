// color_spots.cpp —— 环上的局部色斑检测
//
// 【第 11 轮】
// 上一轮用环上的**平均**色度，两类重叠。
// 但缺陷可能是**局部**的：环上有一段颜色不对，整圈平均后被稀释。
//
// 做法：沿环的 720 个角度逐个量色度，然后找
//   · 与环自身中位色度的偏离
//   · 连续偏离的最长角度
// 这与"墨量"不同的是，色度对曝光几乎不敏感，
// 所以万一缺陷是"局部烧蚀深浅不同"，会体现在这里。
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
        std::printf("usage: markloc_colorspots <modelRoot> <tmpl.png> <image...>\n");
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
    const int N = 720;

    std::printf("%-18s %9s %9s %9s %9s\n", "file", "chrMed", "chrMin", "minRatio", "lowDeg");
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

        std::vector<cv::Mat> ch;
        cv::split(bgr, ch);
        cv::Mat rg, bg, chr;
        cv::subtract(ch[2], ch[1], rg, cv::noArray(), CV_32F);
        cv::subtract(ch[0], ch[1], bg, cv::noArray(), CV_32F);
        cv::magnitude(rg, bg, chr);

        const double th = tplRot * CV_PI / 180.0;
        const double ca = std::cos(th), sa = std::sin(th);
        std::vector<double> prof(N, 0.0);
        for (int t = 0; t < N; ++t) {
            const double ang = 2.0 * CV_PI * t / N;
            // 沿半径方向取最大值：环有厚度，取最饱和的那个点最能反映刻印
            double best = 0;
            for (double rr = 0.88; rr <= 1.12; rr += 0.02) {
                const double u = A * rr * std::cos(ang), v = B * rr * std::sin(ang);
                const int x = cvRound(cx + u * ca - v * sa);
                const int y = cvRound(cy + u * sa + v * ca);
                if (x < 0 || y < 0 || x >= chr.cols || y >= chr.rows) continue;
                best = std::max(best, (double)chr.at<float>(y, x));
            }
            prof[t] = best;
        }
        std::vector<double> s = prof;
        std::sort(s.begin(), s.end());
        const double med = s[s.size() / 2];
        const double mn = s.front();
        // 相对中位低于 0.5 的角度，找最长连续段
        std::vector<char> low(N, 0);
        for (int t = 0; t < N; ++t) low[t] = (med > 1e-6 && prof[t] < med * 0.5) ? 1 : 0;
        int best = 0, cur = 0;
        for (int k = 0; k < 2 * N; ++k) {
            if (low[k % N]) {
                if (++cur > best) best = cur;
            } else {
                cur = 0;
            }
        }

        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::printf("%-18s %9.2f %9.2f %9.3f %9.2f\n", fn.toStdString().substr(0, 18).c_str(), med,
                    mn, mn / std::max(1e-6, med), 360.0 * best / N);
    }
    return 0;
}
