// logo_size_probe.cpp —— 量 logo 的实际像素尺寸，判断两批是否同分辨率
//
// 【为什么必须量这个】
// scale_probe 显示 logo 中心位置的比例在 1688 和 1496 两批里差 5%。
// 这有两种可能，含义完全不同：
//
//   可能一：两批是同一物理构图的缩放（只是分辨率不同）
//           → logo 的像素尺寸应与图宽成正比，a/W 恒定
//           → 按比例换算成立
//
//   可能二：两批是不同的构图（镜头距离/对位不同）
//           → logo 的像素尺寸不随图宽变化，a 绝对值接近
//           → 按比例换算是错的，必须用绝对像素 + 逐图定位
//
// 直接量环的像素半径：从椭圆中心沿若干方向找环的墨，得到实际半径。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <QFile>
#include <QFileInfo>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

static bool readGray(const QString& path, cv::Mat& out) {
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
        std::printf("usage: markloc_logosize <tmpl.png> <list.txt>\n");
        return 2;
    }
    cv::Mat tmpl;
    if (!readGray(QString::fromUtf8(argv[1]), tmpl)) return 1;

    std::vector<std::string> files;
    {
        QFile f(QString::fromUtf8(argv[2]));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return 1;
        while (!f.atEnd()) {
            const QString line = QString::fromUtf8(f.readLine()).trimmed();
            if (!line.isEmpty()) files.push_back(line.toStdString());
        }
    }

    // 模板内椭圆：中心与半轴（模板坐标）
    const double tA = 132.1, tB = 76.75, tRot = -13.78;
    const double tCx = 127.8, tCy = 116.0;

    std::printf("%-40s %5s %7s %7s %8s %8s %8s\n", "file", "W", "rxMean", "ryMean",
                "a/W", "a/H", "ncc");
    std::printf("%s\n", std::string(96, '-').c_str());

    for (const auto& p : files) {
        cv::Mat g;
        if (!readGray(QString::fromUtf8(p.c_str()), g)) continue;
        if (g.cols < tmpl.cols || g.rows < tmpl.rows) continue;

        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        const double cx = mloc.x + tCx, cy = mloc.y + tCy;

        chardet::Config cfg;
        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);

        // 沿多个方向从中心向外，找最亮（环的墨）的半径
        // 环是闭合的，所以每个方向都会在某个半径遇到墨
        const double rot = tRot * CV_PI / 180.0;
        const double ca = std::cos(-rot), sa = std::sin(-rot);
        std::vector<double> radii;
        for (int k = 0; k < 72; ++k) {
            const double th = 2.0 * CV_PI * k / 72.0;
            // 模板椭圆上的点（参数角 th）在本图中的方向
            const double u = tA * std::cos(th), v = tB * std::sin(th);
            const double dx = u * ca - v * sa;
            const double dy = u * sa + v * ca;
            const double len = std::hypot(dx, dy);
            if (len < 1e-6) continue;
            const double ux = dx / len, uy = dy / len;
            double bestR = -1, bestVal = 0;
            for (double r = 60; r < 190; r += 0.5) {
                const int px = static_cast<int>(std::lround(cx + ux * r));
                const int py = static_cast<int>(std::lround(cy + uy * r));
                if (px < 0 || py < 0 || px >= sig.cols || py >= sig.rows) break;
                const double val = sig.at<float>(py, px);
                if (val > bestVal) {
                    bestVal = val;
                    bestR = r;
                }
            }
            if (bestR > 0 && bestVal > 0.35) radii.push_back(bestR);
        }
        if (radii.size() < 30) {
            std::printf("%-40s 找不到环\n",
                        QFileInfo(QString::fromUtf8(p.c_str())).fileName().toStdString()
                            .substr(0, 38).c_str());
            continue;
        }
        std::sort(radii.begin(), radii.end());
        const double rMean = radii[radii.size() / 2];
        const QString fn = QFileInfo(QString::fromUtf8(p.c_str())).fileName();
        std::printf("%-40s %5d %7.1f %7.1f %8.4f %8.4f %8.3f\n",
                    fn.toStdString().substr(0, 38).c_str(), g.cols, rMean, rMean,
                    rMean / g.cols, rMean / g.rows, mval);
    }
    return 0;
}
