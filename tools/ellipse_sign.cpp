// check_ellipse_pos.cpp —— 椭圆的两种平移符号，哪个才真压在环上
//
// 【为什么需要】
// GUI 里椭圆跟着配准平移，方向可能是 +sh 也可能是 -sh。
// 单看代码推不出来（warpAffine 的 WARP_INVERSE_MAP 让符号绕了一圈），
// 所以直接量：在图上找环的实际亮度中心，看哪个符号的椭圆离它近。
//
// 量法：沿每个椭圆的参数角取一圈采样点，比较两种符号下这一圈上的"墨量"
// （越亮墨越多）。压得准的椭圆，整圈都落在环上，均值高、方差小。
#include <algorithm>
#include <cmath>
#include <cstdio>
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

// 沿椭圆采一圈，返回该圈上的平均亮度。ell 是模型坐标，dx/dy 是额外平移。
static double ringMean(const cv::Mat& img, const chardet::EllipseGeom& ell, double dx, double dy,
                       int n = 720) {
    const double th = ell.angleDeg * CV_PI / 180.0;
    const double ca = std::cos(th), sa = std::sin(th);
    double sum = 0.0;
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        const double t = 2.0 * CV_PI * i / n;
        // 参数式：(a cos t, b sin t) 再按长轴角度旋转
        const double u = ell.a * std::cos(t);
        const double v = ell.b * std::sin(t);
        const int x = cvRound(ell.center.x + dx + u * ca - v * sa);
        const int y = cvRound(ell.center.y + dy + u * sa + v * ca);
        if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) continue;
        sum += img.at<uchar>(y, x);
        ++cnt;
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: markloc_ellcheck <modelRoot> <image...>\n");
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

    std::printf("%-46s %8s %10s %10s %10s\n", "file", "phase", "noShift", "plusSh", "minusSh");
    for (int a = 2; a < argc; ++a) {
        cv::Mat g;
        if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g) || g.size() != model.freq.size()) {
            std::printf("%-46s  (skip: size/read)\n", argv[a]);
            continue;
        }
        // 用原图（不是信号图）量墨量：信号图经过高通过滤，环的灰度被拉平了，
        // 反而不好看出"压没压在环上"。
        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

        // 复算相位相关位移（与 GUI 同法）
        std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);
        cv::Mat r32, s32;
        model.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : units) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
        }
        double dx = 0.0, dy = 0.0;
        if (inkArea.area() > 0) {
            inkArea.x = std::max(0, inkArea.x - 20);
            inkArea.y = std::max(0, inkArea.y - 20);
            inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
            inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            s32(inkArea).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
                dx = sh.x;
                dy = sh.y;
            }
        }

        // 三种落点各量一圈墨量。压制最准的那个应该最高。
        const double mNo = ringMean(g, pat.ell, 0, 0);
        const double mPlus = ringMean(g, pat.ell, dx, dy);
        const double mMinus = ringMean(g, pat.ell, -dx, -dy);

        // 文件名取尾段，便于对齐
        std::string name(QString::fromLocal8Bit(argv[a]).toStdString());
        const auto slash = name.find_last_of("/\\");
        if (slash != std::string::npos) name = name.substr(slash + 1);
        std::printf("%-46s %6.1f,%6.1f %10.1f %10.1f %10.1f\n", name.substr(0, 46).c_str(), dx,
                    dy, mNo, mPlus, mMinus);
    }
    return 0;
}
