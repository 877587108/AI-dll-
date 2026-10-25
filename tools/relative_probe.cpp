// relative_probe.cpp —— 相对墨量判据的可行性探查
//
// 【第 3 轮的动机】
// 绝对阈值（inkThr / preGrayThr）在第 2 轮被证明不可用：
// ngv2 与 unset 的环墨量分布重叠（p5 都在 0.73~0.98），
// 把阈值抬到能测出 ngv2 的程度，unset 就误报 58 张。
//
// 根本原因是"绝对"：不同图的曝光、墨水浓度、镜头状态都不同，
// 同一个数值在两张图上的含义不一样。
//
// 换相对判据：每个角度算一个"墨量比"
//     ratio(theta) = peak(theta) / median(peak over all theta)
// 即"这个角度的墨，相对这张图自己的环平均水平，够不够"。
// 这样曝光整体偏暗不会影响结论，因为分子分母一起变。
//
// 这个工具只做探查：打印每个样本 ratio 的分布，
// 看 ngv2 与 unset 能不能被一个共同的相对阈值分开。
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

struct RatioStat {
    double minRatio = 1.0;    // 最淡的角度：peak / 中位
    double p5Ratio = 1.0;     // 5% 分位
    int    lowAngles = 0;     // ratio < 0.4 的角度数
    double longestLowDeg = 0; // 连续低墨的最长角度
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: markloc_relprobe <modelRoot> <tmpl.png> <image...>\n");
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
    const double LOWERATIO = argc > 3 ? 0.0 : 0.0;

    std::printf("%-16s %8s %8s %8s %10s\n", "file", "minRatio", "p5Ratio", "lowCnt", "longestDeg");
    for (int a = 3; a < argc; ++a) {
        cv::Mat g;
        if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g)) continue;

        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        const double cx = mloc.x + tplCx, cy = mloc.y + tplCy;
        const double A = tplA, B = tplB;

        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

        const double kScale = A / std::max(1.0, B);
        const double rot = tplRot * CV_PI / 180.0;
        const double cR = std::cos(-rot), sR = std::sin(-rot);
        cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -cx, 0, 1, -cy, 0, 0, 1);
        cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
        cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
        cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
        cv::Mat M3 = T2 * Sc * Rm * T1;
        cv::Mat M;
        M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
        cv::Mat straight;
        cv::warpAffine(sig, straight, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);

        const int pad = static_cast<int>(A * 0.45) + 8;
        cv::Rect roi(static_cast<int>(cx - A - pad), static_cast<int>(cy - A - pad),
                     static_cast<int>(2 * (A + pad)), static_cast<int>(2 * (A + pad)));
        roi &= cv::Rect(0, 0, straight.cols, straight.rows);
        const cv::Point2f cInRoi(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
        const double maxR = A + pad;
        const int nRadius = static_cast<int>(std::lround(maxR));
        cv::Mat pol;
        cv::warpPolar(straight(roi), pol, cv::Size(nRadius, 720), cInRoi, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        cv::Mat polT;
        cv::transpose(pol, polT);
        pol = polT;

        const int bLo = static_cast<int>(std::lround(A * 0.75));
        const int bHi = std::min(static_cast<int>(std::lround(A * 1.35)), pol.rows - 1);
        const int Wcols = pol.cols;

        // 每个角度的墨量峰值
        std::vector<double> peak(Wcols, 0.0);
        for (int x = 0; x < Wcols; ++x) {
            double mx = 0;
            for (int r = bLo; r <= bHi; ++r) mx = std::max(mx, (double)pol.at<float>(r, x));
            peak[x] = mx;
        }
        // 相对判据：除以本图中位数
        std::vector<double> sorted = peak;
        std::sort(sorted.begin(), sorted.end());
        const double med = sorted[sorted.size() / 2];
        RatioStat st;
        if (med > 1e-9) {
            std::vector<double> ratio(Wcols, 0.0);
            for (int x = 0; x < Wcols; ++x) ratio[x] = peak[x] / med;
            std::vector<double> rs = ratio;
            std::sort(rs.begin(), rs.end());
            st.minRatio = rs.front();
            st.p5Ratio = rs[static_cast<size_t>(rs.size() * 0.05)];
            for (double v : ratio)
                if (v < 0.4) ++st.lowAngles;
            // 绕圈找最长连续低墨段
            int best = 0, cur = 0;
            for (int k = 0; k < 2 * Wcols; ++k) {
                if (ratio[k % Wcols] < 0.4) {
                    if (++cur > best) best = cur;
                } else {
                    cur = 0;
                }
            }
            st.longestLowDeg = 360.0 * best / Wcols;
        }
        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::printf("%-16s %8.3f %8.3f %8d %10.2f\n", fn.toStdString().substr(0, 16).c_str(),
                    st.minRatio, st.p5Ratio, st.lowAngles, st.longestLowDeg);
    }
    (void)LOWERATIO;
    return 0;
}
