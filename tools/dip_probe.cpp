// dip_probe.cpp —— 区段内的"整体凹陷"度量
//
// 【上一版的错误】
// seg_probe 用 peakMin（区段内最低的墨量）作为判据。
// 但最小值对单个噪声点极其敏感：正常料只要有一个角度偶发低值，
// peakMin 就会很低，于是误报 64 张。收窄区段反而让误报更多，就是这个原因。
//
// 正确的量应该回答"这一整段是不是都偏低"，而不是"有没有一个点偏低"。
// 这里改用三种更稳健的度量：
//   · segMean  ：区段内墨量的平均值（整体水平）
//   · segP25   ：区段内 25% 分位（忽略最低的噪声，反映"这段的下限有多低"）
//   · dipDepth ：区段均值相对"全环中位"的凹陷深度，即 (环中位 - 区段均值)
//                —— 这是**相对**度量，能吸收整张图的曝光与墨水浓度差异
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

int main(int argc, char** argv) {
    if (argc < 6) {
        std::printf("usage: markloc_dipprobe <modelRoot> <tmpl.png> <fromDeg> <toDeg> <image...>\n");
        return 2;
    }
    const double fromDeg = std::atof(argv[3]);
    const double toDeg = std::atof(argv[4]);

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

    std::printf("区段 %.0f~%.0f 度\n", fromDeg, toDeg);
    std::printf("%-18s %8s %8s %9s %8s\n", "file", "ringMed", "segMean", "segP25", "dipDepth");
    for (int a = 5; a < argc; ++a) {
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
        cv::warpPolar(straight(roi), pol, cv::Size(nRadius, N), cInRoi, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        cv::Mat polT;
        cv::transpose(pol, polT);
        pol = polT;

        const int bLo = static_cast<int>(std::lround(A * 0.72));
        const int bHi = std::min(static_cast<int>(std::lround(A * 1.38)), pol.rows - 1);

        // 每个角度的墨量峰值
        std::vector<double> prof(N, 0.0);
        for (int x = 0; x < N; ++x) {
            double pk = 0;
            for (int r = bLo; r <= bHi; ++r) pk = std::max(pk, (double)pol.at<float>(r, x));
            prof[x] = pk;
        }
        // 先做环形中值滤波（窗口 11 个采样 ≈ 5.5 度），去掉单点噪声
        std::vector<double> smooth(N, 0.0);
        {
            const int hw = 5;
            std::vector<double> buf;
            for (int x = 0; x < N; ++x) {
                buf.clear();
                for (int k = -hw; k <= hw; ++k) buf.push_back(prof[(x + k + N) % N]);
                std::nth_element(buf.begin(), buf.begin() + buf.size() / 2, buf.end());
                smooth[x] = buf[buf.size() / 2];
            }
        }

        std::vector<double> ring = smooth;
        std::sort(ring.begin(), ring.end());
        const double ringMed = ring[ring.size() / 2];

        const int t0 = static_cast<int>(std::lround(fromDeg * N / 360.0));
        const int t1 = static_cast<int>(std::lround(toDeg * N / 360.0));
        std::vector<double> seg;
        for (int t = t0; t < t1; ++t) seg.push_back(smooth[((t % N) + N) % N]);
        if (seg.empty()) continue;
        double sum = 0;
        for (double v : seg) sum += v;
        const double segMean = sum / seg.size();
        std::vector<double> ss = seg;
        std::sort(ss.begin(), ss.end());
        const double segP25 = ss[ss.size() * 25 / 100];

        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::printf("%-18s %8.3f %8.3f %9.3f %8.3f\n", fn.toStdString().substr(0, 18).c_str(),
                    ringMed, segMean, segP25, ringMed - segMean);
    }
    return 0;
}
