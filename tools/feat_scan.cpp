// feat_scan.cpp —— 一次扫描多个候选特征，找能区分两类的维度
//
// 【背景】
// 前六轮查过：角度缺口、环粗细、墨量峰值、绝对阈值、相对比值、拓扑条数，
// 全部无法区分 ngv2(7) 与 unset(158)。环上的"墨"这个维度已经查尽。
//
// 这一轮换思路：不再只盯环上，改为扫描一批**互相独立**的统计量，
// 包括整图、环带的空域纹理、频域能量、局部曲率等。
// 目的是先用广撒网的方式找候选，再对候选做严格验证。
//
// 输出一行一个样本，列是各特征，方便外部排序 / 找分界。
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

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: markloc_featscan <modelRoot> <tmpl.png> <image...>\n");
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

    std::printf("%-18s", "file");
    std::printf("%8s %8s %8s %8s %8s %8s %8s %8s\n", "peak", "gStd", "ringThk", "thkStd",
                "fftHi", "gradMag", "entropy", "curvStd");
    for (int ai = 3; ai < argc; ++ai) {
        cv::Mat g;
        if (!readGrayU8(QString::fromLocal8Bit(argv[ai]), g)) continue;

        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        const double cx = mloc.x + tplCx, cy = mloc.y + tplCy;
        const double A = tplA, B = tplB;

        // 取 logo 邻域（比环略大），后面的纹理/频域统计都在这一块上做
        const int pad = static_cast<int>(A * 0.45) + 8;
        cv::Rect roi(static_cast<int>(cx - A - pad), static_cast<int>(cy - A - pad),
                     static_cast<int>(2 * (A + pad)), static_cast<int>(2 * (A + pad)));
        roi &= cv::Rect(0, 0, g.cols, g.rows);
        if (roi.area() <= 0) continue;
        const cv::Mat gRoi = g(roi);

        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

        // 拉直 + 极坐标（与判据一致）
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

        // --- 特征 1：整块 logo 区域的灰度标准差（对比度）
        cv::Scalar mu, sd;
        cv::meanStdDev(gRoi, mu, sd);

        // --- 特征 2：环的厚度（每条墨带沿半径方向的宽度）及其波动
        std::vector<double> thick;
        for (int x = 0; x < Wcols; x += 4) {
            int run = 0, best = 0;
            for (int r = bLo; r <= bHi; ++r) {
                if (pol.at<float>(r, x) >= 0.35f) {
                    if (++run > best) best = run;
                } else {
                    run = 0;
                }
            }
            thick.push_back(best);
        }
        const double thkMean = [&] {
            double s = 0;
            for (double v : thick) s += v;
            return s / std::max<size_t>(1, thick.size());
        }();
        double thkVar = 0;
        for (double v : thick) thkVar += (v - thkMean) * (v - thkMean);
        const double thkStd = std::sqrt(thkVar / std::max<size_t>(1, thick.size()));

        // --- 特征 3：高频能量占比（刻印边缘的锐利度）
        cv::Mat g32;
        gRoi.convertTo(g32, CV_32F);
        cv::Mat lap;
        cv::Laplacian(g32, lap, CV_32F, 3);
        const double fftHi = cv::norm(lap) / std::max(1.0, (double)gRoi.total());

        // --- 特征 4：梯度幅值均值（整体的"墨感"强弱）
        cv::Mat gx, gy;
        cv::Sobel(g32, gx, CV_32F, 1, 0, 3);
        cv::Sobel(g32, gy, CV_32F, 0, 1, 3);
        cv::Mat mag;
        cv::magnitude(gx, gy, mag);
        const double gradMag = cv::mean(mag)[0];

        // --- 特征 5：灰度熵（信息量；断墨会让分布更集中）
        int hist[256] = {0};
        for (int y = 0; y < gRoi.rows; ++y)
            for (int x = 0; x < gRoi.cols; ++x) ++hist[gRoi.at<uchar>(y, x)];
        double ent = 0;
        const double tot = static_cast<double>(gRoi.total());
        for (int i = 0; i < 256; ++i) {
            if (!hist[i]) continue;
            const double p = hist[i] / tot;
            ent -= p * std::log2(p);
        }

        // --- 特征 6：环半径曲线的二阶差分（局部曲率突变；断口/毛刺会体现）
        std::vector<double> rProf(Wcols, 0.0);
        for (int x = 0; x < Wcols; ++x) {
            int bestR = bLo;
            double bv = -1;
            for (int r = bLo; r <= bHi; ++r) {
                const double v = pol.at<float>(r, x);
                if (v > bv) {
                    bv = v;
                    bestR = r;
                }
            }
            rProf[x] = bestR;
        }
        std::vector<double> curv;
        for (int x = 0; x < Wcols; ++x) {
            const double d2 = rProf[(x + 1) % Wcols] - 2 * rProf[x] + rProf[(x - 1 + Wcols) % Wcols];
            curv.push_back(std::abs(d2));
        }
        const double curvStd = std::sqrt([&] {
            const double m = [&] {
                double s = 0;
                for (double v : curv) s += v;
                return s / curv.size();
            }();
            double s = 0;
            for (double v : curv) s += (v - m) * (v - m);
            return s / curv.size();
        }());

        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[ai])).fileName();
        std::printf("%-18s%8.3f %8.2f %8.2f %8.2f %8.4f %8.3f %8.3f %8.3f\n",
                    fn.toStdString().substr(0, 18).c_str(), mval, sd[0], thkMean, thkStd, fftHi,
                    gradMag, ent, curvStd);
        (void)median;
    }
    return 0;
}
