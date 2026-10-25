// ring_scan.cpp —— 沿椭圆环逐角度打印采样值
//
// 【要回答的问题】
// ngv2/1_00 的 SC 环左上角肉眼可见一个明显缺口（见 out/ngv2_logo_zoom.png），
// 但 fitnew 报"最长缺口 1.50°"，几乎认为环是完好的。
//
// 这个矛盾必须查清楚：
//   要么缺口真的很小（肉眼看到的是错觉）
//   要么采样方法把缺口跨过去了（法向剖面取最大值，跨度太大）
//
// 做法：不猜，把每个角度上的采样值全打出来，
// 同时给出"法向取最大值"和"只用中心单点"两种结果，对比看差多少。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

int main(int argc, char** argv) {
    if (argc < 8) {
        std::printf("usage: markloc_ringscan <modelDir> <image> cx cy a b rotDeg [anglesN]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat g;
    if (!readGrayU8(QString::fromLocal8Bit(argv[2]), g)) {
        std::printf("read failed\n");
        return 1;
    }
    const double cx = std::atof(argv[3]), cy = std::atof(argv[4]);
    const double a = std::atof(argv[5]), b = std::atof(argv[6]);
    const double rot = std::atof(argv[7]) * CV_PI / 180.0;
    const int N = argc >= 9 ? std::atoi(argv[8]) : 720;

    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    const double inkThr = model.config.strokeHi;
    std::printf("椭圆 中心(%.1f,%.1f) 半轴 %.1f x %.1f 角 %.2f°  N=%d  门槛 %.3f\n", cx, cy, a, b,
                rot * 180 / CV_PI, N, inkThr);

    const double ca = std::cos(rot), sa = std::sin(rot);

    // 两种采样：法向取最大值 / 只用中心
    auto sample = [&](double th, int halfK, double* px, double* py) {
        const double u = a * std::cos(th), v = b * std::sin(th);
        const double x = cx + u * ca - v * sa;
        const double y = cy + u * sa + v * ca;
        double nx = std::cos(th) / a, ny = std::sin(th) / b;
        const double nl = std::hypot(nx, ny);
        nx /= nl;
        ny /= nl;
        const double wnx = nx * ca - ny * sa;
        const double wny = nx * sa + ny * ca;
        double peak = 0.0;
        for (int k = -halfK; k <= halfK; ++k) {
            const int xi = static_cast<int>(std::lround(x + wnx * k));
            const int yi = static_cast<int>(std::lround(y + wny * k));
            if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
            peak = std::max(peak, static_cast<double>(sig.at<float>(yi, xi)));
        }
        if (px) *px = x;
        if (py) *py = y;
        return peak;
    };

    // 打印每个角度的值，文件方便细看
    FILE* fp = std::fopen("out/ringscan.csv", "wb");
    if (fp) std::fprintf(fp, "deg,x,y,peakHalf5,peakHalf4,peakCenter\n");

    int onHalf5 = 0, onHalf4 = 0, onCenter = 0;
    std::vector<char> v5(N, 0), v4(N, 0), v0(N, 0);
    for (int t = 0; t < N; ++t) {
        const double th = 2.0 * CV_PI * t / N;
        double px, py;
        const double p5 = sample(th, 5, &px, &py);
        const double p4 = sample(th, 4, nullptr, nullptr);
        const double p0 = sample(th, 0, nullptr, nullptr);
        v5[t] = p5 >= inkThr;
        v4[t] = p4 >= inkThr;
        v0[t] = p0 >= inkThr;
        onHalf5 += v5[t];
        onHalf4 += v4[t];
        onCenter += v0[t];
        if (fp) std::fprintf(fp, "%.1f,%.1f,%.1f,%.4f,%.4f,%.4f\n", 360.0 * t / N, px, py, p5, p4, p0);
    }
    if (fp) std::fclose(fp);

    std::printf("\n有墨比例：法向±5 %.4f   法向±4 %.4f   中心单点 %.4f\n",
                static_cast<double>(onHalf5) / N, static_cast<double>(onHalf4) / N,
                static_cast<double>(onCenter) / N);

    // 最长连续缺口
    auto maxGap = [&](const std::vector<char>& v) {
        int mgl = 0, cur = 0;
        for (int i = 0; i < 2 * N; ++i) {
            if (!v[i % N]) {
                ++cur;
                mgl = std::max(mgl, std::min(cur, N));
            } else
                cur = 0;
        }
        return 360.0 * mgl / N;
    };
    std::printf("最长缺口：法向±5 %.2f°   法向±4 %.2f°   中心单点 %.2f°\n", maxGap(v5), maxGap(v4),
                maxGap(v0));

    // 列出中心单点采样下的所有缺口段（>=1°）
    std::printf("\n中心单点采样下的缺口段（>=1°）：\n");
    int startAt = -1;
    for (int t = 0; t < N; ++t)
        if (v0[t]) { startAt = t; break; }
    if (startAt < 0) {
        std::printf("  整圈无墨\n");
        return 0;
    }
    int cur = 0, cs = -1;
    for (int k = 1; k <= N; ++k) {
        const int t = (startAt + k) % N;
        if (!v0[t]) {
            if (cur == 0) cs = t;
            ++cur;
        } else if (cur > 0) {
            const double deg = 360.0 * cur / N;
            if (deg >= 1.0) {
                const int tMid = (cs + cur / 2) % N;
                const double thMid = 2.0 * CV_PI * tMid / N;
                double mx, my;
                sample(thMid, 0, &mx, &my);
                std::printf("  %.2f° 起始%.1f° @(%.0f,%.0f)  法向±5的值 %.3f\n", deg,
                            360.0 * cs / N, mx, my, sample(thMid, 5, nullptr, nullptr));
            }
            cur = 0;
        }
    }
    return 0;
}
