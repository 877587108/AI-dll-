// fitnew.cpp —— 在新规格图上定位 logo 椭圆
//
// 【为什么需要单独一个工具】
// 新规格图是 1496x1496，旧模型是 1280x1024（由 strokeSignal 与 model.freq 决定）。
// 之前一直拿旧模型去测新图，结果是"完全无效"——不是判据不好，是量纲不对。
// 现在要在新规格上重建区域，得先知道新规格里 logo 到底在哪、多大。
//
// 做法：不猜区域，全图找"椭圆环"。
//   1. 全图算 strokeSignal
//   2. 在左右上半区找一个椭圆环：对候选中心/半轴/角度做网格搜索，
//      沿椭圆采样，要求周上有墨比例最高
//   3. 报告最佳椭圆的中心、半轴、旋转、有墨比例、最长缺口
//
// 网格搜索不是优雅解法，但它不依赖任何先验假设 ——
// 我不知道 logo 在新图里的位置，猜一个区域再拟合，可能又猜错。
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

struct Ell {
    double cx, cy, a, b, rot;
};

static double coverage(const cv::Mat& sig, const Ell& e, int N = 360, double thr = 0.35,
                       int normalHalf = 4, double* maxGapDeg = nullptr) {
    const double ca = std::cos(e.rot), sa = std::sin(e.rot);
    int on = 0;
    std::vector<char> onv(N, 0);
    for (int t = 0; t < N; ++t) {
        const double th = 2.0 * CV_PI * t / N;
        const double u = e.a * std::cos(th), v = e.b * std::sin(th);
        const double px = e.cx + u * ca - v * sa;
        const double py = e.cy + u * sa + v * ca;
        double nx = std::cos(th) / e.a, ny = std::sin(th) / e.b;
        const double nl = std::hypot(nx, ny);
        nx /= nl;
        ny /= nl;
        const double wnx = nx * ca - ny * sa, wny = nx * sa + ny * ca;
        double peak = 0.0;
        for (int k = -normalHalf; k <= normalHalf; ++k) {
            const int xi = static_cast<int>(std::lround(px + wnx * k));
            const int yi = static_cast<int>(std::lround(py + wny * k));
            if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
            peak = std::max(peak, static_cast<double>(sig.at<float>(yi, xi)));
        }
        onv[t] = peak >= thr ? 1 : 0;
        on += onv[t];
    }
    if (maxGapDeg) {
        int mgl = 0, cur = 0;
        for (int i = 0; i < 2 * N; ++i) {
            const int t = i % N;
            if (!onv[t]) {
                ++cur;
                mgl = std::max(mgl, std::min(cur, N));
            } else {
                cur = 0;
            }
        }
        *maxGapDeg = 360.0 * mgl / N;
    }
    return static_cast<double>(on) / N;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: markloc_fitnew <modelDir> <image> [x0 y0 x1 y1]\n");
        return 2;
    }
    const QString modelDir = QString::fromLocal8Bit(argv[1]);
    const QString imgPath = QString::fromLocal8Bit(argv[2]);

    chardet::Model model;
    const QString vdir = modelDir.endsWith(QStringLiteral("v00")) ? modelDir
                                                                 : modelDir + QStringLiteral("/v00");
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat g;
    if (!readGrayU8(imgPath, g)) {
        std::printf("read failed\n");
        return 1;
    }
    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    std::printf("图 %dx%d  模型 %dx%d\n", g.cols, g.rows, model.freq.cols, model.freq.rows);
    if (g.size() != model.freq.size()) {
        std::printf("【尺寸不同】strokeSignal 的归一化基准与模型不同，")
        ;
        std::printf("这在旧链路里是致命的；这里只用信号形状定位，不受影响。\n");
    }

    // 搜索范围：logo 在芯片上表面，通常在上半区偏左
    int x0 = 100, y0 = 100, x1 = g.cols - 100, y1 = g.rows / 2;
    if (argc >= 7) {
        x0 = std::atoi(argv[3]);
        y0 = std::atoi(argv[4]);
        x1 = std::atoi(argv[5]);
        y1 = std::atoi(argv[6]);
    }
    std::printf("搜索区域 (%d,%d)-(%d,%d)\n", x0, y0, x1, y1);

    // 网格搜索。半轴范围按"图宽的 8%~30%"给，因为不知道缩放比例。
    double bestCov = -1;
    Ell best{0, 0, 1, 1, 0};
    const int stepC = 12;
    const int stepA = 8;
    for (int cx = x0; cx <= x1; cx += stepC) {
        for (int cy = y0; cy <= y1; cy += stepC) {
            for (int a = static_cast<int>(g.cols * 0.06); a <= static_cast<int>(g.cols * 0.28);
                 a += stepA) {
                for (int bi = static_cast<int>(g.cols * 0.035);
                     bi <= static_cast<int>(g.cols * 0.16); bi += stepA) {
                    for (int deg = 0; deg < 180; deg += 15) {
                        Ell e{static_cast<double>(cx), static_cast<double>(cy),
                              static_cast<double>(a), static_cast<double>(bi),
                              deg * CV_PI / 180.0};
                        const double cov = coverage(sig, e, 180);
                        if (cov > bestCov) {
                            bestCov = cov;
                            best = e;
                        }
                    }
                }
            }
        }
    }
    std::printf("粗搜最佳：中心(%.0f,%.0f) 半轴%.0f x %.0f 角%.1f° cov=%.3f\n", best.cx, best.cy,
                best.a, best.b, best.rot * 180 / CV_PI, bestCov);

    // 精修：小步长局部搜索
    for (int round = 0; round < 4; ++round) {
        const double sc = round == 0 ? 4.0 : 2.0;
        // 局部枚举
        double localBest = -1;
        Ell localE = best;
        for (int dcx = -8; dcx <= 8; ++dcx) {
            for (int dcy = -8; dcy <= 8; ++dcy) {
                for (int da = -6; da <= 6; ++da) {
                    for (int db = -6; db <= 6; ++db) {
                        for (int dd = -6; dd <= 6; ++dd) {
                            Ell e{best.cx + dcx * sc, best.cy + dcy * sc, best.a + da * sc,
                                  best.b + db * sc, best.rot + dd * 0.004};
                            if (e.a < 5 || e.b < 5) continue;
                            const double cov = coverage(sig, e, 180);
                            if (cov > localBest) {
                                localBest = cov;
                                localE = e;
                            }
                        }
                    }
                }
            }
        }
        best = localE;
        bestCov = localBest;
        std::printf("精修%d：中心(%.1f,%.1f) 半轴%.1f x %.1f 角%.2f° cov=%.4f\n", round, best.cx,
                    best.cy, best.a, best.b, best.rot * 180 / CV_PI, bestCov);
    }

    double gapDeg = 0;
    const double finalCov = coverage(sig, best, 720, 0.35, 5, &gapDeg);
    std::printf("\n=== 结果 ===\n中心 (%.1f, %.1f)  半轴 %.1f x %.1f  旋转 %.2f°\n",
                best.cx, best.cy, best.a, best.b, best.rot * 180 / CV_PI);
    std::printf("有墨比例 %.4f  最长缺口 %.2f°\n", finalCov, gapDeg);
    // regions.txt 需要的是正外接框
    const int bx = static_cast<int>(std::floor(best.cx - std::hypot(best.a * std::cos(best.rot),
                                                                    best.b * std::sin(best.rot))));
    const int by = static_cast<int>(std::floor(best.cy - std::hypot(best.a * std::sin(best.rot),
                                                                    best.b * std::cos(best.rot))));
    const int bw = static_cast<int>(std::ceil(2 * std::hypot(best.a * std::cos(best.rot),
                                                             best.b * std::sin(best.rot))));
    const int bh = static_cast<int>(std::ceil(2 * std::hypot(best.a * std::sin(best.rot),
                                                             best.b * std::cos(best.rot))));
    std::printf("#region N ellipse %d %d %d %d 0\n", bx, by, bw, bh);

    cv::Mat vis;
    cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);
    cv::ellipse(vis,
                cv::RotatedRect(cv::Point2f(static_cast<float>(best.cx), static_cast<float>(best.cy)),
                                cv::Size2f(static_cast<float>(best.a * 2),
                                           static_cast<float>(best.b * 2)),
                                static_cast<float>(best.rot * 180 / CV_PI)),
                cv::Scalar(0, 0, 255), 2);
    cv::imwrite("out/fitnew.png", vis);
    std::printf("已写出 out/fitnew.png\n");
    return 0;
}
