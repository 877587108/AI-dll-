// polar_viz.cpp —— 极坐标展开验证
//
// 【目的】
// 把 SC logo 的椭圆环按极坐标展开成 (角度, 半径) 矩形，
// 环在展开图上应该是一条水平亮带，断口在角度方向上是亮带上的一段空缺。
//
// 这个工具只做可视化 + 数值统计，不做判定。
// 先把"能不能看见断口"这一步做实，再谈判据。
//
// 关键参数：半径范围 [rInner, rOuter]。
// 椭圆不是圆，同一角度上环的半径随角度变化，
// 所以不能用一个固定的 [rmin,rmax] 卡死 ——
// 要么先把椭圆拉直成正圆再看，要么用足够宽的半径窗。
// 这里两种都出图，对比哪种能让环保持在同一行。
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
    if (argc < 4) {
        std::printf("usage: markloc_polar <modelDir> <image> <outPrefix> [cx cy a b rotDeg]\n");
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
    const std::string prefix = argv[3];

    double cx = 602.0, cy = 440.0, a = 137.0, b = 74.0;
    double rotDeg = 164.54;
    if (argc >= 9) {
        cx = std::atof(argv[4]);
        cy = std::atof(argv[5]);
        a = std::atof(argv[6]);
        b = std::atof(argv[7]);
        rotDeg = std::atof(argv[8]);
    }

    cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    const double rot = rotDeg * CV_PI / 180.0;

    // ---- 方法 A：先把椭圆拉直成正圆，再极坐标展开
    //
    // 椭圆在同一角度上的半径不是常数（长轴方向 137、短轴方向 74），
    // 直接极坐标展开会得到一条波浪线，而不是一条水平带，
    // 后面的"按半径限制找断口"就没法做。
    //
    // 变换按四步显式构造，不用 getRotationMatrix2D ——
    // 第一版用了它并叠加缩放，展开图上的环是弯的，
    // 因为该函数返回的矩阵已经含"把旋转中心搬回该点"的平移，
    // 再乘缩放就把中心挪走了。
    //
    //   1. 平移到原点       p1 = p - c
    //   2. 旋转 -rot        p2 = R(-rot) p1
    //   3. y 方向除以 k     p3 = S p2        (k = a/b)
    //   4. 平移回去         p4 = p3 + c
    const double kScale = a / std::max(1.0, b);
    const double cR = std::cos(-rot), sR = std::sin(-rot);
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -cx, 0, 1, -cy, 0, 0, 1);
    cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
    // 乘以 kScale（把短轴 b 拉长到 a），不是除以。
    // 第一版写成除以，自检报短轴端距 40（应 137）——这个自检就是为抓这个错误加的。
    cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
    cv::Mat M3 = T2 * Sc * Rm * T1;
    cv::Mat M;
    M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
    cv::Mat sigStraight;
    cv::warpAffine(sig, sigStraight, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);
    std::printf("拉直变换：缩放比 a/b = %.3f\n", kScale);

    // 自检：长短轴端点变换后应都落在离中心 a 的位置（正圆）
    {
        auto apply = [&](double px, double py, double* ox, double* oy) {
            *ox = M.at<double>(0, 0) * px + M.at<double>(0, 1) * py + M.at<double>(0, 2);
            *oy = M.at<double>(1, 0) * px + M.at<double>(1, 1) * py + M.at<double>(1, 2);
        };
        double x0, y0, x1, y1;
        apply(cx + a * std::cos(rot), cy + a * std::sin(rot), &x0, &y0);
        apply(cx + b * std::cos(rot + CV_PI / 2), cy + b * std::sin(rot + CV_PI / 2), &x1, &y1);
        const double d0 = std::hypot(x0 - cx, y0 - cy);
        const double d1 = std::hypot(x1 - cx, y1 - cy);
        std::printf("自检：长轴端距中心 %.2f(应%.2f)  短轴端距中心 %.2f(应%.2f)%s\n", d0, a, d1, a,
                    (std::abs(d0 - a) < 2 && std::abs(d1 - a) < 2) ? "  OK" : "  <<< 变换有误");
    }

    // ---- 极坐标展开
    const double maxR = a * 1.35;
    // 【warpPolar 的 dsize 约定，别记错】
    //   dsize.width  -> 半径方向的分辨率（像素数）
    //   dsize.height -> 角度方向的分辨率（像素数）
    // 第一版写成了 Size(1440, 185)，等于"1440 个半径 x 185 个角度"，
    // 恰好反了，展开图上的环是散开的，不是一条带。
    // 参考资料：https://cloud.tencent.com.cn/developer/article/2370931?policyId=1003
    const int nAngle = 1440;
    const int nRadius = static_cast<int>(std::lround(maxR));
    cv::Mat pol;
    cv::warpPolar(sigStraight, pol, cv::Size(nRadius, nAngle),
                  cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)), maxR,
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    // 现在 pol 的 cols = 半径 0..maxR，rows = 角度 0..360。
    // 转成"横轴=角度、纵轴=半径"方便看和做半径限制。
    cv::Mat polT;
    cv::transpose(pol, polT);
    pol = polT;
    std::printf("极坐标图 %dx%d（宽=角度 0..360° %d 列，高=半径 0..%.1f %d 行）\n", pol.cols, pol.rows,
                nAngle, maxR, nRadius);

    // 归一化到 0..255 便于看图
    cv::Mat polV;
    pol.convertTo(polV, CV_8U, 255.0 / std::max(1.0, model.config.strokeHi * 2.5));
    cv::imwrite(prefix + "_polar.png", polV);

    // 把环附近放大也存一张。transpose 之后 rows=半径，所以按行裁。
    const int rc = static_cast<int>(std::lround(a));
    cv::Rect band(0, std::max(0, rc - 20), pol.cols, std::min(40, pol.rows - std::max(0, rc - 20)));
    band &= cv::Rect(0, 0, pol.cols, pol.rows);
    if (band.area() > 0) {
        cv::Mat bandV;
        cv::resize(polV(band), bandV, cv::Size(), 1.0, 3.0, cv::INTER_NEAREST);
        cv::imwrite(prefix + "_band.png", bandV);
        std::printf("环带（半径 %d±20，纵向放大 3 倍）-> %s_band.png\n", rc, prefix.c_str());
    }
    cv::Mat straightV;
    const double scale = 255.0 / std::max(1.0, model.config.strokeHi * 2.5);
    sigStraight.convertTo(straightV, CV_8U, scale);
    cv::Rect crop(std::max(0, static_cast<int>(cx - a * 1.2)),
                  std::max(0, static_cast<int>(cy - a * 1.2)), std::min(g.cols, static_cast<int>(a * 2.4)),
                  std::min(g.rows, static_cast<int>(a * 2.4)));
    crop &= cv::Rect(0, 0, g.cols, g.rows);
    if (crop.area() > 0) cv::imwrite(prefix + "_straight.png", straightV(crop));

    // ---- 数值：对每一列（角度）在环带内取最大值，看周向的 0/1
    //
    // 半径限制就落在这里：只在 [rc - halfBand, rc + halfBand] 这一圈里找墨。
    // 环外面的笔画（SC 两个字母）不在这个半径范围内，天然被排除。
    // 拉直之后环正好落在半径 a 上，所以中心取 a、窗口取 ±halfBand。
    const int halfBand = 12;
    std::vector<double> perAngle(pol.cols, 0.0);
    for (int x = 0; x < pol.cols; ++x) {
        double peak = 0.0;
        for (int r = std::max(0, rc - halfBand); r <= std::min(pol.rows - 1, rc + halfBand); ++r) {
            peak = std::max(peak, static_cast<double>(pol.at<float>(r, x)));
        }
        perAngle[x] = peak;
    }
    int on = 0;
    std::vector<char> onv(pol.cols, 0);
    for (int x = 0; x < pol.cols; ++x) {
        onv[x] = perAngle[x] >= model.config.strokeHi;
        on += onv[x];
    }
    std::printf("环带内（半径 %d±%d）有墨比例 %.4f\n", rc, halfBand,
                static_cast<double>(on) / pol.cols);

    // 【补救：跟踪环的半径曲线，而不是用固定窗口】
    //
    // 下面的扫描证明了一条死路：
    //   ±2 能看见 10° 缺口，±8 缺口就完全消失。
    // 原因不是窗口"碰巧"补上了，而是**环在极坐标里的半径随角度起伏**：
    // 拉直只是把椭圆变成圆，刻印本身不圆、中心估计也有偏差，
    // 于是环在极坐标图上是一条上下摆动的波浪，不是水平直线。
    // 窗口窄了，波谷处正常有墨的地方被当成缺口；
    // 窗口宽了，真缺口被两侧的墨填平。两头都不行。
    //
    // 正确做法：先按角度跟踪出环的实际半径 r(θ)，再沿这条曲线判有无墨。
    // 跟踪办法：对每个角度，在窗口内找亮度最大的那个半径（即环所在位置），
    // 再用中值滤波把 r(θ) 抹平（缺口的几个角度没有墨，
    // 那里的 argmax 会乱跳，必须靠邻居把它压回去）。
    std::printf("\n=== 沿半径曲线跟踪（不用固定窗口）===\n");
    {
        const int trackHalf = 25;
        std::vector<double> rTheta(pol.cols, rc);
        std::vector<double> peakTheta(pol.cols, 0.0);
        for (int x = 0; x < pol.cols; ++x) {
            double bestV = -1e9;
            int bestR = rc;
            for (int r = std::max(0, rc - trackHalf); r <= std::min(pol.rows - 1, rc + trackHalf);
                 ++r) {
                const double v = pol.at<float>(r, x);
                if (v > bestV) {
                    bestV = v;
                    bestR = r;
                }
            }
            peakTheta[x] = bestV;
            rTheta[x] = bestR;
        }
        // 中值滤波抹平 r(θ)：半窗 15 个角度（约 3.75°）。
        // 缺口窄于这个宽度，缺口处的 r 会被邻居拉回正常位置；
        // 环半径的正常起伏周期以几十度计，会被保留。
        std::vector<double> rSmooth(pol.cols, rc);
        const int medW = 15;
        std::vector<double> buf;
        for (int x = 0; x < pol.cols; ++x) {
            buf.clear();
            for (int k = -medW; k <= medW; ++k) buf.push_back(rTheta[(x + k + pol.cols) % pol.cols]);
            std::nth_element(buf.begin(), buf.begin() + buf.size() / 2, buf.end());
            rSmooth[x] = buf[buf.size() / 2];
        }
        {
            double mn = 1e9, mx = -1e9;
            for (double v : rSmooth) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            std::printf("环半径跟踪：范围 %.0f..%.0f（起伏 %.0f px）  a=%.0f\n", mn, mx, mx - mn, a);
            std::printf("起伏 %s\n", (mx - mn) > 3 ? "超过 3px，固定窗口不可用" : "较小，固定窗口勉强可用");
        }
        for (int side : {1, 2, 3}) {
            int cnt = 0;
            std::vector<char> v(pol.cols, 0);
            for (int x = 0; x < pol.cols; ++x) {
                const int cc = static_cast<int>(std::lround(rSmooth[x]));
                double pk = 0.0;
                for (int k = -side; k <= side; ++k) {
                    const int r = std::clamp(cc + k, 0, pol.rows - 1);
                    pk = std::max(pk, static_cast<double>(pol.at<float>(r, x)));
                }
                v[x] = pk >= model.config.strokeHi;
                cnt += v[x];
            }
            int m = 0, c = 0;
            const int Wc = pol.cols;
            for (int i = 0; i < 2 * Wc; ++i) {
                if (!v[i % Wc]) {
                    ++c;
                    m = std::max(m, std::min(c, Wc));
                } else
                    c = 0;
            }
            std::printf("沿曲线±%d：有墨比例 %.4f  最长缺口 %.2f°\n", side,
                        static_cast<double>(cnt) / Wc, 360.0 * m / Wc);
        }
    }

    // 【窗口宽度扫描】
    // 上一轮报"有墨比例 1.0000、最长缺口 0"，但拉直图上半点钟方向明明断了一截。
    // 最大嫌疑是半径窗口 ±12 太宽：环在极坐标图上不是严格水平的一条线，
    // 而是上下起伏的波浪，窗口一宽，缺口两侧的墨就在半径方向补上了。
    //
    // 把窗口从窄到宽扫一遍，看缺口什么时候消失。
    // 窄窗口能看见、宽窗口看不见，就说明问题在窗口，不在算法别的地方。
    std::printf("\n窗口扫描（每列取半径窗口内最大值，是否过门槛）：\n");
    std::printf("%8s %10s %12s\n", "±半宽", "有墨比例", "最长缺口°");
    for (int hb : {2, 3, 4, 5, 6, 8, 10, 12, 16, 20}) {
        std::vector<char> v(pol.cols, 0);
        int cnt = 0;
        for (int x = 0; x < pol.cols; ++x) {
            double pk = 0.0;
            for (int r = std::max(0, rc - hb); r <= std::min(pol.rows - 1, rc + hb); ++r)
                pk = std::max(pk, static_cast<double>(pol.at<float>(r, x)));
            v[x] = pk >= model.config.strokeHi;
            cnt += v[x];
        }
        int m = 0, c = 0;
        const int Wc = pol.cols;
        for (int i = 0; i < 2 * Wc; ++i) {
            if (!v[i % Wc]) {
                ++c;
                m = std::max(m, std::min(c, Wc));
            } else
                c = 0;
        }
        std::printf("%8d %10.4f %12.2f\n", hb, static_cast<double>(cnt) / Wc, 360.0 * m / Wc);
    }

    // 最长缺口
    const int W = pol.cols;
    int mgl = 0, cur = 0;
    for (int i = 0; i < 2 * W; ++i) {
        if (!onv[i % W]) {
            ++cur;
            mgl = std::max(mgl, std::min(cur, W));
        } else
            cur = 0;
    }
    std::printf("最长缺口 %.2f°\n", 360.0 * mgl / W);

    // 列出缺口段
    std::printf("\n缺口段（>=0.5°）：\n");
    int startAt = -1;
    for (int t = 0; t < W; ++t)
        if (onv[t]) { startAt = t; break; }
    if (startAt >= 0) {
        int c2 = 0, cs = -1;
        for (int k = 1; k <= W; ++k) {
            const int t = (startAt + k) % W;
            if (!onv[t]) {
                if (c2 == 0) cs = t;
                ++c2;
            } else if (c2 > 0) {
                const double deg = 360.0 * c2 / W;
                if (deg >= 0.5) std::printf("  %.2f° 起始 %.1f°\n", deg, 360.0 * cs / W);
                c2 = 0;
            }
        }
    }
    return 0;
}
