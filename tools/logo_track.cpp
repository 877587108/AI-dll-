// logo_track.cpp —— 用模板匹配逐图定位 logo，再用椭圆量环的完整性
//
// 【为什么这个方法，而不是前几版的"拟合"】
// 前三版都在图里"找椭圆"：迭代筛点、最大连通域、按形状挑连通域。
// 全部失败 —— 图里和环一样显眼的东西太多（Si l a n 几个字母、
// 定位圆点、引脚高光），纯几何判据选不对。
//
// 模板匹配不需要"猜哪块是环"：拿一张环完好的图裁出模板，
// 在每张图上找它的最佳位置。零件位置变了它也跟得上。
//
// 定位之后做什么：
//   1. 沿椭圆采一圈，量"有墨比例"和周向最长缺口 —— 这是环断没断
//   2. 量半径方向的墨量分布 —— 这是环缺没缺墨（变淡）
// 缺墨和断裂是两件事，分开量。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
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
    if (argc < 4) {
        std::printf("usage: markloc_logotrack <modelDir> <imageDir> <tmpl.png> [--csv out.csv]\n");
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
    if (!readGrayU8(QString::fromLocal8Bit(argv[3]), tmpl)) {
        std::printf("模板读取失败\n");
        return 1;
    }
    // 模板里的椭圆参数（相对模板左上角）。
    //
    // 【第一版这几个数是凭 232x176 的外接框估的，估错了】
    // 估出来的中心 (147,117) 实际应为 (127.8,116)，差了 19px ——
    // 结果全批"环覆盖 0.71、最长缺口 60°"，把好好的环说成断的。
    //
    // 现在是实测值：对模板图做连通域椭圆拟合得到
    //   中心(607.8,416.0) 轴 264.2x153.5 角 76.22°（RotatedRect 约定，
    //   size.width 沿 angle 方向，所以长轴角度是 76.22+90 = -13.78°）
    // 模板裁剪起点 (480,300)，换算到模板内即下面这四个数。
    //
    // 程序启动时会先在模板图上自检：覆盖应该接近 1.0，
    // 否则说明这几个数和模板对不上，测出来的结果都不能信。
    const double tplCx = 127.8, tplCy = 116.0;
    const double tplA = 132.1, tplB = 76.75;
    const double tplRotDeg = -13.78;

    QString csvPath;
    for (int i = 4; i + 1 < argc; i += 2)
        if (!std::strcmp(argv[i], "--csv")) csvPath = QString::fromLocal8Bit(argv[i + 1]);

    QStringList files;
    for (const QFileInfo& fi : QDir(QString::fromLocal8Bit(argv[2]))
                                   .entryInfoList({QStringLiteral("*.tif"), QStringLiteral("*.png"),
                                                   QStringLiteral("*.bmp")},
                                                  QDir::Files, QDir::Name)) {
        if (!fi.completeBaseName().endsWith(QStringLiteral("_00"))) continue;
        files << fi.absoluteFilePath();
    }
    std::printf("模板 %dx%d，共 %d 张 _00 图\n", tmpl.cols, tmpl.rows,
                static_cast<int>(files.size()));

    // 【自检：先在模板自身上量一遍】
    // 这几个椭圆参数如果和模板对不上，后面全批的数都是错的，
    // 而且错得很像"真有缺陷"（覆盖低、缺口大）。
    // 第一版就是这么错的：中心估偏 19px，全批报"缺口 60°"。
    {
        const int N = 360;
        const double rot = tplRotDeg * CV_PI / 180.0;
        const double ca = std::cos(rot), sa = std::sin(rot);
        std::vector<char> onv(N, 0);
        for (int t = 0; t < N; ++t) {
            const double th = 2.0 * CV_PI * t / N;
            const double u = tplA * std::cos(th), v = tplB * std::sin(th);
            const double px = tplCx + u * ca - v * sa;
            const double py = tplCy + u * sa + v * ca;
            double nx = std::cos(th) / tplA, ny = std::sin(th) / tplB;
            const double nl = std::hypot(nx, ny);
            nx /= nl;
            ny /= nl;
            const double wnx = nx * ca - ny * sa, wny = nx * sa + ny * ca;
            double peak = 0;
            for (int k = -3; k <= 3; ++k) {
                const int xi = static_cast<int>(std::lround(px + wnx * k));
                const int yi = static_cast<int>(std::lround(py + wny * k));
                if (xi < 0 || yi < 0 || xi >= tmpl.cols || yi >= tmpl.rows) continue;
                peak = std::max(peak, static_cast<double>(tmpl.at<uchar>(yi, xi)));
            }
            onv[t] = peak >= 120;
        }
        int c = 0;
        for (char b : onv) c += b;
        const double cov = static_cast<double>(c) / N;
        int mgl = 0, cu = 0;
        for (int i = 0; i < 2 * N; ++i) {
            if (!onv[i % N]) {
                ++cu;
                mgl = std::max(mgl, std::min(cu, N));
            } else
                cu = 0;
        }
        std::printf("[自检] 在模板自身上量：覆盖 %.3f  最长缺口 %.2f°%s\n", cov, 360.0 * mgl / N,
                    cov >= 0.97 ? "  OK" : "  <<< 椭圆参数和模板对不上，下面结果不可信");
        if (cov < 0.97) return 1;
    }
    std::printf("\n");

    std::printf("%-24s %7s %7s %8s %7s %7s %8s %8s\n", "file", "mx", "my", "peak", "cover",
                "maxGap", "meanInk", "minRatio");

    std::vector<double> vCover, vGap, vMeanInk, vMinRatio, vPeak;

    FILE* csv = nullptr;
    if (!csvPath.isEmpty()) {
        csv = std::fopen(csvPath.toStdString().c_str(), "wb");
        if (csv) std::fprintf(csv, "file,mx,my,peak,cover,maxGapDeg,meanInk,minInkRatio\n");
    }

    for (const QString& path : files) {
        cv::Mat g;
        if (!readGrayU8(path, g)) continue;
        // 模板必须完整落在图内
        if (tmpl.cols > g.cols || tmpl.rows > g.rows) continue;
        // 模板匹配：在原图上找模板
        cv::Mat res;
        cv::matchTemplate(g, tmpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(res, nullptr, &mval, nullptr, &mloc);

        // 环在整图上的椭圆参数
        double cx = mloc.x + tplCx;
        double cy = mloc.y + tplCy;

        // 小范围精定位：在 ±6px 内微调中心，让椭圆上的墨最多。
        // 模板匹配给的是"整块 logo"的最优位置，环本身可能还差一两像素。
        auto ringScore = [&](double ccx, double ccy, int* onCount) {
            const int N = 360;
            const double rot = tplRotDeg * CV_PI / 180.0;
            const double ca = std::cos(rot), sa = std::sin(rot);
            int cnt = 0;
            for (int t = 0; t < N; ++t) {
                const double th = 2.0 * CV_PI * t / N;
                const double u = tplA * std::cos(th), v = tplB * std::sin(th);
                const double px = ccx + u * ca - v * sa;
                const double py = ccy + u * sa + v * ca;
                // 局部阈值：用图自身的墨水平（不跨图比较绝对值）
                double peak = 0;
                for (int k = -3; k <= 3; ++k) {
                    double nx = std::cos(th) / tplA, ny = std::sin(th) / tplB;
                    const double nl = std::hypot(nx, ny);
                    nx /= nl;
                    ny /= nl;
                    const double wnx = nx * ca - ny * sa, wny = nx * sa + ny * ca;
                    const int xi = static_cast<int>(std::lround(px + wnx * k));
                    const int yi = static_cast<int>(std::lround(py + wny * k));
                    if (xi < 0 || yi < 0 || xi >= g.cols || yi >= g.rows) continue;
                    peak = std::max(peak, static_cast<double>(g.at<uchar>(yi, xi)));
                }
                if (peak >= 120) ++cnt;   // 固定灰度门槛：环上的墨远亮于此
            }
            if (onCount) *onCount = cnt;
            return static_cast<double>(cnt) / N;
        };
        double bestCov = -1;
        for (int dx = -6; dx <= 6; dx += 2) {
            for (int dy = -6; dy <= 6; dy += 2) {
                const double c = ringScore(cx + dx, cy + dy, nullptr);
                if (c > bestCov) {
                    bestCov = c;
                    // 记住最优偏移
                }
            }
        }
        // 用最优位置重新算一遍
        int bestDx = 0, bestDy = 0;
        {
            double bc = -1;
            for (int dx = -6; dx <= 6; dx += 2)
                for (int dy = -6; dy <= 6; dy += 2) {
                    const double c = ringScore(cx + dx, cy + dy, nullptr);
                    if (c > bc) {
                        bc = c;
                        bestDx = dx;
                        bestDy = dy;
                    }
                }
            bestCov = bc;
            cx += bestDx;
            cy += bestDy;
        }

        // 在最优位置上量：覆盖、缺口、环上墨的平均强度、最暗处与平均的比
        const int N = 360;
        const double rot = tplRotDeg * CV_PI / 180.0;
        const double ca = std::cos(rot), sa = std::sin(rot);
        std::vector<char> onv(N, 0);
        std::vector<double> inkVals(N, 0.0);
        int onCount = 0;
        for (int t = 0; t < N; ++t) {
            const double th = 2.0 * CV_PI * t / N;
            const double u = tplA * std::cos(th), v = tplB * std::sin(th);
            const double px = cx + u * ca - v * sa;
            const double py = cy + u * sa + v * ca;
            double nx = std::cos(th) / tplA, ny = std::sin(th) / tplB;
            const double nl = std::hypot(nx, ny);
            nx /= nl;
            ny /= nl;
            const double wnx = nx * ca - ny * sa, wny = nx * sa + ny * ca;
            double peak = 0;
            for (int k = -3; k <= 3; ++k) {
                const int xi = static_cast<int>(std::lround(px + wnx * k));
                const int yi = static_cast<int>(std::lround(py + wny * k));
                if (xi < 0 || yi < 0 || xi >= g.cols || yi >= g.rows) continue;
                peak = std::max(peak, static_cast<double>(g.at<uchar>(yi, xi)));
            }
            inkVals[t] = peak;
            onv[t] = peak >= 120;
            onCount += onv[t];
        }
        const double cover = static_cast<double>(onCount) / N;

        // 环形最长缺口
        int mgl = 0, cur = 0;
        for (int i = 0; i < 2 * N; ++i) {
            if (!onv[i % N]) {
                ++cur;
                mgl = std::max(mgl, std::min(cur, N));
            } else
                cur = 0;
        }
        const double maxGap = 360.0 * mgl / N;

        // 环上墨量的统计：只看判为有墨的点，避免缺口拉低均值
        double sum = 0;
        int cnt = 0;
        for (int t = 0; t < N; ++t)
            if (onv[t]) {
                sum += inkVals[t];
                ++cnt;
            }
        const double meanInk = cnt > 0 ? sum / cnt : 0.0;
        // 最暗的 10% 与均值的比：反映"某段明显偏淡"
        std::vector<double> sortedInk;
        for (int t = 0; t < N; ++t)
            if (onv[t]) sortedInk.push_back(inkVals[t]);
        std::sort(sortedInk.begin(), sortedInk.end());
        const double p10 = sortedInk.empty() ? 0.0 : sortedInk[sortedInk.size() / 10];
        const double ratio = meanInk > 1e-6 ? p10 / meanInk : 0.0;

        const std::string stem = QFileInfo(path).completeBaseName().toStdString();
        std::printf("%-24s %7d %7d %8.3f %7.3f %7.2f %8.1f %8.3f\n", stem.c_str(), mloc.x,
                    mloc.y, mval, cover, maxGap, meanInk, ratio);

        vPeak.push_back(mval);
        vCover.push_back(cover);
        vGap.push_back(maxGap);
        vMeanInk.push_back(meanInk);
        vMinRatio.push_back(ratio);
        if (csv)
            std::fprintf(csv, "%s,%d,%d,%.4f,%.4f,%.2f,%.2f,%.4f\n", stem.c_str(), mloc.x, mloc.y,
                         mval, cover, maxGap, meanInk, ratio);
    }
    if (csv) std::fclose(csv);

    auto stats = [](std::vector<double>& v, const char* name) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        std::printf("%-12s 均值%9.3f  中位%9.3f  最小%9.3f  最大%9.3f\n", name, mean,
                    v[v.size() / 2], v.front(), v.back());
    };
    std::printf("\n=== 汇总（%d 张）===\n", static_cast<int>(vCover.size()));
    stats(vPeak, "模板峰值");
    stats(vCover, "环覆盖");
    stats(vGap, "最长缺口");
    stats(vMeanInk, "环平均墨量");
    stats(vMinRatio, "最暗/均值");
    return 0;
}
