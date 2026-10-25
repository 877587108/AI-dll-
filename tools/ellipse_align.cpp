// ellipse_align.cpp —— 椭圆是否要逐图精对齐
//
// 【要回答的问题】
// 84 张 OK 图上的环半径起伏在 12~48px，而墨线只有 5~8px 宽。
// 起伏这么大，断点检测没法用：窗口窄了把波谷当缺口，宽了把真缺口填平。
//
// 两种可能，修法完全不同：
//   A. 椭圆参数（中心/半轴/角度）本身不准  -> 改参数就行
//   B. 每张图零件位置不同，固定参数天然对不上 -> 必须逐图精对齐
//
// 判别办法：对每张图，在中心附近做小范围搜索，
// 找使"半径起伏最小"的那个中心偏移。
//   如果最优偏移几乎都是 0 -> A
//   如果最优偏移因图而异、而且能明显降低起伏 -> B
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

#include "batchcore.hpp"
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

// 给定椭圆，量环半径的起伏
static double radiusSwing(const cv::Mat& sig, double cx, double cy, double a, double b, double rotDeg,
                          double thr, double* outCov = nullptr) {
    const int N = 360;
    const double rot = rotDeg * CV_PI / 180.0;
    const double ca = std::cos(rot), sa = std::sin(rot);
    std::vector<double> rOn;
    int onCount = 0;
    for (int t = 0; t < N; ++t) {
        const double th = 2.0 * CV_PI * t / N;
        const double u = a * std::cos(th), v = b * std::sin(th);
        const double px = cx + u * ca - v * sa;
        const double py = cy + u * sa + v * ca;
        // 法向
        double nx = std::cos(th) / a, ny = std::sin(th) / b;
        const double nl = std::hypot(nx, ny);
        nx /= nl;
        ny /= nl;
        const double wnx = nx * ca - ny * sa, wny = nx * sa + ny * ca;
        // 沿法向找最亮的偏移
        double bestV = -1e9, bestK = 0;
        for (int k = -20; k <= 20; ++k) {
            const int xi = static_cast<int>(std::lround(px + wnx * k));
            const int yi = static_cast<int>(std::lround(py + wny * k));
            if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
            const double v2 = sig.at<float>(yi, xi);
            if (v2 > bestV) {
                bestV = v2;
                bestK = k;
            }
        }
        if (bestV >= thr) {
            rOn.push_back(bestK);
            ++onCount;
        }
    }
    if (outCov) *outCov = static_cast<double>(onCount) / N;
    if (rOn.size() < 8) return 1e9;
    // 用中位数中心做稳健起伏
    std::vector<double> s = rOn;
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    const double med = s[s.size() / 2];
    double mn = 1e9, mx = -1e9;
    for (double v : rOn) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    (void)med;
    return mx - mn;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: markloc_ellipsealign <modelDir> <imageDir>\n");
        return 2;
    }
    chardet::Model model;
    const QString vdir = QDir(QString::fromLocal8Bit(argv[1]))
                             .absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    // 找图案区
    const chardet::RegionInfo* pat = nullptr;
    for (const auto& r : model.regions)
        if (r.isPattern()) {
            pat = &r;
            break;
        }
    if (!pat) {
        std::printf("模型里没有图案区\n");
        return 1;
    }
    std::printf("图案区：中心(%.1f,%.1f) 半轴 %.2fx%.2f 角 %.2f° e=%.3f\n", pat->ell.center.x,
                pat->ell.center.y, pat->ell.a, pat->ell.b, pat->ell.angleDeg,
                pat->ell.eccentricity());

    // 【角度敏感性验证】
    // 既然椭圆框允许设置角度，就要证明这个角度真的起作用。
    // 做法：把角度换几个值，看环半径起伏怎么变。
    // 如果改角度起伏几乎不动，说明角度没被用上、或者用错了。
    if (argc >= 4 && std::strcmp(argv[3], "--angle-test") == 0) {
        QString first;
        for (const QFileInfo& fi : QDir(QString::fromLocal8Bit(argv[2]))
                                       .entryInfoList({QStringLiteral("*.tif"), QStringLiteral("*.png"),
                                                       QStringLiteral("*.bmp")},
                                                      QDir::Files, QDir::Name)) {
            if (fi.completeBaseName().endsWith(QStringLiteral("_00"))) {
                first = fi.absoluteFilePath();
                break;
            }
        }
        if (first.isEmpty()) {
            std::printf("没有找到 _00 图\n");
            return 1;
        }
        cv::Mat gg;
        if (!readGrayU8(first, gg) || gg.size() != model.freq.size()) {
            std::printf("读图失败\n");
            return 1;
        }
        const cv::Mat sg =
            chardet::strokeSignal(gg, cv::Rect(0, 0, gg.cols, gg.rows), model.config, false);
        std::printf("\n角度(°)   半径起伏(px)   有墨比例\n");
        for (double ang : {0.0, -13.27, -30.0, 13.27, 45.0, 90.0}) {
            double cov = 0;
            const double sw = radiusSwing(sg, pat->ell.center.x, pat->ell.center.y, pat->ell.a,
                                          pat->ell.b, ang, model.config.strokeHi, &cov);
            std::printf("%8.2f %12.1f %10.4f\n", ang, sw, cov);
        }
        return 0;
    }

    std::vector<chardet::CharUnit> base;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        if (model.regions[ri].isPattern()) continue;
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) base.push_back(std::move(x));
    }

    QStringList files;
    for (const QFileInfo& fi : QDir(QString::fromLocal8Bit(argv[2]))
                                   .entryInfoList({QStringLiteral("*.tif"), QStringLiteral("*.png"),
                                                   QStringLiteral("*.bmp")},
                                                  QDir::Files, QDir::Name)) {
        if (!fi.completeBaseName().endsWith(QStringLiteral("_00"))) continue;
        files << fi.absoluteFilePath();
    }
    std::printf("待测 %d 张\n\n", static_cast<int>(files.size()));

    std::printf("%-30s %8s %8s %8s %8s\n", "file", "swing0", "bestDx", "bestDy", "swingBest");
    std::vector<double> swings0, swingsBest;
    std::vector<int> dxs, dys;
    for (const QString& path : files) {
        cv::Mat g;
        if (!readGrayU8(path, g) || g.size() != model.freq.size()) continue;
        const cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

        // 基准：整图配准后的位移
        int adx = 0, ady = 0;
        {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : base) {
                const cv::Rect bb = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (bb.width <= 0 || bb.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | bb) : bb;
            }
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
                    adx = static_cast<int>(std::lround(sh.x));
                    ady = static_cast<int>(std::lround(sh.y));
                }
            }
        }
        const double cx0 = pat->ell.center.x + adx;
        const double cy0 = pat->ell.center.y + ady;
        const double s0 = radiusSwing(sig, cx0, cy0, pat->ell.a, pat->ell.b, pat->ell.angleDeg,
                                      model.config.strokeHi);

        // 在 ±14px 内搜中心，找起伏最小的
        double best = 1e9;
        int bdx = 0, bdy = 0;
        for (int ddx = -14; ddx <= 14; ddx += 2) {
            for (int ddy = -14; ddy <= 14; ddy += 2) {
                const double s = radiusSwing(sig, cx0 + ddx, cy0 + ddy, pat->ell.a, pat->ell.b,
                                             pat->ell.angleDeg, model.config.strokeHi);
                if (s < best) {
                    best = s;
                    bdx = ddx;
                    bdy = ddy;
                }
            }
        }
        const std::string stem = QFileInfo(path).completeBaseName().toStdString();
        std::printf("%-30s %8.0f %8d %8d %8.0f\n", stem.c_str(), s0, bdx, bdy, best);
        swings0.push_back(s0);
        swingsBest.push_back(best);
        dxs.push_back(bdx);
        dys.push_back(bdy);
    }

    auto stats = [](std::vector<double>& v, const char* name) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        std::printf("%-12s 均值 %.1f  中位 %.1f  最小 %.0f  最大 %.0f\n", name, mean,
                    v[v.size() / 2], v.front(), v.back());
    };
    std::printf("\n=== 汇总 ===\n");
    stats(swings0, "原起伏");
    stats(swingsBest, "精对齐后");
    if (!dxs.empty()) {
        std::sort(dxs.begin(), dxs.end());
        std::sort(dys.begin(), dys.end());
        std::printf("最优 dx 中位 %d（范围 %d..%d）  dy 中位 %d（范围 %d..%d）\n",
                    dxs[dxs.size() / 2], dxs.front(), dxs.back(), dys[dys.size() / 2], dys.front(),
                    dys.back());
    }
    return 0;
}
