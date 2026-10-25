// breakprobe.cpp —— 断点检测的诚实探针
//
// 【我犯过的错，写在最前面】
// 版本一：拿区域框当椭圆采样，cov 只有 2%~23%。
//         错在框是正的、椭圆是斜的，框根本不是椭圆的外接框。
// 版本二：改从墨拟合椭圆，cov 报出 1.000、最长缺口 0°。
//         数字漂亮到我该警觉 —— 果然验证代码本身有 bug：
//         法向偏移写成了 a+k / b+k（沿两个半轴分别加），
//         正确做法是沿法向单位向量偏移。而且 -- 更根本的问题 --
//         我一直在验证"拟合对没对"，却没验证"断口能不能被测出来"。
//
// 这个工具做三件事，按顺序，每件都必须先证明"能动"：
//   1. 拟合椭圆，报告拟合质量（残差、参数量级）
//   2. 沿椭圆采样，报告 cov / 最长缺口 / 每一段缺口
//   3. 【关键】人为在环上挖一个洞，看断口数是否真的从 0 变成 1
//      如果挖了洞还报 0，那这个检测器就是死的，前面所有数字都没有意义。
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

// 椭圆描述：中心 + 半轴 + 旋转（弧度）
struct Ell {
    double cx = 0, cy = 0, a = 1, b = 1, rot = 0;
    void pointAt(double th, double* px, double* py) const {
        const double u = a * std::cos(th), v = b * std::sin(th);
        const double ca = std::cos(rot), sa = std::sin(rot);
        *px = cx + u * ca - v * sa;
        *py = cy + u * sa + v * ca;
    }
};

// 沿椭圆采样，返回每个角度的"有墨峰值"
static std::vector<double> sampleRing(const cv::Mat& sig, const Ell& e, int N, double normalHalf) {
    std::vector<double> out(N, 0.0);
    const double ca = std::cos(e.rot), sa = std::sin(e.rot);
    for (int t = 0; t < N; ++t) {
        const double th = 2.0 * CV_PI * t / N;
        double px, py;
        e.pointAt(th, &px, &py);
        // 法向：椭圆在该点的外法向
        double nx = std::cos(th) / e.a, ny = std::sin(th) / e.b;
        const double nl = std::hypot(nx, ny);
        nx /= nl;
        ny /= nl;
        // 旋转到全图坐标
        const double wnx = nx * ca - ny * sa;
        const double wny = nx * sa + ny * ca;
        double peak = 0.0;
        for (double k = -normalHalf; k <= normalHalf; k += 0.5) {
            const int xi = static_cast<int>(std::lround(px + wnx * k));
            const int yi = static_cast<int>(std::lround(py + wny * k));
            if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
            peak = std::max(peak, static_cast<double>(sig.at<float>(yi, xi)));
        }
        out[t] = peak;
    }
    return out;
}

// 从 0/1 序列找断口：连续 0 且两侧各有足够长的连续 1
struct GapInfo {
    int start, len, flank;
};
static std::vector<GapInfo> findGaps(const std::vector<char>& on, int minFlank, int mergeGap = 6) {
    const int N = static_cast<int>(on.size());
    int startAt = -1;
    for (int t = 0; t < N; ++t)
        if (on[t]) { startAt = t; break; }
    std::vector<GapInfo> out;
    if (startAt < 0) return out;
    int cur = 0, curStart = -1;
    for (int k = 1; k <= N; ++k) {
        const int t = (startAt + k) % N;
        if (!on[t]) {
            if (cur == 0) curStart = t;
            ++cur;
        } else {
            if (cur > 0) {
                out.push_back({curStart, cur, 0});
                cur = 0;
            }
        }
    }
    // 合并间隔很短的相邻缺口。
    // 不合并的话，缺口边缘残留的零星墨点会把一个大洞切成几段，
    // 结果是"洞越大越检不出来" —— 实测 8° 洞被切成 2 段、15° 洞切成 5 段。
    {
        std::vector<GapInfo> merged;
        for (const auto& g : out) {
            if (!merged.empty()) {
                auto& last = merged.back();
                const int endOfLast = (last.start + last.len) % N;
                const int between = (g.start - endOfLast + N) % N;
                if (between <= mergeGap) {
                    last.len += between + g.len;
                    continue;
                }
            }
            merged.push_back(g);
        }
        if (merged.size() >= 2) {
            const GapInfo first = merged.front();
            auto& last = merged.back();
            const int endOfLast = (last.start + last.len) % N;
            const int between = (first.start - endOfLast + N) % N;
            if (between <= mergeGap) {
                last.len += between + first.len;
                merged.erase(merged.begin());
            }
        }
        out.swap(merged);
    }
    // 合并之后再量两侧墨段长度
    for (auto& g : out) {
        int before = 0, after = 0;
        for (int j = 1; j <= N; ++j) {
            if (!on[(g.start - j + 2 * N) % N]) break;
            ++before;
        }
        for (int j = 0; j < N; ++j) {
            if (!on[(g.start + g.len + j) % N]) break;
            ++after;
        }
        g.flank = std::min(before, after);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: markloc_breakprobe <modelDir> <image> [x y w h]\n");
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

    cv::Rect roi(474, 315, 322, 225);
    if (argc >= 7) {
        roi = cv::Rect(std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                       std::atoi(argv[6]));
    }
    cv::Mat g;
    if (!readGrayU8(imgPath, g)) {
        std::printf("read failed\n");
        return 1;
    }
    if (g.size() != model.freq.size()) {
        std::printf("size mismatch\n");
        return 1;
    }
    roi &= cv::Rect(0, 0, g.cols, g.rows);
    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    cv::Mat sub = sig(roi).clone();
    cv::Mat bin;
    cv::threshold(sub, bin, model.config.strokeHi, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255);

    std::printf("=== 1. 拟合椭圆 ===\n");
    std::vector<cv::Point2f> pts;
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x)
            if (bin.at<uchar>(y, x)) pts.emplace_back(static_cast<float>(roi.x + x),
                                                      static_cast<float>(roi.y + y));
    std::printf("区域 %dx%d @(%d,%d) 墨点 %d\n", roi.width, roi.height, roi.x, roi.y,
                static_cast<int>(pts.size()));

    std::vector<cv::Point2f> cur = pts;
    cv::RotatedRect best;
    double bestResid = 1e9;
    int bestN = 0;
    for (int iter = 0; iter < 8; ++iter) {
        if (cur.size() < 30) break;
        const cv::RotatedRect rr = cv::fitEllipse(cur);
        const double ca = std::cos(rr.angle * CV_PI / 180.0);
        const double sa = std::sin(rr.angle * CV_PI / 180.0);
        const double a = std::max(1.0, rr.size.width * 0.5);
        const double b = std::max(1.0, rr.size.height * 0.5);
        std::vector<double> res(cur.size());
        for (size_t i = 0; i < cur.size(); ++i) {
            const double dx = cur[i].x - rr.center.x, dy = cur[i].y - rr.center.y;
            const double u = (dx * ca + dy * sa) / a;
            const double v = (-dx * sa + dy * ca) / b;
            res[i] = std::abs(std::hypot(u, v) - 1.0);
        }
        std::vector<double> sorted = res;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const double med = sorted[sorted.size() / 2];
        std::printf("  迭代%d 点%5d 轴 %.1fx%.1f 角%.1f 中位残差%.4f\n", iter,
                    static_cast<int>(cur.size()), rr.size.width, rr.size.height, rr.angle, med);
        // 迭代会不断收缩点集，残差必然下降。所以不能只看残差，
        // 还要看"这些点够不够铺满一周"：点太少时残差再小也没有代表性。
        if (med < bestResid && cur.size() >= 800) {
            bestResid = med;
            best = rr;
            bestN = static_cast<int>(cur.size());
        }
        std::vector<double> s2 = res;
        std::sort(s2.begin(), s2.end());
        const double thr = s2[static_cast<size_t>(s2.size() * 0.60)];
        std::vector<cv::Point2f> next;
        for (size_t i = 0; i < cur.size(); ++i)
            if (res[i] <= thr) next.push_back(cur[i]);
        if (next.size() == cur.size()) break;
        cur.swap(next);
    }
    // cv::fitEllipse 的 size.width/height 已是全轴长，半轴要除 2
    Ell e;
    e.cx = best.center.x;
    e.cy = best.center.y;
    e.a = best.size.width * 0.5;
    e.b = best.size.height * 0.5;
    e.rot = best.angle * CV_PI / 180.0;
    std::printf("采用：中心(%.1f,%.1f) 半轴 %.1f x %.1f 角%.2f° 支撑点%d 残差%.4f\n", e.cx, e.cy,
                e.a, e.b, best.angle, bestN, bestResid);

    // 参数方程的自洽检查：算出的外接框该和墨的外接框接近
    double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (int t = 0; t < 3600; ++t) {
        double px, py;
        e.pointAt(2.0 * CV_PI * t / 3600.0, &px, &py);
        x0 = std::min(x0, px);
        x1 = std::max(x1, px);
        y0 = std::min(y0, py);
        y1 = std::max(y1, py);
    }
    std::printf("参方外接框 (%.0f,%.0f,%.0f,%.0f)\n", x0, y0, x1 - x0, y1 - y0);

    std::printf("\n=== 2. 沿椭圆采样 ===\n");
    const int N = 720;
    const double inkThr = model.config.strokeHi;
    auto peaks = sampleRing(sig, e, N, 6.0);
    std::vector<char> on(N);
    int nOn = 0;
    for (int t = 0; t < N; ++t) {
        on[t] = peaks[t] >= inkThr ? 1 : 0;
        nOn += on[t];
    }
    std::printf("有墨比例 %.3f\n", static_cast<double>(nOn) / N);
    auto gaps = findGaps(on, 8);
    int maxGap = 0, maxBreak = 0;
    for (const auto& gp : gaps) {
        const double deg = 360.0 * gp.len / N;
        maxGap = std::max(maxGap, gp.len);
        if (gp.len >= 2 && gp.len <= 24 && gp.flank >= 8) maxBreak = std::max(maxBreak, gp.len);
        if (deg >= 1.0) {
            const int tMid = (gp.start + gp.len / 2) % N;
            double mx, my;
            e.pointAt(2.0 * CV_PI * tMid / N, &mx, &my);
            std::printf("  缺口 %.2f° 起点%.0f° 两侧墨段%d @(%0.f,%.0f)\n", deg,
                        360.0 * gp.start / N, gp.flank, mx, my);
        }
    }
    std::printf("最长缺口 %.2f°  判为断口 %.2f°\n", 360.0 * maxGap / N, 360.0 * maxBreak / N);

    std::printf("\n=== 3. 能动性验证：人为在环上挖洞 ===\n");
    // 这是整个工具存在的理由。如果挖了洞断口号还是 0，检测器就是死的。
    //
    // 【第一轮验证的结论：检测器确实是死的】
    // 挖 2/4/8/15° 的洞，断口数全是 0。原因不是椭圆拟合错，
    // 而是采样本身：sampleRing 在法向上搜 ±6px 取最大值，
    // 而洞只有沿周向的长度、法向上依然是满墨 —— 剖面一横跨，
    // 就从洞两侧的墨里捞到了峰值。周向的缺口在法向剖面上根本不显现。
    //
    // 所以这里改成沿周向挖一条完整贯通的带（法向 ±RINGBAND 全清），
    // 这样任何采样方式都躲不开。只有这样才能验证"周向缺口检测"这一段逻辑。
    const double RINGBAND = 12.0;
    for (double holeDeg : {0.5, 1.0, 2.0, 4.0, 8.0, 15.0}) {
        cv::Mat sig2 = sig.clone();
        const int half = static_cast<int>(holeDeg * N / 360.0 / 2);
        int cleared = 0;
        for (int t = -half; t <= half; ++t) {
            const int tt = ((t % N) + N) % N;
            for (double k = -RINGBAND; k <= RINGBAND; k += 0.5) {
                const double a2 = e.a + k, b2 = e.b + k;
                const double th = 2.0 * CV_PI * tt / N;
                const double u = a2 * std::cos(th), v = b2 * std::sin(th);
                const double ca = std::cos(e.rot), sa = std::sin(e.rot);
                const int xi = static_cast<int>(std::lround(e.cx + u * ca - v * sa));
                const int yi = static_cast<int>(std::lround(e.cy + u * sa + v * ca));
                if (xi >= 0 && yi >= 0 && xi < sig2.cols && yi < sig2.rows) {
                    sig2.at<float>(yi, xi) = 0.0f;
                    ++cleared;
                }
            }
        }
        auto p2 = sampleRing(sig2, e, N, 6.0);
        std::vector<char> on2(N);
        for (int t = 0; t < N; ++t) on2[t] = p2[t] >= inkThr ? 1 : 0;
        auto g2 = findGaps(on2, 8);
        int nb = 0, mg = 0;
        for (const auto& gp : g2) {
            mg = std::max(mg, gp.len);
            if (gp.flank >= 8) ++nb;
        }
        std::printf("  挖 %4.1f° 洞(清%dpx) -> 缺口段%d个 最长%6.2f° 断口%d\n", holeDeg, cleared,
                    static_cast<int>(g2.size()), 360.0 * mg / N, nb);
    }
    return 0;
}
