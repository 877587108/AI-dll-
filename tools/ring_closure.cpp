// ring_closure.cpp —— 圆环封闭性检测（不依赖极坐标投影）
//
// 【为什么投影找不到断口】
// 现有的极坐标做法是：把环展开成横带，再按角度看"这条半径上有没有墨"。
// 它有一个结构性缺陷：
//   · 每个角度只回答"这一列有没有墨"，不回答"整条曲线连不连得起来"
//   · 环上某处断开后，只要旁边的墨横向延伸过来，那一列照样"有墨"
//   · 环的封闭性（拓扑性质）在展开后丢失了
//
// 所以断口可能在投影上表现为"没有波谷"，或波谷很浅，被判成正常。
//
// 【正确的做法：在图像上直接做封闭性检测】
//   ① 在拉直后的图上，取环所在的那个环形区域
//   ② 二值化得到"墨"的连通域
//   ③ 把环的那一圈当作一条闭合曲线，检查它是否首尾相连：
//      沿环的每一个角度，从中心向外找到"最外层的墨"，
//      然后检查相邻角度之间这个最外层墨是否连续（距离跳变 <= 阈值）
//      只要有一处跳变过大 = 环在这里断开/缺失
//
// 这比"列内有墨"严格得多：它要求外圈墨在**角度方向上连续成链**，
// 也就是真的闭合。
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

struct ClosureResult {
    bool valid = false;
    double outerRadiusMed = 0;   // 环外缘半径的中位数
    double worstGapDeg = 0;      // 最长的"外缘缺失"角度跨度
    int    gapCount = 0;         // 断点个数
    double chainCoverage = 0;    // 外缘连续的覆盖率
    double maxJump = 0;          // 外缘半径的最大跳变(px)
};

// 核心：沿角度跟踪"最外层墨的半径"，并要求它连续成链
static ClosureResult checkClosure(const chardet::Model& model, const cv::Mat& tmpl,
                                  const cv::Mat& g, double inkThr, int N,
                                  double minOuterRatio, double maxOuterRatio,
                                  int maxJumpPx) {
    ClosureResult out;
    const double tplCx = 127.8, tplCy = 116.0, tplA = 132.1, tplB = 76.75, tplRot = -13.78;

    cv::Mat mres;
    cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
    cv::Point mloc;
    double mval = 0;
    cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
    const double cx = mloc.x + tplCx, cy = mloc.y + tplCy;
    const double A = tplA, B = tplB;

    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);

    // 拉直（与判据同一套矩阵），之后环变成半径 A 的正圆
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
    if (roi.area() <= 0) return out;
    const cv::Point2f cc(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
    const cv::Mat sub = straight(roi);
    const double maxR = A + pad;

    const int nRadius = static_cast<int>(std::lround(maxR));
    cv::Mat pol;
    cv::warpPolar(sub, pol, cv::Size(nRadius, N), cc, maxR,
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    cv::Mat polT;
    cv::transpose(pol, polT);
    pol = polT;   // rows=半径 cols=角度

    const int rLo = static_cast<int>(std::lround(A * minOuterRatio));
    const int rHi = std::min(static_cast<int>(std::lround(A * maxOuterRatio)), pol.rows - 1);
    if (rHi <= rLo + 2) return out;

    // 对每个角度，找出"最外层过阈值的墨"所在的半径
    // 关键是**最外层**：环是闭合曲线，它一定是该角度上最外面的那条墨带
    std::vector<double> outer(N, -1.0);
    for (int x = 0; x < N; ++x) {
        for (int r = rHi; r >= rLo; --r) {
            if (pol.at<float>(r, x) >= inkThr) {
                outer[x] = r;
                break;
            }
        }
    }
    // 没有任何墨的角度保持 -1

    std::vector<double> valid;
    for (double v : outer)
        if (v >= 0) valid.push_back(v);
    if (valid.size() < static_cast<size_t>(N) * 0.5) return out;
    std::sort(valid.begin(), valid.end());
    out.outerRadiusMed = valid[valid.size() / 2];
    out.valid = true;

    // 判定断点：某个角度没有墨（-1），或半径相对中位偏离过大
    std::vector<char> broken(N, 0);
    double mx = 0;
    for (int x = 0; x < N; ++x) {
        if (outer[x] < 0) {
            broken[x] = 1;
            continue;
        }
        // 与前后相邻角度的跳变
        const int nx = (x + 1) % N;
        if (outer[nx] >= 0) {
            const double j = std::abs(outer[nx] - outer[x]);
            mx = std::max(mx, j);
            if (j > maxJumpPx) broken[x] = 1;
        }
    }
    out.maxJump = mx;

    // 最长连续断点跨度
    int startAt = -1;
    for (int t = 0; t < N; ++t)
        if (!broken[t]) {
            startAt = t;
            break;
        }
    if (startAt < 0) {
        out.chainCoverage = 0;
        out.worstGapDeg = 360;
        return out;
    }
    int best = 0, cur = 0, bestStart = 0, curStart = 0;
    for (int k = 1; k <= N; ++k) {
        const int t = (startAt + k) % N;
        if (broken[t]) {
            if (cur == 0) curStart = k;
            ++cur;
            if (cur > best) {
                best = cur;
                bestStart = curStart;
            }
        } else {
            cur = 0;
        }
    }
    out.worstGapDeg = 360.0 * best / N;
    out.gapCount = best > 0 ? 1 : 0;
    out.chainCoverage = 1.0 - static_cast<double>(best) / N;
    (void)bestStart;
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: markloc_closure <modelRoot> <tmpl.png> <image...>\n");
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

    const double inkThr = 0.35;
    const int N = 720;
    const double minOuter = 0.80, maxOuter = 1.40;
    const int maxJumpPx = 6;

    std::printf("%-18s %8s %9s %8s %9s\n", "file", "outerR", "worstGap", "chainCov", "maxJump");
    for (int a = 3; a < argc; ++a) {
        cv::Mat g;
        if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g)) continue;
        const auto r = checkClosure(model, tmpl, g, inkThr, N, minOuter, maxOuter, maxJumpPx);
        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        if (!r.valid) {
            std::printf("%-18s  无效\n", fn.toStdString().substr(0, 18).c_str());
            continue;
        }
        std::printf("%-18s %8.1f %8.2f度 %8.4f %9.1f\n", fn.toStdString().substr(0, 18).c_str(),
                    r.outerRadiusMed, r.worstGapDeg, r.chainCoverage, r.maxJump);
    }
    return 0;
}
