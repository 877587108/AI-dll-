// topo_probe.cpp —— 环与内部字母的拓扑关系
//
// 【第 4 轮的动机】
// 前几轮证明：环本身的墨量（角度缺口、粗细、峰值、相对比值）
// 在 ngv2 与 unset 之间重叠，分不开。
// 那么缺陷很可能不在"环的墨够不够"，而在"环与内部结构的关系"。
//
// 具体看一件事：沿每个角度从圆心向外走，会穿过几条"墨带"？
//   · 正常：先穿过 S/C 字母（1~2 条），再穿过环（1 条）
//   · 若环与字母之间有笔画相连，中间那条会接上，条数变少
//   · 若环上某处断开，外圈那条会缺失
//
// 这个工具把"每个角度的墨带条数"和"最外圈墨带的半径"打出来，
// 供判断该用什么量区分两类。
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
    if (argc < 4) {
        std::printf("usage: markloc_topoprobe <modelRoot> <tmpl.png> <image...>\n");
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

    std::printf("%-16s %8s %8s %10s %10s\n", "file", "avgRuns", "p90Runs", "outerRStd", "outerRRange");
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
        const int innerLo = static_cast<int>(std::lround(A * 0.15));
        const int Wcols = pol.cols;

        std::vector<int> runs(Wcols, 0);
        std::vector<double> outerR(Wcols, 0.0);
        for (int x = 0; x < Wcols; ++x) {
            // 从圆心到带外沿，数"墨带条数"（连续过阈值算一条）
            int cnt = 0;
            bool in = false;
            double lastOn = 0;
            for (int r = innerLo; r <= bHi; ++r) {
                const bool on = pol.at<float>(r, x) >= 0.35f;
                if (on) {
                    if (!in) {
                        ++cnt;
                        in = true;
                    }
                    lastOn = r;
                } else {
                    in = false;
                }
            }
            runs[x] = cnt;
            outerR[x] = lastOn;
        }

        std::vector<int> rs = runs;
        std::sort(rs.begin(), rs.end());
        const double avgRuns = [&] {
            double s = 0;
            for (int v : runs) s += v;
            return s / std::max<size_t>(1, runs.size());
        }();
        const int p90 = rs[static_cast<size_t>(rs.size() * 0.9)];
        // 最外圈半径的稳定性：环若有局部凹陷/断开，这个值会跳
        std::vector<double> orv = outerR;
        std::sort(orv.begin(), orv.end());
        double ms = 0;
        for (double v : outerR) ms += v;
        ms /= std::max<size_t>(1, outerR.size());
        double var = 0;
        for (double v : outerR) var += (v - ms) * (v - ms);
        const double sd = std::sqrt(var / std::max<size_t>(1, outerR.size()));
        const double rng = orv.back() - orv.front();

        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::printf("%-16s %8.2f %8d %10.2f %10.1f\n", fn.toStdString().substr(0, 16).c_str(),
                    avgRuns, p90, sd, rng);
    }
    return 0;
}
