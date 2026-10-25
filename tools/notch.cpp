// notch.cpp —— 定点检测圆环左上角缺口
//
// 【背景】
// 目视确认：ngv2 全部 7 张、unset 高分那批，缺失都在**圆环左上角的同一个位置**，
// 形态是环线断开一小段（约 2~5 度）。而 image/ok 的干净料此处环线完整闭合。
//
// 【为什么之前全部失败】
// 之前的做法都是在**整圈 360 度**上找最大断口/最深凹陷：
//   · 缺口只有 2~5 度，占整圈的 0.6%~1.4%
//   · 环带纹理的逐角度波动幅度与缺口深度相当
//   · 于是"全局最大值"被纹理噪声主导，缺口淹没在里面
//
// 【现在的方法】
//   · 只在缺口所在的**一个小角度窗口**内做检测
//   · 窗口内的环线必须连续；一旦窗口内出现"环线上没有任何墨"的角度段 = 缺口
//   · 关键是用**沿环方向的连续性**判定，而不是逐列取最大值
//
// 拉直后环是半径 A 的正圆。对窗口内每个角度，沿半径方向找环线墨带的中心，
// 然后检查这串中心点是否连续（相邻角度之间半径不跳变、且每列都有墨）。
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

static const double kA = 132.1, kB = 76.75, kRot = -13.78;
static const double kTplCx = 127.8, kTplCy = 116.0;

int main(int argc, char** argv) {
    if (argc < 7) {
        std::printf("usage: markloc_notch <modelRoot> <tmpl.png> <listOrDir> <fromDeg> <toDeg> "
                    "<mode>\n");
        std::printf("  mode: list=从文件读路径(每行一个)\n");
        return 2;
    }
    const double fromDeg = std::atof(argv[4]);
    const double toDeg = std::atof(argv[5]);

    chardet::Model model;
    const QString vdir =
        QDir(QString::fromUtf8(argv[1])).absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    cv::Mat tmpl;
    if (!readGrayU8(QString::fromUtf8(argv[2]), tmpl)) {
        std::printf("模板读取失败\n");
        return 1;
    }

    std::vector<std::string> files;
    {
        QFile f(QString::fromUtf8(argv[3]));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            std::printf("列表读取失败\n");
            return 1;
        }
        while (!f.atEnd()) {
            const QString line = QString::fromUtf8(f.readLine()).trimmed();
            if (!line.isEmpty()) files.push_back(line.toStdString());
        }
    }

    const int N = 3600;   // 0.1 度一个采样
    const double A = kA;
    const int rLo = static_cast<int>(std::lround(A * 0.82));
    const int rHi = static_cast<int>(std::lround(A * 1.18));

    std::printf("窗口 %.1f~%.1f 度\n", fromDeg, toDeg);
    std::printf("%-46s %8s %9s %9s %8s\n", "file", "ringMed", "缺口宽", "窗口最低", "判定");

    for (const auto& path : files) {
        cv::Mat g;
        if (!readGrayU8(QString::fromUtf8(path.c_str()), g)) {
            std::printf("[读失败] %s\n", path.c_str());
            continue;
        }
        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        if (mval < 0.6) {
            std::printf("%-46s 定位失败(%.2f)\n",
                        QFileInfo(QString::fromUtf8(path.c_str())).fileName().toStdString().substr(0, 44).c_str(),
                        mval);
            continue;
        }
        const double cx = mloc.x + kTplCx, cy = mloc.y + kTplCy;

        const cv::Mat sig =
            chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
        const double k = kA / std::max(1.0, kB);
        const double th = kRot * CV_PI / 180.0;
        const double cR = std::cos(-th), sR = std::sin(-th);
        cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -cx, 0, 1, -cy, 0, 0, 1);
        cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
        cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, k, 0, 0, 0, 1);
        cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
        cv::Mat M3 = T2 * Sc * Rm * T1;
        cv::Mat M;
        M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
        cv::Mat straight;
        cv::warpAffine(sig, straight, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);

        const int pad = static_cast<int>(A * 0.35) + 8;
        cv::Rect roi(static_cast<int>(cx - A - pad), static_cast<int>(cy - A - pad),
                     static_cast<int>(2 * (A + pad)), static_cast<int>(2 * (A + pad)));
        roi &= cv::Rect(0, 0, straight.cols, straight.rows);
        if (roi.area() <= 0) continue;
        const cv::Point2f cc(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
        const double maxR = A + pad;
        const int nR = static_cast<int>(std::lround(maxR));
        cv::Mat pol;
        cv::warpPolar(straight(roi), pol, cv::Size(nR, N), cc, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        cv::Mat polT;
        cv::transpose(pol, polT);
        pol = polT;   // rows=半径 cols=角度(0.1度)

        // 全环中位墨量（用于归一化）
        std::vector<double> allPeak;
        for (int x = 0; x < N; x += 5) {
            double pk = 0;
            for (int r = rLo; r <= rHi && r < pol.rows; ++r)
                pk = std::max(pk, (double)pol.at<float>(r, x));
            allPeak.push_back(pk);
        }
        std::sort(allPeak.begin(), allPeak.end());
        const double ringMed = allPeak[allPeak.size() / 2];

        // 窗口内逐角度取环带峰值
        const int t0 = static_cast<int>(std::lround(fromDeg * 10));
        const int t1 = static_cast<int>(std::lround(toDeg * 10));
        int nW = t1 - t0;
        if (nW < 1) nW = 1;
        std::vector<double> prof(nW, 0.0);
        for (int i = 0; i < nW; ++i) {
            const int x = ((t0 + i) % N + N) % N;
            double pk = 0;
            for (int r = rLo; r <= rHi && r < pol.rows; ++r)
                pk = std::max(pk, (double)pol.at<float>(r, x));
            prof[i] = pk;
        }
        // 归一化：窗口内墨量 / 全环中位
        std::vector<double> rel(nW);
        for (int i = 0; i < nW; ++i) rel[i] = ringMed > 1e-6 ? prof[i] / ringMed : 0.0;

        // 缺口判定：相对墨量低于某比例即视为"无墨"
        // 用 0.45（环中位的 45%）作为"没有环线"的门限
        const double missThr = 0.45;
        int bestRun = 0, cur = 0;
        double minRel = 1e9;
        for (int i = 0; i < nW; ++i) {
            minRel = std::min(minRel, rel[i]);
            if (rel[i] < missThr) {
                ++cur;
                bestRun = std::max(bestRun, cur);
            } else {
                cur = 0;
            }
        }
        const double gapDeg = bestRun * 0.1;
        const QString fn = QFileInfo(QString::fromUtf8(path.c_str())).fileName();
        std::printf("%-46s %8.3f %8.1f度 %9.3f %8s\n", fn.toStdString().substr(0, 44).c_str(),
                    ringMed, gapDeg, minRel, gapDeg >= 1.0 ? "NG" : "OK");
    }
    return 0;
}
