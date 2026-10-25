// tmpl_cmp.cpp —— 多算法横向对比（优先模板比对）
//
// 【为什么回到模板比对】
// 之前的"极坐标投影墨量"和"外缘半径跳变"都在度量**局部标量**：
//   · 每个角度只看一条半径带是否过阈值
//   · 或者只看最外层半径的跳变
// 这些量描述的是"某处有多暗/有多毛糙"，不是"这里和标准环长得不一样"。
//
// 模板比对换了一个问题：**这块区域与正常环的差异有多大**。
// 正常环是从大量正常料里学出来的，噪声与纹理在平均中被抹掉，
// 剩下的就是真正的结构性差异。
//
// 本程序把样本拉直到统一坐标系，与正常模板做多种比对，
// 一次性输出多个指标，便于横向比较哪个真正有区分力。
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

static const double kTplCx = 127.8, kTplCy = 116.0, kTplA = 132.1, kTplB = 76.75;
static const double kTplRot = -13.78;

// 把一张图拉直到统一的模型坐标系，裁出以环心为中心的方块
static bool straightenLogo(const chardet::Model& model, const cv::Mat& tmpl, const cv::Mat& g,
                           int half, cv::Mat& outPatch, cv::Point2d& center) {
    if (g.empty() || g.cols < tmpl.cols || g.rows < tmpl.rows) {
        std::printf("  [diag] 图像尺寸异常 %dx%d\n", g.cols, g.rows);
        return false;
    }
    cv::Mat mres;
    cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
    cv::Point mloc;
    double mval = 0;
    cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
    if (mval < 0.3) {
        std::printf("  [diag] 匹配分 %.4f 过低\n", mval);
        return false;
    }
    const double cx = mloc.x + kTplCx, cy = mloc.y + kTplCy;
    center = cv::Point2d(cx, cy);

    const cv::Mat sig =
        chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), model.config, false);
    const double kScale = kTplA / std::max(1.0, kTplB);
    const double rot = kTplRot * CV_PI / 180.0;
    const double cR = std::cos(-rot), sR = std::sin(-rot);
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -cx, 0, 1, -cy, 0, 0, 1);
    cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
    cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
    cv::Mat M3 = T2 * Sc * Rm * T1;
    cv::Mat M;
    M3(cv::Rect(0, 0, 3, 2)).copyTo(M);

    // 输出到环心固定的画布上
    const int S = half * 2;
    cv::Mat Mout = M.clone();
    Mout.at<double>(0, 2) += half - cx;
    Mout.at<double>(1, 2) += half - cy;
    cv::Mat full;
    cv::warpAffine(sig, full, Mout, cv::Size(S, S), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0.0);
    outPatch = full;
    return true;
}

struct Metrics {
    double tplCorr = 0;      // 1. 归一化互相关（整块）
    double ringCorr = 0;     // 2. 只看环带的互相关
    double segCorr = 0;      // 3. 240~272 区段的互相关
    double absDiff = 0;      // 4. 环带平均绝对差
    double deficit = 0;      // 5. 环带"比模板暗"的量（只算变暗的方向）
    double segDeficit = 0;   // 6. 区段的变暗量
    double worstCol = 0;     // 7. 环带最差的单列相关
};

static double corrOf(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat aa, bb;
    a.convertTo(aa, CV_64F);
    b.convertTo(bb, CV_64F);
    cv::Scalar ma, sa, mb, sb;
    cv::meanStdDev(aa, ma, sa);
    cv::meanStdDev(bb, mb, sb);
    if (sa[0] < 1e-6 || sb[0] < 1e-6) return 0.0;
    return (aa - ma[0]).dot(bb - mb[0]) / (aa.total() * sa[0] * sb[0]);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("usage: markloc_tmplcmp <modelRoot> <tmpl.png> <refList.txt> <image...>\n");
        std::printf("  refList.txt : 用于建立正常模板的文件列表（每行一个路径）\n");
        return 2;
    }
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

    const int half = 170;   // 画布 340x340，环半径 132 居中
    const double A = kTplA;

    // ---------- 建立正常模板 ----------
    std::vector<std::string> refs;
    {
        QFile f(QString::fromUtf8(argv[3]));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            std::printf("参考列表读取失败\n");
            return 1;
        }
        while (!f.atEnd()) {
            QString line = QString::fromUtf8(f.readLine()).trimmed();
            if (line.isEmpty()) continue;
            refs.push_back(line.toStdString());
        }
    }
    if (refs.empty()) {
        std::printf("参考列表为空\n");
        return 1;
    }
    cv::Mat acc = cv::Mat::zeros(half * 2, half * 2, CV_64F);
    int nOk = 0;
    for (const auto& r : refs) {
        cv::Mat g, p;
        cv::Point2d c;
        if (!readGrayU8(QString::fromUtf8(r.c_str()), g)) continue;
        if (!straightenLogo(model, tmpl, g, half, p, c)) continue;
        cv::Mat pd;
        p.convertTo(pd, CV_64F);
        acc += pd;
        ++nOk;
    }
    if (nOk < 5) {
        std::printf("有效参考样本不足 (%d)\n", nOk);
        return 1;
    }
    acc /= nOk;
    cv::Mat ref;
    acc.convertTo(ref, CV_32F);
    std::printf("正常模板由 %d 张正常料平均建立\n", nOk);

    // 环带掩码与区段掩码（画布坐标，环心在 half,half）
    cv::Mat ringMask = cv::Mat::zeros(half * 2, half * 2, CV_8U);
    cv::circle(ringMask, cv::Point(half, half), static_cast<int>(std::lround(A * 1.35)), 255, -1);
    cv::circle(ringMask, cv::Point(half, half), static_cast<int>(std::lround(A * 0.72)), 0, -1);
    cv::Mat segMask = cv::Mat::zeros(half * 2, half * 2, CV_8U);
    {
        const double r0 = A * 0.60, r1 = A * 1.45;
        std::vector<cv::Point> pts;
        for (int t = 238; t <= 272; t += 1) {
            const double rad = t * CV_PI / 180.0;
            pts.push_back(cv::Point(static_cast<int>(half + r1 * std::cos(rad)),
                                    static_cast<int>(half + r1 * std::sin(rad))));
        }
        for (int t = 272; t >= 238; t -= 1) {
            const double rad = t * CV_PI / 180.0;
            pts.push_back(cv::Point(static_cast<int>(half + r0 * std::cos(rad)),
                                    static_cast<int>(half + r0 * std::sin(rad))));
        }
        cv::fillPoly(segMask, std::vector<std::vector<cv::Point>>{pts}, 255);
    }

    std::printf("%-18s %8s %8s %8s %8s %8s %8s %8s\n", "file", "tplCorr", "ringCorr", "segCorr",
                "absDiff", "deficit", "segDefic", "worstCol");
    // 待测列表：优先从 UTF-8 文本文件读取（避免 Windows 命令行代码页破坏中文路径）
    std::vector<std::string> tests;
    if (argc == 5) {
        QFile tf(QString::fromUtf8(argv[4]));
        if (tf.open(QIODevice::ReadOnly | QIODevice::Text)) {
            while (!tf.atEnd()) {
                const QString line = QString::fromUtf8(tf.readLine()).trimmed();
                if (!line.isEmpty()) tests.push_back(line.toStdString());
            }
        }
    } else {
        for (int a = 4; a < argc; ++a) tests.push_back(argv[a]);
    }

    for (const auto& path : tests) {
        cv::Mat g, p;
        cv::Point2d c;
        if (!readGrayU8(QString::fromUtf8(path.c_str()), g)) {
            std::printf("[diag] 读取失败: %s\n", path.c_str());
            continue;
        }
        const QString fn = QFileInfo(QString::fromUtf8(path.c_str())).fileName();
        if (!straightenLogo(model, tmpl, g, half, p, c)) {
            std::printf("%-18s  对齐失败\n", fn.toStdString().substr(0, 18).c_str());
            continue;
        }

        Metrics m;
        cv::Mat maskF;
        ringMask.convertTo(maskF, CV_32F, 1.0 / 255.0);
        cv::Mat segF;
        segMask.convertTo(segF, CV_32F, 1.0 / 255.0);
        m.tplCorr = corrOf(p, ref);
        m.ringCorr = corrOf(p.mul(maskF), ref.mul(maskF));
        m.segCorr = corrOf(p.mul(segF), ref.mul(segF));

        cv::Mat diff = p - ref;   // 负值=比模板暗（墨缺）
        cv::Mat absd;
        cv::absdiff(p, ref, absd);
        m.absDiff = cv::mean(absd, ringMask)[0];

        cv::Mat dark = diff.clone();
        cv::threshold(dark, dark, 0.0, 0.0, cv::THRESH_TOZERO_INV);   // 只留负值
        m.deficit = -cv::mean(dark, ringMask)[0];
        m.segDeficit = -cv::mean(dark, segMask)[0];

        // 最差单列：环带内逐列相关，取最小
        double wc = 1.0;
        for (int x = 0; x < half * 2; ++x) {
            std::vector<float> a1, b1;
            for (int y = 0; y < half * 2; ++y) {
                if (!ringMask.at<uchar>(y, x)) continue;
                a1.push_back(p.at<float>(y, x));
                b1.push_back(ref.at<float>(y, x));
            }
            if (a1.size() < 10) continue;
            cv::Mat ma(1, (int)a1.size(), CV_32F, a1.data());
            cv::Mat mb(1, (int)b1.size(), CV_32F, b1.data());
            wc = std::min(wc, corrOf(ma, mb));
        }
        m.worstCol = wc;

        std::printf("%-18s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n",
                    fn.toStdString().substr(0, 18).c_str(), m.tplCorr, m.ringCorr, m.segCorr,
                    m.absDiff, m.deficit, m.segDeficit, m.worstCol);
    }
    return 0;
}
