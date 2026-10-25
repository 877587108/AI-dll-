// pca_logo.cpp —— 用正常料建立 logo 外观模型，测异常样本的重建残差
//
// 【为什么换这个思路】
// 前几轮把"环的墨量"这一族统计量查尽了（角度缺口、粗细、峰值、相对比值、
// 拓扑条数、8 个纹理特征及其两两/三三组合），结果都是两类同分布。
//
// 根因可能是：这些统计量都是**手工设计的单一项**，而缺陷可能表现为
// 一个"整体不像正常样"的模式，落在很多项的微妙组合里 —— 单项都正常，
// 合起来不对。人眼判断"这块看起来不对劲"靠的就是这种整体外观比对。
//
// 所以改用**外观建模**：
//   ① 把每张图的环按极坐标拉直、归一化到同一尺寸
//   ② 用 158 张正常料做 PCA，得到"正常环外观"的主成分子空间
//   ③ 把待测环投影到这个子空间再重建
//   ④ 重建残差 = 这张环有多少"正常料从未出现过的结构"
//
// 这与逐项统计量的区别：PCA 捕捉的是各像素之间的联合协方差，
// 能表达"这个位置的这种组合从未见过"，而单项阈值只能表达"这个量偏大"。
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

// 把一张图处理成固定尺寸的"环带向量"
//
// 归一化要小心：不同图的曝光与墨水浓度不同，若直接用原始灰度，
// PCA 的第一主成分会被整体亮度占满，学不到形状。
// 这里对每张环带先做 0~255 归一化，再做 z-score，
// 目的就是让 PCA 专注"结构"，而不是"亮度"。
static bool ringVector(const chardet::Model& model, const cv::Mat& tmpl, const cv::Mat& g,
                       cv::Mat& vec, double* matchPeak) {
    const double tplCx = 127.8, tplCy = 116.0, tplA = 132.1, tplB = 76.75, tplRot = -13.78;
    cv::Mat mres;
    cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
    cv::Point mloc;
    double mval = 0;
    cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
    if (matchPeak) *matchPeak = mval;

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
    if (roi.area() <= 0) return false;
    const cv::Point2f cInRoi(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
    const double maxR = A + pad;
    const int nRadius = static_cast<int>(std::lround(maxR));
    cv::Mat pol;
    cv::warpPolar(straight(roi), pol, cv::Size(nRadius, 360), cInRoi, maxR,
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
    cv::Mat polT;
    cv::transpose(pol, polT);
    pol = polT;   // rows=半径 cols=角度

    // 只取环所在的半径带，并把整条带缩放成固定尺寸（72 角度 x 24 半径）
    const int bLo = static_cast<int>(std::lround(A * 0.70));
    const int bHi = std::min(static_cast<int>(std::lround(A * 1.40)), pol.rows - 1);
    if (bHi <= bLo + 2) return false;
    cv::Mat band = pol(cv::Rect(0, bLo, pol.cols, bHi - bLo + 1));
    cv::Mat fixed;
    cv::resize(band, fixed, cv::Size(72, 24), 0, 0, cv::INTER_AREA);

    // 归一化：先 0~255，再 z-score
    cv::Mat f32;
    fixed.convertTo(f32, CV_32F);
    cv::normalize(f32, f32, 0, 255, cv::NORM_MINMAX);
    cv::Scalar mu, sd;
    cv::meanStdDev(f32, mu, sd);
    f32 = (f32 - mu[0]) / std::max(1e-6, sd[0]);

    vec = f32.reshape(1, 1).clone();
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("usage: markloc_pcalogo <modelRoot> <tmpl.png> <normalDir> <testImg...>\n");
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

    // ---- 训练：用正常料建 PCA
    QDir ndir(QString::fromLocal8Bit(argv[3]));
    QStringList normals;
    for (const QFileInfo& fi : ndir.entryInfoList({QStringLiteral("*_00.tif")}, QDir::Files, QDir::Name))
        normals << fi.absoluteFilePath();
    std::printf("训练集 %d 张\n", static_cast<int>(normals.size()));

    cv::Mat X;   // 每行一个样本
    for (const QString& p : normals) {
        cv::Mat g;
        if (!readGrayU8(p, g)) continue;
        cv::Mat v;
        if (!ringVector(model, tmpl, g, v, nullptr)) continue;
        X.push_back(v);
    }
    if (X.rows < 10) {
        std::printf("训练样本太少(%d)\n", X.rows);
        return 1;
    }
    std::printf("特征维度 %d，样本 %d\n", X.cols, X.rows);

    // PCA：保留能解释 95% 方差的主成分
    cv::PCA pca(X, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.95);
    std::printf("保留主成分 %d 个\n\n", pca.eigenvectors.rows);

    // 先在训练集上算残差分布，作为"正常"的参照
    std::vector<double> trainRes;
    for (int i = 0; i < X.rows; ++i) {
        const cv::Mat proj = pca.project(X.row(i));
        const cv::Mat rec = pca.backProject(proj);
        trainRes.push_back(cv::norm(X.row(i), rec, cv::NORM_L2));
    }
    std::vector<double> tr = trainRes;
    std::sort(tr.begin(), tr.end());
    const double trMed = tr[tr.size() / 2];
    const double trMax = tr.back();
    const double trP99 = tr[static_cast<size_t>(tr.size() * 0.99)];
    std::printf("训练集残差：中位 %.3f  p99 %.3f  最大 %.3f\n", trMed, trP99, trMax);

    std::printf("\n%-20s %8s %10s %8s\n", "file", "resid", "vs p99", "峰值");
    for (int a = 4; a < argc; ++a) {
        cv::Mat g;
        if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g)) continue;
        cv::Mat v;
        double pk = 0;
        if (!ringVector(model, tmpl, g, v, &pk)) continue;
        const cv::Mat proj = pca.project(v);
        const cv::Mat rec = pca.backProject(proj);
        const double res = cv::norm(v, rec, cv::NORM_L2);
        const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
        std::printf("%-20s %8.3f %10.2f %8.3f\n", fn.toStdString().substr(0, 20).c_str(), res,
                    res / std::max(1e-6, trP99), pk);
    }
    return 0;
}
