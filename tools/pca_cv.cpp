// pca_cv.cpp —— PCA 外观模型的留一交叉验证
//
// 【为什么必须做这个】
// 上面 pca_logo 在 158 张正常料上训练、又在同一批上算残差，
// 训练集 p99 可能被"记住"了样本，导致残差被低估。
// 但更关键的问题是：ngv2 的残差高，会不会只是因为
// "它们与训练集的光照/构图略有不同"，而不是真有缺陷？
//
// 留一法能回答这个：
//   对每个正常料，用其余 157 张训练，再算它的残差
//   —— 这是"完全没见过的新正常料"的残差分布，无偏。
// 然后把 ngv2 的残差与这个无偏分布比。
//
// 另外还加一个更严格的对照：把 unset 的 _01 帧（同批连拍的其它帧）
// 当作"未参与训练的正常料"，看它们的残差是否也在低位。
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

static bool ringVector(const chardet::Model& model, const cv::Mat& tmpl, const cv::Mat& g,
                       cv::Mat& vec) {
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
    pol = polT;
    const int bLo = static_cast<int>(std::lround(A * 0.70));
    const int bHi = std::min(static_cast<int>(std::lround(A * 1.40)), pol.rows - 1);
    if (bHi <= bLo + 2) return false;
    cv::Mat band = pol(cv::Rect(0, bLo, pol.cols, bHi - bLo + 1));
    cv::Mat fixed;
    cv::resize(band, fixed, cv::Size(72, 24), 0, 0, cv::INTER_AREA);
    cv::Mat f32;
    fixed.convertTo(f32, CV_32F);
    cv::normalize(f32, f32, 0, 255, cv::NORM_MINMAX);
    cv::Scalar mu, sd;
    cv::meanStdDev(f32, mu, sd);
    f32 = (f32 - mu[0]) / std::max(1e-6, sd[0]);
    vec = f32.reshape(1, 1).clone();
    return true;
}

// 用 X 中除 skip 行以外的样本训练 PCA，返回第 i 行的残差
static double looResidual(const cv::Mat& X, int skip, double varKeep) {
    cv::Mat Xt;
    for (int i = 0; i < X.rows; ++i)
        if (i != skip) Xt.push_back(X.row(i));
    if (Xt.rows < 10) return -1;
    cv::PCA pca(Xt, cv::Mat(), cv::PCA::DATA_AS_ROW, varKeep);
    const cv::Mat proj = pca.project(X.row(skip));
    const cv::Mat rec = pca.backProject(proj);
    return cv::norm(X.row(skip), rec, cv::NORM_L2);
}

// 用完整训练集测一张新图
static double testResidual(const cv::Mat& X, const cv::Mat& v, double varKeep) {
    cv::PCA pca(X, cv::Mat(), cv::PCA::DATA_AS_ROW, varKeep);
    const cv::Mat proj = pca.project(v);
    const cv::Mat rec = pca.backProject(proj);
    return cv::norm(v, rec, cv::NORM_L2);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: markloc_pcacv <modelRoot> <tmpl.png> <normalDir> [testImg...]\n");
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
    const double varKeep = 0.95;

    // ---- 训练集：158 张正常料的 _00
    QDir ndir(QString::fromLocal8Bit(argv[3]));
    QStringList normals;
    for (const QFileInfo& fi : ndir.entryInfoList({QStringLiteral("*_00.tif")}, QDir::Files, QDir::Name))
        normals << fi.absoluteFilePath();

    cv::Mat X;
    for (const QString& p : normals) {
        cv::Mat g;
        if (!readGrayU8(p, g)) continue;
        cv::Mat v;
        if (!ringVector(model, tmpl, g, v)) continue;
        X.push_back(v);
    }
    std::printf("训练集 %d 张，维度 %d\n", X.rows, X.cols);

    // ---- 留一交叉验证：得到"没见过的正常料"的无偏残差分布
    std::vector<double> loo;
    loo.reserve(X.rows);
    for (int i = 0; i < X.rows; ++i) {
        const double r = looResidual(X, i, varKeep);
        if (r >= 0) loo.push_back(r);
    }
    std::sort(loo.begin(), loo.end());
    const double lMed = loo[loo.size() / 2];
    const double lP95 = loo[static_cast<size_t>(loo.size() * 0.95)];
    const double lP99 = loo[static_cast<size_t>(loo.size() * 0.99)];
    const double lMax = loo.back();
    std::printf("\n留一法（无偏）正常料残差：中位 %.3f  p95 %.3f  p99 %.3f  最大 %.3f\n", lMed,
                lP95, lP99, lMax);

    // ---- 测外部图
    if (argc > 4) {
        std::printf("\n%-22s %9s %10s %s\n", "file", "resid", "vs LOO max", "判定(阈值=LOO max)");
        int nTest = 0, nFlag = 0;
        for (int a = 4; a < argc; ++a) {
            cv::Mat g;
            if (!readGrayU8(QString::fromLocal8Bit(argv[a]), g)) continue;
            cv::Mat v;
            if (!ringVector(model, tmpl, g, v)) continue;
            const double r = testResidual(X, v, varKeep);
            ++nTest;
            const bool flag = r > lMax;
            if (flag) ++nFlag;
            const QString fn = QFileInfo(QString::fromLocal8Bit(argv[a])).fileName();
            std::printf("%-22s %9.3f %10.2f %s\n", fn.toStdString().substr(0, 22).c_str(), r,
                        r / std::max(1e-6, lMax), flag ? "超出" : "");
        }
        std::printf("\n测了 %d 张，超出 LOO 最大值 %d 张\n", nTest, nFlag);
    }
    return 0;
}
