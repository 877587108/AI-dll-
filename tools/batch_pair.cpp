// batch_pair.cpp —— 同批样本之间的相似性 / 差异分析
//
// 【第 9 轮的动机】
// 前面所有尝试都是"ngv2 vs unset"两类对比，全部失败。
// 但 ngv2 是同一批料（文件名连续编号），它们之间可能存在一个**共享的、但
// 与正常料不同的**结构 —— 只是这个结构在单张统计量上体现不出来。
//
// 做法：把 ngv2 的 7 张两两比较，找出"它们彼此高度一致、
// 但一致的那个模式在正常料里不常见"的部分。
//
// 具体：
//   ① 环带向量化（与 PCA 工具同一套归一化）
//   ② 算 ngv2 内部的平均向量（batch centroid）
//   ③ 算每个 ngv2 到 centroid 的距离  -> "这批料的内部一致性"
//   ④ 算每张正常料到 centroid 的距离  -> "正常料离这批有多远"
// 如果 ngv2 内部很紧、而正常料离得远，就说明存在批次特征。
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

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("usage: markloc_batchpair <modelRoot> <tmpl.png> <batchDir> <normalDir>\n");
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

    QDir bdir(QString::fromLocal8Bit(argv[3]));
    QDir ndir(QString::fromLocal8Bit(argv[4]));
    std::vector<cv::Mat> batch, normals;
    QStringList bn, nn;
    for (const QFileInfo& fi : bdir.entryInfoList({QStringLiteral("*_00.bmp"), QStringLiteral("*_00.tif")},
                                                  QDir::Files, QDir::Name)) {
        cv::Mat g, v;
        if (!readGrayU8(fi.absoluteFilePath(), g)) continue;
        if (!ringVector(model, tmpl, g, v)) continue;
        batch.push_back(v);
        bn << fi.fileName();
    }
    for (const QFileInfo& fi : ndir.entryInfoList({QStringLiteral("*_00.tif")}, QDir::Files, QDir::Name)) {
        cv::Mat g, v;
        if (!readGrayU8(fi.absoluteFilePath(), g)) continue;
        if (!ringVector(model, tmpl, g, v)) continue;
        normals.push_back(v);
        nn << fi.fileName();
    }
    if (batch.size() < 3) {
        std::printf("批次样本太少\n");
        return 1;
    }
    std::printf("批次 %d 张，正常料 %d 张\n", (int)batch.size(), (int)normals.size());

    // 批次质心
    cv::Mat centroid = cv::Mat::zeros(1, batch[0].cols, CV_32F);
    for (const auto& v : batch) centroid += v;
    centroid /= static_cast<double>(batch.size());

    // 批次内部两两距离（说明这批料彼此有多像）
    double sumIn = 0, maxIn = 0;
    int cntIn = 0;
    for (size_t i = 0; i < batch.size(); ++i)
        for (size_t j = i + 1; j < batch.size(); ++j) {
            const double d = cv::norm(batch[i], batch[j], cv::NORM_L2);
            sumIn += d;
            maxIn = std::max(maxIn, d);
            ++cntIn;
        }
    const double meanIn = cntIn ? sumIn / cntIn : 0;

    // 正常料两两距离（参照）
    double sumNo = 0, maxNo = 0;
    int cntNo = 0;
    for (size_t i = 0; i < normals.size(); ++i)
        for (size_t j = i + 1; j < normals.size(); ++j) {
            const double d = cv::norm(normals[i], normals[j], cv::NORM_L2);
            sumNo += d;
            maxNo = std::max(maxNo, d);
            ++cntNo;
        }
    const double meanNo = cntNo ? sumNo / cntNo : 0;

    std::printf("\n批次内部两两距离: 均值 %.2f  最大 %.2f\n", meanIn, maxIn);
    std::printf("正常料两两距离:   均值 %.2f  最大 %.2f\n", meanNo, maxNo);

    std::printf("\n各样本到【批次质心】的距离:\n");
    std::printf("%-24s %10s\n", "file", "toCentroid");
    for (size_t i = 0; i < batch.size(); ++i)
        std::printf("%-24s %10.2f\n", bn[(int)i].toStdString().substr(0, 24).c_str(),
                    cv::norm(batch[i], centroid, cv::NORM_L2));

    std::printf("\n正常料中到批次质心最近的 10 张（越小说明越像这批）:\n");
    std::vector<std::pair<double, int>> dv;
    for (size_t i = 0; i < normals.size(); ++i)
        dv.emplace_back(cv::norm(normals[i], centroid, cv::NORM_L2), (int)i);
    std::sort(dv.begin(), dv.end());
    for (size_t k = 0; k < dv.size() && k < 10; ++k)
        std::printf("  %10.2f  %s\n", dv[k].first,
                    nn[dv[k].second].toStdString().substr(0, 40).c_str());
    return 0;
}
