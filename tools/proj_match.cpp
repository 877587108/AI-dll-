// projmatch.cpp —— 用已有图的投影做基准，匹配新图
//
// 用户思路：和已有的图做投影比对。
//
// 为什么可行：同一颗料在同一工位拍的图，投影曲线（每列/每行的梯度能量累加）
// 形状是稳定的——引脚在固定列、塑封体边在固定列。
// 换参数后只是裁剪窗口/尺度变了，曲线形状还在，只是被平移或缩放。
//
// 所以不逐个找边，而是整条曲线做一维互相关：
//   1. 从旧规格的图算基准投影曲线（多张平均，压掉噪声）
//   2. 对新图算它的投影曲线
//   3. 在 ±50% 的尺度范围、±300px 的平移范围内找最佳匹配
//   4. 最佳匹配的 (scale, shift) 就是两个规格之间的映射关系
//
// 先决条件：旧规格的曲线在自己内部必须稳定。本工具先验证这一条，
// 不稳定就说明基准本身不可信，后面全是空的。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

static cv::Mat readAny(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::vector<uchar> buf;
    uchar tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(f);
    if (buf.empty()) return {};
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

// 一维投影：返回归一化后的曲线（均值 0、方差 1），便于跨图比较
static std::vector<double> project(const cv::Mat& gray, bool vertical) {
    cv::Mat g;
    cv::GaussianBlur(gray, g, cv::Size(0, 0), 1.5);
    cv::Mat gx, gy;
    cv::Sobel(g, gx, CV_32F, 1, 0, 3);
    cv::Sobel(g, gy, CV_32F, 0, 1, 3);
    const cv::Mat& m = vertical ? gx : gy;
    cv::Mat a;
    cv::absdiff(m, cv::Mat::zeros(m.size(), m.type()), a);
    const int n = vertical ? gray.cols : gray.rows;
    std::vector<double> p(n, 0.0);
    for (int y = 0; y < a.rows; ++y) {
        const float* row = a.ptr<float>(y);
        for (int x = 0; x < a.cols; ++x) p[vertical ? x : y] += row[x];
    }
    // 归一化
    double mean = 0;
    for (double v : p) mean += v;
    mean /= n;
    double var = 0;
    for (double v : p) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / n);
    if (sd > 1e-9)
        for (double& v : p) v = (v - mean) / sd;
    return p;
}

// 两条曲线在给定尺度与平移下的相关系数
// model 为基准（长 n0），img 为待测（长 n1）
// 认为 img 的坐标 x 对应 model 的 (x - shift) / scale
static double score(const std::vector<double>& model, const std::vector<double>& img,
                    double scale, int shift) {
    const int n0 = (int)model.size(), n1 = (int)img.size();
    double s0 = 0, s1 = 0, s00 = 0, s11 = 0, s01 = 0;
    int cnt = 0;
    for (int x = 0; x < n1; ++x) {
        const double xm = (x - shift) / scale;
        const int xi = (int)std::lround(xm);
        if (xi < 0 || xi >= n0) continue;
        const double a = model[xi], b = img[x];
        s0 += a; s1 += b; s00 += a * a; s11 += b * b; s01 += a * b;
        ++cnt;
    }
    if (cnt < n0 / 2) return -2.0;
    const double ma = s0 / cnt, mb = s1 / cnt;
    const double va = s00 / cnt - ma * ma, vb = s11 / cnt - mb * mb;
    if (va < 1e-9 || vb < 1e-9) return -2.0;
    return (s01 / cnt - ma * mb) / std::sqrt(va * vb);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_projmatch <基准图Dir> <待测图Dir>\n");
        return 2;
    }
    // ---- 1. 从基准目录建平均投影
    std::vector<std::string> base;
    for (const auto& e : fs::directory_iterator(fs::path(argv[1]))) {
        if (!e.is_regular_file()) continue;
        const std::string stem = e.path().stem().string();
        if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
        base.push_back(e.path().string());
    }
    std::sort(base.begin(), base.end());
    if (base.empty()) {
        std::printf("基准目录无图\n");
        return 1;
    }

    std::vector<double> avgV, avgH;
    std::vector<std::vector<double>> allV, allH;
    const int N = std::min<int>(12, (int)base.size());   // 取前 12 张做基准
    for (int i = 0; i < N; ++i) {
        cv::Mat g = readAny(base[i]);
        if (g.empty()) continue;
        auto pv = project(g, true);
        auto ph = project(g, false);
        allV.push_back(pv);
        allH.push_back(ph);
        if (avgV.empty()) { avgV = pv; avgH = ph; }
        else {
            for (size_t k = 0; k < pv.size(); ++k) avgV[k] += pv[k];
            for (size_t k = 0; k < ph.size(); ++k) avgH[k] += ph[k];
        }
    }
    for (double& v : avgV) v /= N;
    for (double& v : avgH) v /= N;

    // ---- 2. 先验证基准内部是否稳定
    std::printf("=== 基准内部稳定性（%d 张旧规格图，两两相关）===\n", N);
    double worst = 1.0, sum = 0;
    int pairs = 0;
    for (size_t i = 0; i < allV.size(); ++i)
        for (size_t j = i + 1; j < allV.size(); ++j) {
            const double c = score(allV[i], allV[j], 1.0, 0);
            worst = std::min(worst, c);
            sum += c;
            ++pairs;
        }
    std::printf("垂直投影相关: 平均 %.4f  最低 %.4f（%d 对）\n", pairs ? sum / pairs : 0, worst,
                pairs);
    std::printf("判读: 若最低值 < 0.70，基准本身不稳，后面的匹配不可信\n");

    // ---- 3. 待测图匹配
    std::vector<std::string> tests;
    for (const auto& e : fs::directory_iterator(fs::path(argv[2]))) {
        if (!e.is_regular_file()) continue;
        const std::string stem = e.path().stem().string();
        if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
        tests.push_back(e.path().string());
    }
    std::sort(tests.begin(), tests.end());
    std::printf("\n=== 待测图匹配 ===\n");
    std::printf("%-16s %8s %8s %8s %8s\n", "文件", "最佳尺度", "最佳平移", "相关", "判定");
    for (const auto& f : tests) {
        cv::Mat g = readAny(f);
        if (g.empty()) continue;
        auto pv = project(g, true);
        double bestS = 1, bestSc = -2;
        int bestShift = 0;
        for (double s = 0.55; s <= 1.45; s += 0.01) {
            for (int sh = -400; sh <= 400; sh += 4) {
                const double c = score(avgV, pv, s, sh);
                if (c > bestSc) { bestSc = c; bestS = s; bestShift = sh; }
            }
        }
        // 细化平移
        for (int sh = bestShift - 4; sh <= bestShift + 4; ++sh) {
            const double c = score(avgV, pv, bestS, sh);
            if (c > bestSc) { bestSc = c; bestShift = sh; }
        }
        char tag[32];
        std::snprintf(tag, sizeof(tag), "%.16s", fs::path(f).filename().stem().string().c_str());
        std::printf("%-16s %8.3f %8d %8.4f %8s\n", tag, bestS, bestShift, bestSc,
                    bestSc > 0.80 ? "好" : (bestSc > 0.60 ? "勉强" : "差"));
    }
    return 0;
}
