// body_proj.cpp —— 用水平/垂直投影找塑封体四条边
//
// 前三次定位（最大暗区 / 最均匀暗区 / 长直线）都失败：
//   暗区法选中整张图（塑封体与背景连通）
//   直线法找不到连续长直线（边被引脚和刻印打断）
//
// 投影法的优势：不要求边连续。对每一列把"垂直方向的梯度能量"累加，
// 塑封体左右边所在的列会形成尖峰（那一列全是亮度突变），
// 即使被引脚打断，被打断的只是少数行，累加后尖峰依旧显著。
//
// 步骤：
//   1. Sobel 求 x 方向梯度（找竖边）与 y 方向梯度（找横边）
//   2. 分别对列、行累加得到一维投影曲线
//   3. 在曲线里找最强的左右两个峰 = 塑封体左右边
//      上下两个峰 = 塑封体上下边
//   4. 峰的位置用抛物线插值做亚像素修正
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

struct BodyRect {
    bool found = false;
    int L = 0, T = 0, R = 0, B = 0;
    double peakL = 0, peakR = 0, peakT = 0, peakB = 0;
    // 用于输出曲线
    std::vector<double> projX, projY;
};

// 在一维曲线里找最强的两个峰（相隔至少 minSep）
static void twoStrongest(const std::vector<double>& v, int minSep, int* i1, int* i2) {
    *i1 = *i2 = -1;
    double b1 = -1, b2 = -1;
    const int n = (int)v.size();
    for (int i = 1; i < n - 1; ++i) {
        if (v[i] < v[i - 1] || v[i] < v[i + 1]) continue;   // 只取局部极大
        if (v[i] > b1) {
            if (*i1 >= 0 && std::abs(i - *i1) < minSep) {
                if (v[i] > b1) { b1 = v[i]; *i1 = i; }
                continue;
            }
            b2 = b1;
            *i2 = *i1;
            b1 = v[i];
            *i1 = i;
        } else if (v[i] > b2 && std::abs(i - *i1) >= minSep) {
            b2 = v[i];
            *i2 = i;
        }
    }
}

static BodyRect locate(const cv::Mat& gray) {
    BodyRect r;
    cv::Mat g;
    cv::GaussianBlur(gray, g, cv::Size(0, 0), 1.5);

    // x 方向梯度 -> 竖边
    cv::Mat gx, gy, ax, ay;
    cv::Sobel(g, gx, CV_32F, 1, 0, 3);
    cv::Sobel(g, gy, CV_32F, 0, 1, 3);
    cv::absdiff(gx, cv::Mat::zeros(gx.size(), gx.type()), ax);
    cv::absdiff(gy, cv::Mat::zeros(gy.size(), gy.type()), ay);

    // 列投影（找左右边）：每一列把该列所有行的 |gx| 累加
    std::vector<double> px(gray.cols, 0.0), py(gray.rows, 0.0);
    for (int y = 0; y < ax.rows; ++y) {
        const float* row = ax.ptr<float>(y);
        for (int x = 0; x < ax.cols; ++x) px[x] += row[x];
    }
    for (int y = 0; y < ay.rows; ++y) {
        const float* row = ay.ptr<float>(y);
        for (int x = 0; x < ay.cols; ++x) py[y] += row[x];
    }
    r.projX = px;
    r.projY = py;

    const int minSepX = std::max(50, gray.cols / 4);
    const int minSepY = std::max(50, gray.rows / 4);
    int l = -1, rr = -1, t = -1, b = -1;
    twoStrongest(px, minSepX, &l, &rr);
    twoStrongest(py, minSepY, &t, &b);
    if (l < 0 || rr < 0 || t < 0 || b < 0) return r;
    if (l > rr) std::swap(l, rr);
    if (t > b) std::swap(t, b);

    r.found = true;
    r.L = l;
    r.R = rr;
    r.T = t;
    r.B = b;
    r.peakL = px[l];
    r.peakR = px[rr];
    r.peakT = py[t];
    r.peakB = py[b];
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: markloc_bodyproj <imageDir...> [--save dir]\n");
        return 2;
    }
    std::string saveDir = "out/trace";
    std::vector<std::string> dirs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--save" && i + 1 < argc) saveDir = argv[++i];
        else dirs.push_back(a);
    }
    for (const auto& d : dirs) {
        const fs::path dir(d);
        if (!fs::is_directory(dir)) continue;
        std::printf("\n===== %s =====\n", d.c_str());
        std::printf("%-42s %20s %8s %10s %10s\n", "文件", "塑封体矩形", "宽高比", "宽高", "四边峰值");
        std::vector<std::string> files;
        for (const auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            const std::string stem = e.path().stem().string();
            if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
            files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
        for (const auto& f : files) {
            cv::Mat gray = readAny(f);
            if (gray.empty()) continue;
            const std::string tag = fs::path(f).filename().stem().string();
            BodyRect r = locate(gray);
            if (!r.found) {
                std::printf("%-42s  未找到边\n", tag.c_str());
                continue;
            }
            const int W = r.R - r.L, H = r.B - r.T;
            const double ar = H > 0 ? (double)W / H : 0;
            char bb[64], wh[32], pk[64];
            std::snprintf(bb, sizeof(bb), "(%d,%d,%d,%d)", r.L, r.T, W, H);
            std::snprintf(wh, sizeof(wh), "%dx%d", W, H);
            std::snprintf(pk, sizeof(pk), "%.0f/%.0f/%.0f/%.0f", r.peakL, r.peakR, r.peakT,
                          r.peakB);
            std::printf("%-42s %20s %8.4f %10s %10s\n", tag.c_str(), bb, ar, wh, pk);

            if (files.size() <= 12) {
                // 出图：原图 + 四边 + 两条投影曲线
                cv::Mat vis;
                cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
                cv::rectangle(vis, cv::Rect(r.L, r.T, W, H), cv::Scalar(0, 255, 0), 3);
                // 投影曲线画在顶部/左侧作为条带
                cv::Mat strip(80, vis.cols, CV_8UC3, cv::Scalar(20, 20, 20));
                double mx = 0;
                for (double v : r.projX) mx = std::max(mx, v);
                if (mx > 0)
                    for (int x = 0; x < vis.cols; ++x) {
                        const int h = (int)(r.projX[x] / mx * 75);
                        cv::line(strip, cv::Point(x, 80), cv::Point(x, 80 - h),
                                 cv::Scalar(0, 200, 255), 1);
                    }
                cv::line(strip, cv::Point(r.L, 0), cv::Point(r.L, 80), cv::Scalar(0, 255, 0), 1);
                cv::line(strip, cv::Point(r.R, 0), cv::Point(r.R, 80), cv::Scalar(0, 255, 0), 1);
                cv::Mat out;
                cv::vconcat(strip, vis, out);
                cv::imwrite((saveDir + "/bodyproj_" + tag + ".png"), out);
            }
        }
    }
    return 0;
}
