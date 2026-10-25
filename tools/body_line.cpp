// body_line.cpp —— 用直线找塑封体矩形
//
// 前一版按"最大暗区 / 最均匀暗区"找塑封体，两次都选中了整张图：
// 塑封体和外部背景在二值图里是连通的，中间没有暗影隔断。
//
// 换法：塑封体是画面里最长的四条直线围成的矩形。
//   1. Canny 求边缘
//   2. HoughLinesP 找长直线（长度 >= 画面短边 30%）
//   3. 按角度分成横/竖两组
//   4. 取最外侧的两条横线、两条竖线，组成矩形
//
// 这样塑封体的边被精确找到，引脚歪不歪都不影响——
// 引脚是几十像素的短块，塑封体边是上千像素的长线，尺度差一个数量级。
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: markloc_bodyline <imageDir...>\n");
        return 2;
    }
    for (int ai = 1; ai < argc; ++ai) {
        const fs::path dir(argv[ai]);
        if (!fs::is_directory(dir)) continue;
        std::printf("\n===== %s =====\n", dir.string().c_str());
        std::printf("%-42s %20s %8s %9s %7s\n", "文件", "塑封体矩形", "宽高比", "宽高像素", "线数");
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

            cv::Mat g;
            cv::GaussianBlur(gray, g, cv::Size(0, 0), 1.5);
            cv::Mat edges;
            cv::Canny(g, edges, 40, 120);

            const int shortSide = std::min(gray.cols, gray.rows);
            const int minLen = (int)(shortSide * 0.30);
            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(edges, lines, 1, CV_PI / 360, 80, minLen, 20);

            std::vector<int> xs, ys;   // 竖线的 x、横线的 y
            int usedH = 0, usedV = 0;
            for (const auto& l : lines) {
                const int dx = l[2] - l[0], dy = l[3] - l[1];
                const double len = std::hypot(dx, dy);
                if (len < minLen) continue;
                const double ang = std::atan2(std::abs(dy), std::abs(dx)) * 180 / CV_PI;
                if (ang < 15) {          // 横线
                    ys.push_back((l[1] + l[3]) / 2);
                    ++usedH;
                } else if (ang > 75) {   // 竖线
                    xs.push_back((l[0] + l[2]) / 2);
                    ++usedV;
                }
            }
            if (xs.empty() || ys.empty()) {
                std::printf("%-42s  未找到矩形边\n", tag.c_str());
                continue;
            }
            std::sort(xs.begin(), xs.end());
            std::sort(ys.begin(), ys.end());
            // 聚类：把相差 < 15px 的线并成一条，取每组的中位数
            auto cluster = [](std::vector<int> v) {
                std::sort(v.begin(), v.end());
                std::vector<int> out;
                size_t i = 0;
                while (i < v.size()) {
                    size_t j = i;
                    while (j + 1 < v.size() && v[j + 1] - v[j] <= 15) ++j;
                    out.push_back(v[(i + j) / 2]);
                    i = j + 1;
                }
                return out;
            };
            const auto cx = cluster(xs), cy = cluster(ys);
            if (cx.size() < 2 || cy.size() < 2) {
                std::printf("%-42s  聚类后不足 2x2\n", tag.c_str());
                continue;
            }
            const int L = cx.front(), R = cx.back();
            const int T = cy.front(), B = cy.back();
            const int W = R - L, H = B - T;
            const double ar = H > 0 ? (double)W / H : 0;
            char bb[64], wh[32];
            std::snprintf(bb, sizeof(bb), "(%d,%d,%d,%d)", L, T, W, H);
            std::snprintf(wh, sizeof(wh), "%dx%d", W, H);
            std::printf("%-42s %20s %8.4f %9s %3d/%d\n", tag.c_str(), bb, ar, wh, usedV, usedH);

            if (files.size() <= 12) {
                cv::Mat vis;
                cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
                cv::rectangle(vis, cv::Rect(L, T, W, H), cv::Scalar(0, 255, 0), 3);
                for (int x : cx) cv::line(vis, cv::Point(x, T), cv::Point(x, B),
                                          cv::Scalar(0, 200, 255), 1);
                for (int y : cy) cv::line(vis, cv::Point(L, y), cv::Point(R, y),
                                          cv::Scalar(0, 200, 255), 1);
                cv::imwrite(std::string("out/trace/bodyline_") + tag + ".png", vis);
                std::printf("     -> out/trace/bodyline_%s.png\n", tag.c_str());
            }
        }
    }
    return 0;
}
