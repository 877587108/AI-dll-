// projalign.cpp —— 用字符框内的垂直投影做逐字符对齐
//
// 为什么换法：现在用 matchTemplate 整块匹配，它把"框里所有像素"
// 当成一个整体，框内空白多、字在框里偏一点，峰值还很高（实测 0.85~0.94），
// 但算出来的位移到底是"字偏了"还是"框画歪了"分不清。
//
// 垂直投影（每列求和，得到一条一维曲线）只关心"墨往哪一列堆"，
// 对垂直方向的空白完全不敏感，算出来的横向位移就是字形重心的位移，
// 物理意义明确，也不受框面积影响。
//
// 做法：
//   1. 模板投影  P_model(x) = sum_y  freq[box.y .. box.y+h][x]
//   2. 图像投影  P_img(x)   = sum_y  sig [box.y .. box.y+h][x]
//   3. 在 ±R 范围内找 P_img 相对 P_model 的平移量，取归一化互相关最大处
//   4. 分别对水平、垂直各做一次（垂直用水平投影）
//
// 判读标准与 warpsign 一致：0109_RT 的命中数应当下降，0026_FT 必须保住。
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

#include "chardet.hpp"
#include "gui/scratch.hpp"

namespace fs = std::filesystem;

static cv::Mat readTif(const std::string& p) {
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

// 一维序列在 [-R,R] 内的最佳平移量（归一去均值互相关）
// a = 参考（模板），b = 待测（图像）。返回使 b 与 a 对齐所需的位移。
static int bestShift1D(const std::vector<float>& a, const std::vector<float>& b, int R) {
    const int n = (int)a.size();
    if (n < 4) return 0;
    auto corr = [&](int d) -> double {
        // 认为 b(x) ≈ a(x - d)，即 b 相对 a 右移了 d
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        int cnt = 0;
        for (int x = 0; x < n; ++x) {
            const int xa = x;
            const int xb = x + d;
            if (xb < 0 || xb >= n) continue;
            const double va = a[xa], vb = b[xb];
            sa += va;
            sb += vb;
            saa += va * va;
            sbb += vb * vb;
            sab += va * vb;
            ++cnt;
        }
        if (cnt < n / 2) return -2.0;   // 重叠太少，不算
        const double ma = sa / cnt, mb = sb / cnt;
        const double va = saa / cnt - ma * ma;
        const double vb = sbb / cnt - mb * mb;
        if (va < 1e-9 || vb < 1e-9) return -2.0;
        return (sab / cnt - ma * mb) / std::sqrt(va * vb);
    };
    int bestD = 0;
    double bestV = -3.0;
    for (int d = -R; d <= R; ++d) {
        const double v = corr(d);
        if (v > bestV) {
            bestV = v;
            bestD = d;
        }
    }
    return bestD;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_projalign <modelDir> <image...> [--radius 16]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    int radius = 16;
    bool useWarp = true;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--radius") radius = std::atoi(argv[i + 1]);
        else if (k == "--warp") useWarp = std::atoi(argv[i + 1]) != 0;
    }

    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--radius" || s == "--warp") { ++i; continue; }
        if (!fs::exists(s)) continue;
        files.push_back(s);
    }

    for (const auto& p8 : files) {
        cv::Mat gray = readTif(p8);
        if (gray.empty() || gray.size() != model.freq.size()) continue;
        std::printf("\n===== %s =====\n", fs::path(p8).filename().string().c_str());

        cv::Mat sig0 =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        std::vector<chardet::CharUnit> units = baseUnits;
        if (!model.ref.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig0.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : units) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig0.cols, sig0.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            inkArea.x = std::max(0, inkArea.x - 20);
            inkArea.y = std::max(0, inkArea.y - 20);
            inkArea.width = std::min(sig0.cols - inkArea.x, inkArea.width + 40);
            inkArea.height = std::min(sig0.rows - inkArea.y, inkArea.height + 40);
            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            sig0(inkArea).convertTo(sC, CV_32F);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::Mat aligned;
                cv::warpAffine(sig0, aligned, M, sig0.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                sig0 = aligned;
            }
        }

        // 逐字符：垂直投影求 dx，水平投影求 dy
        std::vector<cv::Point> sh(units.size(), cv::Point(0, 0));
        std::vector<char> ok(units.size(), 0);
        std::printf("%4s %14s %8s %8s\n", "序", "框 w x h", "dx投影", "dy投影");
        for (size_t ui = 0; ui < units.size(); ++ui) {
            const cv::Rect box = units[ui].bbox & cv::Rect(0, 0, sig0.cols, sig0.rows);
            if (box.width <= 4 || box.height <= 4) continue;
            // 取框外扩一点的范围，避免字形偏出框后投影被截断
            cv::Rect sr(box.x - radius, box.y - radius, box.width + 2 * radius,
                        box.height + 2 * radius);
            sr &= cv::Rect(0, 0, sig0.cols, sig0.rows);
            if (sr.width < box.width || sr.height < box.height) continue;

            const cv::Mat tpl = model.freq(box);
            const cv::Mat smp = sig0(sr);

            // 垂直投影：每列求和，长度 = 框宽
            // 在 sr 范围内滑动，窗口宽 = box.width
            std::vector<float> pm(box.width, 0.0f);
            for (int x = 0; x < box.width; ++x) {
                double s = 0;
                const int xx = box.x - sr.x + x;
                if (xx < 0 || xx >= sr.width) continue;
                for (int y = box.y - sr.y; y < box.y - sr.y + box.height; ++y) {
                    if (y < 0 || y >= sr.height) continue;
                    s += smp.at<float>(y, xx);
                }
                pm[x] = (float)s;
            }
            // 模板同样的投影
            std::vector<float> qm(box.width, 0.0f);
            for (int x = 0; x < box.width; ++x) {
                double s = 0;
                for (int y = 0; y < tpl.rows; ++y) s += tpl.at<uchar>(y, x);
                qm[x] = (float)s;
            }
            const int ddx = bestShift1D(qm, pm, radius);

            // 水平投影：每行求和
            std::vector<float> pmY(box.height, 0.0f);
            for (int y = 0; y < box.height; ++y) {
                double s = 0;
                const int yy = box.y - sr.y + y;
                if (yy < 0 || yy >= sr.height) continue;
                for (int x = box.x - sr.x; x < box.x - sr.x + box.width; ++x) {
                    if (x < 0 || x >= sr.width) continue;
                    s += smp.at<float>(yy, x);
                }
                pmY[y] = (float)s;
            }
            std::vector<float> qmY(box.height, 0.0f);
            for (int y = 0; y < box.height; ++y) {
                double s = 0;
                for (int x = 0; x < tpl.cols; ++x) s += tpl.at<uchar>(y, x);
                qmY[y] = (float)s;
            }
            const int ddy = bestShift1D(qmY, pmY, radius);

            sh[ui] = cv::Point(ddx, ddy);
            ok[ui] = 1;
            std::printf("%4d %14s %+8d %+8d\n", units[ui].index,
                        (std::to_string(box.width) + "x" + std::to_string(box.height)).c_str(),
                        ddx, ddy);
        }

        // 搬像素 + 搬框
        cv::Mat warped = sig0.clone();
        if (useWarp) {
            for (size_t ui = 0; ui < units.size(); ++ui) {
                if (!ok[ui]) continue;
                const cv::Point s = sh[ui];
                if (s.x == 0 && s.y == 0) continue;
                if (std::abs(s.x) > radius || std::abs(s.y) > radius) continue;
                cv::Rect box = units[ui].bbox;
                box.x -= 6;
                box.y -= 6;
                box.width += 12;
                box.height += 12;
                box &= cv::Rect(0, 0, sig0.cols, sig0.rows);
                if (box.width <= 2 || box.height <= 2) continue;
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, s.x, 0, 1, s.y);
                cv::Mat piece;
                cv::warpAffine(sig0(box), piece, M, box.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REFLECT);
                piece.copyTo(warped(box));
                units[ui].bbox.x += s.x;
                units[ui].bbox.y += s.y;
            }
        } else {
            // 只挪框（与生产侧 perCharAlign 等价）
            for (size_t ui = 0; ui < units.size(); ++ui) {
                if (!ok[ui]) continue;
                units[ui].bbox.x += sh[ui].x;
                units[ui].bbox.y += sh[ui].y;
            }
        }

        markloc::ScratchParams sp;
        markloc::ScratchStats st;
        auto hits = markloc::findScratches(warped, model.freq, units, cfg, sp, &st);
        int best = 0;
        for (const auto& h : hits) best = std::max(best, (int)h.composite);
        std::printf("  -> 投影对齐%s: 命中 %d 条，最高 S=%d\n", useWarp ? "+搬像素" : "(只挪框)",
                    (int)hits.size(), best);
        cv::imwrite("out/trace/projalign_" + fs::path(p8).filename().string().substr(0, 4) + ".png",
                    warped);
    }
    return 0;
}
