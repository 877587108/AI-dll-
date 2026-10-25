// logobreak2.cpp —— logo 逐点断连检测（局部比对，不用全局统计）
//
// 为什么换法：上一版用极坐标扫描数"环被切成几段"，NG 与 OK 完全重叠
// （NG 环段数 3~8，OK 2~11）。原因是断连是**局部**缺陷，
// 全局统计把它淹没在亮度/采样波动里。
//
// 新做法：只和模型比**同一位置的局部结构**。
//   1. 在各自图上找到 logo 椭圆（不假设坐标，靠形状搜索）
//   2. 用椭圆的仿射参数把图上的 logo 归一化到模型 logo 的坐标系
//      （这一步吸收新旧规格的尺度和位移差异）
//   3. 逐像素比较"模型说有墨 / 图上没墨"的位置，聚成团
//   4. 只保留"连续成段"的缺失——那是断口，不是磨损
//
// 输出：断口数量、最大断口长度、断口处的信号缺口深度。
// 判读：OK 的断口应当少而短；断开一处就破坏环的连续性，视为断连。
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

// 在图上找 logo 椭圆。返回椭圆；找不到返回空。
// 判据：外接框长宽比 1.2~3.2、填充率 < 0.45（中间空 = 环）、面积够大。
static bool findLogoEllipse(const cv::Mat& bin, cv::RotatedRect* out, cv::Rect* roiOut) {
    cv::Mat lb, st, ce;
    const int n = cv::connectedComponentsWithStats(bin, lb, st, ce, 8, CV_32S);
    int best = -1;
    double bestScore = -1;
    for (int i = 1; i < n; ++i) {
        const int a = st.at<int>(i, cv::CC_STAT_AREA);
        const int w = st.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = st.at<int>(i, cv::CC_STAT_HEIGHT);
        if (a < 2000 || w < 80 || h < 40) continue;
        const double ar = (double)w / h;
        if (ar < 1.2 || ar > 3.2) continue;
        const double fill = (double)a / (w * (double)h);
        if (fill > 0.45) continue;
        const double sc = a * (1.0 - std::abs(ar - 1.7));
        if (sc > bestScore) {
            bestScore = sc;
            best = i;
        }
    }
    if (best < 0) return false;
    const cv::Rect roi(st.at<int>(best, cv::CC_STAT_LEFT), st.at<int>(best, cv::CC_STAT_TOP),
                       st.at<int>(best, cv::CC_STAT_WIDTH), st.at<int>(best, cv::CC_STAT_HEIGHT));
    *out = cv::RotatedRect(cv::Point2f(roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f),
                           cv::Size2f(roi.width / 2.0f, roi.height / 2.0f), 0);
    *roiOut = roi;
    return true;
}

struct Result {
    bool found = false;
    int breaks = 0;        // 断口数量
    int maxBreakLen = 0;   // 最长断口在环周上的长度（采样点数）
    double meanDepth = 0;  // 断口处信号相对模型墨的缺口
    int totalMissing = 0;  // 所有缺失像素数
    cv::Mat vis;           // 可视化
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_logobreak2 <modelDir> <imageDir...> [--out dir]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;

    // 模型侧 logo：regions.txt 里区域 3 的那个大框
    const auto& rg = model.regions[3];
    cv::Mat freqFull = model.freq;
    cv::Mat modelLogoF = freqFull(rg.rect & cv::Rect(0, 0, freqFull.cols, freqFull.rows)).clone();
    // 模型 logo 的椭圆：在模型自己身上找一次，坐标固定
    cv::Mat modelBin;
    cv::threshold(modelLogoF, modelBin, 38, 255, cv::THRESH_BINARY);
    cv::RotatedRect mEll;
    cv::Rect mRoi;
    if (!findLogoEllipse(modelBin, &mEll, &mRoi)) {
        std::printf("模型里找不到 logo 环，退出\n");
        return 1;
    }
    std::printf("模型 logo: 中心(%.1f,%.1f) 轴(%.1f,%.1f) 外框(%d,%d,%d,%d)\n", mEll.center.x,
                mEll.center.y, mEll.size.width, mEll.size.height, mRoi.x, mRoi.y, mRoi.width,
                mRoi.height);

    for (int ai = 2; ai < argc; ++ai) {
        const fs::path dir(argv[ai]);
        if (!fs::is_directory(dir)) continue;
        std::printf("\n===== %s =====\n", dir.string().c_str());
        std::printf("%-42s %7s %8s %8s %9s\n", "文件", "断口数", "最长断口", "总缺失", "缺口深度");

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

            cv::Mat sig =
                chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
            // 图上找 logo：用信号二值化
            cv::Mat bin;
            cv::compare(sig, 0.25, bin, cv::CMP_GT);
            cv::Mat bk = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
            cv::morphologyEx(bin, bin, cv::MORPH_OPEN, bk);

            cv::RotatedRect ell;
            cv::Rect roi;
            if (!findLogoEllipse(bin, &ell, &roi)) {
                std::printf("%-42s  未找到 logo\n", tag.c_str());
                continue;
            }

            // 用椭圆的外接框把图上的 logo 缩放到模型 logo 的外接框尺寸
            // 这一步吸收新旧规格的尺度差异
            const cv::Rect srcBox = ell.boundingRect() & cv::Rect(0, 0, sig.cols, sig.rows);
            if (srcBox.width < 40 || srcBox.height < 30) continue;
            cv::Mat aligned;
            cv::resize(sig(srcBox), aligned, cv::Size(rg.rect.width, rg.rect.height), 0, 0,
                       cv::INTER_LINEAR);

            // 模型 logo 与对齐后图像都归一到模型坐标系
            cv::Mat mBin;
            cv::compare(modelLogoF, 38, mBin, cv::CMP_GT);
            cv::Mat kOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
            cv::morphologyEx(mBin, mBin, cv::MORPH_OPEN, kOpen);

            // 细微平移搜索：让对齐后的图与模型最佳重合
            int bdx = 0, bdy = 0;
            double bv = -2;
            for (int dy = -6; dy <= 6; ++dy)
                for (int dx = -6; dx <= 6; ++dx) {
                    cv::Rect r(dx, dy, aligned.cols - std::abs(0), aligned.rows);
                    const cv::Rect src(std::max(0, dx), std::max(0, dy),
                                       aligned.cols - std::abs(dx), aligned.rows - std::abs(dy));
                    const cv::Rect dst(std::max(0, -dx), std::max(0, -dy),
                                       aligned.cols - std::abs(dx), aligned.rows - std::abs(dy));
                    if (src.width < 20 || src.height < 20) continue;
                    cv::Mat a, b;
                    aligned(src).convertTo(a, CV_32F);
                    modelLogoF(dst).convertTo(b, CV_32F, 1.0 / 255.0);
                    cv::Scalar ma, mb, sa, sb;
                    cv::meanStdDev(a, ma, sa);
                    cv::meanStdDev(b, mb, sb);
                    if (sa[0] < 1e-6 || sb[0] < 1e-6) continue;
                    const double c = (a - ma[0]).dot(b - mb[0]) / (a.total() * sa[0] * sb[0]);
                    if (c > bv) { bv = c; bdx = dx; bdy = dy; }
                    (void)r;
                }
            cv::Mat best = aligned;
            if (bdx || bdy) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, bdx, 0, 1, bdy);
                cv::warpAffine(aligned, best, M, aligned.size(), cv::INTER_LINEAR,
                               cv::BORDER_REPLICATE);
            }

            // 逐点比：模型说有墨，图上没有 = 缺失
            cv::Mat tBin;
            cv::compare(best, 0.20, tBin, cv::CMP_GT);
            cv::Mat miss;
            cv::bitwise_and(mBin, ~tBin, miss);
            // 只留成规模的缺失（断口），去掉边缘毛刺
            cv::Mat missOpen;
            cv::morphologyEx(miss, missOpen, cv::MORPH_OPEN,
                             cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
            const int totalMissing = cv::countNonZero(missOpen);

            // 数缺失块，最大块的长边当"最长断口"
            cv::Mat lb, st, ce;
            const int nc = cv::connectedComponentsWithStats(missOpen, lb, st, ce, 8, CV_32S);
            int breaks = 0, maxLen = 0;
            double depthSum = 0;
            int depthN = 0;
            for (int i = 1; i < nc; ++i) {
                const int a = st.at<int>(i, cv::CC_STAT_AREA);
                const int w = st.at<int>(i, cv::CC_STAT_WIDTH);
                const int h = st.at<int>(i, cv::CC_STAT_HEIGHT);
                if (a < 30) continue;   // 太小的算磨损，不算断口
                ++breaks;
                maxLen = std::max(maxLen, std::max(w, h));
                // 缺口深度：该块内模型墨强度（应亮）的平均
                double s = 0;
                int c = 0;
                for (int y = st.at<int>(i, cv::CC_STAT_TOP);
                     y < st.at<int>(i, cv::CC_STAT_TOP) + h; ++y)
                    for (int x = st.at<int>(i, cv::CC_STAT_LEFT);
                         x < st.at<int>(i, cv::CC_STAT_LEFT) + w; ++x) {
                        if (lb.at<int>(y, x) != i) continue;
                        s += modelLogoF.at<uchar>(y, x) / 255.0;
                        ++c;
                    }
                if (c) { depthSum += s / c; ++depthN; }
            }

            std::printf("%-42s %7d %8d %8d %9.3f\n", tag.c_str(), breaks, maxLen, totalMissing,
                        depthN ? depthSum / depthN : 0.0);

            cv::Mat vis;
            cv::cvtColor(best, vis, cv::COLOR_GRAY2BGR);
            cv::Mat mv;
            cv::cvtColor(missOpen, mv, cv::COLOR_GRAY2BGR);
            vis.setTo(cv::Scalar(0, 0, 255), missOpen);
            cv::imwrite(std::string("out/trace/detect_") + tag + ".png", vis);
        }
    }
    return 0;
}
