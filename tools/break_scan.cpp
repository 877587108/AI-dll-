// break_scan.cpp —— 不依赖位置的断点识别
//
// 用户要求：不要对位置产生依赖，靠"造成笔画断连的断点"这个特征去找。
//
// 判据的本质：断口把一条笔画切成两段。
//   磨损 = 边缘少了墨，但笔画主线还是一整条
//   断口 = 沿笔画方向，出现"两侧有墨、中间空"的裂缝，主线被切开
//
// 关键是不预设断口在哪。做法用纯局部算子的组合：
//
//   1. 形态学闭运算（大核）把断口"补上"，得到【应该连续】的笔画形态 C
//   2. 断口 = C 里被补上的那部分，且那里原始图没有墨
//      —— 闭运算只在"两段挨得近的墨之间"填补，正好是断口的定义
//   3. 对每个补出来的块，量它的几何：长度、宽度、两端是否接着墨
//      —— 两端都接墨、宽度 >= 笔画宽的一半、长度 >= 最小的，才是断口
//
// 这个算子天然与位置无关：断口无论落在环的哪里，只要它切断了笔画就会被补出来。
// 磨损补不出来（边缘没有对侧可连）。
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

static bool findLogoBox(const cv::Mat& bin, cv::Rect* out) {
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
        if ((double)a / (w * (double)h) > 0.45) continue;
        const double sc = a * (1.0 - std::abs(ar - 1.7));
        if (sc > bestScore) { bestScore = sc; best = i; }
    }
    if (best < 0) return false;
    *out = cv::Rect(st.at<int>(best, cv::CC_STAT_LEFT), st.at<int>(best, cv::CC_STAT_TOP),
                    st.at<int>(best, cv::CC_STAT_WIDTH), st.at<int>(best, cv::CC_STAT_HEIGHT));
    return true;
}

struct BreakInfo {
    cv::Rect at;
    int len = 0;       // 断口长度（沿断开方向的跨度）
    int area = 0;
    bool twoSided = false;   // 两端都接着墨
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf(
            "用法: markloc_breakscan <imageDir...> [--minLen 4] [--maxLen 30] [--close 11]\n");
        return 2;
    }
    int minLen = 4, maxLen = 30, closeK = 11;
    std::vector<std::string> dirs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--minLen" && i + 1 < argc) minLen = std::atoi(argv[++i]);
        else if (a == "--maxLen" && i + 1 < argc) maxLen = std::atoi(argv[++i]);
        else if (a == "--close" && i + 1 < argc) closeK = std::atoi(argv[++i]);
        else dirs.push_back(a);
    }

    for (const auto& d : dirs) {
        const fs::path dir(d);
        if (!fs::is_directory(dir)) continue;
        std::printf("\n===== %s =====  (闭核 %d, 断口长 %d~%d px)\n", d.c_str(), closeK, minLen,
                    maxLen);
        std::printf("%-40s %7s %7s %8s %10s\n", "文件", "断点数", "最长", "总断口面积", "两端都接墨");

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
            cv::GaussianBlur(gray, g, cv::Size(0, 0), 2.0);
            double mn, mx;
            cv::minMaxLoc(g, &mn, &mx);
            if (mx - mn < 1) continue;
            cv::Mat gn;
            g.convertTo(gn, CV_32F, 1.0 / (mx - mn), -mn / (mx - mn));

            cv::Mat bin;
            {
                // Otsu 只接受 8U/16U，先把归一化结果转成 8U
                cv::Mat gn8;
                gn.convertTo(gn8, CV_8U, 255.0);
                cv::threshold(gn8, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            }
            cv::Rect logo;
            if (!findLogoBox(bin, &logo)) {
                std::printf("%-40s  未找到 logo\n", tag.c_str());
                continue;
            }

            cv::Mat roiB = bin(logo).clone();
            cv::Mat roiF = gn(logo).clone();

            // 只处理环带：椭圆边界 ±8% 范围内
            const int W = roiB.cols, H = roiB.rows;
            cv::Mat ringMask = cv::Mat::zeros(roiB.size(), CV_8U);
            cv::ellipse(ringMask, cv::Point(W / 2, H / 2),
                        cv::Size((int)(W * 0.46), (int)(H * 0.46)), 0, 0, 360, cv::Scalar(255),
                        (int)(std::min(W, H) * 0.16));
            cv::Mat ringB;
            cv::bitwise_and(roiB, ringMask, ringB);

            // 1. 闭运算补断口
            const cv::Mat kc = cv::getStructuringElement(cv::MORPH_ELLIPSE, {closeK, closeK});
            cv::Mat closed;
            cv::morphologyEx(ringB, closed, cv::MORPH_CLOSE, kc);

            // 2. 补出来的部分 = closed - ringB
            cv::Mat filled;
            cv::subtract(closed, ringB, filled);
            // 去掉环内外边缘本来就不是墨的区域：只保留"两侧本来就是墨"的填充
            cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
            cv::Mat ringDil;
            cv::dilate(ringB, ringDil, kd);
            cv::bitwise_and(filled, ringDil, filled);
            // 去掉细碎噪点
            cv::morphologyEx(filled, filled,
                             cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

            // 3. 逐块量几何
            cv::Mat lb, st, ce;
            const int nc = cv::connectedComponentsWithStats(filled, lb, st, ce, 8, CV_32S);
            std::vector<BreakInfo> breaks;
            for (int i = 1; i < nc; ++i) {
                const int a = st.at<int>(i, cv::CC_STAT_AREA);
                const int w = st.at<int>(i, cv::CC_STAT_WIDTH);
                const int h = st.at<int>(i, cv::CC_STAT_HEIGHT);
                if (a < 12) continue;
                const int lb_ = std::max(w, h);
                const int sb_ = std::min(w, h);
                if (lb_ < minLen || lb_ > maxLen) continue;
                BreakInfo b;
                b.at = cv::Rect(st.at<int>(i, cv::CC_STAT_LEFT),
                                st.at<int>(i, cv::CC_STAT_TOP), w, h);
                b.len = lb_;
                b.area = a;
                // 两端都接墨：块沿长轴方向的两端外侧是否有墨
                bool two = true;
                if (w >= h) {
                    for (int side : {-1, 1}) {
                        const int xx = side < 0 ? b.at.x - 2 : b.at.x + b.at.width + 1;
                        bool any = false;
                        for (int y = b.at.y; y < b.at.y + b.at.height && !any; ++y)
                            if (xx >= 0 && xx < W && y >= 0 && y < H && ringB.at<uchar>(y, xx))
                                any = true;
                        if (!any) two = false;
                    }
                } else {
                    for (int side : {-1, 1}) {
                        const int yy = side < 0 ? b.at.y - 2 : b.at.y + b.at.height + 1;
                        bool any = false;
                        for (int x = b.at.x; x < b.at.x + b.at.width && !any; ++x)
                            if (yy >= 0 && yy < H && x >= 0 && x < W && ringB.at<uchar>(yy, x))
                                any = true;
                        if (!any) two = false;
                    }
                }
                b.twoSided = two;
                breaks.push_back(b);
            }
            int maxL = 0, totArea = 0, twoCount = 0;
            for (const auto& b : breaks) {
                maxL = std::max(maxL, b.len);
                totArea += b.area;
                if (b.twoSided) ++twoCount;
            }
            std::printf("%-40s %7zu %7d %8d %10d\n", tag.c_str(), breaks.size(), maxL, totArea,
                        twoCount);
        }
    }
    return 0;
}
