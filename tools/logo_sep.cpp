// logosep.cpp —— logo 断连的分离度：NG（断字）vs OK
//
// 用户要求：造成断连的墨迹对识别影响更大，要给它更高的分数，
// 并据此把 logo 断连的 NG 和 OK 区分开。
//
// 关键约束：NG 是 1496x1496 新规格，OK 是 1688x1460 旧规格。
// 不能整图对齐比较，必须各自在自己的 logo 区域内、用自己的尺度量。
//
// 所以本工具不依赖整图配准，只做两件事：
//   1. 在每张图上找到 logo 椭圆（用形状搜索，不假设坐标）
//   2. 在 logo 区域内量"环的连通性"：完整环是 1 个大连通块，
//      断了就会分裂成多块，或者出现一段明显缺口
//
// 输出每张图的三个量，最后按 OK / NG 两组打印分布与分界点：
//   ringParts  主环分裂成的块数（越大越像断）
//   gapRatio   环上最长连续缺口占周长的比例
//   ringArea   环的总面积（排除"整块缺失"的情形）
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

struct RingStat {
    bool found = false;
    int ringParts = 0;     // 主环分裂块数
    double gapRatio = 0;   // 最长缺口 / 周长
    double ringArea = 0;   // 环面积（占椭圆框比例）
    double coverage = 0;   // 环上被判为有墨的角度比例
    cv::RotatedRect ellipse;
};

// 在给定图像里找最像"椭圆环"的结构，并量它的连通性
static RingStat measureRing(const cv::Mat& gray, const char* tag, bool saveVis) {
    RingStat rs;
    cv::Mat g;
    if (gray.channels() > 1) cv::cvtColor(gray, g, cv::COLOR_BGR2GRAY);
    else g = gray;
    // 归一化亮度，摆脱新老规格曝光差异
    cv::Mat gb;
    cv::GaussianBlur(g, gb, cv::Size(0, 0), 2.0);
    double mn, mx;
    cv::minMaxLoc(gb, &mn, &mx);
    if (mx - mn < 1) return rs;
    cv::Mat gn;
    gb.convertTo(gn, CV_8U, 255.0 / (mx - mn), -mn * 255.0 / (mx - mn));

    // 用 Otsu 找"亮的东西"（logo 是亮的刻印）
    cv::Mat bin;
    cv::threshold(gn, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 找所有连通块，挑"外接椭圆长宽比接近 2:1、面积适中"的那一个当 logo 环
    cv::Mat lb, st, ce;
    const int n = cv::connectedComponentsWithStats(bin, lb, st, ce, 8, CV_32S);
    int bestLbl = -1;
    double bestScore = -1;
    for (int i = 1; i < n; ++i) {
        const int a = st.at<int>(i, cv::CC_STAT_AREA);
        const int w = st.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = st.at<int>(i, cv::CC_STAT_HEIGHT);
        if (a < 2000 || w < 80 || h < 40) continue;
        const double ar = (double)w / h;
        if (ar < 1.2 || ar > 3.2) continue;   // logo 椭圆约 1.6~2.0
        const double fill = (double)a / (w * (double)h);
        // 环的填充率低（中间是空的），文字块的填充率高
        if (fill > 0.45) continue;
        const double sc = a * (1.0 - std::abs(ar - 1.7));
        if (sc > bestScore) {
            bestScore = sc;
            bestLbl = i;
        }
    }
    if (bestLbl < 0) {
        std::printf("  [%s] 未找到 logo 环候选\n", tag);
        return rs;
    }
    rs.found = true;
    const int x0 = st.at<int>(bestLbl, cv::CC_STAT_LEFT);
    const int y0 = st.at<int>(bestLbl, cv::CC_STAT_TOP);
    const int w0 = st.at<int>(bestLbl, cv::CC_STAT_WIDTH);
    const int h0 = st.at<int>(bestLbl, cv::CC_STAT_HEIGHT);
    const cv::Rect roi = cv::Rect(x0, y0, w0, h0) & cv::Rect(0, 0, bin.cols, bin.rows);

    // 找出这个环所在的整块区域（含内部字母），再单独割出"环带"
    cv::Mat mask = (lb == bestLbl);
    // 环带的几何：以内切椭圆为基准，取距离椭圆边界 ±3% 的带
    const cv::Point2f c(roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f);
    const cv::Size2f ax(roi.width / 2.0f, roi.height / 2.0f);
    rs.ellipse = cv::RotatedRect(c, ax, 0);

    // 极坐标扫描：沿椭圆一周采样，看每一角度上是否有墨
    const int N = 720;
    std::vector<int> on(N, 0);
    int onCount = 0;
    for (int t = 0; t < N; ++t) {
        const double th = 2.0 * CV_PI * t / N;
        bool hit = false;
        for (double rr = 0.93; rr <= 1.07 && !hit; rr += 0.01) {
            const int xx = (int)std::lround(c.x + ax.width * rr * std::cos(th));
            const int yy = (int)std::lround(c.y + ax.height * rr * std::sin(th));
            if (xx < 0 || yy < 0 || xx >= bin.cols || yy >= bin.rows) continue;
            if (bin.at<uchar>(yy, xx)) hit = true;
        }
        on[t] = hit ? 1 : 0;
        onCount += on[t];
    }
    rs.coverage = (double)onCount / N;

    // 最长连续缺口（环形，所以要绕一圈找）
    int maxGap = 0, cur = 0;
    for (int k = 0; k < 2 * N; ++k) {
        if (!on[k % N]) {
            ++cur;
            maxGap = std::max(maxGap, cur);
        } else {
            cur = 0;
        }
    }
    maxGap = std::min(maxGap, N);
    rs.gapRatio = (double)maxGap / N;

    // 环被切成几段：数"有墨段"的个数
    int segs = 0;
    for (int k = 0; k < N; ++k)
        if (on[k] && !on[(k + N - 1) % N]) ++segs;
    rs.ringParts = segs;
    rs.ringArea = (double)cv::countNonZero(mask) / (roi.width * (double)roi.height);

    if (saveVis) {
        cv::Mat vis;
        cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);
        cv::ellipse(vis, rs.ellipse, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(vis, roi, cv::Scalar(255, 160, 0), 1);
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s  parts=%d gap=%.3f cov=%.3f", tag, rs.ringParts,
                      rs.gapRatio, rs.coverage);
        cv::putText(vis, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::imwrite(std::string("out/trace/ring_") + tag + ".png", vis);
    }
    return rs;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: markloc_logosep <imageDir...>\n");
        return 2;
    }
    for (int ai = 1; ai < argc; ++ai) {
        const fs::path dir(argv[ai]);
        if (!fs::is_directory(dir)) continue;
        std::printf("\n===== %s =====\n", dir.string().c_str());
        std::printf("%-46s %6s %8s %8s %8s\n", "文件", "环段数", "最长缺口", "覆盖率", "环面积");
        std::vector<std::string> files;
        for (const auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            const std::string stem = e.path().stem().string();
            // 只取 _00 视角
            if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
            files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
        for (const auto& f : files) {
            cv::Mat g = readAny(f);
            if (g.empty()) continue;
            std::string tag = fs::path(f).filename().stem().string();
            const RingStat rs = measureRing(g, tag.c_str(), true);
            if (!rs.found) continue;
            std::printf("%-46s %6d %8.3f %8.3f %8.3f\n", tag.c_str(), rs.ringParts, rs.gapRatio,
                        rs.coverage, rs.ringArea);
        }
    }
    return 0;
}
