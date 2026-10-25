// oksep.cpp —— NG vs OK 一次性对比，多量并排看哪个能分开
//
// 前提（用户确认）：re_ok/1_00、re_ok/2_00 是合格基准。
// 用户同时确认 ngv2 里的 7 颗都是 logo 断连。
//
// 方法：每张图都经过完全相同的流水线，不做任何按图的参数调整：
//   归一化 -> Otsu -> 找 logo 环 -> 裁 logo 区域 -> 缩放到统一尺寸
//   -> 在统一坐标系下量多个候选量
//
// 候选量（都不依赖具体位置）：
//   ringParts   环在 Otsu 下分裂成的块数
//   maxGapA     环上最长连续暗段占周长比例
//   missRatio   "闭运算能补上、但原始图没墨"的面积 / 环面积（断口的直接度量）
//   twoSided   上面那些补丁里，两端都接墨的个数
//   edgeRough   环带面积 / 周长（越粗糙越大，用于排除"粗糙度"这个混淆因素）
//
// 判读：NG 全部同侧、OK 全部另一侧，才算分开。有任何交叉就是没分开。
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

struct Metric {
    int ringParts = 0;
    double maxGapA = 0;
    double missRatio = 0;
    int twoSided = 0;
    double edgeRough = 0;
    bool found = false;
};

static Metric analyze(const cv::Mat& gray, int closeK, bool saveVis, const char* tag) {
    Metric m;
    cv::Mat g;
    cv::GaussianBlur(gray, g, cv::Size(0, 0), 2.0);
    double mn, mx;
    cv::minMaxLoc(g, &mn, &mx);
    if (mx - mn < 1) return m;
    cv::Mat gn;
    g.convertTo(gn, CV_32F, 1.0 / (mx - mn), -mn / (mx - mn));
    cv::Mat gn8;
    gn.convertTo(gn8, CV_8U, 255.0);
    cv::Mat bin;
    cv::threshold(gn8, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 找 logo 环
    cv::Mat lb, st, ce;
    const int n = cv::connectedComponentsWithStats(bin, lb, st, ce, 8, CV_32S);
    int best = -1;
    double bs = -1;
    for (int i = 1; i < n; ++i) {
        const int a = st.at<int>(i, cv::CC_STAT_AREA);
        const int w = st.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = st.at<int>(i, cv::CC_STAT_HEIGHT);
        if (a < 2000 || w < 80 || h < 40) continue;
        const double ar = (double)w / h;
        if (ar < 1.2 || ar > 3.2) continue;
        if ((double)a / (w * (double)h) > 0.45) continue;
        const double sc = a * (1.0 - std::abs(ar - 1.7));
        if (sc > bs) { bs = sc; best = i; }
    }
    if (best < 0) return m;

    // 统一坐标系：用找到的环的外接框，缩放到固定尺寸
    cv::Rect bb(st.at<int>(best, cv::CC_STAT_LEFT), st.at<int>(best, cv::CC_STAT_TOP),
                st.at<int>(best, cv::CC_STAT_WIDTH), st.at<int>(best, cv::CC_STAT_HEIGHT));
    bb &= cv::Rect(0, 0, bin.cols, bin.rows);
    if (bb.width < 60 || bb.height < 30) return m;
    const int TW = 300, TH = 200;
    cv::Mat roiB, roiF;
    cv::resize(bin(bb), roiB, cv::Size(TW, TH), 0, 0, cv::INTER_NEAREST);
    cv::resize(gn(bb), roiF, cv::Size(TW, TH), 0, 0, cv::INTER_LINEAR);

    // 环带掩膜：椭圆边界 ±8%
    cv::Mat ringMask = cv::Mat::zeros(roiB.size(), CV_8U);
    cv::ellipse(ringMask, cv::Point(TW / 2, TH / 2), cv::Size((int)(TW * 0.455), (int)(TH * 0.455)),
                0, 0, 360, cv::Scalar(255), (int)(TH * 0.15));
    cv::Mat ringB;
    cv::bitwise_and(roiB, ringMask, ringB);
    m.found = true;

    // 环块数
    cv::Mat lb2, st2, ce2;
    int nc2 = cv::connectedComponentsWithStats(ringB, lb2, st2, ce2, 8, CV_32S);
    for (int i = 1; i < nc2; ++i)
        if (st2.at<int>(i, cv::CC_STAT_AREA) >= 80) ++m.ringParts;

    // 极角扫描：最长连续暗段
    const int N = 720;
    std::vector<int> on(N, 0);
    int onN = 0;
    for (int t = 0; t < N; ++t) {
        const double th = 2 * CV_PI * t / N;
        bool hit = false;
        for (double rr = 0.90; rr <= 1.10 && !hit; rr += 0.01) {
            const int x = (int)std::lround(TW / 2.0 + TW * 0.455 * rr * std::cos(th));
            const int y = (int)std::lround(TH / 2.0 + TH * 0.455 * rr * std::sin(th));
            if (x < 0 || y < 0 || x >= TW || y >= TH) continue;
            if (ringB.at<uchar>(y, x)) hit = true;
        }
        on[t] = hit;
        onN += on[t];
    }
    int maxGap = 0, cur = 0;
    for (int k = 0; k < 2 * N; ++k) {
        if (!on[k % N]) { ++cur; maxGap = std::max(maxGap, cur); }
        else cur = 0;
    }
    m.maxGapA = (double)std::min(maxGap, N) / N;

    // 闭运算补出来的面积占环面积比（断口直接度量）
    const cv::Mat kc = cv::getStructuringElement(cv::MORPH_ELLIPSE, {closeK, closeK});
    cv::Mat closed;
    cv::morphologyEx(ringB, closed, cv::MORPH_CLOSE, kc);
    cv::Mat filled;
    cv::subtract(closed, ringB, filled);
    cv::Mat ringDil;
    cv::dilate(ringB, ringDil, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    cv::bitwise_and(filled, ringDil, filled);
    cv::morphologyEx(filled, filled, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    const int ringArea = cv::countNonZero(ringB);
    m.missRatio = ringArea > 0 ? (double)cv::countNonZero(filled) / ringArea : 0;

    // 两端都接墨的补丁数
    cv::Mat lb3, st3, ce3;
    const int nc3 = cv::connectedComponentsWithStats(filled, lb3, st3, ce3, 8, CV_32S);
    for (int i = 1; i < nc3; ++i) {
        const int a = st3.at<int>(i, cv::CC_STAT_AREA);
        const int w = st3.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = st3.at<int>(i, cv::CC_STAT_HEIGHT);
        if (a < 20) continue;
        if (std::max(w, h) < 4 || std::max(w, h) > 34) continue;
        bool two = true;
        if (w >= h) {
            for (int s : {-1, 1}) {
                const int xx = s < 0 ? st3.at<int>(i, cv::CC_STAT_LEFT) - 2
                                     : st3.at<int>(i, cv::CC_STAT_LEFT) + w + 1;
                bool any = false;
                for (int y = st3.at<int>(i, cv::CC_STAT_TOP);
                     y < st3.at<int>(i, cv::CC_STAT_TOP) + h && !any; ++y)
                    if (xx >= 0 && xx < TW && y >= 0 && y < TH && ringB.at<uchar>(y, xx)) any = true;
                if (!any) two = false;
            }
        } else {
            for (int s : {-1, 1}) {
                const int yy = s < 0 ? st3.at<int>(i, cv::CC_STAT_TOP) - 2
                                     : st3.at<int>(i, cv::CC_STAT_TOP) + h + 1;
                bool any = false;
                for (int x = st3.at<int>(i, cv::CC_STAT_LEFT);
                     x < st3.at<int>(i, cv::CC_STAT_LEFT) + w && !any; ++x)
                    if (yy >= 0 && yy < TH && x >= 0 && x < TW && ringB.at<uchar>(yy, x)) any = true;
                if (!any) two = false;
            }
        }
        if (two) ++m.twoSided;
    }

    // 粗糙度：环面积 / 极角扫描中"有墨"的采样数（每采样点平均厚度）
    m.edgeRough = onN > 0 ? (double)ringArea / onN : 0;

    if (saveVis) {
        cv::Mat vis;
        cv::cvtColor(ringB, vis, cv::COLOR_GRAY2BGR);
        vis.setTo(cv::Scalar(0, 0, 255), filled);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s parts=%d gap=%.3f miss=%.3f two=%d rough=%.1f", tag,
                      m.ringParts, m.maxGapA, m.missRatio, m.twoSided, m.edgeRough);
        cv::Mat o(vis.rows + 24, vis.cols, CV_8UC3, cv::Scalar(20, 20, 20));
        vis.copyTo(o(cv::Rect(0, 24, vis.cols, vis.rows)));
        cv::putText(o, buf, cv::Point(4, 17), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::imwrite(std::string("out/trace/sep_") + tag + ".png", o);
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_oksep <ngDir> <okDir> [--close 11]\n");
        return 2;
    }
    int closeK = 11;
    for (int i = 3; i + 1 < argc; i += 2)
        if (std::strcmp(argv[i], "--close") == 0) closeK = std::atoi(argv[i + 1]);

    auto runDir = [&](const std::string& d, const char* label) {
        std::vector<std::pair<std::string, Metric>> rows;
        std::vector<std::string> files;
        for (const auto& e : fs::directory_iterator(fs::path(d))) {
            if (!e.is_regular_file()) continue;
            const std::string stem = e.path().stem().string();
            if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
            files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
        for (const auto& f : files) {
            cv::Mat g = readAny(f);
            if (g.empty()) continue;
            std::string tag = fs::path(f).filename().stem().string();
            Metric m = analyze(g, closeK, true, (std::string(label) + "_" + tag).c_str());
            if (m.found) rows.push_back({tag, m});
        }
        std::printf("\n===== %s =====\n", label);
        std::printf("%-14s %8s %8s %9s %8s %8s\n", "文件", "环块数", "最长缺口", "补平面积比",
                    "两端接墨", "环粗糙度");
        for (const auto& r : rows)
            std::printf("%-14s %8d %8.3f %9.4f %8d %8.1f\n", r.first.c_str(), r.second.ringParts,
                        r.second.maxGapA, r.second.missRatio, r.second.twoSided,
                        r.second.edgeRough);
        return rows;
    };
    auto ng = runDir(argv[1], "NG");
    auto ok = runDir(argv[2], "OK");
    (void)ng;
    (void)ok;
    return 0;
}
