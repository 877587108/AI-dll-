// logobreak.cpp —— 定位 logo 环上的断点
//
// 用户指出：断的是 logo（椭圆环），不是字符。
//
// 前面我看 SA6HY1F 的 H 就下了"断字"的结论，位置看错了一次。
// 这次直接量：把 logo 框内的墨迹做连通块分析，
// 一个完整的椭圆环是 1 个大连通块（加上内部的 SC 两个字母共 3 个）；
// 环断了就会多出块来，或者主块面积明显偏小。
//
// 同时对旧模型里的 logo 做同样分析作为基准，两边的块数/面积一对比就知道断没断。
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

struct BlockInfo {
    int n = 0;
    int largest = 0;
    double largestFrac = 0;
    std::vector<int> areas;
    cv::Mat labels;
    cv::Mat stats, cent;
};

static BlockInfo analyze(const cv::Mat& bin) {
    BlockInfo b;
    cv::Mat lbl, st, ct;
    const int n = cv::connectedComponentsWithStats(bin, lbl, st, ct, 8, CV_32S);
    b.n = n - 1;  // 去掉背景标签 0
    b.labels = lbl;
    b.stats = st;
    b.cent = ct;
    int total = 0;
    for (int i = 1; i < n; ++i) {
        const int a = st.at<int>(i, cv::CC_STAT_AREA);
        b.areas.push_back(a);
        total += a;
        if (a > b.largest) b.largest = a;
    }
    std::sort(b.areas.rbegin(), b.areas.rend());
    b.largestFrac = total > 0 ? (double)b.largest / total : 0.0;
    return b;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_logobreak <modelDir> <image> [--thr 0.35]\n");
        std::printf("      (需要在 tools/region_align 定出的 logo 框内分析)\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    double thr = 0.35;
    for (int i = 3; i + 1 < argc; i += 2)
        if (std::strcmp(argv[i], "--thr") == 0) thr = std::atof(argv[i + 1]);

    cv::Mat gray = readAny(argv[2]);
    if (gray.empty()) {
        std::printf("读图失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    const cv::Mat sig =
        chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    const auto& rg = model.regions[3];
    // 新图上 logo 的定位框（region_align 实测：(427,319,328,229)）
    //
    // 外扩 30%：断点可能就在框边上。框若不盖住断口，二值化后环看起来是连续的，
    // 就会把"没看到"误判成"没断"。这一步是排除性检查，不是调参。
    cv::Rect logoNew(427, 319, 328, 229);
    {
        const int ex = (int)(logoNew.width * 0.15), ey = (int)(logoNew.height * 0.15);
        logoNew.x -= ex;
        logoNew.y -= ey;
        logoNew.width += 2 * ex;
        logoNew.height += 2 * ey;
    }
    cv::Rect logoNewC = logoNew & cv::Rect(0, 0, sig.cols, sig.rows);

    // 基准：旧模型 ref 上的同一区域（同样外扩，保持可比）
    cv::Mat ref8;
    model.ref.convertTo(ref8, CV_8U);
    cv::Rect refR = rg.rect;
    {
        const int ex = (int)(refR.width * 0.15), ey = (int)(refR.height * 0.15);
        refR.x -= ex;
        refR.y -= ey;
        refR.width += 2 * ex;
        refR.height += 2 * ey;
    }
    refR &= cv::Rect(0, 0, ref8.cols, ref8.rows);
    cv::Mat refLogo = ref8(refR).clone();

    auto thrMat = [&](const cv::Mat& m, bool isRef) {
        cv::Mat b;
        if (isRef)
            cv::threshold(m, b, 60, 255, cv::THRESH_BINARY);   // ref 是 x80 的 8 位图
        else
            cv::compare(m, thr, b, cv::CMP_GT);
        return b;
    };

    // 新图 logo 用的是信号，先裁再阈值
    cv::Mat sigLogo = sig(logoNewC).clone();
    cv::Mat binNew = thrMat(sigLogo, false);
    cv::Mat binRef = thrMat(refLogo, true);
    // 两者尺寸可能不同，统一缩放后再比（只为看形态）
    if (binNew.size() != binRef.size()) {
        cv::resize(binNew, binNew, binRef.size(), 0, 0, cv::INTER_NEAREST);
    }

    // 去掉小噪点：形态学开运算，避免把噪声算成断块
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::morphologyEx(binRef, binRef, cv::MORPH_OPEN, k);
    cv::morphologyEx(binNew, binNew, cv::MORPH_OPEN, k);

    BlockInfo br = analyze(binRef);
    BlockInfo bn = analyze(binNew);

    auto show = [](const char* tag, const BlockInfo& b) {
        std::printf("%s: 连通块 %d 个，最大块 %d px（占 %.1f%%）\n", tag, b.n, b.largest,
                    b.largestFrac * 100);
        std::printf("     面积前 8: ");
        for (size_t i = 0; i < b.areas.size() && i < 8; ++i) std::printf("%d ", b.areas[i]);
        std::printf("\n");
    };
    std::printf("=== logo 区域连通性对比 ===\n");
    show("旧模型 ref ", br);
    show("新图      ", bn);

    // ---- 出对比图：参考 / 新图 / 连通块着色
    cv::Mat colRef, colNew;
    cv::Mat visRef, visNew;
    cv::cvtColor(binRef, visRef, cv::COLOR_GRAY2BGR);
    cv::cvtColor(binNew, visNew, cv::COLOR_GRAY2BGR);
    // 每个连通块给随机颜色，断点会显示成两块不同颜色
    auto colorize = [&](const BlockInfo& b) {
        cv::Mat o(b.labels.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < b.labels.rows; ++y)
            for (int x = 0; x < b.labels.cols; ++x) {
                const int id = b.labels.at<int>(y, x);
                if (id <= 0) continue;
                o.at<cv::Vec3b>(y, x) = cv::Vec3b(60 + (id * 97) % 195, 60 + (id * 53) % 195,
                                                  60 + (id * 151) % 195);
            }
        return o;
    };
    colRef = colorize(br);
    colNew = colorize(bn);

    auto lab = [](cv::Mat& m, const char* t) {
        cv::Mat o(m.rows + 24, m.cols, CV_8UC3, cv::Scalar(25, 25, 25));
        m.copyTo(o(cv::Rect(0, 24, m.cols, m.rows)));
        cv::putText(o, t, cv::Point(5, 17), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
        m = o;
    };
    char t1[96], t2[96];
    std::snprintf(t1, sizeof(t1), "OLD model logo: %d blocks", br.n);
    std::snprintf(t2, sizeof(t2), "NEW image logo: %d blocks", bn.n);
    lab(visRef, t1);
    lab(visNew, t2);
    lab(colRef, "OLD connected components");
    lab(colNew, "NEW connected components");
    cv::Mat out;
    cv::hconcat(std::vector<cv::Mat>{visRef, colRef, visNew, colNew}, out);
    cv::imwrite("out/trace/logo_break.png", out);
    std::printf("\n对比图: out/trace/logo_break.png (%dx%d)\n", out.cols, out.rows);
    return 0;
}
