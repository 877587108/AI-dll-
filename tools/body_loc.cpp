// bodyloc.cpp —— 定位塑封体，给出裁剪参数
//
// 用户思路：用塑封体做锚点裁剪，有大小差异再拉伸。
//
// 为什么塑封体比 logo 可靠：塑封体是刚体，四边笔直、面积由封装规格决定，
// 不受刻印质量影响。logo 环的粗细和亮度本身就不均匀，拿它当锚点会漂。
//
// 做法：
//   1. 用 Otsu 把图分成"亮的结构"和"暗的塑封体面"
//   2. 塑封体面是画面中央最大的一块暗区（引脚是亮的、背景也可能亮）
//   3. 求这块暗区的最小外接矩形 = 塑封体
//   4. 输出外接矩形的宽高比、面积占比，供两个规格对比
//
// 判读：若两个规格的宽高比接近（差值 < 3%），说明只是裁剪不同、无需拉伸；
//       若差异大，就需要各向异性拉伸。
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

struct Body {
    bool found = false;
    cv::Rect box;
    double aspect = 0;    // 宽 / 高
    double areaFrac = 0;  // 塑封体面积 / 整图面积
    cv::Mat mask;
};

// 找塑封体：中央最大的一块"暗而均匀"的区域
static Body findBody(const cv::Mat& gray) {
    Body b;
    cv::Mat g;
    cv::GaussianBlur(gray, g, cv::Size(0, 0), 3.0);
    double mn, mx;
    cv::minMaxLoc(g, &mn, &mx);
    if (mx - mn < 10) return b;
    cv::Mat gn;
    g.convertTo(gn, CV_32F, 1.0 / (mx - mn), -mn / (mx - mn));
    cv::Mat gn8;
    gn.convertTo(gn8, CV_8U, 255.0);

    // 塑封体面是暗的：取 Otsu 的下侧
    cv::Mat bin;
    cv::threshold(gn8, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    // 去掉细碎
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15}));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {25, 25}));

    cv::Mat lb, st, ce;
    const int n = cv::connectedComponentsWithStats(bin, lb, st, ce, 8, CV_32S);
    int best = -1;
    double bs = -1;
    const cv::Point2f center(gray.cols / 2.0f, gray.rows / 2.0f);
    for (int i = 1; i < n; ++i) {
        const int a = st.at<int>(i, cv::CC_STAT_AREA);
        if (a < gray.total() * 0.05) continue;   // 塑封体至少占画面 5%
        const int w = st.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = st.at<int>(i, cv::CC_STAT_HEIGHT);
        // ---- 关键：塑封体面是"最均匀"的暗块
        //
        // 只按面积选会选到整张图（塑封体和外部暗背景连成一片）。
        // 塑封体是同一块材料，内部亮度方差小；背景里有引脚、阴影、反光，
        // 方差大得多。用"面积大 + 方差小"共同打分。
        cv::Mat shape = (lb == i);
        cv::Scalar mean, stddev;
        cv::meanStdDev(gn8, mean, stddev, shape);
        if (mean[0] > 128) continue;   // 塑封体是暗的
        const double uniformity = 1.0 / (1.0 + stddev[0] / 20.0);
        // 塑封体应近似矩形：填充率高
        const double fill = (double)a / (w * (double)h);
        const cv::Point2f c(ce.at<double>(i, 0), ce.at<double>(i, 1));
        const double d = std::hypot(c.x - center.x, c.y - center.y);
        const double sc = a * uniformity * fill * fill / (1.0 + d);
        if (sc > bs) { bs = sc; best = i; }
    }
    if (best < 0) return b;

    cv::Mat m = (lb == best);
    m.convertTo(m, CV_8U);
    // 填洞：塑封体面上的刻印是亮的，会把面打成筛子
    {
        cv::Mat filled = m.clone();
        cv::Mat ff;
        // flood fill 从 (0,0) 开始，反色后填洞
        cv::Mat inv;
        cv::bitwise_not(m, inv);
        cv::Mat lbl2;
        cv::Mat stats2, cent2;
        cv::connectedComponentsWithStats(inv, lbl2, stats2, cent2, 8, CV_32S);
        // 背景是最大的那块
        int bgLbl = 1;
        int bgArea = 0;
        for (int i = 1; i < stats2.rows; ++i)
            if (stats2.at<int>(i, cv::CC_STAT_AREA) > bgArea) {
                bgArea = stats2.at<int>(i, cv::CC_STAT_AREA);
                bgLbl = i;
            }
        cv::Mat holes = (lbl2 != bgLbl) & inv;   // 不在背景里的"非面"= 洞
        cv::bitwise_or(m, holes, filled);
        m = filled;
        (void)ff;
    }

    std::vector<cv::Point> pts;
    cv::findNonZero(m, pts);
    if (pts.empty()) return b;
    b.box = cv::boundingRect(pts);
    b.aspect = b.box.height > 0 ? (double)b.box.width / b.box.height : 0;
    b.areaFrac = (double)cv::countNonZero(m) / gray.total();
    b.mask = m;
    b.found = true;
    return b;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: markloc_bodyloc <imageDir...>\n");
        return 2;
    }
    for (int ai = 1; ai < argc; ++ai) {
        const fs::path dir(argv[ai]);
        if (!fs::is_directory(dir)) continue;
        std::printf("\n===== %s =====\n", dir.string().c_str());
        std::printf("%-42s %18s %8s %9s\n", "文件", "塑封体外框", "宽高比", "面积占比");
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
            Body b = findBody(gray);
            if (!b.found) {
                std::printf("%-42s  未找到塑封体\n", tag.c_str());
                continue;
            }
            char bb[64];
            std::snprintf(bb, sizeof(bb), "(%d,%d,%d,%d)", b.box.x, b.box.y, b.box.width,
                          b.box.height);
            std::printf("%-42s %18s %8.4f %9.4f\n", tag.c_str(), bb, b.aspect, b.areaFrac);
            if (files.size() <= 12) {
                cv::Mat vis;
                cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
                cv::rectangle(vis, b.box, cv::Scalar(0, 255, 0), 2);
                cv::imwrite(std::string("out/trace/body_") + tag + ".png", vis);
            }
        }
    }
    return 0;
}
