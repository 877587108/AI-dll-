// boxaudit.cpp —— 字符框体检：框是不是切错了
//
// 上一轮从对比图上看到的问题（用户指出的）：
//   "Silan" 5 个字母只切出 4 个框，l 和 a 并成一个框，i 被切成一条缝
//   "Q046R8NLP" 里 8 的框与 N 的框紧贴、N 与 L 重叠
//
// 这直接毁掉 charsHit：它数的是"线段扫过几个字符框"，
// 框重叠/并合时这个数就是虚的，composite = 深度 × 跨越字数 被成倍放大。
//
// 本工具不改任何东西，只回答三个问题：
//   1. 22 个框的宽度分布：有没有明显的一小撮（被切碎的）和一大撮（并合的）
//   2. 每个框里实际有多少字形墨：墨量过少的框说明框错了位置
//   3. 框之间有重叠吗：重叠框会让 charsHit 虚高
#include <algorithm>
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: markloc_boxaudit <modelDir>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型加载失败: %s\n", argv[1]);
        return 1;
    }
    const int inkThr = static_cast<int>(model.config.blankFreq * 255.0);

    std::printf("区域 %zu 个\n", model.regions.size());
    int totalChars = 0;
    std::vector<int> widths;

    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        const auto& rg = model.regions[ri];
        std::printf("\n===== 区域 %zu  外框(%d,%d,%d,%d)  声明字符 %zu =====\n", ri, rg.rect.x,
                    rg.rect.y, rg.rect.width, rg.rect.height, rg.chars.size());
        std::printf("%4s %-24s %6s %8s %8s %7s\n", "序", "框 x,y,w,h", "宽", "框内墨量", "占框比例",
                    "长宽比");
        for (size_t ci = 0; ci < rg.chars.size(); ++ci) {
            const cv::Rect b = rg.chars[ci].bbox;
            const cv::Rect c = b & cv::Rect(0, 0, model.freq.cols, model.freq.rows);
            int ink = 0;
            if (c.area() > 0)
                ink = cv::countNonZero(model.freq(c) > inkThr);
            const double ratio = (c.area() > 0) ? static_cast<double>(ink) / c.area() : 0.0;
            const double ar = b.height > 0 ? static_cast<double>(b.width) / b.height : 0.0;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%d,%d,%d,%d", b.x, b.y, b.width, b.height);
            std::printf("%4zu %-24s %6d %8d %8.3f %7.2f%s\n", ci, buf, b.width, ink, ratio, ar,
                        (ratio < 0.05 ? "   <- 框内几乎无墨" : ""));
            widths.push_back(b.width);
            ++totalChars;
        }
    }
    std::printf("\n共 %d 个字符框\n", totalChars);

    // 宽度分布：排序后看有没有自然的断点
    std::sort(widths.begin(), widths.end());
    std::printf("宽度分布(升序):");
    for (int w : widths) std::printf(" %d", w);
    std::printf("\n");
    if (!widths.empty()) {
        const double med = widths[widths.size() / 2];
        std::printf("宽度中位 %.1f，最小 %d，最大 %d\n", med, widths.front(), widths.back());
        std::printf("窄于中位 60%%（疑似被切碎）的框：");
        for (int w : widths)
            if (w < med * 0.6) std::printf(" %d", w);
        std::printf("\n");
        std::printf("宽于中位 160%%（疑似并合）的框：");
        for (int w : widths)
            if (w > med * 1.6) std::printf(" %d", w);
        std::printf("\n");
    }

    // 框间重叠：同一区域内两两求交
    std::printf("\n===== 框间重叠（同区域内）=====\n");
    int overlaps = 0;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        const auto& chars = model.regions[ri].chars;
        for (size_t i = 0; i < chars.size(); ++i)
            for (size_t j = i + 1; j < chars.size(); ++j) {
                const cv::Rect inter = chars[i].bbox & chars[j].bbox;
                if (inter.area() > 0) {
                    std::printf("  区域%zu  #%zu 与 #%zu 重叠 %dx%d\n", ri, i, j, inter.width,
                                inter.height);
                    ++overlaps;
                }
            }
    }
    if (overlaps == 0) std::printf("  无重叠\n");
    return 0;
}
