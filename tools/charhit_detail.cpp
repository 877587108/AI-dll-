// charhit_detail.cpp —— 一条线段到底压过了哪些字符框
//
// 【要回答的问题】
// 0101 的真划痕（品红线）明显斜穿了好几个字，
// 但 charsHit 只算出 1 或 2。严重度 = depth × charsHit，
// 所以 charsHit 被低估会直接压掉严重度。
//
// 这个工具给定一条线段，把每个字符框与它的关系逐个打出来：
//   · 线段是否经过这个框（用真实采样，不用轴对齐 bbox）
//   · 真实经过的采样点数
//   · 轴对齐 bbox 是否相交（现在用的判据）
//   · 框宽、框内占比
// 两者不一致的地方，就是当前判据丢分的地方。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "chardet.hpp"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::printf("usage: markloc_charhitdetail <modelDir> x1 y1 x2 y2\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model) || !model.valid()) {
        std::printf("model load failed\n");
        return 1;
    }
    const cv::Point p1(std::atoi(argv[2]), std::atoi(argv[3]));
    const cv::Point p2(std::atoi(argv[4]), std::atoi(argv[5]));

    std::vector<chardet::CharUnit> units;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        if (model.regions[ri].isPattern()) continue;
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) units.push_back(std::move(x));
    }

    const double len = cv::norm(p2 - p1);
    const cv::Rect seg(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::abs(p2.x - p1.x) + 1,
                       std::abs(p2.y - p1.y) + 1);
    std::printf("线段 (%d,%d)-(%d,%d) 长%.0f  轴对齐bbox %dx%d @(%d,%d)\n", p1.x, p1.y, p2.x,
                p2.y, len, seg.width, seg.height, seg.x, seg.y);

    // 真实采样：线段上有哪些点落在框内
    const int nS = std::max(2, static_cast<int>(len * 4));
    std::printf("\n%-4s %-26s %6s %6s %6s %8s %8s\n", "idx", "bbox(x,y,w,h)", "真实", "bbox交",
                "框宽", "真实占比", "bbox占比");
    int nReal = 0, nBbox = 0;
    for (size_t ui = 0; ui < units.size(); ++ui) {
        const cv::Rect ub = units[ui].bbox;
        if (ub.width <= 0 || ub.height <= 0) continue;
        int in = 0;
        for (int t = 0; t <= nS; ++t) {
            const double rr = static_cast<double>(t) / nS;
            const int x = static_cast<int>(std::lround(p1.x + (p2.x - p1.x) * rr));
            const int y = static_cast<int>(std::lround(p1.y + (p2.y - p1.y) * rr));
            if (ub.contains(cv::Point(x, y))) ++in;
        }
        const bool bboxHit = (ub & seg).area() > 0;
        if (in > 0) ++nReal;
        if (bboxHit) ++nBbox;
        if (in == 0 && !bboxHit) continue;   // 无关的框不打印
        std::printf("%-4zu (%4d,%4d,%3d,%3d)       %6d %6s %6d %8.3f %8.3f\n", ui, ub.x, ub.y,
                    ub.width, ub.height, in, bboxHit ? "是" : "否", ub.width,
                    static_cast<double>(in) / (nS + 1),
                    static_cast<double>((ub & seg).area()) / (ub.area() + 1e-9));
    }
    std::printf("\n真实穿过 %d 个框；轴对齐 bbox 判据数出 %d 个框\n", nReal, nBbox);
    if (nReal != nBbox) {
        std::printf("【判据有出入】差 %d 个。严重度 = depth x charsHit，这个差会直接传导。\n",
                    nBbox - nReal);
    }
    return 0;
}
