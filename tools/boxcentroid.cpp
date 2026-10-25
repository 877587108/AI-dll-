// 字符框偏移的可靠判定
//
// 前面按"字墨外接框溢出"判，结果 84 张里 132 个框被判溢出，
// 但抽查发现大部分是假的：
//   · #10 左右各只有 1~2 点墨，质心偏 3px —— 框明明很准，
//     是连通域把邻居的墨连进来了
//   · #17 是 21px 宽的 i，左右各有 140/245 点墨 —— 两侧都有墨，
//     那是相邻字符，不是本字溢出
//
// 真正可靠的读数是"墨迹质心相对框心的偏移"：
// 框偏了，质心必然偏；邻居对称带墨时质心不移。用它当主判据。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "chardet.hpp"

static bool readGray(const std::string& path, cv::Mat& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::vector<unsigned char> buf;
    unsigned char tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof(tmp), fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(fp);
    if (buf.empty()) return false;
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: boxcentroid <modelDir> <image...> [阈px=10]\n");
        return 2;
    }
    const double thr = (argc > 3) ? std::atof(argv[3]) : 10.0;
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);

    // 统计每个字符框跨图像的质心偏移
    std::vector<std::vector<double>> dxBy(units.size()), dyBy(units.size());
    for (int a = 2; a < argc; ++a) {
        cv::Mat gray;
        if (!readGray(argv[a], gray)) continue;
        const cv::Rect full(0, 0, gray.cols, gray.rows);
        const cv::Mat sig = chardet::strokeSignal(gray, full, model.config, false);
        for (size_t i = 0; i < units.size(); ++i) {
            const cv::Rect box = units[i].bbox & full;
            if (box.width <= 4 || box.height <= 4) continue;
            // 只在框内统计质心。框外的墨不参与——否则邻居的墨会把质心拉走，
            // 而这正是要避免的混淆。
            double sx = 0, sy = 0, sw = 0;
            for (int y = box.y; y < box.y + box.height; ++y)
                for (int x = box.x; x < box.x + box.width; ++x) {
                    const double v = sig.at<float>(y, x);
                    if (v < model.config.strokeHi) continue;
                    sx += x;
                    sy += y;
                    sw += 1;
                }
            if (sw < 50) continue;
            const double cx = sx / sw, cy = sy / sw;
            dxBy[i].push_back(cx - (box.x + box.width / 2.0));
            dyBy[i].push_back(cy - (box.y + box.height / 2.0));
        }
    }

    std::printf("%-4s %-16s %6s %8s %8s %8s\n", "#", "模型框", "样本", "平均dx", "平均dy",
                "最大偏移");
    int bad = 0;
    for (size_t i = 0; i < units.size(); ++i) {
        if (dxBy[i].empty()) continue;
        double ax = 0, ay = 0, mx = 0;
        for (size_t k = 0; k < dxBy[i].size(); ++k) {
            ax += dxBy[i][k];
            ay += dyBy[i][k];
            mx = std::max(mx, std::hypot(dxBy[i][k], dyBy[i][k]));
        }
        ax /= dxBy[i].size();
        ay /= dyBy[i].size();
        const bool off = std::hypot(ax, ay) > thr;
        if (off) ++bad;
        std::printf("%-4zu %-16s %6zu %+7.1f %+7.1f %7.1f%s\n", i,
                    (std::to_string(units[i].bbox.x) + "," + std::to_string(units[i].bbox.y) +
                     "," + std::to_string(units[i].bbox.width) + "," +
                     std::to_string(units[i].bbox.height)).c_str(),
                    dxBy[i].size(), ax, ay, mx, off ? "  <- 偏移" : "");
    }
    std::printf("\n平均偏移超过 %.0fpx 的框: %d 个\n", thr, bad);
    return 0;
}
