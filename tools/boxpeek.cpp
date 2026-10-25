// 指定字符框的墨迹剖析
//
// 框到底偏了多少，看三样东西最直接：
//   1. 框内、框左、框右、框上、框下各有多少墨
//   2. 墨迹沿 x 和 y 的分布（哪一侧是空的）
//   3. 墨迹质心相对框心的偏移
// 只报"溢出 19px"不够，要知道溢出的是哪个方向、那边有多少墨。
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
    if (argc < 4) {
        std::printf("用法: boxpeek <modelDir> <image> <index...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    cv::Mat gray;
    if (!readGray(argv[2], gray)) {
        std::printf("图片读取失败\n");
        return 1;
    }
    const cv::Rect full(0, 0, gray.cols, gray.rows);
    cv::Mat sig = chardet::strokeSignal(gray, full, model.config, false);
    const std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);

    for (int a = 3; a < argc; ++a) {
        const int idx = std::atoi(argv[a]);
        if (idx < 0 || idx >= static_cast<int>(units.size())) continue;
        const cv::Rect box = units[idx].bbox & full;

        // 以框为中心取一大块，量各方向的墨
        const int R = 60;
        const cv::Rect big = (cv::Rect(box.x - R, box.y - R, box.width + 2 * R,
                                       box.height + 2 * R) &
                              full);
        std::printf("\n=== 框 #%d  (%d,%d,%d,%d) ===\n", idx, box.x, box.y, box.width,
                    box.height);

        // 逐列、逐行的墨量（在 big 范围内）
        std::vector<double> col(big.width, 0.0), row(big.height, 0.0);
        for (int y = 0; y < big.height; ++y)
            for (int x = 0; x < big.width; ++x) {
                const double v = sig.at<float>(big.y + y, big.x + x);
                col[x] += v;
                row[y] += v;
            }

        // 框在 big 坐标系里的位置
        const int bx0 = box.x - big.x, by0 = box.y - big.y;
        const int bx1 = bx0 + box.width, by1 = by0 + box.height;

        // 每侧 15px 内的墨量占比
        auto sideInk = [&](int x0, int x1, int y0, int y1) {
            double s = 0;
            for (int y = std::max(0, y0); y < std::min(big.height, y1); ++y)
                for (int x = std::max(0, x0); x < std::min(big.width, x1); ++x)
                    s += sig.at<float>(big.y + y, big.x + x);
            return s;
        };
        const double inInk = sideInk(bx0, bx1, by0, by1);
        const double leftInk = sideInk(bx0 - 15, bx0, by0, by1);
        const double rightInk = sideInk(bx1, bx1 + 15, by0, by1);
        const double topInk = sideInk(bx0, bx1, by0 - 15, by0);
        const double botInk = sideInk(bx0, bx1, by1, by1 + 15);
        std::printf("墨量(相对框内 %.0f): 左 %.0f  右 %.0f  上 %.0f  下 %.0f\n", inInk, leftInk,
                    rightInk, topInk, botInk);

        // 墨迹质心
        double sx = 0, sy = 0, sw = 0;
        for (int y = 0; y < big.height; ++y)
            for (int x = 0; x < big.width; ++x) {
                const double v = sig.at<float>(big.y + y, big.x + x);
                if (v < model.config.strokeHi) continue;
                sx += x * v;
                sy += y * v;
                sw += v;
            }
        if (sw > 0) {
            const double cx = sx / sw, cy = sy / sw;
            std::printf("墨迹质心相对框心: (%+.1f, %+.1f)  [框心在 big(%.0f,%.0f)]\n",
                        cx - (bx0 + box.width / 2.0), cy - (by0 + box.height / 2.0),
                        bx0 + box.width / 2.0, by0 + box.height / 2.0);
        }

        // 打印列剖面，标出框边界
        std::printf("列剖面(每格=4px, |是框边): ");
        for (int x = 0; x < big.width; x += 4) {
            if (x == bx0 || (x < bx0 && x + 4 > bx0)) std::printf("|");
            else if (x == bx1 || (x < bx1 && x + 4 > bx1)) std::printf("|");
            const double v = col[x];
            std::printf("%c", v < 1 ? '.' : v < 20 ? '-' : v < 60 ? '+' : '#');
        }
        std::printf("\n");
        std::printf("行剖面(每格=4px, |是框边): ");
        for (int y = 0; y < big.height; y += 4) {
            if (y == by0 || (y < by0 && y + 4 > by0)) std::printf("|");
            else if (y == by1 || (y < by1 && y + 4 > by1)) std::printf("|");
            const double v = row[y];
            std::printf("%c", v < 1 ? '.' : v < 20 ? '-' : v < 60 ? '+' : '#');
        }
        std::printf("\n");
    }
    return 0;
}
