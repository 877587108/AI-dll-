// imgdiff.cpp —— 两张图到底差多少（原始像素级）
//
// 前面工具对 re_ok/1_00 和 ngv2/1_00 输出了完全相同的四个数，
// 但两个文件的 MD5 不同。要么是工具把差异丢了，要么是两张图本来就一样。
//
// 本工具直接算原始像素差，不经过任何归一化、对齐、缩放。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_imgdiff <a> <b>\n");
        return 2;
    }
    cv::Mat a = readAny(argv[1]);
    cv::Mat b = readAny(argv[2]);
    if (a.empty() || b.empty()) {
        std::printf("读图失败\n");
        return 1;
    }
    std::printf("A %dx%d  B %dx%d\n", a.cols, a.rows, b.cols, b.rows);
    if (a.size() != b.size()) {
        std::printf("尺寸不同，无法逐像素比\n");
        return 0;
    }
    cv::Mat d;
    cv::absdiff(a, b, d);
    double mn, mx, mean;
    cv::minMaxLoc(d, &mn, &mx);
    mean = cv::mean(d)[0];
    int nz = cv::countNonZero(d);
    std::printf("像素差: 最大 %.0f  均值 %.3f  非零像素 %d / %d (%.2f%%)\n", mx, mean, nz,
                (int)d.total(), 100.0 * nz / d.total());

    // 差得最多的位置在哪一块
    if (nz > 0) {
        cv::Mat thr;
        cv::threshold(d, thr, 10, 255, cv::THRESH_BINARY);
        if (cv::countNonZero(thr) > 0) {
            const std::vector<cv::Point> pts = [&] {
                std::vector<cv::Point> v;
                cv::findNonZero(thr, v);
                return v;
            }();
            std::printf("差异>10 的像素 %zu 个，包围盒 ", pts.size());
            cv::Rect bb = cv::boundingRect(pts);
            std::printf("(%d,%d,%d,%d)\n", bb.x, bb.y, bb.width, bb.height);
        }
        cv::imwrite("out/trace/imgdiff.png", d * 4);
        std::printf("差异图(4x增强): out/trace/imgdiff.png\n");
    }
    return 0;
}
