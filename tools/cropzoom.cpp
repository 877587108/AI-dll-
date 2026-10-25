// 裁剪放大：把命中区域切出来，看清线上到底有没有东西
// 用法: markloc/cropzoom <image> <x> <y> <w> <h> <scale> <out.png>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static bool readGray(const char* path, cv::Mat& out) {
    FILE* fp = std::fopen(path, "rb");
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
    if (argc < 8) {
        std::printf("用法: markloc_cropzoom <image> <x> <y> <w> <h> <scale> <out.png>\n");
        return 2;
    }
    cv::Mat g;
    if (!readGray(argv[1], g)) {
        std::printf("读取失败\n");
        return 1;
    }
    cv::Rect r(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
    r &= cv::Rect(0, 0, g.cols, g.rows);
    const int sc = std::max(1, std::atoi(argv[6]));
    cv::Mat crop = g(r).clone();
    cv::Mat big;
    cv::resize(crop, big, cv::Size(), sc, sc, cv::INTER_NEAREST);
    std::filesystem::path p(argv[7]);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    cv::imwrite(argv[7], big);
    std::printf("-> %s  (%dx%d)\n", argv[7], big.cols, big.rows);
    return 0;
}
