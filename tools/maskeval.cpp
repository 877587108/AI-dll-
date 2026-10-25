// 掩膜覆盖度实验
//
// 假设：0109 判 NG 是因为"模型笔画掩膜盖不住真实笔画"，
// 未被盖住的笔画像素进入 Hough，被连成"划痕"。
//
// 这里只测一个量：划痕搜索区里"信号亮"的像素有多少没被模型掩膜覆盖。
// 这个数越大，越多真实笔画会漏进 Hough。
// 不统计命中条数——那需要完整复刻 findScratches，容易复刻出偏差，
// 反而误导判断。
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

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
        std::printf("用法: maskeval <modelDir> <image> [maxImages]\n");
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
    const chardet::Config cfg = model.config;
    std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    markloc::ScratchParams P;
    const cv::Rect band = markloc::scratchBand(sig.size(), base, P.growPx);
    const cv::Mat sBand = sig(band);
    const cv::Mat fBand = model.freq(band);

    cv::Mat weak;
    cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
    const int sigCnt = cv::countNonZero(weak);
    std::printf("搜索区: (%d,%d,%d,%d)  信号亮像素=%d\n", band.x, band.y, band.width, band.height,
                sigCnt);
    std::printf("\n%-10s %-9s %s\n", "blankFreq", "dilateK", "未被模型覆盖的信号亮像素");
    for (double bf : {0.15, 0.10, 0.06, 0.03, 0.01}) {
        for (int dk : {9, 21, 41, 81}) {
            cv::Mat modelInk;
            cv::compare(fBand, static_cast<int>(bf * 255), modelInk, cv::CMP_GT);
            const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {dk, dk});
            cv::Mat modelDil, notModel, left;
            cv::dilate(modelInk, modelDil, kd);
            cv::bitwise_not(modelDil, notModel);
            cv::bitwise_and(weak, notModel, left);
            const int leftCnt = cv::countNonZero(left);
            std::printf("%-10.2f %-9d %5d / %d = %5.1f%%\n", bf, dk, leftCnt, sigCnt,
                        100.0 * leftCnt / std::max(1, sigCnt));
        }
    }
    (void)argv[3];
    return 0;
}
