// 信号 vs 模型 的逐像素对账
//
// 怀疑：0109_RT 判 NG 是因为"信号说这里有墨、模型说这里是空白"，
// 两者矛盾的地方被当成划痕。
// 做法：把命中框位置的 信号图 和 模型频率图 并排放大存盘，
// 一眼就能看出是模型漏了笔画，还是信号上真有额外亮痕。
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

static bool writeImage(const std::string& path, const cv::Mat& img) {
    std::vector<unsigned char> buf;
    if (!cv::imencode(".png", img, buf)) return false;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::fwrite(buf.data(), 1, buf.size(), fp);
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::printf("用法: freqvs <modelDir> <image> <x> <y> <w> <h>\n");
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
    cv::Rect r(std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]));
    r.x -= 30;
    r.y -= 30;
    r.width += 60;
    r.height += 60;
    r &= cv::Rect(0, 0, gray.cols, gray.rows);

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // 三联：原图 | 信号(归一到0-255) | 模型频率
    cv::Mat a = gray(r).clone();
    cv::Mat b;
    cv::normalize(sig(r), b, 0, 255, cv::NORM_MINMAX);
    b.convertTo(b, CV_8U);
    cv::Mat c = model.freq(r).clone();

    cv::cvtColor(a, a, cv::COLOR_GRAY2BGR);
    cv::cvtColor(b, b, cv::COLOR_GRAY2BGR);
    cv::cvtColor(c, c, cv::COLOR_GRAY2BGR);
    // 标注
    cv::putText(a, "RAW", cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(b, "SIGNAL", cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(c, "MODEL FREQ", cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);

    cv::Mat out;
    cv::hconcat(std::vector<cv::Mat>{a, b, c}, out);
    cv::resize(out, out, cv::Size(), 2.5, 2.5, cv::INTER_NEAREST);
    writeImage("out/trace/freqvs.png", out);

    // 统计：信号明显亮、但模型说空白 的像素占比
    const cv::Mat s = sig(r), f = model.freq(r);
    int both = 0, sigOnly = 0, sigBright = 0;
    for (int y = 0; y < s.rows; ++y) {
        for (int x = 0; x < s.cols; ++x) {
            const bool isSig = s.at<float>(y, x) > cfg.missStrokeLo * 0.7;
            const bool isInk = f.at<uchar>(y, x) > static_cast<int>(cfg.blankFreq * 255);
            if (isSig) ++sigBright;
            if (isSig && isInk) ++both;
            if (isSig && !isInk) ++sigOnly;
        }
    }
    std::printf("信号亮像素=%d  模型也认=%d  只有信号认=%d (%.1f%%)\n", sigBright, both, sigOnly,
                sigBright > 0 ? 100.0 * sigOnly / sigBright : 0.0);
    std::printf("图 -> out/trace/freqvs.png\n");
    return 0;
}
