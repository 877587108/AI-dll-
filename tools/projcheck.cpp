// 垂直投影对齐可行性验证
//
// 问题：垂直投影能不能扩大对齐范围？
// 做法：拿真实图算垂直/水平投影剖面，看峰值结构是否足够尖锐。
//   · 行间距清晰 -> 投影有意义
//   · 只有一坨气泡 -> 投影没用
// 同时用投影法做平移估计，和相位相关的结果对比。
#include <cmath>
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

// 垂直投影：每列墨量求和
static std::vector<float> colProfile(const cv::Mat& sign) {
    std::vector<float> p(sign.cols, 0.f);
    for (int x = 0; x < sign.cols; ++x) {
        float s = 0;
        for (int y = 0; y < sign.rows; ++y) s += sign.at<float>(y, x);
        p[x] = s;
    }
    return p;
}

// 水平投影：每行墨量求和
static std::vector<float> rowProfile(const cv::Mat& sign) {
    std::vector<float> p(sign.rows, 0.f);
    for (int y = 0; y < sign.rows; ++y) {
        float s = 0;
        for (int x = 0; x < sign.cols; ++x) s += sign.at<float>(y, x);
        p[y] = s;
    }
    return p;
}

// 关键指标：投影剖面的"峰均比"。峰越尖，1D 匹配越可靠。
// 行结构清晰时，行间空白处投影接近 0，行中心处很高。
static void reportProfile(const char* name, const std::vector<float>& p) {
    double sum = 0, mx = 0;
    for (float v : p) {
        sum += v;
        mx = std::max<double>(mx, v);
    }
    const double mean = sum / p.size();
    double var = 0;
    for (float v : p) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / p.size());
    int nearZero = 0;
    for (float v : p)
        if (v < mean * 0.25) ++nearZero;
    std::printf("  %s: 均值=%.0f 峰值=%.0f 峰均比=%.1f 变异系数=%.2f 近零列占比=%.0f%%\n", name,
                mean, mx, mx / std::max(1.0, mean), sd / std::max(1.0, mean),
                100.0 * nearZero / p.size());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: projcheck <modelDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    for (int a = 2; a < argc; ++a) {
        cv::Mat gray;
        if (!readGray(argv[a], gray)) {
            std::printf("跳过（读取失败）: %s\n", argv[a]);
            continue;
        }
        const cv::Mat sig =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), model.config, false);
        std::printf("%s\n", argv[a]);
        reportProfile("垂直投影", colProfile(sig));
        reportProfile("水平投影", rowProfile(sig));
    }
    return 0;
}
