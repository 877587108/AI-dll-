// 统计 strokeSignal 在"字符框内"的亮度分位数
//
// 目的：给自适应阈值 strokeHi = p90 找一个有实测依据的上/下界。
// 只统计 22 个字符框并集区域内的像素，因为管脚/治具/芯片边缘的反光
// 强度远超笔画，算全图会把分位数抬得没有意义。
//
// 用法: markloc_p90stat <modelDir> <listFile>
//   listFile 每行一个图片路径
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

namespace fs = std::filesystem;

// MSVC /std:c++20 删除了 fs::u8path，必须显式构造
static fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

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

// 在 mask 为真的像素上取分位
static double pctOn(const cv::Mat& s, const cv::Mat& mask, double q) {
    std::vector<float> v;
    v.reserve(static_cast<size_t>(cv::countNonZero(mask)));
    for (int y = 0; y < s.rows; ++y) {
        const float* sp = s.ptr<float>(y);
        const uchar* mp = mask.ptr<uchar>(y);
        for (int x = 0; x < s.cols; ++x) {
            if (mp[x]) v.push_back(sp[x]);
        }
    }
    if (v.empty()) return 0.0;
    const size_t k = static_cast<size_t>(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_p90stat <modelDir> <listFile>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    // 22 个字符框的并集掩膜
    cv::Mat boxMask = cv::Mat::zeros(model.freq.size(), CV_8U);
    for (const auto& u : base) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, boxMask.cols, boxMask.rows);
        if (b.width > 0 && b.height > 0) boxMask(b).setTo(255);
    }

    // 路径直接从 std::filesystem 取，不经过中间列表文件。
    // 曾用 out/oklist.txt 当输入，中文文件名在生成列表时被写成 "(??)"，
    // 导致 42 张 _FT_ 图全部"读取失败"，结论只基于一半样本。
    std::vector<std::string> files;
    {
        const std::filesystem::path arg = u8p(argv[2]);
        std::error_code ec;
        if (std::filesystem::is_directory(arg, ec)) {
            for (const auto& e :
                 std::filesystem::directory_iterator(arg, ec)) {
                if (!e.is_regular_file()) continue;
                const std::string ext = e.path().extension().string();
                if (ext == ".tif" || ext == ".tiff" || ext == ".png" || ext == ".bmp" ||
                    ext == ".jpg" || ext == ".jpeg") {
                    // 模型只对应 v00（_00）视角。其它视角的图与模型尺寸不符，
                    // 字符框落在别处，强行统计会把背景噪声当成笔画分位数。
                    const std::string stem = e.path().stem().string();
                    if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
                    files.push_back(e.path().string());
                }
            }
            std::sort(files.begin(), files.end());
        } else {
            FILE* fp = std::fopen(argv[2], "r");
            if (!fp) {
                std::printf("列表读取失败\n");
                return 1;
            }
            char line[4096];
            while (std::fgets(line, sizeof(line), fp)) {
                std::string s(line);
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
                if (!s.empty()) files.push_back(s);
            }
            std::fclose(fp);
        }
    }

    std::printf("%-46s %8s %8s %8s %8s\n", "image", "p50", "p90", "p95", "p99");
    std::printf("--- 明细（TAB 分隔，供脚本解析）---\n");
    std::vector<double> p90s;
    std::vector<double> ratios;
    for (const auto& f : files) {
        cv::Mat g;
        if (!readGray(f, g)) {
            std::printf("%-46s 读取失败\n", f.c_str());
            continue;
        }
        if (g.size() != model.freq.size()) {
            std::printf("%-46s 尺寸不符\n", f.c_str());
            continue;
        }
        const cv::Mat s = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
        if (s.type() != CV_32F) {
            std::printf("%-46s 信号类型非 CV_32F\n", f.c_str());
            continue;
        }
        const double p50 = pctOn(s, boxMask, 0.50);
        const double p90 = pctOn(s, boxMask, 0.90);
        const double p95 = pctOn(s, boxMask, 0.95);
        const double p99 = pctOn(s, boxMask, 0.99);
        std::string name = f;
        const size_t slash = name.find_last_of("/\\");
        if (slash != std::string::npos) name = name.substr(slash + 1);
        std::printf("PCT\t%s\t%.4f\t%.4f\t%.4f\t%.4f\n", name.c_str(), p50, p90, p95, p99);
        p90s.push_back(p90);
        if (p50 > 1e-6) ratios.push_back(p90 / p50);
    }

    if (!p90s.empty()) {
        std::sort(p90s.begin(), p90s.end());
        std::printf("\n统计 %zu 张\n", p90s.size());
        std::printf("p90 of p90: min=%.3f  p10=%.3f  p50=%.3f  p90=%.3f  max=%.3f\n",
                    p90s.front(), p90s[p90s.size() / 10], p90s[p90s.size() / 2],
                    p90s[p90s.size() * 9 / 10], p90s.back());
        int below35 = 0, above60 = 0;
        for (double v : p90s) {
            if (v < 0.35) ++below35;
            if (v > 0.60) ++above60;
        }
        std::printf("p90 < 0.35 的: %d 张  (会被下界卡住)\n", below35);
        std::printf("p90 > 0.60 的: %d 张  (会被上界卡住)\n", above60);
    }
    if (!ratios.empty()) {
        std::sort(ratios.begin(), ratios.end());
        std::printf("\np90/p50 比值: min=%.4f  p10=%.4f  p50=%.4f  p90=%.4f  max=%.4f\n",
                    ratios.front(), ratios[ratios.size() / 10], ratios[ratios.size() / 2],
                    ratios[ratios.size() * 9 / 10], ratios.back());
    }
    return 0;
}
