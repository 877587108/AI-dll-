// ref 与 freq 谁更适合当配准参照？
//
// 背景：runDetect 的全局配准用 model_.ref 当参照（DefectViewWindow.cpp:1067），
// 我手写的复刻工具一直用 model_.freq，两者结果差很多——复刻出 0 条命中，
// 真实判定是 1 条 S=1938。差异必须查清，否则后面所有"改一处验一处"
// 都是在错误的基线上验证。
//
// 做法：对每张图分别用 ref / freq 做相位相关，拿到的位移各自套到信号上，
// 然后量"信号与参照之间的失配"作为配准好坏的客观分数：
//   cost = 笔画区(参照>阈)里 1 - s / 笔画区均值   （参照有墨处 信号 缺多少）
// 越小越好。同时对 logostab 那 4 个区域分别报，logo 区是已知最难的地方。
//
// 用法: markloc_refcmp <modelDir> <image|imageDir> [...]
#include <algorithm>
#include <cmath>
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

/// 配准后的失配代价：参照有笔画的地方，信号缺多少
static double mismatch(const cv::Mat& sig, const cv::Mat& ref32, const cv::Mat& mask,
                       const cv::Rect& roi) {
    const cv::Mat s = sig(roi), r = ref32(roi), m = mask(roi);
    double sumS = 0, sumR = 0;
    int n = 0;
    for (int y = 0; y < s.rows; ++y)
        for (int x = 0; x < s.cols; ++x) {
            if (!m.at<uchar>(y, x)) continue;
            sumS += s.at<float>(y, x);
            sumR += r.at<float>(y, x);
            ++n;
        }
    if (n == 0 || sumR <= 0) return 0.0;
    return 1.0 - (sumS / n) / (sumR / n);
}

/// 用给定参照求位移并把信号重采样过去
static cv::Mat alignWith(const cv::Mat& sig, const cv::Mat& ref32, const cv::Rect& area,
                         cv::Point2d* outShift) {
    cv::Mat sC, rC;
    sig(area).copyTo(sC);
    ref32(area).copyTo(rC);
    const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
    if (outShift) *outShift = sh;
    if (std::abs(sh.x) >= 60 || std::abs(sh.y) >= 60) return sig;
    if (std::abs(sh.x) <= 0.01 && std::abs(sh.y) <= 0.01) return sig;
    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
    cv::Mat aligned;
    cv::warpAffine(sig, aligned, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
    return aligned;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_refcmp <modelDir> <image|dir>...\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;

    // 收集图片
    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const fs::path p = u8p(argv[i]);
        if (fs::is_directory(p)) {
            for (const auto& e : fs::directory_iterator(p)) {
                if (!e.is_regular_file()) continue;
                const std::string stem = e.path().stem().string();
                if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
                files.push_back(e.path().string());
            }
        } else {
            files.push_back(argv[i]);
        }
    }
    std::sort(files.begin(), files.end());

    cv::Mat ref32, freq32;
    model.ref.convertTo(ref32, CV_32F);
    model.freq.convertTo(freq32, CV_32F);
    // 参照有笔画的地方 = 模型的墨
    cv::Mat refInk;
    cv::compare(model.ref, cfg.blankFreq * 255, refInk, cv::CMP_GT);

    // 字符并集，外扩 20px（与 runDetect 一致）
    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);
    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, model.freq.cols, model.freq.rows);
        if (b.width <= 0 || b.height <= 0) continue;
        inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(model.freq.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(model.freq.rows - inkArea.y, inkArea.height + 40);

    // 区域表：0..3 来自 regions.txt，正好是 logostab 用的那 4 块
    std::vector<cv::Rect> zones;
    for (const auto& rg : model.regions) {
        const cv::Rect r = rg.rect & cv::Rect(0, 0, model.freq.cols, model.freq.rows);
        if (r.area() > 0) zones.push_back(r);
    }
    if (zones.empty()) zones.push_back(inkArea);

    std::printf("区域数=%zu  搜索区=(%d,%d,%d,%d)\n", zones.size(), inkArea.x, inkArea.y,
                inkArea.width, inkArea.height);
    std::printf("%-22s %-14s %-14s %7s %7s %7s %7s %7s\n", "文件", "ref位移", "freq位移", "ref代价",
                "freq代价", "ref0", "ref1", "ref2");

    // 累计：谁当参照代价更低
    int winRef = 0, winFreq = 0, tie = 0;
    double sumRef = 0, sumFreq = 0;
    int n = 0;
    for (const auto& f : files) {
        cv::Mat gray;
        if (!readGray(f, gray) || gray.size() != model.freq.size()) {
            std::printf("%-22s 读取失败或尺寸不符\n", fs::path(f).stem().string().c_str());
            continue;
        }
        const cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg,
                                                  false);
        cv::Point2d shRef, shFreq;
        const cv::Mat aRef = alignWith(sig, ref32, inkArea, &shRef);
        const cv::Mat aFreq = alignWith(sig, freq32, inkArea, &shFreq);

        const double cRef = mismatch(aRef, ref32, refInk, inkArea);
        const double cFreq = mismatch(aFreq, ref32, refInk, inkArea);
        sumRef += cRef;
        sumFreq += cFreq;
        ++n;
        if (cRef < cFreq - 1e-4) ++winRef;
        else if (cFreq < cRef - 1e-4) ++winFreq;
        else ++tie;

        std::string bare = fs::path(f).stem().string();
        // 只显示前 18 个字符，避免超长文件名把表打乱
        if (bare.size() > 18) bare = bare.substr(0, 18);
        std::printf("%-22s (%5.2f,%5.2f) (%5.2f,%5.2f) %7.2f%% %7.2f%%", bare.c_str(), shRef.x,
                    shRef.y, shFreq.x, shFreq.y, cRef * 100, cFreq * 100);
        for (size_t z = 0; z < std::min<size_t>(3, zones.size()); ++z) {
            std::printf(" %6.1f%%", mismatch(aRef, ref32, refInk, zones[z]) * 100);
        }
        std::printf("\n");
    }
    std::printf("\n=== 汇总 (%d 张) ===\n", n);
    if (n > 0) {
        std::printf("ref  平均代价 %.2f%%\n", sumRef / n * 100);
        std::printf("freq 平均代价 %.2f%%\n", sumFreq / n * 100);
    }
    std::printf("ref 更低 %d 张 | freq 更低 %d 张 | 持平 %d 张\n", winRef, winFreq, tie);
    return 0;
}
