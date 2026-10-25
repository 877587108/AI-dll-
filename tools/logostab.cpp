// logo 区稳定性诊断
//
// 背景：region idx=3 的 char[0] bbox=(512,345,279,177) 是一个 279x177 的
// 巨型"字符框"，圈住的是 Silan 椭圆标记 + 下方圆形标记，不是字。
//
// 用户裁定：logo 区算检测区域（选项 B）。
// 但"算"不等于"现状能做好"。本工具回答一个问题：
//   84 张图里，这个 logo 区的内容本身稳不稳？
//
// 判据：
//   - 如果 84 张的 logo 区信号高度一致（逐像素方差小），那么平均模型是可靠的，
//     现在的环状漏出只是阈值/膨胀核的问题，可以修。
//   - 如果 84 张本身位置/粗细就有差异，那么平均模型必然糊，
//     掩膜永远盖不住，选项 B 在物理上不可达，必须换思路。
//
// 用法: markloc/logostab <modelDir> <imageDir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_logostab <modelDir> <imageDir> [regionIndex]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    const int ri = (argc >= 4) ? std::atoi(argv[3]) : 3;
    if (ri < 0 || ri >= static_cast<int>(model.regions.size())) {
        std::printf("region %d 不存在\n", ri);
        return 1;
    }
    const auto& reg = model.regions[static_cast<size_t>(ri)];
    const cv::Rect roi = reg.rect & cv::Rect(0, 0, model.freq.cols, model.freq.rows);
    std::printf("region %d: rect=(%d,%d,%d,%d) 字数=%zu\n", ri, reg.rect.x, reg.rect.y,
                reg.rect.width, reg.rect.height, reg.chars.size());
    for (const auto& c : reg.chars) {
        std::printf("  char[%d] bbox=(%d,%d,%d,%d)\n", c.index, c.bbox.x, c.bbox.y, c.bbox.width,
                    c.bbox.height);
    }

    std::vector<std::string> files;
    {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(u8p(argv[2]), ec)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext != ".tif" && ext != ".tiff" && ext != ".png" && ext != ".bmp") continue;
            const std::string stem = e.path().stem().string();
            if (stem.size() < 3 || stem.compare(stem.size() - 3, 3, "_00") != 0) continue;
            files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    }

    // 累加 84 张的 logo 区信号，逐像素统计均值/标准差
    cv::Mat sum = cv::Mat::zeros(roi.size(), CV_64F);
    cv::Mat sumSq = cv::Mat::zeros(roi.size(), CV_64F);
    int used = 0;
    for (const auto& f : files) {
        cv::Mat g;
        if (!readGray(f, g) || g.size() != model.freq.size()) continue;
        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
        if (sig.size() != model.freq.size()) continue;

        // 复刻推理侧全局配准，把每张图先对到基准
        cv::Mat r32, s32;
        model.freq.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        const auto units = chardet::toAllUnits(model);
        cv::Rect inkArea;
        for (const auto& u : units) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
        }
        if (inkArea.area() > 0) {
            inkArea.x = std::max(0, inkArea.x - 20);
            inkArea.y = std::max(0, inkArea.y - 20);
            inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
            inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            s32(inkArea).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::Mat al;
                cv::warpAffine(sig, al, M, sig.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                sig = al;
            }
        }

        const cv::Mat patch = sig(roi);
        cv::Mat p64, p64sq;
        patch.convertTo(p64, CV_64F);
        cv::multiply(p64, p64, p64sq);
        sum += p64;
        sumSq += p64sq;
        ++used;
    }
    if (used == 0) {
        std::printf("没有可用样本\n");
        return 1;
    }

    const cv::Mat mean = sum / used;
    cv::Mat var = (sumSq / used) - mean.mul(mean);
    cv::max(var, 0.0, var);
    cv::Mat stddev;
    cv::sqrt(var, stddev);

    // 只在"多数图认为有内容"的像素上统计标准差，背景噪声没意义
    cv::Mat content;
    cv::compare(mean, 0.10, content, cv::CMP_GT);  // 均值 > 0.10 视为有内容
    const int contentPx = cv::countNonZero(content);

    std::vector<double> sdOnContent;
    for (int y = 0; y < stddev.rows; ++y) {
        const double* s = stddev.ptr<double>(y);
        const uchar* c = content.ptr<uchar>(y);
        for (int x = 0; x < stddev.cols; ++x)
            if (c[x]) sdOnContent.push_back(s[x]);
    }
    if (sdOnContent.empty()) {
        std::printf("有内容的像素为 0\n");
        return 1;
    }
    std::sort(sdOnContent.begin(), sdOnContent.end());
    auto q = [&](double f) { return sdOnContent[static_cast<size_t>(f * (sdOnContent.size() - 1))]; };

    double meanOfContent = 0.0;
    for (int y = 0; y < mean.rows; ++y) {
        const double* m = mean.ptr<double>(y);
        const uchar* c = content.ptr<uchar>(y);
        for (int x = 0; x < mean.cols; ++x)
            if (c[x]) meanOfContent += m[x];
    }
    meanOfContent /= static_cast<double>(contentPx);

    std::printf("\n样本 %d 张（已配准）\n", used);
    std::printf("logo 区有内容像素: %d / %d (%.1f%%)\n", contentPx,
                roi.width * roi.height, 100.0 * contentPx / (roi.width * roi.height));
    std::printf("有内容像素的平均信号值: %.3f\n", meanOfContent);
    std::printf("\n逐像素标准差（只统计有内容处）:\n");
    std::printf("  min=%.4f  p50=%.4f  p90=%.4f  max=%.4f\n", sdOnContent.front(), q(0.50),
                q(0.90), sdOnContent.back());
    std::printf("  标准差 / 均值 = %.1f%%\n", 100.0 * q(0.50) / std::max(1e-6, meanOfContent));

    // 对比：字符区（region 0/1/2）同样口径，作为参照
    //
    // 注意：mean/stddev 只覆盖 roi 尺寸（roi 是 region ri 的矩形），
    // 不能拿全图坐标去索引它——之前这里直接用 rr.y 越界，直接崩。
    // 因此参照区必须用全图尺寸的累加缓冲。
    std::printf("\n=== 参照：各 region 的 std/mean（全图口径重算）===\n");
    {
        cv::Mat sumF = cv::Mat::zeros(model.freq.size(), CV_64F);
        cv::Mat sumSqF = cv::Mat::zeros(model.freq.size(), CV_64F);
        int usedF = 0;
        for (const auto& f : files) {
            cv::Mat g;
            if (!readGray(f, g) || g.size() != model.freq.size()) continue;
            cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
            if (sig.size() != model.freq.size()) continue;
            cv::Mat r32b, s32b;
            model.freq.convertTo(r32b, CV_32F);
            sig.convertTo(s32b, CV_32F);
            const auto unitsF = chardet::toAllUnits(model);
            cv::Rect ia;
            for (const auto& u : unitsF) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                ia = ia.area() > 0 ? (ia | b) : b;
            }
            if (ia.area() > 0) {
                ia.x = std::max(0, ia.x - 20);
                ia.y = std::max(0, ia.y - 20);
                ia.width = std::min(sig.cols - ia.x, ia.width + 40);
                ia.height = std::min(sig.rows - ia.y, ia.height + 40);
                cv::Mat rCb, sCb;
                r32b(ia).copyTo(rCb);
                s32b(ia).copyTo(sCb);
                const cv::Point2d shf = cv::phaseCorrelate(sCb, rCb, cv::noArray());
                if (std::abs(shf.x) < 60 && std::abs(shf.y) < 60 &&
                    (std::abs(shf.x) > 0.01 || std::abs(shf.y) > 0.01)) {
                    const cv::Mat Mf = (cv::Mat_<double>(2, 3) << 1, 0, -shf.x, 0, 1, -shf.y);
                    cv::Mat alf;
                    cv::warpAffine(sig, alf, Mf, sig.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE);
                    sig = alf;
                }
            }
            cv::Mat p64f, p64fsq;
            sig.convertTo(p64f, CV_64F);
            cv::multiply(p64f, p64f, p64fsq);
            sumF += p64f;
            sumSqF += p64fsq;
            ++usedF;
        }
        if (usedF > 0) {
            const cv::Mat meanF = sumF / usedF;
            cv::Mat varF = (sumSqF / usedF) - meanF.mul(meanF);
            cv::max(varF, 0.0, varF);
            cv::Mat sdF;
            cv::sqrt(varF, sdF);

            for (int r2 = 0; r2 < static_cast<int>(model.regions.size()); ++r2) {
                const cv::Rect rr = model.regions[static_cast<size_t>(r2)].rect &
                                    cv::Rect(0, 0, model.freq.cols, model.freq.rows);
                double m2 = 0.0, s2 = 0.0;
                int n2 = 0;
                for (int y = 0; y < rr.height; ++y) {
                    const double* mp = meanF.ptr<double>(rr.y + y);
                    const double* sp = sdF.ptr<double>(rr.y + y);
                    for (int x = 0; x < rr.width; ++x) {
                        const double mv = mp[rr.x + x];
                        if (mv > 0.10) {
                            m2 += mv;
                            s2 += sp[rr.x + x];
                            ++n2;
                        }
                    }
                }
                if (n2 > 0) {
                    std::printf("  region %d%s: 均值=%.3f 平均标准差=%.4f  std/mean=%.1f%%\n",
                                r2, r2 == ri ? "(logo)" : "", m2 / n2, s2 / n2,
                                100.0 * (s2 / n2) / (m2 / n2));
                }
            }
        }
    }
    return 0;
}
