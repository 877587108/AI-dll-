// 双指标验证：多模板比对（含频率图对齐）
//
// 用法: markloc_dual <模型目录> <模板图...> -- <待测图...>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#include "chardet.hpp"
#include "common/console_utf8.hpp"

namespace fs = std::filesystem;

namespace {

std::string w2u(const std::wstring& w) {
    if (w.empty()) return {};
    int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string o(n, '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), o.data(), n, nullptr, nullptr);
    return o;
}

fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

bool rd(const std::string& p, cv::Mat& o) {
    std::ifstream f(u8p(p), std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0) return false;
    std::vector<uchar> b((size_t)n);
    f.read(reinterpret_cast<char*>(b.data()), n);
    if (!f) return false;
    o = cv::imdecode(b, cv::IMREAD_GRAYSCALE);
    return !o.empty();
}

// reg
cv::Mat loadSignal(const std::string& path, const chardet::Model& cm, const std::vector<chardet::CharUnit>& units) {
    cv::Mat g;
    if (!rd(path, g) || g.size() != cm.freq.size()) return cv::Mat();

    cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cm.config, false);
    if (cm.ref.empty()) return sig;

    cv::Mat r32, s32;
    cm.ref.convertTo(r32, CV_32F);
    sig.convertTo(s32, CV_32F);

    cv::Rect ia;
    for (const auto& u : units) ia = ia.area() > 0 ? (ia | u.bbox) : u.bbox;
    if (ia.area() <= 0) return sig;
    ia.x = std::max(0, ia.x - 20);
    ia.y = std::max(0, ia.y - 20);
    ia.width = std::min(sig.cols - ia.x, ia.width + 40);
    ia.height = std::min(sig.rows - ia.y, ia.height + 40);

    cv::Mat rC, sC;
    r32(ia).copyTo(rC);
    s32(ia).copyTo(sC);
    const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
    if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
        cv::warpAffine(sig, sig, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
    }
    return sig;
}

/// 从一张图提取所有字符的归一化 patch
std::vector<cv::Mat> extractPatches(const cv::Mat& sig, const chardet::Model& cm,
                                    const std::vector<chardet::CharUnit>& units) {
    std::vector<cv::Mat> out;
    out.reserve(units.size());
    for (const auto& u : units) {
        cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width <= 4 || b.height <= 4) {
            out.push_back(cv::Mat());
            continue;
        }
        cv::Mat patch;
        sig(b).convertTo(patch, CV_32F);
        cv::threshold(patch, patch, 0.25, 0.0, cv::THRESH_TOZERO);
        cv::Mat r;
        cv::resize(patch, r, cv::Size(cm.templateSize, cm.templateSize), 0, 0, cv::INTER_LINEAR);
        double mx = 0;
        cv::minMaxLoc(r, nullptr, &mx);
        if (mx > 1e-6) r *= (1.0 / mx);
        out.push_back(r);
    }
    return out;
}

}  // namespace

int main() {
    markloc::enableUtf8Console();

    int wc = 0;
    LPWSTR* wv = ::CommandLineToArgvW(::GetCommandLineW(), &wc);
    std::vector<std::string> a;
    if (wv) {
        for (int i = 0; i < wc; ++i) a.push_back(w2u(wv[i]));
        ::LocalFree(wv);
    }
    if (a.size() < 4) return 2;

    size_t sep = 0;
    for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "--") {
            sep = i;
            break;
        }
    }
    if (sep == 0) return 2;

    chardet::Model cm;
    if (!chardet::loadModel(a[1], cm)) return 1;

    std::vector<chardet::CharUnit> base = chardet::toAllUnits(cm);
    const int nChar = static_cast<int>(base.size());

    // 两种用法：
    //   普通：<模型> <模板图...> -- <待测图...>  → 用命令行给的图现建库
    //   读库：<模型> --frommodel -- <待测图...>  → 直接用模型目录里已有的模板库
    //         （与 GUI 的实际路径一致，避免"测的库和用的库不是同一个"）
    int nTemplate = 0;
    bool fromModel = false;
    for (size_t i = 2; i < sep; ++i) {
        if (a[i] == "--frommodel") fromModel = true;
    }
    if (fromModel) {
        nTemplate = static_cast<int>(cm.charTemplates.empty() ? 0 : cm.charTemplates[0].size());
        if (cm.charTemplates.empty()) {
            std::printf("模型目录内没有模板库(templates.png/txt)\n");
            return 1;
        }
        std::printf("模板 %d 张（读自模型目录），字符 %d 个\n\n", nTemplate, nChar);
    } else {
        cm.templateSize = 64;
        cm.charTemplates.assign(static_cast<size_t>(nChar), {});
        for (size_t i = 2; i < sep; ++i) {
            if (a[i] == "--frommodel") continue;
            const cv::Mat sig = loadSignal(a[i], cm, base);
            if (sig.empty()) continue;
            const auto patches = extractPatches(sig, cm, base);
            for (int c = 0; c < nChar && c < (int)patches.size(); ++c) {
                if (!patches[static_cast<size_t>(c)].empty()) {
                    cm.charTemplates[static_cast<size_t>(c)].push_back(
                        patches[static_cast<size_t>(c)]);
                }
            }
            ++nTemplate;
        }
        std::printf("模板 %d 张（命令行现建），字符 %d 个\n\n", nTemplate, nChar);
    }
    std::printf("%-32s %9s %9s %9s\n", "probe", "maxDiff", "secondDiff", "meanDiff");

    for (size_t i = sep + 1; i < a.size(); ++i) {
        const cv::Mat sig = loadSignal(a[i], cm, base);
        if (sig.empty()) continue;

        std::vector<chardet::CharUnit> units = base;
        const auto mr = chardet::diffMultiTemplate(sig, cm, units);
        if (!mr.valid) continue;

        double sum = 0;
        int cnt = 0;
        for (double d : mr.perChar) {
            if (d > 0) {
                sum += d;
                ++cnt;
            }
        }

        const fs::path fp = u8p(a[i]);
        const std::string nm(reinterpret_cast<const char*>(fp.filename().u8string().c_str()));
        std::printf("%-32s %9.4f %9.4f %9.4f\n", nm.substr(0, 32).c_str(), mr.maxDiff,
                    mr.secondDiff, cnt > 0 ? sum / cnt : 0.0);
    }
    return 0;
}
