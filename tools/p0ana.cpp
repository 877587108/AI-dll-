// P0 分析：找出能区分「缺笔画 / 改笔画」的特征
//
// 对每张图统计每个字符的多维特征，取最异常的那个，与 OK 样本对比。

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

void registerImage(cv::Mat& sig, const chardet::Model& cm,
                   const std::vector<chardet::CharUnit>& all) {
    if (cm.ref.empty()) return;
    cv::Mat r32, s32;
    cm.ref.convertTo(r32, CV_32F);
    sig.convertTo(s32, CV_32F);

    cv::Rect ia;
    for (const auto& u : all) {
        ia = ia.area() > 0 ? (ia | u.bbox) : u.bbox;
    }
    if (ia.area() <= 0) return;
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
}

void alignPerChar(cv::Mat& sig, const chardet::Model& cm,
                  std::vector<chardet::CharUnit>& all) {
    for (auto& u : all) {
        cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (box.width <= 4 || box.height <= 4) continue;

        constexpr int kS = 12;
        cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
        sr &= cv::Rect(0, 0, sig.cols, sig.rows);
        if (sr.width < box.width || sr.height < box.height) continue;

        cv::Mat mS, tS;
        cm.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
        sig(sr).convertTo(tS, CV_32F);

        cv::Mat res;
        cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)), res,
                          cv::TM_CCOEFF_NORMED);
        cv::Point mL;
        double mv = 0;
        cv::minMaxLoc(res, nullptr, &mv, nullptr, &mL);
        if (mv < 0.30) continue;

        const cv::Rect nb(box.x + mL.x - kS, box.y + mL.y - kS, box.width, box.height);
        if ((nb & cv::Rect(0, 0, sig.cols, sig.rows)) != nb) continue;
        u.bbox = nb;
    }
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
    if (a.size() < 3) return 2;

    chardet::Model cm;
    if (!chardet::loadModel(a[1], cm)) return 1;

    std::vector<chardet::CharUnit> baseUnits;
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        auto u = chardet::toUnits(cm, static_cast<int>(ri));
        for (auto& x : u) baseUnits.push_back(x);
    }

    std::printf("%-32s %9s %9s %9s %9s\n", "sample", "maxMiss", "maxLocal", "maxAreaDev",
                "meanRatio");

    for (size_t fi = 2; fi < a.size(); ++fi) {
        cv::Mat g;
        if (!rd(a[fi], g) || g.size() != cm.freq.size()) continue;

        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cm.config, false);
        std::vector<chardet::CharUnit> units = baseUnits;
        registerImage(sig, cm, units);
        alignPerChar(sig, cm, units);

        double maxMiss = 0, maxExtra = 0, maxAreaDev = 0, sumRatio = 0;
        int nValid = 0;

        for (const auto& u : units) {
            const auto nc = chardet::diffNcc(sig, cm.freq, u, cm.config);
            if (!nc.valid) continue;
            maxMiss = std::max(maxMiss, nc.missRatio);
            maxExtra = std::max(maxExtra, nc.localExtra);
            sumRatio += nc.missRatio + nc.extraRatio;
            ++nValid;

            cv::Mat mf, tf;
            cm.freq(u.bbox).convertTo(mf, CV_32F, 1.0 / 255.0);
            sig(u.bbox).convertTo(tf, CV_32F);
            cv::Mat mb, tb;
            cv::compare(mf, cm.config.missFreq, mb, cv::CMP_GT);
            cv::compare(tf, cm.config.strokeHi, tb, cv::CMP_GT);
            const int mi = cv::countNonZero(mb);
            const int ti = cv::countNonZero(tb);
            if (mi > 100) {
                maxAreaDev = std::max(maxAreaDev, std::abs(ti - mi) / static_cast<double>(mi));
            }
        }

        const fs::path fp = u8p(a[fi]);
        const std::string nm(reinterpret_cast<const char*>(fp.filename().u8string().c_str()));
        std::printf("%-32s %9.3f %9.3f %9.3f %9.3f\n", nm.substr(0, 32).c_str(), maxMiss, maxExtra,
                    maxAreaDev, nValid > 0 ? sumRatio / nValid : 0.0);
    }
    return 0;
}
