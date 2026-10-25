// 方向 D 验证：多模板比对
//
// 思路：不再用 84 张的平均频率图，而是直接拿每张 OK 样本的字符实例做模板。
// 待检字符与"最近的那张模板"比对，取最小差异作为该字符的异常分。
//
// 为什么可能有效：
//   平均图是 84 张的交集，边缘被平均掉、细节被抹平；
//   真实模板保留了完整的笔画形态，改笔画这种局部结构变化更容易暴露。
//
// 本工具只做验证，不改变生产代码。

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

/// 把一个字符的笔画信号裁出来，归一到固定尺寸
cv::Mat normalizePatch(const cv::Mat& sig, const cv::Rect& box, int size) {
    const cv::Rect b = box & cv::Rect(0, 0, sig.cols, sig.rows);
    if (b.width <= 2 || b.height <= 2) return cv::Mat();

    cv::Mat patch;
    sig(b).convertTo(patch, CV_32F);
    // 裁掉低于门槛的部分再取包围盒，消除边缘空白差异
    cv::Mat out;
    cv::resize(patch, out, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
    return out;
}

struct Sample {
    std::vector<cv::Mat> patches;  // 每个字符一个归一化 patch
};

/// 与一组模板比对，返回每个字符的最小差异（1 - 最大相关系数）
double minDiffToTemplates(const cv::Mat& probe, const std::vector<cv::Mat>& templates,
                          double* bestCorr) {
    double best = -1.0;
    for (const auto& t : templates) {
        if (t.size() != probe.size()) continue;
        cv::Mat a = probe.reshape(1, 1);
        cv::Mat b = t.reshape(1, 1);
        cv::Mat am = a - cv::mean(a)[0];
        cv::Mat bm = b - cv::mean(b)[0];
        const double den = std::sqrt(am.dot(am) * bm.dot(bm));
        if (den <= 1e-9) continue;
        best = std::max(best, am.dot(bm) / den);
    }
    if (bestCorr) *bestCorr = best;
    return best < 0 ? 1.0 : (1.0 - best);
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
    // 用法: markloc_multi <模型目录> <模板图1> <模板图2> ... -- <待测图...>
    if (a.size() < 5) {
        std::printf("usage: markloc_multi <model> <tmpl...> -- <probe...>\n");
        return 2;
    }

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

    std::vector<chardet::CharUnit> base;
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        auto u = chardet::toUnits(cm, static_cast<int>(ri));
        for (auto& x : u) base.push_back(x);
    }
    const int nChar = static_cast<int>(base.size());
    constexpr int kPatch = 64;

    // ---- 提取模板库
    // templates[c] = 该字符在每张 OK 样本中的 patch
    std::vector<std::vector<cv::Mat>> templates(static_cast<size_t>(nChar));
    int nTemplate = 0;

    for (size_t i = 2; i < sep; ++i) {
        cv::Mat g;
        if (!rd(a[i], g) || g.size() != cm.freq.size()) continue;
        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cm.config, false);

        // 整图配准（字符区）
        if (!cm.ref.empty()) {
            cv::Mat r32, s32;
            cm.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect ia;
            for (const auto& u : base) ia = ia.area() > 0 ? (ia | u.bbox) : u.bbox;
            ia.x = std::max(0, ia.x - 20);
            ia.y = std::max(0, ia.y - 20);
            ia.width = std::min(sig.cols - ia.x, ia.width + 40);
            ia.height = std::min(sig.rows - ia.y, ia.height + 40);
            cv::Mat rC, sC;
            r32(ia).copyTo(rC);
            s32(ia).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
                cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::warpAffine(sig, sig, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
            }
        }

        // 逐字符对齐后取 patch
        for (int c = 0; c < nChar; ++c) {
            cv::Rect box = base[static_cast<size_t>(c)].bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (box.width <= 4 || box.height <= 4) continue;

            constexpr int kS = 12;
            cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
            sr &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (sr.width < box.width || sr.height < box.height) continue;

            cv::Mat mS, tS;
            cm.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig(sr).convertTo(tS, CV_32F);
            cv::Mat res;
            cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)),
                              res, cv::TM_CCOEFF_NORMED);
            cv::Point mL;
            double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &mL);
            if (mv < 0.30) continue;
            const cv::Rect nb(box.x + mL.x - kS, box.y + mL.y - kS, box.width, box.height);
            if ((nb & cv::Rect(0, 0, sig.cols, sig.rows)) != nb) continue;

            cv::Mat patch = normalizePatch(sig, nb, kPatch);
            if (!patch.empty()) templates[static_cast<size_t>(c)].push_back(patch);
        }
        ++nTemplate;
    }

    std::printf("模板图 %d 张，字符 %d 个，每字符平均 %.1f 张模板\n\n", nTemplate, nChar,
                nTemplate > 0 ? (double)nTemplate : 0.0);

    // ---- 待测图逐个比对
    std::printf("%-32s %10s %10s %10s\n", "probe", "maxMinDiff", "meanMinDiff", "maxCorr");

    for (size_t i = sep + 1; i < a.size(); ++i) {
        cv::Mat g;
        if (!rd(a[i], g) || g.size() != cm.freq.size()) continue;
        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cm.config, false);

        if (!cm.ref.empty()) {
            cv::Mat r32, s32;
            cm.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect ia;
            for (const auto& u : base) ia = ia.area() > 0 ? (ia | u.bbox) : u.bbox;
            ia.x = std::max(0, ia.x - 20);
            ia.y = std::max(0, ia.y - 20);
            ia.width = std::min(sig.cols - ia.x, ia.width + 40);
            ia.height = std::min(sig.rows - ia.y, ia.height + 40);
            cv::Mat rC, sC;
            r32(ia).copyTo(rC);
            s32(ia).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
                cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::warpAffine(sig, sig, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
            }
        }

        double maxDiff = 0, sumDiff = 0, maxCorr = 0;
        int nCount = 0;

        for (int c = 0; c < nChar; ++c) {
            cv::Rect box = base[static_cast<size_t>(c)].bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (box.width <= 4 || box.height <= 4) continue;

            constexpr int kS = 12;
            cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
            sr &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (sr.width < box.width || sr.height < box.height) continue;

            cv::Mat mS, tS;
            cm.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig(sr).convertTo(tS, CV_32F);
            cv::Mat res;
            cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)),
                              res, cv::TM_CCOEFF_NORMED);
            cv::Point mL;
            double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &mL);
            if (mv < 0.30) continue;
            const cv::Rect nb(box.x + mL.x - kS, box.y + mL.y - kS, box.width, box.height);
            if ((nb & cv::Rect(0, 0, sig.cols, sig.rows)) != nb) continue;

            cv::Mat patch = normalizePatch(sig, nb, kPatch);
            if (patch.empty() || templates[static_cast<size_t>(c)].empty()) continue;

            double corr = 0;
            const double d = minDiffToTemplates(patch, templates[static_cast<size_t>(c)], &corr);
            maxDiff = std::max(maxDiff, d);
            sumDiff += d;
            maxCorr = std::max(maxCorr, corr);
            ++nCount;
        }

        const fs::path fp = u8p(a[i]);
        const std::string nm(reinterpret_cast<const char*>(fp.filename().u8string().c_str()));
        std::printf("%-32s %10.4f %10.4f %10.4f\n", nm.substr(0, 32).c_str(), maxDiff,
                    nCount > 0 ? sumDiff / nCount : 0.0, maxCorr);
    }
    return 0;
}
