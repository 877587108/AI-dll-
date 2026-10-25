// Diagnose why the grid-difference pass finds nothing.
//
// Reports the distribution of every intermediate quantity so a threshold can be
// placed against evidence rather than guessed. Without this it is impossible to
// tell "the model difference is genuinely small" apart from "a filter upstream
// wiped the signal out".
//
// Usage: markloc_grillediag <modelDir> <image...>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#include "chardet.hpp"

namespace fs = std::filesystem;

namespace {

fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}
std::string toU8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}
#ifdef _WIN32
std::string wideToU8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                        nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n,
                          nullptr, nullptr);
    return out;
}
std::vector<std::string> wideArgs() {
    std::vector<std::string> out;
    int argc = 0;
    LPWSTR* argv = ::CommandLineToArgvW(::GetCommandLineW(), &argc);
    if (!argv) return out;
    for (int i = 0; i < argc; ++i) out.push_back(wideToU8(argv[i]));
    ::LocalFree(argv);
    return out;
}
#endif

bool readGray(const std::string& path, cv::Mat& out) {
#ifdef _WIN32
    const std::wstring w = u8p(path).wstring();
    std::ifstream f(w.c_str(), std::ios::binary);
#else
    std::ifstream f(u8p(path), std::ios::binary);
#endif
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0) return false;
    std::vector<uchar> b(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(b.data()), n);
    if (!f) return false;
    out = cv::imdecode(b, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef _WIN32
    const std::vector<std::string> av = wideArgs();
#else
    const std::vector<std::string> av(argv, argv + argc);
#endif
    if (av.size() < 3) {
        std::cout << "usage: markloc_grillediag <modelDir> <image...>\n";
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(av[1], model) || !model.valid()) {
        std::cout << "model load failed\n";
        return 1;
    }
    chardet::Config cfg;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    std::cout << "file,variant,p99,p999,above50,above40,above30\n";

    for (size_t k = 2; k < av.size(); ++k) {
        cv::Mat g;
        if (!readGray(av[k], g) || g.size() != model.freq.size()) continue;
        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);

        if (!model.ref.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : base) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            if (inkArea.area() > 0) {
                cv::Rect roi = inkArea;
                roi.x = std::max(0, roi.x - 20);
                roi.y = std::max(0, roi.y - 20);
                roi.width = std::min(sig.cols - roi.x, roi.width + 40);
                roi.height = std::min(sig.rows - roi.y, roi.height + 40);
                cv::Mat rC, sC;
                r32(roi).copyTo(rC);
                s32(roi).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat shifted;
                    cv::warpAffine(sig, shifted, M, sig.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                    sig = shifted;
                }
            }
        }

        cv::Rect band;
        for (const auto& u : base) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            band = band.area() > 0 ? (band | b) : b;
        }
        if (band.area() <= 0) continue;

        const cv::Mat sBand = sig(band);
        cv::Mat mBand;
        model.freq(band).convertTo(mBand, CV_32F);

        // Same rescaling the evaluator uses, so the numbers here are the numbers
        // the thresholds are applied to.
        const double kInkScale = 255.0 / std::max(1e-6, cfg.strokeHi);
        cv::Mat imgInk;
        cv::multiply(sBand, kInkScale, imgInk);
        cv::threshold(imgInk, imgInk, 255.0, 255.0, cv::THRESH_TRUNC);

        cv::Mat s3;
        cv::blur(imgInk, s3, cv::Size(3, 3));
        cv::Mat floored;
        s3.copyTo(floored);
        floored.setTo(0.0, s3 < 50.0);

        auto pct = [](const cv::Mat& m, double p) {
            std::vector<float> v(m.begin<float>(), m.end<float>());
            if (v.empty()) return 0.0f;
            std::sort(v.begin(), v.end());
            const size_t i = static_cast<size_t>(p * (v.size() - 1));
            return v[i];
        };

        // The model frequency map is an average over 84 samples, so its stroke
        // edges are eroded compared with any single image. Subtracting it as-is
        // leaves a systematic "image wider than model" band along every stroke,
        // which swamps a thin scratch. Taking the maximum of the model over a
        // small neighbourhood says "the model allows ink here", which is the
        // only fair comparison given the alignment error.
        for (int d : {1, 3, 5, 7, 9}) {
            cv::Mat mDil;
            if (d <= 1) {
                mBand.copyTo(mDil);
            } else {
                cv::dilate(mBand, mDil, cv::getStructuringElement(cv::MORPH_RECT, {d, d}));
            }
            cv::Mat df;
            cv::subtract(floored, mDil, df);
            cv::blur(df, df, cv::Size(3, 3));
            int a50 = 0, a40 = 0, a30 = 0;
            for (int y = 0; y < df.rows; ++y) {
                const float* r = df.ptr<float>(y);
                for (int x = 0; x < df.cols; ++x) {
                    if (r[x] > 50) ++a50;
                    if (r[x] > 40) ++a40;
                    if (r[x] > 30) ++a30;
                }
            }
            std::cout << toU8(u8p(av[k]).filename()) << ",dil" << d << ',' << std::fixed
                      << std::setprecision(1) << pct(df, 0.99) << ',' << pct(df, 0.999) << ','
                      << a50 << ',' << a40 << ',' << a30 << '\n';
        }
    }
    return 0;
}

