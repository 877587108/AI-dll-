// Scratch vs texture comparison sheet.
//
// The question this answers: does a scratch look different from the moulding
// texture in a way a threshold could key on? Before tuning any parameter the
// operator needs to confirm that the region being measured is actually the
// scratch, and to see an OK part at the same location for reference.
//
// Output is a side-by-side sheet per NG image:
//   left   - NG part, cropped region, raw grayscale
//   middle - NG part, same crop, excess-ink mask (what the algorithm sees)
//   right  - OK part, same crop, raw grayscale (reference texture)
//
// Usage:
//   markloc_scratchcmp <modelDir> <outDir> <ngImage> <okImage> <cx> <cy> <half>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

#ifdef _WIN32
// Convert a wide string to UTF-8.
std::string wideToU8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                        nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n,
                          nullptr, nullptr);
    return out;
}

// Windows hands the process a UTF-16 command line. Taking argv from main()
// would deliver codepage-936 bytes, which fs::path then decodes as UTF-8 and
// aborts on. Parse the wide command line instead.
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

[[maybe_unused]] std::string toU8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

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

bool writePng(const std::string& path, const cv::Mat& img) {
    std::vector<uchar> buf;
    if (!cv::imencode(".png", img, buf)) return false;
#ifdef _WIN32
    const std::wstring w = u8p(path).wstring();
    std::ofstream f(w.c_str(), std::ios::binary);
#else
    std::ofstream f(u8p(path), std::ios::binary);
#endif
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return static_cast<bool>(f);
}

// Upscale with nearest interpolation so pixel-level structure stays visible:
// a scratch is a few pixels wide, and a smooth resize would hide exactly the
// detail being judged.
cv::Mat zoom(const cv::Mat& src, int factor) {
    cv::Mat out;
    cv::resize(src, out, cv::Size(), factor, factor, cv::INTER_NEAREST);
    return out;
}

void label(cv::Mat& img, const std::string& text) {
    cv::rectangle(img, cv::Rect(0, 0, img.cols, 26), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, text, cv::Point(6, 19), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 1);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> av;
#ifdef _WIN32
    av = wideArgs();
#else
    av.assign(argv, argv + argc);
#endif
    if (av.size() < 8) {
        std::cout << "usage: markloc_scratchcmp <modelDir> <outDir> <ngImage> <okImage> <cx> <cy> <half>\n";
        return 2;
    }
    const std::string modelDir = av[1];
    const std::string outDir = av[2];
    const std::string ngPath = av[3];
    const std::string okPath = av[4];
    const int cx = std::atoi(av[5].c_str());
    const int cy = std::atoi(av[6].c_str());
    const int half = std::atoi(av[7].c_str());

    chardet::Model model;
    if (!chardet::loadModel(modelDir, model) || !model.valid()) {
        std::cout << "model load failed\n";
        return 1;
    }
    chardet::Config cfg;

    cv::Mat ngG, okG;
    if (!readGray(ngPath, ngG)) {
        std::cout << "read failed: ng\n";
        return 1;
    }
    if (!readGray(okPath, okG)) {
        std::cout << "read failed: ok\n";
        return 1;
    }

    const cv::Rect crop(cx - half, cy - half, half * 2, half * 2);
    const cv::Rect cropN = crop & cv::Rect(0, 0, ngG.cols, ngG.rows);
    const cv::Rect cropO = crop & cv::Rect(0, 0, okG.cols, okG.rows);
    if (cropN.width < 10 || cropO.width < 10) {
        std::cout << "crop outside image\n";
        return 1;
    }

    // ---- excess-ink mask on the NG crop, same construction as the detector
    cv::Mat ngSig = chardet::strokeSignal(ngG, cv::Rect(0, 0, ngG.cols, ngG.rows), cfg, false);
    cv::Mat excess;
    cv::compare(ngSig(cropN), cfg.missStrokeLo * 0.7, excess, cv::CMP_GT);
    cv::Mat modelInk;
    cv::compare(model.freq(cropN), static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);
    cv::Mat modelDil, notModel;
    cv::dilate(modelInk, modelDil, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::bitwise_not(modelDil, notModel);
    cv::bitwise_and(excess, notModel, excess);

    constexpr int kZoom = 4;
    cv::Mat a, b, c;
    cv::cvtColor(ngG(cropN), a, cv::COLOR_GRAY2BGR);
    cv::cvtColor(excess, b, cv::COLOR_GRAY2BGR);
    cv::cvtColor(okG(cropO), c, cv::COLOR_GRAY2BGR);
    a = zoom(a, kZoom);
    b = zoom(b, kZoom);
    c = zoom(c, kZoom);

    label(a, "NG raw");
    label(b, "NG excess-ink mask");
    label(c, "OK raw (same location)");

    // pad the three panels to a common height before hconcat
    const int H = std::max({a.rows, b.rows, c.rows});
    for (cv::Mat* m : {&a, &b, &c}) {
        if (m->rows < H) {
            cv::copyMakeBorder(*m, *m, 0, H - m->rows, 0, 0, cv::BORDER_CONSTANT,
                               cv::Scalar(0, 0, 0));
        }
    }
    cv::Mat sheet;
    cv::hconcat(std::vector<cv::Mat>{a, b, c}, sheet);

    std::error_code ec;
    fs::create_directories(u8p(outDir), ec);
    std::ostringstream name;
    name << outDir << "/cmp_" << std::setw(4) << std::setfill('0') << cx << "_" << std::setw(4)
         << std::setfill('0') << cy << ".png";
    if (!writePng(name.str(), sheet)) {
        std::cout << "write failed\n";
        return 1;
    }
    std::cout << "wrote " << name.str() << "  crop=" << cropN.x << "," << cropN.y << " "
              << cropN.width << "x" << cropN.height << "\n";
    return 0;
}
