// 渲染标注样式，用来验证 GUI 里"缺陷框 + 右侧调门槛提示"的实际样子。
//
// 直接复用 src/gui/annotate.hpp，而不是在这里另抄一份绘制代码：抄一遍就等于
// 验证了另一段程序，界面真正画出来的东西仍然没被看过。
//
// 用法：markloc_annotedemo <outDir> [image]
//   不给 image 时自造一张灰底图；给 image 时在真图上画，便于看叠加效果。

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
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

#include "gui/annotate.hpp"

namespace fs = std::filesystem;

namespace {

fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
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

// 与 GUI 完全相同的分档与文案，用来复现标注
void annotate(cv::Mat& vis, cv::Rect box, cv::Point a1, cv::Point a2, double composite,
              double cutCertain, double cutObvious, double cutSuspect) {
    const int band = (composite >= cutCertain) ? 3
                     : (composite >= cutObvious) ? 2
                     : (composite >= cutSuspect) ? 1
                                                 : 0;
    static const cv::Scalar kColor[4] = {cv::Scalar(170, 170, 170), cv::Scalar(0, 215, 255),
                                         cv::Scalar(0, 140, 255), cv::Scalar(0, 0, 255)};
    static const char* kTag[4] = {"FAINT", "SUSPECT", "OBVIOUS", "CERTAIN"};
    static const char* kUp[4] = {"SUSPECT", "OBVIOUS", "CERTAIN", "CERTAIN"};

    if (band >= 1) cv::line(vis, a1, a2, kColor[band], band >= 2 ? 3 : 2, cv::LINE_AA);
    cv::rectangle(vis, box, kColor[band], 2);

    char sev[128];
    std::snprintf(sev, sizeof(sev), "S=%d %s", static_cast<int>(composite), kTag[band]);
    std::vector<std::string> note;
    note.push_back(sev);
    if (band < 3) {
        const double cutCurr = (band == 2) ? cutCertain : (band == 1) ? cutObvious : cutSuspect;
        char up[128];
        std::snprintf(up, sizeof(up), "UP %s: %d->%d", kUp[band], static_cast<int>(cutCurr),
                      static_cast<int>(composite));
        note.push_back(up);
    }
    if (band > 0) {
        const double cutCurr = (band == 3) ? cutCertain : (band == 2) ? cutObvious : cutSuspect;
        char dn[128];
        std::snprintf(dn, sizeof(dn), "DN %s: %d->%d", kTag[band - 1], static_cast<int>(cutCurr),
                      static_cast<int>(composite) + 1);
        note.push_back(dn);
    }
    markloc::drawNoteRight(vis, box, kColor[band], note);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    const std::vector<std::string> av = wideArgs();
#else
    const std::vector<std::string> av(argv, argv + argc);
#endif
    if (av.size() < 2) {
        std::cout << "usage: markloc_annotedemo <outDir> [image]\n";
        return 2;
    }
    const std::string outDir = av[1];

    cv::Mat g;
    if (av.size() >= 3) {
        if (!readGray(av[2], g)) {
            std::cout << "read failed\n";
            return 1;
        }
    } else {
        g = cv::Mat(700, 1100, CV_8UC1, cv::Scalar(40));
        // 画几个假字符，模拟"划痕压在字上"
        cv::putText(g, "QO46R8", cv::Point(80, 380), cv::FONT_HERSHEY_SIMPLEX, 3.0,
                    cv::Scalar(210), 6);
        cv::putText(g, "SA6JZ1", cv::Point(80, 520), cv::FONT_HERSHEY_SIMPLEX, 3.0,
                    cv::Scalar(210), 6);
    }

    cv::Mat vis;
    cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);

    // 四个真实出现过的严重度，覆盖四个档位
    const double kCut[3] = {1500.0, 1000.0, 300.0};
    const struct { double score; cv::Point a1, a2; } cases[] = {
        {2289.0, {60, 300}, {960, 320}},     // CERTAIN
        {1321.0, {60, 430}, {820, 448}},     // OBVIOUS
        {520.0, {60, 560}, {700, 575}},      // SUSPECT
        {73.0, {60, 640}, {300, 650}},       // FAINT
    };
    for (const auto& c : cases) {
        cv::Rect box(std::min(c.a1.x, c.a2.x) - 4, std::min(c.a1.y, c.a2.y) - 8,
                     std::abs(c.a2.x - c.a1.x) + 9, std::abs(c.a2.y - c.a1.y) + 17);
        annotate(vis, box, c.a1, c.a2, c.score, kCut[0], kCut[1], kCut[2]);
    }

    std::error_code ec;
    fs::create_directories(u8p(outDir), ec);
    writePng(outDir + "/annotated.png", vis);
    std::cout << "wrote " << outDir << "/annotated.png\n";
    return 0;
}
