// 批量评估工具：不依赖 Qt，直接调 core 的 analyze 做全量回归
// 用法: markloc_eval <image_root> <model_dir> [--view NN] [--set k=v ...] [--verbose]
//
// 输出两种口径的 TP/FP/FN/TN，与 GUI 批量窗口口径一致：
//   按图：每张图独立判定
//   按零件：同产品任一视角 NG 则产品 NG
// 用于在沙箱无法构建 GUI 时验证 core 改动。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

#include "common/console_utf8.hpp"
#include "core/char_core.hpp"
#include "core/naming.hpp"

using namespace markloc::charinsp;

namespace {

std::filesystem::path u8path(const std::string& s) {
    return std::filesystem::path(
        std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

std::string u8str(const std::u8string& s) {
    return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

std::string wideToUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                       nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n,
                          nullptr, nullptr);
    return out;
}

bool readGray(const std::string& path, cv::Mat& out) {
    std::ifstream f(u8path(path), std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const auto n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0) return false;
    std::vector<uchar> buf(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(buf.data()), n);
    if (!f) return false;
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

}  // namespace

int main() {
    markloc::enableUtf8Console();
    int wargc = 0;
    LPWSTR* wargv = ::CommandLineToArgvW(::GetCommandLineW(), &wargc);
    std::vector<std::string> args;
    if (wargv) {
        for (int i = 0; i < wargc; ++i) args.push_back(wideToUtf8(wargv[i]));
        ::LocalFree(wargv);
    }
    if (args.size() < 3) {
        std::printf("用法: markloc_eval <image_root> <model_dir> [--view NN] [--set k=v]... "
                    "[--verbose]\n");
        return 2;
    }
    const std::string root = args[1];
    const std::string modelDir = args[2];
    int viewFilter = -1;
    bool verbose = false;
    std::string disableRoi;   // 逗号分隔的区域名或序号，如 "code" 或 "0,3"
    std::string enableRoi;    // 强制开启（覆盖 params 中的 enabled=0）
    std::map<std::string, double> overrides;

    for (size_t i = 3; i < args.size(); ++i) {
        if (args[i] == "--view" && i + 1 < args.size()) {
            viewFilter = std::atoi(args[++i].c_str());
        } else if (args[i] == "--disable-roi" && i + 1 < args.size()) {
            disableRoi = args[++i];
        } else if (args[i] == "--enable-roi" && i + 1 < args.size()) {
            enableRoi = args[++i];
        } else if (args[i] == "--set" && i + 1 < args.size()) {
            const std::string kv = args[++i];
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                overrides[kv.substr(0, eq)] = std::atof(kv.substr(eq + 1).c_str());
            }
        } else if (args[i] == "--verbose") {
            verbose = true;
        }
    }

    Model model;
    std::string err;
    if (!model.loadFrom(modelDir, &err)) {
        std::printf("模型加载失败: %s\n", err.c_str());
        return 1;
    }

    // 收集样本
    std::vector<std::string> files = naming::collectImages(root);
    if (files.empty()) {
        std::printf("未找到样本: %s\n", root.c_str());
        return 1;
    }

    int imgTp = 0, imgFp = 0, imgTn = 0, imgFn = 0, skipped = 0, decodeFail = 0;
    int total = 0;

    struct ProdAgg {
        bool truthNg = false;
        bool predictedNg = false;
        double maxScore = 0.0;
        std::array<double, kViewCount> viewMax{};
        int images = 0;
    };
    std::map<std::string, ProdAgg> prod;

    for (const std::string& f : files) {
        const int v = naming::viewOf(f);
        if (v < 0) { ++skipped; continue; }
        if (viewFilter >= 0 && v != viewFilter) { ++skipped; continue; }
        if (!model.valid(v)) { ++skipped; continue; }
        // 遵循 enabled 开关：未启用的视角不参与判定（与 analyzeProduct 一致）。
        // 显式给了 --view 或 --set enabled=1 时例外，便于逐视角调参。
        if (viewFilter < 0 && !model.params[static_cast<size_t>(v)].enabled &&
            overrides.find("enabled") == overrides.end()) {
            ++skipped;
            continue;
        }

        ++total;
        cv::Mat gray;
        if (!readGray(f, gray)) {
            ++decodeFail;
            continue;
        }

        Params p = model.params[static_cast<size_t>(v)];
        // 按名称/序号开启或关闭指定区域
        auto applyRoiToggle = [&](const std::string& list, bool enable) {
            if (list.empty()) return;
            std::string tok;
            std::stringstream ss(list);
            while (std::getline(ss, tok, ',')) {
                while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
                while (!tok.empty() && tok.back() == ' ') tok.pop_back();
                for (int k = 0; k < 4; ++k) {
                    std::string trimmed = p.rois[k].name;
                    while (!trimmed.empty() && trimmed.back() == ' ') trimmed.pop_back();
                    if (tok == trimmed || tok == std::to_string(k)) p.rois[k].enabled = enable;
                }
            }
        };
        applyRoiToggle(enableRoi, true);
        applyRoiToggle(disableRoi, false);
        // 应用覆盖
        for (const auto& [k, val] : overrides) {
            if (k == "edgeTB") p.edgeTB = static_cast<int>(val);
            else if (k == "edgeLR") p.edgeLR = static_cast<int>(val);
            else if (k == "sigma") p.sigma = val;
            else if (k == "strokeHi") p.strokeHi = val;
            else if (k == "strokeLo") p.strokeLo = val;
            else if (k == "pKeep") p.pKeep = val;
            else if (k == "dilateK") p.dilateK = static_cast<int>(val);
            else if (k == "scratchT") p.scratchT = val;
            else if (k == "scratchMinCoverage") p.scratchMinCoverage = val;
            else if (k == "scratchMinLen") p.scratchMinLen = val;
            else if (k == "appendageT") p.appendageT = val;
            else if (k == "eccMin") p.eccMin = val;
            else if (k == "enabled") p.enabled = val != 0.0;
            else if (k == "checkMissing") p.checkMissing = val != 0.0;
            else if (k == "missFreq") p.missFreq = val;
            else if (k == "missStrokeLo") p.missStrokeLo = val;
            else if (k == "checkChars") p.checkChars = val != 0.0;
            else if (k == "charMissRatio") p.charMissRatio = val;
            else if (k == "charExtraRatio") p.charExtraRatio = val;
        }

        const Result r = analyze(gray, model.views[static_cast<size_t>(v)], p, false);

        // 真值：目录名
        const std::string dirName = u8str(u8path(f).parent_path().filename().u8string());
        bool truthNg = false;
        bool hasTruth = false;
        if (dirName == "ng" || dirName == "NG") { truthNg = true; hasTruth = true; }
        else if (dirName == "ok" || dirName == "OK") { truthNg = false; hasTruth = true; }

        if (hasTruth) {
            if (truthNg) { if (r.ng) ++imgTp; else ++imgFn; }
            else { if (r.ng) ++imgFp; else ++imgTn; }
        }

        const std::string pid = naming::productOf(f);
        auto& a = prod[pid];
        a.images++;
        if (hasTruth && truthNg) a.truthNg = true;
        if (r.ng) a.predictedNg = true;
        a.maxScore = std::max(a.maxScore, r.score);
        a.viewMax[static_cast<size_t>(v)] = std::max(a.viewMax[static_cast<size_t>(v)], r.score);

        if (verbose) {
            std::printf("  %s v%02d %s score=%.3f  [logo %.2f line1 %.2f line2 %.2f code %.2f]\n",
                        pid.c_str(), v, r.ng ? "NG" : "OK", r.score, r.regions[0].score,
                        r.regions[1].score, r.regions[2].score, r.regions[3].score);
        }
    }

    int pTp = 0, pFp = 0, pTn = 0, pFn = 0;
    for (const auto& [pid, a] : prod) {
        if (a.truthNg) { if (a.predictedNg) ++pTp; else ++pFn; }
        else { if (a.predictedNg) ++pFp; else ++pTn; }
    }

    std::printf("\n===== markloc_eval =====\n");
    std::printf("样本 %d 张（跳过 %d，解码失败 %d）\n", total, skipped, decodeFail);
    std::printf("\n[按图]\n  TP=%d FP=%d FN=%d TN=%d\n", imgTp, imgFp, imgFn, imgTn);
    const int imgLabeled = imgTp + imgFp + imgFn + imgTn;
    if (imgLabeled > 0) {
        const double acc = 100.0 * (imgTp + imgTn) / imgLabeled;
        const double prec = (imgTp + imgFp) > 0 ? 100.0 * imgTp / (imgTp + imgFp) : 0.0;
        const double rec = (imgTp + imgFn) > 0 ? 100.0 * imgTp / (imgTp + imgFn) : 0.0;
        const double f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
        std::printf("  正确率 %.2f%%  精确率 %.2f%%  召回率 %.2f%%  F1 %.2f%%\n", acc, prec, rec,
                    f1);
    }
    std::printf("\n[按零件]\n  TP=%d FP=%d FN=%d TN=%d\n", pTp, pFp, pFn, pTn);
    const int pLabeled = pTp + pFp + pFn + pTn;
    if (pLabeled > 0) {
        const double acc = 100.0 * (pTp + pTn) / pLabeled;
        const double prec = (pTp + pFp) > 0 ? 100.0 * pTp / (pTp + pFp) : 0.0;
        const double rec = (pTp + pFn) > 0 ? 100.0 * pTp / (pTp + pFn) : 0.0;
        const double f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
        std::printf("  正确率 %.2f%%  精确率 %.2f%%  召回率 %.2f%%  F1 %.2f%%\n", acc, prec, rec,
                    f1);
    }

    // 列出误判
    std::printf("\n[误判明细]\n");
    bool anyBad = false;
    for (const auto& [pid, a] : prod) {
        if (a.truthNg != a.predictedNg) {
            anyBad = true;
            std::printf("  %s 真值=%s 判定=%s 最高分=%.3f\n", pid.c_str(),
                        a.truthNg ? "NG" : "OK", a.predictedNg ? "NG" : "OK", a.maxScore);
        }
    }
    if (!anyBad) std::printf("  （无）\n");
    return 0;
}
