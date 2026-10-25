// locate_bench.cpp —— 定位算法测试台
//
// ================================================================
// 用途
// ================================================================
//
// 加载一个定位插件 DLL，对一批图跑定位，评估结果并输出：
//   · CSV：每个字符的定位框、置信度、与模板的相对偏差
//   · 叠画图：把定位框画在原图上，一眼看出偏没偏
//   · 统计：各字符的偏差分布、失败率
//
// ================================================================
// 为什么需要「与模板的相对偏差」这个指标
// ================================================================
//
// 定位对不对，不能只看置信度。NCC 0.92 也可能定位到了隔壁的字上。
// 真正要问的是：**定位出的框，相对它该在的位置偏了多少。**
//
// 但"该在的位置"是什么？
//   · 若无人工标注真值，就用「模板框按整图尺度换算」的位置作为参照
//   · 这不是真值，是"如果整图只是等比缩放+平移，它应该在的地方"
//   · 偏差大 = 要么定位错了，要么这张图的形变不是等比缩放+平移
//
// 这个指标的价值在于：**同一批图上比较不同插件**。
// 两个插件谁的偏差小，谁就更稳，不需要人工标注。
//
// ================================================================
// 用法
// ================================================================
//
//   markloc_locatebench --plugin ncc.dll --model model/v00 --tmpl tmpl.png \
//                        --list images.txt --out out/
//
//   选项：
//     --plugin PATH       插件 DLL 路径
//     --model DIR         模型目录（含 freq.png / regions.txt）
//     --tmpl PATH         模板图（用于在待测图里粗略定位，先给插件一个初值）
//     --list FILE         图片列表（UTF-8，每行一个）
//     --out DIR           输出目录
//     --param k=v         传给插件的参数，可重复
//     --max N             最多处理 N 张
//     --list-params       只列出插件参数就退出
//     --debug N           把第 N 张的匹配响应存下来
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
// windows.h 的 min/max 宏会破坏 std::max/std::min，
// 必须在包含任何标准库头之前定义 NOMINMAX。
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "../include/locate_plugin/locate_plugin.hpp"

namespace {

// ---------------------------------------------------------------- 工具

bool readGray(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray b = f.readAll();
    f.close();
    if (b.isEmpty()) return false;
    const std::vector<uchar> buf(b.begin(), b.end());
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

std::vector<std::string> readList(const char* path) {
    std::vector<std::string> out;
    QFile f(QString::fromUtf8(path));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return out;
    while (!f.atEnd()) {
        const QString line = QString::fromUtf8(f.readLine()).trimmed();
        if (!line.isEmpty()) out.push_back(line.toStdString());
    }
    return out;
}

// ---------------------------------------------------------------- DLL 加载

class Plugin {
public:
    ~Plugin() { unload(); }

    bool load(const std::string& path, std::string* err) {
#ifdef _WIN32
        // 用 UTF-8 → 宽字符，避免中文路径问题
        const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wpath(static_cast<size_t>(wn), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wn);

        hmod_ = LoadLibraryW(wpath.c_str());
        if (!hmod_) {
            const DWORD e = GetLastError();
            if (err) *err = "LoadLibrary 失败，错误码 " + std::to_string(e);
            return false;
        }
        using CreateFn = locate_api_t* (*)();
        using DestroyFn = void (*)(locate_api_t*);
        auto create = reinterpret_cast<CreateFn>(GetProcAddress(hmod_, "locate_create"));
        destroy_ = reinterpret_cast<DestroyFn>(GetProcAddress(hmod_, "locate_destroy"));
        if (!create) {
            if (err) *err = "DLL 里找不到 locate_create";
            unload();
            return false;
        }
        api_ = create();
        if (!api_) {
            if (err) *err = "locate_create 返回空";
            unload();
            return false;
        }
        if (api_->apiVersion != LOCATE_PLUGIN_API_VERSION) {
            if (err) *err = "接口版本不匹配：插件 " + std::to_string(api_->apiVersion) +
                            "，宿主 " + std::to_string(LOCATE_PLUGIN_API_VERSION);
            unload();
            return false;
        }
        return true;
#else
        (void)path;
        if (err) *err = "仅支持 Windows";
        return false;
#endif
    }

    void unload() {
        if (api_ && api_->shutdown) api_->shutdown();
        if (api_ && destroy_) destroy_(api_);
        api_ = nullptr;
#ifdef _WIN32
        if (hmod_) {
            FreeLibrary(hmod_);
            hmod_ = nullptr;
        }
#endif
    }

    locate_api_t* api() const { return api_; }

private:
#ifdef _WIN32
    HMODULE hmod_ = nullptr;
#endif
    void (*destroy_)(locate_api_t*) = nullptr;
    locate_api_t* api_ = nullptr;
};

// ---------------------------------------------------------------- 主流程

struct Options {
    std::string pluginPath;
    std::string modelDir;
    std::string tmplPath;
    std::string listPath;
    std::string outDir;
    std::vector<std::pair<std::string, double>> params;
    int maxImages = 0;
    bool listParams = false;
    int debugImage = -1;
};

void usage() {
    std::printf(
        "用法: markloc_locatebench --plugin DLL --model DIR --tmpl PNG --list FILE [选项]\n"
        "  --out DIR        输出目录（默认 out/）\n"
        "  --param k=v      传给插件，可重复\n"
        "  --max N          最多处理 N 张\n"
        "  --list-params    列出插件参数后退出\n"
        "  --debug N        存第 N 张的匹配过程\n");
}

/// 从模型目录读字符框（模型坐标）
bool loadModelBoxes(const std::string& modelDir, std::vector<TemplateBox>& boxes, int& modelW,
                    int& modelH, std::string* err) {
    const QString dir = QString::fromUtf8(modelDir.c_str());

    // 读模型图尺寸
    cv::Mat ref;
    if (!readGray(QDir(dir).absoluteFilePath("freq.png"), ref) &&
        !readGray(QDir(dir).absoluteFilePath("ref.png"), ref)) {
        if (err) *err = "读不到 freq.png / ref.png";
        return false;
    }
    modelW = ref.cols;
    modelH = ref.rows;

    // 读 regions.txt
    QFile f(QDir(dir).absoluteFilePath("regions.txt"));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (err) *err = "读不到 regions.txt";
        return false;
    }
    int idx = 0;
    while (!f.atEnd()) {
        QString line = QString::fromUtf8(f.readLine()).trimmed();
        if (line.isEmpty() || line.startsWith('#')) continue;
        const auto parts = line.split(' ', Qt::SkipEmptyParts);
        if (parts.isEmpty()) continue;
        // 行格式（regions.txt v2）：
        //   <首行> 区域数
        //   <区域行> <序号> <x> <y> <宽> <高> <字符数>          —— 6 个字段
        //   <字符行> bbox(x y w h) extend(x y w h) <序号>        —— 9 个字段
        if (parts.size() == 6) continue;          // 区域行，跳过
        if (parts.size() >= 9) {
            TemplateBox b;
            b.x = parts[0].toInt();
            b.y = parts[1].toInt();
            b.w = parts[2].toInt();
            b.h = parts[3].toInt();
            b.index = idx++;
            if (b.w > 3 && b.h > 3) boxes.push_back(b);
        }
    }
    if (boxes.empty()) {
        if (err) *err = "regions.txt 里没有字符框";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::printf("选项 %s 缺参数\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--plugin") opt.pluginPath = next();
        else if (a == "--model") opt.modelDir = next();
        else if (a == "--tmpl") opt.tmplPath = next();
        else if (a == "--list") opt.listPath = next();
        else if (a == "--out") opt.outDir = next();
        else if (a == "--max") opt.maxImages = std::atoi(next());
        else if (a == "--debug") opt.debugImage = std::atoi(next());
        else if (a == "--list-params") opt.listParams = true;
        else if (a == "--param") {
            const std::string kv = next();
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                opt.params.emplace_back(kv.substr(0, eq), std::atof(kv.c_str() + eq + 1));
            }
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        }
    }
    if (opt.outDir.empty()) opt.outDir = "out";

    // ---- 加载插件
    Plugin plugin;
    std::string err;
    if (!plugin.load(opt.pluginPath, &err)) {
        std::printf("加载插件失败: %s\n", err.c_str());
        return 1;
    }
    locate_api_t* api = plugin.api();
    std::printf("插件: %s （接口版本 %d）\n", api->name ? api->name : "?", api->apiVersion);

    if (opt.listParams) {
        int n = 0;
        if (api->listParams) {
            const char** ps = api->listParams(&n);
            for (int i = 0; i < n; ++i) std::printf("  %s\n", ps[i]);
        }
        return 0;
    }

    // ---- 初始化插件
    if (api->init && !api->init(opt.modelDir.c_str())) {
        std::printf("插件 init 失败（模型目录 %s）\n", opt.modelDir.c_str());
        return 1;
    }
    if (api->setParams && !opt.params.empty()) {
        std::vector<LocateParam> lp;
        for (const auto& kv : opt.params) {
            LocateParam p;
            p.key = kv.first.c_str();
            p.value = kv.second;
            lp.push_back(p);
        }
        api->setParams(lp.data(), static_cast<int>(lp.size()));
        std::printf("参数: ");
        for (const auto& kv : opt.params) std::printf("%s=%g ", kv.first.c_str(), kv.second);
        std::printf("\n");
    }

    // ---- 读模型字符框
    std::vector<TemplateBox> boxes;
    int modelW = 0, modelH = 0;
    if (!loadModelBoxes(opt.modelDir, boxes, modelW, modelH, &err)) {
        std::printf("读模型失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("模型: %dx%d，%zu 个字符框\n", modelW, modelH, boxes.size());

    // ---- 读图列表
    const auto files = readList(opt.listPath.c_str());
    if (files.empty()) {
        std::printf("图片列表为空: %s\n", opt.listPath.c_str());
        return 1;
    }
    std::printf("待测图: %zu 张\n\n", files.size());

    // ---- 模板图（用于给插件一个初始位置参考）
    cv::Mat tmpl;
    if (!opt.tmplPath.empty()) readGray(QString::fromUtf8(opt.tmplPath.c_str()), tmpl);

    // ---- 输出 CSV
    const QString csvPath = QDir(QString::fromUtf8(opt.outDir.c_str()))
                                .absoluteFilePath("locate_result.csv");
    FILE* csv = std::fopen(csvPath.toUtf8().constData(), "w");
    if (csv) {
        std::fprintf(csv,
                     "image_file,image_w,image_h,char_index,accepted,conf,x,y,w,h,"
                     "dx,dy,abs_dx,abs_dy,ink,global_ok\n");
    }

    // ---- 逐图处理
    int nOk = 0, nFail = 0, nTotalChars = 0, nAccepted = 0, nGlobalOk = 0;
    double sumAbsDx = 0, sumAbsDy = 0, sumInk = 0;
    std::vector<double> allDx, allDy, allInk;

    const int limit = opt.maxImages > 0 ? std::min(opt.maxImages, static_cast<int>(files.size()))
                                        : static_cast<int>(files.size());
    for (int fi = 0; fi < limit; ++fi) {
        const std::string& path = files[static_cast<size_t>(fi)];
        const QString qp = QString::fromUtf8(path.c_str());
        const QString fn = QFileInfo(qp).fileName();

        cv::Mat gray;
        if (!readGray(qp, gray)) {
            std::printf("[%d/%d] %-38s 读取失败\n", fi + 1, limit,
                        fn.toStdString().substr(0, 36).c_str());
            ++nFail;
            continue;
        }

        // 笔画信号（全图，保证尺度估计一致）
        chardet::Config cfg;
        const cv::Mat signal =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

        // ---- 调用插件
        const auto t0 = std::chrono::steady_clock::now();
        LocateResult res = api->detect(&signal, gray.cols, gray.rows, modelW, modelH, boxes.data(),
                                       static_cast<int>(boxes.size()));
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!res.chars || res.charCount <= 0) {
            std::printf("[%d/%d] %-38s 无结果 %s\n", fi + 1, limit,
                        fn.toStdString().substr(0, 36).c_str(),
                        res.message ? res.message : "");
            ++nFail;
            continue;
        }

        // ---- 评估
        // 参照位置：用插件报告的整图仿射（scale/offset），而非图幅比例。
        // 之前用 gray.cols/modelW 当参照，但真实变换不是图幅比例缩放
        // （实测 scale≈1.0 + 小平移），导致假偏差 50px+。
        // 这里的 dx/dy 现在是"定位框相对插件自身变换的残差"，
        // 反映的是逐字符精调的量，不是绝对准确度。
        // 绝对准确度用墨量验证（见下方 ink）。
        const double sx = res.scaleX > 0 ? res.scaleX : 1.0;
        const double sy = res.scaleY > 0 ? res.scaleY : 1.0;
        const double ox = res.offsetX;
        const double oy = res.offsetY;

        int acc = 0;
        double maxAbs = 0;
        for (int i = 0; i < res.charCount; ++i) {
            const LocatedChar& lc = res.chars[i];
            const TemplateBox& tb = boxes[static_cast<size_t>(i)];
            const double ex = tb.x * sx + ox;
            const double ey = tb.y * sy + oy;
            const double dx = lc.x - ex;
            const double dy = lc.y - ey;
            const double adx = std::abs(dx), ady = std::abs(dy);
            maxAbs = std::max(maxAbs, std::max(adx, ady));
            allDx.push_back(dx);
            allDy.push_back(dy);

            // 墨量验证：定位框内灰度 > inkThr 的像素占比。
            // 这是独立于任何参照系假设的客观准确度指标：
            // 框套在字符上 → 墨量高；框偏了 → 墨量低。
            double ink = 0.0;
            {
                const int inkThr = 100;
                cv::Rect rr(lc.x, lc.y, lc.w, lc.h);
                rr &= cv::Rect(0, 0, gray.cols, gray.rows);
                if (rr.width > 0 && rr.height > 0) {
                    const cv::Mat patch = gray(rr);
                    ink = static_cast<double>(cv::countNonZero(patch > inkThr)) /
                          (patch.rows * patch.cols);
                }
            }

            if (lc.accepted) {
                ++acc;
                ++nAccepted;
                sumAbsDx += adx;
                sumAbsDy += ady;
                sumInk += ink;
                allInk.push_back(ink);
            }
            if (csv) {
                std::fprintf(csv,
                             "\"%s\",%d,%d,%d,%d,%.4f,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d\n",
                             fn.toUtf8().constData(), gray.cols, gray.rows, lc.index,
                             lc.accepted ? 1 : 0, lc.confidence, lc.x, lc.y, lc.w, lc.h, dx, dy,
                             adx, ady, ink, res.globalOk ? 1 : 0);
            }
        }
        nTotalChars += res.charCount;
        if (res.globalOk) ++nGlobalOk;
        ++nOk;

        std::printf("[%d/%d] %-38s %d/%d 定位  max|d|=%.1f  scale=%.3f,%.3f  %.1fms\n", fi + 1,
                    limit, fn.toStdString().substr(0, 36).c_str(), acc, res.charCount, maxAbs,
                    sx, sy, ms);

        // ---- 叠画
        const QString visPath =
            QDir(QString::fromUtf8(opt.outDir.c_str()))
                .absoluteFilePath(QFileInfo(qp).completeBaseName() + "_locate.png");
        cv::Mat vis;
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        for (int i = 0; i < res.charCount; ++i) {
            const LocatedChar& lc = res.chars[i];
            const cv::Scalar col = lc.accepted ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(vis, cv::Rect(lc.x, lc.y, lc.w, lc.h), col, 1, cv::LINE_AA);
        }
        if (csv) {
            std::fflush(csv);
        }
        cv::imwrite(visPath.toUtf8().constData(), vis);

        if (fi == opt.debugImage) {
            std::printf("  （调试：第 %d 张已存 %s）\n", fi,
                        visPath.toUtf8().constData());
        }
    }

    if (csv) std::fclose(csv);
    if (api->shutdown) api->shutdown();

    // ---- 统计
    std::printf("\n===== 汇总 =====\n");
    std::printf("图: 成功 %d / 失败 %d / 共 %d\n", nOk, nFail, limit);
    std::printf("字符: 定位成功 %d / 总数 %d （%.1f%%）\n", nAccepted, nTotalChars,
                nTotalChars > 0 ? 100.0 * nAccepted / nTotalChars : 0.0);
    std::printf("整图可信（>=80%%字符成功）: %d / %d\n", nGlobalOk, nOk);
    if (nAccepted > 0) {
        std::printf("平均偏差: |dx|=%.2f px, |dy|=%.2f px\n", sumAbsDx / nAccepted,
                    sumAbsDy / nAccepted);
    }
    if (!allDx.empty()) {
        std::vector<double> v = allDx;
        std::sort(v.begin(), v.end());
        std::printf("dx 分布: 中位 %.2f, p90 %.2f, 最大 %.2f\n", v[v.size() / 2],
                    v[v.size() * 9 / 10], v.back());
        v = allDy;
        std::sort(v.begin(), v.end());
        std::printf("dy 分布: 中位 %.2f, p90 %.2f, 最大 %.2f\n", v[v.size() / 2],
                    v[v.size() * 9 / 10], v.back());
    }
    if (!allInk.empty()) {
        std::sort(allInk.begin(), allInk.end());
        double inkMed = allInk[allInk.size() / 2];
        int inkGood = 0;
        for (double v : allInk) if (v >= 0.15) ++inkGood;
        std::printf("墨量验证(独立准确度): 均值 %.3f, 中位 %.3f, 高墨(>=0.15) %d/%zu\n",
                    sumInk / allInk.size(), inkMed, inkGood, allInk.size());
    }
    std::printf("CSV: %s\n", csvPath.toUtf8().constData());
    return 0;
}
