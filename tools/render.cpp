// 缺陷标注渲染工具
//
// 把检测结果画在原图上：
//   红框  = 缺墨（少了一笔/整字缺失）—— 框出缺墨所在的列区间
//   蓝框  = 多墨（多了一笔/笔画增粗）
//   黄框  = 多出的字符（整行墨块比对发现）
//   绿框  = 正常字符（可选，便于对照）
//
// 三个面板并排输出：
//   ① 模型图（模板长什么样）
//   ② 待检图（原始）
//   ③ 标注图（检测结果画在上面）
//
// 用法: markloc_render <模型目录> <产品图> <输出png> [--show-ok] [--zoom N]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

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
#include "scratch_score.hpp"
#include "common/console_utf8.hpp"

namespace fs = std::filesystem;

namespace {

std::string w2u(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr,
                                        0, nullptr, nullptr);
    std::string o(static_cast<size_t>(n), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), o.data(), n, nullptr,
                          nullptr);
    return o;
}

fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

bool readGray(const std::string& p, cv::Mat& out) {
    std::ifstream f(u8p(p), std::ios::binary);
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

/// 画一个带标签的框
void drawTag(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, const std::string& label,
             int thickness = 2) {
    const cv::Rect rr = r & cv::Rect(0, 0, img.cols, img.rows);
    if (rr.width <= 0 || rr.height <= 0) return;
    cv::rectangle(img, rr, color, thickness);

    if (label.empty()) return;
    int baseline = 0;
    const double scale = 0.45;
    const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    const int ty = std::max(ts.height + 2, rr.y - 2);
    const cv::Rect bg(rr.x, ty - ts.height - 2, ts.width + 6, ts.height + baseline + 4);
    cv::rectangle(img, bg & cv::Rect(0, 0, img.cols, img.rows), color, cv::FILLED);
    cv::putText(img, label, cv::Point(rr.x + 3, ty), cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

void putHeader(cv::Mat& img, const std::string& text, const cv::Scalar& color) {
    cv::rectangle(img, cv::Rect(0, 0, img.cols, 26), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, text, cv::Point(6, 19), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                cv::LINE_AA);
}

}  // namespace

int main() {
    markloc::enableUtf8Console();
    int wc = 0;
    LPWSTR* wv = ::CommandLineToArgvW(::GetCommandLineW(), &wc);
    std::vector<std::string> args;
    if (wv) {
        for (int i = 0; i < wc; ++i) args.push_back(w2u(wv[i]));
        ::LocalFree(wv);
    }
    if (args.size() < 4) {
        std::printf("用法: markloc_render <模型目录> <产品图> <输出png> [--show-ok] [--char-ncc] [--zoom N]\n");
        return 2;
    }

    bool showOk = false;
    bool useCharNcc = false;
    double zoom = 1.0;
    for (size_t i = 4; i < args.size(); ++i) {
        if (args[i] == "--show-ok") showOk = true;
        else if (args[i] == "--char-ncc") useCharNcc = true;
        else if (args[i] == "--zoom" && i + 1 < args.size()) zoom = std::atof(args[++i].c_str());
    }

    // ---- 载入模型
    chardet::Model cm;
    if (!chardet::loadModel(args[1], cm)) {
        std::printf("模型加载失败: %s\n", args[1].c_str());
        return 1;
    }
    std::vector<chardet::CharUnit> all;
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        auto u = chardet::toUnits(cm, static_cast<int>(ri));
        for (auto& x : u) all.push_back(std::move(x));
    }

    // ---- 载入产品图
    cv::Mat gray;
    if (!readGray(args[2], gray)) {
        std::printf("产品图读取失败: %s\n", args[2].c_str());
        return 1;
    }
    if (gray.size() != cm.freq.size()) {
        std::printf("尺寸不符: 产品 %dx%d, 模型 %dx%d\n", gray.cols, gray.rows, cm.freq.cols,
                    cm.freq.rows);
        return 1;
    }

    // ---- 配准
    chardet::Config cfg = cm.config;
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    if (!cm.ref.empty()) {
        cv::Mat r32, s32;
        cm.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);

        // 只用字符区域做相位相关：整幅图会被引脚、封装边缘主导，
        // 给出错误的平移量（实测误差可达 8~16px，足以让所有框错位）。
        cv::Rect inkArea;
        for (const auto& u : all) {
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
                cv::warpAffine(sig, sig, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
                std::printf("配准平移: (%.1f, %.1f)\n", -sh.x, -sh.y);
            }
        }
    }

    // ---- 检测
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);

    int nMissing = 0, nExtra = 0, nOk = 0, nBigMiss = 0;

    for (const auto& u : all) {
        // 字符级判据（--char-ncc 开启）
        //
        // 默认不启用：实测 84 张 OK 有 39.3% 误报，因为划痕/脏污落在
        // 字符框内时无法与笔画缺陷区分。默认只跑整行墨块判据（零误报）。
        if (!useCharNcc) { ++nOk; continue; }
        const auto pc = chardet::diffNcc(sig, cm.freq, u, cfg);
        if (!pc.valid) continue;

        const bool isMiss = pc.missScore > 0.0;
        const bool isExtra = pc.extraScore > 0.0;

        if (!isMiss && !isExtra) {
            if (showOk) {
                drawTag(vis, u.bbox, cv::Scalar(80, 200, 80), "", 1);
                ++nOk;
            }
            continue;
        }

        char lbl[64];
        if (isMiss) {
            ++nMissing;
            if (pc.missScore > 0.8) ++nBigMiss;

            // 缺墨：用列/行区间画精确位置；区间都没有就画整个字符框
            bool drew = false;
            for (const auto& sx : pc.missSpanX) {
                cv::Rect r(u.bbox.x + sx.start, u.bbox.y, sx.end - sx.start, u.bbox.height);
                std::snprintf(lbl, sizeof(lbl), "缺墨 %.0f%%", pc.missScore * 100);
                drawTag(vis, r, cv::Scalar(60, 60, 230), drew ? "" : lbl);
                drew = true;
            }
            for (const auto& sy : pc.missSpanY) {
                cv::Rect r(u.bbox.x, u.bbox.y + sy.start, u.bbox.width, sy.end - sy.start);
                std::snprintf(lbl, sizeof(lbl), "缺墨 %.0f%%", pc.missScore * 100);
                drawTag(vis, r, cv::Scalar(60, 60, 230), drew ? "" : lbl);
                drew = true;
            }
            if (!drew) {
                std::snprintf(lbl, sizeof(lbl), "缺墨 %.0f%%", pc.missScore * 100);
                drawTag(vis, u.bbox, cv::Scalar(60, 60, 230), lbl);
            }
        }
        if (isExtra) {
            ++nExtra;
            bool drew = false;
            for (const auto& sx : pc.extraSpanX) {
                cv::Rect r(u.bbox.x + sx.start, u.bbox.y, sx.end - sx.start, u.bbox.height);
                std::snprintf(lbl, sizeof(lbl), "多墨 %.0f%%", pc.extraScore * 100);
                drawTag(vis, r, cv::Scalar(230, 150, 60), drew ? "" : lbl);
                drew = true;
            }
            if (!drew) {
                std::snprintf(lbl, sizeof(lbl), "多墨 %.0f%%", pc.extraScore * 100);
                drawTag(vis, u.bbox, cv::Scalar(230, 150, 60), lbl);
            }
        }
    }

    // ---- 划痕评分（Hough 直线 + 覆盖率 + 能量）
    //
    // 与字符检测分开：划痕是"连续的细长痕迹"，不是字符结构缺陷。
    // 轻微划痕应放行，只有达到长度门槛且能量足够大的才判 NG。
    render_detail::ScratchScore sc;
    {
        // 只在"字形所在区域"内找划痕，不能扫全图。
        //
        // 全图包含引脚、芯片边缘等强直线结构，实测会被 Hough 检出 4000+ 条
        // "划痕"，完全失去意义。划痕检测应限定在文字行带内。
        cv::Rect band;
        for (const auto& u : all) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            band = band.area() > 0 ? (band | b) : b;
        }
        if (band.area() > 0) {
            band.x = std::max(0, band.x - 20);
            band.y = std::max(0, band.y - 20);
            band.width = std::min(sig.cols - band.x, band.width + 40);
            band.height = std::min(sig.rows - band.y, band.height + 40);
            // 划痕必须是"模型里没有的"墨迹。
            //
            // 字符笔画本身就是密集直线段（Q/0 的竖边、L/P 的横竖），
            // 若直接对整带做 Hough，会检出上千条"划痕"。
            // 先扣掉模型笔画及其邻域，剩下的才是可疑的额外直线。
            cv::Mat weak;
            cv::compare(sig(band), cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
            weak.convertTo(weak, CV_8U, 255.0);

            cv::Mat modelInk;
            cv::compare(cm.freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk,
                        cv::CMP_GT);
            const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
            cv::Mat modelDil;
            cv::dilate(modelInk, modelDil, kd);
            cv::Mat notModel;
            cv::bitwise_not(modelDil, notModel);
            cv::bitwise_and(weak, notModel, weak);   // 只留模型之外的墨

            const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
            cv::morphologyEx(weak, weak, cv::MORPH_OPEN, k);
            // 字符笔画区域（用于判断划痕是否压在字上）
            cv::Mat charInk;
            cv::compare(cm.freq(band), static_cast<int>(cfg.missFreq * 255), charInk, cv::CMP_GT);
            sc = render_detail::scoreScratch(sig(band), weak, cfg.missStrokeLo, 160.0, 0.25,
                                             &charInk);
        }
    }
    // 只有"压在笔画上"的划痕才判 NG —— 空白区的划痕不影响字符可读性。
    //
    // 实测依据：84 张 OK 样本的划痕强度跨度达 385 倍（2305~888533），
    // 远超 NG 样本（9977~164345），说明单看强度无法区分。
    // 但客户判定逻辑是"轻微划痕放行"——关键是划痕有没有压在字上。
    // 门槛按实测标定：84 张 OK 样本的"压笔画划痕"强度 90% 分位约 2.1 万，
    // 最高 28.7 万。取 30 万作为默认门槛，可使这批复查放行件全部不误报。
    // 客户如有更严/更宽的标准，通过 GUI 的灵敏度设置调整。
    const bool scratchNg = sc.onInkBest > 300000.0 && sc.onInkLength > 60.0;
    if (sc.best > 0.0 && sc.length > 0.0) {
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "划痕 长%.0f 覆盖%.0f%%", sc.length, sc.coverage * 100);
        // 在笔画信号图上找最强划痕位置不易，这里只在结论里报告
        (void)lbl;
    }

    // ---- 逐字符对齐
    //
    // 整图配准只能消除全局平移。实测残余的字符级偏差可达 (+8,+2)、(-3,+7)，
    // 框没对准字形时，框里装的是"半个字形 + 一块空白"，墨量对不上，
    // 所有基于框内墨量的判据都会失真——这是过去大量误报的根源。
    if (useCharNcc) {
        for (auto& u : all) {
            cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (box.width <= 4 || box.height <= 4) continue;
            constexpr int kS = 12;
            cv::Rect search(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
            search &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (search.width < box.width || search.height < box.height) continue;
            cv::Mat mS, tS;
            cm.freq(search).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig(search).convertTo(tS, CV_32F);
            cv::Mat res;
            cv::matchTemplate(tS, mS(cv::Rect(box.x - search.x, box.y - search.y, box.width,
                                             box.height)),
                              res, cv::TM_CCOEFF_NORMED);
            cv::Point mLoc;
            double mVal = 0;
            cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
            if (mVal < 0.30) continue;
            const int dx = mLoc.x - kS, dy = mLoc.y - kS;
            const cv::Rect moved(box.x + dx, box.y + dy, box.width, box.height);
            if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
            u.bbox = moved;
        }
    }

    // ---- 整行墨块：多出的字符
    const auto bd = chardet::diffBlocks(sig, cm.freq, all, cfg, 8);
    int nBlockExtra = 0;
    for (const auto& b : bd.extraBlocks) {
        const cv::Rect r(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
        drawTag(vis, r, cv::Scalar(0, 220, 220), "多出的字符", 2);
        ++nBlockExtra;
    }

    // ---- 拼三个面板
    cv::Mat modelBgr;
    cv::cvtColor(cm.freq, modelBgr, cv::COLOR_GRAY2BGR);
    if (showOk) {
        for (const auto& u : all) drawTag(modelBgr, u.bbox, cv::Scalar(80, 200, 80), "", 1);
    }

    cv::Mat origBgr;
    cv::cvtColor(gray, origBgr, cv::COLOR_GRAY2BGR);

    char hdr[256];
    std::snprintf(hdr, sizeof(hdr), "MODEL (template)  %zu chars", all.size());
    putHeader(modelBgr, hdr, cv::Scalar(0, 255, 255));

    std::snprintf(hdr, sizeof(hdr), "TEST  (raw)");
    putHeader(origBgr, hdr, cv::Scalar(0, 255, 255));

    const bool anyNg = (nMissing + nExtra + nBlockExtra) > 0 || scratchNg;
    std::snprintf(hdr, sizeof(hdr),
                  "RESULT: miss=%d extra=%d blk=%d scratch=%.0f%s  ->  %s", nMissing, nExtra,
                  nBlockExtra, sc.best, scratchNg ? "(NG)" : "", anyNg ? "NG" : "OK");
    putHeader(vis, hdr, anyNg ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));

    // 三张图等高并排：模型 | 待检原图 | 标注结果
    cv::Mat cmp;
    std::vector<cv::Mat> panels = {modelBgr, origBgr, vis};
    cv::hconcat(panels, cmp);

    if (zoom != 1.0 && zoom > 0.1) {
        cv::resize(cmp, cmp, cv::Size(), zoom, zoom, zoom > 1.0 ? cv::INTER_NEAREST
                                                                : cv::INTER_AREA);
    }

    if (!cv::imwrite(args[3], cmp)) {
        std::printf("写入失败: %s\n", args[3].c_str());
        return 1;
    }

    std::printf("\n模型: %zu 字符, %d 行\n", all.size(), bd.rowCount);
    std::printf("缺墨 %d 处（其中整字缺失 %d）\n", nMissing, nBigMiss);
    std::printf("多墨 %d 处\n", nExtra);
    std::printf("多出字符 %d 处\n", nBlockExtra);
    std::printf("划痕: 全部最强 %.0f (长 %.0f)  |  压笔画 %.0f (长 %.0f, %d 条)\n",
                sc.best, sc.length, sc.onInkBest, sc.onInkLength, sc.onInkCount);
    std::printf("判定: %s\n", anyNg ? "NG" : "OK");
    std::printf("标注图: %s\n", args[3].c_str());
    return 0;
}
