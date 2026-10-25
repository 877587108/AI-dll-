// chase0101.cpp —— 专查 0101：它到底有没有划痕，S 卡在哪
//
// 用与生产代码完全一致的对齐链（相位相关 -> 逐字符投影对齐 + 搬像素），
// 逐条列出 0101 的全部命中，并给出每条命中的原料来源：
//   框内比例  这条线是否在字符框里
//   inkDist   到模型墨迹的距离（大=离开字形=真划痕）
//   漏出比例  亮但模型说没墨的比例
//   平行中位  与字形笔画方向的平行度（高=沿笔画走=字形漏出）
//
// 判读：若 0101 的最高 S 低于门槛但明显高于其他清洁件，它可能是
// "真缺陷但信号弱"；若与清洁件同分布，它就是标错的干净件。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

namespace fs = std::filesystem;

static cv::Mat readTif(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::vector<uchar> buf;
    uchar tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(f);
    if (buf.empty()) return {};
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

static int bestShift1D(const std::vector<double>& a, const std::vector<double>& b, int R) {
    const int n = (int)a.size();
    if (n < 4) return 0;
    int best = 0;
    double bv = -2.0;
    for (int d = -R; d <= R; ++d) {
        double sa = 0, sb = 0, sa2 = 0, sb2 = 0, sab = 0;
        int cnt = 0;
        for (int x = 0; x < n; ++x) {
            const int xb = x + d;
            if (xb < 0 || xb >= n) continue;
            const double va = a[x], vb = b[xb];
            sa += va; sb += vb; sa2 += va * va; sb2 += vb * vb; sab += va * vb; ++cnt;
        }
        if (cnt < n / 2) continue;
        const double ma = sa / cnt, mb = sb / cnt;
        const double va = sa2 / cnt - ma * ma;
        const double vb = sb2 / cnt - mb * mb;
        if (va < 1e-9 || vb < 1e-9) continue;
        const double v = (sab / cnt - ma * mb) / std::sqrt(va * vb);
        if (v > bv) { bv = v; best = d; }
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_chase0101 <modelDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    const cv::Mat modelInkFull = (model.freq > static_cast<int>(cfg.blankFreq * 255));

    for (int ai = 2; ai < argc; ++ai) {
        cv::Mat gray = readTif(argv[ai]);
        if (gray.empty() || gray.size() != model.freq.size()) continue;
        std::printf("\n===== %s =====\n", fs::path(argv[ai]).filename().string().c_str());

        cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        std::vector<chardet::CharUnit> units = baseUnits;

        // 全局对齐
        if (!model.ref.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : units) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            inkArea.x = std::max(0, inkArea.x - 20);
            inkArea.y = std::max(0, inkArea.y - 20);
            inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
            inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            sig(inkArea).convertTo(sC, CV_32F);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::Mat aligned;
                cv::warpAffine(sig, aligned, M, sig.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                sig = aligned;
            }
            std::printf("全局对齐 dx=%+.1f dy=%+.1f\n", sh.x, sh.y);
        }

        // 逐字符投影对齐 + 搬像素（与生产代码一致）
        std::vector<cv::Point> shifts(units.size(), cv::Point(0, 0));
        int moved = 0;
        {
            std::vector<cv::Point> sv(units.size(), cv::Point(0, 0));
            std::vector<char> ok(units.size(), 0);
            for (size_t ui = 0; ui < units.size(); ++ui) {
                const cv::Rect box = units[ui].bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (box.width <= 8 || box.height <= 8) continue;
                constexpr int kS = 8;
                cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
                sr &= cv::Rect(0, 0, sig.cols, sig.rows);
                if (sr.width < box.width || sr.height < box.height) continue;
                cv::Mat fm;
                model.freq(box).convertTo(fm, CV_32F, 1.0 / 255.0);
                std::vector<double> pm(box.width, 0.0), qm(box.width, 0.0);
                for (int x = 0; x < box.width; ++x) {
                    double s = 0;
                    const int xx = box.x - sr.x + x;
                    if (xx >= 0 && xx < sr.width)
                        for (int y = box.y - sr.y; y < box.y - sr.y + box.height; ++y)
                            if (y >= 0 && y < sr.height) s += sig.at<float>(sr.y + y, sr.x + xx);
                    pm[x] = s;
                    double t = 0;
                    for (int y = 0; y < fm.rows; ++y) t += fm.at<float>(y, x);
                    qm[x] = t;
                }
                const int ddx = bestShift1D(qm, pm, kS);
                std::vector<double> pmY(box.height, 0.0), qmY(box.height, 0.0);
                for (int y = 0; y < box.height; ++y) {
                    double s = 0;
                    const int yy = box.y - sr.y + y;
                    if (yy >= 0 && yy < sr.height)
                        for (int x = box.x - sr.x; x < box.x - sr.x + box.width; ++x)
                            if (x >= 0 && x < sr.width) s += sig.at<float>(sr.y + yy, sr.x + x);
                    pmY[y] = s;
                    double t = 0;
                    for (int x = 0; x < fm.cols; ++x) t += fm.at<float>(y, x);
                    qmY[y] = t;
                }
                const int ddy = bestShift1D(qmY, pmY, kS);
                sv[ui] = cv::Point(ddx, ddy);
                ok[ui] = 1;
            }
            cv::Mat warped = sig.clone();
            for (size_t ui = 0; ui < units.size(); ++ui) {
                if (!ok[ui]) continue;
                const cv::Point s = sv[ui];
                shifts[ui] = s;
                if (s.x == 0 && s.y == 0) continue;
                if (std::abs(s.x) > 8 || std::abs(s.y) > 8) continue;
                cv::Rect box = units[ui].bbox;
                box.x -= 6; box.y -= 6; box.width += 12; box.height += 12;
                box &= cv::Rect(0, 0, sig.cols, sig.rows);
                if (box.width <= 2 || box.height <= 2) continue;
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, s.x, 0, 1, s.y);
                cv::Mat piece;
                cv::warpAffine(sig(box), piece, M, box.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REFLECT);
                piece.copyTo(warped(box));
                units[ui].bbox.x += s.x;
                units[ui].bbox.y += s.y;
                ++moved;
            }
            sig = warped;
        }
        std::printf("逐字符搬了 %d 个字的像素\n", moved);

        // 距离图与梯度（用于方向判据）
        cv::Mat notInk, dist;
        cv::bitwise_not(modelInkFull, notInk);
        cv::distanceTransform(notInk, dist, cv::DIST_L2, 3);
        cv::Mat inkF, gx, gy;
        modelInkFull.convertTo(inkF, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(inkF, inkF, cv::Size(5, 5), 0);
        cv::Sobel(inkF, gx, CV_32F, 1, 0, 3);
        cv::Sobel(inkF, gy, CV_32F, 0, 1, 3);

        markloc::ScratchParams sp;
        markloc::ScratchStats st;
        auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp, &st);
        std::printf("候选=%d 过短=%d 覆盖不足=%d 距离不足=%d 方向不足=%d 严重度不足=%d 通过=%d 最高S=%.0f\n",
                    st.candidates, st.belowLen, st.belowCov, st.belowInkDist, st.belowDir,
                    st.belowComposite, st.passed, st.bestComposite);
        if (hits.empty()) continue;
        std::sort(hits.begin(), hits.end(),
                  [](const markloc::ScratchHit& a, const markloc::ScratchHit& b) {
                      return a.composite > b.composite;
                  });
        std::printf("%3s %-22s %6s %6s %6s %6s %8s %8s %8s\n", "#", "线段", "len", "深", "跨",
                    "框内", "inkDist", "漏出", "平行中位");
        for (size_t i = 0; i < hits.size() && i < 12; ++i) {
            const auto& h = hits[i];
            const int n = std::max(2, (int)h.len);
            const double dl = std::max(1e-6, h.len);
            const double ux = (h.a2.x - h.a1.x) / dl, uy = (h.a2.y - h.a1.y) / dl;
            std::vector<float> ds;
            std::vector<double> cs;
            int inBox = 0, lit = 0, leak = 0, tot = 0, vert = 0;
            for (int t = 0; t <= n; ++t) {
                const double rr = (double)t / n;
                const cv::Point p((int)std::lround(h.a1.x + (h.a2.x - h.a1.x) * rr),
                                  (int)std::lround(h.a1.y + (h.a2.y - h.a1.y) * rr));
                if (p.x < 0 || p.y < 0 || p.x >= sig.cols || p.y >= sig.rows) continue;
                ++tot;
                for (const auto& cu : units)
                    if (cu.bbox.contains(p)) { ++inBox; break; }
                if (sig.at<float>(p) > 0.105f) {
                    ++lit;
                    ds.push_back(dist.at<float>(p));
                    if (model.freq.at<uchar>(p) <= (int)(cfg.blankFreq * 255)) ++leak;
                }
                const double gxv = gx.at<float>(p), gyv = gy.at<float>(p);
                const double gm = std::hypot(gxv, gyv);
                if (gm >= 0.02) {
                    const double c = std::abs(ux * (-gyv / gm) + uy * (gxv / gm));
                    cs.push_back(c);
                    if (c < 0.35) ++vert;
                }
            }
            float p50 = -1;
            if (!ds.empty()) { std::sort(ds.begin(), ds.end()); p50 = ds[ds.size() / 2]; }
            double par = -1;
            if (!cs.empty()) { std::sort(cs.begin(), cs.end()); par = cs[cs.size() / 2]; }
            char seg[32];
            std::snprintf(seg, sizeof(seg), "(%d,%d)-(%d,%d)", h.a1.x, h.a1.y, h.a2.x, h.a2.y);
            std::printf("%3zu %-22s %6.0f %6.0f %6d %6.2f %8.1f %8.2f %8.3f  S=%.0f\n", i, seg,
                        h.len, h.depth, h.charsHit, tot ? (double)inBox / tot : 0.0, p50,
                        lit ? (double)leak / lit : 0.0, par, h.composite);
        }
    }
    return 0;
}
