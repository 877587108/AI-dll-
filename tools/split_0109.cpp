// split_0109.cpp —— 把 0109_RT / 0109_FT 的命中逐条拆开对比
//
// 目的：0109_RT 和 0109_FT 同号不同位，命中都判 NG。
// 我要的是"每条命中到底是同一类东西，还是两类"的量化答案。
//
// 用户已确认：0109 这两张都没有划痕。所以判读标准是确定的——
// 凡是让这两条命中的 S 掉到门槛以下的量，方向就是对的；
// 但要同时看 NG 侧 4 张真缺陷会不会被一起放过。
//
// 已确证的事实（上一轮，有工具输出佐证）：
//   S=1938 那条线在原图上不存在，放大 7 倍看到的是字形缝里的空白
//   线的原料是字母 N 的对角笔画：连续 73px signal 达 1.5，而模型 freq 是 0
//   该笔画在 84 张上的命中率中位仅 13.7%，对应 freq≈35，恰在门槛 38 之下
// 所以正确的问题不是"那里亮不亮"，而是"那段亮像素是不是字形的一部分"。
//
// 每条命中输出三个纯几何判别量（不含调参）：
//   inBoxFrac  采样点落在字符框内的比例      真划痕应跨越字形
//   inkDistP50 亮像素到最近 modelInk 的距离   字形笔画应贴近 modelInk
//   leakFrac   "亮但 freq<=38" 占亮像素比例  字形漏出应偏高
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_split0109 <modelDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型加载失败: %s\n", argv[1]);
        return 1;
    }
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);
    std::printf("基础字符单元: %zu\n", baseUnits.size());

    // modelInk 与推理门槛一致（scratch.hpp:194 modelInk = freq > blankFreq*255）
    cv::Mat inkMask = (model.freq > static_cast<int>(cfg.blankFreq * 255.0));
    cv::Mat inv, dist;
    cv::bitwise_not(inkMask, inv);
    cv::distanceTransform(inv, dist, cv::DIST_L2, 3);

    const float kWeak = static_cast<float>(cfg.missStrokeLo * 0.7);  // scratch.hpp:190

    for (int ai = 2; ai < argc; ++ai) {
        const std::string path = argv[ai];
        cv::Mat gray = readTif(path);
        if (gray.empty()) {
            std::printf("读图失败: %s\n", path.c_str());
            continue;
        }
        const std::string base = fs::path(path).filename().string();
        std::printf("\n===== %s =====\n", base.c_str());

        // ---- 与 batchcore::detectOne 完全同路
        cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg,
                                            false);

        std::vector<chardet::CharUnit> units = baseUnits;
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
                    cv::Mat aligned;
                    cv::warpAffine(sig, aligned, M, sig.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE);
                    sig = aligned;
                    std::printf("全局对齐 dx=%+.2f dy=%+.2f\n", sh.x, sh.y);
                }
            }
        }

        for (auto& u : units) {
            const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (box.width <= 4 || box.height <= 4) continue;
            constexpr int kSearch = 12;
            cv::Rect search(box.x - kSearch, box.y - kSearch, box.width + 2 * kSearch,
                            box.height + 2 * kSearch);
            search &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (search.width < box.width || search.height < box.height) continue;
            cv::Mat mS, tS;
            model.freq(search).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig(search).convertTo(tS, CV_32F);
            cv::Mat res;
            cv::matchTemplate(tS, mS(cv::Rect(box.x - search.x, box.y - search.y, box.width,
                                              box.height)),
                              res, cv::TM_CCOEFF_NORMED);
            cv::Point mLoc;
            double mVal = 0;
            cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
            if (mVal < 0.30) continue;
            const cv::Rect moved(box.x + mLoc.x - kSearch, box.y + mLoc.y - kSearch, box.width,
                                 box.height);
            if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
            u.bbox = moved;
        }

        markloc::ScratchParams sp;
        markloc::ScratchStats st;
        const auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp, &st);
        std::printf("候选=%d 过短=%d 覆盖不足=%d 能量不足=%d 严重度不足=%d 通过=%d 最高S=%.0f\n",
                    st.candidates, st.belowLen, st.belowCov, st.belowEnergy, st.belowComposite,
                    st.passed, st.bestComposite);

        for (size_t i = 0; i < hits.size(); ++i) {
            const markloc::ScratchHit& h = hits[i];
            const int n = std::max(2, static_cast<int>(h.len));
            std::vector<float> ds;
            int inBox = 0, lit = 0, leak = 0;
            // 框外连续段数：线离开所有字符框后连续走了多少段。
            // 这是"划痕能否走出字形区"的量；清白件的漏出只会沿字形边缘，
            // 出了字形就该断，真划痕会一路穿到空白处。
            int outsideRuns = 0;
            bool prevOut = false;
            for (int t = 0; t <= n; ++t) {
                const double u = static_cast<double>(t) / n;
                const cv::Point p(static_cast<int>(std::lround(h.a1.x + (h.a2.x - h.a1.x) * u)),
                                  static_cast<int>(std::lround(h.a1.y + (h.a2.y - h.a1.y) * u)));
                if (p.x < 0 || p.y < 0 || p.x >= sig.cols || p.y >= sig.rows) continue;
                bool inside = false;
                for (const auto& cu : units)
                    if (cu.bbox.contains(p)) {
                        inside = true;
                        break;
                    }
                if (inside) {
                    ++inBox;
                    prevOut = false;
                } else {
                    if (!prevOut && t > 0) ++outsideRuns;
                    prevOut = true;
                }
                if (sig.at<float>(p) > kWeak) {
                    ++lit;
                    ds.push_back(dist.at<float>(p));
                    if (model.freq.at<uchar>(p) <= static_cast<int>(cfg.blankFreq * 255.0))
                        ++leak;
                }
            }
            float p50 = -1.f;
            if (!ds.empty()) {
                std::sort(ds.begin(), ds.end());
                p50 = ds[ds.size() / 2];
            }
            std::printf(
                "  #%zu (%d,%d)-(%d,%d) len=%.0f depth=%.0f wide=%.1f cov=%.2f "
                "chars=%d band=%d S=%d | 框内=%.2f 框外段=%d inkDistP50=%.1f 漏出=%.2f\n",
                i, h.a1.x, h.a1.y, h.a2.x, h.a2.y, h.len, h.depth, h.widthPx, h.cov, h.charsHit,
                h.band, static_cast<int>(h.composite),
                n ? static_cast<double>(inBox) / (n + 1) : 0.0, outsideRuns, p50,
                lit ? static_cast<double>(leak) / lit : 0.0);
        }
    }
    return 0;
}
