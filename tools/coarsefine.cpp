// 粗-精两级对齐验证
//
// 结构：投影粗对齐 -> 相位相关精对齐 -> 逐字匹配
//
// 粗对齐只负责把图拉回到相位相关能接住的范围内（<60px）；
// 精对齐用 2D 纹理做亚像素级的修正。
// 判据：整幅平移 N 像素后，检测结果是否仍然判合格。
//
// 关键设计：粗对齐结果按峰值余量把关。余量太低就不采用，
// 宁可交给相位相关自己尽力，也不能拿一个错的峰把图带偏。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

static bool readGray(const std::string& path, cv::Mat& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::vector<unsigned char> buf;
    unsigned char tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof(tmp), fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(fp);
    if (buf.empty()) return false;
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

static cv::Mat shiftImage(const cv::Mat& src, int dx, int dy) {
    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::Mat out;
    cv::warpAffine(src, out, M, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return out;
}

static std::vector<double> profile(const cv::Mat& s, bool vertical) {
    std::vector<double> p(vertical ? s.cols : s.rows, 0.0);
    for (int y = 0; y < s.rows; ++y) {
        const float* row = s.ptr<float>(y);
        for (int x = 0; x < s.cols; ++x) p[vertical ? x : y] += row[x];
    }
    return p;
}

struct ShiftEst {
    int shift = 0;
    double peak = 0;
    double margin = 0;
};

static ShiftEst bestShift(const std::vector<double>& a, const std::vector<double>& b, int lo,
                          int hi) {
    const int n = static_cast<int>(a.size());
    ShiftEst e;
    double bestV = -1e18, secondV = -1e18;
    for (int s = lo; s <= hi; ++s) {
        double num = 0, d1 = 0, d2 = 0;
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            const int j = i + s;
            if (j < 0 || j >= n) continue;
            num += a[i] * b[j];
            d1 += a[i] * a[i];
            d2 += b[j] * b[j];
            ++cnt;
        }
        if (cnt < n / 2 || d1 < 1e-9 || d2 < 1e-9) continue;
        const double v = num / std::sqrt(d1 * d2);
        if (v > bestV) {
            if (std::abs(s - e.shift) > 4) secondV = std::max(secondV, bestV);
            bestV = v;
            e.shift = s;
        } else if (v > secondV && std::abs(s - e.shift) > 4) {
            secondV = v;
        }
    }
    e.peak = bestV;
    e.margin = (secondV < -1e17) ? 999.0 : bestV - secondV;
    return e;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: coarsefine <modelDir> <image> <maxShift> [useCoarse=1] [minMargin=0.01]\n");
        return 2;
    }
    const int maxShift = std::atoi(argv[3]);
    const bool useCoarse = (argc > 4) ? std::atoi(argv[4]) != 0 : true;
    const double minMargin = (argc > 5) ? std::atof(argv[5]) : 0.01;

    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    cv::Mat gray;
    if (!readGray(argv[2], gray)) {
        std::printf("图片读取失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    const cv::Rect full(0, 0, gray.cols, gray.rows);
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    cv::Mat ref32;
    model.ref.convertTo(ref32, CV_32F);
    const std::vector<double> refV = profile(ref32, true);
    const std::vector<double> refH = profile(ref32, false);
    cv::Rect inkArea;
    for (const auto& u : base) {
        const cv::Rect b = u.bbox & full;
        if (b.width > 0 && b.height > 0) inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(gray.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(gray.rows - inkArea.y, inkArea.height + 40);

    markloc::ScratchParams sp;
    std::printf("模式=%s 余量门限=%.3f\n", useCoarse ? "粗+精" : "仅精", minMargin);
    std::printf("%6s %8s %8s %8s %10s\n", "位移", "划痕数", "最大S", "档位", "粗对齐采用");

    int firstBad = -1;
    for (int t = 0; t <= maxShift; t += 8) {
        int worstCount = 0, worstS = 0, worstB = 0, applied = 0;
        const int dirs[4][2] = {{1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
        for (const auto& d : dirs) {
            const cv::Mat img = shiftImage(gray, d[0] * t, d[1] * t);
            cv::Mat sig = chardet::strokeSignal(img, full, cfg, false);
            auto units = base;
            for (auto& u : units) u.bbox += cv::Point(d[0] * t, d[1] * t);

            // ---- 粗对齐：投影
            if (useCoarse) {
                // 搜索范围不能无脑放大。投影剖面有周期性：字符行距、字距都会
                // 制造等高的伪峰，窗口开到 ±300 时伪峰必然进入争夺。
                // 粗对齐的职责只是把图拉进精对齐能接住的范围内，
                // 所以搜索范围按"精对齐失效边界"来给，不由数据决定。
                constexpr int kCoarseR = 90;
                const ShiftEst eh = bestShift(refH, profile(sig, false), -kCoarseR, kCoarseR);
                const ShiftEst ev = bestShift(refV, profile(sig, true), -kCoarseR, kCoarseR);
                // 两个方向必须同时采用或同时放弃。
                // 分别判断会造成"半对齐"：一个方向修好了、另一个还错着几十像素，
                // 这种错位比完全不动更糟——精对齐面对的是横竖不等的畸变，
                // 两边都追不上。64px 时正是这样：dx 余量 0.040 通过、
                // dy 余量 0.000 被拒，只修了横向，结果 22 条误报。
                const bool useBoth = (eh.margin >= minMargin && ev.margin >= minMargin) &&
                                     (eh.shift != 0 || ev.shift != 0);
                int cy = useBoth ? eh.shift : 0;
                int cx = useBoth ? ev.shift : 0;
                // 把估计值打出来。之前只看"采不采用"，等于在盲猜到底挪了多少。
                if (t % 32 == 0 && d[0] > 0) {
                    std::printf("      [真实(%+d,%+d)] 投影估 dy=%+d(余%.3f) dx=%+d(余%.3f)\n",
                                d[0] * t, d[1] * t, eh.shift, eh.margin, ev.shift, ev.margin);
                }

                // 粗对齐只在"精对齐打不准"时才该出手。
                // 精对齐能修的时候让粗对齐插一脚，等于用 1D 的粗估
                // 去覆盖 2D 的精确结果，纯属倒退。
                bool needCoarse = true;
                {
                    cv::Mat s32;
                    sig.convertTo(s32, CV_32F);
                    cv::Mat rC, sC;
                    ref32(inkArea).copyTo(rC);
                    s32(inkArea).copyTo(sC);
                    const cv::Point2d probeSh = cv::phaseCorrelate(sC, rC, cv::noArray());
                    needCoarse = !(std::abs(probeSh.x) < 60 && std::abs(probeSh.y) < 60);
                }
                if (!needCoarse) {
                    cx = 0;
                    cy = 0;
                }
                if (cy != 0 || cx != 0) {
                    // 方向语义已用 tools/warpsem.cpp 实测确认：
                    //   M=[1,0,-s] 加 WARP_INVERSE_MAP  ==  平移 +s（沿正方向走）
                    // 所以"把多走的 cx 拉回来"要写成 +cx（配 INVERSE 后等效 -cx）。
                    // 这里原先写 -cx，等效于又往前走一份，位移被放大一倍。
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, cx, 0, 1, cy);
                    cv::Mat al;
                    cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE);
                    sig = al;
                    for (auto& u : units) u.bbox -= cv::Point(cx, cy);
                    ++applied;
                }
            }

            // ---- 精对齐：相位相关（字符区取窗）
            {
                cv::Mat s32;
                sig.convertTo(s32, CV_32F);
                cv::Mat rC, sC;
                ref32(inkArea).copyTo(rC);
                s32(inkArea).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat al;
                    cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE);
                    sig = al;
                    // 图像挪了，字符框必须同步挪，否则框和字直接错位。
                    // 粗对齐两处都改了，这里漏改过——64px 时估计明明是对的
                    // (+65 vs +64)、粗对齐也确实生效，结果还是 46 条误报。
                    for (auto& u : units) u.bbox -= cv::Point(cvRound(sh.x), cvRound(sh.y));
                }
            }

            // ---- 逐字匹配
            for (auto& u : units) {
                const cv::Rect box = u.bbox & full;
                if (box.width <= 4 || box.height <= 4) continue;
                constexpr int kS = 12;
                cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
                sr &= full;
                if (sr.width < box.width || sr.height < box.height) continue;
                cv::Mat mS, tS;
                model.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
                sig(sr).convertTo(tS, CV_32F);
                cv::Mat res;
                cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)),
                                  res, cv::TM_CCOEFF_NORMED);
                cv::Point ml; double mv = 0;
                cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
                if (mv < 0.30) continue;
                const cv::Rect moved(box.x + ml.x - kS, box.y + ml.y - kS, box.width, box.height);
                if ((moved & full) != moved) continue;
                u.bbox = moved;
            }

            const auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp);
            for (const auto& h : hits) {
                worstS = std::max(worstS, static_cast<int>(h.composite));
                worstB = std::max(worstB, h.band);
            }
            worstCount = std::max(worstCount, static_cast<int>(hits.size()));
        }
        std::printf("%6d %8d %8d %8d %10d%s\n", t, worstCount, worstS, worstB, applied,
                    (worstCount > 0 && firstBad < 0) ? "  <- 首次误报" : "");
        if (worstCount > 0 && firstBad < 0) firstBad = t;
    }
    std::printf("\n结论：%s 模式下 <= %d px 无误报，首次误报 %d px\n",
                useCoarse ? "粗+精" : "仅精", firstBad < 0 ? maxShift : firstBad - 8,
                firstBad < 0 ? -1 : firstBad);
    return 0;
}
