// 跨越字数判据的对比：bbox 相交 vs 线段真跨字
//
// 现状（src/gui/scratch.hpp:304-312）：
//   把线段两端点围成的**轴对齐矩形**去和每个字符框求交，数相交个数。
// 问题一：矩形不是线段。一条竖直的短线段，bbox 只有几十像素宽，
//         但只要它跨过相邻两个字的上下方向，bbox 就能同时盖住两个字。
// 问题二：重复计数。region1 的 #6 在"压住的字符框"里出现了两次——
//         因为 toAllUnits 生成的是 22 个单元而不是 4 个区域 22 个字，
//         index 只在区域内唯一，同一 index 会被不同区域复用。
//         实测 0109_RT 那条 S=1938 的线 charsHit=4，压住的却是 #5 #6 #6 #7，
//         即 3 个不同的字被算成 4 个。
//
// 本工具对同一条候选线并排算三种口径，供判断改哪一种、会改掉多少：
//   A 现状：线段 bbox ∩ 字符框，个数（会有重复）
//   B 去重：同上，但按 (区域,index) 去重
//   C 真跨：线段真正穿过的字符框，去重
// 并给出每种口径下的严重度，以及跨档位的差异。
//
// 用法: markloc/crossrule <modelDir> <image>...
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

namespace fs = std::filesystem;

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

/// 线段是否真的穿过矩形：逐点采样 + Liang-Barsky 裁剪
static bool segmentCrossesRect(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Rect& r) {
    // Liang-Barsky
    double t0 = 0.0, t1 = 1.0;
    const double dx = p2.x - p1.x, dy = p2.y - p1.y;
    const double p[4] = {-dx, dx, -dy, dy};
    const double q[4] = {p1.x - r.x, r.x + r.width - p1.x, p1.y - r.y, r.y + r.height - p1.y};
    for (int i = 0; i < 4; ++i) {
        if (std::abs(p[i]) < 1e-12) {
            if (q[i] < 0) return false;   // 平行且在外
            continue;
        }
        const double t = q[i] / p[i];
        if (p[i] < 0) {
            if (t > t1) return false;
            if (t > t0) t0 = t;
        } else {
            if (t < t0) return false;
            if (t < t1) t1 = t;
        }
    }
    // 要求真正穿过一段长度，而不是擦一个角
    return (t1 - t0) * cv::norm(p2 - p1) >= 2.0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_crossrule <modelDir> <image>...\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    markloc::ScratchParams P;

    for (int a = 2; a < argc; ++a) {
        cv::Mat gray;
        if (!readGray(argv[a], gray) || gray.size() != model.freq.size()) {
            std::printf("%s: 读取失败\n", argv[a]);
            continue;
        }
        std::vector<chardet::CharUnit> units;
        std::vector<int> unitRegion;   // 平行表：CharUnit 没有 region 字段
        for (size_t ri = 0; ri < model.regions.size(); ++ri) {
            auto u = chardet::toUnits(model, static_cast<int>(ri));
            for (auto& x : u) {
                units.push_back(std::move(x));
                unitRegion.push_back(static_cast<int>(ri));
            }
        }
        cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

        // ref 配准（与 detectOne 一致）
        cv::Mat r32;
        model.ref.convertTo(r32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : units) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.area() > 0) inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
        }
        inkArea.x = std::max(0, inkArea.x - 20);
        inkArea.y = std::max(0, inkArea.y - 20);
        inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
        inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
        cv::Mat sC, rC;
        sig(inkArea).copyTo(sC);
        r32(inkArea).copyTo(rC);
        const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat al;
            cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            sig = al;
        }
        // 逐字符平移框（与 detectOne 一致）
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

        const auto hits = markloc::findScratches(sig, model.freq, units, cfg, P);
        if (hits.empty()) {
            std::printf("\n%s: 无命中\n", fs::path(argv[a]).stem().string().c_str());
            continue;
        }
        std::printf("\n=== %s (%zu 条) ===\n", fs::path(argv[a]).stem().string().c_str(),
                    hits.size());
        std::printf("%-22s %-6s %5s %5s %5s %8s %8s %8s\n", "线段", "深度", "A现状", "B去重",
                    "C真跨", "S_A", "S_B", "S_C");
        int hitA = 0, hitB = 0, hitC = 0;
        for (const auto& h : hits) {
            const cv::Point2d p1(h.a1.x, h.a1.y), p2(h.a2.x, h.a2.y);
            const cv::Rect seg(std::min(h.a1.x, h.a2.x), std::min(h.a1.y, h.a2.y),
                               std::abs(h.a2.x - h.a1.x) + 1, std::abs(h.a2.y - h.a1.y) + 1);
            // A: bbox 相交计数（含重复），与 scratch.hpp 现状一致
            int ca = 0;
            std::vector<std::string> ids;
            for (size_t i = 0; i < units.size(); ++i) {
                if ((units[i].bbox & seg).area() <= 0) continue;
                ++ca;
                ids.push_back(cv::format("r%d#%d", unitRegion[i], units[i].index));
            }
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            const int cb = static_cast<int>(ids.size());
            // C: 线段真的穿过
            int cc = 0;
            for (const auto& u : units) {
                if (segmentCrossesRect(p1, p2, u.bbox)) ++cc;
            }
            const double sA = h.depth * ca, sB = h.depth * cb, sC = h.depth * cc;
            if (sA >= P.minComposite) ++hitA;
            if (sB >= P.minComposite) ++hitB;
            if (sC >= P.minComposite) ++hitC;
            std::printf("%-22s %-6.0f %5d %5d %5d %8.0f %8.0f %8.0f\n",
                        cv::format("(%d,%d)-(%d,%d)", h.a1.x, h.a1.y, h.a2.x, h.a2.y).c_str(),
                        h.depth, ca, cb, cc, sA, sB, sC);
        }
        std::printf("过门槛条数： A=现%d  B=去重%d  C=真跨%d\n", hitA, hitB, hitC);
    }
    return 0;
}
