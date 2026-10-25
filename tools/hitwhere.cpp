// detectOne 的逐命中坐标复刻器
//
// 为什么要有它：批量 CSV 只报"最严重的那一条"的 S/跨字/深度，
// 没有坐标。分析误报必须知道那条线落在哪、离字符框多远，
// 否则只能对着 S 猜。本工具复刻 batchcore.hpp::detectOne 的完整流程
// （strokeSignal → ref 全局配准 → 逐字符 bbox 平移 → findScratches），
// 然后把每一条命中连同坐标、与最近字符框的距离一起打出来。
//
// 关键点：detectOne **只移动 u.bbox，不重采样 signal 像素**
// （batchcore.hpp:163-176）。所以逐个命中要看它压在哪个框上，
// 以及该框的原位置和移动量差多少。
//
// 用法: markloc/hitwhere <modelDir> <image>
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_hitwhere <modelDir> <image>\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("读取失败或尺寸不符\n");
        return 1;
    }

    std::vector<chardet::CharUnit> base;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) base.push_back(std::move(x));
    }

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // ---- 与 detectOne 逐行对齐：ref 参照整图配准
    cv::Mat r32, s32;
    model.ref.convertTo(r32, CV_32F);
    sig.convertTo(s32, CV_32F);
    cv::Rect inkArea;
    for (const auto& u : base) {
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
    s32(inkArea).copyTo(sC);
    const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
    std::printf("全局配准 (ref): (%+.2f, %+.2f)\n", sh.x, sh.y);
    if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
        (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
        cv::Mat al;
        cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
        sig = al;
    }

    // ---- 逐字符平移（只动 bbox，不动像素）—— 与 batchcore.hpp:143-177 一致
    std::vector<chardet::CharUnit> units = base;
    std::printf("\n=== 逐字符对齐（只动框不动像素）===\n");
    std::printf("%-6s %-22s %-22s %6s %6s %6s\n", "index", "原框", "移动后", "dx", "dy", "mVal");
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
        const int dx = mLoc.x - kSearch, dy = mLoc.y - kSearch;
        std::printf("%-6d %-22s ", u.index, cv::format("(%d,%d %dx%d)", box.x, box.y, box.width,
                                                       box.height).c_str());
        if (mVal < 0.30) {
            std::printf("%-22s %6s %6s %6.2f  [放弃]\n", "-", "-", "-", mVal);
            continue;
        }
        const cv::Rect moved(box.x + dx, box.y + dy, box.width, box.height);
        std::printf("%-22s %+6d %+6d %6.2f\n",
                    cv::format("(%d,%d %dx%d)", moved.x, moved.y, moved.width, moved.height).c_str(),
                    dx, dy, mVal);
        u.bbox = moved;
    }

    // ---- 找划痕
    markloc::ScratchParams P;
    markloc::ScratchStats ss;
    const auto hits = markloc::findScratches(sig, model.freq, units, cfg, P, &ss);
    std::printf("\n候选=%d 长度挡=%d 覆盖挡=%d 严重度挡=%d 最好=%.0f 命中=%zu\n", ss.candidates,
                ss.belowLen, ss.belowCov, ss.belowComposite, ss.bestComposite, hits.size());

    std::printf("\n=== 逐命中 ===\n");
    std::printf("%-7s %-5s %-5s %-5s %-6s %-24s %s\n", "S", "跨字", "深", "长", "框内?", "线段端点",
                "压住的字符框");
    for (const auto& h : hits) {
        // 这条线扫过哪些字框（findScratches 用同一判据：线段 bbox 与框相交）
        const cv::Rect seg = h.box;
        std::string which;
        int nIn = 0;
        for (const auto& u : units) {
            if ((u.bbox & seg).area() <= 0) continue;
            ++nIn;
            which += cv::format("#%d ", u.index);
        }
        std::printf("%-7.0f %-5d %-5.0f %-5.0f %-6d %-24s %s\n", h.composite, h.charsHit, h.depth,
                    h.len, nIn,
                    cv::format("(%d,%d)-(%d,%d)", h.a1.x, h.a1.y, h.a2.x, h.a2.y).c_str(),
                    which.c_str());
    }

    // ---- 全部候选（含未过门槛的），看"差一点"的那条在哪
    std::printf("\n=== 过长度门槛的候选中，严重度前 8（不论是否过门槛）===\n");
    printf("（略：findScratches 不返回未过门槛的明细，需要时改 scratch.hpp 加 out 参数）\n");
    return 0;
}
