// warpsign.cpp —— 逐字符平移的方向和符号到底该是哪个
//
// 上一轮 perCharWarp 让误报 1->8，怀疑是把字形搬反了方向。
// 不再猜：把四种组合都跑一遍，看哪一种让命中数下降。
//
// 四种组合 = {warpAffine 正向, WARP_INVERSE_MAP} x {用 (dx,dy), 用 (-dx,-dy)}
//
// 判读：0109_RT 是误报件，命中数应当越少越好；
//       0026_FT 是真缺陷，命中数不能掉。
//       两者同时满足的组合才是对的。
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

struct Cfg {
    bool inverseMap;
    int sign;
    const char* name;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_warpsign <modelDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    const Cfg cfgs[] = {
        {true, +1, "INVERSE_MAP, 用 (+dx,+dy)"},
        {true, -1, "INVERSE_MAP, 用 (-dx,-dy)"},
        {false, +1, "正向(无标志), 用 (+dx,+dy)"},
        {false, -1, "正向(无标志), 用 (-dx,-dy)"},
    };

    for (int ai = 2; ai < argc; ++ai) {
        cv::Mat gray = readTif(argv[ai]);
        if (gray.empty() || gray.size() != model.freq.size()) continue;
        std::printf("\n===== %s =====\n", fs::path(argv[ai]).filename().string().c_str());

        cv::Mat sig0 =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        std::vector<chardet::CharUnit> units0 = baseUnits;
        if (!model.ref.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig0.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : units0) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig0.cols, sig0.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            inkArea.x = std::max(0, inkArea.x - 20);
            inkArea.y = std::max(0, inkArea.y - 20);
            inkArea.width = std::min(sig0.cols - inkArea.x, inkArea.width + 40);
            inkArea.height = std::min(sig0.rows - inkArea.y, inkArea.height + 40);
            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            sig0(inkArea).convertTo(sC, CV_32F);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::Mat aligned;
                cv::warpAffine(sig0, aligned, M, sig0.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                sig0 = aligned;
            }
        }

        for (const auto& c : cfgs) {
            cv::Mat sig = sig0.clone();
            std::vector<chardet::CharUnit> units = baseUnits;

            // 逐字符求位移（与推理侧一致）
            std::vector<cv::Point> sh(units.size(), cv::Point(0, 0));
            std::vector<char> ok(units.size(), 0);
            for (size_t ui = 0; ui < units.size(); ++ui) {
                const cv::Rect box = units[ui].bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (box.width <= 4 || box.height <= 4) continue;
                constexpr int kS = 12;
                cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
                sr &= cv::Rect(0, 0, sig.cols, sig.rows);
                if (sr.width < box.width || sr.height < box.height) continue;
                cv::Mat mS, tS;
                model.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
                sig(sr).convertTo(tS, CV_32F);
                cv::Mat res;
                cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width,
                                                  box.height)),
                                  res, cv::TM_CCOEFF_NORMED);
                cv::Point ml;
                double mv = 0;
                cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
                if (mv < 0.45) continue;
                sh[ui] = cv::Point(ml.x - kS, ml.y - kS);
                ok[ui] = 1;
            }

            // 按配置搬像素
            cv::Mat warped = sig.clone();
            for (size_t ui = 0; ui < units.size(); ++ui) {
                if (!ok[ui]) continue;
                const cv::Point s = sh[ui] * c.sign;
                if (s.x == 0 && s.y == 0) continue;
                if (std::abs(s.x) > 14 || std::abs(s.y) > 14) continue;
                cv::Rect box = units[ui].bbox;
                box.x -= 6;
                box.y -= 6;
                box.width += 12;
                box.height += 12;
                box &= cv::Rect(0, 0, sig.cols, sig.rows);
                if (box.width <= 2 || box.height <= 2) continue;
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, s.x, 0, 1, s.y);
                cv::Mat piece;
                if (c.inverseMap)
                    cv::warpAffine(sig(box), piece, M, box.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REFLECT);
                else
                    cv::warpAffine(sig(box), piece, M, box.size(), cv::INTER_LINEAR,
                                   cv::BORDER_REFLECT);
                piece.copyTo(warped(box));
                units[ui].bbox.x += s.x;
                units[ui].bbox.y += s.y;
            }

            markloc::ScratchParams sp;
            markloc::ScratchStats st;
            auto hits = markloc::findScratches(warped, model.freq, units, cfg, sp, &st);
            int best = 0;
            for (const auto& h : hits) best = std::max(best, (int)h.composite);
            std::printf("  %-28s 命中 %2d 条  最高S %5d\n", c.name, (int)hits.size(), best);
        }

        // 基线：不搬
        std::vector<chardet::CharUnit> units = baseUnits;
        for (auto& u : units) {
            const cv::Rect box = u.bbox & cv::Rect(0, 0, sig0.cols, sig0.rows);
            if (box.width <= 4 || box.height <= 4) continue;
            constexpr int kS = 12;
            cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
            sr &= cv::Rect(0, 0, sig0.cols, sig0.rows);
            if (sr.width < box.width || sr.height < box.height) continue;
            cv::Mat mS, tS;
            model.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig0(sr).convertTo(tS, CV_32F);
            cv::Mat res;
            cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)),
                              res, cv::TM_CCOEFF_NORMED);
            cv::Point ml;
            double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
            if (mv < 0.30) continue;
            const cv::Rect moved(box.x + ml.x - kS, box.y + ml.y - kS, box.width, box.height);
            if ((moved & cv::Rect(0, 0, sig0.cols, sig0.rows)) != moved) continue;
            u.bbox = moved;
        }
        markloc::ScratchParams sp;
        markloc::ScratchStats st;
        auto hits = markloc::findScratches(sig0, model.freq, units, cfg, sp, &st);
        int best = 0;
        for (const auto& h : hits) best = std::max(best, (int)h.composite);
        std::printf("  %-28s 命中 %2d 条  最高S %5d\n", "【基线】只挪框不挪像素", (int)hits.size(),
                    best);
    }
    return 0;
}
