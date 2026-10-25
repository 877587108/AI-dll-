// 字符位移边界探针
//
// 问题：字符对齐算法到底能兜住多大的位移？
// 做法：拿一张合格样图，整幅平移指定像素后重新检测，看结论是否稳定。
//
// 平移分两个方向各测一遍：
//   · 全图平移（模拟来料位置偏）—— 相位相关 + 逐字匹配两级兜底
//   · 仅字形区域平移（模拟打标机字位漂移）—— 只有逐字匹配能救
// 判据：一张合格图挪了位置之后仍然判合格，才算对齐算法兜住了。
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// 整体平移：内容移动到 (+dx,+dy)，空出来的地方用边缘像素填充。
static cv::Mat shiftImage(const cv::Mat& src, int dx, int dy) {
    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::Mat out;
    cv::warpAffine(src, out, M, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

// 只平移字形区域，模拟打标字位漂移。
//
// 注意：不能把"所有字框并集"整块搬走——那块区域里除了字还有大片背景，
// 整块涂平会连字一起抹掉，结果就是一张空图，探针拿它测什么都是"零误报"。
// 必须逐个字符框搬运，每个框单独清底、单独贴到新位置。
static cv::Mat shiftInkOnly(const cv::Mat& src, const std::vector<chardet::CharUnit>& units,
                            int dx, int dy) {
    cv::Mat out = src.clone();
    // 先把所有字符内容抠出来，再统一清底，最后贴回去，避免互相覆盖。
    std::vector<std::pair<cv::Rect, cv::Mat>> pieces;
    for (const auto& u : units) {
        const cv::Rect box = u.bbox & cv::Rect(0, 0, src.cols, src.rows);
        if (box.width <= 0 || box.height <= 0) continue;
        pieces.emplace_back(box, src(box).clone());
    }
    double bgSum = 0;
    int bgN = 0;
    for (const auto& p : pieces) {
        // 取框外一圈作为背景亮度参考
        const cv::Rect ring = (cv::Rect(p.first.x - 2, p.first.y - 2, p.first.width + 4,
                                        p.first.height + 4) &
                               cv::Rect(0, 0, src.cols, src.rows));
        const cv::Rect inner = p.first;
        for (int y = ring.y; y < ring.y + ring.height; ++y) {
            for (int x = ring.x; x < ring.x + ring.width; ++x) {
                if (inner.contains(cv::Point(x, y))) continue;
                bgSum += src.at<unsigned char>(y, x);
                ++bgN;
            }
        }
    }
    const double bg = bgN > 0 ? bgSum / bgN : 0.0;
    for (const auto& p : pieces) cv::rectangle(out, p.first, cv::Scalar(bg), cv::FILLED);
    for (const auto& p : pieces) {
        const cv::Rect dst = (p.first + cv::Point(dx, dy)) & cv::Rect(0, 0, src.cols, src.rows);
        if (dst.width <= 0 || dst.height <= 0) continue;
        // dst 相对 p.first 的偏移再减掉本次平移量，才是源块里的起点。
        const cv::Rect take(dst.x - p.first.x - dx, dst.y - p.first.y - dy, dst.width,
                            dst.height);
        if ((take & cv::Rect(0, 0, p.second.cols, p.second.rows)) != take) continue;
        p.second(take).copyTo(out(dst));
    }
    return out;
}

struct Probe {
    int shift = 0;
    int count = -1;
    int worstS = 0;
    int worstBand = 0;
    float alignErr = 0;   // 对齐后框心与真实字形的偏差
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: alignlimit <modelDir> <image> [mode=full|ink] [maxShift=80]\n");
        return 2;
    }
    const std::string mode = (argc > 3) ? argv[3] : "full";
    const int maxShift = (argc > 4) ? std::atoi(argv[4]) : 80;

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
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);
    cv::Rect ink;
    for (const auto& u : base) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, gray.cols, gray.rows);
        if (b.width > 0 && b.height > 0) ink = ink.area() > 0 ? (ink | b) : b;
    }
    std::printf("模型=%s 图=%s 模式=%s 字形区=(%d,%d,%d,%d)\n", argv[1], argv[2], mode.c_str(),
                ink.x, ink.y, ink.width, ink.height);

    // 自检：把平移图导出来，肉眼确认字形只在一处，没有重影/残影。
    // 生成器有 bug 时探针会给出"误报"，那是在测我自己的错，不是测算法。
    if (const char* dumpDir = std::getenv("ALIGNLIMIT_DUMP")) {
        for (const int s : {0, 8, 16, 20, 32}) {
            const cv::Mat probe = (mode == "ink") ? shiftInkOnly(gray, base, s, 0)
                                                  : shiftImage(gray, s, s);
            char name[512];
            std::snprintf(name, sizeof(name), "%s\\%s_%02d.png", dumpDir, mode.c_str(), s);
            const int cx0 = std::max(0, ink.x - 120);
            const int cy0 = std::max(0, ink.y - 120);
            const int cx1 = std::min(probe.cols, ink.x + ink.width + 120);
            const int cy1 = std::min(probe.rows, ink.y + ink.height + 120);
            if (cx1 <= cx0 || cy1 <= cy0) continue;
            const cv::Rect crop(cx0, cy0, cx1 - cx0, cy1 - cy0);
            cv::imwrite(name, probe(crop));
        }
        std::printf("自检图已导出到 %s\n", dumpDir);
    }

    std::printf("%6s %8s %8s %8s %10s\n", "位移", "划痕数", "最大S", "档位", "对齐残差");

    markloc::ScratchParams sp;
    int firstBad = -1;
    for (int s = 0; s <= maxShift; s += 4) {
        // 四个方向都测，取最坏的那个
        const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        Probe worst;
        worst.count = 0;
        for (const auto& d : dirs) {
            const int dx = d[0] * s, dy = d[1] * s;
            const cv::Mat img = (mode == "ink") ? shiftInkOnly(gray, base, dx, dy)
                                                : shiftImage(gray, dx, dy);
            cv::Mat sig = chardet::strokeSignal(img, cv::Rect(0, 0, img.cols, img.rows), cfg, false);
            auto units = base;
            for (auto& u : units) u.bbox += cv::Point(dx, dy);

            // 相位相关：参照图（母版）和样图在同一窗口取窗。
            // 样图被平移了 (dx,dy)，所以内容现在位于 ink+(dx,dy)；
            // 参照图内容仍在 ink。两者取同一个"内容本来应在"的窗口即可对齐。
            cv::Rect window(ink.x - 20, ink.y - 20, ink.width + 40, ink.height + 40);
            window &= cv::Rect(0, 0, sig.cols, sig.rows);
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            if (window.area() > 0) {
                cv::Mat rC, sC;
                r32(window).copyTo(rC);
                s32(window).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                // 相位相关的可信半径。产品代码写死 60；这里留一个旋钮，
                // 用来分别测「相位相关」和「逐字匹配」各自的上限。
                const double pcLimit = std::getenv("ALIGNLIMIT_PC") ?
                                           std::atof(std::getenv("ALIGNLIMIT_PC")) : 60.0;
                // sh 就是"样图相对参照图多走了多少"，对准需要移回去 -sh。
                if (std::abs(sh.x) < pcLimit && std::abs(sh.y) < pcLimit &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat al;
                    cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE);
                    sig = al;
                    // 图像被拉回参照系，字符框也要跟着回到参照系。
                    for (auto& u : units) u.bbox -= cv::Point(cvRound(sh.x), cvRound(sh.y));
                }
            }
            // 逐字匹配（±12 搜索半径）
            float maxErr = 0;
            for (auto& u : units) {
                const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (box.width <= 4 || box.height <= 4) continue;
                constexpr int kS = 12;
                cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
                sr &= cv::Rect(0, 0, sig.cols, sig.rows);
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
                u.bbox += cv::Point(ml.x - kS, ml.y - kS);
                maxErr = std::max(maxErr, static_cast<float>(std::abs(ml.x - kS - dx)) +
                                              std::abs(ml.y - kS - dy));
            }
            const auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp);
            int worstS = 0, worstB = 0;
            for (const auto& h : hits) {
                worstS = std::max(worstS, static_cast<int>(h.composite));
                worstB = std::max(worstB, h.band);
            }
            if (static_cast<int>(hits.size()) > worst.count || worstS > worst.worstS) {
                worst.count = static_cast<int>(hits.size());
                worst.worstS = worstS;
                worst.worstBand = worstB;
                worst.alignErr = maxErr;
            }
        }
        std::printf("%6d %8d %8d %8d %10.1f%s\n", s, worst.count, worst.worstS, worst.worstBand,
                    worst.alignErr, (worst.count > 0 && firstBad < 0) ? "  <- 首次误报" : "");
        if (worst.count > 0 && firstBad < 0) firstBad = s;
    }
    std::printf("\n结论：在 %s 模式下，位移 <= %d px 时无误报；首次误报出现在 %d px。\n",
                mode.c_str(), (firstBad < 0 ? maxShift : firstBad - 4),
                firstBad < 0 ? -1 : firstBad);
    return 0;
}
