// 0109 误报溯源
//
// 目的：0109 在数据集里放在 ok/，判定却是 NG（RT: S=1938 确定，FT: S=1187 可疑）。
// 要分清两件事：
//   1) 图上真有划痕，那就是数据集标错了，模型没错；
//   2) 图上是别的东西被当成划痕，那就是检出的问题。
//
// 做法：跑完整检出流程，把每个命中框单独裁出来放大存盘，
// 同时在整图上画框，便于一眼看出它们落在哪里。
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

static bool writeImage(const std::string& path, const cv::Mat& img) {
    std::vector<unsigned char> buf;
    const std::string ext = ".png";
    if (!cv::imencode(ext, img, buf)) return false;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::fwrite(buf.data(), 1, buf.size(), fp);
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: scratchtrace <modelDir> <image> <outPrefix>\n");
        return 2;
    }
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
    const std::string prefix = argv[3];
    const chardet::Config cfg = model.config;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);
    markloc::ScratchParams sp;

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat r32, s32;
    model.ref.convertTo(r32, CV_32F);
    sig.convertTo(s32, CV_32F);
    cv::Rect ink;
    for (const auto& u : base) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width > 0 && b.height > 0) ink = ink.area() > 0 ? (ink | b) : b;
    }
    ink.x = std::max(0, ink.x - 20);
    ink.y = std::max(0, ink.y - 20);
    ink.width = std::min(sig.cols - ink.x, ink.width + 40);
    ink.height = std::min(sig.rows - ink.y, ink.height + 40);
    cv::Mat rC, sC;
    r32(ink).copyTo(rC);
    s32(ink).copyTo(sC);
    const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
    std::printf("配准: (%+.2f, %+.2f)\n", sh.x, sh.y);
    // SCRATCHTRACE_NOGLOBAL=1 跳过全局平移，只做逐字对齐。
    // 用来判定"全局配准把信号推歪了"这个假设：若跳过之后命中数骤降，
    // 说明问题在全局平移而非逐字对齐。
    const bool noGlobal = std::getenv("SCRATCHTRACE_NOGLOBAL") != nullptr;
    if (noGlobal) {
        std::printf("  [跳过全局平移]\n");
    } else if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
        cv::Mat al;
        cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
        sig = al;
    }
    std::vector<chardet::CharUnit> units = base;
    for (auto& u : units) {
        cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
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
        cv::Point ml;
        double mv = 0;
        cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
        if (mv < 0.30) continue;
        const cv::Rect moved(box.x + ml.x - kS, box.y + ml.y - kS, box.width, box.height);
        if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
        u.bbox = moved;
    }

    const auto hits = markloc::findScratches(sig, model.freq, units, cfg, sp);
    std::printf("命中 %zu 条\n", hits.size());

    // 整图：画所有命中框 + 所有字符框
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto& u : units)
        cv::rectangle(vis, u.bbox, cv::Scalar(0, 200, 0), 1);   // 绿=字符框
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        cv::rectangle(vis, h.box, cv::Scalar(0, 0, 255), 2);    // 红=划痕命中
        char lbl[32];
        std::snprintf(lbl, sizeof(lbl), "#%d S=%d", static_cast<int>(i),
                      static_cast<int>(h.composite));
        cv::putText(vis, lbl, cv::Point(h.box.x, std::max(14, h.box.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    writeImage(prefix + "_all.png", vis);

    // 逐条：从原图裁出命中框（外扩 40px），放大 3 倍，旁边贴一联信号图
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        cv::Rect c = h.box;
        c.x -= 40;
        c.y -= 40;
        c.width += 80;
        c.height += 80;
        c &= cv::Rect(0, 0, gray.cols, gray.rows);
        if (c.width < 8 || c.height < 8) continue;

        cv::Mat cropRaw = gray(c);
        cv::Mat cropSig = sig(c);
        cv::Mat sig8;
        cv::normalize(sig(c), sig8, 0, 255, cv::NORM_MINMAX);
        sig8.convertTo(sig8, CV_8U);

        // 在裁图上框出命中区域（相对坐标）
        cv::Mat visRaw, visSig;
        cv::cvtColor(cropRaw, visRaw, cv::COLOR_GRAY2BGR);
        cv::cvtColor(sig8, visSig, cv::COLOR_GRAY2BGR);
        const cv::Rect rel(h.box.x - c.x, h.box.y - c.y, h.box.width, h.box.height);
        cv::rectangle(visRaw, rel, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(visSig, rel, cv::Scalar(0, 0, 255), 1);

        cv::Mat side;
        cv::hconcat(visRaw, visSig, side);
        cv::Mat big;
        cv::resize(side, big, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);

        char nm[512];
        std::snprintf(nm, sizeof(nm), "%s_hit%d.png", prefix.c_str(), static_cast<int>(i));
        writeImage(nm, big);
        std::printf("  #%d 框=(%d,%d,%d,%d) S=%d 深=%.0f 长=%.0f 跨=%d -> %s\n",
                    static_cast<int>(i), h.box.x, h.box.y, h.box.width, h.box.height,
                    static_cast<int>(h.composite), h.depth, h.len, h.charsHit, nm);
    }

    // 每个命中的信号统计：峰值、均值，以及与 strokeHi / 弱阈值的相对关系。
    //
    // 若峰值普遍只有 strokeHi 的一小半，说明这些"划痕"其实是笔画软边缘
    // 穿过弱阈值留下的像素，而不是真正的亮线。
    std::printf("\n--- 命中框内信号统计 (strokeHi=%.3f 弱阈值=%.3f) ---\n", cfg.strokeHi,
                cfg.missStrokeLo * 0.7);
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        const cv::Rect b = h.box & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width < 2 || b.height < 2) continue;
        double mn = 0, mx = 0;
        cv::minMaxLoc(sig(b), &mn, &mx);
        const double mean = cv::mean(sig(b))[0];
        std::printf("  #%d  峰=%.3f 均=%.3f  峰/strokeHi=%.2f  框内像素%%>弱阈值=%.1f%%\n",
                    static_cast<int>(i), mx, mean, mx / std::max(1e-6, cfg.strokeHi),
                    100.0 * cv::countNonZero(sig(b) > cfg.missStrokeLo * 0.7) / b.area());
    }
    // 模型频率图在命中框内的取值，以及"模型认为是墨"的像素比例。
    //
    // 这是判定的核心：weak = 信号>弱阈值 且 NOT(模型频率>blankFreq*255 再膨胀)。
    // 若命中框内 模型墨迹占比接近 0，说明模型没说这里有笔画，
    // 于是笔画被原样留进 weak，最后被当成划痕报出来。
    std::printf("\n--- 命中框内模型频率统计 (blankFreq*255=%d) ---\n",
                static_cast<int>(cfg.blankFreq * 255));
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        const cv::Rect b = h.box & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width < 2 || b.height < 2) continue;
        const cv::Mat f = model.freq(b);
        double fmn = 0, fmx = 0;
        cv::minMaxLoc(f, &fmn, &fmx);
        const double fmean = cv::mean(f)[0];
        const int inkPct =
            100 * cv::countNonZero(f > static_cast<int>(cfg.blankFreq * 255)) / b.area();
        std::printf("  #%d  freq峰=%.0f 均=%.0f  模型墨迹占比=%d%%\n", static_cast<int>(i), fmx,
                    fmean, inkPct);
    }
    return 0;
}
