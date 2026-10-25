// 划痕判定一致性诊断
//
// 目的：解释"交互式检出 10 处、批量检出 1 处"这个矛盾。
// 做法：在同一进程里用与两条路径完全相同的方式各跑一遍，逐条列出差异。
#include <cstdio>
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

static void report(const char* tag, const std::vector<markloc::ScratchHit>& hits) {
    std::printf("%s: %zu 条\n", tag, hits.size());
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& h = hits[i];
        std::printf("   [%zu] S=%4d band=%d 长=%.0f 深=%.0f 跨=%d 框=(%d,%d,%d,%d)\n", i,
                    static_cast<int>(h.composite), h.band, h.len, h.depth, h.charsHit, h.box.x,
                    h.box.y, h.box.width, h.box.height);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: scratchcmp2 <modelDir> <image>\n");
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
    const chardet::Config cfg = model.config;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    markloc::ScratchParams sp;

    // ---- 路线 A：批处理（batchcore.hpp 的写法）
    {
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
        std::printf("A 批处理 配准: (%+.2f, %+.2f)\n", sh.x, sh.y);
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
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
            cv::Point ml; double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
            if (mv < 0.30) continue;
            const cv::Rect moved(box.x + ml.x - kS, box.y + ml.y - kS, box.width, box.height);
            if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
            u.bbox = moved;
        }
        report("A 批处理", markloc::findScratches(sig, model.freq, units, cfg, sp));
    }

    // ---- 路线 B：交互式（DefectViewWindow::runDetect 的写法，含亚像素插值）
    {
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
        std::printf("B 交互式 配准: (%+.2f, %+.2f)\n", sh.x, sh.y);
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
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
            cv::Point ml; double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
            if (mv < 0.30) continue;
            double subX = 0, subY = 0;
            if (ml.x > 0 && ml.x + 1 < res.cols) {
                const double l = res.at<float>(ml.y, ml.x - 1);
                const double c = res.at<float>(ml.y, ml.x);
                const double r2 = res.at<float>(ml.y, ml.x + 1);
                const double den = 2.0 * c - l - r2;
                if (std::abs(den) > 1e-9) subX = std::clamp(0.5 * (r2 - l) / den, -1.0, 1.0);
            }
            if (ml.y > 0 && ml.y + 1 < res.rows) {
                const double u2 = res.at<float>(ml.y - 1, ml.x);
                const double c = res.at<float>(ml.y, ml.x);
                const double d = res.at<float>(ml.y + 1, ml.x);
                const double den = 2.0 * c - u2 - d;
                if (std::abs(den) > 1e-9) subY = std::clamp(0.5 * (d - u2) / den, -1.0, 1.0);
            }
            const int dx = static_cast<int>(std::lround(ml.x - kS + subX));
            const int dy = static_cast<int>(std::lround(ml.y - kS + subY));
            const cv::Rect moved(box.x + dx, box.y + dy, box.width, box.height);
            if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
            u.bbox = moved;
        }
        report("B 交互式", markloc::findScratches(sig, model.freq, units, cfg, sp));
    }
    return 0;
}
