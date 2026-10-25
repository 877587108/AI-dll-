// 字符框是否完整包住字符
//
// 之前的 boxdiag 用"离框心最近的连通域"算 recall，只覆盖一个字的一坨墨。
// 字符常由多笔组成（i 的点 + 竖、P 的竖 + 环），只取一坨会漏掉别的笔，
// 得出"框很准"的假结论。
//
// 这里换个更直接的判据：把属于该字符的所有墨迹取出来，求其外接矩形，
// 然后看这个外接矩形是否被字符框完整包含。
//   · 外接矩形完全在框内        -> 包住
//   · 有墨迹落在框边之外        -> 没包住，报出各边超出多少像素
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "chardet.hpp"

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
        std::printf("用法: boxwrap <modelDir> <image> [align 0/1]\n");
        return 2;
    }
    const int doAlign = (argc > 3) ? std::atoi(argv[3]) : 1;
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
    cv::Mat sig = chardet::strokeSignal(gray, full, cfg, false);
    cv::Mat ref32;
    model.ref.convertTo(ref32, CV_32F);

    // ---- 与产品一致的两级对齐
    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);
    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & full;
        if (b.width > 0 && b.height > 0) inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
    {
        cv::Mat s32;
        sig.convertTo(s32, CV_32F);
        cv::Mat rC, sC;
        ref32(inkArea).copyTo(rC);
        s32(inkArea).copyTo(sC);
        const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
        const double pcLimit = std::getenv("BOXWRAP_PC") ? std::atof(std::getenv("BOXWRAP_PC")) : 60.0;
        if (std::abs(sh.x) < pcLimit && std::abs(sh.y) < pcLimit &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat al;
            cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            sig = al;
            std::printf("整图配准: (%+.1f, %+.1f)\n", sh.x, sh.y);
        } else {
            std::printf("整图配准: 跳过 (%.1f, %.1f)\n", sh.x, sh.y);
        }
    }
    if (doAlign) {
        for (auto& u : units) {
            cv::Rect box = u.bbox & full;
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
            cv::Point ml;
            double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
            if (mv < 0.30) continue;
            const cv::Rect moved(box.x + ml.x - kS, box.y + ml.y - kS, box.width, box.height);
            if ((moved & full) != moved) continue;
            u.bbox = moved;
        }
    }

    // ---- 墨迹掩膜
    cv::Mat inkMask;
    cv::compare(sig, cfg.strokeHi, inkMask, cv::CMP_GT);
    const int inkTotal = cv::countNonZero(inkMask);
    if (inkTotal == 0) {
        std::printf("墨迹掩膜为空，无法判断\n");
        return 1;
    }
    std::printf("阈值=%.2f 全图墨点数=%d\n\n", cfg.strokeHi, inkTotal);

    // 全局连通域，一次算完
    cv::Mat labels, stats, centroids;
    const int nlab = cv::connectedComponentsWithStats(inkMask, labels, stats, centroids, 8, CV_32S);

    // 判定外扩量：字符框各边再多算 pad 像素。
    // 墨迹边缘本身有 1~3px 的扩散（描边、反光），把这种扩散算成"没包住"
    // 会淹没真正的问题。pad 可调，用来看结果对边缘扩散的敏感度。
    const int pad = std::getenv("BOXWRAP_PAD") ? std::atoi(std::getenv("BOXWRAP_PAD")) : 0;
    std::printf("判定外扩=%d px\n\n", pad);

    std::printf("%-4s %-16s %-16s %-16s %7s %7s\n", "#", "字符框", "字墨外接框", "外接框-字符框",
                "覆盖率", "判定");
    int nWrap = 0, nBad = 0, nTest = 0;
    double sumCover = 0;
    for (size_t i = 0; i < units.size(); ++i) {
        const cv::Rect box = units[i].bbox & full;
        if (box.width <= 4 || box.height <= 4) continue;

        // 收集"属于这个字符"的墨：质心落在框向外扩 50% 的范围内，且与框有交叠。
        // 用外扩范围而不是框本身，是为了把"跑到框外的那部分笔"也认回来——
        // 否则框外的墨会被当成别人的，永远测不出"没包住"。
        const int ex = std::max(10, box.width / 2), ey = std::max(10, box.height / 2);
        const cv::Rect wide = (cv::Rect(box.x - ex, box.y - ey, box.width + 2 * ex,
                                        box.height + 2 * ey) &
                               full);
        cv::Rect glyph;
        double glyphArea = 0;
        for (int li = 1; li < nlab; ++li) {
            const int area = stats.at<int>(li, cv::CC_STAT_AREA);
            if (area < 40) continue;
            const cv::Point2d c(centroids.at<double>(li, 0), centroids.at<double>(li, 1));
            if (!wide.contains(cv::Point(cvRound(c.x), cvRound(c.y)))) continue;
            cv::Rect comp(stats.at<int>(li, cv::CC_STAT_LEFT), stats.at<int>(li, cv::CC_STAT_TOP),
                          stats.at<int>(li, cv::CC_STAT_WIDTH),
                          stats.at<int>(li, cv::CC_STAT_HEIGHT));
            // 必须与字符框有交集，否则是隔壁的字
            if ((comp & box).area() <= 0) continue;
            glyph = glyph.area() > 0 ? (glyph | comp) : comp;
            glyphArea += area;
        }
        if (glyph.area() <= 0) continue;
        ++nTest;

        // 覆盖率：字形墨点里有多少落在框内
        double inBox = 0;
        for (int li = 1; li < nlab; ++li) {
            if (stats.at<int>(li, cv::CC_STAT_AREA) < 40) continue;
            const cv::Point2d c(centroids.at<double>(li, 0), centroids.at<double>(li, 1));
            if (!wide.contains(cv::Point(cvRound(c.x), cvRound(c.y)))) continue;
            cv::Rect comp(stats.at<int>(li, cv::CC_STAT_LEFT), stats.at<int>(li, cv::CC_STAT_TOP),
                          stats.at<int>(li, cv::CC_STAT_WIDTH),
                          stats.at<int>(li, cv::CC_STAT_HEIGHT));
            if ((comp & box).area() <= 0) continue;
            cv::Mat m = (labels == li);
            const double a = cv::countNonZero(m(box & full));
            inBox += a;
        }
        const double cover = glyphArea > 0 ? inBox / glyphArea : 0;
        sumCover += cover;

        const int overL = box.x - glyph.x;
        const int overT = box.y - glyph.y;
        const int overR = (glyph.x + glyph.width) - (box.x + box.width);
        const int overB = (glyph.y + glyph.height) - (box.y + box.height);
        // 外扩 pad 之后仍溢出，才算真的没包住
        const bool wrapped = overL <= pad && overT <= pad && overR <= pad && overB <= pad;
        if (wrapped) ++nWrap; else ++nBad;

        char clip[64];
        std::snprintf(clip, sizeof(clip), "L%d T%d R%d B%d", std::max(0, overL),
                      std::max(0, overT), std::max(0, overR), std::max(0, overB));
        std::printf("%-4zu %-16s %-16s %-16s %6.1f%% %7s\n", i,
                    (std::to_string(box.x) + "," + std::to_string(box.y) + "," +
                     std::to_string(box.width) + "," + std::to_string(box.height)).c_str(),
                    (std::to_string(glyph.x) + "," + std::to_string(glyph.y) + "," +
                     std::to_string(glyph.width) + "," + std::to_string(glyph.height)).c_str(),
                    clip, cover * 100.0, wrapped ? "包住" : "溢出");
    }
    std::printf("\n参与判定=%d  完整包住=%d  有溢出=%d  平均覆盖率=%.1f%%\n", nTest, nWrap, nBad,
                nTest > 0 ? sumCover / nTest * 100.0 : 0.0);
    return 0;
}
