// 导出字符框判定可视化
//
// 把字符框、字墨外接框、溢出边画到图上，肉眼看一次胜过读十行数字。
// 中文路径用 imdecode 读、imencode 写，避开 imread/imwrite 的编码问题。
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

static bool writePng(const std::string& path, const cv::Mat& img) {
    std::vector<unsigned char> buf;
    if (!cv::imencode(".png", img, buf)) return false;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::fwrite(buf.data(), 1, buf.size(), fp);
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: boxviz <modelDir> <image> <out.png> [pad=2]\n");
        return 2;
    }
    const int pad = (argc > 4) ? std::atoi(argv[4]) : 2;
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
    const cv::Rect full(0, 0, gray.cols, gray.rows);
    cv::Mat sig = chardet::strokeSignal(gray, full, model.config, false);
    cv::Mat ref32;
    model.ref.convertTo(ref32, CV_32F);
    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);

    // 与产品一致的整图配准
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
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::Mat al;
            cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            sig = al;
        }
    }
    // 逐字匹配
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
        cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)), res,
                          cv::TM_CCOEFF_NORMED);
        cv::Point ml;
        double mv = 0;
        cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
        if (mv < 0.30) continue;
        const cv::Rect moved(box.x + ml.x - kS, box.y + ml.y - kS, box.width, box.height);
        if ((moved & full) != moved) continue;
        u.bbox = moved;
    }

    cv::Mat inkMask;
    cv::compare(sig, model.config.strokeHi, inkMask, cv::CMP_GT);
    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(inkMask, labels, stats, centroids, 8, CV_32S);

    // 画在灰度图上，用彩色标注
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    int nBad = 0;
    for (const auto& u : units) {
        const cv::Rect box = u.bbox & full;
        if (box.width <= 4 || box.height <= 4) continue;
        const int ex = std::max(10, box.width / 2), ey = std::max(10, box.height / 2);
        const cv::Rect wide =
            (cv::Rect(box.x - ex, box.y - ey, box.width + 2 * ex, box.height + 2 * ey) & full);
        cv::Rect glyph;
        for (int li = 1; li < stats.rows; ++li) {
            if (stats.at<int>(li, cv::CC_STAT_AREA) < 40) continue;
            const cv::Point2d c(centroids.at<double>(li, 0), centroids.at<double>(li, 1));
            if (!wide.contains(cv::Point(cvRound(c.x), cvRound(c.y)))) continue;
            const cv::Rect comp(stats.at<int>(li, cv::CC_STAT_LEFT),
                                stats.at<int>(li, cv::CC_STAT_TOP),
                                stats.at<int>(li, cv::CC_STAT_WIDTH),
                                stats.at<int>(li, cv::CC_STAT_HEIGHT));
            if ((comp & box).area() <= 0) continue;
            glyph = glyph.area() > 0 ? (glyph | comp) : comp;
        }
        if (glyph.area() <= 0) continue;
        const int ov = std::max({box.x - glyph.x, box.y - glyph.y,
                                 (glyph.x + glyph.width) - (box.x + box.width),
                                 (glyph.y + glyph.height) - (box.y + box.height)});
        const bool ok = ov <= pad;
        if (!ok) ++nBad;
        // 框：绿=包住，红=溢出
        cv::rectangle(vis, box, ok ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255), 2);
        // 字墨外接框：黄细线
        cv::rectangle(vis, glyph, cv::Scalar(0, 220, 255), 1);
        if (!ok) {
            char t[32];
            std::snprintf(t, sizeof(t), "ov%d", ov);
            cv::putText(vis, t, cv::Point(box.x, box.y - 4), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
    if (!writePng(argv[3], vis)) {
        std::printf("写出失败\n");
        return 1;
    }
    std::printf("已写出 %s  溢出框=%d\n", argv[3], nBad);
    return 0;
}
