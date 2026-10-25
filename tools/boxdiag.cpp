// 字符框贴合度诊断
//
// 目的：量化"框有没有框住字形"，而不是看图猜。
// 做法：在信号图上，对每个字符框统计
//   - 框内墨量占该字符总墨量的比例（recall：框住的部分 / 全部墨）
//   - 框外但很近（±20px）的墨量（说明框偏了，把真字形留在了外面）
// 判别依据：正常件的框应该 recall > 0.85，且框外余墨接近 0。
//
// 为什么需要这个工具：
// 0109 的两条路径给出 10 处 vs 1 处划痕，两者唯一的差别是逐字符对齐的
// 实现细节。要判断哪个对，必须先知道框本身准不准。
#include <cstdio>
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
    if (argc < 4) {
        std::printf("用法: boxdiag <modelDir> <image> [align 1/0]\n");
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
    const bool doAlign = (argc < 4) || std::string(argv[3]) != "0";
    const chardet::Config cfg = model.config;

    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

    // 整图配准（与两条检测路径相同）
    {
        cv::Mat r32, s32;
        model.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);
        cv::Rect ink;
        for (const auto& u : base) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width > 0 && b.height > 0) ink = ink.area() > 0 ? (ink | b) : b;
        }
        if (ink.area() > 0) {
            ink.x = std::max(0, ink.x - 20);
            ink.y = std::max(0, ink.y - 20);
            ink.width = std::min(sig.cols - ink.x, ink.width + 40);
            ink.height = std::min(sig.rows - ink.y, ink.height + 40);
            cv::Mat rC, sC;
            r32(ink).copyTo(rC);
            s32(ink).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::Mat al;
                cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
                sig = al;
                std::printf("整图配准平移: (%+.2f, %+.2f)\n", sh.x, sh.y);
            }
        }
    }

    std::vector<chardet::CharUnit> units = chardet::toAllUnits(model);
    std::vector<cv::Rect> originals;
    for (const auto& u : units) originals.push_back(u.bbox);

    if (doAlign) {
        for (auto& u : units) {
            cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (box.width <= 4 || box.height <= 4) continue;
            constexpr int kSearch = 12;
            cv::Rect search(box.x - kSearch, box.y - kSearch, box.width + 2 * kSearch,
                            box.height + 2 * kSearch);
            search &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (search.width < box.width || search.height < box.height) continue;
            cv::Mat mS, tS;
            model.freq(search).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig(search).convertTo(tS, CV_32F);
            const cv::Rect tr(box.x - search.x, box.y - search.y, box.width, box.height);
            cv::Mat res;
            cv::matchTemplate(tS, mS(tr), res, cv::TM_CCOEFF_NORMED);
            cv::Point ml;
            double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
            if (mv < 0.30) continue;
            const cv::Rect moved(box.x + ml.x - kSearch, box.y + ml.y - kSearch, box.width,
                                 box.height);
            if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
            u.bbox = moved;
        }
    }

    // 二值墨迹：信号高于阈值即算墨。
    //
    // 注意 strokeSignal 返回的是 CV_32F，范围 0~1，不是 0~255 的 8 位图。
    // 一开始我按 8 位图算 strokeHi*255，掩膜全空、recall 全变成 1.00，
    // 工具"确认了框完全准确"——假结论比没有结论更危险。
    cv::Mat inkMask;
    cv::compare(sig, cfg.strokeHi, inkMask, cv::CMP_GT);
    const int inkTotal = cv::countNonZero(inkMask);
    std::printf("信号 %dx%d type=%d 阈值=%.2f 全图墨点数=%d (%.2f%%)\n", sig.cols, sig.rows,
                sig.type(), cfg.strokeHi, inkTotal,
                100.0 * inkTotal / static_cast<double>(sig.cols) / sig.rows);
    if (inkTotal == 0) {
        std::printf("!! 墨迹掩膜为空，后续 recall 全部无意义\n");
        return 1;
    }

    std::printf("\n%-4s %-18s %-18s %6s %6s %6s %5s\n", "#", "原始框(x,y,w,h)", "对齐后(x,y,w,h)",
                "recall", "框外墨", "位移", "质心");
    double sumRecall = 0;
    int n = 0, bad = 0;
    for (size_t i = 0; i < units.size(); ++i) {
        const cv::Rect orig = originals[i] & cv::Rect(0, 0, sig.cols, sig.rows);
        const cv::Rect cur = units[i].bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (orig.width <= 0 || cur.width <= 0) continue;

        // 取"该字符应有的全部墨迹"的范围。
        //
        // 原来的扩窗只有 ±20px，而框本身就有 100+px 宽——这个窗口几乎
        // 完全落在框内，框外墨量永远是 0，recall 永远是 1.00。
        // 那是在用工具确认工具，什么也没测出来。
        // 改为按框宽高各扩 60%，才能容下"框偏了 10~20px"这种情况下的真墨迹。
        cv::Rect wide = cur;
        const int ex = std::max(12, cur.width * 3 / 5);
        const int ey = std::max(12, cur.height * 3 / 5);
        wide.x -= ex; wide.y -= ey; wide.width += 2 * ex; wide.height += 2 * ey;
        wide &= cv::Rect(0, 0, sig.cols, sig.rows);

        // 只算"属于这个字符"的墨：以框中心最近的连通域为准。
        //
        // 直接数扩窗里的全部墨会吃进左右邻居的字，recall 被整体拉低到 0.3~0.5，
        // 那测的是"字符有多密"而不是"框准不准"。
        // 判据：连通域质心落在扩窗内、且其质心离本框中心最近——即这一坨墨
        // 就是这个字。然后看它有多少落在框里。
        cv::Mat labels, stats, centroids;
        cv::Mat wideMask = inkMask(wide);
        const int nlab = cv::connectedComponentsWithStats(wideMask, labels, stats, centroids, 8, CV_32S);

        const cv::Point2d boxC(cur.x + cur.width / 2.0, cur.y + cur.height / 2.0);
        int bestLab = -1;
        double bestD = 1e18;
        for (int li = 1; li < nlab; ++li) {
            const int area = stats.at<int>(li, cv::CC_STAT_AREA);
            if (area < 60) continue;   // 噪点
            const cv::Point2d c(centroids.at<double>(li, 0) + wide.x,
                                centroids.at<double>(li, 1) + wide.y);
            const double d = std::hypot(c.x - boxC.x, c.y - boxC.y);
            if (d < bestD) { bestD = d; bestLab = li; }
        }

        double recall = 0.0, outside = 0.0;
        bool hasComp = false;
        if (bestLab >= 0) {
            cv::Mat compMask = (labels == bestLab);
            const double compArea = cv::countNonZero(compMask);
            // 框在 wide 坐标系里的位置
            const cv::Rect curInWide(cur.x - wide.x, cur.y - wide.y, cur.width, cur.height);
            const cv::Rect valid = curInWide & cv::Rect(0, 0, labels.cols, labels.rows);
            const double inBox = valid.width > 0 ? cv::countNonZero(compMask(valid)) : 0.0;
            recall = compArea > 0 ? inBox / compArea : 1.0;
            outside = compArea - inBox;
            hasComp = true;
        }
        // 质心相对框中心的偏移：这是"框准不准"最直接的读数
        const double cdx = hasComp ? (centroids.at<double>(bestLab, 0) + wide.x - boxC.x) : 0.0;
        const double cdy = hasComp ? (centroids.at<double>(bestLab, 1) + wide.y - boxC.y) : 0.0;
        const int dx = cur.x - orig.x, dy = cur.y - orig.y;

        sumRecall += recall;
        ++n;
        if (recall < 0.85) ++bad;
        std::printf("%-4zu %-18s %-18s %6.2f %6.0f %+3d,%+3d %+5.1f,%+5.1f\n", i,
                    (std::to_string(orig.x) + "," + std::to_string(orig.y) + "," +
                     std::to_string(orig.width) + "," + std::to_string(orig.height)).c_str(),
                    (std::to_string(cur.x) + "," + std::to_string(cur.y) + "," +
                     std::to_string(cur.width) + "," + std::to_string(cur.height)).c_str(),
                    recall, outside, dx, dy, cdx, cdy);
    }
    std::printf("\n字符框数=%d 平均recall=%.3f recall<0.85的框=%d\n", n,
                n > 0 ? sumRecall / n : 0.0, bad);
    return 0;
}
