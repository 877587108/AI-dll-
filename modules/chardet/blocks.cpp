// 整行字符级比对（多字 / 少字 / 字符错位）
//
// 为什么用"按字符区间统计"而不是"数墨块"：
//   数墨块对阈值极其敏感——同一个字在不同图里会被切成 1 块或 2 块，
//   于是模板那块报"缺失"、待检那块报"多出"，产生大量固定位置的误报。
//   实测 OK 样本稳定报 5~7 个缺失，全是这种切分抖动。
//
//   字符的 x 区间由选区给出，是稳定的。按区间统计墨量再比较，
//   就绕开了切分不稳定的问题。
//
// 能发现：
//   · 某个字符位置完全没有墨      → 少字
//   · 字符区间之外出现成块的新墨  → 多字
//   · 字符整体横向偏移            → 排列异常

#include "chardet.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace chardet {

namespace {

/// 按行把字符分组（y 范围重叠的归为同一行）
struct RowGroup {
    int y0 = 0;
    int y1 = 0;
    std::vector<CharUnit> units;
};

std::vector<RowGroup> groupRows(const std::vector<CharUnit>& units, const cv::Rect& img) {
    std::vector<CharUnit> valid;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & img;
        if (b.width > 0 && b.height > 0) valid.push_back(u);
    }
    std::sort(valid.begin(), valid.end(),
              [](const CharUnit& a, const CharUnit& b) { return a.bbox.y < b.bbox.y; });

    std::vector<RowGroup> rows;
    for (const auto& u : valid) {
        const int y0 = u.bbox.y;
        const int y1 = u.bbox.y + u.bbox.height;
        if (!rows.empty() && y0 <= rows.back().y1 + 6) {
            rows.back().y1 = std::max(rows.back().y1, y1);
            rows.back().units.push_back(u);
        } else {
            RowGroup g;
            g.y0 = y0;
            g.y1 = y1;
            g.units.push_back(u);
            rows.push_back(g);
        }
    }
    for (auto& r : rows) {
        std::sort(r.units.begin(), r.units.end(),
                  [](const CharUnit& a, const CharUnit& b) { return a.bbox.x < b.bbox.x; });
    }
    return rows;
}

/// 统计某个矩形内的墨像素数（二值图）
int inkIn(const cv::Mat& bin, const cv::Rect& r, const cv::Rect& img) {
    const cv::Rect rr = r & img;
    if (rr.width <= 0 || rr.height <= 0) return 0;
    return cv::countNonZero(bin(rr));
}

}  // namespace

BlockDiff diffBlocks(const cv::Mat& signal, const cv::Mat& modelFreq,
                     const std::vector<CharUnit>& units, const Config& cfg, int margin) {
    BlockDiff out;
    if (signal.empty() || modelFreq.empty() || units.empty()) return out;
    if (signal.type() != CV_32F) return out;
    if (signal.size() != modelFreq.size()) return out;

    const cv::Rect img(0, 0, signal.cols, signal.rows);

    // 二值化：两者用等效阈值（已标定，量纲匹配）
    cv::Mat mBin, tBin;
    cv::compare(modelFreq, static_cast<int>(cfg.missFreq * 255.0), mBin, cv::CMP_GT);
    cv::compare(signal, cfg.strokeHi, tBin, cv::CMP_GT);

    const auto rows = groupRows(units, img);
    out.rowCount = static_cast<int>(rows.size());

    for (const auto& row : rows) {
        const int y0 = std::max(0, row.y0 - margin);
        const int y1 = std::min(img.height - 1, row.y1 + margin);
        const int bandH = std::max(1, y1 - y0 + 1);

        // ① 逐字符区间比较
        BlockDiff::RowCount rc;
        rc.y0 = y0;
        rc.y1 = y0 + bandH;
        rc.expected = static_cast<int>(row.units.size());

        for (const auto& u : row.units) {
            const cv::Rect b = u.bbox & img;
            if (b.width <= 0 || b.height <= 0) continue;

            const int mInk = inkIn(mBin, b, img);
            const int tInk = inkIn(tBin, b, img);
            ++out.modelBlocks;   // 复用字段：字符总数

            // 该字符位有墨 → 认为这个字在
            if (mInk > 0 && static_cast<double>(tInk) / mInk >= cfg.charMissingInkRatio) {
                ++rc.found;
                ++out.testBlocks;
                continue;
            }

            // 该字符位置完全没墨 → 少字
            ++out.missingCount;
            ++rc.missingChars;
            BlockDiff::Block bk;
            bk.x0 = b.x;
            bk.x1 = b.x + b.width;
            bk.y0 = y0;
            bk.y1 = y0 + bandH;
            out.missingBlocks.push_back(bk);
        }

        // ② 整行范围内找"字符区间之外"的成块新墨 → 多字
        //
        // 行带从最左字符的左边缘到最右字符的右边缘，再向两侧各留一个字宽，
        // 覆盖"行尾多一个字"的情形。
        const int rowX0 = row.units.front().bbox.x;
        const int rowX1 = row.units.back().bbox.x + row.units.back().bbox.width;
        const int avgW = std::max(20, (rowX1 - rowX0) / std::max(1, static_cast<int>(row.units.size())));
        const int scanX0 = std::max(0, rowX0 - avgW);
        const int scanX1 = std::min(img.width, rowX1 + avgW);
        if (scanX1 <= scanX0) continue;

        // 把"字符区间"在扫描带内标成已覆盖，其余为待检查区
        cv::Mat covered = cv::Mat::zeros(cv::Size(scanX1 - scanX0, bandH), CV_8U);
        for (const auto& u : row.units) {
            const cv::Rect b = u.bbox & img;
            if (b.width <= 0) continue;
            const int x0 = std::max(scanX0, b.x) - scanX0;
            const int x1 = std::min(scanX1, b.x + b.width) - scanX0;
            if (x1 > x0) covered(cv::Rect(x0, 0, x1 - x0, bandH)).setTo(255);
        }

        const cv::Rect scan(scanX0, y0, scanX1 - scanX0, bandH);

        // 待检在扫描带内的墨，排除字符区间
        cv::Mat tScan;
        tBin(scan).copyTo(tScan);
        cv::Mat notCovered;
        cv::bitwise_not(covered, notCovered);
        cv::bitwise_and(tScan, notCovered, tScan);

        // 模型在同一区域（字符之外）也应当没有墨；若模板本身有残留则扣除
        cv::Mat mScan;
        mBin(scan).copyTo(mScan);
        cv::bitwise_and(mScan, notCovered, mScan);

        // 形态学开运算去掉刻印噪点
        const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
        cv::morphologyEx(tScan, tScan, cv::MORPH_OPEN, k);

        cv::Mat labels, stats, cents;
        const int n = cv::connectedComponentsWithStats(tScan, labels, stats, cents, 8, CV_32S);
        for (int i = 1; i < n; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            // 门槛要严：多出的必须是"像个字符"的墨块。
            // 太松会把背景划痕、脏斑当成多余字符——实测 84 张 OK 里
            // 有 9 张因此误报，全部是这一侧造成的。
            // 字符的高度通常占行带的 60% 以上，宽度至少半个字宽，
            // 面积也要达到字符量级。
            if (area < 300) continue;
            if (h < bandH * 6 / 10) continue;          // 太矮，不像字符
            if (w < (scanX1 - scanX0) / 40) continue;  // 太窄，不像字符
            if (w > (scanX1 - scanX0) / 4) continue;   // 太宽，可能是大片脏污

            const cv::Rect blk(stats.at<int>(i, cv::CC_STAT_LEFT) + scanX0,
                               stats.at<int>(i, cv::CC_STAT_TOP) + y0, w, h);

            // 该位置模型本来就有墨 → 不是多出的
            if (inkIn(mScan, cv::Rect(blk.x - scanX0, blk.y - y0, w, h), img) > area / 3) continue;

            ++out.extraCount;
            ++rc.extraChars;
            BlockDiff::Block bk;
            bk.x0 = blk.x;
            bk.x1 = blk.x + w;
            bk.y0 = y0;
            bk.y1 = y0 + bandH;
            out.extraBlocks.push_back(bk);
        }

        ++out.rowCount;
        out.rows.push_back(rc);
    }

    out.valid = out.rowCount > 0;
    return out;
}

}  // namespace chardet
