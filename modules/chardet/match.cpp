#include "chardet.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace chardet {

const char* defectName(DefectType t) {
    switch (t) {
        case DefectType::Missing: return "少墨";
        case DefectType::Dirt: return "脏污";
        case DefectType::Scratch: return "划痕";
        case DefectType::Extra: return "多墨";
        case DefectType::Shape: return "字形偏移";
        default: return "无";
    }
}

namespace {

/// 单个连通块的信息
struct Blob {
    cv::Rect rect;       // 在原图中的位置
    int area = 0;
    double elong = 1.0;  // 长宽比（长边/短边）
    int length = 0;      // 长边像素长度
};

/// 找出二值图里的所有连通块（过滤掉过小的）
std::vector<Blob> findBlobs(const cv::Mat& bin, int minArea, const cv::Point& offset) {
    std::vector<Blob> out;
    cv::Mat labels, stats, cents;
    const int n = cv::connectedComponentsWithStats(bin, labels, stats, cents, 8, CV_32S);
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < minArea) continue;

        Blob b;
        b.area = area;
        b.rect = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT) + offset.x,
                          stats.at<int>(i, cv::CC_STAT_TOP) + offset.y,
                          stats.at<int>(i, cv::CC_STAT_WIDTH),
                          stats.at<int>(i, cv::CC_STAT_HEIGHT));
        const int w = b.rect.width;
        const int h = b.rect.height;
        b.length = std::max(w, h);
        const int shortSide = std::max(1, std::min(w, h));
        b.elong = static_cast<double>(b.length) / shortSide;
        out.push_back(b);
    }
    return out;
}

/// 判据一：少墨
///
/// 按"笔画连通块"逐段检查，而不是整框算比例。
///
/// 为什么必须逐段：logo 区是一个 279×177 的大框，缺的可能只是 i 上的一个点
/// （约 100px）。按整框比例算只有 1%，永远达不到门槛；
/// 但按"这个点是一个独立的笔画段，它整段没了"来判，就能抓出来。
void checkMissing(const cv::Mat& sRoi, const cv::Mat& freqF, const cv::Rect& r,
                  const Config& cfg, CharMatch& cm) {
    // 模板里"应有墨"的区域
    cv::Mat mustInk;
    cv::compare(freqF, cfg.missFreq, mustInk, cv::CMP_GT);

    // 整字缺失判定：框内实际墨量 vs 应有墨量
    cv::Mat strong;
    cv::compare(sRoi, cfg.strokeHi, strong, cv::CMP_GT);
    const int inkN = cv::countNonZero(mustInk);
    const int actualInk = cv::countNonZero(strong);
    const double inkRatio = inkN > 0 ? static_cast<double>(actualInk) / inkN : 1.0;

    if (inkN > 0 && inkRatio < cfg.charMissingInkRatio) {
        cm.missingChar = true;
        cm.missRatio = 1.0 - inkRatio;
        cm.missScore = std::max(1.5, cm.missRatio / std::max(1e-6, cfg.charT));
        return;
    }

    // 逐段检查：把"应有墨"切成一笔一笔的连通块
    const auto mustBlobs = findBlobs(mustInk, cfg.missMinBlobArea, cv::Point(0, 0));
    if (mustBlobs.empty()) return;

    int totalInk = 0;
    int totalGone = 0;
    int worstMissing = 0;
    double worstRatio = 0.0;

    for (const Blob& b : mustBlobs) {
        const cv::Rect br = b.rect & cv::Rect(0, 0, sRoi.cols, sRoi.rows);
        if (br.width <= 0 || br.height <= 0) continue;

        const int should = cv::countNonZero(mustInk(br));
        if (should < cfg.missMinBlobArea) continue;

        cv::Mat strongIn;
        cv::compare(sRoi(br), cfg.missStrokeLo, strongIn, cv::CMP_GT);
        const int have = cv::countNonZero(strongIn);

        const int gone = std::max(0, should - have);
        totalInk += should;
        totalGone += gone;

        const double ratio = static_cast<double>(gone) / should;
        if (ratio > worstRatio) {
            worstRatio = ratio;
            worstMissing = gone;
        }

        // 整段消失（该段 80% 以上没墨）→ 直接判少墨
        if (ratio > 0.80) {
            cm.missScore = std::max(cm.missScore, 1.2);
        }
    }

    cm.missRatio = totalInk > 0 ? static_cast<double>(totalGone) / totalInk : 0.0;

    // 按最严重的那一段打分（而不是整体比例）——这正是解决大框稀释的关键
    if (worstRatio > cfg.charMissRatio && worstMissing >= cfg.missMinBlobArea) {
        const double segScore = (worstRatio - cfg.charMissRatio) / std::max(1e-6, cfg.charT);
        cm.missScore = std::max(cm.missScore, segScore);
    }
}

/// 判据二/三：脏污与划痕
///
/// 共同点：在"应为空白"的地方出现了墨。
/// 区别在形状：
///   脏污 = 面积小、近似圆形
///   划痕 = 细长、长宽比大
void checkExtraBlobs(const cv::Mat& sExt, const cv::Mat& freqExt, const cv::Mat& neighborInk,
                     const cv::Rect& rext, const cv::Rect& r, const Config& cfg, CharMatch& cm,
                     int inkN) {
    // 应为空白（模板频率低）但当前有强墨
    cv::Mat lowFreq, strong, extraPx;
    cv::compare(freqExt, cfg.blankFreq, lowFreq, cv::CMP_LT);
    cv::compare(sExt, cfg.strokeHi, strong, cv::CMP_GT);
    cv::bitwise_and(strong, lowFreq, extraPx);
    // 排除相邻字符的笔画
    cv::Mat notNeighbor;
    cv::bitwise_not(neighborInk, notNeighbor);
    cv::bitwise_and(extraPx, notNeighbor, extraPx);

    cv::Mat extra8;
    extraPx.convertTo(extra8, CV_8U, 255.0);
    const cv::Mat kOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::morphologyEx(extra8, extra8, cv::MORPH_OPEN, kOpen);   // 去孤立噪点

    const auto blobs = findBlobs(extra8, cfg.dirtMinArea, cv::Point(rext.x, rext.y));

    int dirtArea = 0;
    int dirtCount = 0;
    int scratchMaxLen = 0;
    int scratchCount = 0;

    for (const Blob& b : blobs) {
        // 区分"笔画变粗"与"独立多余墨"
        //
        // 不能靠位置阈值判断：
        //   用 missFreq(0.50) → 增粗的边缘条带被判成划痕
        //   用 blankFreq(0.15) → 划痕被当成笔画附近，反而漏检
        //
        // 改用距离：该块像素与模板笔画的最小距离。
        //   大部分像素紧贴笔画(<=2px) → 属于"增粗"，归入多墨
        //   离笔画远                     → 独立多余墨，归入脏污或划痕
        const cv::Rect local(b.rect.x - rext.x, b.rect.y - rext.y, b.rect.width, b.rect.height);
        const cv::Rect loc = local & cv::Rect(0, 0, freqExt.cols, freqExt.rows);
        bool onInk = false;
        if (loc.width > 0 && loc.height > 0) {
            cv::Mat inkMask;
            cv::compare(freqExt, cfg.blankFreq, inkMask, cv::CMP_GT);
            const cv::Mat k2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
            cv::Mat inkDil;
            cv::dilate(inkMask, inkDil, k2);

            const int inNeighbor = cv::countNonZero(inkDil(loc));
            const int area = std::max(1, b.area);
            onInk = static_cast<double>(inNeighbor) / area > 0.60;
        }

        if (onInk) {
            // 紧贴笔画：归入多墨（由 checkThicknessAndShape 计分）
            continue;
        }

        // 划痕：细长
        if (b.elong >= cfg.scratchMinElong && b.length >= cfg.scratchMinLen) {
            ++scratchCount;
            scratchMaxLen = std::max(scratchMaxLen, b.length);
            continue;
        }
        // 脏污：面积在范围内
        if (b.area <= cfg.dirtMaxArea) {
            dirtArea += b.area;
            ++dirtCount;
        }
    }

    cm.dirtBlobs = dirtCount;
    cm.scratchCount = scratchCount;
    cm.scratchLen = scratchMaxLen;

    // 脏污归一化
    //
    // 不能用「脏污面积 / 应有笔画面积」——实测字符框应有笔画 3660px，
    // 一个 10×10 的污点只占 2.7%，永远达不到 12% 的门槛，
    // 这种归一化会让小污点被笔画总量稀释掉。
    //
    // 改用「污点等效直径 / 字符高度」：污点是局部缺陷，
    // 只要它在字符尺度上足够显眼就该报出来。
    if (dirtArea > 0 && r.height > 0) {
        const double equivDia = 2.0 * std::sqrt(dirtArea / CV_PI);
        const double rel = equivDia / static_cast<double>(r.height);
        // 污点直径达到字符高度的 8% 即达到门槛
        constexpr double kDirtRel = 0.08;
        if (rel > kDirtRel) {
            cm.dirtScore = (rel - kDirtRel) / std::max(1e-6, cfg.charT) * 3.0;
        }
    }
    // 划痕归一化：长度相对框尺寸
    if (scratchMaxLen > 0) {
        const int refLen = std::max(40, std::min(r.width, r.height));
        const double spanRatio = static_cast<double>(scratchMaxLen) / refLen;
        if (spanRatio >= cfg.scratchMinLen * 1.0 / refLen) {
            cm.scratchScore = spanRatio;
        }
    }
}

/// 判据四：多墨（笔画变粗）与字形偏移
void checkThicknessAndShape(const cv::Mat& sRoi, const cv::Mat& sExt, const cv::Mat& freqF,
                            const cv::Mat& freqExt, const cv::Rect& r, const Config& cfg,
                            CharMatch& cm, int inkN) {
    // 多墨：在"应有墨"的区域，当前墨量明显超出
    // （用笔画面积对比：当前强墨面积 / 模板应有面积）
    cv::Mat strong;
    cv::compare(sRoi, cfg.strokeHi, strong, cv::CMP_GT);
    const int actualInk = cv::countNonZero(strong);
    if (inkN > 0) {
        const double over = static_cast<double>(actualInk) / inkN - 1.0;
        if (over > cfg.charExtraRatio) {
            cm.extraScore = (over - cfg.charExtraRatio) / std::max(1e-6, cfg.charT);
            cm.extraRatio = over;
        }
    }

    // 字形偏移：质心位移
    double sumFx = 0, sumFy = 0, sumF = 0;
    double sumSx = 0, sumSy = 0, sumS = 0;
    for (int y = 0; y < sRoi.rows; ++y) {
        const float* frow = freqF.ptr<float>(y);
        const float* srow = sRoi.ptr<float>(y);
        for (int x = 0; x < sRoi.cols; ++x) {
            const float f = frow[x];
            const float sv = srow[x] > cfg.strokeHi ? srow[x] : 0.0f;
            if (f > 0.3f) { sumFx += x * f; sumFy += y * f; sumF += f; }
            if (sv > 0.0f) { sumSx += x * sv; sumSy += y * sv; sumS += sv; }
        }
    }
    if (sumF > 1e-3 && sumS > 1e-3) {
        const double cx = sumFx / sumF, cy = sumFy / sumF;
        const double ax = sumSx / sumS, ay = sumSy / sumS;
        const double dist = std::sqrt((cx - ax) * (cx - ax) + (cy - ay) * (cy - ay));
        const double scale = std::max(1.0, static_cast<double>(r.height));
        const double rel = dist / scale;
        constexpr double kShapeThr = 0.08;
        if (rel > kShapeThr) cm.shapeScore = (rel - kShapeThr) * 4.0;
    }
    (void)sExt;
    (void)freqExt;
}

}  // namespace


std::vector<CharMatch> match(const cv::Mat& aligned, const std::vector<CharUnit>& units,
                             const Config& cfg) {
    std::vector<CharMatch> out;
    if (aligned.empty() || units.empty() || aligned.type() != CV_32F) return out;

    const cv::Rect imgRect(0, 0, aligned.cols, aligned.rows);

    for (const CharUnit& u : units) {
        if (u.freq.size() != aligned.size()) continue;
        const cv::Rect r = u.bbox & imgRect;
        if (r.width <= 0 || r.height <= 0) continue;

        cv::Rect rext = u.extend & imgRect;
        if (rext.width <= 0 || rext.height <= 0) rext = r;

        const cv::Mat sRoi = aligned(r);
        const cv::Mat sExt = aligned(rext);

        cv::Mat freqF;
        u.freq(r).convertTo(freqF, CV_32F, 1.0 / 255.0);

        // 扩展框的模板：字符框内取模板值，框外视为空白
        cv::Mat freqExt = cv::Mat::zeros(rext.size(), CV_32F);
        for (int y = 0; y < rext.height; ++y) {
            const int gy = rext.y + y;
            const int ly = gy - r.y;
            const uchar* frowFull = u.freq.ptr<uchar>(gy);
            float* fext = freqExt.ptr<float>(y);
            for (int x = 0; x < rext.width; ++x) {
                const int gx = rext.x + x;
                const int lx = gx - r.x;
                if (ly >= 0 && ly < r.height && lx >= 0 && lx < r.width) {
                    fext[x] = frowFull[gx] / 255.0f;
                }
            }
        }

        // 扣掉相邻字符的地盘：扩展框有 12px 余量，
        // 同一行相邻字符只隔约 20px，邻居笔画会落进来被误判成多余墨。
        cv::Mat neighborInk = cv::Mat::zeros(rext.size(), CV_8U);
        for (const CharUnit& other : units) {
            if (&other == &u) continue;
            const cv::Rect inter = other.bbox & rext;
            if (inter.width <= 0 || inter.height <= 0) continue;
            const cv::Rect local(inter.x - rext.x, inter.y - rext.y, inter.width, inter.height);
            neighborInk(local).setTo(255);
        }

        cv::Mat mustInk;
        cv::compare(freqF, cfg.missFreq, mustInk, cv::CMP_GT);
        const int inkN = cv::countNonZero(mustInk);

        CharMatch cm;
        cm.index = u.index;
        cm.bbox = r;

        checkMissing(sRoi, freqF, r, cfg, cm);
        checkExtraBlobs(sExt, freqExt, neighborInk, rext, r, cfg, cm, inkN);
        checkThicknessAndShape(sRoi, sExt, freqF, freqExt, r, cfg, cm, inkN);

        const double scores[] = {cm.missScore, cm.dirtScore, cm.scratchScore, cm.extraScore,
                                 cm.shapeScore};
        const DefectType types[] = {DefectType::Missing, DefectType::Dirt, DefectType::Scratch,
                                    DefectType::Extra, DefectType::Shape};
        int best = 0;
        for (int i = 1; i < 5; ++i) {
            if (scores[i] > scores[best]) best = i;
        }
        cm.score = scores[best];
        cm.type = cm.score > 0.0 ? types[best] : DefectType::None;

        if (cm.score > 0.0) {
            char buf[160];
            switch (cm.type) {
                case DefectType::Missing:
                    if (cm.missingChar) {
                        std::snprintf(buf, sizeof(buf), "少墨：整字缺失");
                    } else {
                        std::snprintf(buf, sizeof(buf), "少墨：%.0f%% 笔画缺失",
                                      cm.missRatio * 100.0);
                    }
                    break;
                case DefectType::Dirt:
                    std::snprintf(buf, sizeof(buf), "脏污：%d 处污点", cm.dirtBlobs);
                    break;
                case DefectType::Scratch:
                    std::snprintf(buf, sizeof(buf), "划痕：%d 条，最长 %dpx", cm.scratchCount,
                                  cm.scratchLen);
                    break;
                case DefectType::Extra:
                    std::snprintf(buf, sizeof(buf), "多墨：笔画增粗 %.0f%%",
                                  cm.extraRatio * 100.0);
                    break;
                case DefectType::Shape:
                    std::snprintf(buf, sizeof(buf), "字形偏移");
                    break;
                default:
                    buf[0] = 0;
                    break;
            }
            cm.label = buf;
        }

        out.push_back(cm);
    }
    return out;
}

}  // namespace chardet
