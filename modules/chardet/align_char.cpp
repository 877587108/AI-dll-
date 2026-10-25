// 在 match() 之前对每个字符框做局部对齐
//
// 模型是逐字符精对齐后平均的，检测时若只做整图配准，
// 残余刻印偏移会让每个字符都有 5~6px 错位，导致大面积误报（实测 9/84）。
// 这里用模板在框附近做小范围搜索，把每个字符校正到最佳位置。
//
// 实现方式：在一个新函数里封装，match() 内部调用。

#include "chardet.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace chardet {
namespace {

/// 在 searchR 范围内为单字符找最佳平移，返回校正量
cv::Point findLocalShift(const cv::Mat& signal, const cv::Mat& templ, const cv::Rect& box,
                         int searchR) {
    const cv::Rect img(0, 0, signal.cols, signal.rows);
    cv::Rect search(box.x - searchR, box.y - searchR, box.width + 2 * searchR,
                    box.height + 2 * searchR);
    search &= img;
    if (search.width < box.width || search.height < box.height) return cv::Point(0, 0);
    if (templ.cols <= 0 || templ.rows <= 0) return cv::Point(0, 0);

    const cv::Mat s = signal(search);
    cv::Mat f32;
    s.convertTo(f32, CV_32F);
    cv::Mat t32;
    templ.convertTo(t32, CV_32F, 1.0 / 255.0);

    if (t32.cols > f32.cols || t32.rows > f32.rows) return cv::Point(0, 0);

    cv::Mat res;
    cv::matchTemplate(f32, t32, res, cv::TM_CCOEFF_NORMED);
    cv::Point mLoc;
    double mVal = 0.0;
    cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
    if (mVal < 0.5) return cv::Point(0, 0);   // 匹配太弱，不校正

    return cv::Point(mLoc.x - searchR, mLoc.y - searchR);
}

}  // namespace

/// 逐字符对齐后的信号（返回整幅图，每个字符被搬到最佳位置）
cv::Mat alignPerChar(const cv::Mat& signal, const std::vector<CharUnit>& units, int searchR,
                     double* meanPeak) {
    if (signal.empty() || units.empty()) return signal;
    if (signal.type() != CV_32F) return signal;

    cv::Mat out = signal.clone();
    double sum = 0;
    int n = 0;

    for (const CharUnit& u : units) {
        const cv::Rect box = u.bbox & cv::Rect(0, 0, signal.cols, signal.rows);
        if (box.width <= 0 || box.height <= 0) continue;

        cv::Mat templ = cv::Mat::zeros(box.size(), CV_8U);
        u.freq(box).copyTo(templ);

        const cv::Point sh = findLocalShift(signal, templ, box, searchR);
        if (sh.x == 0 && sh.y == 0) continue;

        // 把该字符按平移量搬到位
        const cv::Rect src(box.x + sh.x, box.y + sh.y, box.width, box.height);
        const cv::Rect srcClipped = src & cv::Rect(0, 0, signal.cols, signal.rows);
        if (srcClipped.width != box.width || srcClipped.height != box.height) continue;

        signal(srcClipped).copyTo(out(box));
        sum += 1;
        ++n;
    }
    if (meanPeak) *meanPeak = n > 0 ? sum / n : 0.0;
    return out;
}

}  // namespace chardet
