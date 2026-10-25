#include "core/char_core.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "chardet.hpp"
#include "core/model_io.hpp"
#include "core/detectors.hpp"
#include "core/warp.hpp"

#include <opencv2/imgproc.hpp>

namespace markloc::charinsp {
// ============================================================================
bool autoCalibrateRois(const std::array<std::vector<cv::Mat>, kViewCount>& okByView,
                       std::array<Params, kViewCount>& ps, std::string* log) {
    if (okByView[0].empty()) {
        if (log) *log += "视角0 无样本，无法标定\n";
        return false;
    }
    const cv::Mat ref = okByView[0].front();
    const cv::Rect refRoi = ps[0].rois[1].rect;   // 用 row1 文字区作匹配模板
    if (refRoi.x < 0 || refRoi.y < 0 || refRoi.x + refRoi.width > ref.cols ||
        refRoi.y + refRoi.height > ref.rows) {
        if (log) *log += "视角0 的 line1 越界\n";
        return false;
    }
    const cv::Mat templ0 = ref(refRoi).clone();

    bool any = false;
    for (int v = 1; v < kViewCount; ++v) {
        if (okByView[static_cast<size_t>(v)].empty()) continue;
        const cv::Mat img = okByView[static_cast<size_t>(v)].front();

        double bestScore = -2.0;
        double bestScale = 1.0;
        cv::Point bestLoc(0, 0);
        for (double s = 0.80; s <= 1.2001; s += 0.02) {
            cv::Mat templ;
            cv::resize(templ0, templ, cv::Size(), s, s, s < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
            if (templ.cols >= img.cols || templ.rows >= img.rows) continue;
            cv::Mat res;
            cv::matchTemplate(img, templ, res, cv::TM_CCOEFF_NORMED);
            double mx = 0;
            cv::Point loc;
            cv::minMaxLoc(res, nullptr, &mx, nullptr, &loc);
            if (mx > bestScore) {
                bestScore = mx;
                bestScale = s;
                bestLoc = loc;
            }
        }
        if (bestScore < 0.35) {
            if (log) *log += "视角" + std::to_string(v) + " 匹配失败(score=" +
                             std::to_string(bestScore) + ")\n";
            continue;
        }

        // 把视角0的四个 ROI 映射到视角 v
        for (int i = 0; i < 4; ++i) {
            const cv::Rect r0 = ps[0].rois[i].rect;
            const int nx = static_cast<int>(std::lround(bestLoc.x + (r0.x - refRoi.x) * bestScale));
            const int ny = static_cast<int>(std::lround(bestLoc.y + (r0.y - refRoi.y) * bestScale));
            const int nw = static_cast<int>(std::lround(r0.width * bestScale));
            const int nh = static_cast<int>(std::lround(r0.height * bestScale));
            ps[static_cast<size_t>(v)].rois[i].rect =
                cv::Rect(std::max(0, nx), std::max(0, ny), nw, nh);
        }
        any = true;
        if (log) {
            *log += "视角" + std::to_string(v) + " 标定: scale=" + std::to_string(bestScale) +
                    " loc=(" + std::to_string(bestLoc.x) + "," + std::to_string(bestLoc.y) +
                    ") score=" + std::to_string(bestScore) + "\n";
        }
    }
    return any;
}

bool buildModel(const std::array<std::vector<cv::Mat>, kViewCount>& okByView,
                std::array<Params, kViewCount>& ps, Model& out, std::string* err) {
    int built = 0;
    for (int v = 0; v < kViewCount; ++v) {
        ViewModel& vm = out.views[v];
        vm = ViewModel{};
        const auto& imgs = okByView[v];
        if (imgs.empty()) continue;

        Params& p = ps[v];
        // 自动判定极性：若负向亮尾强于正向，则视为暗刻印
        {
            const cv::Mat m0 = imgs.front()(p.rois[1].rect);
            cv::Mat mf, bg;
            m0.convertTo(mf, CV_32F);
            cv::GaussianBlur(mf, bg, cv::Size(0, 0), p.sigma);
            cv::Mat dd;
            cv::subtract(mf, bg, dd);
            double mx = 0, mn = 0;
            cv::minMaxLoc(dd, &mn, &mx);
            p.invert = (-mn) > mx;
        }
        bool failed = false;
        for (int i = 0; i < 4; ++i) {
            const cv::Rect roi = p.rois[i].rect;
            if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > imgs.front().cols ||
                roi.y + roi.height > imgs.front().rows) {
                if (err) *err = "视角 " + std::to_string(v) + " 区域越界: " + kRoiNames[i];
                failed = true;
                break;
            }

            cv::Mat sRef = strokeSignal(imgs.front(), roi, p);
            cv::Mat ref8;
            sRef.convertTo(ref8, CV_8U, 80.0, 0.0);

            cv::Mat count = cv::Mat::zeros(sRef.size(), CV_32F);
            int used = 0;
            for (size_t k = 0; k < imgs.size(); ++k) {
                if (imgs[k].size() != imgs.front().size()) continue;
                cv::Mat s = strokeSignal(imgs[k], roi, p);
                if (k > 0) {
                    cv::Mat warp;
                    if (detail::estimateWarp(s, sRef, warp)) s = detail::applyWarp(s, warp);
                }
                cv::Mat mask;
                cv::compare(s, p.strokeHi, mask, cv::CMP_GT);
                cv::Mat mf;
                mask.convertTo(mf, CV_32F, 1.0 / 255.0);
                count += mf;
                ++used;
            }
            if (used == 0) {
                failed = true;
                break;
            }

            cv::Mat freq = count / static_cast<double>(used);
            cv::Mat freq8;
            freq.convertTo(freq8, CV_8U, 255.0, 0.0);

            cv::Mat keep;
            cv::compare(freq, p.pKeep, keep, cv::CMP_GT);
            cv::Mat tol = cv::Mat::zeros(freq.size(), CV_8U);
            keep.convertTo(tol, CV_8U, 255.0, 0.0);
            const int dk = (p.dilateK % 2 == 0) ? p.dilateK + 1 : p.dilateK;
            const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {dk, dk});
            cv::dilate(tol, tol, kernel);

            vm.rois[i].sRef = ref8;
            vm.rois[i].freq = freq8;
            vm.rois[i].tolMask = tol;
            vm.rois[i].ok = true;
        }
        if (failed) continue;

        // ---- 整图频率统计（供任意位置选区的模板使用）
        //
        // 与区域频率不同的是：这里覆盖整幅芯片，不局限于 4 个 ROI。
        // 用户可以在图上任意位置框选文字，模板直接从这里取。
        {
            const cv::Rect full(0, 0, imgs.front().cols, imgs.front().rows);
            cv::Mat sRefFull = strokeSignal(imgs.front(), full, p);
            cv::Mat refFull8;
            sRefFull.convertTo(refFull8, CV_8U, 80.0, 0.0);

            cv::Mat cntFull = cv::Mat::zeros(sRefFull.size(), CV_32F);
            int usedFull = 0;
            for (size_t k = 0; k < imgs.size(); ++k) {
                if (imgs[k].size() != imgs.front().size()) continue;
                cv::Mat s = strokeSignal(imgs[k], full, p);
                if (k > 0) {
                    cv::Mat warp;
                    if (detail::estimateWarp(s, sRefFull, warp)) s = detail::applyWarp(s, warp);
                }
                cv::Mat mask;
                cv::compare(s, p.strokeHi, mask, cv::CMP_GT);
                cv::Mat mf;
                mask.convertTo(mf, CV_32F, 1.0 / 255.0);
                cntFull += mf;
                ++usedFull;
            }
            if (usedFull > 0) {
                cv::Mat freqFull = cntFull / static_cast<double>(usedFull);
                // 全图尺寸大，存 8bit 即可（0~255 表示 0.0~1.0）
                cv::Mat freqFull8;
                freqFull.convertTo(freqFull8, CV_8U, 255.0, 0.0);
                vm.fullFreq = freqFull8;
                vm.fullRef = refFull8;
                vm.fullOk = true;
            }
        }

        vm.ok = true;
        ++built;
    }

    if (built == 0) {
        if (err) *err = "没有任何视角建模成功";
        return false;
    }
    return true;
}

}  // namespace markloc::charinsp