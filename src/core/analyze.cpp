#include "core/char_core.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "core/char_detector.hpp"
#include "core/detectors.hpp"
#include "core/warp.hpp"

#include <opencv2/imgproc.hpp>

namespace markloc::charinsp {
// ============================================================================
Result analyze(const cv::Mat& gray, const ViewModel& vm, const Params& p, bool wantEvidence) {
    Result res;
    if (!vm.ok) return res;

    for (int i = 0; i < 4; ++i) {
        const Roi& roi = p.rois[i];
        const ModelRoi& mr = vm.rois[i];
        if (!roi.enabled) continue;   // 该视角下本区域已关闭
        if (!mr.ok) continue;
        if (roi.rect.x < 0 || roi.rect.y < 0 || roi.rect.x + roi.rect.width > gray.cols ||
            roi.rect.y + roi.rect.height > gray.rows) {
            continue;
        }

        RoiResult rr;
        RoiEvidence ev;

        cv::Mat sImg = strokeSignal(gray, roi.rect, p);
        cv::Mat refF, imgF;
        mr.sRef.convertTo(refF, CV_32F, 1.0 / 80.0);
        sImg.convertTo(imgF, CV_32F);

        cv::Mat patchF;
        gray(roi.rect).convertTo(patchF, CV_32F);

        cv::Mat warp;
        cv::Mat aligned = imgF;
        cv::Mat alignedPatch = patchF;
        double ecc = -1;
        if (detail::estimateWarp(imgF, refF, warp, &ecc)) {
            aligned = detail::applyWarp(imgF, warp);
            alignedPatch = detail::applyWarp(patchF, warp);
        }
        rr.ecc = ecc;
        if (mr.tolMask.size() != aligned.size()) {
            continue;   // 模板与图像 ROI 尺寸不符，跳过该区域
        }

        cv::Mat usable = cv::Mat::ones(aligned.size(), CV_8U) * 255;
        const int h = aligned.rows, w = aligned.cols;
        if (h > 2 * p.edgeTB && w > 2 * p.edgeLR) {
            usable(cv::Rect(0, 0, w, p.edgeTB)).setTo(0);
            usable(cv::Rect(0, h - p.edgeTB, w, p.edgeTB)).setTo(0);
            usable(cv::Rect(0, 0, p.edgeLR, h)).setTo(0);
            usable(cv::Rect(w - p.edgeLR, 0, p.edgeLR, h)).setTo(0);
        }

        cv::Mat inTol;
        cv::compare(mr.tolMask, 127, inTol, cv::CMP_GT);
        cv::Mat allowed;
        cv::bitwise_not(inTol, allowed);

        auto makeResidual = [&](double thr) {
            cv::Mat strong;
            cv::compare(aligned, thr, strong, cv::CMP_GT);
            cv::Mat r;
            cv::bitwise_and(strong, allowed, r);
            cv::bitwise_and(r, usable, r);
            cv::Mat r8;
            r.convertTo(r8, CV_8U, 255.0);
            return r8;
        };

        cv::Mat extra = makeResidual(p.strokeHi);
        cv::Mat weak = makeResidual(p.strokeLo);

        // 检测分支通过统一契约执行。替换某一算法时只需改对应 detector 实现。
        const det::DetectorInput detectorInput{aligned, extra, weak, mr.freq, p, roi.checkAppendage};
        double regionScore = det::runDetectors(detectorInput, rr);

        // 分支E 字符级比对在 ROI 循环外统一处理（见下方，基于整图频率）

        // 可靠性门控：ECC 不足则该区域不计分。
        // 配准失败时的残差是错位伪差，不能当真。
        const double eccGate = p.eccMin;
        rr.reliable = (ecc >= eccGate);
        if (!rr.reliable) regionScore = 0.0;
        rr.score = regionScore;
        if (wantEvidence) {
            alignedPatch.convertTo(ev.aligned8, CV_8U);
            aligned.convertTo(ev.s8, CV_8U, 80.0, 0.0);
            ev.tol8 = mr.tolMask.clone();
            ev.extra8 = extra.clone();
            ev.weak8 = weak.clone();
            ev.valid = true;
            res.evidence[i] = std::move(ev);
        }

        res.regions[i] = rr;
        res.score = std::max(res.score, regionScore);
        if (regionScore > res.regionMax) {
            res.regionMax = regionScore;
            res.worstRegion = i;
        }
    }

    // 字符级检测是独立适配模块，避免主流程直接依赖 chardet 的模型和函数。
    chars::appendResults(gray, vm, p, res);

    res.ng = res.score >= kNgThreshold;
    return res;
}

// ============================================================================
ProductResult analyzeProduct(const std::array<cv::Mat, kViewCount>& images,
                             const std::array<bool, kViewCount>& present, const Model& m,
                             const std::array<Params, kViewCount>& ps, bool wantEvidence) {
    ProductResult pr;
    for (int v = 0; v < kViewCount; ++v) {
        if (!present[v] || images[v].empty() || !m.views[v].ok) continue;
        if (!ps[static_cast<size_t>(v)].enabled) continue;   // 未启用则不参与判定
        pr.present[v] = true;
        pr.views[v] = analyze(images[v], m.views[v], ps[v], wantEvidence);
        pr.viewScore[v] = pr.views[v].score;
        pr.regionScore = std::max(pr.regionScore, pr.views[v].regionMax);
        if (pr.viewScore[v] > pr.score) {
            pr.score = pr.viewScore[v];
            pr.worstView = v;
        }
    }
    // 产品分 = 各视角四区域最高分
    pr.score = pr.regionScore;
    pr.ng = pr.score >= kNgThreshold;
    return pr;
}

}  // namespace markloc::charinsp
