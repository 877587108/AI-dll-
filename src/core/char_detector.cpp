#include "core/char_detector.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "chardet.hpp"
#include "core/warp.hpp"

namespace markloc::charinsp::chars {

void appendResults(const cv::Mat& gray, const ViewModel& vm, const Params& p, Result& res) {
    if (!p.checkChars || !vm.hasManualBoxes || !vm.fullOk || vm.fullFreq.empty() ||
        !vm.chardetModel) {
        return;
    }

    const cv::Rect full(0, 0, gray.cols, gray.rows);
    cv::Mat sFull = strokeSignal(gray, full, p);
    cv::Mat alignedFull = sFull;
    if (!vm.fullRef.empty() && vm.fullRef.size() == sFull.size()) {
        cv::Mat refFullF;
        vm.fullRef.convertTo(refFullF, CV_32F, 1.0 / 80.0);
        cv::Mat warp;
        double eccFull = -1;
        if (detail::estimateWarp(sFull, refFullF, warp, &eccFull)) {
            alignedFull = detail::applyWarp(sFull, warp);
        }
    }

    chardet::Config cfg;
    cfg.sigma = p.sigma;
    cfg.strokeHi = p.strokeHi;
    cfg.missStrokeLo = p.missStrokeLo;
    cfg.missFreq = p.missFreq;
    cfg.charMissRatio = p.charMissRatio;
    cfg.charExtraRatio = p.charExtraRatio;
    cfg.charT = p.charT;

    const chardet::Model& cm = *vm.chardetModel;
    double worstChar = 0.0;
    std::vector<chardet::CharUnit> allUnits;
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        const auto units = chardet::toUnits(cm, static_cast<int>(ri));
        for (const auto& u : units) allUnits.push_back(u);

        const auto matches = chardet::match(alignedFull, units, cfg);
        for (const auto& mc : matches) {
            CharResult cr;
            cr.roiIndex = static_cast<int>(ri);
            cr.charIndex = mc.index;
            cr.rect = mc.bbox;
            cr.score = mc.score;
            cr.missScore = mc.missScore;
            cr.extraScore = mc.extraScore;
            cr.shapeScore = mc.shapeScore;
            cr.missRatio = mc.missRatio;
            cr.extraRatio = mc.extraRatio;
            cr.label = mc.label;
            res.chars.push_back(std::move(cr));
            worstChar = std::max(worstChar, mc.score);
        }
    }

    if (p.useCharNcc) {
        for (const auto& u : allUnits) {
            const auto nc = chardet::diffNcc(alignedFull, cm.freq, u, cfg);
            if (!nc.valid || nc.score <= 0.0) continue;

            CharResult cr;
            cr.roiIndex = -1;
            cr.charIndex = u.index;
            cr.rect = u.bbox;
            cr.score = nc.score;
            cr.missScore = nc.missScore;
            cr.extraScore = nc.extraScore;
            cr.missRatio = nc.missRatio;
            cr.extraRatio = nc.extraRatio;
            cr.label = nc.missRatio > nc.extraRatio ? "判据2:缺墨" : "判据2:多墨";
            res.chars.push_back(std::move(cr));
            worstChar = std::max(worstChar, nc.score);
        }
    }

    if (!allUnits.empty()) {
        const auto bd = chardet::diffBlocks(alignedFull, cm.freq, allUnits, cfg, 8);
        if (bd.valid && (bd.extraCount > 0 || bd.missingCount > 0)) {
            worstChar = std::max(worstChar, 1.5);
            res.blockExtra = bd.extraCount;
            res.blockMissing = bd.missingCount;
        }
    }

    res.score = std::max(res.score, worstChar);
    if (worstChar > res.regionMax) {
        res.regionMax = worstChar;
        res.worstRegion = -1;
    }
}

} // namespace markloc::charinsp::chars
