#include "core/char_core.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "chardet.hpp"
#include "core/detectors.hpp"
#include "core/warp.hpp"

#include <opencv2/imgproc.hpp>

namespace markloc::charinsp {
// ============================================================================
cv::Mat makeExplain(const cv::Mat& gray, const Params& p, const Result& r,
                    const std::string& title) {
    (void)gray;
    constexpr int LABEL_W = 96;
    constexpr int GAP = 8;

    int panelW = 0;
    std::array<int, 4> rowH{};
    for (int i = 0; i < 4; ++i) {
        if (!r.evidence[i].valid) continue;
        panelW = std::max(panelW, r.evidence[i].aligned8.cols);
        rowH[i] = r.evidence[i].aligned8.rows;
    }
    if (panelW == 0) return {};

    const int headerH = 74;
    int totalH = headerH;
    for (int i = 0; i < 4; ++i) totalH += rowH[i] + GAP;

    cv::Mat canvas(totalH, LABEL_W + 4 * panelW + 3 * GAP, CV_8UC3, cv::Scalar(24, 24, 28));

    std::string safeTitle;
    safeTitle.reserve(title.size());
    for (unsigned char ch : title) safeTitle.push_back(ch < 0x80 ? static_cast<char>(ch) : '?');
    if (safeTitle.size() > 62) safeTitle = safeTitle.substr(0, 62) + "...";

    cv::putText(canvas, safeTitle, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    char buf[192];
    std::snprintf(buf, sizeof(buf), "view score = %.3f  ->  %s", r.score, r.ng ? "NG" : "OK");
    cv::putText(canvas, buf, cv::Point(8, 42), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                r.ng ? cv::Scalar(60, 60, 255) : cv::Scalar(80, 220, 80), 1, cv::LINE_AA);

    const char* colNames[4] = {"1) original", "2) stroke S", "3) tolerance", "4) residual"};
    for (int c = 0; c < 4; ++c) {
        const int x = LABEL_W + c * (panelW + GAP);
        cv::putText(canvas, colNames[c], cv::Point(x + 4, 68), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(200, 200, 120), 1, cv::LINE_AA);
    }

    int y = headerH;
    for (int i = 0; i < 4; ++i) {
        const RoiEvidence& e = r.evidence[i];
        const RoiResult& rr = r.regions[i];

        cv::putText(canvas, p.rois[i].name, cv::Point(6, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        std::snprintf(buf, sizeof(buf), "%.2f", rr.score);
        cv::putText(canvas, buf, cv::Point(6, y + 44), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    rr.score >= kNgThreshold ? cv::Scalar(60, 60, 255) : cv::Scalar(80, 220, 80), 1,
                    cv::LINE_AA);
        if (!rr.reliable) {
            cv::putText(canvas, "ecc low", cv::Point(6, y + 64), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
        }

        if (!e.valid) {
            y += rowH[i] + GAP;
            continue;
        }

        cv::Mat p1, p2, p3, p4;
        cv::cvtColor(e.aligned8, p1, cv::COLOR_GRAY2BGR);
        cv::applyColorMap(e.s8, p2, cv::COLORMAP_JET);
        cv::cvtColor(e.tol8, p3, cv::COLOR_GRAY2BGR);
        p3.setTo(cv::Scalar(80, 200, 80), e.tol8);
        cv::cvtColor(e.aligned8, p4, cv::COLOR_GRAY2BGR);
        p4.convertTo(p4, -1, 0.40, 0.0);
        p4.setTo(cv::Scalar(0, 165, 255), e.weak8);
        p4.setTo(cv::Scalar(0, 0, 255), e.extra8);

        const cv::Mat* panels[4] = {&p1, &p2, &p3, &p4};
        for (int c = 0; c < 4; ++c) {
            const cv::Mat& src = *panels[c];
            if (src.cols > panelW || src.rows > rowH[i]) continue;
            cv::Mat dst = canvas(cv::Rect(LABEL_W + c * (panelW + GAP), y, src.cols, src.rows));
            src.copyTo(dst);
        }
        y += rowH[i] + GAP;
    }
    return canvas;
}

// ============================================================================

}  // namespace markloc::charinsp