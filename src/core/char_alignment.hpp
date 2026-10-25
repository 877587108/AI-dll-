#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "chardet.hpp"

namespace markloc {

struct CharOffset {
    cv::Rect box;                  // Always in model coordinates.
    cv::Point2f sourceOffset;       // Input glyph position minus model position.
    double confidence = 0.0;
    bool accepted = false;
};

// Match against the ORIGINAL model box. Never take a new template from a box
// already moved into image coordinates. Search origin may be clipped at an edge.
inline std::vector<CharOffset> locateCharacters(const cv::Mat& signal, const cv::Mat& freq,
                                                const std::vector<chardet::CharUnit>& units,
                                                int radius = 24) {
    std::vector<CharOffset> out;
    if (signal.empty() || signal.type() != CV_32F || freq.size() != signal.size()) return out;
    const cv::Rect bounds(0, 0, signal.cols, signal.rows);
    for (const auto& u : units) {
        CharOffset a;
        a.box = u.bbox;
        if ((a.box & bounds) != a.box || a.box.width <= 4 || a.box.height <= 4) {
            out.push_back(a);
            continue;
        }
        const cv::Rect search = cv::Rect(a.box.x - radius, a.box.y - radius,
                                        a.box.width + 2 * radius, a.box.height + 2 * radius) & bounds;
        cv::Mat templ, response;
        freq(a.box).convertTo(templ, CV_32F, 1.0 / 255.0);
        cv::Scalar mean, stddev;
        cv::meanStdDev(templ, mean, stddev);
        if (stddev[0] < 1e-5) { out.push_back(a); continue; }
        cv::matchTemplate(signal(search), templ, response, cv::TM_CCOEFF_NORMED);
        cv::Point peak;
        cv::minMaxLoc(response, nullptr, &a.confidence, nullptr, &peak);
        // A peak on the search boundary cannot establish a complete location.
        if (!std::isfinite(a.confidence) || a.confidence < 0.50 ||
            peak.x == 0 || peak.y == 0 || peak.x + 1 == response.cols || peak.y + 1 == response.rows) {
            out.push_back(a);
            continue;
        }
        const auto subpixel = [](double l, double c, double r) {
            const double den = 2.0 * c - l - r;
            return den > 1e-9 ? std::clamp(0.5 * (r - l) / den, -0.5, 0.5) : 0.0;
        };
        const double dx = peak.x + search.x - a.box.x +
            subpixel(response.at<float>(peak.y, peak.x - 1), response.at<float>(peak),
                     response.at<float>(peak.y, peak.x + 1));
        const double dy = peak.y + search.y - a.box.y +
            subpixel(response.at<float>(peak.y - 1, peak.x), response.at<float>(peak),
                     response.at<float>(peak.y + 1, peak.x));
        a.sourceOffset = cv::Point2f(static_cast<float>(dx), static_cast<float>(dy));
        a.accepted = true;
        out.push_back(a);
    }
    return out;
}

// A character keeps one translation across its entire box; interpolate only in
// the inter-character gaps. This handles a suffix displaced relative to a prefix
// without scaling the glyphs or cutting/reflection-padding individual patches.
// All samples come from the unchanged input image, in a single remap. Output
// boxes, frequency mask and output signal therefore share MODEL coordinates.
// Optional display image receives the same map so evidence is drawn at the
// coordinates that were actually inspected.
inline cv::Mat alignCharacterSignal(const cv::Mat& signal, const std::vector<CharOffset>& offsets,
                                    cv::Mat* displayImage = nullptr) {
    struct Row { cv::Rect bounds; std::vector<CharOffset> anchors; };
    std::vector<Row> rows;
    for (const auto& a : offsets) {
        if (!a.accepted) continue;
        auto row = std::find_if(rows.begin(), rows.end(), [&](const Row& r) {
            const int overlap = std::min(r.bounds.y + r.bounds.height, a.box.y + a.box.height) -
                                std::max(r.bounds.y, a.box.y);
            return overlap > 0.5 * std::min(r.bounds.height, a.box.height);
        });
        if (row == rows.end()) rows.push_back({a.box, {a}});
        else { row->bounds |= a.box; row->anchors.push_back(a); }
    }
    if (rows.empty()) return signal.clone();
    for (auto& r : rows) std::sort(r.anchors.begin(), r.anchors.end(),
        [](const CharOffset& a, const CharOffset& b) { return a.box.x < b.box.x; });
    cv::Mat mapX(signal.size(), CV_32F), mapY(signal.size(), CV_32F);
    for (int y = 0; y < signal.rows; ++y) {
        float* mx = mapX.ptr<float>(y);
        float* my = mapY.ptr<float>(y);
        for (int x = 0; x < signal.cols; ++x) {
            mx[x] = static_cast<float>(x);
            my[x] = static_cast<float>(y);
        }
    }
    cv::Mat weights = cv::Mat::zeros(signal.size(), CV_32F);
    constexpr int feather = 32;
    const cv::Rect bounds(0, 0, signal.cols, signal.rows);
    for (const auto& r : rows) {
        const cv::Rect area = cv::Rect(r.bounds.x - feather, r.bounds.y - feather,
            r.bounds.width + 2 * feather, r.bounds.height + 2 * feather) & bounds;
        std::vector<cv::Point2f> shiftAtX(area.width);
        std::vector<float> weightAtX(area.width);
        for (int x = area.x; x < area.x + area.width; ++x) {
            cv::Point2f d = r.anchors.back().sourceOffset;
            for (size_t i = 0; i < r.anchors.size(); ++i) {
                const auto& a = r.anchors[i];
                if (x < a.box.x + a.box.width) {
                    d = a.sourceOffset;
                    if (i > 0 && x < a.box.x) {
                        const auto& prev = r.anchors[i - 1];
                        const int start = prev.box.x + prev.box.width - 1;
                        const float t = std::clamp(static_cast<float>(x - start) /
                                                   std::max(1, a.box.x - start), 0.0f, 1.0f);
                        d = prev.sourceOffset * (1.0f - t) + a.sourceOffset * t;
                    }
                    break;
                }
            }
            shiftAtX[x - area.x] = d;
            const int outside = std::max({r.bounds.x - x, x - (r.bounds.x + r.bounds.width - 1), 0});
            weightAtX[x - area.x] = std::max(0.0f, 1.0f - outside / static_cast<float>(feather));
        }
        for (int y = area.y; y < area.y + area.height; ++y) {
            float* mx = mapX.ptr<float>(y);
            float* my = mapY.ptr<float>(y);
            float* w = weights.ptr<float>(y);
            const int outside = std::max({r.bounds.y - y, y - (r.bounds.y + r.bounds.height - 1), 0});
            const float wy = std::max(0.0f, 1.0f - outside / static_cast<float>(feather));
            for (int x = area.x; x < area.x + area.width; ++x) {
                const float weight = wy * weightAtX[x - area.x];
                if (weight <= w[x]) continue;
                const auto shift = shiftAtX[x - area.x] * weight;
                // Sample input(x + dx, y + dy) at the fixed model coordinate.
                mx[x] = x + shift.x;
                my[x] = y + shift.y;
                w[x] = weight;
            }
        }
    }
    cv::Mat aligned;
    cv::remap(signal, aligned, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    if (displayImage && displayImage->size() == signal.size()) {
        cv::Mat view;
        cv::remap(*displayImage, view, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        *displayImage = view;
    }
    return aligned;
}
} // namespace markloc
