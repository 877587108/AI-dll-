// Known displacements test the sampling direction independently of real-data labels.
#include "core/char_alignment.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        cv::Mat freq = cv::Mat::zeros(240, 600, CV_8U);
        std::vector<chardet::CharUnit> units;
        std::vector<cv::Point> truth;
        const char* glyphs = "Q4R8NF";
        for (int row = 0; row < 2; ++row) for (int i = 0; i < 6; ++i) {
            chardet::CharUnit u;
            u.bbox = cv::Rect((row == 0 ? 5 : 15) + i * 95, 15 + row * 110, 70, 85);
            u.extend = u.bbox;
            units.push_back(u);
            truth.emplace_back(row == 0 ? (i < 2 ? 0 : 3 * (i - 1)) : -8,
                               row == 0 ? 2 : -3);
            cv::putText(freq, std::string(1, glyphs[i]), u.bbox.tl() + cv::Point(8, 65),
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255), 3, cv::LINE_AA);
        }
        cv::Mat input = cv::Mat::zeros(freq.size(), CV_32F);
        for (size_t i = 0; i < units.size(); ++i) {
            cv::Mat patch;
            freq(units[i].bbox).convertTo(patch, CV_32F, 1.0 / 255);
            patch.copyTo(input(units[i].bbox + truth[i]));
        }
        const auto loc = markloc::locateCharacters(input, freq, units);
        for (size_t i = 0; i < loc.size(); ++i) {
            require(loc[i].accepted, "known glyph rejected (including clipped search window)");
            require(cv::norm(loc[i].sourceOffset - cv::Point2f(truth[i])) < 0.6,
                    "wrong displacement/direction");
        }
        cv::Mat display = input.clone();
        const auto result = markloc::alignCharacterSignal(input, loc, &display);
        require(cv::norm(display, result, cv::NORM_INF) == 0, "display and detector maps differ");
        const auto residual = markloc::locateCharacters(result, freq, units);
        for (const auto& a : residual) {
            require(a.accepted && cv::norm(a.sourceOffset) < 0.6, "prefix/suffix drift remains");
        }
        require(cv::norm(result, markloc::alignCharacterSignal(input, loc), cv::NORM_INF) == 0,
                "repeated detection depends on prior state");
        auto blank = markloc::locateCharacters(cv::Mat::zeros(input.size(), CV_32F), freq, units);
        for (const auto& a : blank) require(!a.accepted, "blank image accepted as located");
        auto flat = markloc::locateCharacters(input, cv::Mat::zeros(freq.size(), CV_8U), units);
        for (const auto& a : flat) require(!a.accepted, "constant template accepted");
        require(cv::norm(input, markloc::alignCharacterSignal(input, blank), cv::NORM_INF) == 0,
                "no reliable anchors must preserve image");
        std::cout << "PASS: prefix/suffix and row shifts, clipped search, fixed coordinates, "
                     "display parity, repeatability, blank inputs\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
