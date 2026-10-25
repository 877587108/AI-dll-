// Suspect candidate ranking.
//
// The client asked for a high-recall pass: show every place the detector finds
// suspicious, score it, and let a human confirm before the sample is filed.
// This tool produces that ranking. It does not decide NG; it only orders
// candidates so the reviewer sees the worst ones first.
//
// Scoring is deliberately coarse (four bands, no numeric score) because the
// underlying features are not separable enough to justify a probability.
//
// Usage:
//   markloc_suspect <modelDir> <outDir> <image...>
//   markloc_suspect <modelDir> <outDir> --dir <folder> [--view 00]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#include "chardet.hpp"

namespace fs = std::filesystem;

namespace {

fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

std::string toU8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

#ifdef _WIN32
std::string wideToU8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                        nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n,
                          nullptr, nullptr);
    return out;
}

std::vector<std::string> wideArgs() {
    std::vector<std::string> out;
    int argc = 0;
    LPWSTR* argv = ::CommandLineToArgvW(::GetCommandLineW(), &argc);
    if (!argv) return out;
    for (int i = 0; i < argc; ++i) out.push_back(wideToU8(argv[i]));
    ::LocalFree(argv);
    return out;
}
#endif

bool readGray(const std::string& path, cv::Mat& out) {
#ifdef _WIN32
    const std::wstring w = u8p(path).wstring();
    std::ifstream f(w.c_str(), std::ios::binary);
#else
    std::ifstream f(u8p(path), std::ios::binary);
#endif
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0) return false;
    std::vector<uchar> b(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(b.data()), n);
    if (!f) return false;
    out = cv::imdecode(b, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

bool writePng(const std::string& path, const cv::Mat& img) {
    std::vector<uchar> buf;
    if (!cv::imencode(".png", img, buf)) return false;
#ifdef _WIN32
    const std::wstring w = u8p(path).wstring();
    std::ofstream f(w.c_str(), std::ios::binary);
#else
    std::ofstream f(u8p(path), std::ios::binary);
#endif
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return static_cast<bool>(f);
}

// One suspicious place: a straight run of excess ink, with how much character
// box it overlaps and how bright it is relative to the background.
struct Cand {
    cv::Point p1, p2;
    double len = 0;
    double energy = 0;
    double meanGray = 0;
    double onCharRatio = 0;   // fraction of the segment lying inside a char box
    int charsHit = 0;
    // Depth, measured as a perpendicular profile rather than as an absolute
    // intensity sum. Length alone cannot separate a shallow scuff from a real
    // scratch, and the raw energy depends on exposure; how much brighter the
    // mark is than the surface immediately beside it, and how wide that bright
    // band is, are the two things the eye actually uses.
    double depth = 0;         // peak-minus-background contrast, 0..255 scale
    double widthPx = 0;       // full width at half maximum of the profile
    int band = 0;             // 0 light, 1 suspect, 2 obvious, 3 certain
};

// Median of a small vector, used for the profile background. The background is
// taken from the outer part of the perpendicular profile so the scratch itself
// cannot bias it.
double medianOf(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Perpendicular depth profile at one point.
//
// Returns the contrast of the brightest sample in the middle of the profile
// against the median of its two ends, and writes the half-maximum width.
double depthAt(const cv::Mat& sig, cv::Point2f c, cv::Point2f perp, double scale,
               double* widthOut) {
    constexpr int kReach = 12;   // half-length of the profile
    std::vector<double> v(2 * kReach + 1, 0.0);
    for (int k = -kReach; k <= kReach; ++k) {
        const float x = c.x + perp.x * k;
        const float y = c.y + perp.y * k;
        const int xi = static_cast<int>(std::lround(x));
        const int yi = static_cast<int>(std::lround(y));
        if (xi < 0 || yi < 0 || xi >= sig.cols || yi >= sig.rows) continue;
        v[k + kReach] = sig.at<float>(yi, xi) * scale;
    }
    std::vector<double> ends;
    for (int k = 0; k < kReach - 4; ++k) {
        ends.push_back(v[k]);
        ends.push_back(v[v.size() - 1 - k]);
    }
    const double bg = medianOf(ends);
    double peak = 0.0;
    for (int k = kReach - 5; k <= kReach + 5; ++k) peak = std::max(peak, v[k]);
    const double contrast = peak - bg;

    int w = 0;
    const double half = bg + contrast * 0.5;
    for (int k = kReach - 6; k <= kReach + 6; ++k) {
        if (v[k] >= half) ++w;
    }
    if (widthOut) *widthOut = static_cast<double>(w);
    return contrast;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef _WIN32
    const std::vector<std::string> av = wideArgs();
#else
    const std::vector<std::string> av(argv, argv + argc);
#endif
    if (av.size() < 4) {
        std::cout << "usage: markloc_suspect <modelDir> <outDir> <image...>\n";
        std::cout << "       markloc_suspect <modelDir> <outDir> --dir <folder> [--view 00]\n";
        std::cout << "              [--tag name] [--gate 1000]\n";
        return 2;
    }
    const std::string modelDir = av[1];
    const std::string outDir = av[2];
    std::vector<std::string> files;
    std::string viewFilter = "00";
    std::string tag;
    // Report gate, mirroring the GUI's scratchMinComposite. Kept as a flag so the
    // measurements stay visible even when they sit below the decision line: a
    // suppressed candidate is evidence too, and hiding it would make the
    // calibration impossible to audit.
    double gate = 1000.0;

    for (size_t i = 3; i < av.size(); ++i) {
        const std::string& a = av[i];
        if (a == "--dir" && i + 1 < av.size()) {
            const fs::path dir = u8p(av[++i]);
            std::error_code ec;
            std::vector<std::string> found;
            for (const auto& e : fs::directory_iterator(dir, ec)) {
                if (!e.is_regular_file()) continue;
                const std::string n = toU8(e.path().filename());
                if (n.size() < 4 || n.compare(n.size() - 4, 4, ".tif") != 0) continue;
                if (!viewFilter.empty()) {
                    const std::string suf = "_" + viewFilter + ".tif";
                    if (n.size() < suf.size() ||
                        n.compare(n.size() - suf.size(), suf.size(), suf) != 0) {
                        continue;
                    }
                }
                found.push_back(toU8(e.path()));
            }
            std::sort(found.begin(), found.end());
            files.insert(files.end(), found.begin(), found.end());
        } else if (a == "--view" && i + 1 < av.size()) {
            viewFilter = av[++i];
        } else if (a == "--tag" && i + 1 < av.size()) {
            tag = av[++i];
        } else if (a == "--gate" && i + 1 < av.size()) {
            gate = std::stod(av[++i]);
        } else {
            files.push_back(a);
        }
    }
    if (files.empty()) {
        std::cout << "no images\n";
        return 1;
    }

    chardet::Model model;
    if (!chardet::loadModel(modelDir, model) || !model.valid()) {
        std::cout << "model load failed\n";
        return 1;
    }
    chardet::Config cfg;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    // Minimum scratch length, in pixels.
    //
    // Three character widths (210 px) was tried and rejected: 0026's confirmed
    // scratch is 123 px long, so the gate threw away a real defect while it did
    // suppress the stroke residual that had been ranking first. The residual and
    // the real scratch overlap in length, so length alone cannot separate them;
    // 60 px only drops specks.
    constexpr double kMinScratchLen = 60.0;

    std::cout << "rank,band,reported,file,candidates,bestLen,bestEnergy,onChar,charsHit,"
                 "depth,width,maxDepth,composite,bx,by\n";

    struct Row {
        std::string file;
        int band = 0;
        bool reported = false;
        int nCand = 0;
        double bestLen = 0, bestEnergy = 0, onChar = 0;
        double bestDepth = 0, bestWidth = 0;
        double maxDepth = 0;
        double composite = 0;
        cv::Point bestAt;
        int charsHit = 0;
        cv::Mat vis;
    };
    std::vector<Row> rows;

    for (const auto& path : files) {
        cv::Mat g;
        if (!readGray(path, g) || g.size() != model.freq.size()) continue;

        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);

        // whole-image registration, same target as the GUI
        if (!model.ref.empty()) {
            cv::Mat r32, s32;
            model.ref.convertTo(r32, CV_32F);
            sig.convertTo(s32, CV_32F);
            cv::Rect inkArea;
            for (const auto& u : base) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
            }
            if (inkArea.area() > 0) {
                inkArea.x = std::max(0, inkArea.x - 20);
                inkArea.y = std::max(0, inkArea.y - 20);
                inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
                inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
                cv::Mat rC, sC;
                r32(inkArea).copyTo(rC);
                s32(inkArea).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat shifted;
                    cv::warpAffine(sig, shifted, M, sig.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE);
                    sig = shifted;
                }
            }
        }

        // ---- excess ink inside the character band
        cv::Rect band;
        for (const auto& u : base) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            band = band.area() > 0 ? (band | b) : b;
        }
        if (band.area() <= 0) continue;

        cv::Mat weak;
        cv::compare(sig(band), cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
        cv::Mat modelInk;
        cv::compare(model.freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk, cv::CMP_GT);
        cv::Mat modelDil, notModel;
        cv::dilate(modelInk, modelDil, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
        cv::bitwise_not(modelDil, notModel);
        cv::bitwise_and(weak, notModel, weak);

        // re-join the pieces a character cut the scratch into
        const cv::Mat kb = cv::getStructuringElement(cv::MORPH_ELLIPSE, {25, 25});
        cv::morphologyEx(weak, weak, cv::MORPH_CLOSE, kb);
        cv::medianBlur(weak, weak, 3);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(weak, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);

        std::vector<Cand> cands;
        const cv::Mat sBand = sig(band);
        for (const auto& ln : lines) {
            const cv::Point p1(ln[0], ln[1]), p2(ln[2], ln[3]);
            const double len = cv::norm(p2 - p1);
            // Minimum scratch length, in pixels.
            //
            // A character stroke that survives the model subtraction leaves a
            // short residual along the stroke; measuring the box widths gives a
            // median of 70 px, so anything below three characters (210 px) is
            // more likely a stroke residual than a scratch. A real defect runs
            // across the text line and is much longer.
            if (len < kMinScratchLen) continue;

            const int steps = std::max(2, static_cast<int>(len));
            double energy = 0;
            int covered = 0, onChar = 0;
            for (int t = 0; t <= steps; ++t) {
                const double rr = static_cast<double>(t) / steps;
                const int x = static_cast<int>(p1.x + (p2.x - p1.x) * rr);
                const int y = static_cast<int>(p1.y + (p2.y - p1.y) * rr);
                if (y < 0 || y >= weak.rows || x < 0 || x >= weak.cols) continue;
                const int ax = x + band.x, ay = y + band.y;
                for (int dy = -4; dy <= 4; ++dy) {
                    const int yy = std::clamp(y + dy, 0, weak.rows - 1);
                    if (weak.at<uchar>(yy, x)) {
                        energy += std::max(0.0f, sBand.at<float>(yy, x) -
                                                     static_cast<float>(cfg.missStrokeLo));
                        if (std::abs(dy) <= 1) {
                            ++covered;
                            break;
                        }
                    }
                }
                for (const auto& u : base) {
                    if ((u.bbox & cv::Rect(ax, ay, 1, 1)).area() > 0) {
                        ++onChar;
                        break;
                    }
                }
            }
            const double ocr = static_cast<double>(onChar) / (steps + 1);
            if (covered < steps / 4) continue;

            Cand c;
            c.p1 = p1 + cv::Point(band.x, band.y);
            c.p2 = p2 + cv::Point(band.x, band.y);
            c.len = len;
            c.energy = energy * len;
            c.meanGray = energy / std::max(1.0, static_cast<double>(covered));
            c.onCharRatio = ocr;

            // Depth: average a perpendicular profile at several points along the
            // run. Averaging keeps one lucky bright pixel from defining the
            // candidate, and the width is averaged the same way.
            {
                const cv::Point2f d(c.p2.x - c.p1.x, c.p2.y - c.p1.y);
                const double dn = std::max(1e-6, cv::norm(d));
                const cv::Point2f dir(static_cast<float>(d.x / dn), static_cast<float>(d.y / dn));
                const cv::Point2f perp(-dir.y, dir.x);
                const double scale = 255.0 / std::max(1e-6, cfg.strokeHi);
                const int steps2 = std::max(4, static_cast<int>(len / 6.0));
                double sumDepth = 0, sumWidth = 0;
                int nSamp = 0;
                for (int t = 1; t < steps2; ++t) {
                    const double f = static_cast<double>(t) / steps2;
                    const cv::Point2f q(static_cast<float>(c.p1.x + d.x * f),
                                        static_cast<float>(c.p1.y + d.y * f));
                    double w = 0;
                    sumDepth += depthAt(sig, q, perp, scale, &w);
                    sumWidth += w;
                    ++nSamp;
                }
                if (nSamp > 0) {
                    c.depth = sumDepth / nSamp;
                    c.widthPx = sumWidth / nSamp;
                }
            }

            for (const auto& u : base) {
                cv::Rect seg(std::min(c.p1.x, c.p2.x), std::min(c.p1.y, c.p2.y),
                             std::abs(c.p2.x - c.p1.x) + 1, std::abs(c.p2.y - c.p1.y) + 1);
                if ((u.bbox & seg).area() > 0) ++c.charsHit;
            }
            cands.push_back(c);
        }

        // Keep the strongest candidate per image; the sheet shows one row per
        // image and the rest would only add noise.
        Cand* best = nullptr;
        double maxDepth = 0;
        for (auto& c : cands) {
            const double sc = c.depth * c.charsHit;
            if (!best || sc > best->depth * best->charsHit) best = &c;
            maxDepth = std::max(maxDepth, c.depth);
        }

        Row r;
        r.file = toU8(u8p(path).filename());
        r.nCand = static_cast<int>(cands.size());
        r.maxDepth = maxDepth;
        if (best) {
            r.bestLen = best->len;
            r.bestEnergy = best->energy;
            r.bestDepth = best->depth;
            r.bestWidth = best->widthPx;
            r.bestAt = best->p1;
            r.onChar = best->onCharRatio;
            r.charsHit = best->charsHit;
            // Composite severity: how deep the mark is, times how many
            // characters it reaches across.
            //
            // Neither term works alone. Depth alone puts OK parts with a deep
            // scuff away from the text at the top — measured, the deepest OK
            // mark is 1204 against an NG range of 78..1038. Span alone is blind
            // to a shallow line. Multiplied they mostly separate, but not
            // cleanly: the seven NG images read 2289 2256 2209 2040 1321 1219
            // (0110 is only 73), while three OK images read 1430 1182 1080.
            // Those three are 0109_RT, 0109_FT and 0101_FT — the customer has
            // confirmed 0101 really is a defect (so that row is a correct
            // detection), and 0109's two shots are the genuine false positives.
            // 0112_RT at 1219 sits below 0109_RT at 1430, so the two populations
            // genuinely overlap and no threshold can separate them: the marks on
            // 0109 are 477/591 deep across three characters, the same magnitude
            // as a real scratch. The bands below mirror the GUI defaults and are
            // deliberately coarse: four bands, not a probability.
            const double composite = best->depth * best->charsHit;
            if (composite >= 1900.0) {
                r.band = 3;
            } else if (composite >= 1300.0) {
                r.band = 2;
            } else if (composite >= 1000.0) {
                r.band = 1;
            } else {
                r.band = 0;
            }
            r.reported = composite >= gate;
            r.composite = composite;
        }

        cv::cvtColor(g, r.vis, cv::COLOR_GRAY2BGR);
        for (const auto& u : base) {
            cv::rectangle(r.vis, u.bbox, cv::Scalar(80, 200, 80), 1);
        }
        if (best) {
            static const cv::Scalar kBands[4] = {
                cv::Scalar(160, 160, 160),  // light grey  - faint
                cv::Scalar(0, 215, 255),    // yellow      - suspect
                cv::Scalar(0, 140, 255),    // orange      - obvious
                cv::Scalar(0, 0, 255)};     // red         - certain
            static const char* kNames[4] = {"faint", "suspect", "obvious", "certain"};
            cv::line(r.vis, best->p1, best->p2, kBands[r.band], 3, cv::LINE_AA);
            cv::putText(r.vis, kNames[r.band], cv::Point(best->p1.x, std::max(20, best->p1.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, kBands[r.band], 2);
        }
        rows.push_back(std::move(r));
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.composite > b.composite;
    });

    std::error_code ec;
    fs::create_directories(u8p(outDir), ec);
    int rank = 0;
    for (const auto& r : rows) {
        ++rank;
        std::cout << rank << ',' << r.band << ',' << (r.reported ? 1 : 0) << ',' << r.file << ','
                  << r.nCand << ',' << std::fixed
                  << std::setprecision(0) << r.bestLen << ',' << r.bestEnergy << ','
                  << std::setprecision(2) << r.onChar << ',' << r.charsHit << ','
                  << std::setprecision(1) << r.bestDepth << ',' << r.bestWidth << ','
                  << r.maxDepth << ',' << std::setprecision(0) << r.composite << ','
                  << r.bestAt.x << ',' << r.bestAt.y << '\n';
        std::ostringstream nm;
        nm << outDir << "/" << (tag.empty() ? std::string("s") : tag) << std::setw(2)
           << std::setfill('0') << rank << "_b" << r.band << ".png";
        writePng(nm.str(), r.vis);
    }

    // ---- numbered contact sheet
    //
    // The reviewer works by number, so the candidates go on one sheet with a
    // large index on each tile. Cropping around the candidate rather than
    // showing the whole part keeps the mark legible: at full frame the marks are
    // a few pixels wide and impossible to judge.
    // The sheet is what the reviewer actually looks at, so it carries only the
    // rows that pass the gate. Sub-gate rows stay in the CSV as evidence.
    {
        constexpr int kTile = 320;
        constexpr int kCols = 4;
        constexpr size_t kMax = 16;
        std::vector<const Row*> shown;
        for (const auto& r : rows) {
            if (r.reported) shown.push_back(&r);
        }
        const size_t n = std::min(shown.size(), kMax);
        const int nRows = static_cast<int>((n + kCols - 1) / kCols);
        if (nRows > 0) {
            cv::Mat sheet(nRows * kTile, kCols * kTile, CV_8UC3, cv::Scalar(32, 32, 32));
            for (size_t i = 0; i < n; ++i) {
                const Row& r = *shown[i];
                const cv::Rect area(r.bestAt.x - kTile / 2, r.bestAt.y - kTile / 2, kTile, kTile);
                cv::Mat tile = cv::Mat::zeros(kTile, kTile, CV_8UC3);
                const cv::Rect valid = area & cv::Rect(0, 0, r.vis.cols, r.vis.rows);
                if (valid.area() > 0) {
                    const cv::Rect dst(valid.x - area.x, valid.y - area.y, valid.width,
                                       valid.height);
                    r.vis(valid).copyTo(tile(dst));
                }
                // number in the corner, own background so it reads on any image
                const int cx = (static_cast<int>(i) % kCols) * kTile;
                const int cy = (static_cast<int>(i) / kCols) * kTile;
                cv::rectangle(tile, cv::Rect(0, 0, 96, 44), cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(tile, std::to_string(i + 1), cv::Point(8, 36),
                            cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 255, 255), 3);
                cv::rectangle(tile, cv::Rect(1, 1, kTile - 2, kTile - 2),
                              cv::Scalar(70, 70, 70), 1);
                tile.copyTo(sheet(cv::Rect(cx, cy, kTile, kTile)));
            }
            std::ostringstream snm;
            snm << outDir << "/" << (tag.empty() ? std::string("sheet") : tag + "_sheet")
                << ".png";
            writePng(snm.str(), sheet);
        }
    }
    return 0;
}
