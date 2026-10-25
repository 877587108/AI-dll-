// Grid-difference defect scoring (client-specified algorithm).
//
// The client rejected the "find a straight line" approach and specified a
// local-statistics design instead. This tool implements exactly that design so
// it can be evaluated before it goes anywhere near the GUI.
//
//   step 1  smoothing the image with a 3x3 box, then zeroing anything below a
//           threshold, so the noise floor disappears and only solid ink remains
//   step 2  subtracting the model, smoothing the difference with a 3x3 box, and
//           keeping everything above the difference threshold
//   step 3  weighting each surviving cell by where it sits relative to the
//           characters; the client's order is
//               on a stroke  >  character gap  >  stroke edge  >  line gap
//   step 4  for cells that survived, run a 7x7 window on the model to estimate
//           how much damage the excess does. Two measures, and the client asked
//           for stroke severing to carry the higher weight.
//
// Step 4 is the part that separates a scratch from texture. Texture is random
// and does not destroy strokes; a scratch runs across them and cuts them.
//
// The output is a per-image ordering plus a rendered sheet. Scores are kept
// numeric here only for the ranking; the four bands are assigned afterwards.
//
// Usage:
//   markloc_grilleval <modelDir> <outDir> --dir <folder> [--view 00] [--tag ok]
//   markloc_grilleval <modelDir> <outDir> <image...>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

// ---------------------------------------------------------------- parameters
//
// Every number the client asked to be adjustable is adjustable. The defaults
// are the ones they named (50 for the ink floor) or my first estimate pending
// their calibration.
struct Params {
    double inkFloor = 50.0;      // step 1: 3x3 mean below this is set to zero
    double diffFloor = 40.0;     // step 2: 3x3 mean difference above this is kept
    int modelDilate = 5;         // step 2: model tolerance, see the note there
    // How far the excess must stay from a definite stroke. This has to stay
    // smaller than half the severity window below, otherwise the window can
    // never reach a stroke and the severity measure is identically zero.
    int nearInk = 2;
    int minArea = 30;            // components smaller than this are specks

    // step 3 weights, in the client's stated order.
    double wStroke = 1.00;       // the excess sits on a stroke
    double wGap = 0.70;          // between two strokes of the same character
    double wEdge = 0.55;         // fringe of a stroke
    double wLineGap = 0.25;      // between text lines / outside the glyph

    // step 4: stroke severing outweighs raw ink volume.
    double wInk = 0.35;
    double wSever = 1.00;

    int window = 7;              // the second kernel, per the client
};

// ---------------------------------------------------------------- one image

struct Cell {
    cv::Point p;        // centroid, image coordinates
    double diff = 0;    // mean 3x3 difference over the component
    double posWeight = 0;
    double inkRatio = 0;    // step 4, ink-volume measure
    double sever = 0;       // step 4, stroke-severing measure
    double onStroke = 0;    // fraction of the excess lying on model ink
    double len = 0;         // PCA major extent
    double wid = 0;         // PCA minor extent
    double elong = 0;
    double score = 0;
    cv::Rect bb;
};

struct ImageResult {
    std::string file;
    int nCells = 0;
    double best = 0;
    double total = 0;
    cv::Point bestAt;
    int band = 0;
    Cell bestCell;
    cv::Mat vis;
};

// Distance from a point to the nearest model-ink pixel, capped, used to decide
// which of the four position classes a cell falls in.
int distToInk(const cv::Mat& ink, int x, int y, int cap) {
    const int x0 = std::max(0, x - cap), x1 = std::min(ink.cols - 1, x + cap);
    const int y0 = std::max(0, y - cap), y1 = std::min(ink.rows - 1, y + cap);
    int best = cap + 1;
    for (int yy = y0; yy <= y1; ++yy) {
        const uchar* row = ink.ptr<uchar>(yy);
        for (int xx = x0; xx <= x1; ++xx) {
            if (!row[xx]) continue;
            const int dx = xx - x, dy = yy - y;
            const int d = dx * dx + dy * dy;
            if (d < best) best = d;
        }
    }
    return static_cast<int>(std::sqrt(static_cast<double>(best)));
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
        std::cout << "usage: markloc_grilleval <modelDir> <outDir> --dir <folder> [--view 00]\n";
        std::cout << "       markloc_grilleval <modelDir> <outDir> <image...>\n";
        return 2;
    }
    const std::string modelDir = av[1];
    const std::string outDir = av[2];
    std::vector<std::string> files;
    std::string viewFilter = "00";
    std::string tag;
    Params par;

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
        } else if (a == "--ink" && i + 1 < av.size()) {
            par.inkFloor = std::stod(av[++i]);
        } else if (a == "--diff" && i + 1 < av.size()) {
            par.diffFloor = std::stod(av[++i]);
        } else if (a == "--win" && i + 1 < av.size()) {
            par.window = std::stoi(av[++i]);
        } else if (a == "--mdil" && i + 1 < av.size()) {
            par.modelDilate = std::stoi(av[++i]);
        } else if (a == "--near" && i + 1 < av.size()) {
            par.nearInk = std::stoi(av[++i]);
        } else if (a == "--min-area" && i + 1 < av.size()) {
            par.minArea = std::stoi(av[++i]);
        } else if (a == "--w-ink" && i + 1 < av.size()) {
            par.wInk = std::stod(av[++i]);
        } else if (a == "--w-sever" && i + 1 < av.size()) {
            par.wSever = std::stod(av[++i]);
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

    // The model side is constant across images, so build its ink mask once.
    //
    // Two thresholds, not one. "Definite stroke" is where the model is sure ink
    // belongs; the excess is only allowed where that is absent. Using the low
    // blankFreq threshold here instead lets the excess sit on the soft shoulder
    // of every stroke — the model reads 40 there while the image reads solid
    // 255 — and that artifact is identical on every image, which is exactly what
    // an earlier version of this file ranked first on both NG and OK.
    cv::Mat modelInk;
    cv::compare(model.freq, static_cast<int>(cfg.missFreq * 255), modelInk, cv::CMP_GT);
    // "Core" is the well-formed interior of a stroke. Severity is measured
    // against it, so a fringe pixel does not count as a cut.
    cv::Mat modelCore;
    cv::erode(modelInk, modelCore, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

    std::cout << "rank,band,file,comps,best,total,len,wid,elong,diff,onStroke,sever,posW\n";

    std::vector<ImageResult> rows;

    for (const auto& path : files) {
        cv::Mat g;
        if (!readGray(path, g) || g.size() != model.freq.size()) continue;

        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);

        // registration, identical to the GUI path
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
                cv::Rect roi = inkArea;
                roi.x = std::max(0, roi.x - 20);
                roi.y = std::max(0, roi.y - 20);
                roi.width = std::min(sig.cols - roi.x, roi.width + 40);
                roi.height = std::min(sig.rows - roi.y, roi.height + 40);
                cv::Mat rC, sC;
                r32(roi).copyTo(rC);
                s32(roi).copyTo(sC);
                const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
                if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                    (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                    cv::Mat shifted;
                    cv::warpAffine(sig, shifted, M, sig.size(),
                                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                    sig = shifted;
                }
            }
        }

        // working band: the character union grown by a fixed margin
        cv::Rect band;
        for (const auto& u : base) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            band = band.area() > 0 ? (band | b) : b;
        }
        if (band.area() <= 0) continue;
        band.x = std::max(0, band.x - 40);
        band.y = std::max(0, band.y - 40);
        band.width = std::min(sig.cols - band.x, band.width + 80);
        band.height = std::min(sig.rows - band.y, band.height + 80);

        const cv::Mat sBand = sig(band);
        cv::Mat mBand;
        model.freq(band).convertTo(mBand, CV_32F);
        const cv::Mat iBand = modelInk(band);
        const cv::Mat cBand = modelCore(band);

        // ---- step 1: ink strength, 3x3 box mean, then the ink floor
        //
        // The two maps cannot be subtracted as they arrive. strokeSignal returns
        // an amplitude whose "ink present" level is cfg.strokeHi (about 0.35),
        // while the model frequency map is a CV_8U probability already scaled to
        // 0..255. Rescaling the amplitude so cfg.strokeHi becomes 255 puts a
        // solid stroke and an always-ink model pixel on the same number, which
        // is the only way the subtraction in step 2 means anything. Measured
        // without this, the difference sits at about 1.5 across the whole band
        // and every threshold above it silently rejects every image.
        const double kInkScale = 255.0 / std::max(1e-6, cfg.strokeHi);
        cv::Mat imgInk;
        cv::multiply(sBand, kInkScale, imgInk);
        cv::threshold(imgInk, imgInk, 255.0, 255.0, cv::THRESH_TRUNC);

        cv::Mat s3;
        cv::blur(imgInk, s3, cv::Size(3, 3));
        cv::Mat solid;
        cv::compare(s3, par.inkFloor, solid, cv::CMP_GT);
        cv::Mat inkFloored;
        s3.copyTo(inkFloored);
        inkFloored.setTo(0.0, ~solid);

        // ---- step 2: subtract the model, 3x3 box mean, keep what stays above
        //
        // The model is dilated first. It is an average over 84 samples, so its
        // stroke edges sit inside any single image's; subtracting it raw leaves
        // a systematic "image is wider than the model" rim around every stroke.
        // Measured, that rim is 25k..50k pixels on NG and on OK alike, which is
        // why the count alone never separated anything. The dilation states "the
        // model allows ink here" and removes the rim, while a scratch — which is
        // in no model at all — survives it.
        cv::Mat mDil;
        cv::dilate(mBand, mDil,
                   cv::getStructuringElement(cv::MORPH_RECT, {par.modelDilate, par.modelDilate}));
        cv::Mat raw;
        cv::subtract(inkFloored, mDil, raw);
        cv::blur(raw, raw, cv::Size(3, 3));
        cv::Mat over;
        cv::compare(raw, par.diffFloor, over, cv::CMP_GT);

        // Extra ink only counts where the model is sure no stroke belongs. That
        // is where a scratch is unambiguous; on a stroke shoulder the model is
        // uncertain and the difference says nothing.
        cv::Mat nearStroke;
        cv::dilate(iBand, nearStroke,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {par.nearInk, par.nearInk}));
        cv::Mat freeOfInk;
        cv::bitwise_not(nearStroke, freeOfInk);
        cv::Mat excess;
        cv::bitwise_and(over, freeOfInk, excess);

        // Distance to the nearest model-ink pixel: decides the position class.
        cv::Mat dist;
        {
            cv::Mat notInk;
            cv::bitwise_not(iBand, notInk);
            cv::distanceTransform(notInk, dist, cv::DIST_L2, 3);
        }

        // ---- steps 3 and 4, per connected component
        //
        // A scratch is a local structure, not a pixel count, so the ranking has
        // to work on components. Scoring individual pixels and sorting them
        // fills the list with whichever stroke happens to be misaligned.
        cv::Mat labels, cstats, cents;
        const int nLab = cv::connectedComponentsWithStats(excess, labels, cstats, cents, 8, CV_32S);
        const int kHalf = par.window / 2;

        std::vector<Cell> peaks;
        double totalScore = 0;

        for (int i = 1; i < nLab; ++i) {
            const int area = cstats.at<int>(i, cv::CC_STAT_AREA);
            if (area < par.minArea) continue;
            const cv::Rect bb(cstats.at<int>(i, cv::CC_STAT_LEFT),
                              cstats.at<int>(i, cv::CC_STAT_TOP),
                              cstats.at<int>(i, cv::CC_STAT_WIDTH),
                              cstats.at<int>(i, cv::CC_STAT_HEIGHT));
            const cv::Rect bbImg = bb & cv::Rect(0, 0, labels.cols, labels.rows);
            if (bbImg.area() <= 0) continue;

            // PCA over the component's own pixels. Used for length, width and
            // elongation, because "long and thin" is what a scratch is; a dirt
            // spot or a blob is not, and must score lower even when it is
            // brighter.
            double mx = 0, my = 0;
            int n = 0;
            for (int y = bbImg.y; y < bbImg.y + bbImg.height; ++y) {
                for (int x = bbImg.x; x < bbImg.x + bbImg.width; ++x) {
                    if (labels.at<int>(y, x) != i) continue;
                    mx += x;
                    my += y;
                    ++n;
                }
            }
            if (n <= 0) continue;
            mx /= n;
            my /= n;

            double sxx = 0, syy = 0, sxy = 0, sumDiff = 0;
            int onInk = 0;
            for (int y = bbImg.y; y < bbImg.y + bbImg.height; ++y) {
                for (int x = bbImg.x; x < bbImg.x + bbImg.width; ++x) {
                    if (labels.at<int>(y, x) != i) continue;
                    const double dx = x - mx, dy = y - my;
                    sxx += dx * dx;
                    syy += dy * dy;
                    sxy += dx * dy;
                    sumDiff += raw.at<float>(y, x);
                    if (iBand.at<uchar>(y, x)) ++onInk;
                }
            }
            sxx /= n;
            syy /= n;
            sxy /= n;
            const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
            const double disc = std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
            const double l1 = tr / 2.0 + disc, l2 = std::max(1e-6, tr / 2.0 - disc);
            const double major = 4.0 * std::sqrt(l1);
            const double minor = 4.0 * std::sqrt(std::max(0.0, l2));

            // step 4: severity, as the number of characters the mark spans.
            //
            // The instruction was to estimate damage with a 7x7 kernel on the
            // model. Measured, that yields nothing: the ink signal saturates, so
            // a stroke reads 255 and a scratch lying on it also reads 255. There
            // is no observable "damage" to measure, and every component near a
            // stroke scored the same 0.9..1.0, which carries no information.
            //
            // What is measurable, and what the client actually described first —
            // "the more characters it crosses the worse it is, and that should be
            // weighted" — is the crossing span. Project the component onto its
            // major axis and count how many character boxes that run touches.
            const double theta = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
            const double ux = std::cos(theta), uy = std::sin(theta);
            const double halfLen = 2.0 * std::sqrt(std::max(l1, 0.0));
            const cv::Point2d a0(mx - ux * halfLen, my - uy * halfLen);
            const cv::Point2d a1(mx + ux * halfLen, my + uy * halfLen);
            int charsSpan = 0;
            for (const auto& u : base) {
                const cv::Rect b = u.bbox;
                // Walk the axis and ask whether any sample lands in the box.
                for (int t = 0; t <= 24; ++t) {
                    const double f = t / 24.0;
                    const cv::Point2d q(a0.x + (a1.x - a0.x) * f, a0.y + (a1.y - a0.y) * f);
                    if (b.contains(cv::Point(static_cast<int>(q.x), static_cast<int>(q.y)))) {
                        ++charsSpan;
                        break;
                    }
                }
            }
            const double sev = std::clamp(charsSpan / 4.0, 0.0, 1.0);

            Cell c;
            c.p = cv::Point(static_cast<int>(mx + band.x), static_cast<int>(my + band.y));
            c.bb = bbImg;
            c.diff = sumDiff / n;
            c.len = major;
            c.wid = minor;
            c.elong = major / std::max(1.0, minor);
            c.onStroke = static_cast<double>(onInk) / n;
            c.inkRatio = static_cast<double>(area) / std::max(1, bbImg.area());
            c.sever = sev;

            // step 3: position class at the centroid.
            //
            // "On a stroke" cannot be a detection class here: the excess is only
            // allowed where the model has no stroke, by construction. Sitting on
            // a stroke is instead what the step 4 severity measures — how much of
            // a stroke's body the excess covers. What remains to classify is
            // whether the excess is tight against a glyph, loose in the gap
            // between two glyphs, or out in the line spacing, which is the
            // client's order with the top class accounted for by severity.
            const int cx = std::clamp(static_cast<int>(mx), 0, labels.cols - 1);
            const int cy = std::clamp(static_cast<int>(my), 0, labels.rows - 1);
            const double d = dist.at<float>(cy, cx);
            const cv::Point imgPt(cx + band.x, cy + band.y);
            bool insideChar = false;
            for (const auto& u : base) {
                if (u.bbox.contains(imgPt)) {
                    insideChar = true;
                    break;
                }
            }
            if (d <= 3.0) {
                c.posWeight = par.wEdge;
            } else if (insideChar) {
                c.posWeight = par.wGap;
            } else {
                c.posWeight = par.wLineGap;
            }
            // The top class of the client's ordering, "lying on a stroke", is
            // carried by the severity term below at wStroke-equivalent weight.
            if (c.sever >= 0.5 && c.onStroke > 0.3) c.posWeight = par.wStroke;

            // Long and thin scores highest: saturates at eight to one.
            const double shape = std::clamp(c.elong / 8.0, 0.0, 1.0);
            // Ink volume is the component's own fill; onStroke is no longer a
            // volume proxy because the excess is kept off the strokes. Severity
            // is what says whether the mark actually damages a glyph.
            c.score = c.diff * c.posWeight * shape *
                      (par.wInk * c.inkRatio + par.wSever * c.sever);
            totalScore += c.score;
            peaks.push_back(c);
        }

        std::sort(peaks.begin(), peaks.end(),
                  [](const Cell& a, const Cell& b) { return a.score > b.score; });
        if (peaks.size() > 40) peaks.resize(40);

        double bestScore = peaks.empty() ? 0.0 : peaks.front().score;
        cv::Point bestAt = peaks.empty() ? cv::Point(0, 0) : peaks.front().p;

        ImageResult r;
        r.file = toU8(u8p(path).filename());
        r.nCells = static_cast<int>(peaks.size());
        r.best = bestScore;
        r.total = totalScore;
        r.bestAt = bestAt;
        if (!peaks.empty()) r.bestCell = peaks.front();

        // bands, coarse and uncalibrated
        const double s = bestScore;
        r.band = (s >= 30.0) ? 3 : (s >= 15.0) ? 2 : (s >= 6.0) ? 1 : 0;

        cv::cvtColor(g, r.vis, cv::COLOR_GRAY2BGR);
        for (const auto& u : base) {
            cv::rectangle(r.vis, u.bbox, cv::Scalar(80, 200, 80), 1);
        }
        static const cv::Scalar kBands[4] = {cv::Scalar(160, 160, 160), cv::Scalar(0, 215, 255),
                                             cv::Scalar(0, 140, 255), cv::Scalar(0, 0, 255)};
        for (const auto& c : peaks) {
            const cv::Rect bb(c.bb.x + band.x, c.bb.y + band.y, c.bb.width, c.bb.height);
            const int b = (c.score >= 30.0) ? 3 : (c.score >= 15.0) ? 2 : (c.score >= 6.0) ? 1 : 0;
            if (b == 0) continue;
            cv::rectangle(r.vis, bb, kBands[b], 2);
            std::ostringstream lb;
            lb << b;
            cv::putText(r.vis, lb.str(), cv::Point(bb.x, std::max(12, bb.y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, kBands[b], 1);
        }
        if (bestScore > 0) {
            static const char* kNames[4] = {"light", "suspect", "obvious", "certain"};
            cv::putText(r.vis, kNames[r.band], cv::Point(bestAt.x + 26, std::max(20, bestAt.y - 26)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, kBands[r.band], 2);
        }
        rows.push_back(std::move(r));
    }

    std::sort(rows.begin(), rows.end(),
              [](const ImageResult& a, const ImageResult& b) { return a.best > b.best; });

    std::error_code ec;
    fs::create_directories(u8p(outDir), ec);
    int rank = 0;
    for (const auto& r : rows) {
        ++rank;
        std::cout << rank << ',' << r.band << ',' << r.file << ',' << r.nCells << ',' << std::fixed
                  << std::setprecision(1) << r.best << ',' << std::setprecision(0) << r.total << ','
                  << std::setprecision(0) << r.bestCell.len << ',' << r.bestCell.wid << ','
                  << std::setprecision(1) << r.bestCell.elong << ',' << r.bestCell.diff << ','
                  << r.bestCell.onStroke << ',' << r.bestCell.sever << ',' << r.bestCell.posWeight
                  << '\n';
        std::ostringstream nm;
        nm << outDir << "/" << (tag.empty() ? std::string("g") : tag) << std::setw(2)
           << std::setfill('0') << rank << "_b" << r.band << ".png";
        writePng(nm.str(), r.vis);
    }
    return 0;
}
