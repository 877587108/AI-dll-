// Scratch depth profiler.
//
// Measures how "deep" a scratch is, in a way that is comparable across images.
// The current scratch criterion gates on energy*length, which mixes length and
// contrast and is therefore impossible to calibrate against a single reference
// defect. This tool reports the primitives separately so the client's
// "this one is the boundary" call can be turned into numbers.
//
// Reported per image:
//   - the char-box union and the +N px grown band
//   - candidate scratch components: elongated, thin, brighter than background
//   - for each candidate: length, width, peak contrast, mean contrast, area,
//     how many character boxes it intersects
//
// Usage:
//   markloc_diagscratch <modelDir> <image...> [--grow N] [--out dir]

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

// Read a grayscale image.
//
// On MSVC the std::ifstream(fs::path) overload goes through the filesystem
// conversion chain, which is fragile for non-ASCII paths under codepage 936.
// Opening through the native wide path keeps the bytes untouched.
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
    std::ofstream f(u8p(path), std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return static_cast<bool>(f);
}

// Thin elongated components are what a scratch looks like once the model
// strokes have been removed. PCA on the pixel cloud gives the major/minor axes
// without assuming the scratch is a straight line: a curved scratch still has
// one dominant direction overall.
struct Comp {
    double len = 0;     // major axis extent, px
    double width = 0;   // minor axis extent, px
    double area = 0;    // pixel count
    double peak = 0;    // max signal inside the component
    double mean = 0;    // mean signal inside the component
    double fill = 0;    // area / (len*width), 1.0 = solid rectangle
    cv::Rect box;
    int charsHit = 0;   // how many char boxes the component touches
};

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::printf("usage: markloc_diagscratch <modelDir> <image...> [--grow N] [--out dir]\n");
        return 2;
    }
    const std::string modelDir = argv[1];
    std::vector<std::string> files;
    std::string outDir;
    int grow = 60;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--grow" && i + 1 < argc) {
            grow = std::atoi(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            outDir = argv[++i];
        } else if (a == "--dir" && i + 1 < argc) {
            // Enumerate from disk instead of taking paths on the command line.
            // Under codepage 936 PowerShell hands argv the ACP bytes, not UTF-8,
            // so a Chinese filename would be mis-decoded and the open would fail.
            const fs::path dir = u8p(argv[++i]);
            std::error_code ec;
            std::vector<std::string> found;
            for (const auto& e : fs::directory_iterator(dir, ec)) {
                if (!e.is_regular_file()) continue;
                const std::string n = toU8(e.path().filename());
                if (n.size() < 4) continue;
                if (n.compare(n.size() - 4, 4, ".tif") != 0) continue;
                found.push_back(toU8(e.path()));
            }
            std::sort(found.begin(), found.end());
            files.insert(files.end(), found.begin(), found.end());
        } else {
            files.push_back(a);
        }
    }

    chardet::Model model;
    if (!chardet::loadModel(modelDir, model) || !model.valid()) {
        std::printf("model load failed: %s\n", modelDir.c_str());
        return 1;
    }
    chardet::Config cfg;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);
    std::printf("model=%s chars=%d grow=%d\n", modelDir.c_str(), static_cast<int>(base.size()),
                grow);
    std::printf(
        "file,chars_union,band,peak,mean,len,width,fill,area,chars_hit\n");

    for (const auto& path : files) {
        cv::Mat g;
        if (!readGray(path, g) || g.size() != model.freq.size()) {
            std::cout << toU8(u8p(path).filename()) << ",READ_FAIL,,,,,,,,\n";
            continue;
        }
        const std::string stem = toU8(u8p(path).stem());

        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);

        // ---- whole-image registration (same as GUI)
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

        // ---- char union and grown band
        cv::Rect band;
        for (const auto& u : base) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            band = band.area() > 0 ? (band | b) : b;
        }
        const cv::Rect grown(band.x - grow, band.y - grow, band.width + 2 * grow,
                             band.height + 2 * grow);
        const cv::Rect roi = grown & cv::Rect(0, 0, sig.cols, sig.rows);
        const cv::Mat sRoi = sig(roi);

        // ---- excess ink = above background AND away from model strokes
        //
        // The background is the same statistical frequency map used everywhere
        // else, so "excess" means "ink the model has never seen here".
        cv::Mat excess;
        cv::compare(sRoi, cfg.missStrokeLo * 0.7, excess, cv::CMP_GT);
        cv::Mat modelInk;
        cv::compare(model.freq(roi), static_cast<int>(cfg.blankFreq * 255), modelInk,
                    cv::CMP_GT);
        cv::Mat modelDil, notModel;
        cv::dilate(modelInk, modelDil,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
        cv::bitwise_not(modelDil, notModel);
        // cv::compare already yields 0/255; do NOT scale again.
        cv::bitwise_and(excess, notModel, excess);

        // ---- re-join the pieces a character cut the scratch into
        //
        // This is the step the current pipeline is missing: subtracting the
        // model strokes splits one scratch into several fragments, and an
        // OPEN(3x3) afterwards deletes the thin ends. A directional close
        // rebuilds the gap along the scratch instead of eroding it away.
        cv::Mat joined;
        {
            const cv::Mat kc = cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15});
            cv::morphologyEx(excess, joined, cv::MORPH_CLOSE, kc);
        }
        const cv::Mat ko = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
        cv::morphologyEx(joined, joined, cv::MORPH_OPEN, ko);

        cv::Mat lbl, stats, cent;
        const int n = cv::connectedComponentsWithStats(joined, lbl, stats, cent, 8);

        double bestPeak = 0, bestMean = 0, bestLen = 0, bestWid = 0, bestFill = 0,
               bestArea = 0;
        int bestChars = 0;
        cv::Rect bestBox;

        for (int i = 1; i < n; ++i) {
            const int a = stats.at<int>(i, cv::CC_STAT_AREA);
            if (a < 80) continue;

            // Bounding box and pixel sums in one pass; avoiding a per-pixel
            // point list keeps this linear and avoids feeding minAreaRect a
            // degenerate point cloud.
            const cv::Rect cb(stats.at<int>(i, cv::CC_STAT_LEFT),
                              stats.at<int>(i, cv::CC_STAT_TOP),
                              stats.at<int>(i, cv::CC_STAT_WIDTH),
                              stats.at<int>(i, cv::CC_STAT_HEIGHT));
            if (cb.width <= 0 || cb.height <= 0) continue;

            double peak = 0, sum = 0;
            int cnt = 0;
            double mx = 0, my = 0;
            for (int y = cb.y; y < cb.y + cb.height; ++y) {
                const int* row = lbl.ptr<int>(y);
                for (int x = cb.x; x < cb.x + cb.width; ++x) {
                    if (row[x] != i) continue;
                    const double v = sRoi.at<float>(y, x);
                    peak = std::max(peak, v);
                    sum += v;
                    mx += x;
                    my += y;
                    ++cnt;
                }
            }
            if (cnt <= 0) continue;
            const double mean = sum / cnt;
            mx /= cnt;
            my /= cnt;

            // Second moments around the centroid give the major/minor extent
            // of a possibly curved shape without assuming it is a line.
            double sxx = 0, syy = 0, sxy = 0;
            for (int y = cb.y; y < cb.y + cb.height; ++y) {
                const int* row = lbl.ptr<int>(y);
                for (int x = cb.x; x < cb.x + cb.width; ++x) {
                    if (row[x] != i) continue;
                    const double dx = x - mx, dy = y - my;
                    sxx += dx * dx;
                    syy += dy * dy;
                    sxy += dx * dy;
                }
            }
            sxx /= cnt;
            syy /= cnt;
            sxy /= cnt;
            const double tr = sxx + syy;
            const double det = sxx * syy - sxy * sxy;
            const double disc = std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
            const double l1 = tr / 2.0 + disc;
            const double l2 = tr / 2.0 - disc;
            // 4*sqrt(eigenvalue) is the full extent along each axis
            const double len = 4.0 * std::sqrt(std::max(0.0, l1));
            const double wid = 4.0 * std::sqrt(std::max(0.0, l2));

            if (len < 40) continue;                 // too short to be a scratch
            if (wid > 18) continue;                 // too fat, that is a blob
            if (len / std::max(1.0, wid) < 2.5) continue;  // not elongated

            // how many character boxes the component touches
            const cv::Rect absBox(cb.x + roi.x, cb.y + roi.y, cb.width, cb.height);
            int hit = 0;
            for (const auto& u : base) {
                if ((u.bbox & absBox).area() > 0) ++hit;
            }

            if (peak > bestPeak) {
                bestPeak = peak;
                bestMean = mean;
                bestLen = len;
                bestWid = wid;
                bestFill = a / std::max(1.0, len * wid);
                bestArea = a;
                bestChars = hit;
                bestBox = absBox;
            }
        }

        // NOTE: never pass a non-ASCII path to printf("%s") on this platform.
        // Under codepage 936 a byte of a Chinese character can be read as a
        // format specifier, which corrupts the argument walk and crashes.
        std::cout << stem << ',' << band.area() << ',' << roi.area() << ',' << std::fixed
                  << std::setprecision(3) << bestPeak << ',' << bestMean << ','
                  << std::setprecision(1) << bestLen << ',' << bestWid << ','
                  << std::setprecision(2) << bestFill << ',' << std::setprecision(0)
                  << bestArea << ',' << bestChars << '\n';
        if (!outDir.empty()) {
            std::error_code ec;
            fs::create_directories(u8p(outDir), ec);
            cv::Mat vis;
            cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);
            for (const auto& u : base) {
                cv::rectangle(vis, u.bbox, cv::Scalar(80, 200, 80), 1);
            }
            cv::rectangle(vis, roi, cv::Scalar(200, 200, 0), 2);
            if (bestArea > 0) {
                cv::rectangle(vis, bestBox, cv::Scalar(0, 0, 255), 2);
            }
            writePng(outDir + "/" + stem + "_probe.png", vis);
            writePng(outDir + "/" + stem + "_excess.png", excess);
            writePng(outDir + "/" + stem + "_joined.png", joined);
        }
    }
    return 0;
}
