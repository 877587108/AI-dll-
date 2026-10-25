// Per-image evaluation tool: mirrors the GUI batch-detection criteria path.
//
// Purpose:
//   1) Answer "which real NG parts are being missed" (the GUI table is not
//      convenient for image-by-image comparison)
//   2) Quickly inspect score distributions when calibrating thresholds
//
// Usage:
//   markloc_chardet_eval <modelDir> <image...>
//   markloc_chardet_eval <modelDir> --dir <folder> [--view 00] [--all-views]
//
// Output columns: file,verdict,blockExtra,miss,extra,maxDiff,secondDiff,ms

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

namespace fs = std::filesystem;

namespace {

// Convert a UTF-8 std::string into fs::path. cv::imread fails on non-ASCII
// paths, so images are decoded from a byte buffer instead.
fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

// Convert an fs::path into a UTF-8 std::string. On a Chinese Windows the
// native narrow encoding is GBK, so path::string() would corrupt the bytes.
std::string toU8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

// Read a grayscale image.
bool readGray(const std::string& path, cv::Mat& out) {
    std::ifstream f(u8p(path), std::ios::binary);
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

// Draw a labelled rectangle, clipped to the image.
void drawTag(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, const std::string& label,
             int thickness) {
    const cv::Rect rr = r & cv::Rect(0, 0, img.cols, img.rows);
    if (rr.width <= 0 || rr.height <= 0) return;
    cv::rectangle(img, rr, color, thickness);
    if (label.empty()) return;
    const int base = std::max(rr.y - 4, 12);
    cv::putText(img, label, cv::Point(rr.x + 2, base), cv::FONT_HERSHEY_SIMPLEX, 0.45, color,
                thickness);
}

// Write a PNG through an encoded buffer so non-ASCII output paths work.
bool writePng(const std::string& path, const cv::Mat& img) {
    std::vector<uchar> buf;
    if (!cv::imencode(".png", img, buf)) return false;
    std::ofstream f(u8p(path), std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
    // Unbuffered stdout so a crash still shows how far execution got.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::printf("usage: markloc_chardet_eval <modelDir> <image...>\n");
        std::printf("       markloc_chardet_eval <modelDir> --dir <folder> [--view 00]\n");
        return 2;
    }

    const std::string modelDir = argv[1];
    std::vector<std::string> files;
    std::string viewFilter = "00";
    std::string renderDir;
    // scratch criterion parameters, mirroring the GUI defaults
    int bridge = 25;
    int scratchGrow = 60;
    double scratchMinLen = 60.0;
    double scratchMinCov = 0.25;
    double scratchMinEnergy = 300000.0;
    double scratchMaxTilt = 20.0;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) {
            const fs::path dir = u8p(argv[++i]);
            std::error_code ec;
            for (const auto& e : fs::directory_iterator(dir, ec)) {
                if (!e.is_regular_file()) continue;
                const std::string n = e.path().filename().string();
                if (n.size() < 4) continue;
                if (n.substr(n.size() - 4) != ".tif") continue;
                if (!viewFilter.empty()) {
                    const std::string suf = "_" + viewFilter + ".tif";
                    if (n.size() < suf.size() ||
                        n.compare(n.size() - suf.size(), suf.size(), suf) != 0) {
                        continue;
                    }
                }
                files.push_back(toU8(e.path()));
            }
        } else if (a == "--view" && i + 1 < argc) {
            viewFilter = argv[++i];
        } else if (a == "--render" && i + 1 < argc) {
            renderDir = argv[++i];
        } else if (a == "--all-views") {
            viewFilter.clear();
        } else if (a == "--bridge" && i + 1 < argc) {
            bridge = std::atoi(argv[++i]);
        } else if (a == "--grow" && i + 1 < argc) {
            scratchGrow = std::atoi(argv[++i]);
        } else if (a == "--scratch-min-len" && i + 1 < argc) {
            scratchMinLen = std::atof(argv[++i]);
        } else if (a == "--scratch-min-cov" && i + 1 < argc) {
            scratchMinCov = std::atof(argv[++i]);
        } else if (a == "--scratch-min-energy" && i + 1 < argc) {
            scratchMinEnergy = std::atof(argv[++i]);
        } else if (a == "--tilt" && i + 1 < argc) {
            scratchMaxTilt = std::atof(argv[++i]);
        } else {
            files.push_back(a);
        }
    }
    if (files.empty()) {
        std::printf("no matching images\n");
        return 1;
    }
    std::sort(files.begin(), files.end());

    chardet::Model model;
    if (!chardet::loadModel(modelDir, model) || !model.valid()) {
        std::printf("model load failed: %s\n", modelDir.c_str());
        return 1;
    }

    chardet::Config cfg;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);
    const bool hasTemplate = !model.charTemplates.empty();
    // NOTE: never hand printf a to_string(...).c_str() directly; the temporary
    // is destroyed before the call.
    const std::string tmplCount =
        hasTemplate ? std::to_string(model.charTemplates[0].size()) : std::string("none");
    std::printf("model=%s chars=%d templates=%s\n", modelDir.c_str(),
                static_cast<int>(base.size()), tmplCount.c_str());
    std::printf("file,verdict,block,miss,extra,maxDiff,secondDiff,scratch,nScrLen,nScrEnergy,ms\n");

    int nNg = 0;
    for (const auto& path : files) {
        const int64 t0 = cv::getTickCount();

        cv::Mat g;
        if (!readGray(path, g) || g.size() != model.freq.size()) {
            std::printf("%s,READ_FAIL,0,0,0,0,0,0\n",
                        toU8(u8p(path).filename()).c_str());
            continue;
        }

        // ---- stroke signal + whole-image registration (same as the GUI)
        cv::Mat sig = chardet::strokeSignal(g, cv::Rect(0, 0, g.cols, g.rows), cfg, false);
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

        // ---- per-character fine alignment
        std::vector<chardet::CharUnit> units = base;
        for (auto& u : units) {
            const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
            if (box.width <= 4 || box.height <= 4) continue;
            constexpr int kSearch = 12;
            cv::Rect search(box.x - kSearch, box.y - kSearch, box.width + 2 * kSearch,
                            box.height + 2 * kSearch);
            search &= cv::Rect(0, 0, sig.cols, sig.rows);
            if (search.width < box.width || search.height < box.height) continue;
            cv::Mat mS, tS;
            model.freq(search).convertTo(mS, CV_32F, 1.0 / 255.0);
            sig(search).convertTo(tS, CV_32F);
            cv::Mat res;
            cv::matchTemplate(tS,
                              mS(cv::Rect(box.x - search.x, box.y - search.y, box.width,
                                          box.height)),
                              res, cv::TM_CCOEFF_NORMED);
            cv::Point mLoc;
            double mVal = 0;
            cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
            if (mVal < 0.30) continue;
            const cv::Rect moved(box.x + mLoc.x - kSearch, box.y + mLoc.y - kSearch, box.width,
                                 box.height);
            if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
            u.bbox = moved;
        }

        // ---- criterion 1: whole-row ink blocks
        int blockExtra = 0;
        int blockMissing = 0;
        {
            const auto bd = chardet::diffBlocks(sig, model.freq, units, cfg, 8);
            if (bd.valid) {
                blockExtra = bd.extraCount;
                blockMissing = bd.missingCount;
            }
        }

        // ---- criterion 2: per-character missing / extra ink
        int miss = 0;
        int extra = 0;
        for (const auto& u : units) {
            const auto nc = chardet::diffNcc(sig, model.freq, u, cfg);
            if (!nc.valid) continue;
            if (nc.missScore > 0.0) ++miss;
            if (nc.extraScore > 0.0) ++extra;
        }

        // ---- criterion 3: scratch, with cross-character re-joining
        //
        // Mirrors the GUI block. The bridge close is the key step: subtracting
        // the model strokes cuts a scratch that crosses several characters into
        // fragments, and each fragment is too short to pass the length gate.
        // Closing along the scratch reconnects them before Hough runs.
        int scratchLines = 0;
        double scratchBestLen = 0.0;
        double scratchBestEnergy = 0.0;
        {
            cv::Rect band;
            for (const auto& u : units) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
                if (b.width <= 0 || b.height <= 0) continue;
                band = band.area() > 0 ? (band | b) : b;
            }
            if (band.area() > 0) {
                const int g = scratchGrow;
                band = cv::Rect(band.x - g, band.y - g, band.width + 2 * g,
                                band.height + 2 * g) &
                       cv::Rect(0, 0, sig.cols, sig.rows);
                const cv::Mat sBand = sig(band);

                cv::Mat weak;
                cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
                cv::Mat modelInk;
                cv::compare(model.freq(band), static_cast<int>(cfg.blankFreq * 255), modelInk,
                            cv::CMP_GT);
                cv::Mat modelDil, notModel;
                cv::dilate(modelInk, modelDil,
                           cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
                cv::bitwise_not(modelDil, notModel);
                cv::bitwise_and(weak, notModel, weak);

                if (bridge > 0) {
                    const int bk = std::max(3, bridge | 1);
                    const cv::Mat kb =
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, {bk, bk});
                    cv::morphologyEx(weak, weak, cv::MORPH_CLOSE, kb);
                }
                cv::medianBlur(weak, weak, 3);

                std::vector<cv::Vec4i> lines;
                cv::HoughLinesP(weak, lines, 1, CV_PI / 180.0, 20, 30.0, 60.0);
                for (const auto& ln : lines) {
                    const double len =
                        cv::norm(cv::Point(ln[0], ln[1]) - cv::Point(ln[2], ln[3]));
                    if (len < scratchMinLen) continue;
                    // Keep near-horizontal segments only. Moulding texture is
                    // isotropic, so it shows up at every angle; a real defect
                    // cuts across a text line and is therefore horizontal.
                    const double ang =
                        std::abs(std::atan2(static_cast<double>(ln[3] - ln[1]),
                                            static_cast<double>(ln[2] - ln[0])));
                    const double dev = std::min(ang, CV_PI - ang) * 180.0 / CV_PI;
                    if (dev > scratchMaxTilt) continue;
                    // energy along the segment
                    const int steps = std::max(2, static_cast<int>(len));
                    double energy = 0;
                    int covered = 0;
                    for (int t = 0; t <= steps; ++t) {
                        const double rr = static_cast<double>(t) / steps;
                        const int x = static_cast<int>(ln[0] + (ln[2] - ln[0]) * rr);
                        const int y = static_cast<int>(ln[1] + (ln[3] - ln[1]) * rr);
                        if (y < 0 || y >= weak.rows || x < 0 || x >= weak.cols) continue;
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
                    }
                    const double cov = static_cast<double>(covered) / (steps + 1);
                    if (cov < scratchMinCov) continue;
                    const double score = energy * len;
                    if (score < scratchMinEnergy) continue;
                    ++scratchLines;
                    scratchBestLen = std::max(scratchBestLen, len);
                    scratchBestEnergy = std::max(scratchBestEnergy, score);
                }
            }
        }

        // ---- multi-template dual metric (recorded only, never decides)
        double maxDiff = 0.0;
        double secondDiff = 0.0;
        if (hasTemplate) {
            std::vector<chardet::CharUnit> allU = units;
            const auto mr = chardet::diffMultiTemplate(sig, model, allU);
            if (mr.valid) {
                maxDiff = mr.maxDiff;
                secondDiff = mr.secondDiff;
            }
        }

        const bool ng = (blockExtra + blockMissing + miss + extra + scratchLines) > 0;
        if (ng) ++nNg;

        const double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
        std::printf("%s,%s,%d,%d,%d,%.4f,%.4f,%d,%.0f,%.0f,%.0f\n",
                    toU8(u8p(path).filename()).c_str(), ng ? "NG" : "OK",
                    blockExtra + blockMissing, miss, extra, maxDiff, secondDiff, scratchLines,
                    scratchBestLen, scratchBestEnergy, ms);

        // ---- optional annotation render: shows every char box so the operator
        // can tell whether a defect sits inside or outside the character boxes
        if (!renderDir.empty()) {
            cv::Mat vis;
            cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);
            for (const auto& u : units) {
                drawTag(vis, u.bbox, cv::Scalar(80, 200, 80), std::string(), 1);
            }
            for (const auto& u : units) {
                const auto nc = chardet::diffNcc(sig, model.freq, u, cfg);
                if (!nc.valid) continue;
                if (nc.missScore > 0.0) {
                    drawTag(vis, u.bbox, cv::Scalar(0, 0, 255), "MISS", 2);
                }
                if (nc.extraScore > 0.0) {
                    drawTag(vis, u.bbox, cv::Scalar(255, 0, 0), "EXTRA", 2);
                }
            }
            const std::string stamp = ng ? "NG" : "OK";
            cv::putText(vis, stamp, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 2.0,
                        ng ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 180, 0), 4);
            std::error_code ec;
            fs::create_directories(u8p(renderDir), ec);
            const fs::path out = u8p(renderDir) / (u8p(path).stem().u8string() + u8".png");
            writePng(toU8(out), vis);
        }
    }

    std::printf("# total %d, verdict NG %d\n", static_cast<int>(files.size()), nNg);
    return 0;
}
