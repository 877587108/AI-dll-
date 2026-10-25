// 判别区域图 —— 把 OK 与 NG 的严重度分布并排摊开，让"门槛该放哪"变成
// 一个看得见的决定，而不是我拍一个数。
//
// 输入是 markloc_suspect 落盘的 CSV 与每行的渲染图（渲染图上已经画好了
// 那一条划痕和位置，所以这里只需要按坐标裁一小块放大，不必重跑检测，
// 数字与图必然一致）。
//
// 输出两样：
//   1) 判别图：横轴 = 严重度 S = 深度 × 跨越字数。NG 画在轴上方，OK 画在
//      轴下方，重叠区着色标出。几条候选门槛画成竖线，并写明每条线的代价。
//   2) 重叠区放大图：把落在重叠区里的每一张原图裁出来放大，标上 S 和分项，
//      供人眼直接判"这条到底算不算"。
//
// usage: markloc_bandplot <ng.csv> <ngDir> <ok.csv> <okDir> <outDir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace {

// MSVC 在 /std:c++20 下已删除 fs::u8path，统一走 u8string 构造。
fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

// 中文路径读不出来，统一走字节流 + imdecode。
bool readBgr(const std::string& path, cv::Mat& out) {
    std::ifstream f(u8p(path), std::ios::binary);
    if (!f) return false;
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) return false;
    out = cv::imdecode(cv::Mat(1, static_cast<int>(buf.size()), CV_8UC1, buf.data()),
                       cv::IMREAD_COLOR);
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

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

struct Row {
    int rank = 0;
    int band = 0;
    bool reported = false;
    std::string file;
    int charsHit = 0;
    double depth = 0;
    double composite = 0;
    int bx = 0;
    int by = 0;
    bool ng = false;   // 来自 ng 目录 = 真缺陷
};

std::vector<Row> loadCsv(const std::string& path, bool ng) {
    std::vector<Row> rows;
    std::ifstream f(u8p(path));
    if (!f) {
        std::cout << "cannot open " << path << "\n";
        return rows;
    }
    // 上游可能是 PowerShell 重定向写出来的，带 UTF-8 BOM，先剥掉；
    // 万一整个文件是 UTF-16（重定向默认编码），字段里会夹 NUL，直接判死，
    // 别让 stoi 抛异常把进程打崩。
    std::string line;
    std::getline(f, line);
    if (line.size() >= 2 && (static_cast<unsigned char>(line[0]) == 0xFF ||
                             static_cast<unsigned char>(line[1]) == 0x00)) {
        std::cout << path << " looks like UTF-16, re-export it as UTF-8\n";
        return rows;
    }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB) {
            line.erase(0, 3);
        }
        const auto c = splitCsv(line);
        if (c.size() < 15) continue;
        Row r;
        try {
            r.rank = std::stoi(c[0]);
            r.band = std::stoi(c[1]);
            r.reported = std::stoi(c[2]) != 0;
            r.file = c[3];
            r.charsHit = std::stoi(c[8]);
            r.depth = std::stod(c[9]);
            r.composite = std::stod(c[12]);
            r.bx = std::stoi(c[13]);
            r.by = std::stoi(c[14]);
        } catch (const std::exception&) {
            std::cout << "skip unparsable row: " << line.substr(0, 60) << "\n";
            continue;
        }
        r.ng = ng;
        rows.push_back(std::move(r));
    }
    return rows;
}

// 文件名的 ASCII 部分，够认出是哪一张就行（OpenCV 画不了中文）。
std::string asciiOf(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c < 0x80) out.push_back(static_cast<char>(c));
    }
    std::string squeezed;
    for (char c : out) {
        if (c == ' ' && !squeezed.empty() && squeezed.back() == ' ') continue;
        squeezed.push_back(c);
    }
    while (!squeezed.empty() && squeezed.back() == ' ') squeezed.pop_back();
    if (squeezed.size() > 44) squeezed = squeezed.substr(0, 42) + "..";
    return squeezed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cout << "usage: markloc_bandplot <ng.csv> <ngDir> <ok.csv> <okDir> <outDir>\n";
        return 2;
    }
    const std::string ngCsv = argv[1];
    const std::string ngDir = argv[2];
    const std::string okCsv = argv[3];
    const std::string okDir = argv[4];
    const std::string outDir = argv[5];

    std::vector<Row> rows = loadCsv(ngCsv, true);
    for (auto& r : loadCsv(okCsv, false)) rows.push_back(std::move(r));
    if (rows.empty()) {
        std::cout << "no rows\n";
        return 1;
    }

    const double kMaxS = 2500.0;

    // ---------------- 判别图 ----------------
    const int W = 1500;
    const int H = 800;
    const int left = 230;
    const int right = 40;
    const int midY = 440;

    cv::Mat plot(H, W, CV_8UC3, cv::Scalar(28, 28, 32));
    auto toX = [&](double s) {
        const double t = std::clamp(s / kMaxS, 0.0, 1.0);
        return left + static_cast<int>(std::lround(t * (W - left - right)));
    };

    // 重叠区：真 NG 最低 1219 与 OK 最高 1430 之间
    const double kOverlapLo = 1182.0;
    const double kOverlapHi = 1430.0;
    cv::rectangle(plot, cv::Rect(toX(kOverlapLo), 0, toX(kOverlapHi) - toX(kOverlapLo), H),
                  cv::Scalar(52, 46, 96), cv::FILLED);

    struct Gate {
        double s;
        const char* label;
        const char* cost;
        cv::Scalar color;
    };
    // 每条门槛线的代价是实测数，不是估的：用 84 张 OK + 7 张 NG 的 composite
    // 逐个比出来的。FP 不含 0101（客户已确认它确实是缺陷，只是被放在 ok/ 里），
    // 所以"报出 0101"不算误报。
    const Gate gates[] = {
        {800.0, "800", "NG 6/7  FP 5", cv::Scalar(130, 210, 130)},
        {1000.0, "1000", "NG 6/7  FP 2", cv::Scalar(80, 230, 80)},
        {1219.0, "1219", "NG 6/7  FP 1", cv::Scalar(0, 220, 255)},
        {1500.0, "1500", "NG 4/7  FP 0", cv::Scalar(0, 110, 255)},
    };
    int gi = 0;
    for (const auto& g : gates) {
        const int x = toX(g.s);
        for (int y = 120; y < H - 60; y += 14) {
            cv::line(plot, cv::Point(x, y), cv::Point(x, std::min(y + 8, H)), g.color, 1);
        }
        // 1430 与 1500 挨得太近，标签错行放，否则叠成一团看不清。
        const int lane = (gi % 2) * 34;
        ++gi;
        cv::putText(plot, g.label, cv::Point(x + 3, 26 + lane), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    g.color, 1, cv::LINE_AA);
        cv::putText(plot, g.cost, cv::Point(x + 3, 46 + lane), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                    g.color, 1, cv::LINE_AA);
    }

    cv::line(plot, cv::Point(left, midY), cv::Point(W - right, midY), cv::Scalar(90, 90, 90), 1);
    for (int s = 0; s <= static_cast<int>(kMaxS); s += 250) {
        const int x = toX(s);
        cv::line(plot, cv::Point(x, midY - 8), cv::Point(x, midY + 8), cv::Scalar(90, 90, 90), 1);
        cv::putText(plot, std::to_string(s), cv::Point(x - 14, midY + 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(150, 150, 150), 1);
    }
    cv::putText(plot, "S = depth x charsHit", cv::Point(left + 4, midY + 56),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    cv::putText(plot, "NG (true defects)", cv::Point(left, 36), cv::FONT_HERSHEY_SIMPLEX, 0.62,
                cv::Scalar(95, 95, 255), 2, cv::LINE_AA);
    cv::putText(plot, "one dot per shot", cv::Point(left, 58), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(150, 150, 200), 1, cv::LINE_AA);
    // OK 的图例放到最底下，避开轴下方那串 81 个点。
    cv::putText(plot, "OK (released)", cv::Point(left, H - 92), cv::FONT_HERSHEY_SIMPLEX, 0.62,
                cv::Scalar(255, 170, 90), 2, cv::LINE_AA);
    cv::putText(plot, "only the 3 that clear 1000 are lifted above the axis",
                cv::Point(left, H - 68), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(160, 140, 120), 1, cv::LINE_AA);

    std::vector<const Row*> ngs;
    std::vector<const Row*> oks;
    for (const auto& r : rows) (r.ng ? ngs : oks).push_back(&r);
    auto byS = [](const Row* a, const Row* b) { return a->composite > b->composite; };
    std::sort(ngs.begin(), ngs.end(), byS);
    std::sort(oks.begin(), oks.end(), byS);

    const double kShowOkAbove = 1000.0;
    const int nAbove = static_cast<int>(ngs.size());

    int slotNg = 0;
    for (const Row* rp : ngs) {
        const Row& r = *rp;
        const int x = toX(r.composite);
        const int y = midY - (20 + slotNg * 30);
        ++slotNg;
        cv::circle(plot, cv::Point(x, y), 6, cv::Scalar(95, 95, 255), cv::FILLED, cv::LINE_AA);
        // 每条都标，包括 0110 那个 73 —— 客户已确认它可漏，图上也要看得见，
        // 否则它就是一个来路不明的孤立红点。
        char lbl[160];
        std::snprintf(lbl, sizeof(lbl), "S=%d  d%d x %dch  %s", static_cast<int>(r.composite),
                      static_cast<int>(r.depth), r.charsHit, asciiOf(r.file).c_str());
        // 用 getTextSize 量实际宽度，别用字数乘常数估 —— 估小了文字会压到点上。
        // 左边放不下（点太靠左）就翻到右边写。
        const cv::Size ts = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.44, 1, nullptr);
        const int tx = (x - 16 - ts.width >= left) ? x - 16 - ts.width : x + 16;
        cv::putText(plot, lbl, cv::Point(tx, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                    cv::Scalar(95, 95, 255), 1, cv::LINE_AA);
    }
    // 0110 单独点一句，省得有人问"这个点为什么没进结果"
    {
        const int y = midY - (20 + slotNg * 30);
        cv::putText(plot, "0110 (S=73): below every gate - customer confirmed it may be missed",
                    cv::Point(left, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                    cv::Scalar(150, 120, 150), 1, cv::LINE_AA);
    }

    // OK 里只有三张过 1000，抬到轴上方，与真 NG 同框才看得出谁高谁低。
    int slotOk = 0;
    for (const Row* rp : oks) {
        const Row& r = *rp;
        const int x = toX(r.composite);
        if (r.composite >= kShowOkAbove) {
            const int y = midY - (20 + (nAbove + 2 + slotOk) * 30);
            ++slotOk;
            cv::circle(plot, cv::Point(x, y), 6, cv::Scalar(255, 170, 90), cv::FILLED, cv::LINE_AA);
            char lbl[160];
            std::snprintf(lbl, sizeof(lbl), "S=%d  d%d x %dch  %s", static_cast<int>(r.composite),
                          static_cast<int>(r.depth), r.charsHit, asciiOf(r.file).c_str());
            const cv::Size ts = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.44, 1, nullptr);
            const int tx = (x - 16 - ts.width >= left) ? x - 16 - ts.width : x + 16;
            cv::putText(plot, lbl, cv::Point(tx, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                        cv::Scalar(255, 170, 90), 1, cv::LINE_AA);
        } else {
            const int y = midY + 20 + (slotOk++ * 8);
            if (y > H - 70) continue;
            cv::circle(plot, cv::Point(x, y), 3, cv::Scalar(170, 130, 95), cv::FILLED,
                       cv::LINE_AA);
        }
    }

    {
        const int x = toX(kOverlapLo) + 6;
        cv::putText(plot, "OVERLAP 1182..1430", cv::Point(x, H - 44), cv::FONT_HERSHEY_SIMPLEX,
                    0.58, cv::Scalar(130, 130, 245), 2, cv::LINE_AA);
        cv::putText(plot, "no single threshold separates NG from FP in here",
                    cv::Point(x, H - 20), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                    cv::Scalar(130, 130, 245), 1, cv::LINE_AA);
    }
    cv::putText(plot, "84 OK + 7 NG, v00", cv::Point(left, H - 34), cv::FONT_HERSHEY_SIMPLEX,
                0.40, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
    cv::putText(plot, "FP excludes 0101 = confirmed defect", cv::Point(left, H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);

    if (!writePng(outDir + "/bandplot.png", plot)) {
        std::cout << "write bandplot failed\n";
        return 1;
    }
    std::cout << "wrote " << outDir << "/bandplot.png\n";

    // ---------------- 重叠区放大图 ----------------
    //
    // 只裁严重度落在 [900, 1600] 的行：这一带是"判据说不好、只能人眼判"的
    // 区间。渲染图上已画好划痕线，放大直接看那条痕够不够深。
    {
        constexpr int kTile = 420;
        constexpr int kCols = 4;
        constexpr double kZoom = 2.4;
        std::vector<const Row*> shown;
        for (const auto& r : rows) {
            if (r.composite >= 900.0 && r.composite <= 1600.0) shown.push_back(&r);
        }
        std::sort(shown.begin(), shown.end(), [](const Row* a, const Row* b) {
            return a->composite > b->composite;
        });
        const int n = static_cast<int>(shown.size());
        const int nRows = (n + kCols - 1) / kCols;
        if (nRows > 0) {
            cv::Mat sheet(nRows * kTile, kCols * kTile, CV_8UC3, cv::Scalar(24, 24, 24));
            for (int i = 0; i < n; ++i) {
                const Row& r = *shown[i];
                char name[512];
                std::snprintf(name, sizeof(name), "%s/%s%02d_b%d.png",
                              (r.ng ? ngDir : okDir).c_str(), r.ng ? "ng" : "ok", r.rank, r.band);
                cv::Mat src;
                if (!readBgr(name, src)) {
                    std::cout << "missing " << name << "\n";
                    continue;
                }
                const int half = static_cast<int>(kTile / kZoom / 2);
                const cv::Rect want(r.bx - half, r.by - half, 2 * half, 2 * half);
                const cv::Rect valid = want & cv::Rect(0, 0, src.cols, src.rows);
                cv::Mat tile(kTile, kTile, CV_8UC3, cv::Scalar(0, 0, 0));
                if (valid.area() > 0) {
                    cv::Mat zoom;
                    cv::resize(src(valid), zoom, cv::Size(), kZoom, kZoom, cv::INTER_CUBIC);
                    const int ox = static_cast<int>((valid.x - want.x) * kZoom);
                    const int oy = static_cast<int>((valid.y - want.y) * kZoom);
                    const int bw = std::min(zoom.cols, kTile - ox);
                    const int bh = std::min(zoom.rows, kTile - oy);
                    if (bw > 0 && bh > 0) {
                        zoom(cv::Rect(0, 0, bw, bh)).copyTo(tile(cv::Rect(ox, oy, bw, bh)));
                    }
                }
                const int cx = (i % kCols) * kTile;
                const int cy = (i / kCols) * kTile;
                cv::rectangle(tile, cv::Rect(0, 0, kTile, 52), cv::Scalar(0, 0, 0), cv::FILLED);
                char head[192];
                std::snprintf(head, sizeof(head), "%s   S=%d", r.ng ? "NG" : "OK",
                              static_cast<int>(r.composite));
                cv::putText(tile, head, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                            r.ng ? cv::Scalar(95, 95, 255) : cv::Scalar(255, 170, 90), 1,
                            cv::LINE_AA);
                char sub[192];
                std::snprintf(sub, sizeof(sub), "depth %d  x  chars %d",
                              static_cast<int>(r.depth), r.charsHit);
                cv::putText(tile, sub, cv::Point(6, 44), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                            cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
                const std::string nm = asciiOf(r.file);
                cv::putText(tile, nm, cv::Point(6, kTile - 10), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                            cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
                cv::rectangle(tile, cv::Rect(1, 1, kTile - 2, kTile - 2),
                              r.ng ? cv::Scalar(95, 95, 220) : cv::Scalar(200, 140, 70), 1);
                tile.copyTo(sheet(cv::Rect(cx, cy, kTile, kTile)));
            }
            if (!writePng(outDir + "/overlap_zoom.png", sheet)) {
                std::cout << "write zoom failed\n";
            } else {
                std::cout << "wrote " << outDir << "/overlap_zoom.png  (" << n << " tiles)\n";
            }
        }
    }
    return 0;
}
