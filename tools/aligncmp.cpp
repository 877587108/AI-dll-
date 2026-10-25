// 三种对齐方式在同批真实数据上的直接对比
//
// 只关心一件事：哪个方法能让字符框真正压在字形上。
// 用"框墨覆盖率"衡量——框内墨量 / 整个字符区墨量，越高说明框越准。
//
// 对比：
//   A 不对齐      —— 底线，看原始偏差
//   B 相位相关    —— 产品现状
//   C 投影对齐    —— 待评估
// 必须三个一起看。只看 C 的绝对值毫无意义，因为参照帧和实拍本就不同源。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "chardet.hpp"

static bool readGray(const std::string& path, cv::Mat& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::vector<unsigned char> buf;
    unsigned char tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof(tmp), fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(fp);
    if (buf.empty()) return false;
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

static std::vector<double> profile(const cv::Mat& s, bool vertical) {
    std::vector<double> p(vertical ? s.cols : s.rows, 0.0);
    for (int y = 0; y < s.rows; ++y) {
        const float* row = s.ptr<float>(y);
        for (int x = 0; x < s.cols; ++x) p[vertical ? x : y] += row[x];
    }
    return p;
}

static int bestShift(const std::vector<double>& a, const std::vector<double>& b, int lo, int hi) {
    const int n = static_cast<int>(a.size());
    int best = 0;
    double bestV = -1e18;
    for (int s = lo; s <= hi; ++s) {
        double num = 0, d1 = 0, d2 = 0;
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            const int j = i + s;
            if (j < 0 || j >= n) continue;
            num += a[i] * b[j];
            d1 += a[i] * a[i];
            d2 += b[j] * b[j];
            ++cnt;
        }
        if (cnt < n / 2 || d1 < 1e-9 || d2 < 1e-9) continue;
        const double v = num / std::sqrt(d1 * d2);
        if (v > bestV) {
            bestV = v;
            best = s;
        }
    }
    return best;
}

// 框墨覆盖率：所有字符框内的墨量占整个字符区墨量的比例
static double coverage(const cv::Mat& sig, const std::vector<chardet::CharUnit>& units) {
    cv::Rect ink;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.width > 0 && b.height > 0) ink = ink.area() > 0 ? (ink | b) : b;
    }
    if (ink.area() <= 0) return 0;
    double in = 0;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.area() > 0) in += cv::sum(sig(b))[0];
    }
    const double all = cv::sum(sig(ink))[0];
    return all > 1e-9 ? in / all * 100.0 : 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: aligncmp <modelDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    cv::Mat ref32;
    model.ref.convertTo(ref32, CV_32F);
    const std::vector<double> refV = profile(ref32, true);
    const std::vector<double> refH = profile(ref32, false);
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);

    // 参照帧的字符区（用于相位相关取窗）
    cv::Rect inkArea;
    for (const auto& u : base) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, ref32.cols, ref32.rows);
        if (b.width > 0 && b.height > 0) inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(ref32.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(ref32.rows - inkArea.y, inkArea.height + 40);

    std::printf("%-34s %7s %7s %7s   %s\n", "图片", "不对齐", "相位相关", "投影对齐", "胜者");
    double sumA = 0, sumB = 0, sumC = 0;
    int winB = 0, winC = 0, winA = 0, n = 0;
    for (int a = 2; a < argc; ++a) {
        cv::Mat gray;
        if (!readGray(argv[a], gray)) continue;
        const cv::Rect full(0, 0, gray.cols, gray.rows);
        const cv::Mat sig = chardet::strokeSignal(gray, full, model.config, false);
        if (sig.size() != ref32.size()) continue;

        // A 不对齐
        const double covA = coverage(sig, base);

        // B 相位相关（与产品同款：字符区取窗）
        cv::Mat sigB = sig.clone();
        {
            cv::Mat r32 = ref32, s32;
            sig.convertTo(s32, CV_32F);
            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            s32(inkArea).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::warpAffine(sig, sigB, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
            }
        }
        const double covB = coverage(sigB, base);

        // C 投影对齐
        const int dy = bestShift(refH, profile(sig, false), -80, 80);
        const int dx = bestShift(refV, profile(sig, true), -80, 80);
        cv::Mat sigC;
        {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -dx, 0, 1, -dy);
            cv::warpAffine(sig, sigC, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
        }
        const double covC = coverage(sigC, base);

        const char* base_name = std::strrchr(argv[a], '\\');
        base_name = base_name ? base_name + 1 : argv[a];
        const char* winner = (covC > covB + 0.5) ? "投影" : (covB > covC + 0.5) ? "相位" : "持平";
        std::printf("%-34.34s %6.1f%% %6.1f%% %6.1f%%   %s\n", base_name, covA, covB, covC,
                    winner);
        if (covC > covB + 0.5) ++winC;
        else if (covB > covC + 0.5) ++winB;
        else ++winA;
        sumA += covA;
        sumB += covB;
        sumC += covC;
        ++n;
    }
    if (n > 0) {
        std::printf("\n平均覆盖率: 不对齐 %.1f%% | 相位相关 %.1f%% | 投影对齐 %.1f%%\n", sumA / n,
                    sumB / n, sumC / n);
        std::printf("胜场: 相位 %d | 投影 %d | 持平 %d (共 %d 张)\n", winB, winC, winA, n);
    }
    return 0;
}
