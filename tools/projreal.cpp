// 投影对齐在真实数据上的可靠性
//
// 合成平移测出的是"同一张图平移自己"，剖面形状不变，峰值天然干净。
// 真实来料每颗料的字迹磨损、墨量、位置都不同，投影剖面形状本身就变了，
// 这才是决定投影法能不能用的场景。
//
// 这里对每一张真实图：
//   1. 算投影给出的位移估计
//   2. 报峰值与峰余量（峰余量小 = 定位不可信）
//   3. 用估计位移对齐后，统计字符框对字形的覆盖率
// 最终要回答：峰余量能不能作为"这次投影可信"的判据。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// 返回 b 相对 a 的位移；同时给出峰值和"离最佳峰 4px 以外的次峰"，
// 两者之差就是定位可信度。差值接近 0 说明投影分不清对错。
struct ShiftEst {
    int shift = 0;
    double peak = 0;
    double margin = 0;
};

static ShiftEst bestShift(const std::vector<double>& a, const std::vector<double>& b, int lo,
                          int hi) {
    const int n = static_cast<int>(a.size());
    ShiftEst e;
    double bestV = -1e18, secondV = -1e18;
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
            if (std::abs(s - e.shift) > 4) secondV = std::max(secondV, bestV);
            bestV = v;
            e.shift = s;
        } else if (v > secondV && std::abs(s - e.shift) > 4) {
            secondV = v;
        }
    }
    e.peak = bestV;
    e.margin = (secondV < -1e17) ? 999.0 : bestV - secondV;
    return e;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: projreal <modelDir> <image...>\n");
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
    const int R = 80;

    std::printf("%-42s %16s %20s %10s\n", "图片", "水平投影(dy/峰/余量)", "垂直投影(dx/峰/余量)",
                "框墨覆盖率");
    int n = 0, lowMargin = 0;
    for (int a = 2; a < argc; ++a) {
        cv::Mat gray;
        if (!readGray(argv[a], gray)) continue;
        const cv::Rect full(0, 0, gray.cols, gray.rows);
        const cv::Mat sig = chardet::strokeSignal(gray, full, model.config, false);
        if (sig.size() != ref32.size()) continue;

        const ShiftEst eh = bestShift(refH, profile(sig, false), -R, R);
        const ShiftEst ev = bestShift(refV, profile(sig, true), -R, R);

        // 把信号按估计位移拉回参照系，再算字符框里装了多少墨、框外漏了多少墨
        const cv::Mat M =
            (cv::Mat_<double>(2, 3) << 1, 0, -ev.shift, 0, 1, -eh.shift);
        cv::Mat al;
        cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);

        const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);
        const cv::Rect ink = [&] {
            cv::Rect r;
            for (const auto& u : base) {
                const cv::Rect b = u.bbox & cv::Rect(0, 0, al.cols, al.rows);
                if (b.width > 0 && b.height > 0) r = r.area() > 0 ? (r | b) : b;
            }
            return r;
        }();
        double inSum = 0;
        for (const auto& u : base) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, al.cols, al.rows);
            if (b.area() > 0) inSum += cv::sum(al(b))[0];
        }
        const double allInk = cv::sum(al(ink))[0];
        const double cover = allInk > 1e-9 ? inSum / allInk : 0.0;

        const char* base_name = std::strrchr(argv[a], '\\');
        base_name = base_name ? base_name + 1 : argv[a];
        std::printf("%-42.42s %5d/%5.3f/%6.3f %5d/%5.3f/%6.3f %9.1f%%\n", base_name, eh.shift,
                    eh.peak, eh.margin, ev.shift, ev.peak, ev.margin, cover * 100.0);
        ++n;
        if (eh.margin < 0.02 || ev.margin < 0.02) ++lowMargin;
    }
    std::printf("\n共 %d 张，峰余量<0.02 的 %d 张\n", n, lowMargin);
    return 0;
}
