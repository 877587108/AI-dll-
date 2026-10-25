// 投影对齐在"已知平移"上的验证
//
// 本工具回答一个问题：给定一个已知的平移量，投影法能不能把它量出来？
//
// 做法：取一张模型图 f，人为制造一个已知平移 (tx,ty)，对平移后的图跑投影对齐，
// 看估计值是否等于真值。这是"同一张图平移自己"的理想场景，
// 剖面形状完全不变，所以它只能证明算法实现是对的，
// **不能**证明投影法在真实来料上可用——真实数据的结论见 markloc_projreal。
//
// 用法: markloc_projalign <modelDir> <image> [--maxShift 20]
//
// 输出：每个真值偏移下的估计值与误差，最后给成功率。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

// 一维序列 b 相对 a 的最佳平移（归一去均值互相关）
// 返回估计位移；peak 给峰值，margin 给"最佳与次佳（距离>4）之差"
static int bestShift1D(const std::vector<double>& a, const std::vector<double>& b, int R,
                       double* peak, double* margin) {
    const int n = static_cast<int>(a.size());
    if (n < 4) return 0;
    std::vector<double> score(2 * R + 1, -2.0);
    for (int d = -R; d <= R; ++d) {
        double sa = 0, sb = 0, sa2 = 0, sb2 = 0, sab = 0;
        int cnt = 0;
        for (int x = 0; x < n; ++x) {
            const int xb = x + d;
            if (xb < 0 || xb >= n) continue;
            const double va = a[x], vb = b[xb];
            sa += va;
            sb += vb;
            sa2 += va * va;
            sb2 += vb * vb;
            sab += va * vb;
            ++cnt;
        }
        if (cnt < n / 2) continue;
        const double ma = sa / cnt, mb = sb / cnt;
        const double va = sa2 / cnt - ma * ma;
        const double vb = sb2 / cnt - mb * mb;
        if (va < 1e-9 || vb < 1e-9) continue;
        score[d + R] = (sab / cnt - ma * mb) / std::sqrt(va * vb);
    }
    int best = 0;
    double bv = -3.0;
    for (int i = 0; i < (int)score.size(); ++i)
        if (score[i] > bv) { bv = score[i]; best = i; }
    double second = -3.0;
    for (int i = 0; i < (int)score.size(); ++i)
        if (std::abs(i - best) > 4 && score[i] > second) second = score[i];
    if (peak) *peak = bv;
    if (margin) *margin = (second > -3.0) ? (bv - second) : 1.0;
    return best - R;
}

static std::vector<double> vprofile(const cv::Mat& s) {
    std::vector<double> p(s.cols, 0.0);
    for (int y = 0; y < s.rows; ++y) {
        const float* row = s.ptr<float>(y);
        for (int x = 0; x < s.cols; ++x) p[x] += row[x];
    }
    return p;
}

static std::vector<double> hprofile(const cv::Mat& s) {
    std::vector<double> p(s.rows, 0.0);
    for (int y = 0; y < s.rows; ++y) {
        const float* row = s.ptr<float>(y);
        for (int x = 0; x < s.cols; ++x) p[y] += row[x];
    }
    return p;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_projalign <modelDir> <image> [--maxShift 20]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    int maxShift = 20;
    for (int i = 2; i + 1 < argc; i += 2)
        if (std::strcmp(argv[i], "--maxShift") == 0) maxShift = std::atoi(argv[i + 1]);

    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("读图/尺寸不符: %s\n", argv[2]);
        return 1;
    }
    const cv::Mat sig =
        chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), model.config, false);

    // 整幅图投影（合成平移场景下这是最干净的用法）
    const std::vector<double> pv = vprofile(sig);
    const std::vector<double> ph = hprofile(sig);

    std::printf("已知平移验证（同一张图平移自己，只证明实现正确）\n");
    std::printf("%8s %8s %9s %8s %9s %8s\n", "真值dx", "估计dx", "峰值", "峰余量", "真值dy",
                "估计dy");
    int okX = 0, okY = 0, nX = 0, nY = 0;
    for (int t = -maxShift; t <= maxShift; ++t) {
        // 只验证水平：把图水平平移 t 后，反向求位移应得到 -t
        cv::Mat shifted;
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, t, 0, 1, 0);
        cv::warpAffine(sig, shifted, M, sig.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        const std::vector<double> sv = vprofile(shifted);
        double pk = 0, mg = 0;
        const int est = bestShift1D(pv, sv, maxShift + 4, &pk, &mg);
        std::printf("%8d %8d %9.4f %8.4f\n", t, est, pk, mg);
        ++nX;
        if (std::abs(est - t) <= 1) ++okX;
    }
    std::printf("\n水平 ±%d 内：%d/%d 误差 <=1px\n", maxShift, okX, nX);
    (void)okY;
    (void)nY;
    (void)ph;
    return 0;
}
