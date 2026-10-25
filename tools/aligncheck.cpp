// aligncheck.cpp —— 对齐体检：这张图到底有没有对齐
//
// 背景：前面所有判据都建立在"图已对齐"之上，但从没验过。
// 0109_RT 报出的全局平移是 dx=-17.73 dy=-7.93，一个字符框才 70px 宽，
// 这个量相当于四分之一个字。如果它是错的，后面所有数都是错的。
//
// 本工具做三件独立的事，互相交叉验证：
//   1. 全局相位相关（与 batchcore 一致）：给出 (dx,dy) 与响应峰值
//   2. 逐字符 matchTemplate：给出每个字符自己的最佳位移与峰值
//      —— 若全局平移正确，逐字位移应全部接近 0
//   3. 未对齐 vs 已对齐 两种情况下，模型墨迹与图像信号的吻合度
//      —— 对齐应该让吻合度变高，不变或变低就说明对齐是错的
//
// 判读：逐字位移中位 |dx|,|dy| 都在 3px 内、峰值 >=0.6，才算"已对齐"。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

namespace fs = std::filesystem;

static cv::Mat readTif(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::vector<uchar> buf;
    uchar tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(f);
    if (buf.empty()) return {};
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

// 逐字符搜位移，返回 (dx,dy,峰值) 列表
struct CharShift {
    int idx;
    int dx, dy;
    double peak;
    cv::Rect box;
};

static std::vector<CharShift> perCharShift(const cv::Mat& sig, const chardet::Model& model,
                                           const std::vector<chardet::CharUnit>& units,
                                           int search = 20) {
    std::vector<CharShift> out;
    for (const auto& u : units) {
        const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (box.width <= 4 || box.height <= 4) continue;
        cv::Rect sr(box.x - search, box.y - search, box.width + 2 * search,
                    box.height + 2 * search);
        sr &= cv::Rect(0, 0, sig.cols, sig.rows);
        if (sr.width < box.width || sr.height < box.height) continue;
        cv::Mat mS, tS;
        model.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
        sig(sr).convertTo(tS, CV_32F);
        cv::Mat res;
        cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)),
                          res, cv::TM_CCOEFF_NORMED);
        cv::Point mLoc;
        double mVal = 0;
        cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
        out.push_back({u.index, mLoc.x - search, mLoc.y - search, mVal, box});
    }
    return out;
}

// 吻合度：模型墨迹处图像应当亮，两者归一化后的相关系数
//
// 早先这版返回 +0.0000 完全不变，是因为 sig/freq 尺寸一致时
// roi 已在上游裁过，这里再裁一次反而把 dot 的维度算错。
// 现在显式用 reshape 成一行再算点积，维度不可能错。
static double agreement(const cv::Mat& sig, const cv::Mat& freq, const cv::Rect& roi) {
    cv::Mat a, b;
    sig(roi).convertTo(a, CV_32F);
    freq(roi).convertTo(b, CV_32F, 1.0 / 255.0);
    a = a.reshape(1, 1);
    b = b.reshape(1, 1);
    if (a.cols != b.cols || a.cols == 0) return 0.0;
    cv::Mat am = a - cv::mean(a)[0];
    cv::Mat bm = b - cv::mean(b)[0];
    const double denom = cv::norm(am) * cv::norm(bm);
    if (denom < 1e-9) return 0.0;
    return am.dot(bm) / denom;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("用法: markloc_aligncheck <modelDir> <image...>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型加载失败: %s\n", argv[1]);
        return 1;
    }
    const chardet::Config& cfg = model.config;
    const std::vector<chardet::CharUnit> baseUnits = chardet::toAllUnits(model);

    for (int ai = 2; ai < argc; ++ai) {
        const std::string p8 = argv[ai];
        cv::Mat gray = readTif(p8);
        if (gray.empty() || gray.size() != model.freq.size()) {
            std::printf("读图/尺寸不符: %s\n", p8.c_str());
            continue;
        }
        std::printf("\n===== %s =====\n", fs::path(p8).filename().string().c_str());

        cv::Mat sigRaw =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);

        // ---- 1. 全局相位相关（与 batchcore.hpp:121 完全同路）
        cv::Mat r32, s32;
        model.ref.convertTo(r32, CV_32F);
        sigRaw.convertTo(s32, CV_32F);
        cv::Rect inkArea;
        for (const auto& u : baseUnits) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, sigRaw.cols, sigRaw.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
        }
        inkArea.x = std::max(0, inkArea.x - 20);
        inkArea.y = std::max(0, inkArea.y - 20);
        inkArea.width = std::min(sigRaw.cols - inkArea.x, inkArea.width + 40);
        inkArea.height = std::min(sigRaw.rows - inkArea.y, inkArea.height + 40);
        cv::Mat rC, sC;
        r32(inkArea).copyTo(rC);
        s32(inkArea).copyTo(sC);
        const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
        std::printf("[1] 全局相位相关: dx=%+.2f dy=%+.2f  (用于 warpAffine 的量)\n", sh.x, sh.y);

        // ---- 2a. 未对齐时逐字符位移（这本身就应接近 0，如果图已对齐）
        auto before = perCharShift(sigRaw, model, baseUnits, 20);
        // ---- 2b. 施加全局平移后再看逐字符残差
        //
        // 注意：这里必须先 clone。
        // cv::Mat sigAl = sigRaw 是浅拷贝，与 src 共享数据；
        // src 和 dst 是同一块内存时 warpAffine 行为未定义，实测一个像素都不改
        // （[3a] 自检会打出来）。生产代码 batchcore.hpp 用的是独立变量 aligned，
        // 所以那边没有这个问题；错的是本工具。
        cv::Mat sigAl = sigRaw.clone();
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::warpAffine(sigRaw, sigAl, M, sigRaw.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
        }
        auto after = perCharShift(sigAl, model, baseUnits, 20);

        auto report = [](const char* tag, const std::vector<CharShift>& v) {
            if (v.empty()) {
                std::printf("[%s] 无有效字符\n", tag);
                return;
            }
            std::vector<int> ax, ay;
            std::vector<double> pk;
            for (const auto& c : v) {
                ax.push_back(std::abs(c.dx));
                ay.push_back(std::abs(c.dy));
                pk.push_back(c.peak);
            }
            std::sort(ax.begin(), ax.end());
            std::sort(ay.begin(), ay.end());
            std::sort(pk.begin(), pk.end());
            int over3 = 0, lowPk = 0;
            for (size_t i = 0; i < v.size(); ++i) {
                if (ax[i] > 3 || ay[i] > 3) ++over3;
                if (pk[i] < 0.6) ++lowPk;
            }
            std::printf("[%s] 字符 %zu 个 | |dx|中位 %d 最大 %d ||dy|中位 %d 最大 %d | 峰值中位 %.3f 最低 %.3f\n",
                        tag, v.size(), ax[ax.size() / 2], ax.back(), ay[ay.size() / 2], ay.back(),
                        pk[pk.size() / 2], pk.front());
            std::printf("      残差>3px 的字符: %d/%zu   峰值<0.6 的字符: %d/%zu\n", over3, v.size(),
                        lowPk, v.size());
        };
        report("2a 未施加全局平移", before);
        report("2b 已施加全局平移", after);

        // ---- 3. 吻合度：对齐应该让它变高
        //
        // 先做一个自检：warpAffine 到底有没有改变像素？
        // 早先这里恒输出 +0.0000，两个差 8px 的输入给出同一个数，
        // 这要么说明 warp 没生效，要么说明 agreement 本身是坏的。
        // 不先证明这一条，下面的判读一律不可信。
        {
            cv::Mat diff;
            cv::absdiff(sigRaw, sigAl, diff);
            double mx = 0;
            cv::minMaxLoc(diff, nullptr, &mx);
            cv::Scalar ms;
            cv::meanStdDev(diff, ms, cv::noArray());
            std::printf("[3a 自检] |未对齐-已对齐| 最大值 %.4f 均值 %.4f  %s\n", mx, ms[0],
                        (mx > 1e-6 ? "warp 生效了" : "**warp 没有改变任何像素，后续不可信**"));
        }
        const double gBefore = agreement(sigRaw, model.freq, inkArea);
        const double gAfter = agreement(sigAl, model.freq, inkArea);
        std::printf("[3] 模型与图像吻合度: 未对齐 %.4f -> 已对齐 %.4f  (%+.4f)\n", gBefore, gAfter,
                    gAfter - gBefore);
        std::printf("    判读: %s\n",
                    (gAfter > gBefore + 1e-4
                         ? "对齐提高了吻合度，方向对"
                         : "对齐没有提高吻合度，这个平移量可能不该施加"));
    }
    return 0;
}
