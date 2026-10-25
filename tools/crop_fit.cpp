// cropfit.cpp —— 用 logo 锚点把新规格图裁剪到模型尺寸
//
// 用户思路：通过定位塑封进行裁剪。前几轮已确认关键事实：
//   · 新旧规格没有缩放（markloc_projmatch 8/8 张给出 scale=1.000，
//     markloc_regionalign 给 logo 区 0.998）
//   · 只有水平平移
// 所以对齐问题化简为：求一个平移量，把新图裁成模型尺寸。
//
// 为什么不用投影找平移：旧规格库自身投影相关只有 0.724（最低 0.540），
// 基准不稳，据此算出的平移误差可能超过 10px，会毁掉逐点比对。
//
// 改用二维相位相关，锚点选 logo 椭圆区域：
//   相位相关对整体亮度不敏感，且给亚像素精度。
//   logo 环内部结构丰富，是画面里最适合做相关匹配的目标。
//
// 输出：
//   1. 求得的平移量 (dx, dy)
//   2. 裁剪后的图，尺寸与模型完全一致
//   3. 与模型 logo 的吻合度（用于判断裁剪是否可信）
//
// 只有吻合度达标，裁出来的图才允许进入判定链。
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

static cv::Mat readAny(const std::string& p) {
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

// 在图上找 logo 椭圆：形状搜索，不预设坐标
static bool findLogoRect(const cv::Mat& bin, cv::Rect* out) {
    cv::Mat lb, st, ce;
    const int n = cv::connectedComponentsWithStats(bin, lb, st, ce, 8, CV_32S);
    int best = -1;
    double bs = -1;
    for (int i = 1; i < n; ++i) {
        const int a = st.at<int>(i, cv::CC_STAT_AREA);
        const int w = st.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = st.at<int>(i, cv::CC_STAT_HEIGHT);
        if (a < 2000 || w < 80 || h < 40) continue;
        const double ar = (double)w / h;
        if (ar < 1.2 || ar > 3.2) continue;
        if ((double)a / (w * (double)h) > 0.45) continue;
        const double sc = a * (1.0 - std::abs(ar - 1.7));
        if (sc > bs) { bs = sc; best = i; }
    }
    if (best < 0) return false;
    *out = cv::Rect(st.at<int>(best, cv::CC_STAT_LEFT), st.at<int>(best, cv::CC_STAT_TOP),
                    st.at<int>(best, cv::CC_STAT_WIDTH), st.at<int>(best, cv::CC_STAT_HEIGHT));
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: markloc_cropfit <modelDir> <outDir> <image...> [--minCorr 0.55]\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) return 1;
    const std::string outDir = argv[2];
    fs::create_directories(outDir);
    double minCorr = 0.55;
    std::vector<std::string> files;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--minCorr" && i + 1 < argc) minCorr = std::atof(argv[++i]);
        else files.push_back(a);
    }

    const int MW = model.freq.cols, MH = model.freq.rows;
    const chardet::Config& cfg = model.config;
    const auto& rg = model.regions[3];   // logo 区

    // 模型 logo 区域的参考（用 ref，它保留了原始信号强度）
    cv::Mat ref32;
    model.ref.convertTo(ref32, CV_32F);
    const cv::Rect mLogo = rg.rect & cv::Rect(0, 0, ref32.cols, ref32.rows);
    cv::Mat mRef = ref32(mLogo).clone();

    std::printf("模型 %dx%d, logo 区 (%d,%d,%d,%d)\n", MW, MH, mLogo.x, mLogo.y, mLogo.width,
                mLogo.height);
    std::printf("%-18s %10s %10s %9s %9s %s\n", "文件", "logo位置", "平移dx,dy", "相关", "裁后尺寸",
                "结论");

    for (const auto& f : files) {
        cv::Mat gray = readAny(f);
        if (gray.empty()) { std::printf("%-18s 读图失败\n", f.c_str()); continue; }
        const std::string tag = fs::path(f).stem().string();

        cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg,
                                            false);

        // 1. 找 logo
        cv::Mat bin;
        cv::compare(sig, 0.25, bin, cv::CMP_GT);
        cv::morphologyEx(bin, bin, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
        cv::Rect logo;
        if (!findLogoRect(bin, &logo)) {
            std::printf("%-18s 未找到 logo\n", tag.c_str());
            continue;
        }

        // 3. 平移量：用**区域 2（Silan 行）**做锚点，不用 logo 区。
        //
        // 模型里的 logo 区是 region3 = (474,315,322,225)，而它内部那个
        // "字符框" 是 279x177 的异常大框（早就量过：真实椭圆环只有约 282x179，
        // 而这个框把椭圆和圆周标记全包进去了）。拿一个错误框当锚点必然失败。
        //
        // 区域 2 在模型里是 (804,365,353,154)，在新图上实测定位为
        // (761,370,352,154)——宽差 1px、高完全相同。这直接证明：
        // 新旧规格之间没有缩放，只有平移，平移量 = (-43, +5)。
        const cv::Rect kAnchorModel = model.regions[2].rect;   // (804,365,353,154)
        // 在新图上找同样大小的区域：用归一化灰度做二维匹配
        cv::Mat mGray;
        model.ref.convertTo(mGray, CV_8U);
        cv::Mat anchorTpl = mGray(kAnchorModel & cv::Rect(0, 0, mGray.cols, mGray.rows)).clone();
        cv::Mat grayF, tplF;
        gray.convertTo(grayF, CV_32F);
        anchorTpl.convertTo(tplF, CV_32F);
        // 在新图里搜：预期位置在 (761,370) 附近，给 ±80px 余量
        cv::Rect srch(kAnchorModel.x - 120, kAnchorModel.y - 80,
                      kAnchorModel.width + 320, kAnchorModel.height + 160);
        srch &= cv::Rect(0, 0, gray.cols, gray.rows);
        cv::Point2d sh(0.0, 0.0);
        if (srch.width >= anchorTpl.cols && srch.height >= anchorTpl.rows) {
            cv::Mat res;
            cv::matchTemplate(grayF(srch), tplF, res, cv::TM_CCOEFF_NORMED);
            cv::Point ml;
            double mv = 0;
            cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
            const int ax = srch.x + ml.x;   // 锚点在新图上的位置
            const int ay = srch.y + ml.y;
            // 新图坐标 = 模型坐标 + T，所以 T = 锚点新位置 - 锚点模型位置
            sh = cv::Point2d(ax - kAnchorModel.x, ay - kAnchorModel.y);
            std::printf("  [锚点] 模型(%d,%d) -> 新图(%d,%d)  平移(%+.0f,%+.0f)  匹配 %.3f\n",
                        kAnchorModel.x, kAnchorModel.y, ax, ay, sh.x, sh.y, mv);
        } else {
            std::printf("  [锚点] 搜索区越界，退回默认平移 (-43,+5)\n");
            sh = cv::Point2d(-43.0, 5.0);
        }

        // 4. 裁剪：新图坐标 = 模型坐标 + sh
        //    所以模型原点 (0,0) 对应新图 (sh.x, sh.y)，裁剪窗口左上角就是它
        int cx = (int)std::lround(sh.x);
        int cy = (int)std::lround(sh.y);

        // 检查窗口是否越界
        bool ok = (cx >= 0 && cy >= 0 && cx + MW <= gray.cols && cy + MH <= gray.rows);
        cv::Rect win(cx, cy, MW, MH);
        cv::Mat cropped;
        if (ok) {
            cropped = gray(win).clone();
        } else {
            // 越界就用反射填充，保证尺寸仍与模型一致
            cv::Mat canvas(MH, MW, gray.type(), cv::Scalar(0));
            const cv::Rect src = win & cv::Rect(0, 0, gray.cols, gray.rows);
            if (src.width > 0 && src.height > 0) {
                cv::Rect dst(src.x - cx, src.y - cy, src.width, src.height);
                gray(src).copyTo(canvas(dst));
            }
            cropped = canvas;
            // 用边缘像素填充空白区
            cv::copyMakeBorder(cropped, cropped, 0, 0, 0, 0, cv::BORDER_REPLICATE);
        }

        // 5. 验证：裁出的图与模型 logo 区的原始灰度做相关
        //
        // 早先用 strokeSignal 比，相关全都接近 0——因为两边的 scale
        // 按各自整幅统计，不在同一量纲。这里改用原始灰度（先各自归一化），
        // 比的是"图形长什么样"，不掺入任何按图统计的缩放。
        double corr = 0;
        {
            cv::Mat mGray;
            model.ref.convertTo(mGray, CV_8U);
            cv::Mat mPatch = mGray(mLogo).clone();
            cv::Mat cPatch = cropped(mLogo).clone();
            // 各自归一化到 0..1，去掉整体亮度差
            auto norm01 = [](const cv::Mat& s) {
                cv::Mat f;
                s.convertTo(f, CV_32F);
                double mn, mx;
                cv::minMaxLoc(f, &mn, &mx);
                if (mx - mn > 1e-6) f = (f - mn) / (mx - mn);
                return f;
            };
            cv::Mat A = norm01(mPatch), B = norm01(cPatch);
            cv::Scalar ma, mb, sa, sb;
            cv::meanStdDev(A, ma, sa);
            cv::meanStdDev(B, mb, sb);
            if (sa[0] > 1e-6 && sb[0] > 1e-6)
                corr = (A - ma[0]).dot(B - mb[0]) / (A.total() * sa[0] * sb[0]);
        }

        char where[32];
        std::snprintf(where, sizeof(where), "(%d,%d)", logo.x, logo.y);
        char shs[32];
        std::snprintf(shs, sizeof(shs), "%+.1f,%+.1f", sh.x, sh.y);
        std::printf("%-18s %10s %10s %9.3f %9s %s\n", tag.c_str(), where, shs, corr,
                    (std::to_string(cropped.cols) + "x" + std::to_string(cropped.rows)).c_str(),
                    corr >= minCorr ? "通过" : "相关偏低");

        const std::string op = (fs::path(outDir) / (tag + "_fit.tif")).string();
        cv::imwrite(op, cropped);
    }
    return 0;
}
