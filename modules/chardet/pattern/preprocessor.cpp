// preprocessor.cpp —— 标准预处理算子：椭圆拉直 + 极坐标展开
//
// ================================================================
// 为什么这段代码必须只有一份
// ================================================================
//
// 这段几何变换在十几轮实验里被重复实现了十几次，
// 每次都因为细节不同导致结论不可比。已知踩过的坑：
//
//   坑一：缩放的写法
//     要"沿短轴方向拉伸 a/b"，所以缩放矩阵作用在**旋转后的 y 轴**上。
//     第一版写成除以 a/b，自检发现短轴端点变换后距中心 40px（应 137px）。
//     靠"长短轴端点必须落在半径 a 上"这个自检抓出来的。
//
//   坑二：仿射方向
//     cv::warpAffine 默认把 dst 坐标映射到 src 取像素，所以要挪内容
//     得在矩阵里写**负**的平移。写成正的话图会往反方向偏一个整环宽，
//     表面上看环还在框里，实际沿周长采样一路擦边。
//
//   坑三：warpPolar 的 dsize 顺序
//     dsize.width 是**半径数**，height 是**角度数**，不是反的。
//     弄反了不报错，只是出来一张转置的图，后面的半径带全错位。
//
// 所以这里做两个自检（见 SelfCheck），变换参数不对直接判失败，
// 而不是带着错的几何往下跑——那种情况下的"断口"结论毫无意义。
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

#include "pattern_api.hpp"

namespace chardet {
namespace pattern {

namespace {

/// 拉直的自检阈值
///
/// 拉直后长轴端点、短轴端点距环心的距离都应该等于 a。
/// 允许 5% 偏差（像素取整 + 椭圆框本身不是完美椭圆）。
constexpr double kSelfCheckTolerance = 0.05;

}  // namespace

/// 标准预处理：仿射拉直 → 可选灰度门限 → 极坐标展开
class StandardPreprocessor final : public IPreprocessor {
public:
    const char* name() const override { return "standard"; }

    bool run(const Input& in, Geometry& geom, cv::Mat& straightened, PolarMap& polar,
             std::string* err) override {
        straightened.release();
        polar = PolarMap{};
        geom = Geometry{};
        if (!in.valid()) {
            setErr(err, "输入无效：信号为空、类型不是 CV_32F，或区域不是椭圆/圆");
            return false;
        }

        const EllipseGeom& ell = in.region.ell;
        const double a = std::max(1.0, static_cast<double>(std::max(ell.a, ell.b)));
        const double b = std::max(1.0, static_cast<double>(std::min(ell.a, ell.b)));
        const double cx = ell.center.x;
        const double cy = ell.center.y;

        // 长短轴归一化：保证 a 是长半轴（角度写在长轴上）
        double rotDeg = ell.angleDeg;
        double aa = ell.a, bb = ell.b;
        if (bb > aa) {
            // 调用方把长短轴写反了：交换并转 90°
            std::swap(aa, bb);
            rotDeg += 90.0;
        }
        const double kScale = aa / std::max(1.0, bb);
        const double rot = rotDeg * CV_PI / 180.0;
        const double cR = std::cos(-rot), sR = std::sin(-rot);

        // M3 = T2 · Sc · Rm · T1
        //   ① 平移到环心原点
        //   ② 反向旋转 -rot，把长轴转到 +x
        //   ③ 沿 y 压缩 a/b，椭圆变正圆
        //   ④ 平移回环心
        cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, -cx, 0, 1, -cy, 0, 0, 1);
        cv::Mat Rm = (cv::Mat_<double>(3, 3) << cR, -sR, 0, sR, cR, 0, 0, 0, 1);
        cv::Mat Sc = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, kScale, 0, 0, 0, 1);
        cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
        const cv::Mat M3 = T2 * Sc * Rm * T1;

        // 自检：长短轴端点变换后都该落在离环心 a 的位置
        if (!selfCheck(aa, bb, rot, cx, cy, M3, err)) return false;

        cv::Mat M;
        M3(cv::Rect(0, 0, 3, 2)).copyTo(M);
        cv::warpAffine(in.signal, straightened, M, in.signal.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, 0.0);

        // ---- 展开之前的灰度门限
        //
        // 环的亮度在一段范围内连续变化（细处弱、粗处强）。
        // 展开后再按 inkThr 二值，等于让"环变淡"这段信息被抹平——
        // 淡到一定程度仍然算有墨，缺口恒为 0，缺陷看不见。
        // 所以在展开之前就把低于门限的像素切掉，让"淡"在展开图上是真空洞。
        if (in.params.preGrayThr > 0.0) {
            cv::threshold(straightened, straightened, in.params.preGrayThr, 0.0,
                          cv::THRESH_TOZERO);
        }

        geom.ell = ell;
        geom.scaleY = kScale;
        geom.affine = M3;
        geom.center = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
        geom.radiusA = static_cast<int>(std::lround(a));
        geom.ok = true;

        // ---- 极坐标展开
        //
        // 只展开以环心为中心的一小块，够覆盖半径带外沿即可，
        // 展开整张图没有意义还慢。
        const int pad = static_cast<int>(a * (in.params.bandOuterRatio + 0.05)) + 8;
        const int half = static_cast<int>(a) + pad;
        cv::Rect roi(static_cast<int>(cx) - half, static_cast<int>(cy) - half, half * 2,
                     half * 2);
        roi &= cv::Rect(0, 0, straightened.cols, straightened.rows);
        if (roi.area() <= 0) {
            setErr(err, "环心周围的裁剪区为空（椭圆超出图像范围）");
            return false;
        }
        const cv::Point2f cc(static_cast<float>(cx - roi.x), static_cast<float>(cy - roi.y));
        const double maxR = static_cast<double>(half);
        const int nR = std::max(1, static_cast<int>(std::lround(maxR)));
        const int nA = std::max(8, in.params.angles);

        cv::Mat pol;
        // 注意 dsize 顺序：width = 半径数，height = 角度数
        cv::warpPolar(straightened(roi), pol, cv::Size(nR, nA), cc, maxR,
                      cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR);
        // 转成"行=半径、列=角度"
        cv::Mat polT;
        cv::transpose(pol, polT);
        polar.img = polT;
        polar.angles = nA;
        polar.radii = nR;
        polar.maxR = maxR;
        return true;
    }

private:
    static void setErr(std::string* err, const char* msg) {
        if (err) *err = msg;
    }

    /// 拉直自检：长短轴端点变换后到环心的距离都应为 a
    static bool selfCheck(double a, double b, double rot, double cx, double cy,
                          const cv::Mat& M3, std::string* err) {
        const double ca = std::cos(rot), sa = std::sin(rot);
        // 长轴端点（参数角 0 与 π），短轴端点（π/2 与 3π/2）
        const double pts[4][2] = {{a, 0}, {-a, 0}, {0, b}, {0, -b}};
        double worst = 0.0;
        for (const auto& p : pts) {
            const double ux = p[0] * ca - p[1] * sa;
            const double uy = p[0] * sa + p[1] * ca;
            const double sx = M3.at<double>(0, 0) * (cx + ux) + M3.at<double>(0, 1) * (cy + uy) +
                              M3.at<double>(0, 2);
            const double sy = M3.at<double>(1, 0) * (cx + ux) + M3.at<double>(1, 1) * (cy + uy) +
                              M3.at<double>(1, 2);
            const double d = std::hypot(sx - cx, sy - cy);
            worst = std::max(worst, std::abs(d - a) / a);
        }
        if (worst > kSelfCheckTolerance) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                          "拉直自检失败：端点距环心偏离 %.1f%%（上限 %.0f%%），"
                          "多半是长短轴或角度写错了",
                          worst * 100.0, kSelfCheckTolerance * 100.0);
            setErr(err, buf);
            return false;
        }
        return true;
    }
};

std::unique_ptr<IPreprocessor> makePreprocessor() {
    return std::make_unique<StandardPreprocessor>();
}

}  // namespace pattern
}  // namespace chardet
