#pragma once

// pattern_types.hpp —— 图案区检测的规范化数据结构
//
// ================================================================
// 为什么要有这个文件
// ================================================================
//
// 以前图案检测只有一个函数 analyzePatternBreak()，500 行，
// 9 个算法阶段硬编码串在一起：
//
//   拉直 → 预灰度阈值 → 极坐标展开 → 环内屏蔽 → 带型连通
//        → 环→缺口 → 缺口合并 → 断点判定 → 打分
//
// 每换一个判据就得整段改写这个函数。十几轮实验下来，
// tools/ 下积了二十多个各抄一份预处理代码的探针程序，
// 每个探针的"拉直"、"定位"、"极化"写法都有细微差别，
// 于是实验之间无法互相比较，也无法确定哪个结论可信。
//
// 这个文件把**所有阶段共享的数据**集中定义一次。
// 后续加算法只需要：
//   ① 实现一个 IDetector（见 pattern_api.hpp）
//   ② 在 det_registry.cpp 里注册
// 不需要再看这个文件。
//
// ================================================================
// 数据流（严格单向，每一层只依赖上一层）
// ================================================================
//
//   Input        原图灰度 → strokeSignal → signal，加上区域与参数
//     ↓
//   Geometry     椭圆几何、拉直矩阵、拉直后的环心与半径
//     ↓
//   PolarMap     拉直图 → 极坐标展开。行=半径，列=角度
//     ↓
//   Profile      逐角度的环带墨量曲线（ink / rel）
//     ↓
//   Finding      某个检测器给出的结论
//     ↓
//   Result       所有检测器结论的汇总
//
// 关键约定：
//   · 一个阶段只读它声明依赖的输入，只写自己的产出字段，不修改上游
//   · 角度度量统一用**度**，索引换算只在 PolarMap::degOf/idxOf 里做一次
//   · 所有坐标若无特别说明都是**全图坐标**

#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "../chardet.hpp"
#include "../config.hpp"

namespace chardet {
namespace pattern {

// ---------------------------------------------------------------- 参数

/// 图案检测参数
///
/// 与 PatternBreakParams 的区别：
///   PatternBreakParams 是"老判据自己的参数"（缺口游程法专有）
///   Params 是"流水线级参数" + 各检测器的参数块
///
/// 保留老结构体是为了让已有调用点与模型文件不受影响；
/// 内部通过 fromLegacy() 转换。
struct Params {
    // ---- 流水线级
    int angles = 720;          // 沿周长采样数（极坐标列数）
    double inkThr = 0.35;      // "这个像素有墨"的强度门槛
    double innerBlockRatio = 0.6;  // 半径小于此比例×a 的区域直接清零
                                   // （图案内部常还有别的元素，会盖住缺口）

    // ---- 环带（Profile 用）
    double bandInnerRatio = 0.80;  // 半径带内沿（×a）
    double bandOuterRatio = 1.20;  // 半径带外沿（×a）

    // ---- 展开之前的灰度门限（0 = 关闭）
    double preGrayThr = 0.0;

    // ---- 选用的检测器（按名字，见 det_registry.cpp）
    std::vector<std::string> detectors{"notch_window", "gap_run"};

    // ---- 汇总门槛
    double minSeverity = 0.5;  // severity 达到此值才判 NG

    /// 从老的 PatternBreakParams 转换（保持既有调用点不变）
    static Params fromLegacy(const PatternBreakParams& L);
};

// ---------------------------------------------------------------- 输入

/// 一次检测的全部输入（流水线内只读）
struct Input {
    cv::Mat signal;        ///< CV_32F 笔画信号，**整图尺寸**
    RegionInfo region;     ///< 必须是 Ellipse 或 Circle
    Config cfg;            ///< 字符模块的通用配置（信号处理用）
    Params params;

    bool valid() const {
        return !signal.empty() && signal.type() == CV_32F && region.isPattern();
    }
};

// ---------------------------------------------------------------- 几何

/// 拉直后的统一坐标系
///
/// 椭圆环在极坐标下半径随角度在 a 与 b 之间摆动，展开出来是波浪线，
/// 没法按半径限制。所以先做仿射把椭圆拉直成正圆：
///   绕环心反向旋转 -rot，再沿短轴方向拉伸 a/b，环就落在固定半径 a 上。
struct Geometry {
    EllipseGeom ell;              ///< 原始椭圆（全图坐标）
    double scaleY = 1.0;          ///< y 方向缩放比 = a/b
    cv::Mat affine;               ///< 3×3 仿射矩阵：原图 → 拉直图
    cv::Point2f center;           ///< 拉直图中的环心
    int radiusA = 0;              ///< 环半径（像素，= 长半轴）
    bool ok = false;

    /// 原图坐标 → 拉直图坐标
    cv::Point2f toStraight(cv::Point2f p) const;
    /// 拉直图坐标 → 原图坐标
    cv::Point2f toOriginal(cv::Point2f p) const;
};

// ---------------------------------------------------------------- 极坐标

/// 极坐标展开图
///
/// 与 cv::warpPolar 的关系：warpPolar 的 dsize.width 是**半径数**、
/// height 是**角度数**，出来的图是"行=角度、列=半径"。
/// 这里统一转置成"行=半径、列=角度"，因为按角度遍历更自然。
struct PolarMap {
    cv::Mat img;        ///< CV_32F；rows = 半径索引 0..radii-1，cols = 角度索引
    int angles = 0;     ///< 角度采样数
    int radii = 0;      ///< 半径采样数
    double maxR = 0.0;  ///< 半径索引 radii-1 对应的像素半径

    bool valid() const { return !img.empty() && angles > 0 && radii > 0; }

    /// 角度索引 → 度
    double degOf(int idx) const {
        return angles > 0 ? 360.0 * static_cast<double>(idx) / angles : 0.0;
    }
    /// 度 → 角度索引（自动折回 [0, angles)）
    int idxOf(double deg) const;

    /// 取某半径、某角度索引处的值（越界返回 0）
    float at(int r, int aIdx) const;
};

// ---------------------------------------------------------------- 剖面

/// 环带剖面：逐角度的墨量曲线
///
/// ink[i] 是第 i 个角度在环带内能取到的最大墨量。
/// 这是所有"沿周长看"的判据的基础量。
struct Profile {
    std::vector<double> ink;   ///< 原始墨量
    std::vector<double> rel;   ///< 除以 ringMedian 后的无量纲值
    double ringMedian = 0.0;   ///< 全环墨量中位数（归一化基准）
    int bandLo = 0;            ///< 实际使用的半径带内沿（像素索引）
    int bandHi = 0;            ///< 实际使用的半径带外沿

    bool valid() const { return !ink.empty() && ink.size() == rel.size(); }
    size_t size() const { return ink.size(); }

    /// 某角度索引处的相对墨量（越界按环形折回）
    double relAt(int i) const;
};

// ---------------------------------------------------------------- 结论

/// 单个检测器的结论
///
/// severity 是**跨检测器可比**的归一化严重度：
///   0 = 完全正常，1 = 严重缺陷
/// 每个检测器自己决定如何把原始分映射到 [0,1]，
/// 这样汇总时不需要知道各检测器的量纲。
struct Finding {
    bool triggered = false;
    double score = 0.0;      ///< 原始分，量纲由检测器定义，仅供显示
    double severity = 0.0;   ///< 归一化严重度 [0, 1]
    std::string detail;      ///< 人读说明（进 GUI 与报告）
    std::vector<cv::Point2f> at;          ///< 缺陷位置（**全图坐标**）
    std::vector<cv::Range> spansDeg;      ///< 缺陷角度区间（度，可能跨 0°）

    /// 便捷构造：一个角度区间的缺陷
    static Finding atSpan(double fromDeg, double toDeg, double sev, double sc,
                          const std::string& d);
};

/// 一次检测的总结果
struct Result {
    bool valid = false;
    Geometry geom;
    PolarMap polar;
    Profile profile;

    /// 各检测器结论，按参数里的顺序
    std::vector<std::pair<std::string, Finding>> findings;

    bool flagged = false;              ///< 任一检测器 severity >= minSeverity
    double severity = 0.0;             ///< 所有检测器的最大 severity
    std::vector<cv::Point2f> breakAt;  ///< 汇总缺陷点（全图坐标）

    /// 按名字取某个检测器的结论，找不到返回 nullptr
    const Finding* find(const std::string& name) const;
};

}  // namespace pattern
}  // namespace chardet
