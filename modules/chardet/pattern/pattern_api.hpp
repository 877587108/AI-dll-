#pragma once

// pattern_api.hpp —— 图案区检测的对外接口
//
// ================================================================
// 设计要点
// ================================================================
//
// 整条检测流水线拆成三类算子，每类一个纯虚接口：
//
//   IPreprocessor   输入信号 → 拉直图 + 极坐标图
//   IProfileBuilder 极坐标图 → 逐角度墨量剖面
//   IDetector       剖面/极坐标图 → 缺陷结论
//
// 三类算子的职责边界是硬的，不能互相越界：
//   · 预处理只管"把图摆正、展开"，不做任何判断
//   · 剖面只管"把环上的墨量量化成一条曲线"，不做判断
//   · 检测器只管"从曲线/图上判缺陷"，**不允许再碰原图信号**
//
// 最后一条最重要。以前每个探针都自己重新算 strokeSignal、自己做模板定位、
// 自己拉直、自己极化——于是每个实验的"输入"其实都不一样，
// 结论之间没法比较。现在检测器拿到的是同一个 PolarMap 和同一个 Profile，
// 差别只在判据本身。
//
// ================================================================
// 怎么加一个新算法
// ================================================================
//
//   1. 在 detector/ 下新建 det_xxx.cpp
//   2. 实现 class XxxDetector : public IDetector，name() 返回唯一名字
//   3. 在 det_registry.cpp 里加一行注册
//   4. 跑 markloc_patternbench 对比新旧检测器
//
// 已有代码一行都不用动。这就是这次重构的全部目的。
//
// ================================================================
// 怎么单独调试某个阶段
// ================================================================
//
// 每个算子都能被单独拉出来跑：
//
//   Input in = makeInput(...);
//   auto pre = std::make_unique<StandardPreprocessor>();
//   Geometry g; cv::Mat straight; PolarMap pm;
//   pre->run(in, g, straight, pm, nullptr);
//   // 此时可以把 straight / pm 存盘看，不必跑检测器
//
// 调试检测器时反过来：先 build 好 PolarMap 和 Profile 再反复调 detect()，
// 不必每次重新做预处理。

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "pattern_types.hpp"

namespace chardet {
namespace pattern {

// ---------------------------------------------------------------- 预处理

/// 预处理算子：把原图信号变成可分析的拉直图与极坐标图
class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;

    /// 唯一名字，用于参数选择与日志
    virtual const char* name() const = 0;

    /// 执行预处理
    ///
    /// @param in          输入（只读）
    /// @param geom        输出：几何信息（拉直矩阵、环心、半径）
    /// @param straightened 输出：拉直后的图（与原图同尺寸，CV_32F）
    /// @param polar       输出：极坐标展开图
    /// @param err         失败时写入原因，可为 nullptr
    /// @return 成功与否
    virtual bool run(const Input& in, Geometry& geom, cv::Mat& straightened, PolarMap& polar,
                     std::string* err) = 0;
};

// ---------------------------------------------------------------- 剖面

/// 剖面算子：从极坐标图提取逐角度的环带墨量曲线
class IProfileBuilder {
public:
    virtual ~IProfileBuilder() = default;
    virtual const char* name() const = 0;

    /// 构建剖面
    ///
    /// @param pm   极坐标图（只读）
    /// @param geom 几何信息（用 radiusA 换算半径带的比例）
    /// @param p    流水线参数
    /// @param out  输出剖面
    /// @return 成功与否（极坐标图无效、环带为空都算失败）
    virtual bool build(const PolarMap& pm, const Geometry& geom, const Params& p,
                       Profile& out) = 0;
};

// ---------------------------------------------------------------- 检测

/// 检测算子：从剖面与极坐标图判定缺陷
///
/// 注意：检测器**只读** PolarMap 和 Profile，不接触原图。
/// 若某个算法确实需要原图（例如在拉直图上做形态学），
/// 那它应该是一个 IPreprocessor 的产出，而不是在检测器里重算。
class IDetector {
public:
    virtual ~IDetector() = default;
    virtual const char* name() const = 0;

    /// 执行检测。返回的 Finding.at 必须是**全图坐标**。
    virtual Finding detect(const Input& in, const Geometry& geom, const PolarMap& pm,
                           const Profile& prof) = 0;
};

// ---------------------------------------------------------------- 注册表

/// 注册全部内置算子（程序启动时调用一次，重复调用无副作用）
void registerBuiltins();

/// 按名字创建检测器，找不到返回 nullptr
std::unique_ptr<IDetector> makeDetector(const std::string& name);

/// 已注册的检测器名字列表（用于帮助信息与参数校验）
std::vector<std::string> detectorNames();

/// 创建标准预处理算子（目前只有一种实现）
std::unique_ptr<IPreprocessor> makePreprocessor();

/// 创建标准剖面算子
std::unique_ptr<IProfileBuilder> makeProfileBuilder();

// ---------------------------------------------------------------- 顶层入口

/// 跑完整流水线
///
/// 这是模块对外的唯一检测入口。等价于：
///   preprocessor.run() → profileBuilder.build() → 依次跑 params.detectors → 汇总
///
/// 任何一个阶段失败都返回 valid=false 的 Result，不抛异常。
Result analyze(const cv::Mat& signal, const RegionInfo& region, const Config& cfg,
               const Params& params);

/// 便捷重载：用老参数结构体
Result analyze(const cv::Mat& signal, const RegionInfo& region, const Config& cfg,
               const PatternBreakParams& legacyParams);

// ---------------------------------------------------------------- 调试

/// 把极坐标图存成可视化 PNG（归一化到 0~255）
bool dumpPolar(const PolarMap& pm, const std::string& path);

/// 把剖面存成 CSV（两列：度, ink, rel）
bool dumpProfile(const Profile& prof, const PolarMap& pm, const std::string& path);

}  // namespace pattern
}  // namespace chardet
