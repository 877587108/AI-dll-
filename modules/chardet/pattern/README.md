# `chardet::pattern` —— 图案区检测流水线

> 位置：`modules/chardet/pattern/`
> 状态：接口层已建立，`notch_window` 与 `gap_run` 两个检测器已注册

---

## 一、这个模块为什么存在

改造前，图案检测只有一个函数 `analyzePatternBreak()`（`pattern_break.cpp`，500 行），把 9 个算法阶段硬编码串在一起：

```
拉直 → 预灰度阈值 → 极坐标展开 → 环内屏蔽 → 带型连通
     → 环→缺口 → 缺口合并 → 断点判定 → 打分
```

**每次换一个判据就得整段改写这个函数。** 实际情况是：为了找到 logo 环左上角那个断口，前后试了九轮——绝对阈值、相对墨量、角度、环厚度、墨量峰值、拓扑、8 种纹理特征 + 82 种组合、PCA 外观、批次相似度、定点窗口。每轮都在 `tools/` 下新写一个探针程序，每个探针自己抄一遍预处理代码。

后果是实验之间**无法互相比较**：每个探针的拉直矩阵、定位偏移、极化方式都有细微差别，于是同一个结论在不同探针下数值对不上，也无法确认哪个是真的。

这个模块把公共部分抽出来，让**换算法 = 实现一个接口 + 注册一行**。

---

## 二、三层接口

```
Input ──► IPreprocessor ──► Geometry + straightened + PolarMap
                                   │
                                   ▼
                          IProfileBuilder ──► Profile
                                   │
                                   ▼
                            IDetector（可多个）──► Finding
```

### 2.1 职责边界（硬约束）

| 层 | 只做 | 绝不做 |
|---|---|---|
| `IPreprocessor` | 把椭圆拉直成正圆、展开成极坐标 | 任何判断 |
| `IProfileBuilder` | 把环上的墨量量化成一条逐角度曲线 | 任何判断 |
| `IDetector` | 从曲线 / 极坐标图判缺陷 | **不接触原图信号** |

最后一条最重要。检测器拿到的必须是同一个 `PolarMap` 和同一个 `Profile`，这样两个检测器的差别才只在判据本身，不在输入。

### 2.2 独立调试每个阶段

```cpp
Input in = makeInput(...);
auto pre = chardet::pattern::makePreprocessor();
Geometry g;
cv::Mat straight;
chardet::pattern::PolarMap pm;
pre->run(in, g, straight, pm, nullptr);
// 此时可以 dumpPolar(pm, "x.png") 看展开图，不必跑检测器
```

调检测器时反过来：先 build 好 `PolarMap` 和 `Profile` 再反复调 `detect()`，不必每次重做预处理。

---

## 三、数据结构

全部定义在 `pattern_types.hpp`，这里是唯一定义处。

### 3.1 `Input` —— 一次检测的全部输入

```cpp
struct Input {
    cv::Mat signal;        // CV_32F，整图尺寸（strokeSignal 的结果）
    RegionInfo region;     // 必须 Ellipse 或 Circle
    Config cfg;
    Params params;
};
```

### 3.2 `Geometry` —— 拉直后的统一坐标系

```cpp
struct Geometry {
    EllipseGeom ell;              // 原始椭圆（全图坐标）
    double scaleY = 1.0;          // y 缩放比 = a/b
    cv::Mat affine;               // 3×3：原图 → 拉直图
    cv::Point2f center;           // 拉直图中的环心
    int radiusA = 0;              // 环半径（= 长半轴）
    bool ok = false;
};
```

提供 `toStraight()` / `toOriginal()` 做双向坐标换算。**所有对外输出的坐标必须转回 `image` 空间**，遵循 `docs/csv-schema.md` §2.5。

### 3.3 `PolarMap` —— 极坐标展开图

```cpp
struct PolarMap {
    cv::Mat img;      // CV_32F；rows = 半径索引，cols = 角度索引
    int angles = 0;
    int radii = 0;
    double maxR = 0.0;
};
```

**注意转置**：`cv::warpPolar` 出来是「行=角度、列=半径」，这里转成「行=半径」，因为按角度遍历更自然。

角度换算**只在这里做**：`degOf(idx)` / `idxOf(deg)`。以前各处自己乘 `360.0 / N`，`N` 一变就到处对不上。

### 3.4 `Profile` —— 逐角度墨量曲线

```cpp
struct Profile {
    std::vector<double> ink;   // 原始墨量
    std::vector<double> rel;   // 除以 ringMedian 后的无量纲值
    double ringMedian = 0.0;
    int bandLo = 0, bandHi = 0;
};
```

`rel` 用中位数归一化而不是均值：缺口会把均值拉低，中位数不受影响，于是 `rel` 反映的是「相对这枚料自身环的浓淡」，天然抵消曝光差异。

### 3.5 `Finding` —— 检测器结论

```cpp
struct Finding {
    bool triggered = false;
    double score = 0.0;      // 原始分，量纲由检测器定义
    double severity = 0.0;   // 归一化 [0,1]，跨检测器可比
    std::string detail;      // 人读说明
    std::vector<cv::Point2f> at;       // 缺陷位置（全图坐标）
    std::vector<cv::Range> spansDeg;   // 缺陷角度区间
};
```

`severity` 是唯一跨检测器可比的量。汇总时只看它，不看 `score`——因为各检测器的 `score` 量纲不同（划痕是「灰度×字数」，断点是「度的百分之一」）。

---

## 四、已注册的检测器

| 名字 | 文件 | 判据 |
|---|---|---|
| `notch_window` | `det_notch_window.cpp` | 240°~272° 窗口内的缺口深度 × 宽度 |
| `gap_run` | `det_gap_run.cpp` | 整圈 360° 找最大缺口，按角度判 |

### 4.1 `notch_window`

**为什么是定点而不是整圈**：缺口只有 1~4°，占整圈 0.3%~1.1%。环带纹理的逐角度波动与缺口深度相当，于是「整圈最大缺口」这个统计量被纹理噪声主导，真缺口淹没在里面。**前面九轮失败全是在整圈上做文章，根源在这里。**

限制到 240°~272°（环的左上角）之后，信号立即清晰：

| 数据 | 窗口内最低相对墨量 |
|---|---|
| ngv2（7 张，全部有缺陷） | 0.041 ~ 0.357 |
| `image/ok` 干净料 | 0.229 ~ 0.643 |

**角度约定**：极坐标 0° = 环的最右侧，90° = 环的最下方，逆时针增加，270° = 环的正上方。所以 240°~272° 覆盖左上角到正上方。

这个位置不是猜的：逐角度统计显示 240° 处 ngv2 与正常料的平均墨量差 +0.203（全环最大），250° 处 +0.160。

### 4.2 `gap_run`

老的整圈判据，保留用于对照。判决仍以 `analyzePatternBreak()` 的老实现为准，本检测器只用于实验对比。

---

## 五、怎么加一个新算法

以加一个「环带墨量比」检测器为例：

### 步骤 1：新建文件

`modules/chardet/pattern/det_ring_ink.cpp`

```cpp
#include "pattern_api.hpp"

namespace chardet {
namespace pattern {

class RingInkDetector final : public IDetector {
public:
    const char* name() const override { return "ring_ink"; }

    Finding detect(const Input& in, const Geometry& geom,
                   const PolarMap& pm, const Profile& prof) override {
        Finding f;
        if (!prof.valid()) { f.detail = "剖面无效"; return f; }

        // 你的判据
        double mean = 0.0;
        for (int i = 0; i < static_cast<int>(prof.size()); ++i)
            mean += prof.relAt(i);
        mean /= prof.size();

        f.score = mean;
        f.severity = std::clamp((1.0 - mean) / 0.5, 0.0, 1.0);
        f.triggered = f.severity >= 0.5;
        f.detail = "环带平均相对墨量 " + std::to_string(mean);
        return f;
    }
};

std::unique_ptr<IDetector> makeRingInkDetector() {
    return std::make_unique<RingInkDetector>();
}

}  // namespace pattern
}  // namespace chardet
```

### 步骤 2：注册（改两行）

`det_registry.cpp`：

```cpp
// 加一行声明
std::unique_ptr<IDetector> makeRingInkDetector();

void registerBuiltins() {
    ...
    r["ring_ink"] = &makeRingInkDetector;   // 加这一行
}
```

### 步骤 3：加进 CMake

`CMakeLists.txt`（模块级和主项目各一处）：

```cmake
modules/chardet/pattern/det_ring_ink.cpp
```

### 步骤 4：跑基准台对比

```bash
markloc_patternbench model/ tmpl.png unset.txt --detectors notch_window,ring_ink
```

**已有代码一行都不用动。** 这就是整个重构的目的。

---

## 六、自检

### 6.1 拉直自检

`preprocessor.cpp` 里有一个自检：长短轴端点变换后到环心的距离都应为 `a`，允许 5% 偏差。

**为什么必须有**：第一版把缩放写成「除以 a/b」，自检报短轴端距 40px（应为 137px），就是靠这个抓出来的。没有自检的话，后面所有角度采样都会擦边，而输出的「断口角度」看上去完全合理。

自检失败直接返回 `false`，**不带着错几何往下跑**——那种情况下的结论毫无意义。

### 6.2 `warpPolar` 的 dsize 顺序

`cv::Size(nR, nA)`：**width 是半径数，height 是角度数**。弄反了不报错，只是出来一张转置的图，半径带全错位。

---

## 七、与 `pattern_break.cpp` 的关系

老的 `analyzePatternBreak()` **一字未改**，仍在 `pattern_break.cpp` 里。它的存在是为了：

1. 不破坏既有调用点（GUI、`batchcore`、多个工具）
2. 提供行为基准——新流水线跑出来的数字必须能与它对比

桥接函数 `analyzePatternBreakV2()` 会用新流水线跑一遍，并把结果转回老的 `PatternBreakResult`。

**计划**：等新判据验证充分后，把老的缺口游程逻辑整个搬进 `det_gap_run.cpp`，然后删除 `pattern_break.cpp`。

---

## 八、待办

| 项 | 状态 |
|---|---|
| 接口层与两个检测器 | 已完成 |
| `markloc_patternbench` 基准台 | 已写，待编译验证 |
| 把 `analyzePatternBreak()` 的逻辑完全迁进 `det_gap_run.cpp` | 未开始 |
| 删除 `pattern_break.cpp` | 未开始 |
| 按 `docs/csv-schema.md` 产出 `*_detect.csv` | 未开始 |
| 把 `tools/` 下 20 多个实验探针合并进基准台 | 未开始 |
