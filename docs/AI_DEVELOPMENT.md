# 定位算法插件 —— AI 开发指南

> 本文档面向 **AI（DeepSeek 等）**，描述如何独立地开发、编译、评估一个定位算法插件。
> 目标是让 AI 无需人工介入，就能完成「写算法 → 编译 → 评估 → 迭代」的完整闭环。

---

## 一、定位算法是做什么的

给定：

- 一张**待测图**（芯片照片，可能是 1496×1496 或 1688×1460 等不同尺寸）
- 一个**模型**（含模板频率图 `freq.png` + 字符框 `regions.txt`，模型坐标 1688×1460）

任务：找出模型里每个**字符框**在待测图上的实际位置。

```
模型字符框（模型坐标）         待测图（实际位置）
┌─────────┐                  ┌─────────┐
│  Q046   │  ──定位──►      │  Q046   │  ← 偏移 + 尺度变化后的位置
└─────────┘                  └─────────┘
```

**难点**：两图尺寸可能不同、有平移、有轻微旋转、有曝光差异。定位算法要鲁棒地处理这些。

---

## 二、平台闭环（AI 的工作流程）

```
① 写算法          复制 plugins/locate_template/plugin_template.cpp
                   实现 detect() 里的定位逻辑
                        │
                        ▼
② 编译             scripts/build_plugin.bat <源码.cpp> <输出.dll>
                   （cl.exe 直编，秒级，不碰 CMake）
                        │
                        ▼
③ 评估             scripts/eval_plugin.bat <dll> <模型> <图列表>
                   输出：定位成功率、偏差分布、逐字符详情
                        │
                        ▼
④ 迭代            读指标 → 判断哪里不行 → 改算法 → 回到①
```

---

## 三、接口契约（唯一需要遵守的规范）

完整定义见 `include/locate_plugin/locate_plugin.hpp`。核心是：

### 3.1 输入（detect 的参数）

```cpp
LocateResult detect(
    const void* signal,        // 待测图的笔画信号（CV_32F，整图尺寸）
                               // static_cast 成 const cv::Mat* 使用
    int imageW, int imageH,    // 待测图尺寸
    int modelW, int modelH,    // 模型图尺寸
    const TemplateBox* boxes,  // 模板字符框（模型坐标），长度 boxCount
    int boxCount
);
```

### 3.2 输出（LocateResult）

```cpp
struct LocateResult {
    float scaleX, scaleY;    // 整图仿射：检测图 = 模型 × scale + offset
    float offsetX, offsetY;
    const LocatedChar* chars;  // 每个字符的定位结果
    int charCount;
    bool globalOk;            // 整图是否可信
    const char* message;      // 失败说明
};
```

### 3.3 硬性规则（违反会导致评估错误）

1. **顺序**：`chars[i]` 必须对应 `boxes[i]`，不能重排
2. **坐标**：`chars[i].x/y/w/h` 必须是**待测图坐标**（原图像素，左上原点）
3. **生命周期**：`chars` 指向的内存必须活到下次 `detect()` 或 `release()`。模板用静态 `std::vector` 保证这一点
4. **不抛异常**：`detect()` 里所有异常要自己 catch，返回空结果。模板已用 try/catch 包住
5. **scale 语义**：`scaleX/scaleY` 是插件报告的整图仿射缩放（待测图 = 模型 × scale + offset）。
   它由插件自己估出，不必等于图幅比例；bench 用它做偏差参照，也仅作诊断。

---

## 四、评估指标（locate_bench 自动算，你不需要算）

评估**不需要人工真值**。有**两类指标**，务必区分：

### 4.1 墨量验证（独立准确度，最权威）

bench 会算每个定位框内的**墨量比例**（灰度 > 100 的像素占比）：

```
墨量 ink = 框内"有墨"像素数 / 框面积
```

这是**独立于任何参照系假设**的客观指标：框套在字符上 → 墨量高；框偏了 / 套空 → 墨量低。

| 指标 | 含义 | 越好 |
|---|---|---|
| 墨量均值 / 中位 | 所有 accepted 框的平均墨量 | 越高 |
| 高墨率(≥0.15) | 墨量达标（套在字符上）的框占比 | 越高，理想 100% |

**判断算法好不好，优先看墨量验证**，而不是偏差（见下）。

### 4.2 偏差（内部一致性，仅参考）

偏差对照的参照位置，**不再用图幅比例**，而是插件自己报告的整图仿射：

```
参照位置 ex = templateBoxes[i].x × res.scaleX + res.offsetX
偏差     dx = chars[i].x - ex
```

这反映的是"逐字符结果相对插件整图变换的残差"，**不是绝对准确度**。

> ⚠️ 为什么不能拿偏差当绝对准确度：
> 实测 unset 相对模型的真实变换是 **scale≈1.0 + 小平移**（不是图幅比例
> 0.886）。若用图幅比例当参照，会算出 50~100px 的假偏差。
> 更关键的是：字符级定位有**内在模糊性**——相邻的相似字符（如同批次的
> 数字、斜杠）会被算法"抓"到错误但同样有墨的位置，此时框仍套着墨
> （墨量高），但偏差大。所以偏差只能反映一致性，墨量才反映"是否真套字"。

| 指标 | 含义 | 越好 |
|---|---|---|
| 定位成功率 | accepted 字符数 / 总字符数 | 越高 |
| 墨量均值 | accepted 框的平均墨量 | 越高 |
| 高墨率(≥0.15) | 墨量达标框占比 | 越高 |
| 整图可信率 | globalOk 的图占比 | 越高 |

**判断算法好不好**：在**同一批图**上跑不同插件，优先比较墨量验证，再参考成功率。

---

## 五、标准测试集

建议固定的评估数据集（列表文件，每行一个图路径）：

| 文件 | 内容 | 用途 |
|---|---|---|
| `out/ok00.txt` | image/ok 的 _00 视角（83 张，1688×1460） | 与模型同尺寸，验证基本定位 |
| `out/unset_scale.txt` | image/unset 的 _00（1496×1496） | 跨尺寸，验证尺度鲁棒性 |

**已知挑战**：unset 是 1496×1496，图幅比例与模型不同。但实测发现，模型与 unset 的**真实变换是 scale≈1.0 + 小平移（约 -20~-47, -17~-33 像素），旋转≈0**——并非图幅比例缩放 0.886。两批图的画幅不同（4:3 vs 正方形），但 logo 和字符的**物理尺寸一致**，只是整体位置偏移。跨分辨率定位的核心难点是**画幅裁切不一致**，不是等比缩放。

---

## 六、快速上手（一个完整例子）

### 1. 复制模板

```bash
cp plugins/locate_template/plugin_template.cpp plugins/locate_myalg/plugin_myalg.cpp
```

### 2. 改三处

- `kName` = "myalg"
- 在 `detect()` 里实现你的算法
- 在 `setParams()` / `apiListParams()` 里加你的参数

### 3. 编译

```bash
scripts/build_plugin.bat plugins/locate_myalg/plugin_myalg.cpp out/myalg.dll
```

### 4. 评估

```bash
scripts/eval_plugin.bat out/myalg.dll release_defectview/model/v00 out/ok00.txt out/eval_myalg
```

### 5. 看结果

输出在 `out/eval_myalg/locate_result.csv`，汇总打印在终端。可视化用：

```bash
build/Release/markloc_locateview.exe
```

GUI 里选你的 DLL + 模型 + 图，跑定位，鼠标悬停看模板/检测图对应区域的对比。

---

## 七、已知的参考实现

| 插件 | 算法 | 特点 | unset 定位 |
|---|---|---|---|
| `plugins/locate_orb/plugin_orb.cpp` | ORB 特征点 + 单应矩阵 + 逐字符 NCC 精调 | **最优**，抗旋转尺度，精调后套字更准 | 220/220（100%），墨量 0.382 |
| `plugins/locate_ncc2/plugin_ncc2.cpp` | 逐字符 NCC 模板匹配（算法①） | 独立逐字，但天花板约 72%，缺全局约束 | 160~152/220 |
| `plugins/locate_ncc/plugin_ncc.cpp` | 整图 NCC 模板匹配 | 基线，整图 NCC 被背景稀释，已证失败 | 0% |
| `plugins/locate_template/plugin_template.cpp` | 同上（教学版） | AI 复制的起点 | — |

### 关键结论（务必理解）

1. **信号域必须一致**：插件收到的 `signal` 是**笔画信号**（高斯模糊去背景后的
   高频残差，笔画亮背景≈0），不是原始灰度。模板若用 `original.png` 的原始灰度
   直接 NCC，相关系数只有 0.30（背景稀释 + 灰度反向）；要先对模板做同样的
   `computeSignal` 变换（见 `plugin_ncc2.cpp`），才进同域比较。

2. **整图 NCC 失败**：芯片图大部分是空白背景，整图归一化相关被背景稀释到
   <0.30，无法超过阈值。局部字符区域 NCC 可达 0.65。

3. **逐字符独立 NCC 有天花板**：缺乏全局一致性约束，相邻相似字符会互相
   "吸"错（如 line2 的 7 个字符被整体误匹配到 line1）。实测最高约 72%。

4. **ORB + 精调最优**：先整图 ORB 求单应矩阵（吸收平移/旋转/尺度），再对
   每个字符在粗定位邻域做 NCC 精调。但精调对"行内相似字符"仍可能抓错
   （墨量仍高，因为抓到的也是字符）——这是字符级定位的固有模糊性。

---

## 八、目录结构

```
include/locate_plugin/
├── locate_plugin.hpp      插件 ABI（接口契约，唯一权威定义）
└── plugin_loader.hpp      DLL 加载器（宿主用）

plugins/
├── locate_ncc/            参考实现：整图 NCC 匹配（已证失败）
├── locate_ncc2/           算法①：逐字符 NCC 匹配
├── locate_orb/            算法③：ORB 特征点 + 精调（最优）
└── locate_template/       AI 开发模板

scripts/
├── build_plugin.bat       一键编译 .cpp → DLL
└── eval_plugin.bat        一键评估

tools/locate_bench.cpp     命令行评估器（宿主；含墨量验证）
src/gui/LocateViewWindow.* 可视化调试平台（宿主）
```

---

## 九、常见坑（AI 踩过的，别重复）

1. **中文路径**：`cv::imread` 在 Windows 按 ANSI 解释路径会失败，用 `readGrayFile()`（模板里已提供）
2. **`matchTemplate` 第 5 参数**：是 mask 不是搜索区。要限搜索区就 `signal(searchRect)` 裁出来再匹配
3. **`min`/`max` 宏冲突**：`windows.h` 的 min/max 宏破坏 `std::max`。插件若包含 windows.h 要先 `#define NOMINMAX`
4. **信号域 ≠ 原始灰度**：插件拿到的 signal 是笔画信号。模板匹配前要对模板做
   同样的去背景变换，否则 NCC 只有 0.30（见第七节关键结论 1）
5. **坐标系**：chars 输出必须是待测图坐标，不是模型坐标、不是拉直图坐标
6. **逐字符匹配会抓错相似字符**：相邻相似字符会导致误匹配，墨量验证可能
   不报错（抓到的也是字符）。评估时区分"墨量"（是否套字）和"偏差"（是否套对）
