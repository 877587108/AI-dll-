# CSV 数据契约 —— 算法与 GUI 的边界规范

> 状态：**已定稿**（2026-10-23）
> 适用范围：`markloacation` 全部图像检测产线
> 约束力：算法侧与 GUI 侧都必须遵守；违反即视为缺陷

---

## 零、这份文档要解决什么

当前代码库的真实状况（调查所得，非推测）：

| 问题 | 证据 |
|---|---|
| 检测逻辑长在渲染代码里 | `DefectViewWindow.cpp` 的 L1383-1466（溢料 PCA）、L1472-1515（污损）、L1527-1570（局部缺墨）三段纯算法，位于 `runDetect()` 渲染流程中 |
| 同一件事两份实现 | `DefectViewWindow.cpp` L1071-1141（strokeSignal + 相位相关配准 + warpAffine + 逐字符对齐）与 `batchcore.hpp:140-215` 逐行重复 |
| 单张页与批量页判定口径不同 | 单张 `DefectViewWindow.cpp:1666-1673` 只按划痕判 NG；批量 `batchcore.hpp:325` 按「划痕 **或** 图案断」判 NG。**同一张图两页结论可能相反** |
| 批量页少 4 条判据 | 批量无溢料、污损、局部缺墨、多模板 |
| CSV 三种互不兼容格式 | GUI 写 12 列（`DefectViewWindow.cpp:1890`）、命令行写 14 列（`defectview_main.cpp:178`）、`out/review/NG.csv` 是 UTF-16LE |
| 没有公共 CSV 工具 | 7 处写入、2 处读取，各自手写 |
| 尺寸不符静默失败 | `batchcore.hpp:126` 的 `if (gray.size() != model.freq.size()) return row;` —— 尺寸不符直接返回空结果，既不报错也不判 NG |

**本规范的目标**：把「算什么」和「怎么画」彻底切开。

- **算法侧**只负责算，产物是 CSV
- **GUI 侧**只负责读 CSV 并渲染，不做任何检测计算
- 各阶段之间**只传递几何**，不传递质量判断

---

## 一、阶段划分

检测拆成三个**互不重叠**的阶段，每阶段独立产出一份 CSV。

```
┌──────────────────────────────────────────────────────────────┐
│  原图                                                        │
└───────────────┬──────────────────────────────────────────────┘
                │
        ┌───────▼────────┐
        │  ① 定位 locate │  找出 logo / 字符区在整图中的位置
        │                │  产物：*_locate.csv（一行一个候选）
        └───────┬────────┘
                │  只传几何：框的坐标、尺寸、角度
        ┌───────▼────────┐
        │  ② 分割 split  │  在已定位区域内切出字符
        │                │  产物：*_split.csv（一行一个字符框）
        └───────┬────────┘
                │  只传几何：字符框
        ┌───────▼────────┐
        │  ③ 检测 detect │  对每个字符框 / 图案区跑各检测器
        │                │  产物：*_detect.csv（一行一个检测对象）
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  GUI 只读渲染  │  读三份 CSV + 原图，画框、上色、显示表格
        └────────────────┘
```

### 阶段职责边界

| 阶段 | 做什么 | 不做什么 |
|---|---|---|
| **定位** | 模板匹配、校准、坐标换算 | 不切字符、不判缺陷 |
| **分割** | 在给定框内做投影切分 / 连通域 | 不判缺陷、不修正定位结果 |
| **检测** | 信号处理、比对、评分、判定 | 不重新定位、不重新切分 |

**关键**：检测阶段**不允许**重新做模板匹配或重新切字符。它拿到的区域就是分割阶段给的，无论那个区域对不对。

---

## 二、CSV 通用约定

### 2.1 文件命名

一次运行（一张图或一个批次）在同一个输出目录下产出：

```
<输出目录>/
├── 0001_locate.csv      # 定位阶段
├── 0001_split.csv       # 分割阶段
├── 0001_detect.csv      # 检测阶段
├── _index.csv           # 批次索引（每图一行）
└── <原图>
```

单张运行时 `<stem>` 取图片文件名主干（去掉扩展名）。
批量运行时，每张图一份三件套，按 `<stem>` 区分。

### 2.2 编码与分隔

| 项 | 规定 |
|---|---|
| 编码 | **UTF-8 无 BOM** |
| 分隔符 | 英文逗号 `,` |
| 换行 | `\n`（不写 `\r\n`） |
| 转义 | 含 `,` `"` 或换行的字段用双引号包裹，内部 `"` 写成 `""` |
| 数值格式 | `%.6g`（自动去除多余零，保留 6 位有效数字） |
| 布尔 | `0` / `1`（不用 `true`/`false`） |
| 空值 | 空字符串（不用 `NA`/`NULL`） |

**禁止 UTF-16**：现有 `out/review/NG.csv` 是 UTF-16LE，那是历史遗留，新格式不再产生。

### 2.3 表头与注释

第一行是表头。表头之前**可以有**注释行，以 `#` 开头：

```
# csv-version=1
# stage=detect
# image_root=D:/markloacation/image
# image_file=unset/Failed [2026-1021-142823-358] (字符) _00.tif
# image_w=1496
# image_h=1496
# generated=2026-10-23T17:30:00
# algorithm=pattern.notch_window+gap_run; chardet.diffNcc
image_file,frame,region_id,char_index,detector,...
```

注释行供人和工具读取元信息。**解析器必须跳过 `#` 开头、以及第一个 `#` 之前的空行**。

### 2.4 路径约定

- **`image_root`**：注释行给出一次，是根目录的**绝对路径**
- **`image_file`**：数据列，是相对于 `image_root` 的路径，**用 `/` 分隔**（即使在 Windows 上）

这样一套 CSV + 图片可以整体搬到任何位置，只要同时改 `image_root`；也可以只发 CSV 给别人看，靠 `image_file` 知道原来叫什么。

**不存绝对路径**：一换机器就失效，且泄露本机目录结构。

### 2.5 坐标约定

每一行都带 `frame` 列，标明该行几何数据属于哪个坐标系：

| `frame` 取值 | 含义 | 谁用 |
|---|---|---|
| `image` | 整图原始像素坐标，原点在左上角 | **默认，绝大多数行** |
| `model` | 模型坐标系（训练时的标准图尺寸） | 仅建模阶段 |
| `straight` | 拉直图坐标（椭圆被仿射成正圆后的坐标） | 仅图案检测内部调试 |

**默认强制用 `image`**。现有 GUI 的两个真 bug 全都源于坐标系混淆：

- 图案椭圆被多移了一次 `(sh.x, sh.y)`，导致偏移 24px
- `warpAffine` 的 `WARP_INVERSE_MAP` 方向写反，内容往反向偏了一个环宽

把「CSV 里写清楚这一行是哪个 frame」定死，渲染器就不需要猜。

### 2.6 尺寸适配（不同大小 / 不同格式的图片）

这是**硬约束**：必须能处理不同尺寸、不同格式的输入。

#### 格式

| 项 | 规定 |
|---|---|
| 读取方式 | `cv::imdecode(buf, cv::IMREAD_UNCHANGED)`，然后自己转灰度 |
| 支持格式 | `.bmp` `.tif` `.tiff` `.png` `.jpg` `.jpeg` |
| 16 位 TIFF | 保留原动态范围（用 `IMREAD_UNCHANGED` 读入是 `CV_16U`），转灰度后**不强制压成 8 位**，由 `strokeSignal` 内部归一化 |
| 带 alpha 的 PNG | 4 通道 → 用 alpha 加权合成到黑底：`gray = sum(rgb * a/255)`，避免透明区被当成白底 |
| 彩色图 | 标准加权 `0.299R + 0.587G + 0.114B`（OpenCV 的 `COLOR_BGR2GRAY`） |

**为什么不用 `IMREAD_GRAYSCALE`**：它会把 16 位直接压成 8 位，丢掉一半动态范围。实测同一张图存 16 位与 8 位，`strokeSignal` 出来的信号尺度差 2 倍，所有阈值全部失准。

#### 尺寸

**不做统一缩放。** 按比例换算坐标与阈值：

```
scale = 实际图宽度 / 模型图宽度
```

- 模型坐标 → 实际图坐标：乘以 `scale`
- 像素阈值（划痕最小长度等）→ 实际图阈值：乘以 `scale`
- CSV 里记录**缩放后的实际阈值**（见 §4.7），不记原始值

**为什么不做统一缩放**：缩放会引入重采样，改变笔画粗细与纹理，而现有阈值（连通 3 像素、环带宽度）都是在原始分辨率下标定的。按比例换算坐标则无损。

#### 尺寸不符时的行为

**删除** `batchcore.hpp:126` 的 `if (gray.size() != model.freq.size()) return row;`。

改为：尺寸不符**不提前拒绝**，而是照常进入各阶段，由各阶段自己判断能否处理，并在 `stage_status` 里如实报告（见 §4.6）。

---

## 三、阶段隔离规则（最重要的一条）

> **任何阶段只能读上一阶段的几何输出，不得读上一阶段的质量字段。**

具体地说：

| 阶段 | 允许读 | 禁止读 |
|---|---|---|
| 分割 | `x, y, w, h, angle, shape` | `ncc, scale, yaw, z, cand_rank, is_chosen` |
| 检测 | `region_id, char_index, x, y, w, h` | `ncc, split_confidence, status` |

**为什么这条这么硬**：

你提出的原话是「不要遗留警告让后面算法偷看」。这句话点出了一个真实的腐蚀模式：

1. 定位阶段发现「这次 ncc 只有 0.62，不太确定」
2. 把这个警告写进 CSV
3. 检测阶段看到警告，于是「放宽阈值，免得误报」
4. 结果：**一个定位错误被下游悄悄吸收了**，报告上只显示「检测通过」
5. 真正的定位问题永远暴露不出来，直到产线上出现批量漏检

正确的做法是：定位不准就在**定位 CSV 自己**里体现（`ncc` 低、候选分散），由看报表的人发现。检测阶段老老实实按给定的框检测，框错了就把错误结果输出来——让错误可见，而不是被掩盖。

**接口层强制**：`charinsp::SplitStage::run()` 的入参类型只有几何（见 §5.2），编译期就拿不到质量字段。

---

## 四、CSV 列定义

### 4.1 `*_locate.csv` —— 定位阶段

**一行一个候选。** 模板匹配天然产生 K 个候选，都写出来才能排查「是不是选错了候选」。

| 列名 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `image_file` | str | — | 相对 `image_root` 的路径 |
| `frame` | enum | — | 固定 `image` |
| `region_id` | int | — | 模型里该区域的序号（0=logo 环，1..N=字符区） |
| `shape` | enum | — | `box` / `ellipse` / `circle` |
| `cand_rank` | int | — | 候选排序，0 = 最优 |
| `is_chosen` | bool | — | 本次实际采用该候选 |
| `x` | float | px | 框左上角 x（`box`）或椭圆中心 x（`ellipse`/`circle`） |
| `y` | float | px | 同上 |
| `w` | float | px | 框宽（`box`）或长轴 2a（`ellipse`/`circle`） |
| `h` | float | px | 框高（`box`）或短轴 2b（`ellipse`/`circle`） |
| `angle` | float | deg | 长轴相对 +x 的角度，逆时针为正。`box` 填 0 |
| `ncc` | float | — | 模板匹配的归一化互相关分数，[-1, 1] |
| `scale` | float | — | 相对模型的缩放比（`实际图宽 / 模型图宽`） |
| `yaw` | float | deg | 绕光轴的旋转角（多视角产品补偿用），无则空 |
| `z` | float | px | 偏移补偿量，无则空 |
| `status` | enum | — | `ok` / `fail:模板无匹配` / `fail:图像过小` |

**`is_chosen` 的唯一性**：同一 `(image_file, region_id)` 下**必须恰好一行** `is_chosen=1`。没有则是定位失败。

**为什么保留所有候选**：定位阶段的错误最难从下游发现。`out/angle_map.png` 那次调查里，椭圆偏 24px 的结果是「断口角度全部偏移 3°」，看上去像是算法阈值问题，实际是定位选错了候选。只有把 `cand_rank` 和各自的 `ncc` 写出来，才能一眼看出。

### 4.2 `*_split.csv` —— 分割阶段

**一行一个字符框。**

| 列名 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `image_file` | str | — | 相对 `image_root` 的路径 |
| `frame` | enum | — | 固定 `image` |
| `region_id` | int | — | 所属区域序号，与定位 CSV 对应 |
| `char_index` | int | — | 区域内从左往右的序号，从 0 开始 |
| `x` | int | px | 字符主体框左上角 x |
| `y` | int | px | 同上 |
| `w` | int | px | 主体框宽 |
| `h` | int | px | 主体框高 |
| `extend_x` | int | px | 扩展比对框左上角 x（比主体框大，用于捕捉「笔画长出去」） |
| `extend_y` | int | px | 同上 |
| `extend_w` | int | px | 扩展框宽 |
| `extend_h` | int | px | 扩展框高 |
| `row` | int | — | 所属文字行号（自动分行时有效），单行区填 0 |
| `status` | enum | — | `ok` / `empty`（该区域没切出字符） |

**为什么同时给主体框和扩展框**：主体框是墨的实际范围，扩展框是允许笔画长出去的外扩范围。检测阶段用主体框判缺墨，用扩展框判「这段多出来的墨算不算溢出到别人地盘」。两者缺一不可。

### 4.3 `*_detect.csv` —— 检测阶段

**一行一个检测对象 × 一个检测器。**

一个字符框跑 4 个检测器（缺墨/多墨/划痕/溢料），就产生 4 行。一个图案区跑 2 个检测器（断点/缺墨），产生 2 行。

**公共列**：

| 列名 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `image_file` | str | — | 相对 `image_root` 的路径 |
| `frame` | enum | — | 固定 `image` |
| `region_id` | int | — | 区域序号 |
| `char_index` | int | — | 字符序号；图案区填 -1 |
| `object_kind` | enum | — | `char` / `pattern` |
| `detector` | str | — | 检测器名，见下表 |
| `score` | float | — | 该检测器的原始分，量纲由检测器定义 |
| `thr` | float | — | **缩放后的实际阈值**（已乘 `scale`） |
| `label` | str | — | 该检测器自己的档位标签 |
| `triggered` | bool | — | `score` 越过 `thr` 并判定不合格 |
| `severity` | float | — | 归一化严重度 [0,1]，用于跨检测器排序 |
| `detail` | str | — | 人读说明，进 GUI 悬浮提示 |
| `geom_kind` | enum | — | `none` / `point` / `rect` / `ellipse` / `line` |
| `geom` | str | px | 几何描述，格式见 §4.4 |

**各检测器的 `label` 取值**（由检测器自己定义，此处列举当前实现）：

| `detector` | 适用对象 | `label` 取值 | `score` 含义 |
|---|---|---|---|
| `missing` | char | `ok` / `ng` | 缺墨比例，[0,1] |
| `extra` | char | `ok` / `ng` | 多墨比例，[0,1] |
| `scratch` | char | `ok` / `suspect` / `obvious` / `certain` | 深度 × 跨字数 |
| `appendage` | char | `ok` / `ng` | 溢料连通块的长宽比 |
| `dirt` | char | `ok` / `ng` | 污损面积 |
| `pattern_break` | pattern | `ok` / `ng` | 断口角度 × 100 |
| `pattern_notch` | pattern | `ok` / `ng` | 窗口缺口深度 × 宽度 |
| `pattern_missing` | pattern | `ok` / `ng` | 环带墨量比 |

**新增检测器只需增加新的 `detector` 取值**，不改表头结构。

### 4.4 `geom` 列的编码

`geom_kind` 决定 `geom` 列的内容格式：

| `geom_kind` | `geom` 格式 | 示例 |
|---|---|---|
| `none` | 空 | `` |
| `point` | `x;y` | `1123.4;886.2` |
| `rect` | `x;y;w;h` | `980;740;120;160` |
| `ellipse` | `cx;cy;a;b;angle` | `652.6;433.1;134.25;77.05;-13.27` |
| `line` | `x1;y1;x2;y2` | `1010;800;1090;860` |
| `points` | `x1;y1\|x2;y2\|...` | `1123;886\|1130;890` |

**用分号而不是逗号**：逗号是 CSV 分隔符，混用会逼着每个字段都加引号，表格软件里也难读。

### 4.5 `_index.csv` —— 批次索引

批量运行时每图一行，供 GUI 列目录、快速筛选。

| 列名 | 类型 | 含义 |
|---|---|---|
| `image_file` | str | 相对路径 |
| `image_w` | int | 原图宽（像素） |
| `image_h` | int | 原图高 |
| `image_format` | str | `bmp` / `tif` / `png` / `jpg` |
| `scale` | float | 相对模型的缩放比 |
| `locate_status` | enum | `done` / `failed:...` |
| `split_status` | enum | `done` / `failed:...` |
| `detect_status` | enum | `done` / `failed:...` |
| `verdict` | enum | `OK` / `NG` / **`NOT_CHECKED`** |
| `ng_detectors` | str | 判 NG 的检测器名，逗号分隔（如 `scratch,pattern_break`） |
| `max_severity` | float | 全图最高 severity |
| `elapsed_ms` | int | 三阶段总耗时 |

**`NOT_CHECKED` 是独立取值，不是 `OK` 的同义词。** 见 §4.6。

### 4.6 `stage_status` 语义

每个阶段都可能没跑成。三种情况**必须区分**：

| 取值 | 含义 | GUI 显示 |
|---|---|---|
| `done` | 正常完成 | 正常渲染 |
| `skipped:<原因>` | 主动跳过（如该区域不存在） | **灰色，标「未检」** |
| `failed:<原因>` | 执行失败（读图失败、拉直自检失败等） | **灰色，标「未检」** |

**把「没验过」显示成「合格」是这类系统最危险的错误。** 现有代码正好犯了这个错：`batchcore.hpp:126` 尺寸不符时返回空结果，然后 `row.ng` 默认 false，报告里就成了一片绿色的 OK。

`_index.csv` 的 `verdict` 只有在三个阶段都是 `done` 时才算数；任何一个是 `failed` 或 `skipped`，`verdict` 必须是 `NOT_CHECKED`。

### 4.7 阈值记录规则

`thr` 列存**缩放后的实际阈值**：

```
thr = 模型里标定的阈值 × scale
```

例如划痕最小长度在模型里标定 40px，实际图宽度是模型的 1.06 倍，则 CSV 里写 `thr=42.4`。

**不存原始值**，因为读 CSV 的人需要的是「这次拿什么标准判的」，而不是「模型里的常量是多少」。原始值在模型文件里，随时可查。

---

## 五、接口定义

### 5.1 算法侧的产出接口

每个阶段实现同一个接口：

```cpp
namespace charinsp {

/// 阶段执行状态
enum class StageStatus { Done, Skipped, Failed };

/// 阶段执行结果
struct StageOutcome {
    StageStatus status = StageStatus::Failed;
    std::string reason;      // Skipped/Failed 时填原因，Done 时为空
    std::string csvPath;     // 写出的 CSV 路径
};

class IStage {
public:
    virtual ~IStage() = default;
    virtual const char* name() const = 0;
    virtual StageOutcome run(const StageInput& in, const std::string& outDir) = 0;
};

}  // namespace charinsp
```

### 5.2 阶段间的输入类型（强制隔离）

这是 §3 那条规则的**编译期实现**——下游阶段的入参类型里**根本没有**上游的质量字段：

```cpp
namespace charinsp {

/// 定位阶段交给分割阶段的几何（只有几何，没有分数）
struct LocatedGeometry {
    int    regionId = 0;
    RegionShape shape = RegionShape::Box;
    cv::Rect2f rect;          // box 用外接框
    EllipseGeom ell;          // ellipse/circle 用
    float  scale = 1.0f;      // 相对模型的缩放比（几何换算需要，不是质量分）
};

/// 分割阶段交给检测阶段的几何（只有框，没有置信度）
struct CharGeometry {
    int regionId = 0;
    int charIndex = 0;
    cv::Rect bbox;
    cv::Rect extend;
    int row = 0;
};

/// 分割阶段的输入：只能拿到几何
struct SplitInput {
    const std::vector<LocatedGeometry>* regions = nullptr;
    // 注意：这里没有 ncc、没有 status、没有 cand_rank
};

/// 检测阶段的输入：只能拿到几何
struct DetectInput {
    const std::vector<CharGeometry>* chars = nullptr;
    const std::vector<LocatedGeometry>* patterns = nullptr;
    // 注意：这里没有 split_confidence、没有 ncc
};

}  // namespace charinsp
```

**分割阶段若想读定位的 `ncc`，编译不过。** 这就是把规则固化到类型系统里，而不是写在文档里靠自觉。

### 5.3 GUI 侧的读取接口

GUI 只依赖 CSV 解析，不依赖任何算法头文件：

```cpp
namespace charinsp::ui {

/// 一份 CSV 解析后的通用表（列名 → 列值）
struct Table {
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;
    std::map<std::string, std::string> meta;   // 注释行里的 key=value
};

/// 读 CSV（跳过注释、处理转义、UTF-8）
Table readCsv(const std::string& path);

/// 按图片聚合三个阶段
struct ImageBundle {
    std::string imagePath;
    Table locate, split, detect;
};
std::vector<ImageBundle> loadBundle(const std::string& dir);

}  // namespace charinsp::ui
```

**GUI 的 `#include` 白名单**：

- 允许：`charinsp::ui::*`（CSV 解析）、Qt、OpenCV 绘图
- **禁止**：`chardet.hpp`、`core/*.hpp`、任何检测算法头文件

这条用 CI 检查：`src/gui/` 下出现 `#include "chardet` 或 `#include "core/` 即判失败。

---

## 六、GUI 边界

### 6.1 允许做

| 类别 | 例子 |
|---|---|
| 读文件 | 读 CSV、读图片、读模型元信息（仅尺寸用于显示） |
| 绘图 | 画框、画椭圆、画文字、画点、着色 |
| 表格 | 显示、按列排序、按列过滤、切换显示哪些检测器 |
| 交互 | 点击框高亮、悬停显示 `detail`、缩放平移 |
| 导出 | 把当前视图存成 PNG（纯截图，不重算） |

### 6.2 禁止做

| 类别 | 现在的违规代码 |
|---|---|
| 任何 `cv::matchTemplate` | — |
| 任何 `chardet::strokeSignal` | `DefectViewWindow.cpp:1071` |
| 任何 `cv::phaseCorrelate` 配准 | `DefectViewWindow.cpp:1101` |
| 任何 `cv::warpAffine` 对齐 | `DefectViewWindow.cpp:1121,1125` |
| 检测逻辑 | 溢料 `L1383-1466`、污损 `L1472-1515`、局部缺墨 `L1527-1570` |
| NG 判定 | `L1666-1673` |
| 阈值重算 | `L2243-2299` 的 `edgesOf()` |

### 6.3 颜色映射规则（GUI 唯一允许的「判断」）

GUI 对每一行只做一件事：

```
if (triggered == 1)   → 红色（NG）
else if (severity > 0.5 × thr_severity) → 黄色（可疑）
else                  → 绿色（OK）
```

**这只是颜色映射，不是判定。** `triggered` 已经由算法算好写进 CSV，GUI 不重新比较 `score` 与 `thr`。

若用户想「临时改个阈值看看」，那是**重新跑算法**，不是在 GUI 里改。

---

## 七、迁移清单

### 7.1 从 GUI 搬到 core 的代码

| 源位置 | 内容 | 目标 |
|---|---|---|
| `DefectViewWindow.cpp:1071` | `strokeSignal` 重算 | `core/pipeline.cpp`（已有，去重） |
| `DefectViewWindow.cpp:1101-1134` | 相位相关配准 + `warpAffine` | `core/char_alignment.hpp`（已有 `locateCharacters`，把配准也并进去） |
| `DefectViewWindow.cpp:1383-1466` | 溢料 PCA 检测 | `core/detect_appendage.cpp`（已有 `scoreAppendage`，补齐调用） |
| `DefectViewWindow.cpp:1472-1515` | 污损检测 | 新建 `core/detect_dirt.cpp` |
| `DefectViewWindow.cpp:1527-1570` | 局部缺墨 30×30 网格 | 新建 `core/detect_localmiss.cpp` |
| `DefectViewWindow.cpp:1579-1637` | 多模板判定 | `core/detect_multitmpl.cpp` |
| `DefectViewWindow.cpp:1666-1673` | NG 判定 | `core/verdict.cpp`（统一的判定入口） |
| `DefectViewWindow.cpp:2243-2299` | 升降档切点重算 | `core/verdict.cpp` |
| `batchcore.hpp:140-215` | 与 GUI 重复的配准 | 删除，改调 `core/char_alignment.hpp` |

### 7.2 新增文件

```
src/core/stage/
├── stage_types.hpp       # StageStatus / StageOutcome / IStage / 各阶段入参
├── locate_stage.cpp      # 定位：产 *_locate.csv
├── split_stage.cpp       # 分割：产 *_split.csv
├── detect_stage.cpp      # 检测：产 *_detect.csv
└── verdict.cpp           # 统一判定（OK/NG/NOT_CHECKED）

src/core/csv/
├── csv_writer.hpp/.cpp   # 写 CSV（转义、UTF-8、注释头）
└── csv_reader.hpp/.cpp   # 读 CSV（跳过注释、反转义）

src/gui/
├── csvmodel.hpp/.cpp     # 把 CSV 读成 Qt 模型，供 TableView
└── visualizer.hpp/.cpp   # 按 CSV 行画框（纯几何，不计算）
```

### 7.3 旧 CSV 处置

**不迁移、不兼容。**

| 旧文件 | 处置 |
|---|---|
| `batch_result.csv`（12 列，`DefectViewWindow.cpp:1890`） | 保留但标废弃，新代码不读 |
| `batch_result.csv`（14 列，`defectview_main.cpp:178`） | 同上 |
| `out/review/NG.csv`（UTF-16LE） | 同上 |
| `out/*.csv` 下的实验产物 | 不动，属于历史记录 |

新格式用新文件名（`*_locate.csv` / `*_split.csv` / `*_detect.csv` / `_index.csv`），与旧的 `batch_result.csv` 不冲突。

### 7.4 迁移顺序

按依赖关系分四步，每步都能独立验证：

```
步骤 1  写 csv_writer / csv_reader + 单元测试
        ↓ 验证：写一个含中文、逗号、引号的字段，读回来必须逐字节一致
步骤 2  定义 stage_types.hpp 的几何类型（LocatedGeometry / CharGeometry）
        ↓ 验证：编译错误证明分割阶段拿不到 ncc
步骤 3  三个阶段各自产出 CSV，先用**现有算法**填数据，不改判据
        ↓ 验证：同一张图，新 CSV 里的分数与旧 GUI 显示的数字一致
步骤 4  GUI 改成只读 CSV
        ↓ 验证：把 CSV 里的 score 手工改一个数，GUI 显示跟着变
```

**步骤 3 是本规范的关键验收点**：不改算法、只改数据通路，如果数字对不上，说明现有算法里有隐藏的状态依赖（例如 `lastAlignX_` 这种未清理的成员），这正是要暴露出来的问题。

---

## 八、验收标准

| 编号 | 标准 | 验证方法 |
|---|---|---|
| V1 | 三份 CSV 都能用 Excel 直接打开，中文不乱码 | 手动打开检查 |
| V2 | 含 `,` `"` 换行的 `detail` 字段能正确往返 | 单元测试 |
| V3 | 分割阶段代码里出现 `ncc` 即编译失败 | 故意加一行验证 |
| V4 | CSV 里的分数与当前 GUI 显示的一致 | 抽 10 张逐项比对 |
| V5 | 改 CSV 里的 `score`，GUI 显示跟着变 | 手工改一行 |
| V6 | 尺寸不符的图 `verdict=NOT_CHECKED`，不是 `OK` | 喂一张 1000×1000 的图 |
| V7 | 16 位 TIFF 与 8 位 PNG 检测结果一致 | 同一图两种格式各跑一次 |
| V8 | `src/gui/` 下无 `#include "chardet` 或 `"core/` | CI grep |
| V9 | 现有回归不退化：`image/ok` 误报 0、`image/ng` 漏检 1、`diagnose_0109` 4/4 | 跑既有脚本 |

---

## 九、术语表

| 术语 | 含义 |
|---|---|
| **对象** | CSV 里一行所代表的东西：候选、字符框、或「对象 × 检测器」 |
| **阶段** | 定位 / 分割 / 检测三者之一 |
| **frame** | 该行几何数据所属的坐标系 |
| **triggered** | 该检测器判定不合格（已由算法写入 CSV） |
| **NOT_CHECKED** | 没验过。与 OK 严格区分 |
| **几何字段** | 坐标、尺寸、角度。**不含**分数、置信度、状态 |
| **质量字段** | 分数、置信度、档位标签、状态。**禁止跨阶段传递** |

---

## 附录：本规范否决过的提案

记录在案，避免以后重复讨论。

| 提案 | 否决理由 |
|---|---|
| `image_id` 列做跨阶段关联 | 每阶段独立文件 + `image_file` 已能定位，加 id 是冗余 |
| `image_sha1` 指纹 | `image_file` + `image_w/h` 已足以判断批次；sha1 只防「同路径换内容」这一种边缘情况 |
| 存绝对路径 | 换机器即失效 |
| 统一缩放到模型尺寸 | 重采样会改变笔画粗细与纹理，现有阈值全在原始分辨率下标的 |
| 掩码/像素写进 CSV | CSV 退化为图像容器，Excel 打不开、diff 比不了 |
| GUI 允许拖阈值重算 | 会让 GUI 重新变成第二套判定标准（当前单张页与批量页结论不一致就是这么来的） |
| 兼容旧的 12 列/14 列/UTF-16 格式 | 没有一列能对应新语义，强行兼容让新代码背历史包袱 |
