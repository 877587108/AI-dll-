# 字符检测模块 — 实现说明（移交文档）

> 本文档面向后续开发智能体。描述 `modules/chardet/` 模块的完整实现方法，
> 包含：字符切片、字符检测、用户自选 ROI、参数保存与训练复用。
>
> **当前状态**：代码可编译、可独立构建，但**阈值未标定，GUI 交互未实际测试**。
> 移交方需自行验证。

---

## 一、模块边界

### 职责

| 能力 | 入口函数 |
|---|---|
| 训练（统计频率图） | `chardet::train()` |
| 字符切片 | `chardet::split()` |
| 字符检测（逐字比对） | `chardet::match()` |
| 模型存取（含区域与参数） | `saveModel()` / `loadModel()` |
| 取模板与比对区域 | `regionTemplate()` / `charTemplates()` / `toUnits()` |
| 笔画信号预处理 | `chardet::strokeSignal()` |

### 依赖

仅 OpenCV + 标准库。**不依赖宿主项目任何头文件**。

### 文件

```
modules/chardet/
├── chardet.hpp      公开接口（153 行）
├── config.hpp       全部可调参数（34 行）
├── split.cpp        垂直投影切分（~170 行）
├── match.cpp        逐字符比对（~125 行）
└── model.cpp        训练 + 模型存取 + 模板提取（~350 行）
```

---

## 二、数据结构

```cpp
// 单个字符
struct CharBox {
    cv::Rect bbox;     // 主体框
    cv::Rect extend;   // 扩展比对框（比 bbox 大，用于捕捉"笔画长出去"）
    int index = 0;     // 从左往右序号
};

// 一个选区及其切出的所有字符
struct RegionInfo {
    int index = 0;
    cv::Rect rect;                 // 选区范围（整图绝对坐标）
    std::vector<CharBox> chars;
};

// 完整模型（内存形式）
struct Model {
    int version = 1;
    int view = -1;
    cv::Size imageSize;
    Config config;                 // 训练时用的参数

    cv::Mat freq;                  // 频率图 CV_8U（0~255 表示 0.0~1.0）
    cv::Mat ref;                   // 配准基准 CV_8U

    std::vector<RegionInfo> regions;

    bool valid() const { return !freq.empty() && !regions.empty(); }
    int totalChars() const;
};
```

---

## 三、字符切片实现

### 文件：`split.cpp`

```cpp
std::vector<CharUnit> split(const cv::Mat& freq8, const Config& cfg);
```

### 算法步骤

```
① 二值化
   bin = freq8 > 255 × binFreq(0.25)

② 去噪
   morphologyEx(bin, MORPH_OPEN, 3×3 矩形核)

③ 垂直投影
   colInk[x] = 该列墨像素数（每列求和 / 255）
   maxCol = 投影峰值

④ 列阈值分段
   colThr = max(1, maxCol × colThrRatio)
   colInk[x] >= colThr 的连续区间 → 候选段

⑤ 迭代合并过窄间隙
   for iter in 0..7:
       medianW = 所有段宽的中位数
       gapThr = max(2, medianW × gapMergeRatio)
       相邻段间隙 < gapThr → 合并
       若本轮无变化 → break

⑥ 每段求真实上下边界
   只在该段列范围内做行投影
   y0 = 首个非零行，y1 = 最后非零行
   h = y1 - y0 + 1
   if (h < freq8.rows × minHeightRatio) continue
   if (墨量 < minH × 2) continue

⑦ 外扩
   上下各 6px，左右各 2px
```

### 关键参数

| 参数 | 默认 | 作用 | 调大效果 |
|---|---|---|---|
| `colThrRatio` | **0.03** | 列投影阈值 | 少粘连，但细笔画（`1`/`L`）易被截断 |
| `gapMergeRatio` | 0.10 | 间隙合并比例 | 更多字被合并成一个 |
| `binFreq` | 0.25 | 频率二值化门槛 | 切分更严格 |
| `minHeightRatio` | 0.08 | 最小字符高度 | 过滤更多矮噪点 |

### 为什么用垂直投影而不是连通域

`J` 的钩、`F` 的横杠内部笔画不连通。连通域分析会把一个字符拆成多块，合并后框会跑偏。
投影只看"某一列有没有墨"，与笔画是否连通无关。

### 为什么 colThrRatio 要设 3% 这么低

15% 时实测把 `1`、`L` 的细竖笔截断，一个字符被切成碎块。
3% 时 line1 切出 9 个、line2 切出 7 个，宽度分布均匀（31~89px）。

---

## 四、字符检测实现

### 文件：`match.cpp`

```cpp
std::vector<CharMatch> match(const cv::Mat& aligned,
                             const std::vector<CharUnit>& units,
                             const Config& cfg);
```

### 输入要求

- `aligned`：当前图笔画信号，**CV_32F**，与模板同尺寸、已配准
- 每个 `CharUnit.freq`：该字符的模板频率图（同尺寸稀疏图）

### 比对三类异常

#### ① 少字 / 字少了一部分

```
mustInk  = freq >= missFreq(0.50)        ← 必然有墨的位置
gone     = S < missStrokeLo(0.15)        ← 现在没墨
missPx   = mustInk AND gone

inkN      = countNonZero(mustInk)
missN     = countNonZero(missPx)
missRatio = missN / inkN

if (missRatio > charMissRatio)
    missScore = (missRatio - charMissRatio) / charT
```

#### ② 多字 / 字多了一部分（如 `0→Q`、`S→6`）

在**扩展框 `extend`** 内找强墨：

```
strong  = S > strokeHi(0.35)
lowFreq = freq < 0.20                     ← 正常样本此处是空白
extraPx = strong AND lowFreq
extraRatio = count(extraPx) / inkN

if (extraRatio > charExtraRatio)
    extraScore = (extraRatio - charExtraRatio) / charT
```

**扩展框的意义**：`S→6` 的多余笔画可能伸到字符框外，只看 `bbox` 看不见。

实现要点：构造 `freqExt` 时，**主体框外区域频率强制置 0**（正常样本那里是空白）：

```cpp
cv::Mat freqExt = cv::Mat::zeros(rext.size(), CV_32F);
for (y, x in rext) {
    if (点在主体框内) freqExt[y][x] = u.freq[gy][gx] / 255.0;
    else              freqExt[y][x] = 0.0f;   // 关键
}
```

#### ③ 字形变化

```
应有笔画质心 (cx, cy)：freq > 0.3 处按 freq 加权
实际笔画质心 (ax, ay)：S > strokeHi 处按 S 加权

dist = sqrt((cx-ax)² + (cy-ay)²)
rel  = dist / 字符高度

if (rel > 0.08)
    shapeScore = (rel - 0.08) × 4.0
```

能抓"该有墨的没了 + 该空的有了"同时发生的复杂变形。

#### 最终

```
score = max(missScore, extraScore, shapeScore)
```

`label` 字段生成可读说明（"缺墨 23%"、"多墨 15%"、"字形偏移"）。

### 字符数量校验（可选）

早期实现过 `checkCount()`（检测整字多/少），但**因算法缺陷已从接口移除**：

- 原做法：重新切分一次再数数
- 缺陷：二值化阈值与建模时不一致 → 所有 OK 样本数量都对不上 → 全部误报

若要重做，必须保证切分时的输入与建模时一致。

---

## 五、用户自选 ROI

### 设计原则

**不预设任何区域**。曾经尝试用行投影自动找文字行，但整幅图包含管脚、治具、芯片边缘的强反光，
其投影强度远超笔画，检出的是背景而非文字。

所以：**区域完全由用户在界面上框选**。

### 数据流

```
用户拖拽画框
    ↓
RegionInfo ri; ri.rect = 框的整图坐标
    ↓
立即在该 rect 内调用 split()
    ↓
chars 坐标平移到整图坐标系
    ↓
加入 model.regions
    ↓
用户点保存 → saveModel() 写入 regions.txt
```

### GUI 侧要点（`src/gui/CharBoxEditor.cpp`）

画布显示的是**模型频率图**（`freq.png`），不是产品原图——频率图是平均结果，没有噪声和单张对位偏差。

画完框立即切分的关键代码：

```cpp
void RegionEditor::onBoxCreated(const QRectF& r) {
    chardet::RegionInfo ri;
    ri.index = model_.regions.size();
    ri.rect = cv::Rect(rel.x(), rel.y(), rel.width(), rel.height());

    // 立即切分
    chardet::Config tuning = model_.config;
    tuning.colThrRatio = colThrSpin_->value() / 100.0;
    tuning.gapMergeRatio = gapRatioSpin_->value() / 100.0;

    const cv::Mat sub = model_.freq(ri.rect);
    const auto units = chardet::split(sub, tuning);

    for (const auto& u : units) {
        chardet::CharBox cb;
        cb.bbox = u.bbox + cv::Point(ri.rect.x, ri.rect.y);   // 平移回整图
        cb.extend = u.extend + cv::Point(ri.rect.x, ri.rect.y);
        ri.chars.push_back(cb);
    }
    model_.regions.push_back(std::move(ri));
    refreshCanvas();
}
```

---

## 六、参数保存与训练复用

### 模型目录结构

```
<modelDir>/
├── model.json      元信息：版本、视角、图像尺寸、全部 Config 参数
├── freq.png        频率图 CV_8U
├── ref.png         配准基准
└── regions.txt     区域与字符框
```

### model.json 格式

```json
{
  "module": "chardet",
  "version": 1,
  "view": 0,
  "imageWidth": 1688,
  "imageHeight": 1460,
  "regionCount": 2,
  "charCount": 16,
  "config": {
    "sigma": 15,
    "strokeHi": 0.35,
    "missStrokeLo": 0.15,
    "missFreq": 0.5,
    "charMissRatio": 0.15,
    "charExtraRatio": 0.12,
    "charT": 1,
    "colThrRatio": 0.03,
    "gapMergeRatio": 0.1,
    "binFreq": 0.25
  }
}
```

**解析方式**：`loadModel()` 用手写的宽松匹配（`line.find("\"key\"")` + `atof`），
不引入第三方 JSON 库。若要加复杂字段，建议换成正规 JSON 库。

### regions.txt 格式

```
<区域数>
<每区域一行：index x y w h charCount>
<每字符一行：x y w h extX extY extW extH index>
```

示例（2 个区域）：
```
2
0 395 790 880 215 9
415 806 88 131 411 802 96 139 0
514 806 77 133 510 802 85 141 1
...
1 480 985 720 200 7
529 964 74 133 525 960 82 141 0
...
```

### 参数复用（训练时继承）

用户的典型需求：调好参数后重新训练，不想丢失阈值。

**当前实现**：`Config` 全部存在 `model.json` 里，`loadModel()` 会读回。
但 `train()` **不使用传入 Model 的 config**，只用调用方传的 `Config`。

**若要实现"继承上次参数"**，调用方需这样写：

```cpp
chardet::Model oldModel;
chardet::Config cfg;           // 默认值
if (chardet::loadModel(oldDir, oldModel)) {
    cfg = oldModel.config;     // 继承上次的参数
}
cfg.colThrRatio = 0.05;        // 再覆盖个别项
chardet::Model newModel = chardet::train(input, cfg, &report);
```

---

## 七、训练实现

### 文件：`model.cpp`

```cpp
Model train(const TrainInput& in, const Config& cfg, TrainReport* report);
```

### 步骤

```
① 过滤尺寸不一致的图
   以第一张为基准尺寸，其余尺寸不同的计入 skipped

② 基准信号
   sRef = computeSignal(imgs[0])

③ 累加频率
   for k in 1..N:
       s = computeSignal(imgs[k])
       # 相位相关对齐
       shift = phaseCorrelate(s, sRef)      ← 注意：输入必须是 CV_32F
       if (|shift| < 30px) warpAffine 应用
       mask = s > strokeHi
       count += mask / 255
       used++

④ freq = count / used
   freq.convertTo(m.freq, CV_8U, 255.0)

⑤ 不预设区域（见第五节）
```

### 内部信号函数

```cpp
cv::Mat computeSignal(const cv::Mat& patch, const Config& cfg, bool invert) {
    // 原图 → 浮点
    cv::GaussianBlur(p, bg, Size(0,0), cfg.sigma);
    cv::subtract(p, bg, d);
    if (invert) cv::multiply(d, -1.0, d);

    // 94 分位做归一化刻度
    hist[512]，遍历 d，bin = lround(d[x]) + 256，clamp[0,511]
    q94 = 累计到 total×0.94 时的 bin - 256
    scale = max(8.0, q94)

    s = d / scale
    cv::threshold(s, s, 0, 0, THRESH_TOZERO);   // 丢弃负残差
    return s;
}
```

**注意**：`invert` 目前固定传 `false`。若某视角是暗刻印，需要外部判定后传入。

---

## 八、消费端用法（供宿主项目集成）

```cpp
// 1. 加载模型
chardet::Model m;
if (!chardet::loadModel(modelDir, m)) return;

// 2. 计算当前图笔画信号
cv::Mat S = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
// 注意：需要先做配准（与 m.ref 对齐），本模块不管配准

// 3. 逐区域比对
for (int ri = 0; ri < m.regions.size(); ++ri) {
    auto units = chardet::toUnits(m, ri);
    auto matches = chardet::match(S, units, m.config);

    for (const auto& mc : matches) {
        if (mc.score > 0) {
            printf("字符%d (%d,%d) 分数 %.2f  %s\n",
                   mc.index, mc.bbox.x, mc.bbox.y, mc.score, mc.label.c_str());
        }
    }
}
```

### 取模板的三种方式

```cpp
// 整个区域的模板（截取 freq 对应范围）
cv::Mat tpl = chardet::regionTemplate(m, regionIndex);

// 每个字符的模板（稀疏图，只在各自 bbox 内有值）
std::vector<cv::Mat> tpls = chardet::charTemplates(m, regionIndex);

// 直接转成可比对的单元（最常用）
std::vector<chardet::CharUnit> units = chardet::toUnits(m, regionIndex);
```

---

## 九、已知问题与改进方向

### 必须修复

| 问题 | 位置 | 说明 |
|---|---|---|
| **配准未内置** | 接口设计 | `match()` 要求传入已配准信号，但模块不提供配准。宿主必须自己用 ECC 对齐到 `m.ref` |
| **`invert` 未自动判定** | `train()` | 固定 `false`，暗刻印视角需要外部判定 |
| **阈值未标定** | `Config` | 全部为经验值，实测打开字符检测后误报严重 |

### 建议改进

1. **内置配准**：提供 `alignTo(model, gray)` 返回配准后的信号
2. **极性自动判定**：比较残差正负尾部极值，决定 `invert`
3. **数量校验重做**：用与建模一致的输入重新切分，或改用"框内是否有墨"判据
4. **JSON 库**：手写解析在字段增多后易出错
5. **切分参数自适应**：不同字体、不同字距需要不同 `colThrRatio`，可考虑用投影曲线的谷值统计自动确定

### 验证状态

- ✅ 模块可独立构建（`cmake -S modules/chardet -B build`）
- ✅ `split()` 在 v00 上切出 line1=9、line2=7，宽度分布合理
- ❌ `match()` 未在真实数据上验证有效性
- ❌ GUI 的画框、删除、保存交互从未实际操作

---

## 十、快速上手

```bash
# 独立构建模块
cmake -S modules/chardet -B modules/chardet/build -DOpenCV_ROOT=D:/opencv
cmake --build modules/chardet/build --config Release
```

```cpp
// 最小示例
#include "chardet.hpp"

chardet::TrainInput in;
in.images = loadYourOkImages();   // std::vector<cv::Mat>，CV_8U
in.view = 0;
chardet::Config cfg;
chardet::TrainReport rep;
chardet::Model m = chardet::train(in, cfg, &rep);

// 手动加一个区域并切分
chardet::RegionInfo ri;
ri.rect = cv::Rect(x, y, w, h);   // 用户画的框
ri.index = 0;
auto units = chardet::split(m.freq(ri.rect), cfg);
for (auto& u : units) {
    chardet::CharBox cb;
    cb.bbox = u.bbox + cv::Point(ri.rect.x, ri.rect.y);
    cb.extend = u.extend + cv::Point(ri.rect.x, ri.rect.y);
    ri.chars.push_back(cb);
}
m.regions.push_back(ri);

chardet::saveModel("out/model", m);
```
