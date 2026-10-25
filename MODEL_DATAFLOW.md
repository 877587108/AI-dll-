# 模型加载 → 识别 数据流

> 只覆盖「模型从磁盘读入内存」到「产出判定结果」这一段。
> 不含训练侧（buildModel 写模型），不含 GUI 评估的统计逻辑。
> 代码位置：`src/core/char_core.cpp` 的 `Model::loadFrom()` 与 `analyze()`。

---

## 一、模型在磁盘上的形态

```
models/char/
├── v00/
│   ├── logo _ref.png      参考笔画信号    ← 配准基准
│   ├── logo _freq.png     正常笔画频率    ← 加载但分析时不直接用
│   ├── logo _tol.png      容差掩膜        ← 决定"哪些地方不报"
│   ├── line1_ref/freq/tol.png
│   ├── line2_ref/freq/tol.png
│   ├── code _ref/freq/tol.png
│   ├── code_stats.txt     inkMean inkStd thickMean thickStd
│   ├── surf_ref.png       全芯片表面参考（可选）
│   ├── surf_tol.png       全芯片表面容差（可选）
│   └── params.txt         17+ 个标量 + 4 行 ROI
└── v01/ … v05/  同上
```

**关键区分**：`_ref`、`_freq`、`_tol` 三个文件里，**识别过程只用 `_ref` 和 `_tol`**。
`_freq` 会被读进内存（`ModelRoi::freq`），但 `analyze()` 不引用它——容差判定已经在建模时固化进 `_tol` 了。

---

## 二、加载：磁盘 → 内存结构

```mermaid
flowchart TD
    A["Model::loadFrom(dir)"] --> B["遍历 v = 0..5"]
    B --> C["拼路径 dir/vNN/"]

    C --> D["遍历 i = 0..3<br/>logo / line1 / line2 / code"]
    D --> E["cv::imread 三个 PNG<br/>_ref.png / _freq.png / _tol.png"]
    E --> F{"三个都非空?"}
    F -->|否| FX["该视角 ok = false<br/>跳出，试下一个视角"]
    F -->|是| G["存入 ViewModel.rois[i]<br/>sRef / freq / tolMask<br/>rois[i].ok = true"]
    G --> D

    D --> H["读 code_stats.txt<br/>4 个 double"]
    H --> I{"读到 4 个值?"}
    I -->|是| I1["codeStat.valid = true"]
    I -->|否| I2["codeStat.valid = false<br/>→ 笔画粗细分支将返回 0"]
    I1 --> J
    I2 --> J["读 surf_ref.png / surf_tol.png"]
    J --> K{"两者非空且尺寸相同?"}
    K -->|是| K1["surfOk = true"]
    K -->|否| K2["surfOk = false<br/>→ 表面分支不执行"]
    K1 --> L
    K2 --> L["loadParams(dir, v, params[v])<br/>读 params.txt"]
    L --> M["vm.ok = true<br/>any = true"]
    M --> B

    B --> N{"any ?"}
    N -->|否| NX["返回 false<br/>『没有有效的视角子目录』"]
    N -->|是| NY["返回 true<br/>Model 就绪"]
```

### 加载后的内存结构

```mermaid
classDiagram
    class Model {
        ViewModel views[6]
        Params params[6]
    }
    class ViewModel {
        ModelRoi rois[4]
        CodeStat codeStat
        Mat surfRef
        Mat surfTol
        bool surfOk
        bool ok
    }
    class ModelRoi {
        Mat sRef     "参考笔画信号 CV_8U"
        Mat freq     "频率图（加载但分析不用）"
        Mat tolMask  "容差掩膜 CV_8U 0|255"
        bool ok
    }
    class CodeStat {
        bool valid
        double inkMean, inkStd
        double thickMean, thickStd
    }
    class Params {
        double sigma, strokeHi, strokeLo
        double scratchT, appendageT
        double eccMin, codeEccMin
        double scratchMinCoverage, scratchMinLen
        double codeZNg
        int edgeTB, edgeLR, dilateK
        bool invert, checkCode, enabled
        Roi rois[4]
    }
    class Roi {
        Rect rect
        bool checkAppendage
        bool enabled
    }

    Model --> ViewModel : views[6]
    Model --> Params : params[6]
    ViewModel --> ModelRoi : rois[4]
    ViewModel --> CodeStat
    Params --> Roi : rois[4]
```

**注意**：`Params` 与 `ViewModel` 是**两个平行数组**，都在 `Model` 里但互不引用。
`analyze()` 把两者同时传入，参数决定"怎么算"，模板决定"和谁比"。

---

## 三、识别：模型数据如何参与计算

这是数据流的核心。模型的三类数据分别进入不同的处理环节。

```mermaid
flowchart TD
    M1["Model.views[v].rois[i].sRef<br/>CV_8U 基准信号"] --> S2
    M2["Model.views[v].rois[i].tolMask<br/>CV_8U 容差掩膜"] --> S4
    M3["Model.views[v].codeStat<br/>4 个统计量"] --> S5
    M4["Model.params[v]<br/>全部阈值与 ROI"] --> S1
    M4 --> S3
    M4 --> S4

    G1["输入图像 gray<br/>CV_8U 全图"] --> S1["① strokeSignal()<br/>按 p.sigma 去背景归一化"]
    S1 --> S1O["sImg：CV_32F 笔画信号"]

    S1O --> S2["② ECC 配准<br/>estimateWarp(sImg, sRef÷80)"]
    S2 --> S2O["aligned：CV_32F<br/>对齐到模型坐标系"]

    S2O --> S3["③ 双重掩膜交集<br/>S>阈值 ∧ NOT tolMask ∧ usable(edgeTB/edgeLR)"]
    S3 --> S3O["extra / weak<br/>CV_8U 0|255 残差掩膜"]

    S3O --> B1["分支A/B：划痕与多余笔画<br/>只用 aligned + extra/weak"]
    B1 --> SC["区域分 = 归一化<br/>÷ scratchT 或 appendageT"]

    S1O --> S5["分支C：笔画粗细<br/>measureCode(sImg) vs codeStat"]
    S5 --> SC

    SC --> GATE{"ecc >= eccMin<br/>或 codeEccMin"}
    GATE -->|否| Z["区域分 = 0"]
    GATE -->|是| KEEP["保留"]
    Z --> OUT
    KEEP --> OUT["Result.regions[i]<br/>Result.score = 四区域最高"]
```

### 模型数据的三条去向

| 模型数据 | 被谁使用 | 用途 |
|---|---|---|
| `rois[i].sRef` | `estimateWarp` | **仅作 ECC 配准基准**，不参与像素比对 |
| `rois[i].tolMask` | `allowed = NOT tolMask` | 决定残差掩膜里哪些像素被排除 |
| `rois[i].freq` | **无** | 加载但识别时不使用 |
| `codeStat` | `scoreCode` | 提供正常笔画粗细的均值/标准差，用于 z-score |
| `params[v]` | 全流程 | 所有阈值、ROI 坐标、门控门槛 |

**一个容易误解的点**：`sRef` 不是"模板匹配"意义上的模板。
识别过程**从不把当前图像和 `sRef` 做像素差**。`sRef` 只用来算一个欧氏变换矩阵，
让当前图像对齐到建模时的坐标系。真正的"比对"全部发生在 `tolMask` 上。

---

## 四、逐区域的数据形态变化

以 logo 区（700×235）为例，走一遍数据尺寸与类型的变化：

```mermaid
flowchart LR
    A["gray<br/>全图 CV_8U"] -->|"roi.rect 裁剪"| B["patch<br/>700×235 CV_32F"]
    B -->|"GaussianBlur + 减 + 归一化"| C["sImg<br/>700×235 CV_32F<br/>值域 [0, +∞)"]
    C -->|"×80 量化"| C2["sRef 加载回来<br/>700×235 CV_8U"]
    C2 -->|"÷80 还原"| C3["refF<br/>700×235 CV_32F"]

    C -->|"输入"| D["estimateWarp<br/>半分辨率 350×117"]
    C3 -->|"输入"| D
    D -->|"2×3 变换矩阵 + ecc 标量"| E["aligned<br/>700×235 CV_32F"]

    E -->|"S > strokeHi"| F["强残差 bool"]
    E -->|"S > strokeLo"| G["弱残差 bool"]
    F -->|"∧ allowed ∧ usable"| H["extra<br/>700×235 CV_8U 0|255"]
    G -->|"∧ allowed ∧ usable"| I["weak<br/>700×235 CV_8U 0|255"]

    H -->|"连通域 + PCA"| J["appendage 标量<br/>面积×平均强度"]
    I -->|"Hough + 密度 + 长度"| K["scratchSevere 标量<br/>能量×长度"]

    J -->|"÷ appendageT"| L["logo 区域分"]
    K -->|"÷ scratchT"| L
    L -->|"取 max"| M["Result.score<br/>CV 无关，纯 double"]
```

---

## 五、门控：模型质量如何影响结果

`ecc` 是唯一一个"由模型与图像的匹配质量决定"的量，它反过来否决分数。

```mermaid
flowchart TD
    A["sRef 与 sImg 尺寸相同?"] -->|否| AX["estimateWarp 返回 false<br/>aligned = 原 sImg<br/>ecc = -1"]
    A -->|是| B["缩到半分辨率<br/>m = identity 2×3"]
    B --> C["cv::findTransformECC<br/>MOTION_EUCLIDEAN<br/>迭代 100 次 / eps 1e-5"]
    C --> D{"抛异常?"}
    D -->|是| DX["ecc = -1"]
    D -->|否| E["ecc = 返回值<br/>平移量 ×2 还原到全分辨率"]
    E --> F{"ecc >= 0 ?"}
    F -->|否| FX["返回 false"]
    F -->|是| G["返回 true + 变换矩阵"]

    AX --> H["rr.ecc = -1"]
    DX --> H
    FX --> H
    G --> I["rr.ecc = ecc 实际值"]

    H --> J{"i == 3 ?<br/>code 区"}
    I --> J
    J -->|是| K["门槛 = codeEccMin (0.55)"]
    J -->|否| L["门槛 = eccMin (0.60)"]
    K --> N{"ecc >= 门槛 ?"}
    L --> N
    N -->|否| O["rr.reliable = false<br/>regionScore = 0"]
    N -->|是| P["rr.reliable = true<br/>保留分数"]
```

**设计意图**：`ecc` 低于门槛意味着图像和模型根本没对齐，此时的残差是错位导致的伪差，
不是真实缺陷。code 区门槛更低是因为内容逐颗不同，配准天然偏差。

---

## 六、一次完整调用的数据流（端到端）

```mermaid
sequenceDiagram
    participant GUI as TrainWindow
    participant LD as Model::loadFrom
    participant AN as analyze
    participant SS as strokeSignal
    participant EW as estimateWarp
    participant DT as detect_*

    GUI->>LD: loadFrom("models/char")
    LD->>LD: 读 6×4×3 个 PNG + code_stats + params
    LD-->>GUI: Model (views[6], params[6])

    loop 每张图、每个视角
        GUI->>AN: analyze(gray, views[v], params[v])
        AN->>SS: strokeSignal(gray, roi)
        SS-->>AN: sImg (CV_32F)
        AN->>EW: estimateWarp(sImg, sRef/80)
        EW-->>AN: warp + ecc
        AN->>AN: aligned = applyWarp(sImg, warp)
        AN->>AN: usable ∧ allowed 掩膜
        AN->>DT: scoreAppendage / scoreScratch / scoreCode
        DT-->>AN: 各分支原始分
        AN->>AN: 归一化 ÷ (scratchT|appendageT)
        AN->>AN: ECC 门控否决
        AN-->>GUI: Result (regions[4], score, ng)
    end

    GUI->>GUI: 按料号聚合取最高分
    GUI->>GUI: score >= 1.0 ⇒ NG
```

---

## 七、模型参数与识别行为的对应关系

| 模型里存的 | 影响识别的哪里 | 调大后的效果 |
|---|---|---|
| `sRef` 像素内容 | ECC 配准质量 | — （重新建模才能改） |
| `tolMask` 白色区域 | `allowed` 掩膜 | 白色越多 → 越多区域不报 → 漏检风险↑ |
| `codeStat.thickMean/Std` |  z-score 分母 | Std 越大 → z 越小 → 越不敏感 |
| `sigma` | `GaussianBlur` 尺度 | 只保留粗结构 → 细节缺陷丢失 |
| `strokeHi` | 强残差阈值 | 少报多余笔画 |
| `strokeLo` | 弱残差阈值 | 少报划痕 |
| `edgeTB/edgeLR` | `usable` 边缘屏蔽 | 边缘区域不参与 → 边界伪线减少 |
| `eccMin/codeEccMin` | 门控门槛 | 更多区域被判不可靠 → 分数归零 |
| `scratchMinLen` | 划痕长度门槛 | 短划痕放行 |
| `scratchMinCoverage` | 划痕密度门槛 | 噪点连线被过滤 |
| `scratchT` | 划痕分分母 | 分数整体变小 → 更宽松 |
| `appendageT` | 多余笔画分母 | 同上 |
| `rois[i].enabled` | 是否参与 | 关闭 = 该区域完全不检测 |
| `p.invert` | 信号极性 | 适配暗刻印视角 |

---

## 八、三个"数据断点"（需要特别注意）

数据流里有三处会静默中断，不会报错：

1. **`_freq.png` 读了不用**
   `ModelRoi::freq` 加载到内存后，`analyze()` 从不引用。它是建模时的中间产物，
   容差已经固化进 `_tol.png`。如果手工改 `_freq` 而不同步改 `_tol`，不会生效。

2. **`codeStat.valid = false` 时分支返回 0**
   `code_stats.txt` 缺失/损坏时，`scoreCode()` 直接 `return 0.0`，
   **不报警、不标记不可靠**。表面上一切正常，实际笔画粗细检测已失效。

3. **`surfOk = false` 时表面分支静默跳过**
   `surf_ref.png` / `surf_tol.png` 缺失或尺寸不一致时 `useSurface` 形同虚设。
   当前所有视角 `useSurface=0`，所以这条暂时无影响。

这三条都记录在 `IMPROVEMENTS.md`（P0-4 等）。
