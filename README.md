# markloc — 芯片刻印缺陷检测

对芯片表面的激光刻印（logo / 型号行 / 批号行）做缺陷检测：
识别 **断线漏印、多余笔画、连续划痕、污损划痕** 四类异常，
输出 OK / NG 判定与可视化证据。

---

## 1. 快速开始

```powershell
# 1) 配置与编译（需要 D:\opencv 与 D:\Program\QtDev6.7.3）
cmake --preset release
cmake --build build --config Release --parallel

# 2) 启动图形界面（批量训练）
.\build\Release\markloc_gui.exe

# 3) 无界面训练（同样的训练逻辑，结果打到 stdout）
.\build\Release\markloc_gui.exe --train image\ok models\char
.\build\Release\markloc_gui.exe --train image\ok models\char --no-eval

# 4) 不依赖 Qt 的全量评估（推荐用于回归）
.\build\Release\markloc_eval.exe image models\char
.\build\Release\markloc_eval.exe image models\char --view 3 --disable-roi logo
.\build\Release\markloc_eval.exe image models\char --view 1 --set edgeTB=40
```

### 图形界面：批量训练

界面已重写为**选文件夹 → 一键训练 → 自动评估**，不再有单图/参数面板。

| 控件 | 作用 |
|---|---|
| **图片文件夹** | 递归扫描其中的 `*.tif`，按文件名末尾 `_00.._05` 自动分视角 |
| **模型输出目录** | 训练结果写入该目录的 `v00..v05/` |
| **包含子文件夹** | 是否递归（`image/` 下含 `ok/`、`ng/` 时需开启） |
| **每视角上限** | 只用前 N 张建模型，0 = 不限 |
| **训练后自动评估** | 训练完立即在**同一数据集**上做 OK/NG 回归 |

训练流程：

```
扫描图片 → 按 _NN 分视角 → 解码 → buildModel 统计正常笔画频率
        → 自动标定每视角 scratchT → 写入 vNN/ → （可选）评估
```

> **阈值自动标定**是训练的必要一步。`buildModel` 只给默认 `scratchT=22000`，
> 而实测各视角需要 13976 ~ 170565（差 12 倍）。标定做法：用全部训练样本算分，
> 取 98 分位作为「正常样本最高分」，反推 `scratchT` 使其落在 0.85（低于判定阈值 1.0）。
> 不做这步，新模型会对 OK 样本大量误报。

评估结果分三块显示：**逐视角训练报告**、**逐件判定表**（六视角分列）、**日志**。
底部汇总行显示按零件口径的准确率与 TP/FP/FN/TN。

> ⚠️ 评估的真值来源：父目录名 `ok`/`ng`，或文件名含 `_Pass`/`_Failed`。
> 两样都没有时只训练不评估（汇总行会注明“无标签”）。

### 环境依赖

| 组件 | 版本 | 位置 |
|---|---|---|
| Visual Studio | 2022 (MSVC 19.44) | — |
| CMake | ≥ 3.20 | — |
| OpenCV | 4.10.0 (vc16, world) | `D:\opencv` |
| Qt | 6.7.3 | `D:\Program\QtDev6.7.3` |
| C++ 标准 | C++20 | — |

OpenCV 路径可在配置时覆盖：`cmake --preset release -DOpenCV_ROOT=<你的路径>`。

---

## 2. 模块结构

模块边界、公共接口和数据结构规范见 [docs/architecture.md](docs/architecture.md) 与 [docs/data-contracts.md](docs/data-contracts.md)。应用层只应包含 `src/core/api.hpp`；检测算法通过 `IDetectionAlgorithm` 或 `detectors.hpp` 的输入契约接入。

```
src/
├── core/                      分层检测核心库
│   ├── api.hpp                应用层唯一入口
│   ├── algorithm_types.*      稳定数据结构与只读模型句柄
│   ├── detection_algorithm.hpp 可替换算法接口
│   ├── pipeline.*             输入校验与流程编排
│   ├── model_repository.*     图片索引与模型加载
│   ├── analyze.cpp            内置算法流程
│   ├── detectors.hpp          ROI 检测器契约
│   └── detect_*.cpp           独立缺陷分支
├── gui/                       Qt 界面层（两个独立程序）
│   ├── main.cpp               评估程序入口（markloc_gui）
│   ├── TrainWindow.hpp/.cpp   评估窗口：识别 + 性能统计
│   ├── train_main.cpp         模型构建入口（markloc_train）
│   ├── TrainToolWindow.hpp/.cpp  训练窗口
│   ├── CharBoxEditor.hpp/.cpp 选区与划字对话框
│   └── BoxCanvas.hpp/.cpp     可交互画框控件
└── common/
    └── console_utf8.hpp       控制台 UTF-8 输出（Windows CP936 环境）

tools/                         调参辅助工具
├── crop.cpp                   裁切 ROI 预览
├── montage.cpp                多图拼版
├── mosaic.cpp                 模板马赛克
├── eval.cpp                   ★ 全量评估（不依赖 Qt，可复现 GUI 口径）
├── probe.cpp                  ★ 单区域分支诊断（逐线 len/cov/energy）
├── probe_bg.cpp               ★ 背景梯度诊断（归一化尺度是否被背景主导）
├── charbox.cpp                字符框可视化
├── charprobe.cpp              字符级检测诊断
├── proj.cpp                   投影曲线诊断
├── resplit.cpp                重新生成字符框
└── autotune.ps1               自动调参（坐标轮换爬山）

models/char/                   六视角模型（交付物，勿删）
├── v00/ … v05/
│   ├── {logo,line1,line2,code}_ref.png    参考笔画信号
│   ├── {logo,line1,line2,code}_freq.png   正常笔画频率图
│   ├── {logo,line1,line2,code}_tol.png    容差掩膜
│   ├── chars_{N}.txt                      自动切分的字符框
│   ├── chars_manual.txt                   ★ 手动选区与字符框（存在时优先）
│   ├── code_stats.txt                     历史保留文件，无实际语义
│   └── params.txt                         该视角的完整参数
```

---

## 3. 各模块职责

### 3.1 `naming` — 图片命名与样本组织

图片命名约定：

```
{料号4位}_{FT|RT}_{其余描述}_{视角2位}.tif
0112_FT_Failed [2026-1010-114030-154] (字符) _00.tif
│    │                    │                 │
│    │                    │                 └─ 视角 00~05
│    │                    └─ 拍摄批次（FT 第一次 / RT 第二次）
│    └─ 料号
```

| 接口 | 作用 |
|---|---|
| `viewOf(path)` | 解析末尾 `_NN` 得视角号，失败返回 -1 |
| `productOf(path)` | 取料号前 4 位 |
| `prefixOf(path)` | 去掉 `_NN` 得前缀 |
| `shotOf(path)` | 返回 `FT` / `RT` / 空 |
| `siblingPath(any, v)` | 拼出同产品指定视角的路径 |
| `findSiblings(any, present)` | 一次找出六视角，`present` 标记存在性 |
| `collectImages(root)` | 批量收集（优先 `ok` / `ng` 子目录） |

**注意**：Windows 下 `std::filesystem` 用窄字符串会按 ANSI 代码页解析，
中文文件名（如含 `(字符)`）必须走 `fs::u8path`，本模块已统一处理。

### 3.2 `warp` — ECC 配准

同一产品不同拍摄存在微小位移，需先对齐再做差分。

| 接口 | 作用 |
|---|---|
| `detail::estimateWarp(img, ref, warp, &ecc)` | 估计变换矩阵，`ecc` 为配准质量 |
| `detail::applyWarp(img, warp)` | 按变换重采样 |

实现要点：**半分辨率估计 + 全分辨率应用**（速度提升约 4 倍），
平移量需乘 2 还原。`ecc < eccMin` 的区域视为不可靠，该区域不计分。

### 3.3 `detect_appendage` — 多余笔画

用于检测刻印周围多出的笔画（如漏印的补笔、异物）。

- 在容差区外取强残差（`S > strokeHi`）
- 连通域分析，面积 < 15px 忽略
- PCA 求主方向伸长比，**> 2.2 判为笔画状**（区别于圆形污点）
- 得分 = `面积 × 平均强度`，仅 logo 区启用

### 3.4 `detect_scratch` — 连续划痕

用于检测字符区域内贯穿的划痕。

| 步骤 | 判据 | 作用 |
|---|---|---|
| 1 | Hough 直线检测 | 生成候选线段 |
| 2 | **密度 ≥ 0.25** | 抑制噪点连线（±1px 窄带内有残差的采样占比） |
| 3 | **长度 ≥ 160px** | 区分严重缺陷与轻微瑕疵 |
| 4 | 得分 = `能量 × 长度` | 长而淡的真实划痕也能拿到高分 |

**为什么用长度而不是对比度**：实测发现严重缺陷是「长而淡」（225px），
轻微瑕疵是「短而亮」（≤121px）。用长度门槛把 0056、0101 两个误报清零，
同时保留 0083 真缺陷。

### 3.5 `detect_missing` — 缺墨断线

检测「本该有墨却缺失」的情况，与多余笔画判据方向相反、正好互补。

判据：
1. 频率图中 `freq ≥ missFreq`（默认 0.50）的位置 = 必然有墨区
2. 当前笔画信号 `S < missStrokeLo`（默认 0.15）= 墨缺失
3. 两者交集 → 缺墨像素
4. 连通域分析，面积 ≥ `missMinArea`（默认 30px）才算
5. 得分 = `面积 × 平均缺失深度`，再除以 `missT` 归一化

### 3.6 `char_split` / `char_manual` — 字符级比对

适用于内容固定的文本行（型号行、批号行）。

**自动切分**：用垂直投影把文字行切成单字符。相比连通域分析，
投影不受字符内部笔画是否连通的影响（如 `J` 的钩、`F` 的横杠）。

可调参数：

| 参数 | 默认 | 作用 |
|---|---|---|
| `colThrRatio` | 0.15 | 列投影阈值。调低→细笔画（`1`/`L` 的竖）不断；调高→少粘连 |
| `gapMergeRatio` | 0.10 | 间隙合并比例。调大→更多字被合并 |
| `binFreq` | 0.25 | 频率图二值化门槛 |

**手动选区**：用户可在芯片图上框选任意文字区域，程序只在该区域内切分。
配置存于 `chars_manual.txt`，优先级高于自动切分。

逐字符比对三类异常：

| 类型 | 判据 |
|---|---|
| 少字 / 字少一部分 | 框内 `freq` 高但 `S` 低的像素占比 > `charMissRatio` |
| 多字 / 字多一部分 | 扩展框内 `freq` 低但 `S` 高的像素占比 > `charExtraRatio` |
| 字形变化 | 应有笔画质心 vs 实际笔画质心，偏移超过字符高度 8% |

### 3.7 `char_core` — 主流程

```cpp
Result analyze(gray, viewModel, params, wantEvidence);          // 单视角
ProductResult analyzeProduct(images, present, model, params, wantEvidence);  // 产品级
cv::Mat makeExplain(gray, params, result, title);               // 四联对比图
```

`analyze` 内部流程：

```
对 4 个 ROI：
├─ strokeSignal        去背景 I − Gaussian(I, σ)，归一化
├─ ECC 配准            质量不足则跳过该区域
├─ 容差掩膜            正常笔画频率 > 0.20 处膨胀 9×9
├─ 分支A 多余笔画      仅 logo
├─ 分支B 连续划痕      全区域
├─ 分支D 缺墨断线      全区域（可选）
└─ 分支E 字符级比对    仅 line1 / line2（可选）
```

`analyzeProduct` 汇总：产品分 = `max(各视角四区域最高分)`。

### 3.8 `gui` — 界面

界面已于 2026-10 重写为**批量训练器**（`TrainWindow`），旧的主窗口 / 批量窗口 / 参数面板已移除。

**唯一的操作流**：

1. 选「图片文件夹」（递归扫描 tif，按 `_NN` 自动分视角）
2. 选「模型输出目录」
3. 点「开始训练」→ 建模 → 标定 `scratchT` → 写入 `vNN/`
4. 勾选「训练后自动评估」时，紧接着在同一数据集上回归，显示按图/按零件口径

**三块结果区**：

| 区域 | 内容 |
|---|---|
| 逐视角训练报告 | 每视角样本数、模型状态、启用的区域、极性判定 |
| 逐件判定表 | 每零件一行，六视角分数分列，误判标红 |
| 日志 | 扫描/解码/建模/标定/评估全过程，含失败原因 |

**无界面模式**：`--train <图片目录> [模型目录] [--no-eval]`，逻辑完全相同，输出到 stdout，便于自动化回归。

**构建说明**：本项目**不使用 CMake AUTOMOC**。`CMakeLists.txt` 里显式调用 `moc.exe`
生成 `moc_TrainWindow.cpp`（受限环境里 `cmake_autogen` 无法创建子进程）。
刻意把 moc 步骤写在 CMake 里而不是手搓，是为了换机器/换 Qt 版本时仍能复现构建。

---

## 4. 参数说明

每个视角一套独立参数，存于 `models/char/vNN/params.txt`：

```
行1: sigma strokeHi strokeLo pKeep dilateK edgeTB edgeLR scratchT appendageT
     eccMin scratchMinCoverage scratchMinLen (保留) (保留) enabled invert
     checkMissing (保留) missFreq missStrokeLo missMinArea missT
     checkChars charMissRatio charExtraRatio charT
行2~5: logo / line1 / line2 / code 的 ROI（x y w h [enabled]）
```

> 字段按位置解析，缺省字段使用默认值（`ROI enabled=1`）。
> 解析按行进行，缺字段不会导致后续字段错位。
> 标注「保留」的位不再有语义，仅为兼容旧模型文件保留。

| 参数 | 默认 | 含义 |
|---|---|---|
| `sigma` | 15 | 去背景高斯核尺度，越大越只保留粗结构 |
| `strokeHi` | 0.35 | 强笔画阈值（多余笔画用） |
| `strokeLo` | 0.20 | 弱笔画阈值（划痕用） |
| `pKeep` | 0.20 | 正常笔画频率阈值，超过即视为“正常区域” |
| `dilateK` | 9 | 容差掩膜膨胀核 |
| `edgeTB` / `edgeLR` | 25 / 10 | 上下 / 左右边缘屏蔽宽度 |
| `scratchT` | 22000 | 划痕归一化分母（分数 = 能量×长度 / 此值） |
| `appendageT` | 210 | 多余笔画归一化分母 |
| `eccMin` | 0.60 | 配准质量门槛，不足则该区域不计分 |
| `scratchMinCoverage` | 0.25 | 划痕密度下限 |
| `scratchMinLen` | 160 | 严重划痕长度门槛（px） |
| `enabled` | 1 | 是否参与产品判定 |
| `invert` | 0 | **极性**：1 表示该视角刻印比背景暗，需反转信号 |
| `checkMissing` | 0 | 缺墨检测开关 |
| `missFreq` | 0.50 | 必然有墨的频率门槛 |
| `missStrokeLo` | 0.15 | 墨缺失的信号门槛 |
| `missMinArea` | 30 | 缺墨连通域最小面积（px） |
| `missT` | 900 | 缺墨归一化分母 |
| `checkChars` | 0 | 字符级比对开关 |
| `charMissRatio` | 0.15 | 单字缺墨比例门槛 |
| `charExtraRatio` | 0.12 | 单字多余墨比例门槛 |
| `charT` | 1.0 | 字符分归一化分母 |

**ROI 行第 5 字段 `enabled`**：按视角关闭某个检测区域。v01 / v03 / v04 的 logo 行均为 `... 0`，
原因见 8.1。

### 极性（`invert`）说明

不同打光下刻印可能比背景**亮**也可能比背景**暗**。算法只取“比背景亮”的笔画，
若某视角是暗刻印，信号极性相反，ECC 会失败（实测 v01 只有 0.48）。

`buildModel` 会自动判定：比较 ROI 内差分的正/负尾部强度，负向更强则置 `invert=1`。
v01 被自动识别为暗刻印，ECC 从 0.48 升到 0.856。

---

## 5. 模型

```
models/char/
├── v00/ … v05/
│   ├── logo _ref.png     参考笔画信号（第一张 OK 图）
│   ├── logo _freq.png    正常笔画出现频率（全样本统计）
│   ├── logo _tol.png     容差掩膜（频率 > pKeep 处膨胀）
│   ├── … line1 / line2 / code 同上
│   ├── code_stats.txt    历史保留文件（无实际语义）
│   ├── surf_ref.png      全芯片表面参考
│   ├── surf_tol.png      全芯片表面容差
│   └── params.txt        本视角参数
```

建模命令：

```powershell
.\build\Release\markloc_gui.exe --buildmodel image\ok models\char
```

建模会自动：判定每个视角的极性、统计正常笔画频率、切分 line1/line2 的字符框。

---

## 6. 判定口径

| 口径 | 定义 | 适用 |
|---|---|---|
| **单图独立** | 每张图独立判 OK/NG，不合并 | 评估算法单张能力 |
| **按零件合并** | 同产品任一图 NG 则该产品 NG | 产线放行判定 |

GUI 训练完成后会**同时输出两种口径**（按图 + 按零件），无需切换。日志里各一段，底部汇总行显示按零件结果。

---

## 7. 实测结果（552 张 / 46 颗产品）

### 按零件合并（产线口径）

```
TP=4  FP=0  FN=0  TN=42
正确率 100%  精确率 100%  召回率 100%  F1 100%
```

| 零件 | 标签 | 产品分 | 触发视角 |
|---|---|---|---|
| 0026 | NG | 2.42 | v00 |
| 0083 | NG | 4.68 | v02 |
| 0110 | NG | 1.10 | v00 |
| 0112 | NG | 3.10 | v02 |

### 单图口径

```
TP=14  FP=0  FN=10  TN=252
正确率 96.38%  精确率 100%  召回率 58.33%  F1 73.68%
```

漏检集中在「缺陷只在某次拍摄显影」的图上（0110_RT、0112_RT）。

### 分视角统计

| 视角 | 图数 | 判 NG | 最大分 | 平均分 | 平均 ECC | 状态 |
|---|---|---|---|---|---|---|
| v00 | 92 | 6 | 2.42 | 0.14 | 0.816 | ✅ 启用 |
| v01 | 92 | 5 | 1.37 | 0.28 | 0.856 | ✅ 启用（关闭 logo 区，见 8.1） |
| v02 | 92 | 6 | 4.68 | 0.22 | 0.795 | ✅ 启用 |
| v03 | 92 | 8 | 2.93 | 0.24 | 0.800 | ✅ 启用（关闭 logo 区，见 8.1） |
| v04 | 92 | 8 | 2.90 | 0.28 | 0.790 | ✅ 启用（关闭 logo 区，见 8.1） |
| v05 | 92 | 2 | 1.06 | 0.08 | 0.737 | ✅ 启用 |

> 六视角全部启用后（按图）：`TP=18 FP=0 FN=30 TN=504`；按零件仍为 `TP=4 FP=0 FN=0 TN=42`。
> 相比仅三视角（TP=14 FP=0），**多抓获 4 张、误报保持 0**。详见 `VIEW_ENABLEMENT.md`。

### 性能

| 指标 | 值 |
|---|---|
| 平均耗时 | 140 ms/张（解码 40 / 识别 93 / 渲染 7） |
| 吞吐 | 7 张/秒 |
| 峰值工作集 | 56 MB |

---

## 8. 已知限制

### 8.1 v01 / v03 / v04 的 logo 区已关闭

这三个视角现已**全部启用**，但各自关闭了 logo 区域（`params.txt` 第 2 行末位 `enabled=0`）。
根因经实测拆解为三条：

| 视角 | 根因 | 证据 |
|---|---|---|
| v01 | 暗刻印背景下**亮度梯度**主导归一化尺度 | 中位/q94 = 0.29（正常视角 v02 为 0.16），背景整片被抬高 |
| v03 / v04 | 背景噪点经归一化后接近笔画强度 + appendage 分支失效 | 低 ECC 区域曾被计分；0116 appendage 分 1.498 而真 NG 0083 为 0 |
| 三者共有 | 低 ECC 区域曾被无条件放行（已统一为 `eccMin` 门控） | 修改前 ECC 0.258 的区域仍计 0.701 分 |

**关键实测**：在 v01 的 logo 区，**真 NG（0110）的 appendage 分 0.128 反而低于 OK（0116）的 1.251**，
说明该分支在此视角测的是背景梯度而非缺陷。关闭 logo 后三个视角均达到 **FP=0**。

**已尝试且确认无效的修法**（勿重复）：

| 方案 | 失败原因 |
|---|---|
| 加大 `edgeTB` | 伪线消失的同时，真 NG 0083 划痕分从 0.41 归零 |
| 提高 `strokeLo` | 真 NG 的 coverage 本就只有 0.20~0.25，先跌破门槛 |
| 提高 `appendageT` | 真 NG appendage 反而最低，先被杀掉 |
| 容差掩膜边界补偿 | 完全无效——主体分来自 appendage 分支，与残差连续性无关 |

**结论**：logo 区在 v01/v03/v04 上**确认放弃**，不是待办事项。
若将来要利用，需在信号层重做（反极性视角改用分布陡峭度估计尺度），不能靠现有判据调参。

### 8.2 缺墨与字符级比对默认关闭

上述两项检测已实现，但阈值未经充分标定，默认关闭
（`checkMissing=0`、`checkChars=0`）。打开后需按实际数据调
`missFreq` / `missStrokeLo` / `charMissRatio` / `charExtraRatio`。

### 8.4 字符切分依赖固定文本内容

字符级比对假设型号行 / 批号行内容固定（`Q046R8NLP`、`SA6JZ1F`）。
若换型后文本内容改变，必须重新建模生成字符框，否则逐字比对会全错。

logo 区含椭圆标，其连续笔画把内部 `SC` 包住，自动切分不可靠，
因此字符级比对只对 line1 / line2 启用。

---

## 9. 调参指南

### 单样本调试流程

```powershell
# 1) 看单颗产品六视角明细
.\build\Release\markloc_gui.exe --selftest "image\ng\0083_FT_Failed ... _00.tif"

# 2) 锁定这几颗，确认改动影响
.\build\Release\markloc_gui.exe --batch image --ids "0083,0112,0116"

# 3) 全量回归
.\build\Release\markloc_gui.exe --batch image
```

### 自动调参

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\autotune.ps1 `
    -View 02 -ScanDir out\scan_v02 -MaxEvals 30
```

坐标轮换爬山：逐个参数试值，目标函数 `TP - 5×FP`，最后按 OK 最大值自动标定阈值。

### 经验值参考

| 现象 | 调整方向 |
|---|---|
| 误报多 | 提高 `scratchT` / `scratchMinLen` |
| 漏检多 | 降低 `scratchT`，或降低 `scratchMinCoverage` |
| 噪点连线误判 | 提高 `scratchMinCoverage` |
| 轻微划痕误判 | 提高 `scratchMinLen` |
| 配准失败 | 检查极性 `invert`，或降低 `eccMin` |
| 纹理误报 | 提高 `pKeep` / 加大 `dilateK` |

---

## 10. 数据目录

```
image/
├── ok/    42 颗 × 12 张（FT/RT × 6 视角）= 504 张
└── ng/    4 颗（0026 / 0083 / 0110 / 0112）× 12 张 = 48 张
```

标签约定：文件名含 `_Failed` 或位于 `ng/` 目录为 NG。

---

## 11. 移交清单

### 可直接运行

- `build/Release/markloc_gui.exe` — 主程序（含 CLI 模式）
- `models/char/` — 六视角冻结模型，实测 100%

### 需要保留的源码

| 文件 | 说明 |
|---|---|
| `src/core/*` | 检测核心，改算法看这里 |
| `src/gui/*` | 界面，改交互看这里 |
| `CMakeLists.txt` / `CMakePresets.json` | 构建配置（含 OpenCV / Qt 路径） |
| `tools/` | 调参辅助 |

### 下一步优先级建议

1. **补缺墨分支** — 目前完全不检测缺失笔画，是最大的功能缺口
2. **缺墨/字符级阈值标定** — 需要在真实数据上验证并调参
3. **表面划痕** — 需先缩小 `surfaceRect` 并重新标定
4. **v01/v03/v04 特征升级** — 加方向判据，不要继续调阈值
