# markloc 改进清单

> 基于对 `src/`、`tools/`、`models/char/*/params.txt`、`CMakeLists.txt` 与 README 的只读审阅整理。
> 每条含：问题、证据（文件:行）、影响、建议改法、验证方式。
> 严重度：**P0** 崩溃/错误结果 · **P1** 性能/正确性隐患 · **P2** 可维护性/文档一致性。

**约定已移除项**：本清单不含“缺墨（断线/漏印）检测功能”——该功能未实现且按要求不做，见 §0。

---

## 0. 范围裁剪：缺墨检测（明确不做）

### 0.1 现状
代码中**不存在**任何缺墨/断线/漏印检测逻辑。全仓 `src/` 检索 `缺墨|漏印|断线|missing` 零命中。

它们只作为**文档性文字**存在于 README：

| 位置 | 原文 | 处理 |
|---|---|---|
| README 第 4 行 | “识别 **断线漏印**、多余笔画、连续划痕、笔画粗细、污损划痕 五类异常” | 改为四类，删除“断线漏印” |
| README 3.6 节标题 | “补两个盲区” | 保留（与缺墨无关） |
| README 8.3 节 | “缺墨（断线 / 漏印）分支未实现”——整节 4 行 | **整节删除** |
| README 11 节“下一步优先级” | 第 1 条“补缺墨分支 — 目前完全不检测缺失笔画，是最大的功能缺口” | 删除该条，后续条目重排 |

### 0.2 需要澄清的边界（**不可误删**）
“缺墨”与现有能力只差一个**信号方向**，删除时务必区分：

- **“多余笔画”分支 A**（`detect_appendage.cpp`）检测的是“容差区外**多出来**的强残差”，即多墨/异物/补笔 —— **保留**。
- **“笔画粗细”分支 C**（`detect_code.cpp`）检测的是笔画**局部变粗**（菌状堆积） —— **保留**。
- 真正的缺墨（应有笔画**缺失**）需要“`freq` 高但当前 `S` 低”的判据，该逻辑不存在，**不新增**。

### 0.3 连带影响
- README §7 实测结果反映的是“不够墨”的现状，删除 §8.3 后建议在该处补一句说明：“当前检测能力为‘多余墨迹类’四项，缺失笔画类不在范围内”，避免移交方误解为已覆盖。
- 无需改动任何 `.cpp`。这是一次**纯文档裁剪**。

---

## 1. P0 — 崩溃与错误结果

### P0-1 `scoreSurface` 可能抛 cv::Exception（尺寸未校验）
- **证据**：`char_core.cpp:179-180` 仅在 `loadFrom` 时校验 `surfRef.size() == surfTol.size()`，未校验是否等于 `p.surfaceRect` 的尺寸；`detect_surface.cpp:36-43` 直接把二者与 `sImg` 做 `convertTo`/`threshold`/`multiply`。
- **触发条件**：`params.txt` 中的 `surfaceRect` 与建模时不一致（手工改参数、复用旧模型、GUI 将来接入 surfaceRect 编辑），且 `useSurface=1`。当前 `useSurface=0` 所以**潜伏未爆**，一旦启用必炸。
- **影响**：未捕获的 `cv::Exception` → 进程终止（批量模式会丢掉整轮结果）。
- **改法**：
  1. `Model::loadFrom` 中增加 `surfRef.size() == 期望尺寸` 的校验，不符则 `vm.surfOk = false` 并记入 `err`；
  2. `scoreSurface` 入口加防御：尺寸不匹配直接 `return`；
  3. 调用 `analyze` 外围加 `try/catch (const cv::Exception&)`，落为“该区域不可靠”而非崩溃。
- **验证**：手工把某视角 `params.txt` 的 `surfaceRect` 改大，置 `useSurface=1`，跑 `--batch image --limit 5` 应不崩溃。

### P0-2 `Params` 越界访问风险（ROI 未经 `Rect` 合法性检查）
- **证据**：`char_core.cpp:212-214` 只检查 `x<0 || y<0 || x+w>cols || y+h>rows`，但未检查 `w<=0 || h<=0`；`roi.rect` 来自 `params.txt` 的 4 个整数，无范围钳制（`char_core.cpp:137-141`）。
- **触发条件**：`params.txt` 中 w/h 写 0 或负数（手改、脚本写坏）。`cv::Rect` 宽为 0 时 `gray(roi)` 得到空 Mat，后续 `strokeSignal` 的直方图循环 `total = 0`，`acc >= total*0.94` 立即成立但 `q94` 落到 `hist[0]-256 = -256`，被 `std::max(8.0, q94)` 救回；但 `estimateWarp` 会在空矩阵上 `resize` 抛异常。
- **影响**：崩溃或静默错误分数。
- **改法**：`loadParams` 读取后钳制 `rect.width = std::max(1, ...)`；`analyze` 增加 `rect.width <= 0 || rect.height <= 0` 的跳过。
- **验证**：临时把 `models/char/v00/params.txt` 第 2 行改成 `480 325 0 0`，`--selftest` 应跳过该区域而非崩溃。

### P0-3 `BatchDialog::runAll` 复用 `decodeFailed` 表达“跳过”
- **证据**：`BatchDialog.cpp:311`（非选中视角）、`:318`（视角未启用）、`:325`（无效视角）三处把**正常的过滤行为**标记为 `decodeFailed = true`，与 `:302` 真正的解码失败共用同一字段。
- **影响**：
  - `refreshRow` 显示“无法处理 ✘”（`:378-382`），把“按你的配置被跳过”误报成“文件读不了”；
  - `recomputeStats` 把被跳过的样本计入 `decodeFail`，报告第一节“无法处理 %3”计数失真；
  - CSV 导出中这些行的 `pred` 列会写成 `OK`（`:638`），**把未判定的样本伪装成 OK 判定**，这是最危险的一点——误读为“这些图判 OK 了”。
- **改法**：`Item` 增加独立状态枚举 `enum class State { Pending, Skipped, DecodeFailed, Done }`，替换布尔字段；`refreshRow`/`recomputeStats`/CSV 分别处理；CSV 对 `Skipped` 输出 `pred=""`（空）而**不是 `OK`**。
- **验证**：`--view 02` 跑全量，检查报告“无法处理”为 0、CSV 中非 02 视角行 `pred` 为空。

### P0-4 code 区结果可能在模型缺失时静默为 0
- **证据**：`char_core.cpp:280-283`，`i==3 && p.checkCode` 时 `regionScore = det::scoreCode(...)`，而 `scoreCode` 在 `stat.valid == false` 时直接返回 `0.0`（`detect_code.cpp:75`）。`codeStat.valid` 只有读到 `code_stats.txt` 才为 true（`char_core.cpp:170-176`）。
- **触发条件**：`code_stats.txt` 缺失/损坏/字段不足 4 个。
- **影响**：笔画粗细检测**静默失效**且不报错，产品永远在该区域得 0 分——一种“看起来正常”的漏检。
- **改法**：`loadFrom` 时若 `codeStat.valid == false` 且 `checkCode == true`，向 `err` 追加警告并让 GUI 状态栏显示；或改 `scoreCode` 返回 `-1` 代表“无效”，由调用方做成“不可靠”而非“0 分”。
- **验证**：临时重命名 `code_stats.txt`，加载模型应有可见警告。

---

## 2. P1 — 性能与正确性隐患

### P1-1 `scoreAppendage` 的 O(连通域数 × ROI 面积) 扫描
- **证据**：`detect_appendage.cpp:26-31`，对**每个**候选连通域 `k`，都完整双重循环扫描整个 ROI（700×235 ≈ 164k 像素）去找 `labels == k` 的像素。
- **影响**：残差碎点（纹理噪声）多时退化为 `n_components × 164k`。这是识别耗时 93ms 中最可能的长尾；批量 552 张时放大成可感知的卡顿。
- **改法**：用 `stats(k, CC_STAT_LEFT/TOP/WIDTH/HEIGHT)` 限定扫描窗口；或一次遍历 `labels` 时按 label 收集进 `std::unordered_map<int, std::vector<Point>>`。
- **验证**：在 logo 区构造含大量小噪点的图，对比改动前后 `--batch` 的“识别 ms”分项。

### P1-2 ECC 为线性运算却逐区域独立执行
- **证据**：`char_core.cpp:229-236`，4 个 ROI 各自调用一次 `estimateWarp`（每次都是 100 次迭代的 `findTransformECC`）。
- **影响**：同一视角下 4 个 ROI 的相对位置由夹具固定，理论上可用**一次全局配准**（或标定阶段的 ROI 映射）代替 4 次。当前占用 4×ECC 时间。
- **注意**：`autoCalibrateRois`（`char_core.cpp:430`）已实现了“视角间 ROI 映射”，但**运行时未使用**，两套机制割裂。
- **改法**（保守版）：保留逐区域 ECC，但把迭代上限按区域特点下调（logo 纹理丰富可 100，line1/line2 可 60）；code 区已跳过门控，也可考虑直接跳过 ECC 计算。
- **验证**：`--batch image` 前后对比耗时与 TP/FP。

### P1-3 `strokeSignal` 手写直方图（热路径标量代码）
- **证据**：`char_core.cpp:39-57`，512-bin 手工循环逐像素 `lround` + 边界钳制，全项目所有分支共用、每张图 4~5 次调用。
- **改法**：改用 `cv::calcHist`（OpenCV 有 SIMD 优化）或 `cv::minMaxLoc` + `cv::sort` 取分位数；至少把 `std::lround` 换成直接取整。
- **验证**：同样用 `--batch` 的“识别 ms”。

### P1-4 魔数 `80.0` 散落三处
- **证据**：`char_core.cpp:223`（`sRef` 读回除 80）、`:528`（`ref8` 存乘 80）、`:292` 与 `warp.cpp:17`、`char_core.cpp:600` 同类换算。
- **隐患**：`S` 理论上限远超 `1.0`；乘 80 存 8bit 会**截断溢出**（`S > 3.1875` 的像素全被裁到 255）。当前归一化后 94 分位约为 1，实际很少触顶，但阈值调小时会静默饱和，且回归时很难发现。
- **改法**：提取 `constexpr double kSignalStoreScale = 80.0;`；在 `strokeSignal` 出口加 `cv::min(s, 1.0)` 显式钳制（或按 `kSignalStoreScale` 动态定标），保证“存储无损”是显式设计而非巧合。
- **验证**：把 `strokeHi` 调到 0.1 触发更多高值像素，检查 `_ref.png` 是否出现纯白饱和区块。

### P1-5 `autoCalibrateRois` 用单张样本作模板
- **证据**：`char_core.cpp:436-443`，`okByView[0].front()` 单张；`:453-467` 在 0.80~1.20 间按 0.02 步长穷举缩放，只做**缩放 + 平移**，无旋转/透视。
- **影响**：单样本对异常图无鲁棒性；若某视角存在夹具旋转，ROI 会系统性偏差，而该偏差只在区域分数异常时才暴露。
- **改法**：模板用前 N 张 OK 图的均值；匹配成功阈值 `0.35`（`:468`）偏低，建议提到 0.5 并把匹配分数写入日志供人工复核。
- **优先级**：中——当前模型已冻结且 `--autocal` 非默认路径，属“备用路径”的健壮性问题。

### P1-6 笔画粗细的 z 完全依赖未见正例的假设
- **证据**：`detect_code.cpp:76-79`，`zDead = max(0.5, codeZNg − 1.5)`，`codeZNg = 3.5` 来自理论值；README §8.4 自承“只验证了不误报，缺正例”。
- **影响**：这是**唯一没有任何正例标定**的分支，其输出却直接参与产品判定。
- **改法**：在能拿到真实崩缺样本前，建议给该分支加“未标定”标志，GUI 上以灰色而非红/绿显示其分数，避免让人误以为它已被验证。
- **验证**：无样本则无法自动验证，至少做到“不误导”。

### P1-7 阈值 `kNgThreshold = 1.0` 与逐视角 `scratchT` 的耦合是隐式的
- **证据**：`char_core.hpp:18` 全局阈值 1.0；各视角 `scratchT` 差异达 **12×**（v02 13976 vs v01 170565）。
- **隐患**：区域分 = `energy×len / scratchT`，所以“1.0 分”在不同视角代表**完全不同的绝对残差量**。跨视角比较分数（GUI 的六视角表格、批量报告第五节“逐件各视角最高分”）在语义上并不严格可比。
- **改法**：文档中明确“视角分仅在该视角内可比”；或引入统一的绝对度量作为辅助列（如原始 `energy×len`）。
- **优先级**：中——不影响判定正确性，影响人读报告的解读。

---

## 3. P2 — 可维护性与文档一致性

### P2-1 `modeSel_` 声明但从未创建（README 承诺的功能不存在）
- **证据**：`BatchDialog.hpp:82` 声明 `QComboBox* modeSel_`；`BatchDialog.cpp:530` 读取它；但 `buildUi()`（`:104-150`）**从未实例化该控件**。同时 `:530` 的 `mergeMode` 变量赋值后**从未使用**（编译器 `/W4` 下应有未使用变量告警）。
- **历史遗留**：README §6 表格写“GUI 批量窗口可随时切换，切换后统计立即重算”，与实际不符。
- **改法**：二选一——(a) 补上控件并让 `recomputeStats` 真正按 `mergeMode` 只输出一种口径；(b) 删除 `modeSel_`、`mergeMode`，README §6 改为“两种口径固定同时输出”。
- **建议**：选 (b)，因为两种口径同时可见对调参更有用（按图看算法能力，按零件看产线放行）。

### P2-2 报告节数与 README 不符
- **证据**：README §3.8 写“报告六节”；`BatchDialog.cpp` 实际输出**七节**（一、样本 / 二、六视角参数 / 三、准确率 / 四、分视角统计 / 五、按零件 / 六、速度 / 七、内存）。
- **改法**：README 改为七节并同步各节标题。

### P2-3 `tools/` 中的 Windows 路径编码隐患（与 `naming` 的正确做法不一致）
- **证据**：`mosaic.cpp:24-35` 用 `fs::path::stem().string()` / `.filename().string()`；`crop.cpp:21`、`montage.cpp:32` 用 `cv::imread(argv[i])`。而 `naming.cpp:13-27` 已实现正确的 `u8path` 转换路径（README §3.1 也专门警告过这个坑）。
- **影响**：项目里的图片文件名普遍含中文（如 `(字符)`），这些工具在 Windows 上可能读不到/解析错文件名。属**调参辅助工具**，不影响主程序。
- **改法**：复用 `naming::` 的实现（`src/core` 已是静态库，工具已链接 `markloc_core`，可直接调用）；`cv::imread` 改为“读文件字节 + `cv::imdecode`”的辅助函数（`MainWindow.cpp:42-53` 已有现成实现，可提取到 `common/`）。

### P2-4 `autotune.ps1` 的目标函数在样本不平衡下失衡
- **证据**：`autotune.ps1:30`，目标 `obj = TP − 5×FP`；`:23` 每次评估都 `& $Exe --batch $ScanDir` 起完整进程。
- **问题**：
  - 样本 42 OK : 4 NG，TP 上限只有 4，而 FP 上限 42。`−5×FP` 的权重意味着**少一个 FP 永远比多一个 TP 更值钱**，搜索会系统性趋于保守（阈值越拉越高），与 README 宣称的“保证不漏检优先”相悖。
  - 每次评估都要重新解码整个扫描目录，`MaxEvals=40` 时约 40 次全量解码。
- **改法**：
  1. 目标函数改为与样本量无关的 F 值或 `TPR − λ·FPR`（`λ` 取 0.5~1），避免被 OK 样本数量主导；
  2. 让 `--batch` 支持“仅输出统计摘要”（跳过 `makeExplain` 渲染），可显著降低单次评估开销；
  3. 若可行，加 `--params-override` 参数避免“改文件→重跑”的 IO 往返。
- **验证**：对比新旧目标函数在 v02 上搜出的参数是否导致 FP 上升。

### P2-5 `runBatch` 的 `--ids` / `--view` 过滤依赖字符串后缀匹配
- **证据**：`main.cpp:200-204`，`stem.endsWith("_" + viewFilter)`。`viewFilter` 未做格式校验，“2”会静默匹配不到任何文件（因为文件名是 `_02`），只在末尾报“没有匹配的样本”。
- **改法**：`viewFilter` 归一化为两位（`QString::number(v).rightJustified(2,'0')`）并校验 0~5，非法直接给出明确错误。
- **附带**：README §1 已正确提醒 PowerShell 下 `--ids` 需加引号，这条可保留。

### P2-6 `buildModel` 对单张图 `front()` 的隐式依赖
- **证据**：`char_core.cpp:506`（极性判定用 `imgs.front()`）、`:519`（越界校验用 `imgs.front()`）、`:598`（表面参考用 `imgs.front()`）。
- **风险**：若目录里第一张是异常/曝光错误图，会污染极性判定与表面模板，且**无任何告警**。`front()` 只是文件名排序第一，不是“最典型”。
- **改法**：极性判定改用前 N 张的统计（正/负尾部强度分别取均值再比）；`_ref` 仍可用 `front()` 作配准基准（逐张 ECC 会自我纠正），但表面模板建议用 `freq` 加权而非直接取单张。
- **注**：`:598` 的 `sRefS` 仅用于配准基准，风险相对低。

### P2-7 `console_utf8.hpp` 未被任何 GUI 目标使用
- **证据**：`enableUtf8Console()` 只被 `tools/*.cpp` 调用；`main.cpp`（GUI + CLI 三模式）**未调用**，但它的 `--selftest`/`--batch` 都会 `std::printf` 中文。
- **影响**：命令行模式下中文输出在 CP936 控制台可能是乱码。
- **改法**：`main.cpp` 的 `main()` 开头调用 `markloc::enableUtf8Console()`（该头文件目前未被 `markloc_gui` 包含，需确认 include 路径——`CMakeLists.txt:26` 已 `target_include_directories(... src)`，可直接 `#include "common/console_utf8.hpp"`）。

### P2-8 `Params` 的 `surface*` 系列参数无 UI 入口
- **证据**：`char_core.hpp:52-58` 定义了 `surfaceRect / surfMinLen / surfMedianK / surfThr / surfAreaT / surfMinElong / useSurface` 共 7 项，但 `MainWindow::buildParamPanel` 一个都没暴露；`saveParams/loadParams` 也**不持久化**这 7 项（只存 `useSurface` 一个布尔，`char_core.cpp:113`）。
- **影响**：`surfMinLen=200`、`surfThr=0.55` 等值只能改代码重新编译，`params.txt` 里改不了。这与 README §8.2 给出的建议“缩小 surfaceRect 避开四边与圆标”**实际无法通过配置实施**——必须改源码。这是 README §8.2 建议落不了地的根因。
- **改法**：`saveParams/loadParams` 增加 surface 参数行（向后兼容：读不到就用默认值）；GUI 加一组“表面区域”控件。**在完成前，README §8.2 应注明“需改源码中的默认值”**。

---

## 4. 建议执行顺序

| 阶段 | 条目 | 理由 |
|---|---|---|
| **第一步** | §0 文档裁剪（去掉缺墨） + P0-3 + P2-1 + P2-2 | 消除误导，成本极低，不碰算法 |
| **第二步** | P0-1、P0-2、P0-4 | 纯防御性加固，不改变任何现有分数，可作为“冻结版本”的安全网 |
| **第三步** | P1-1、P1-3、P1-4 | 性能与精度，需用 `--batch image` 回归确认 TP/FP 不变 |
| **第四步** | P2-4、P2-5、P2-7 | 工具链易用性 |
| **第五步** | P2-8 → 解锁 README §8.2 的表面划痕分支 | 功能面最大的一块，需先解决配置能力才谈得上调参 |
| **暂缓** | P1-2、P1-5、P1-6、P1-7 | P1-2 收益有限、风险高；P1-5 是非默认路径；P1-6 无样本无法验证；P1-7 只影响解读 |

## 5. 回归基线（每步改动后必跑）

```powershell
# 单视角快速比对（改/划痕分支时）
.\build\Release\markloc_gui.exe --batch image --view 02

# 全量回归：期望 TP=4 FP=0 FN=0 TN=42（按零件）、TP=14 FP=0 FN=10 TN=252（按图）
.\build\Release\markloc_gui.exe --batch image

# 抽检真缺陷是否仍有分
.\build\Release\markloc_gui.exe --batch image --ids "0026,0083,0110,0112"
```

> ⚠️ 基线只有 4 颗 NG，任何“100%”都应视为弱证据。**引入新改动时优先看 FN 是否增加，而不是看总体百分比。**
