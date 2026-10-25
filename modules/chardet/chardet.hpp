#pragma once

// chardet —— 字符级检测模块（自包含）
//
// 职责（模块内完成，不依赖宿主项目）：
//   1. 训练：从 OK 样本统计笔画频率图
//   2. 切分：在频率图上自动切出字符区域
//   3. 输出：写成一个自包含的模型目录
//   4. 消费：其他模块读这个模型目录，即可得到模板与比对区域
//
// 模型目录布局（自包含，可整体拷贝）：
//   <modelDir>/
//   ├── model.json        元信息（版本、视角、参数、尺寸）
//   ├── freq.png          归一化笔画频率图（0~255 表示 0.0~1.0）
//   ├── ref.png           配准基准信号
//   └── regions.txt       切分结果：每个区域的字符框
//
// regions.txt 格式（v2，行式）：
//   # regions v2
//   #region <序号> <形状> <x> <y> <宽> <高> <旋转角> <字符数>
//   <每个字符一行：bbox x y w h  extend x y w h  序号>
//
// 形状取值 box / ellipse / circle。
// ellipse/circle 的 <x y 宽 高 旋转角> 直接定义闭合曲线：
//   中心 = (x+宽/2, y+高/2)，长半轴 = max(宽,高)/2，短半轴 = min(宽,高)/2，
//   长轴角度 = 旋转角（逆时针为正）。
// 椭圆不再从墨点拟合 —— 框就是定义，离心率由宽高比唯一确定。
//
// 依赖：仅 OpenCV + 标准库

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "config.hpp"

namespace chardet {

// ---------------------------------------------------------------- 基础类型

/// 单个字符
struct CharBox {
    cv::Rect bbox;     // 字符主体框
    cv::Rect extend;   // 扩展比对框
    int index = 0;     // 字符内序号（从左往右）
};

/// 区域形状类型
///
/// 不同形状要用不同的判据：
///   · Box    字符区：矩形框，跑完整的字符级检测（缺墨/多墨/划痕/断连）
///   · Ellipse 椭圆环：沿闭合曲线扫连续性，断点=环被切断
///   · Circle  圆形：同上，只是长短轴相等
///
/// 为什么必须分开：椭圆环用矩形框去比对毫无意义——
/// 框里既有环也有背景，一个 322x225 的框里环只占一条线，
/// 任何"框内墨量"的判据都被背景稀释掉了。必须沿曲线本身看。
enum class RegionShape {
    Box = 0,
    Ellipse = 1,
    Circle = 2,
};

/// 一个比对区域（一个选区及其切出的所有字符）
///
/// 两种区域用两套几何描述，互不干扰：
///
///   字符区（Box）     用 rect（轴对齐矩形）圈定范围，里面切出 chars。
///
///   图案区（Ellipse/Circle）  用 ell 描述的闭合曲线，
///                    常用参数是中心、长半轴、短半轴、长轴角度。
///                    这时 rect 只作为它的轴对齐外接框同步维护，
///                    方便按矩形做初步裁剪；判据一律走 ell。
///
/// 为什么图案区不能只靠 rect：轴对齐矩形存不下"半轴 + 角度"。
/// 同一个外接框能对应多个不同角度、不同半轴的椭圆，反推是多解的。
/// 而椭圆环在实物上多半是斜的（SC logo 实测长轴 -13.3°），
/// 不存角度就没法贴合，沿周长采样会一路擦边。
struct EllipseGeom {
    cv::Point2f center;        // 中心
    float a = 1.0f;            // 长半轴
    float b = 1.0f;            // 短半轴
    float angleDeg = 0.0f;     // 长轴相对 +x 的角度，逆时针为正

    /// 离心率：0 = 正圆，越接近 1 越扁。由 a/b 唯一确定，不单独存。
    double eccentricity() const {
        const float ma = std::max(a, b), mb = std::min(a, b);
        if (ma < 1e-6f) return 0.0;
        const double r = static_cast<double>(mb) / ma;
        return std::sqrt(std::max(0.0, 1.0 - r * r));
    }
};

struct RegionInfo {
    int index = 0;                 // 区域序号
    cv::Rect rect;                 // 选区范围（整图坐标）；图案区里是外接框
    std::vector<CharBox> chars;    // 该区域内切出的字符

    RegionShape shape = RegionShape::Box;

    /// 图案区的椭圆参数（shape != Box 时有效）。圆就令 a == b。
    EllipseGeom ell;

    int charCount() const { return static_cast<int>(chars.size()); }
    bool isPattern() const { return shape != RegionShape::Box; }

    /// 由 ell 生成，或解析 regions.txt 后由 syncRectFromEll() 维护。
    void syncRectFromEll() {
        const double ca = std::cos(ell.angleDeg * CV_PI / 180.0);
        const double sa = std::sin(ell.angleDeg * CV_PI / 180.0);
        const double a = ell.a, b = ell.b;
        const double hw = std::hypot(a * ca, b * sa);
        const double hh = std::hypot(a * sa, b * ca);
        rect = cv::Rect(static_cast<int>(std::lround(ell.center.x - hw)),
                        static_cast<int>(std::lround(ell.center.y - hh)),
                        static_cast<int>(std::lround(2 * hw)),
                        static_cast<int>(std::lround(2 * hh)));
    }
};

/// 完整模型（内存形式）
struct Model {
    int version = 1;
    int view = -1;                       // 视角号
    cv::Size imageSize;                  // 训练图尺寸
    Config config;                       // 训练时用的参数

    cv::Mat freq;                        // 频率图 CV_8U，0~255 表示 0.0~1.0
    cv::Mat ref;                         // 配准基准 CV_8U

    std::vector<RegionInfo> regions;     // 切分出的区域与字符

    /// 多模板库：`charTemplates[字符全局索引]` = 该字符的若干张真实样例
    ///
    /// 为什么需要它：频率图是 84 张的交集，细节被平均化，
    /// 改笔画这类 2%~12% 的局部变化会淹没在平均化误差里。
    /// 真实模板保留了完整形态，能抓住「与任何模板都不像」的情况。
    std::vector<std::vector<cv::Mat>> charTemplates;
    int templateSize = 64;               // 每个模板归一化后的边长

    /// 模型图存在即为有效。区域可以稍后由用户在模型图上框选得到，
    /// 因此 regions 为空不影响模型本身可用。
    bool valid() const { return !freq.empty(); }
    /// 是否已经划出了可比对的区域
    bool hasRegions() const { return !regions.empty(); }
    int totalChars() const;
};

// ---------------------------------------------------------------- 训练

/// 训练输入：同一视角的一组 OK 图像（尺寸需一致）
struct TrainInput {
    std::vector<cv::Mat> images;   // CV_8U 灰度
    int view = 0;                  // 视角号
};

/// 训练结果
struct TrainReport {
    bool ok = false;
    int usedImages = 0;
    int skipped = 0;
    int autoRegions = 0;
    int autoChars = 0;
    double elapsedMs = 0.0;
    std::string message;
};

/// 训练并生成模型图
Model train(const TrainInput& in, const Config& cfg, TrainReport* report = nullptr);

// ---------------------------------------------------------------- 精对齐

/// 精对齐报告
struct RefineReport {
    bool ok = false;
    int usedImages = 0;        // 参与累加的样本数
    int totalChars = 0;        // 字符框总数
    int refinedPairs = 0;      // 成功精对齐的 (样本,字符) 对数
    int skippedWeak = 0;       // 因匹配质量差被跳过的对数
    double meanPeak = 0.0;     // 平均匹配峰值
    double minPeak = 1.0;      // 最低匹配峰值
    double elapsedMs = 0.0;
    std::string message;
};

/// 用初版模型图 + 用户划定的字符框，做逐字符精确对齐后重新平均。
///
/// 原理：初版模型图由粗对齐得到，仍有刻印偏移（实测 σ≈6px）导致笔画发糊。
/// 本函数对每张样本的每个字符，在字符框周围 ±searchRadius 内搜索最佳位置，
/// 只累加匹配良好的样本，从而消除逐字符的位置漂移。
///
/// images  —— 同一视角的 OK 样本（CV_8U）
/// coarse  —— 初版模型图（train 的产物），其 regions 为用户划定的字符框
/// searchRadius —— 每个字符框周围的搜索半径（px）
/// minPeak —— 匹配峰值低于此值的样本不参与该字符累加
/// cancelled —— 可选的中断标志；被置 true 时提前返回（用于 GUI 关闭时取消）
/// onProgress —— 可选的进度回调：参数为 (已完成样本数, 总样本数)
Model refine(const std::vector<cv::Mat>& images, const Model& coarse, const Config& cfg,
             int searchRadius = 30, double minPeak = 0.60, RefineReport* report = nullptr,
             const std::atomic<bool>* cancelled = nullptr,
             const std::function<void(int, int)>* onProgress = nullptr);

// ---------------------------------------------------------------- 模型存取

/// 保存为自包含目录（会创建目录）
bool saveModel(const std::string& modelDir, const Model& m);

/// 读取模型目录
bool loadModel(const std::string& modelDir, Model& m);

/// 是否是有效模型目录
bool isModelDir(const std::string& modelDir);

// ---------------------------------------------------------------- 消费：模板与比对区域

/// 从模型中取出某区域的模板频率图（只保留该区域选区内）
cv::Mat regionTemplate(const Model& m, int regionIndex);

/// 从模型中取出某区域所有字符的模板（每字符一份，同尺寸稀疏图）
std::vector<cv::Mat> charTemplates(const Model& m, int regionIndex);

/// 把模型里的区域转成可直接比对的字符单元
std::vector<struct CharUnit> toUnits(const Model& m, int regionIndex);

/// 把全部区域的字符单元合并成一个列表（顺序与 charTemplates 索引一致）
std::vector<CharUnit> toAllUnits(const Model& m);

/// 多模板比对结果
struct MultiResult {
    bool valid = false;

    /// 每个字符与「最近模板」的差异（1 - 余弦相似度）
    /// 值越大说明该字符和模板库里的任何正常样例都不像
    std::vector<double> perChar;

    /// 差异最大的字符（= 最可疑的那个）
    double maxDiff = 0.0;
    /// 第二大的字符（用于排除单点噪声）
    double secondDiff = 0.0;
};

/// 多模板比对
///
/// 与同「平均频率图」比对的区别：
///   频率图是 84 张的交集，细节被平均化，改笔画这类 2%~12% 的局部变化
///   会淹没在平均化误差里。真实模板保留完整形态，能抓住
///   「与任何模板都不像」的情况。
///
/// units 会被原地更新为对齐后的框位置。
MultiResult diffMultiTemplate(const cv::Mat& signal, const Model& model,
                              std::vector<CharUnit>& units);

// ---------------------------------------------------------------- 切分

/// 从频率图切分字符（供训练时自动切分用）
std::vector<struct CharUnit> split(const cv::Mat& freq8, const Config& cfg);

// ---------------------------------------------------------------- 比对

/// 单个字符单元
struct CharUnit {
    cv::Rect bbox;
    cv::Rect extend;
    cv::Mat freq;      // 该字符的模板（与传入信号同尺寸）
    int index = 0;
};

/// 缺陷类型
enum class DefectType {
    None = 0,
    Missing,      // 少墨（笔画缺失，含整字缺失）
    Dirt,         // 脏污（局部多余墨点）
    Scratch,      // 划痕（长条状多余墨）
    Extra,        // 多墨（笔画变粗 / 多一笔）
    Shape,        // 字形偏移
};

const char* defectName(DefectType t);

/// 单字符比对结果
struct CharMatch {
    int index = 0;
    cv::Rect bbox;

    // 各判据独立分数（0 = 无该缺陷）
    double missScore = 0.0;      // 少墨
    double dirtScore = 0.0;      // 脏污
    double scratchScore = 0.0;   // 划痕
    double extraScore = 0.0;     // 多墨
    double shapeScore = 0.0;     // 字形偏移

    double score = 0.0;          // 以上最大值
    DefectType type = DefectType::None;

    // 中间量（便于调参）
    double missRatio = 0.0;
    double extraRatio = 0.0;
    bool missingChar = false;    // true = 整字缺失
    int dirtBlobs = 0;           // 脏污连通块数
    int scratchCount = 0;        // 划痕条数
    int scratchLen = 0;          // 最长划痕长度

    std::string label;
};

/// 逐字符比对
/// aligned: 当前图像笔画信号 CV_32F（需与模板同尺寸、已配准）
std::vector<CharMatch> match(const cv::Mat& aligned, const std::vector<CharUnit>& units,
                             const Config& cfg);


// ---- 字符级判据（分组预期）

/// 归一化互相关比对结果
struct NccResult {
    bool valid = false;
    cv::Rect box;
    double ncc = 1.0;      // 已不使用，保留字段兼容
    double localExtra = 0.0;  // 最大新增墨连通块 / 模型墨量（抗稀释）
    double missRatio = 0.0;   // 必然有墨处却无墨的比例
    double extraRatio = 0.0;  // 应为空白处却有墨的比例
    double missScore = 0.0;
    double extraScore = 0.0;
    double score = 0.0;

    // 定位：缺陷像素所在的列/行区间（相对框）
    std::vector<cv::Range> missSpanX;
    std::vector<cv::Range> missSpanY;
    std::vector<cv::Range> extraSpanX;
    std::vector<cv::Range> extraSpanY;
};

/// 分组预期判据
///
/// NCC 对待检图的线性变换（整体变粗/变细/变亮/变暗）不敏感，
/// 只反映各位置的相对分布。而"缺一笔/多一笔"破坏的正是相对分布。
/// 这是唯一能同时绕开"模型是交集图"和"笔画粗细个体差异"的判据。
NccResult diffNcc(const cv::Mat& signal, const cv::Mat& modelFreq, const CharUnit& unit,
                  const Config& cfg);

/// 骨架拓扑比对结果

// ---- 整行墨块比对（多字/少字，最可靠）

/// 整行墨块比对结果（多字 / 少字）
struct BlockDiff {
    struct Block {
        int x0 = 0, x1 = 0;   // 像素坐标
        int y0 = 0, y1 = 0;
    };

    bool valid = false;
    int rowCount = 0;              // 处理的文字行数
    int modelBlocks = 0;           // 模型总墨块数
    int testBlocks = 0;            // 待检总墨块数
    int extraCount = 0;            // 多出的墨块数
    int missingCount = 0;          // 缺失的墨块数
    std::vector<Block> extraBlocks;    // 多出的位置
    std::vector<Block> missingBlocks;  // 缺失的位置

    /// 逐行字符数量校验
    ///
    /// 回答"这一行本该有几个字、实际有几个"。
    /// 与 extraCount/missingCount 的区别：那两个是按墨块统计的，
    /// 这里按字符区间逐个核对，得到的是字符级的增减数量。
    struct RowCount {
        int y0 = 0, y1 = 0;   // 行带范围
        int expected = 0;     // 模型该行字符数
        int found = 0;        // 待检实际字符数
        int extraChars = 0;   // 多出几个
        int missingChars = 0; // 缺少几个
        bool ok() const { return extraChars == 0 && missingChars == 0; }
    };
    std::vector<RowCount> rows;
};

/// 整行墨块比对：检测"多出一个字符"/"少一个字符"
///
/// 为什么用墨块而不是投影积分：
///   模型（84 张平均）与待检（单张）在整行尺度的灰度量级差异极大，
///   实测 5 种量纲处理方式（二值化/尺度归一/连续信号/直方图匹配/去尺度强制）
///   全部失败——OK 与 NG 的积分值完全重叠。
///
///   但"有几块墨、每块在哪"是量纲无关的：模型 9 块、待检 10 块且多出的
///   那块位置独立 → 就是多字，与灰度无关。
///
/// 局限：只能发现整块的多/缺，笔画级缺陷由 diffInBox 处理。
BlockDiff diffBlocks(const cv::Mat& signal, const cv::Mat& modelFreq,
                     const std::vector<CharUnit>& units, const Config& cfg, int margin = 8);

// ---- 图案区域的断点检测（椭圆 / 圆）
//
// 与字符区完全分开的一套判据。字符区问"墨量对不对"，
// 图案区只问"这条闭合曲线有没有被切断"。
//
// 为什么不复用字符级的判据：椭圆环在一个 322x225 的框里只占一条 5~8px 宽的线，
// 框内 90% 是背景。任何"框内墨量比例"都会被背景稀释到分不出来。

/// 椭圆/圆断点检测参数
struct PatternBreakParams {
    bool enabled = true;
    int angles = 720;         // 沿周长采样的角度数
    double inkThr = 0.35;     // 该角度"有墨"的强度门槛（与 strokeHi 一致）
    int radiusTol = 3;        // 沿跟踪出的半径曲线两侧允许的偏差(px)
    // 环内屏蔽比例：半径小于 innerBlockRatio × 长半轴的区域完全不参与判断。
    // 图案内部常还有别的元素（SC 环里就有 S、C 两个字母），
    // 它们会在"沿法向找最亮"时把缺口补上，必须屏蔽掉。
    double innerBlockRatio = 0.7;

    // ---- 带型连通判定（bandConnected 方式，见 .cpp 里"方法取舍"一节）
    //
    // 【为什么不再用"跟踪半径曲线"】
    // 老做法是：每个角度找半径方向最亮的位置 r(θ)，再沿这条曲线判有无墨。
    // 它假设"环在任何一个角度都只对应一个半径"。但实测的环是一段粗一段细的
    // 闭合带，"最亮处"会在这条带的两条边缘之间来回跳，跳动幅度就等于
    // 起伏量（83 张 OK 上中位 28px）。所以起伏量测到的是环的粗细变化，
    // 不是对准误差 —— 0109_RT 起伏 48px，但覆盖率 1.000、断口 0。
    // 拿它当"对准"的代理，就会把环完好的图判成"没验过"。
    //
    // 新做法改为：一个角度只要在整条半径带里有一段连续有墨，就算该角度
    // "有墨"。环粗一段细一段完全不影响，也不需要"对准检查"这道安全网。
    bool bandConnected = true;
    // 半径带的下界比例：低于 bandInnerRatio × a 的区域不看。
    // 必须高于 innerBlockRatio 才有意义（那里已经被清零）。
    double bandInnerRatio = 0.75;
    // 半径带的上界比例：高于 bandOuterRatio × a 也不看。
    // 环外沿实测在 1.05a 以内，留到 1.35a 是给粗笔画的环留余量。
    double bandOuterRatio = 1.35;
    // 该角度"有墨"要求半径带里至少有多少个连续像素超过 inkThr。
    // 取 3：单像素的噪点不算，5~8px 宽的环则绰绰有余。
    int minRunPixels = 3;

    // ---- 展开前的灰度阈值（preGrayThr）
    //
    // 【为什么要多这一道】
    // strokeSignal 出来的信号里，环的亮度在一段范围内连续变化（细的地方弱、
    // 粗的地方强）。极坐标展开之后再按 inkThr 判二值，等于让"环变淡"
    // 这段信息被后面的归一化和三像素连通抹平 —— 淡到一定程度仍然算有墨，
    // 于是缺口恒为 0，缺陷看不见。
    //
    // 更合适的做法是**在展开之前**就把信号里低于某个灰度的部分切掉：
    //   ① 先拉直（椭圆 -> 正圆）
    //   ② 按 preGrayThr 做一次硬阈值，低于它的像素直接置 0
    //   ③ 再做极坐标展开
    // 这样"淡"在展开图上是真实存在的空洞，展开后的连通判定才能看见它。
    //
    // 与 inkThr 的区别要分清：
    //   preGrayThr 作用在**展开之前**的拉直图上，是对图像本身的门槛；
    //   inkThr     作用在**展开之后**的极坐标图上，是对"该角度算不算有墨"的门槛。
    // 前者决定"什么算墨"，后者决定"多少个像素算一条线"，两者不能互相替代。
    //
    // 取 0 表示关闭这一步（回到原来的行为）。
    double preGrayThr = 0.0;

    double minGapAngle = 0.6; // 缺口最小角度(度)：小于此算噪声
    double maxGapAngle = 12.0;// 缺口最大角度(度)：大于此算整块缺失而非断连
    int minFlankCount = 8;    // 缺口两侧至少各有多少个连续有墨采样点
    int mergeGapCount = 6;    // 相邻缺口间隔小于这么多采样点就并成一段

    // ---- 打分门槛（与字符区的划痕门槛完全独立）
    //
    // 两套分数各用各的门槛，不合并成一个数。
    // 理由：划痕分 = 深度 × 跨字数，单位是"灰度×字数"；
    // 断点分 = 缺口角度 × 100，单位是"度的百分之一"。
    // 两者量纲不同，强行相加会让门槛失去物理含义，
    // 而且以后调其中一套会莫名其妙影响另一套。
    //
    // 【门槛数值的依据，别凭感觉改】
    // 实测 83 张 OK 图（环全部完好）上的断点分分布：
    //   76 张 = 0，5 张 = 100，1 张 = 150，1 张 = 250
    // 也就是说刻印本身的粗糙最多能造出 2.5° 的假缺口（250 分）。
    // 所以门槛必须高于 250，否则 7 张 OK 图会误报 ——
    // 第一版把 minScore 写成 100，就是这么误报的。
    double cutSuspect = 400.0;   // 4.0° 以上算可疑
    double cutObvious = 800.0;   // 8.0° 以上算明显
    double cutCertain = 1200.0;  // 12.0° 以上算确定
    double minScore = 400.0;     // 低于此不报（必须 > OK 集的最大噪声 250）
};

/// 图案区的椭圆参数（全图坐标）。
/// 注意：这些不是从墨点拟合的，而是直接由选框算出来的
/// （中心=框中心，半轴=框宽高的一半，长轴角度=RegionInfo::angleDeg）。
/// 名字保留 fit 是为了少改调用点，别被误导。
struct EllipseFit {
    double cx = 0, cy = 0;   // 中心
    double a = 1, b = 1;     // 半轴
    double rot = 0;          // 旋转（弧度）
    int support = 0;         // 支撑点数
    double resid = 1.0;      // 中位残差，越小越贴合
};

/// 图案区档位：与字符区的 ScratchBand 同一套语义，
/// 但分数来源完全不同，所以单独取名，别混用。
enum class PatternBand { None = 0, Suspect = 1, Obvious = 2, Certain = 3 };

inline const char* patternBandName(PatternBand b) {
    switch (b) {
        case PatternBand::Certain: return "确定";
        case PatternBand::Obvious: return "明显";
        case PatternBand::Suspect: return "可疑";
        default: return "无";
    }
}

/// 断点检测结果
struct PatternBreakResult {
    bool valid = false;        // 区域参数是否可用（是图案区且有尺寸）
    bool fitted = false;       // 是否成功从墨拟合出椭圆环
    bool allMissing = false;   // 区域内几乎无墨：整条曲线没印出来，不是断点
    // 对准质量：椭圆和实际环贴不贴得上。
    // 没对上时下面的断口结论一律不作数（breaks/score 已清零），
    // 调用方要能和"查过没问题"分开显示 —— 这是"没验过"，不是"合格"。
    bool alignOk = false;
    double radiusLineWidth = 0;  // 环在半径方向的墨线宽度(px)，对准判据的参照
    int breaks = 0;            // 断口数量
    int gapCount = 0;          // 缺口段数（含不算断口的）
    double maxGapAngle = 0;    // 周上最长缺口的角度
    double maxBreakAngle = 0;  // 判为断口的最长缺口角度
    double coverage = 0;       // 周上被判为有墨的比例
    double radiusSwing = 0;    // 跟踪出的环半径在角度方向上的起伏(px)。
                               // 这个数比墨线宽度还大的话，说明框和环对不上。
    int innerBlocked = 0;      // 被屏蔽的环内半径(px)，0 = 没屏蔽
    int bandOuter = 0;         // 半径带外沿(px)，0 = 未使用带型判定
    double preGrayThr = 0.0;   // 本次使用的展开前灰度阈值，0 = 未启用
    // 起伏量超过门槛，但带型判定下结论仍然有效。
    // 只用于显示"这张环的粗细变化较大"，不参与判 NG。
    bool swingOverLimit = false;
    EllipseFit fit;            // 椭圆参数（来自选框，不是拟合）
    std::vector<cv::Point2f> breakAt;  // 每个断口的全图坐标

    // ---- 打分（与划痕分完全独立）
    double score = 0.0;        // 断点分 = maxBreakAngle × 100
    PatternBand band = PatternBand::None;
    bool flagged = false;      // score >= minScore，即"该报"
};

/// 图案区域断点检测
///
/// region 必须是 Ellipse 或 Circle 形状；Box 区域直接返回 valid=false。
/// signal 为 CV_32F 的笔画信号，尺寸需与整图一致。
PatternBreakResult analyzePatternBreak(const cv::Mat& signal, const RegionInfo& region,
                                       const Config& cfg, const PatternBreakParams& P);

// ---- 新模式：模块化流水线（pattern/ 子目录）
//
// 上面那个函数是重构前的老实现，保留是为了不破坏既有调用点。
// 内部把 9 个阶段硬编码串在一起，换判据必须整段改写。
//
// 新实现在 pattern/ 下，拆成三层接口：
//   预处理：signal → 拉直图 + 极坐标图
//   剖面  ：极坐标图 → 逐角度墨量曲线
//   检测器：曲线 → 缺陷结论（可注册多个，各自独立打分）
//
// 用法：
//   pattern::Params p;
//   p.detectors = {"notch_window", "gap_run"};
//   pattern::Result r = pattern::analyze(signal, region, cfg, p);
//   if (!r.valid) { /* 没验过，不能当合格 */ }
//   if (r.flagged) { /* 有缺陷 */ }
//
// 加算法：在 pattern/detector/ 实现 IDetector，在 det_registry.cpp 注册一行。
// 详见 modules/chardet/pattern/README.md 与 IMPLEMENTATION.md。
namespace pattern {
struct Result;
}

/// 用新流水线跑一遍，同时把结果转回老结构体
///
/// detail 非空时回填完整的流水线结果（含各检测器结论），便于 GUI 显示。
PatternBreakResult analyzePatternBreakV2(const cv::Mat& signal, const RegionInfo& region,
                                         const Config& cfg, const PatternBreakParams& P,
                                         pattern::Result* detail);

// ---- 预处理

/// 计算笔画信号（去背景 + 归一化），供比对前使用
/// 这是模块自带的预处理，宿主项目无需实现
cv::Mat strokeSignal(const cv::Mat& gray, const cv::Rect& roi, const Config& cfg, bool invert);

}  // namespace chardet
