#pragma once

// 字符区缺陷检测核心算法（GUI 与命令行共用）
// 六个打光视角，每个视角有独立的模型（模板 + 统计量）与独立参数
//
// 单视角流程：去背景归一化 → 笔画信号 → 与正常笔画频率模板比对
//             → 多余笔画 / 连续划痕 / 缺墨 / 字符级 四个分支评分

#include <opencv2/core.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "chardet.hpp"

namespace markloc::charinsp {

constexpr int kViewCount = 6;
constexpr double kNgThreshold = 1.0;   // 分数 >= 此值判 NG

/// 四个区域的名称（下标与 Params::rois 对应）
inline constexpr const char* kRoiNames[4] = {"logo ", "line1", "line2", "code "};

/// 检测区域定义
struct Roi {
    cv::Rect rect;
    std::string name;
    bool checkAppendage = false;
    // 该区域是否参与本视角的判定。某些视角下个别区域会被背景梯度/噪点主导，
    // 此时按视角关闭该区域比强行调参可靠。
    bool enabled = true;
};

/// 单个视角的可调参数
struct Params {
    // 信号
    double sigma = 15.0;
    double strokeHi = 0.35;
    double strokeLo = 0.20;
    // 模板
    double pKeep = 0.20;
    int dilateK = 9;
    // 边缘屏蔽
    int edgeTB = 25;
    int edgeLR = 10;
    // 评分阈值
    double scratchT = 22000.0;
    double appendageT = 210.0;
    double eccMin = 0.60;
    // 划痕约束
    double scratchMinCoverage = 0.25;
    double scratchMinLen = 160.0;
    // 极性：true 表示该视角下刻印比背景暗（需反转笔画信号）
    bool invert = false;
    // ---- 缺墨（断线/漏印）检测
    // 判据：该像素在正常样本中"几乎必然有笔画"（频率 >= missFreq），
    //       但当前图像此处笔画信号很弱（S < missStrokeLo）。
    // 与多余笔画判据方向相反，正好互补。
    bool checkMissing = true;      // 是否检测缺墨
    double missFreq = 0.50;        // "必然有墨"的频率门槛（0~1）
    double missStrokeLo = 0.15;    // 低于此信号视为"墨缺失"
    int missMinArea = 30;          // 缺墨连通域最小面积（px）
    double missT = 900.0;          // 缺墨归一化分母
    // ---- 字符级比对
    // 型号行/批号行内容固定，可切出单字符逐个比对，定位到"第几个字有问题"。
    bool checkChars = true;        // 是否启用字符级比对
    // 分组预期判据（实验性）。实测 84 张 OK 有 39.3% 误报，
    // 因为划痕/脏污落在字符框内时与笔画缺陷无法区分。默认关闭。
    bool useCharNcc = false;
    double charMissRatio = 0.15;   // 某字符内缺墨像素占其笔画像素的比例门槛
    double charExtraRatio = 0.12;  // 某字符内多余墨像素的比例门槛
    double charT = 1.0;            // 字符分归一化分母
    // 是否参与产品判定（未标定的视角可关闭）
    bool enabled = true;
    // 区域
    std::array<Roi, 4> rois;

    Params();
};

/// 单区域模型
struct ModelRoi {
    cv::Mat sRef;
    cv::Mat freq;
    cv::Mat tolMask;
    bool ok = false;
};

/// 单个视角的模型
struct ViewModel {
    std::array<ModelRoi, 4> rois;
    /// 整图频率图（覆盖整幅芯片）
    cv::Mat fullFreq;      // CV_8U，与输入图同尺寸
    cv::Mat fullRef;       // CV_8U，整图配准基准
    bool fullOk = false;

    /// 字符级模型（由 chardet 模块自包含管理）
    std::shared_ptr<chardet::Model> chardetModel;
    bool hasManualBoxes = false;
    bool hasManualRegions = false;
    bool ok = false;
};

/// 六视角模型集合
struct Model {
    std::array<ViewModel, kViewCount> views;
    std::array<Params, kViewCount> params;   // 每视角独立参数（随模型保存）

    /// 布局：dir/v00/{logo ,line1,line2,code }_ref|freq|tol.png
    bool loadFrom(const std::string& dir, std::string* err = nullptr);
    bool valid(int view) const;
    bool anyValid() const;
};

/// 中间证据（可视化）
struct RoiEvidence {
    cv::Mat aligned8;
    cv::Mat s8;
    cv::Mat tol8;
    cv::Mat extra8;
    cv::Mat weak8;
    bool valid = false;
};

/// 单区域结果
struct RoiResult {
    double ecc = -1.0;
    double scratch = 0.0;
    double scratchCoverage = 0.0;
    double scratchLen = 0.0;
    double scratchMeanS = 0.0;
    double scratchSevere = 0.0;
    double appendage = 0.0;
    double missing = 0.0;         // 缺墨原始量（面积 × 缺失深度）
    double missingArea = 0.0;     // 缺墨连通域面积
    double worstChar = 0.0;       // 字符级比对中最严重的单字得分
    int worstCharIndex = -1;      // 该字符的序号（从左往右，从 0 开始）
    double score = 0.0;
    bool reliable = false;
};

/// 单个字符的检测结果（供界面逐字显示位置与分数）
struct CharResult {
    int roiIndex = -1;        // 属于哪个区域（0=logo 1=line1 2=line2）
    int charIndex = -1;       // 该区域内从左往右的序号
    cv::Rect rect;            // 字符框（整图坐标；公共 DTO 会明确坐标系）
    double score = 0.0;       // 该字符的综合分
    double missScore = 0.0;   // 缺墨分
    double extraScore = 0.0;  // 多墨分
    double shapeScore = 0.0;  // 字形变化分
    double missRatio = 0.0;
    double extraRatio = 0.0;
    std::string label;        // 异常说明
};

/// 单视角结果
struct Result {
    std::array<RoiResult, 4> regions;
    double regionMax = 0.0;      // 四区域最高分
    int worstRegion = -1;        // 最高分来自哪个区域
    double score = 0.0;
    bool ng = false;
    /// 整行墨块比对：多出/缺失的字符块数（最可靠的判据，40 张 OK 零误报）
    int blockExtra = 0;
    int blockMissing = 0;
    std::array<RoiEvidence, 4> evidence;
    /// 逐字符明细（仅前三个区域）
    std::vector<CharResult> chars;
    /// 字符数量校验（每个区域一条）
    struct CountInfo {
        int roiIndex = -1;
        int expected = 0;
        int actual = 0;
        bool matched = true;
        double score = 0.0;
        std::string detail;
    };
    std::vector<CountInfo> charCounts;
};

/// 一颗产品的结果（最多 6 视角）
struct ProductResult {
    std::array<Result, kViewCount> views;
    std::array<bool, kViewCount> present{};
    std::array<double, kViewCount> viewScore{};
    double score = 0.0;
    double regionScore = 0.0;      // 各视角四区域最高分
    int worstView = -1;
    bool ng = false;
};

/// 去背景并归一化，返回笔画信号 S（CV_32F）
cv::Mat strokeSignal(const cv::Mat& gray, const cv::Rect& roi, const Params& p, double* q94 = nullptr);

/// 单视角检测
Result analyze(const cv::Mat& gray, const ViewModel& vm, const Params& p, bool wantEvidence = true);

/// 整颗产品：逐视角独立计算，产品分 = 各视角最大值
ProductResult analyzeProduct(const std::array<cv::Mat, kViewCount>& images,
                             const std::array<bool, kViewCount>& present, const Model& m,
                             const std::array<Params, kViewCount>& ps, bool wantEvidence = true);

/// 生成四联对比图（BGR）
cv::Mat makeExplain(const cv::Mat& gray, const Params& p, const Result& r, const std::string& title);

/// 建模：按视角分别统计正常笔画频率
bool buildModel(const std::array<std::vector<cv::Mat>, kViewCount>& okByView,
                std::array<Params, kViewCount>& ps, Model& out, std::string* err = nullptr);

/// 自动标定：以视角0为基准，用多尺度模板匹配估计其余视角的 ROI（缩放+平移）
bool autoCalibrateRois(const std::array<std::vector<cv::Mat>, kViewCount>& okByView,
                       std::array<Params, kViewCount>& ps, std::string* log = nullptr);

/// 保存单个视角模型到 dir/vXX/
bool saveViewModel(const std::string& dir, int view, const ViewModel& vm, std::string* err = nullptr);

/// 保存/读取单个视角参数到 dir/vXX/params.txt
bool saveParams(const std::string& dir, int view, const Params& p);
bool loadParams(const std::string& dir, int view, Params& p);

/// 把当前区域定义导出成用户可编辑的 rois.txt（带注释说明）
/// 已存在则覆盖。返回是否成功。
bool exportRoisFile(const std::string& dir, int view, const Params& p);

}  // namespace markloc::charinsp
