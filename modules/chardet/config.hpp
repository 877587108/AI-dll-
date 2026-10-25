#pragma once

// chardet 模块配置
//
// 独立于宿主项目，只包含字符级检测需要的参数。

namespace chardet {

/// 字符级检测的可调参数
struct Config {
    // ---- 信号预处理
    double sigma = 15.0;         // 去背景高斯尺度

    // ---- 笔画阈值
    double strokeHi = 0.35;      // 强笔画阈值：当前图"有墨"的判定
    double missStrokeLo = 0.15;  // 低于此信号视为"墨缺失"

    // ---- 模板频率门槛
    double missFreq = 0.50;      // "应有墨"的频率门槛
    double blankFreq = 0.15;     // "应为空白"的频率上限（查脏污/划痕用）

    // ---- 判据一：少墨（笔画缺失）
    double charMissRatio = 0.15;        // 单个笔画段缺失超过此比例 → 少墨
    double charMissingInkRatio = 0.10;  // 整字缺失：框内墨量低于此比例
    int missMinBlobArea = 12;           // 只检查面积 >= 此值的笔画段

    // ---- 判据二：脏污（局部多余墨）
    double charExtraRatio = 0.12;  // 多余墨占应有笔画的比例门槛
    int dirtMinArea = 15;          // 脏污连通块面积下限
    int dirtMaxArea = 400;         // 脏污面积上限（更大的可能是划痕）

    // ---- 判据三：划痕（长条状多余墨）
    int scratchMinLen = 40;        // 划痕最小长度（px）
    double scratchMinElong = 3.0;  // 长宽比门槛（长/宽）

    // ---- 归一化
    double charT = 1.0;            // 分数归一化分母

    // ---- 切分参数
    double colThrRatio = 0.03;    // 列投影阈值（占峰值比例）
    double gapMergeRatio = 0.10;  // 间隙合并比例（占中位字宽）
    double binFreq = 0.25;        // 频率图二值化门槛
    double minHeightRatio = 0.08; // 最小字符高度（占区域高度比例）
    double rowThrRatio = 0.10;    // 行投影阈值（占峰值比例，用于自动分带）
};

}  // namespace chardet
