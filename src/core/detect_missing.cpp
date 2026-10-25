// 缺墨（断线 / 漏印）检测
//
// 与其它三个分支方向相反：它们找"多出来的墨"，这里找"本该有却缺失的墨"。
//
// 判据：
//   freq 图记录了每个像素在正常样本中"有笔画"的概率。
//   · freq >= missFreq       → 该位置几乎必然有笔画（"必然有墨区"）
//   · 当前 S < missStrokeLo  → 此处笔画信号很弱（墨缺失）
//   两者同时成立 → 计为缺墨像素
//
// 为什么这样可行：
//   "必然有墨区"是通过统计大量 OK 样本得到的，字符主体的核心像素频率接近 1.0。
//   若某颗产品漏印/断线，这些核心像素会突然变空——这是模板比对做不到的，
//   因为模板比对比的是"内容"，而这里比的是"有没有墨"。

#include "core/detectors.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace markloc::charinsp::det {

double scoreMissing(const cv::Mat& aligned, const cv::Mat& freq, const Params& p,
                    RoiResult& rr) {
    if (aligned.empty() || freq.empty()) return 0.0;
    if (aligned.size() != freq.size()) return 0.0;

    // ① 必然有墨区：freq >= missFreq
    cv::Mat freqF;
    freq.convertTo(freqF, CV_32F, 1.0 / 255.0);
    cv::Mat mustInk;
    cv::compare(freqF, p.missFreq, mustInk, cv::CMP_GE);

    // ② 墨缺失区：S < missStrokeLo
    cv::Mat gone;
    cv::compare(aligned, p.missStrokeLo, gone, cv::CMP_LT);

    // ③ 交集：本该有墨，但现在没有
    cv::Mat missPx;
    cv::bitwise_and(mustInk, gone, missPx);

    const int n = cv::countNonZero(missPx);
    if (n < p.missMinArea) {
        rr.missing = 0.0;
        rr.missingArea = 0.0;
        return 0.0;
    }

    // ④ 连通域分析：零散的缺失像素（噪声）不算，需要成块
    cv::Mat miss8;
    missPx.convertTo(miss8, CV_8U, 255.0);
    // 轻微闭运算把断线两端连起来，避免被噪声切碎
    const cv::Mat ck = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::morphologyEx(miss8, miss8, cv::MORPH_CLOSE, ck);

    cv::Mat labels, stats, cents;
    const int cnt = cv::connectedComponentsWithStats(miss8, labels, stats, cents, 8, CV_32S);

    double best = 0.0;
    double bestArea = 0.0;
    for (int k = 1; k < cnt; ++k) {
        const int area = stats.at<int>(k, cv::CC_STAT_AREA);
        if (area < p.missMinArea) continue;

        // ⑤ 缺失深度：该连通域内 S 离 missStrokeLo 差多少
        //    深 = 墨完全没了；浅 = 只是变淡，不算严重
        double depthSum = 0.0;
        int px = 0;
        const int left = stats.at<int>(k, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(k, cv::CC_STAT_TOP);
        const int wid = stats.at<int>(k, cv::CC_STAT_WIDTH);
        const int hei = stats.at<int>(k, cv::CC_STAT_HEIGHT);
        for (int y = top; y < top + hei; ++y) {
            const int* lrow = labels.ptr<int>(y);
            const float* srow = aligned.ptr<float>(y);
            const float* frow = freqF.ptr<float>(y);
            for (int x = left; x < left + wid; ++x) {
                if (lrow[x] != k) continue;
                // 只统计真属于"必然有墨"的像素
                if (frow[x] < p.missFreq) continue;
                depthSum += std::max(0.0, p.missStrokeLo - srow[x]);
                ++px;
            }
        }
        if (px <= 0) continue;

        const double meanDepth = depthSum / px;
        // 得分 = 缺失面积 × 平均缺失深度
        // 面积大但没有完全缺失（只是变淡）得分低；
        // 面积小但完全消失，也能得到可观分数。
        const double score = static_cast<double>(px) * meanDepth;
        if (score > best) {
            best = score;
            bestArea = static_cast<double>(px);
        }
    }

    rr.missing = best;
    rr.missingArea = bestArea;
    return best / std::max(1.0, p.missT);
}

}  // namespace markloc::charinsp::det
