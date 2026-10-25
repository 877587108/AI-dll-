// 多模板比对（方向 D）
//
// 核心思路：不再拿 84 张的平均频率图当唯一参考，而是保留若干张真实样例做模板库。
// 待检字符与「模板库中最近的那张」比对，差异越小说明越像某个正常样例。
//
// 为什么这能检出改笔画：
//   频率图是交集区域，比任何单张图都窄，边缘被平均化；
//   改一笔（0→Q）这种局部结构变化，在交集图上只剩下 2% 的像素差异，
//   与 OK 件的刻印波动同量级。而在真实模板上，它是「轮廓上多出一个突起」，
//   与最近模板仍然存在明显差异。
//
// 双指标：
//   maxDiff    —— 差异最大的那个字符（抓「某个字符刻错了」）
//   secondDiff —— 第二大的字符（排除单点噪声，只有真正孤立的一个缺陷才够格）

#include "chardet.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace chardet {

namespace {

/// 把一个字符的笔画信号裁出来，归一化到固定尺寸
cv::Mat makePatch(const cv::Mat& sig, const cv::Rect& box, int size) {
    const cv::Rect b = box & cv::Rect(0, 0, sig.cols, sig.rows);
    if (b.width <= 2 || b.height <= 2) return cv::Mat();

    cv::Mat patch;
    sig(b).convertTo(patch, CV_32F);

    // 只保留笔画核心：边缘的弱信号在不同样本间抖动很大，参与比较会污染结果
    cv::threshold(patch, patch, 0.25, 0.0, cv::THRESH_TOZERO);

    cv::Mat out;
    cv::resize(patch, out, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);

    // 幅值归一化，抵消整体亮度差异
    double mx = 0;
    cv::minMaxLoc(out, nullptr, &mx);
    if (mx > 1e-6) out *= (1.0 / mx);
    return out;
}

/// 两张 patch 的差异 = 1 - 余弦相似度
double patchDiff(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.empty()) return 1.0;
    const cv::Mat fa = a.reshape(1, 1);
    const cv::Mat fb = b.reshape(1, 1);
    const double na = std::sqrt(fa.dot(fa));
    const double nb = std::sqrt(fb.dot(fb));
    if (na <= 1e-9 || nb <= 1e-9) return 1.0;
    const double c = fa.dot(fb) / (na * nb);
    return 1.0 - std::clamp(c, -1.0, 1.0);
}

}  // namespace

std::vector<CharUnit> toAllUnits(const Model& m) {
    std::vector<CharUnit> out;
    for (size_t ri = 0; ri < m.regions.size(); ++ri) {
        auto u = toUnits(m, static_cast<int>(ri));
        for (auto& x : u) out.push_back(std::move(x));
    }
    return out;
}

MultiResult diffMultiTemplate(const cv::Mat& signal, const Model& model,
                              std::vector<CharUnit>& units) {
    MultiResult out;
    if (model.charTemplates.empty() || units.empty()) return out;
    if (signal.type() != CV_32F) return out;

    out.perChar.resize(units.size());
    std::vector<double> diffs;
    diffs.reserve(units.size());

    for (size_t i = 0; i < units.size(); ++i) {
        auto& u = units[i];
        if (i >= model.charTemplates.size()) break;
        const auto& tmpls = model.charTemplates[i];
        if (tmpls.empty()) continue;

        cv::Rect box = u.bbox & cv::Rect(0, 0, signal.cols, signal.rows);
        if (box.width <= 4 || box.height <= 4) continue;

        // ---- 阶段一：用频率图把框对齐到实际字形位置
        //
        // 为什么不能直接拿模型框裁 patch：
        //   刻印位置有 6~8px 的个体偏移。模板是精确对齐的，
        //   待检 patch 若偏了 7px，裁出来是「半个字形 + 一块空白」，
        //   余弦相似度会被无谓拉低 0.1~0.2，淹没改笔画本身那点差异。
        //
        // 为什么用频率图而不是真实模板来对齐：
        //   对齐只需要「大致位置」，频率图虽然细节被平均化，
        //   用来定位足够准（实测匹配峰值 0.9）；而真实模板正是要判别的对象，
        //   拿它来对齐会造成循环依赖。
        {
            constexpr int kS = 12;
            cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
            sr &= cv::Rect(0, 0, signal.cols, signal.rows);
            if (sr.width >= box.width && sr.height >= box.height) {
                cv::Mat mS, tS;
                model.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
                signal(sr).convertTo(tS, CV_32F);
                cv::Mat res;
                cv::matchTemplate(tS,
                                  mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)),
                                  res, cv::TM_CCOEFF_NORMED);
                cv::Point mL;
                double mv = 0;
                cv::minMaxLoc(res, nullptr, &mv, nullptr, &mL);
                if (mv >= 0.30) {
                    const cv::Rect cand(box.x + mL.x - kS, box.y + mL.y - kS, box.width,
                                        box.height);
                    if ((cand & cv::Rect(0, 0, signal.cols, signal.rows)) == cand) {
                        box = cand;
                        u.bbox = cand;   // 回写给调用方，后续判据用同一个位置
                    }
                }
            }
        }

        // ---- 阶段二：在已对齐的位置上裁 patch，与真实模板比对
        const cv::Mat patch = makePatch(signal, box, model.templateSize);
        if (patch.empty()) continue;

        double best = 1.0;
        for (const auto& t : tmpls) {
            best = std::min(best, patchDiff(patch, t));
        }

        out.perChar[i] = best;
        diffs.push_back(best);
    }

    if (diffs.empty()) return out;

    std::sort(diffs.begin(), diffs.end(), std::greater<double>());
    out.maxDiff = diffs[0];
    out.secondDiff = diffs.size() > 1 ? diffs[1] : 0.0;
    out.valid = true;
    return out;
}

}  // namespace chardet
