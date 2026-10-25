// plugin_orb.cpp —— ORB 特征点定位插件
//
// ================================================================
// 算法：ORB 特征点 + 单应矩阵
// ================================================================
//
// 问题：模型图（1688×1460）和待测图（如 unset 1496×1496）之间
//       不仅有缩放和平移，还有约 -5° 的旋转。
//       纯 NCC 模板匹配对旋转敏感，导致全部定位失败。
//
// 方案：
//   1. 用 ORB 在标准样张（original.png）和待测图上检测特征点
//   2. BFMatcher + RANSAC 求单应矩阵 H（吸收缩放、旋转、平移）
//   3. 用 H 把模型字符框映射到待测图坐标
//
// 前提：模型目录里需要一张"标准样张" original.png，
//       它与模型坐标一致（即字符框直接套在它上面）。
//       训练时从 image/ok 里选一张质量最好的原始图存入。
//
// 为什么用 ORB 而不是 SIFT：SIFT 有专利/需要 contrib，ORB 在
// 核心库且对旋转尺度鲁棒，适合芯片上的字符结构。
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "locate_plugin/locate_plugin.hpp"

namespace {

constexpr const char* kName = "orb";

struct Params {
    int nfeatures = 4000;      // ORB 特征点数
    double ratioThr = 0.75;    // ratio test 阈值（越低越严格，误匹配越少）
    double ransacThr = 3.0;    // RANSAC 重投影误差阈值(px)
    double minInliers = 12;    // 最少内点数，低于此判整图定位失败
    double minScale = 0.8;     // 单应矩阵尺度的合理下限
    double maxScale = 1.2;     // 上限
    int refine = 1;            // 是否做逐字符 NCC 精调（1 开启）
    int refineR = 12;          // 精调搜索半径(px)，粗定位框邻域 ±R
    double refineMinConf = 0.3;// 精调最低置信度，低于此不更新位置
};

Params g_params;

cv::Mat readGrayFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) return {};
    return cv::imdecode(cv::Mat(1, static_cast<int>(buf.size()), CV_8U, buf.data()),
                        cv::IMREAD_GRAYSCALE);
}

class OrbLocator {
public:
    bool init(const char* modelDir) {
        if (!modelDir) return false;
        const std::string dir(modelDir);
        // 标准样张：优先 original.png，退回到 ref.png / freq.png
        original_ = readGrayFile(dir + "/original.png");
        if (original_.empty()) original_ = readGrayFile(dir + "/model.png");
        if (original_.empty()) original_ = readGrayFile(dir + "/freq.png");
        if (original_.empty()) return false;

        // 预计算标准样张的 ORB 特征
        orb_ = cv::ORB::create(g_params.nfeatures);
        orb_->detectAndCompute(original_, cv::noArray(), kpRef_, desRef_);
        return !desRef_.empty();
    }

    void setParams(const LocateParam* ps, int n) {
        for (int i = 0; i < n; ++i) {
            const std::string k = ps[i].key ? ps[i].key : "";
            const double v = ps[i].value;
            if (k == "nfeatures") g_params.nfeatures = static_cast<int>(std::max(100.0, v));
            else if (k == "ratioThr") g_params.ratioThr = std::clamp(v, 0.3, 1.0);
            else if (k == "ransacThr") g_params.ransacThr = std::max(0.5, v);
            else if (k == "minInliers") g_params.minInliers = std::max(4.0, v);
            else if (k == "minScale") g_params.minScale = std::max(0.1, v);
            else if (k == "maxScale") g_params.maxScale = std::max(g_params.minScale, v);
            else if (k == "refine") g_params.refine = v > 0 ? 1 : 0;
            else if (k == "refineR") g_params.refineR = static_cast<int>(std::clamp(v, 2.0, 60.0));
            else if (k == "refineMinConf")
                g_params.refineMinConf = std::clamp(v, 0.0, 1.0);
        }
    }

    LocateResult detect(const cv::Mat* signal, int imageW, int imageH, int modelW, int modelH,
                        const TemplateBox* boxes, int count) {
        LocateResult r;
        chars_.clear();
        chars_.resize(static_cast<size_t>(std::max(0, count)));

        if (!signal || signal->empty() || signal->type() != CV_32F) {
            r.message = "signal 无效";
            return r;
        }
        if (desRef_.empty()) {
            r.message = "标准样张特征未就绪";
            return r;
        }

        // 信号转 8 位灰度用于特征检测
        cv::Mat gray;
        cv::normalize(*signal, gray, 0, 255, cv::NORM_MINMAX, CV_8U);

        // 待测图特征
        std::vector<cv::KeyPoint> kpTest;
        cv::Mat desTest;
        orb_->detectAndCompute(gray, cv::noArray(), kpTest, desTest);
        if (desTest.empty()) {
            r.message = "待测图无特征点";
            return r;
        }

        // 匹配：knnMatch + ratio test（滤除歧义匹配）
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desRef_, desTest, knn, 2);
        std::vector<cv::DMatch> matches;
        matches.reserve(knn.size());
        for (const auto& k : knn) {
            if (k.size() == 2 && k[0].distance < g_params.ratioThr * k[1].distance) {
                matches.push_back(k[0]);
            } else if (k.size() == 1) {
                matches.push_back(k[0]);
            }
        }
        std::sort(matches.begin(), matches.end(),
                  [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

        // 取前 200 个匹配求单应
        const int nMatch = std::min(200, static_cast<int>(matches.size()));
        if (nMatch < 8) {
            r.message = "匹配点不足";
            return r;
        }
        std::vector<cv::Point2f> pRef, pTest;
        pRef.reserve(static_cast<size_t>(nMatch));
        pTest.reserve(static_cast<size_t>(nMatch));
        for (int i = 0; i < nMatch; ++i) {
            pRef.push_back(kpRef_[static_cast<size_t>(matches[static_cast<size_t>(i)].queryIdx)].pt);
            pTest.push_back(
                kpTest[static_cast<size_t>(matches[static_cast<size_t>(i)].trainIdx)].pt);
        }

        cv::Mat mask;
        cv::Mat H = cv::findHomography(pRef, pTest, cv::RANSAC, g_params.ransacThr, mask);
        if (H.empty()) {
            r.message = "单应矩阵求解失败";
            return r;
        }
        const int inliers = mask.empty() ? 0 : cv::countNonZero(mask);
        if (inliers < g_params.minInliers) {
            r.message = "内点不足";
            return r;
        }

        // 校验尺度：对 H 左上 2×2 做 SVD，奇异值 = 两个轴的缩放
        // （不能直接 det，旋转会混入）
        cv::Mat M2 = H(cv::Rect(0, 0, 2, 2)).clone();
        cv::Mat w, u, vt;
        cv::SVD::compute(M2, w, u, vt);
        const double s1 = w.at<double>(0, 0);
        const double s2 = w.at<double>(1, 0);
        if (s1 < g_params.minScale || s1 > g_params.maxScale || s2 < g_params.minScale ||
            s2 > g_params.maxScale) {
            r.message = "尺度异常";
            return r;
        }

        // 提取缩放和偏移（用于 LocateResult 的 scale/offset 字段）
        // 用 SVD 奇异值作为真缩放（H 对角线元素在有旋转时 ≠ 纯缩放，
        // 会混入旋转耦合，导致报告的 scale 虚高/虚低）。
        r.scaleX = static_cast<float>(s1);
        r.scaleY = static_cast<float>(s2);
        r.offsetX = static_cast<float>(H.at<double>(0, 2));
        r.offsetY = static_cast<float>(H.at<double>(1, 2));

        // 用 H 映射每个字符框
        int accepted = 0;
        for (int i = 0; i < count; ++i) {
            const TemplateBox& tb = boxes[i];
            LocatedChar& lc = chars_[static_cast<size_t>(i)];
            lc.index = tb.index;

            // 框的四个角经 H 变换
            const double cx[4] = {tb.x, tb.x + tb.w, tb.x + tb.w, tb.x};
            const double cy[4] = {tb.y, tb.y, tb.y + tb.h, tb.y + tb.h};
            double minX = 1e18, minY = 1e18, maxX = -1e18, maxY = -1e18;
            for (int c = 0; c < 4; ++c) {
                const double w = H.at<double>(2, 0) * cx[c] + H.at<double>(2, 1) * cy[c] +
                                 H.at<double>(2, 2);
                const double px = (H.at<double>(0, 0) * cx[c] + H.at<double>(0, 1) * cy[c] +
                                   H.at<double>(0, 2)) / w;
                const double py = (H.at<double>(1, 0) * cx[c] + H.at<double>(1, 1) * cy[c] +
                                   H.at<double>(1, 2)) / w;
                minX = std::min(minX, px);
                maxX = std::max(maxX, px);
                minY = std::min(minY, py);
                maxY = std::max(maxY, py);
            }
            lc.x = static_cast<int>(std::lround(minX));
            lc.y = static_cast<int>(std::lround(minY));
            lc.w = static_cast<int>(std::lround(maxX - minX));
            lc.h = static_cast<int>(std::lround(maxY - minY));

            // 框是否在有效范围内
            const bool inBounds = lc.x >= 0 && lc.y >= 0 && lc.x + lc.w <= imageW &&
                                  lc.y + lc.h <= imageH && lc.w > 3 && lc.h > 3;
            lc.confidence = inliers > 0 ? static_cast<float>(inliers) / static_cast<float>(nMatch)
                                        : 0.0f;
            lc.accepted = inBounds;
            if (lc.accepted) ++accepted;
        }

        // ---- 逐字符 NCC 精调（可选）
        // ORB 用整图单应矩阵定位到"整体正确"，但对单个字符可能有
        // 几到几十像素的残差（单应矩阵是全局近似，局部有非线性）。
        // 这里对每个字符框，用标准样张的字符区域做模板，在粗定位
        // 邻域内做 NCC 搜索，把框精调到亚像素级、更紧贴字符。
        if (g_params.refine && !original_.empty()) {
            const int sr = g_params.refineR;
            const double refScale = std::clamp(s1, 0.5, 2.0);
            for (int i = 0; i < count; ++i) {
                LocatedChar& lc = chars_[static_cast<size_t>(i)];
                if (!lc.accepted) continue;
                const TemplateBox& tb = boxes[i];

                // 模板：标准样张的模型字符区域
                cv::Rect tmplRect(tb.x, tb.y, tb.w, tb.h);
                tmplRect &= cv::Rect(0, 0, original_.cols, original_.rows);
                if (tmplRect.width < 5 || tmplRect.height < 5) continue;
                cv::Mat tmpl;
                cv::resize(original_(tmplRect), tmpl, cv::Size(lc.w, lc.h), 0, 0,
                           cv::INTER_LINEAR);

                // 搜索邻域：粗定位框 ±sr
                cv::Rect search(lc.x - sr, lc.y - sr, lc.w + 2 * sr, lc.h + 2 * sr);
                search &= cv::Rect(0, 0, gray.cols, gray.rows);
                if (search.width <= tmpl.cols || search.height <= tmpl.rows) continue;

                cv::Mat res;
                cv::matchTemplate(gray(search), tmpl, res, cv::TM_CCOEFF_NORMED);
                double mval = 0;
                cv::Point mloc;
                cv::minMaxLoc(res, nullptr, &mval, nullptr, &mloc);
                if (mval > g_params.refineMinConf) {
                    lc.x = search.x + mloc.x;
                    lc.y = search.y + mloc.y;
                    lc.confidence = static_cast<float>(mval);
                }
            }
        }

        r.chars = chars_.data();
        r.charCount = static_cast<int>(chars_.size());
        r.globalOk = count > 0 && accepted >= count * 8 / 10;
        return r;
    }

    void release(LocateResult* r) {
        if (r) r->chars = nullptr;
    }

private:
    cv::Mat original_;
    cv::Ptr<cv::ORB> orb_;
    std::vector<cv::KeyPoint> kpRef_;
    cv::Mat desRef_;
    std::vector<LocatedChar> chars_;
};

OrbLocator g_locator;
LocateResult g_result;
std::string g_paramHelp[10];
const char* g_paramPtrs[10];
int g_paramCount = 0;
locate_api_t g_api{};

bool apiInit(const char* modelDir) { return g_locator.init(modelDir); }
void apiSetParams(const LocateParam* p, int n) { g_locator.setParams(p, n); }

const char** apiListParams(int* count) {
    g_paramHelp[0] = "nfeatures=4000   ORB 特征点数";
    g_paramHelp[1] = "ratioThr=0.75    ratio test 阈值（越低越严格）";
    g_paramHelp[2] = "ransacThr=3.0    RANSAC 重投影误差阈值(px)";
    g_paramHelp[3] = "minInliers=12    最少内点数，低于此判失败";
    g_paramHelp[4] = "minScale=0.8     单应尺度下限";
    g_paramHelp[5] = "maxScale=1.2     单应尺度上限";
    g_paramHelp[6] = "refine=1         逐字符NCC精调（1开0关）";
    g_paramHelp[7] = "refineR=12       精调搜索半径(px)";
    g_paramHelp[8] = "refineMinConf=0.3 精调最低置信度";
    g_paramCount = 9;
    for (int i = 0; i < g_paramCount; ++i) g_paramPtrs[i] = g_paramHelp[i].c_str();
    if (count) *count = g_paramCount;
    return g_paramPtrs;
}

LocateResult apiDetect(const void* signal, int imageW, int imageH, int modelW, int modelH,
                       const TemplateBox* boxes, int count) {
    try {
        g_result = g_locator.detect(static_cast<const cv::Mat*>(signal), imageW, imageH, modelW,
                                    modelH, boxes, count);
    } catch (const std::exception&) {
        g_result = LocateResult{};
        g_result.message = "插件异常";
    } catch (...) {
        g_result = LocateResult{};
        g_result.message = "插件未知异常";
    }
    return g_result;
}

void apiRelease(LocateResult* r) { g_locator.release(r); }
void apiShutdown() {}

}  // namespace

LOCATE_EXPORT locate_api_t* locate_create() {
    g_api.apiVersion = LOCATE_PLUGIN_API_VERSION;
    g_api.name = kName;
    g_api.init = &apiInit;
    g_api.setParams = &apiSetParams;
    g_api.listParams = &apiListParams;
    g_api.detect = &apiDetect;
    g_api.release = &apiRelease;
    g_api.shutdown = &apiShutdown;
    return &g_api;
}

LOCATE_EXPORT void locate_destroy(locate_api_t*) {}
