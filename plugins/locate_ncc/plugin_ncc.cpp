// plugin_ncc.cpp —— 参考定位插件：归一化互相关（NCC）模板匹配
//
// ================================================================
// 这是基线实现，也是现有 markloc 内部用的方法
// ================================================================
//
// 算法：
//   对每个模板字符框：
//     1. 从模型频率图裁出该字符作为模板
//     2. 按整图尺度把框和搜索半径换算到待测图
//     3. 在换算后的框周围 ±radius 内做 TM_CCOEFF_NORMED
//     4. 取峰值，做亚像素插值
//     5. 峰值在搜索边界上 → 拒绝（无法确定完整位置）
//     6. 置信度低于门槛 → 拒绝
//
// 为什么用 NCC 而不是平方差：NCC 对线性的亮度变化免疫。
// 同一颗芯片不同曝光下，平方差会随曝光漂移，NCC 不会。
//
// ================================================================
// 已知局限（写在这里是为了让对比实验有据可依）
// ================================================================
//
//   · 对旋转敏感：旋转超过约 2° 就开始掉分
//   · 对尺度敏感：整图缩放估计稍错，所有字都偏
//   · 笔画细的字符（如 "1"、"l"）NCC 峰值平坦，容易定偏
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/locate_plugin/locate_plugin.hpp"

namespace {

/// 读灰度图
///
/// 不能用 cv::imread：它在 Windows 上按 ANSI 代码页解释路径，
/// 中文路径一律失败。这里先用标准库按字节读进来再 imdecode。
cv::Mat readGrayFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    if (buf.empty()) return {};
    return cv::imdecode(cv::Mat(1, static_cast<int>(buf.size()), CV_8U, buf.data()),
                        cv::IMREAD_GRAYSCALE);
}

// ---- 默认参数
struct Params {
    double radius = 24.0;      // 搜索半径（模型像素）
    double minConf = 0.50;     // 最低置信度
    double edgeReject = 1.0;   // 峰值离搜索边界小于此像素数就拒绝
    double smooth = 0.0;       // 搜索前的高斯平滑 sigma，0=不做
};

Params g_params;
std::string g_paramHelp[6];
const char* g_paramPtrs[6];
std::vector<LocatedChar> g_chars;

}  // namespace

class NccLocator {
public:
    bool init(const char* modelDir) {
        if (!modelDir) return false;
        const std::string dir(modelDir);
        modelFreq_ = readGrayFile(dir + "/freq.png");
        if (modelFreq_.empty()) modelFreq_ = readGrayFile(dir + "\\freq.png");
        return !modelFreq_.empty();
    }

    void setParams(const LocateParam* ps, int n) {
        for (int i = 0; i < n; ++i) {
            const std::string k = ps[i].key ? ps[i].key : "";
            const double v = ps[i].value;
            if (k == "radius") g_params.radius = std::max(2.0, v);
            else if (k == "minConf") g_params.minConf = std::clamp(v, 0.0, 1.0);
            else if (k == "edgeReject") g_params.edgeReject = std::max(0.0, v);
            else if (k == "smooth") g_params.smooth = std::max(0.0, v);
        }
    }

    LocateResult detect(const cv::Mat* signal, int imageW, int imageH, int modelW, int modelH,
                        const TemplateBox* boxes, int count) {
        LocateResult r;
        g_chars.clear();
        g_chars.resize(static_cast<size_t>(std::max(0, count)));

        if (!signal || signal->empty() || signal->type() != CV_32F) {
            r.message = "signal 无效";
            return r;
        }
        if (modelFreq_.empty()) {
            r.message = "模型频率图未加载";
            return r;
        }

        // 整图尺度：两图尺寸可能不同
        const double sx = modelW > 0 ? static_cast<double>(imageW) / modelW : 1.0;
        const double sy = modelH > 0 ? static_cast<double>(imageH) / modelH : 1.0;
        r.scaleX = static_cast<float>(sx);
        r.scaleY = static_cast<float>(sy);

        cv::Mat work = *signal;
        if (g_params.smooth > 0.05) {
            cv::Mat tmp;
            cv::GaussianBlur(work, tmp, cv::Size(0, 0), g_params.smooth);
            work = tmp;
        }

        const cv::Rect bounds(0, 0, work.cols, work.rows);
        int accepted = 0;

        for (int i = 0; i < count; ++i) {
            const TemplateBox& tb = boxes[i];
            LocatedChar& lc = g_chars[static_cast<size_t>(i)];
            lc.index = tb.index;

            // 模板框 → 待测图坐标
            cv::Rect box(static_cast<int>(std::lround(tb.x * sx)),
                         static_cast<int>(std::lround(tb.y * sy)),
                         std::max(2, static_cast<int>(std::lround(tb.w * sx))),
                         std::max(2, static_cast<int>(std::lround(tb.h * sy))));
            lc.x = box.x;
            lc.y = box.y;
            lc.w = box.width;
            lc.h = box.height;
            lc.accepted = false;
            lc.confidence = 0.0f;

            // 模板：从模型频率图对应位置裁出
            const cv::Rect tbox(tb.x, tb.y, tb.w, tb.h);
            if ((tbox & cv::Rect(0, 0, modelFreq_.cols, modelFreq_.rows)) != tbox) continue;
            if (tbox.width <= 3 || tbox.height <= 3) continue;

            cv::Mat templ;
            modelFreq_(tbox).convertTo(templ, CV_32F, 1.0 / 255.0);
            // 缩放到待测图尺度
            if (std::abs(sx - 1.0) > 0.01 || std::abs(sy - 1.0) > 0.01) {
                cv::resize(templ, templ, box.size(), 0, 0, cv::INTER_LINEAR);
            }
            cv::Scalar mean, stddev;
            cv::meanStdDev(templ, mean, stddev);
            if (stddev[0] < 1e-5) continue;   // 纯色模板，无法匹配

            // 搜索区：框外扩 radius（按尺度换算）
            const int rad = static_cast<int>(std::lround(g_params.radius * 0.5 * (sx + sy)));
            cv::Rect search(box.x - rad, box.y - rad, box.width + 2 * rad, box.height + 2 * rad);
            search &= bounds;
            if (search.width <= templ.cols || search.height <= templ.rows) continue;

            cv::Mat response;
            cv::matchTemplate(work(search), templ, response, cv::TM_CCOEFF_NORMED);
            cv::Point peak;
            double mval = 0;
            cv::minMaxLoc(response, nullptr, &mval, nullptr, &peak);
            if (!std::isfinite(mval)) continue;

            const bool onEdge = peak.x < g_params.edgeReject ||
                                peak.y < g_params.edgeReject ||
                                peak.x + 1 >= response.cols - g_params.edgeReject ||
                                peak.y + 1 >= response.rows - g_params.edgeReject;

            // 亚像素插值（抛物线拟合）
            const auto subpixel = [](double l, double c, double rr) {
                const double den = 2.0 * c - l - rr;
                return den > 1e-9 ? std::clamp(0.5 * (rr - l) / den, -0.5, 0.5) : 0.0;
            };
            double subX = 0, subY = 0;
            if (peak.x > 0 && peak.x + 1 < response.cols) {
                subX = subpixel(response.at<float>(peak.y, peak.x - 1), response.at<float>(peak.y, peak.x),
                                response.at<float>(peak.y, peak.x + 1));
            }
            if (peak.y > 0 && peak.y + 1 < response.rows) {
                subY = subpixel(response.at<float>(peak.y - 1, peak.x), response.at<float>(peak.y, peak.x),
                                response.at<float>(peak.y + 1, peak.x));
            }

            lc.confidence = static_cast<float>(std::clamp(mval, 0.0, 1.0));
            lc.x = static_cast<int>(std::lround(search.x + peak.x + subX));
            lc.y = static_cast<int>(std::lround(search.y + peak.y + subY));
            lc.w = templ.cols;
            lc.h = templ.rows;
            lc.accepted = !onEdge && mval >= g_params.minConf;
            if (lc.accepted) ++accepted;
        }

        r.scaleX = static_cast<float>(sx);
        r.scaleY = static_cast<float>(sy);
        r.chars = g_chars.data();
        r.charCount = static_cast<int>(g_chars.size());
        r.globalOk = count > 0 && accepted >= count * 8 / 10;
        return r;
    }

    void release(LocateResult* r) {
        if (r) r->chars = nullptr;
    }

    const cv::Mat& modelFreq() const { return modelFreq_; }

private:
    cv::Mat modelFreq_;
};

// ---------------------------------------------------------------- C 导出

namespace {
NccLocator g_locator;
LocateResult g_result;

bool apiInit(const char* modelDir) {
    return g_locator.init(modelDir);
}

void apiSetParams(const LocateParam* p, int n) {
    g_locator.setParams(p, n);
}

const char** apiListParams(int* count) {
    g_paramHelp[0] = "radius=24.0      搜索半径（模型像素），越大越能容错但越慢";
    g_paramHelp[1] = "minConf=0.50     最低 NCC 置信度，低于则判该字定位失败";
    g_paramHelp[2] = "edgeReject=1.0   峰值距搜索边界小于此值则拒绝（防止定位到边缘）";
    g_paramHelp[3] = "smooth=0.0       匹配前的高斯平滑 sigma，0=不做";
    for (int i = 0; i < 4; ++i) g_paramPtrs[i] = g_paramHelp[i].c_str();
    if (count) *count = 4;
    return g_paramPtrs;
}

LocateResult apiDetect(const void* signal, int imageW, int imageH, int modelW, int modelH,
                       const TemplateBox* boxes, int count) {
    try {
        const cv::Mat* sig = static_cast<const cv::Mat*>(signal);
        g_result = g_locator.detect(sig, imageW, imageH, modelW, modelH, boxes, count);
    } catch (const std::exception& e) {
        g_result = LocateResult{};
        g_result.message = "插件异常";
    } catch (...) {
        g_result = LocateResult{};
        g_result.message = "插件未知异常";
    }
    return g_result;
}

void apiRelease(LocateResult* r) {
    g_locator.release(r);
}

void apiShutdown() {}

const char* kName = "ncc";

locate_api_t g_api{};

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
