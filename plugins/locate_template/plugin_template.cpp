// plugin_template.cpp —— 定位算法插件模板
//
// ================================================================
// 这是 AI 开发定位算法插件的起点。
// 复制本文件 → 改名 → 实现 detect() → 编译成 DLL → 加载评估。
//
// 你（AI）只需要做三件事：
//   1. 改 kName 为你的算法名
//   2. 实现 NccLocator::detect() 里的算法逻辑
//   3. 编译：build_plugin.bat <你的.cpp> <输出.dll>
//
// 其余（导出函数、参数、内存管理、异常兜底）都已在模板里写好，
// 不要改它们。
// ================================================================
//
// 【接口契约速览】（完整定义见 include/locate_plugin/locate_plugin.hpp）
//
//   输入（detect 的参数）：
//     signal       待测图的笔画信号（CV_32F，整图尺寸）
//     imageW/H     待测图尺寸（像素）
//     modelW/H     模型图尺寸（像素）
//     templateBoxes 模板字符框（模型坐标），共 boxCount 个
//
//   输出（LocateResult）：
//     scaleX/scaleY/offsetX/offsetY  整图仿射（检测图 = 模型 × scale + offset）
//     chars[]   每个字符的定位框（待测图坐标）+ 置信度 + accepted
//     globalOk  整图是否可信
//
// 【坐标系】chars[i].x/y/w/h 必须是待测图坐标（原图像素）。
// 【顺序】chars[i] 必须与 templateBoxes[i] 一一对应。
// 【生命周期】chars 指向的内存必须活到下一次 detect() 或 release()。
//            模板里用静态 vector 保证这一点。
// ================================================================
//
// 【评估指标】（由 locate_bench 计算，你不需要算）
//   定位成功率 = accepted 的字符数 / 总字符数
//   偏差 dx/dy = chars[i].x - templateBoxes[i].x × scale
//   评估用"与模板框按整图尺度换算位置的偏差"做参照，不需要人工真值。
// ================================================================

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

#include "locate_plugin/locate_plugin.hpp"

namespace {

// ================================================================
// ① 你的算法名（显示用，改成有意义的名字）
// ================================================================
constexpr const char* kName = "template";

// ================================================================
// ② 你的可调参数（默认值 + 说明）
// 这些参数会出现在 locate_bench --list-params 里，
// 用户/评估脚本通过 --param key=value 覆盖。
// ================================================================
struct Params {
    // 示例：搜索半径（模型像素）
    double radius = 24.0;
    // 示例：最低置信度
    double minConf = 0.50;
    // 在这里加你自己的参数...
};

Params g_params;

// ================================================================
// 工具：读灰度图（不能用 cv::imread，中文路径会失败）
// ================================================================
cv::Mat readGrayFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) return {};
    return cv::imdecode(cv::Mat(1, static_cast<int>(buf.size()), CV_8U, buf.data()),
                        cv::IMREAD_GRAYSCALE);
}

// ================================================================
// ③ 你的定位算法实现
// ================================================================
class Locator {
public:
    bool init(const char* modelDir) {
        if (!modelDir) return false;
        const std::string dir(modelDir);
        // 读模型频率图（模板来源）
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
            // 在这里解析你自己的参数...
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
        if (modelFreq_.empty()) {
            r.message = "模型频率图未加载";
            return r;
        }

        // 整图尺度：两图尺寸可能不同，用图幅比例做初始估计
        const double sx = modelW > 0 ? static_cast<double>(imageW) / modelW : 1.0;
        const double sy = modelH > 0 ? static_cast<double>(imageH) / modelH : 1.0;
        r.scaleX = static_cast<float>(sx);
        r.scaleY = static_cast<float>(sy);
        r.offsetX = 0.0f;
        r.offsetY = 0.0f;

        // ============================================================
        // ④ 在这里写你的定位算法
        //
        // 目标：对每个 templateBoxes[i]，算出它在待测图上的实际位置，
        //       填入 chars_[i].x/y/w/h，并设置 accepted 和 confidence。
        //
        // 下面的参考实现是 NCC 模板匹配，你可以替换成：
        //   · 特征点匹配（抗旋转）
        //   · 相位相关（整体平移）
        //   · 光流（局部形变）
        //   · 深度学习特征
        //   · 尺度自适应的多级匹配
        // ============================================================
        const cv::Rect bounds(0, 0, signal->cols, signal->rows);
        int accepted = 0;

        for (int i = 0; i < count; ++i) {
            const TemplateBox& tb = boxes[i];
            LocatedChar& lc = chars_[static_cast<size_t>(i)];
            lc.index = tb.index;

            // 初始：按整图尺度换算的近似位置
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

            // 从模型频率图裁模板
            const cv::Rect tbox(tb.x, tb.y, tb.w, tb.h);
            if ((tbox & cv::Rect(0, 0, modelFreq_.cols, modelFreq_.rows)) != tbox) continue;
            if (tbox.width <= 3 || tbox.height <= 3) continue;

            cv::Mat templ;
            modelFreq_(tbox).convertTo(templ, CV_32F, 1.0 / 255.0);
            if (std::abs(sx - 1.0) > 0.01 || std::abs(sy - 1.0) > 0.01) {
                cv::resize(templ, templ, box.size(), 0, 0, cv::INTER_LINEAR);
            }
            cv::Scalar mean, stddev;
            cv::meanStdDev(templ, mean, stddev);
            if (stddev[0] < 1e-5) continue;

            // 搜索区
            const int rad = static_cast<int>(std::lround(g_params.radius * 0.5 * (sx + sy)));
            cv::Rect search(box.x - rad, box.y - rad, box.width + 2 * rad, box.height + 2 * rad);
            search &= bounds;
            if (search.width <= templ.cols || search.height <= templ.rows) continue;

            cv::Mat response;
            cv::matchTemplate((*signal)(search), templ, response, cv::TM_CCOEFF_NORMED);
            cv::Point peak;
            double mval = 0;
            cv::minMaxLoc(response, nullptr, &mval, nullptr, &peak);

            // 亚像素插值
            const auto subpixel = [](double l, double c, double rr) {
                const double den = 2.0 * c - l - rr;
                return den > 1e-9 ? std::clamp(0.5 * (rr - l) / den, -0.5, 0.5) : 0.0;
            };
            double subX = 0, subY = 0;
            if (peak.x > 0 && peak.x + 1 < response.cols)
                subX = subpixel(response.at<float>(peak.y, peak.x - 1),
                                response.at<float>(peak.y, peak.x),
                                response.at<float>(peak.y, peak.x + 1));
            if (peak.y > 0 && peak.y + 1 < response.rows)
                subY = subpixel(response.at<float>(peak.y - 1, peak.x),
                                response.at<float>(peak.y, peak.x),
                                response.at<float>(peak.y + 1, peak.x));

            lc.confidence = static_cast<float>(std::clamp(mval, 0.0, 1.0));
            lc.x = static_cast<int>(std::lround(search.x + peak.x + subX));
            lc.y = static_cast<int>(std::lround(search.y + peak.y + subY));
            lc.w = templ.cols;
            lc.h = templ.rows;

            // 边界拒绝 + 置信度门槛
            const bool onEdge = peak.x < 1 || peak.y < 1 || peak.x + 1 >= response.cols ||
                                peak.y + 1 >= response.rows;
            lc.accepted = !onEdge && mval >= g_params.minConf;
            if (lc.accepted) ++accepted;
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
    cv::Mat modelFreq_;
    std::vector<LocatedChar> chars_;
};

// ================================================================
// 导出函数 —— 不要改
// ================================================================
Locator g_locator;
LocateResult g_result;
std::string g_paramHelp[8];
const char* g_paramPtrs[8];
int g_paramCount = 0;

locate_api_t g_api{};

bool apiInit(const char* modelDir) { return g_locator.init(modelDir); }

void apiSetParams(const LocateParam* p, int n) { g_locator.setParams(p, n); }

const char** apiListParams(int* count) {
    g_paramHelp[0] = "radius=24.0     搜索半径（模型像素）";
    g_paramHelp[1] = "minConf=0.50    最低置信度，低于则判定位失败";
    g_paramCount = 2;
    for (int i = 0; i < g_paramCount; ++i) g_paramPtrs[i] = g_paramHelp[i].c_str();
    if (count) *count = g_paramCount;
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
