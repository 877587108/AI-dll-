// plugin_ncc2.cpp —— 逐字符 NCC 模板匹配定位插件（算法①）
//
// ================================================================
// 为什么是"逐字符"而不是"整图"
// ================================================================
//
// 整图 NCC 模板匹配在这个场景必然失败（实测相关系数只有 0.30）：
// 芯片图大部分是空白背景，两图间的背景不完全一致（光照、边框、
// 噪声），这些背景把整图的归一化相关系数稀释到阈值以下。
// 而**局部字符区域的 NCC 高达 0.65**（diag_ncc2.py 实测）。
//
// 所以算法①（模板匹配）的正确形态是：**逐字符独立匹配**——
// 每个字符框用标准样张（original.png）里的字符区域做模板，
// 在待测图上独立搜索，避免整图背景稀释。
//
// ================================================================
// 与算法③（ORB 特征点）的区别
// ================================================================
//
//   ORB：整图求单应矩阵（吸收旋转/透视），再映射所有框。抗旋转，
//        但对单个字符有全局近似残差。
//   本插件：每个字符独立 NCC 搜索，不假设整图刚体变换。更精确
//            （亚像素级），但对旋转敏感、可能匹配到相邻字符。
//   两者互补：ORB 适合整体对齐，逐字符 NCC 适合精确套字。
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

constexpr const char* kName = "ncc2";

struct Params {
    int searchR = 80;          // 搜索半径(px)，覆盖整体平移
    double minScale = 0.9;     // 每字符尺度搜索下限
    double maxScale = 1.1;     // 上限
    double scaleStep = 0.02;   // 尺度步长
    double minConf = 0.5;      // 每字符最低置信度（低于则 accepted=false）
    double sigma = 15.0;       // 去背景高斯尺度（与 chardet::Config 一致）
    int useAspectInit = 0;     // 初始位置用图幅比例(1)/恒等(0)，实测恒等更优
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

// 与宿主 chardet::computeSignal 等价的笔画信号变换：
// 高斯模糊去背景 → 原图减背景（高频残差）→ 负值归零 → 归一化。
// 模板和待测图都进这个域，NCC 才有可比性。原始灰度 vs 笔画信号
// 直接相关只有 0.30 左右，同域后远高于此。
cv::Mat computeSignal(const cv::Mat& patch, double sigma) {
    cv::Mat p;
    patch.convertTo(p, CV_32F);
    cv::Mat bg;
    cv::GaussianBlur(p, bg, cv::Size(0, 0), sigma);
    cv::Mat d = p - bg;
    cv::threshold(d, d, 0.0, 0.0, cv::THRESH_TOZERO);
    double mn, mx;
    cv::minMaxLoc(d, &mn, &mx);
    if (mx <= 1e-6) return d;
    d.convertTo(d, CV_32F, 1.0 / mx);
    return d;
}

class Ncc2Locator {
public:
    bool init(const char* modelDir) {
        if (!modelDir) return false;
        const std::string dir(modelDir);
        original_ = readGrayFile(dir + "/original.png");
        if (original_.empty()) original_ = readGrayFile(dir + "/model.png");
        if (original_.empty()) original_ = readGrayFile(dir + "/freq.png");
        if (original_.empty()) return false;
        // 标准样张也进笔画信号域
        originalSig_ = computeSignal(original_, g_params.sigma);
        return !originalSig_.empty();
    }

    void setParams(const LocateParam* ps, int n) {
        for (int i = 0; i < n; ++i) {
            const std::string k = ps[i].key ? ps[i].key : "";
            const double v = ps[i].value;
            if (k == "searchR") g_params.searchR = static_cast<int>(std::clamp(v, 20.0, 300.0));
            else if (k == "minScale") g_params.minScale = std::max(0.5, v);
            else if (k == "maxScale") g_params.maxScale = std::max(g_params.minScale, v);
            else if (k == "scaleStep") g_params.scaleStep = std::clamp(v, 0.005, 0.1);
            else if (k == "minConf") g_params.minConf = std::clamp(v, 0.0, 1.0);
            else if (k == "sigma") g_params.sigma = std::max(1.0, v);
            else if (k == "useAspectInit") g_params.useAspectInit = v > 0 ? 1 : 0;
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
        if (originalSig_.empty()) {
            r.message = "标准样张信号未就绪";
            return r;
        }

        // 信号转 8 位灰度（匹配用），保持与模板同域
        cv::Mat gray;
        cv::normalize(*signal, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat tpl8;
        cv::normalize(originalSig_, tpl8, 0, 255, cv::NORM_MINMAX, CV_8U);

        // 初始尺度
        const double initSx =
            g_params.useAspectInit && modelW > 0 ? static_cast<double>(imageW) / modelW : 1.0;
        const double initSy =
            g_params.useAspectInit && modelH > 0 ? static_cast<double>(imageH) / modelH : 1.0;

        int accepted = 0;
        for (int i = 0; i < count; ++i) {
            const TemplateBox& tb = boxes[i];
            LocatedChar& lc = chars_[static_cast<size_t>(i)];
            lc.index = tb.index;

            // 模板：笔画信号域的字符区域
            cv::Rect tmplRect(tb.x, tb.y, tb.w, tb.h);
            tmplRect &= cv::Rect(0, 0, tpl8.cols, tpl8.rows);
            if (tmplRect.width < 5 || tmplRect.height < 5) {
                lc.accepted = false;
                continue;
            }
            const cv::Mat tmpl = tpl8(tmplRect);

            const double cx0 = (tb.x + tb.w * 0.5) * initSx;
            const double cy0 = (tb.y + tb.h * 0.5) * initSy;

            double bestConf = -1, bestScale = initSx;
            int bestX = 0, bestY = 0;
            for (double s = g_params.minScale; s <= g_params.maxScale + 1e-6;
                 s += g_params.scaleStep) {
                cv::Mat t;
                cv::resize(tmpl, t, cv::Size(std::max(1, cvRound(tmpl.cols * s)),
                                             std::max(1, cvRound(tmpl.rows * s))),
                           0, 0, cv::INTER_LINEAR);
                const int sr = g_params.searchR;
                const int x0 = static_cast<int>(cx0) - sr;
                const int y0 = static_cast<int>(cy0) - sr;
                const int ww = t.cols + 2 * sr;
                const int hh = t.rows + 2 * sr;
                if (x0 < 0 || y0 < 0 || x0 + ww > gray.cols || y0 + hh > gray.rows) {
                    cv::Rect want(x0, y0, ww, hh);
                    cv::Rect inter = want & cv::Rect(0, 0, gray.cols, gray.rows);
                    if (inter.width <= t.cols || inter.height <= t.rows) continue;
                    cv::Mat res;
                    cv::matchTemplate(gray(inter), t, res, cv::TM_CCOEFF_NORMED);
                    double mv = 0;
                    cv::Point ml;
                    cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
                    if (mv > bestConf) { bestConf = mv; bestScale = s; bestX = inter.x + ml.x; bestY = inter.y + ml.y; }
                } else {
                    cv::Mat res;
                    cv::matchTemplate(gray(cv::Rect(x0, y0, ww, hh)), t, res,
                                      cv::TM_CCOEFF_NORMED);
                    double mv = 0;
                    cv::Point ml;
                    cv::minMaxLoc(res, nullptr, &mv, nullptr, &ml);
                    if (mv > bestConf) { bestConf = mv; bestScale = s; bestX = x0 + ml.x; bestY = y0 + ml.y; }
                }
            }

            if (bestConf < g_params.minConf) {
                lc.x = static_cast<int>(cx0 - tb.w * initSx * 0.5);
                lc.y = static_cast<int>(cy0 - tb.h * initSy * 0.5);
                lc.w = static_cast<int>(tb.w * initSx);
                lc.h = static_cast<int>(tb.h * initSy);
                lc.confidence = static_cast<float>(bestConf);
                lc.accepted = false;
                continue;
            }
            lc.x = bestX;
            lc.y = bestY;
            lc.w = static_cast<int>(std::lround(tb.w * bestScale));
            lc.h = static_cast<int>(std::lround(tb.h * bestScale));
            lc.confidence = static_cast<float>(bestConf);
            const bool inBounds = lc.x >= 0 && lc.y >= 0 && lc.x + lc.w <= imageW &&
                                  lc.y + lc.h <= imageH;
            lc.accepted = inBounds;
            if (lc.accepted) ++accepted;
        }

        r.scaleX = static_cast<float>(initSx);
        r.scaleY = static_cast<float>(initSy);
        r.offsetX = 0;
        r.offsetY = 0;
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
    cv::Mat originalSig_;
    std::vector<LocatedChar> chars_;
};

Ncc2Locator g_locator;
LocateResult g_result;
std::string g_paramHelp[8];
const char* g_paramPtrs[8];
int g_paramCount = 0;
locate_api_t g_api{};

bool apiInit(const char* modelDir) { return g_locator.init(modelDir); }
void apiSetParams(const LocateParam* p, int n) { g_locator.setParams(p, n); }

const char** apiListParams(int* count) {
    g_paramHelp[0] = "searchR=80       每字符搜索半径(px)";
    g_paramHelp[1] = "minScale=0.9     每字符尺度下限";
    g_paramHelp[2] = "maxScale=1.1     每字符尺度上限";
    g_paramHelp[3] = "scaleStep=0.02   尺度步长";
    g_paramHelp[4] = "minConf=0.5      每字符最低置信度";
    g_paramHelp[5] = "sigma=15.0       去背景高斯尺度";
    g_paramHelp[6] = "useAspectInit=0  初始位置用图幅比例(1)/恒等(0)";
    g_paramCount = 7;
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
