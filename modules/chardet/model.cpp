#include "chardet.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace chardet {

namespace fs = std::filesystem;

// ============================================================================
// 内部：笔画信号
// ============================================================================
namespace {

cv::Mat computeSignal(const cv::Mat& patch, const Config& cfg, bool invert) {
    cv::Mat p;
    patch.convertTo(p, CV_32F);
    cv::Mat bg;
    cv::GaussianBlur(p, bg, cv::Size(0, 0), cfg.sigma);
    cv::Mat d;
    cv::subtract(p, bg, d);
    if (invert) cv::multiply(d, -1.0, d);

    // 残差直方图（用于估计归一化刻度）
    std::vector<int> hist(512, 0);
    for (int y = 0; y < d.rows; ++y) {
        const float* row = d.ptr<float>(y);
        for (int x = 0; x < d.cols; ++x) {
            int b = static_cast<int>(std::lround(row[x])) + 256;
            b = std::clamp(b, 0, 511);
            ++hist[static_cast<size_t>(b)];
        }
    }
    const long total = static_cast<long>(d.rows) * d.cols;
    // 归一化刻度：只用"确实有笔画"的像素来估计，避免背景大面积亮区把刻度拉低
    //
    // 做法：先算出 99.5 分位作为参考上限，再取 90~99.5 分位之间的中位数。
    // 这样刻度由真实笔画强度决定，而不是被背景亮块主导。
    auto quantile = [&](double q) {
        const long target = static_cast<long>(total * q);
        long a = 0;
        for (int b = 0; b < 512; ++b) {
            a += hist[static_cast<size_t>(b)];
            if (a >= target) return static_cast<double>(b - 256);
        }
        return static_cast<double>(511 - 256);
    };
    const double q995 = quantile(0.995);
    const double q90 = quantile(0.90);
    // 用高分位（99.5%）作为刻度，使笔画落在 0~1 的较窄范围内，不会过度饱和
    double scale = std::max(8.0, q995 * 0.55 + q90 * 0.45);

    cv::Mat s;
    d.convertTo(s, CV_32F, 1.0 / scale);
    cv::threshold(s, s, 0.0, 0.0, cv::THRESH_TOZERO);
    return s;
}

// 简易 JSON：只支持本模块用到的平坦结构
std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o.push_back('\\');
        o.push_back(c);
    }
    return o;
}

}  // namespace

cv::Mat strokeSignal(const cv::Mat& gray, const cv::Rect& roi, const Config& cfg, bool invert) {
    const cv::Rect r = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width <= 0 || r.height <= 0) return {};
    return computeSignal(gray(r), cfg, invert);
}

// ============================================================================
// 训练
// ============================================================================

int Model::totalChars() const {
    int n = 0;
    for (const auto& r : regions) n += r.charCount();
    return n;
}

Model train(const TrainInput& in, const Config& cfg, TrainReport* report) {
    Model m;
    m.config = cfg;
    m.view = in.view;

    TrainReport rep;

    // ① 过滤尺寸不一致的图
    cv::Size refSize;
    std::vector<cv::Mat> imgs;
    for (const auto& im : in.images) {
        if (im.empty() || im.type() != CV_8U) { ++rep.skipped; continue; }
        if (refSize.width == 0) refSize = im.size();
        if (im.size() != refSize) { ++rep.skipped; continue; }
        imgs.push_back(im);
    }
    if (imgs.empty()) {
        rep.message = "没有可用图像（尺寸不一致或为空）";
        if (report) *report = rep;
        return m;
    }

    const auto t0 = std::chrono::steady_clock::now();
    m.imageSize = refSize;

    // ② 基准信号
    const cv::Rect full(0, 0, refSize.width, refSize.height);
    cv::Mat sRef = computeSignal(imgs.front(), cfg, false);
    cv::Mat ref8;
    sRef.convertTo(ref8, CV_8U, 80.0, 0.0);
    m.ref = ref8;

    // ③ 累加频率
    cv::Mat count = cv::Mat::zeros(sRef.size(), CV_32F);
    int used = 0;
    for (size_t k = 0; k < imgs.size(); ++k) {
        cv::Mat s = computeSignal(imgs[k], cfg, false);
        // 后续图像对齐到基准（相位相关，纯平移）
        //
        // 方向说明（实测验证过）：
        //   phaseCorrelate(B, A) 返回的是把 A 移到 B 所需的位移，
        //   因此要把 B 对齐到 A，必须用 **负** 的返回值。
        //   曾误用 +shift，导致对齐后误差反而从 0.184 升到 0.238。
        if (k > 0) {
            cv::Mat a32, b32;
            sRef.convertTo(a32, CV_32F);
            s.convertTo(b32, CV_32F);
            const cv::Point2d shift = cv::phaseCorrelate(b32, a32, cv::noArray());
            if (std::abs(shift.x) < 60 && std::abs(shift.y) < 60 &&
                (std::abs(shift.x) > 0.01 || std::abs(shift.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -shift.x, 0, 1, -shift.y);
                cv::warpAffine(s, s, M, s.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
            }
        }
        // 累加：二值笔画掩膜
        //
        // 曾试过改用连续信号累加（按 strokeHi 软阈值），
        // 结果边缘过渡反而从 3px 增到 5px、梯度锐度从 2.06 降到 1.03。
        // 原因：连续信号保留了笔画内部的灰度起伏，这些起伏在样本间差异大，
        //       平均后被放大成噪声；二值化把"有墨/无墨"归一，边界位置反而更稳。
        // 因此保留二值化。
        cv::Mat mask;
        cv::compare(s, cfg.strokeHi, mask, cv::CMP_GT);
        cv::Mat mf;
        mask.convertTo(mf, CV_32F, 1.0 / 255.0);
        count += mf;
        ++used;
    }
    if (used == 0) {
        rep.message = "没有成功统计任何样本";
        if (report) *report = rep;
        return m;
    }

    cv::Mat freq = count / static_cast<double>(used);
    freq.convertTo(m.freq, CV_8U, 255.0, 0.0);

    // ④ 不预设区域：区域完全由用户在设计器中画 ROI 决定
    //
    // 曾经尝试过自动行检测（行投影找文字带），但整幅图包含管脚、治具、
    // 芯片边缘等强反光，其投影强度远超笔画，导致检出的是背景而非文字行。
    // 因此改为：训练只产出频率图，区域由用户在界面上框选。

    const auto t1 = std::chrono::steady_clock::now();
    rep.ok = true;
    rep.usedImages = used;
    rep.autoRegions = static_cast<int>(m.regions.size());
    rep.autoChars = m.totalChars();
    rep.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rep.message = "训练完成";
    if (report) *report = rep;
    return m;
}

// ============================================================================
// 精对齐：用用户划定的字符框消除刻印偏移
// ============================================================================

Model refine(const std::vector<cv::Mat>& images, const Model& coarse, const Config& cfg,
             int searchRadius, double minPeak, RefineReport* report,
             const std::atomic<bool>* cancelled,
             const std::function<void(int, int)>* onProgress) {
    RefineReport rep;
    Model out = coarse;          // 沿用 regions / config / ref
    auto isCancelled = [cancelled]() {
        return cancelled != nullptr && cancelled->load();
    };

    if (coarse.freq.empty()) {
        rep.message = "初版模型图为空";
        if (report) *report = rep;
        return out;
    }
    if (!coarse.hasRegions()) {
        rep.message = "初版模型图里没有字符框，请先划出选区";
        if (report) *report = rep;
        return out;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const cv::Size sz = coarse.freq.size();

    // 过滤尺寸不一致的样本
    std::vector<cv::Mat> imgs;
    for (const auto& im : images) {
        if (im.empty() || im.type() != CV_8U) continue;
        if (im.size() != sz) continue;
        imgs.push_back(im);
    }
    if (imgs.empty()) {
        rep.message = "没有与模型图尺寸一致的样本";
        if (report) *report = rep;
        return out;
    }

    // 初版模型图的信号（作为模板来源）
    // freq 是 8 位频率图，转成浮点后作为"标准字形"使用
    cv::Mat coarseF;
    coarse.freq.convertTo(coarseF, CV_32F, 1.0 / 255.0);

    // 累计器：每个字符一份
    struct CharAcc {
        cv::Mat sum;     // 累加的笔画二值掩膜（浮点）
        int n = 0;       // 参与累加的次数
    };
    std::vector<std::vector<CharAcc>> acc(coarse.regions.size());
    for (size_t ri = 0; ri < coarse.regions.size(); ++ri) {
        acc[ri].resize(coarse.regions[ri].chars.size());
        for (auto& a : acc[ri]) {
            a.sum = cv::Mat::zeros(sz, CV_32F);
        }
    }

    double peakSum = 0.0;
    int peakCount = 0;
    double peakMin = 1.0;

    // 逐样本、逐字符对齐并累加
    for (size_t k = 0; k < imgs.size(); ++k) {
        // 支持中断：GUI 关闭窗口时置位，避免线程继续跑完
        if (isCancelled()) {
            rep.message = "已取消";
            if (report) *report = rep;
            return out;
        }
        // 进度回报（每张样本一次）
        if (onProgress && *onProgress) {
            (*onProgress)(static_cast<int>(k), static_cast<int>(imgs.size()));
        }
        cv::Mat s = computeSignal(imgs[k], cfg, false);

        // 先做整图粗对齐到 ref（消除整体摆放偏差），方向为 -shift
        if (!out.ref.empty()) {
            cv::Mat r32, s32;
            out.ref.convertTo(r32, CV_32F);
            s.convertTo(s32, CV_32F);
            const cv::Point2d sh = cv::phaseCorrelate(s32, r32, cv::noArray());
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                cv::warpAffine(s, s, M, s.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
            }
        }

        // 笔画二值（用于累加）
        cv::Mat bin;
        cv::compare(s, cfg.strokeHi, bin, cv::CMP_GT);
        cv::Mat binF;
        bin.convertTo(binF, CV_32F, 1.0 / 255.0);

        for (size_t ri = 0; ri < coarse.regions.size(); ++ri) {
            const auto& region = coarse.regions[ri];
            for (size_t ci = 0; ci < region.chars.size(); ++ci) {
                const cv::Rect box = region.chars[ci].bbox & cv::Rect(0, 0, sz.width, sz.height);
                if (box.width <= 0 || box.height <= 0) continue;

                ++rep.totalChars;

                // 搜索半径按框尺寸自适应。
                //
                // matchTemplate 代价 = 搜索区 × 模板，大框（如整个 logo）会暴涨。
                // 但刻印偏移是绝对像素量级（σ≈6px），与框大小无关，
                // 所以大框不需要更大的搜索半径——按短边 8% 缩放即可，
                // 既覆盖 ±2σ，又不让计算量随框面积平方增长。
                const int shortSide = std::min(box.width, box.height);
                const int radius =
                    std::clamp(static_cast<int>(shortSide * 0.08), 6, searchRadius);

                // 模板：初版模型图在该框内的字形
                const cv::Mat templ = coarseF(box);

                // 搜索区：框外扩 radius（见上面的自适应半径）
                cv::Rect search(box.x - radius, box.y - radius,
                                box.width + 2 * radius, box.height + 2 * radius);
                search &= cv::Rect(0, 0, s.cols, s.rows);   // 用信号的实际尺寸
                if (search.width < templ.cols || search.height < templ.rows) continue;
                // matchTemplate 要求模板不大于搜索区，且两者同类型
                if (templ.cols <= 0 || templ.rows <= 0) continue;

                cv::Mat res;
                cv::matchTemplate(s(search), templ, res, cv::TM_CCOEFF_NORMED);
                cv::Point mLoc;
                double mVal = 0.0;
                cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);

                peakSum += mVal;
                ++peakCount;
                peakMin = std::min(peakMin, mVal);

                if (mVal < minPeak) {
                    ++rep.skippedWeak;
                    continue;   // 该样本此字匹配太差，不参与累加
                }

                // 最佳位置相对原框的偏移
                const int dx = mLoc.x - radius;
                const int dy = mLoc.y - radius;

                // 把该样本这个字符的笔画搬到基准位置累加
                const cv::Rect src(box.x + dx, box.y + dy, box.width, box.height);
                const cv::Rect srcClipped = src & cv::Rect(0, 0, sz.width, sz.height);
                if (srcClipped.width != box.width || srcClipped.height != box.height) continue;

                acc[ri][ci].sum(box) += binF(srcClipped);
                ++acc[ri][ci].n;
                ++rep.refinedPairs;
            }
        }
    }

    // 归一化：每个字符只在"有样本的框内"写回平均值
    cv::Mat outF = coarseF.clone();
    for (size_t ri = 0; ri < coarse.regions.size(); ++ri) {
        const auto& region = coarse.regions[ri];
        for (size_t ci = 0; ci < region.chars.size(); ++ci) {
            CharAcc& a = acc[ri][ci];
            if (a.n == 0) continue;      // 该字没有可用样本，保留初版
            const cv::Rect box = region.chars[ci].bbox & cv::Rect(0, 0, sz.width, sz.height);
            if (box.width <= 0 || box.height <= 0) continue;
            a.sum(box) /= static_cast<double>(a.n);
            a.sum(box).copyTo(outF(box));
        }
    }

    outF.convertTo(out.freq, CV_8U, 255.0, 0.0);

    const auto t1 = std::chrono::steady_clock::now();
    rep.ok = true;
    rep.usedImages = static_cast<int>(imgs.size());
    rep.meanPeak = peakCount > 0 ? peakSum / peakCount : 0.0;
    rep.minPeak = peakMin;
    rep.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rep.message = "精对齐完成";
    if (report) *report = rep;
    return out;
}

// ============================================================================
// 模型存取
// ============================================================================

bool saveModel(const std::string& modelDir, const Model& m) {
    if (!m.valid()) return false;
    std::error_code ec;
    fs::create_directories(modelDir, ec);

    // 频率图与基准
    if (!cv::imwrite(modelDir + "/freq.png", m.freq)) return false;
    if (!m.ref.empty()) cv::imwrite(modelDir + "/ref.png", m.ref);

    // 元信息
    {
        std::ofstream os(modelDir + "/model.json");
        if (!os) return false;
        os << "{\n";
        os << "  \"module\": \"chardet\",\n";
        os << "  \"version\": " << m.version << ",\n";
        os << "  \"view\": " << m.view << ",\n";
        os << "  \"imageWidth\": " << m.imageSize.width << ",\n";
        os << "  \"imageHeight\": " << m.imageSize.height << ",\n";
        os << "  \"regionCount\": " << m.regions.size() << ",\n";
        os << "  \"charCount\": " << m.totalChars() << ",\n";
        os << "  \"config\": {\n";
        os << "    \"sigma\": " << m.config.sigma << ",\n";
        os << "    \"strokeHi\": " << m.config.strokeHi << ",\n";
        os << "    \"missStrokeLo\": " << m.config.missStrokeLo << ",\n";
        os << "    \"missFreq\": " << m.config.missFreq << ",\n";
        os << "    \"charMissRatio\": " << m.config.charMissRatio << ",\n";
        os << "    \"charExtraRatio\": " << m.config.charExtraRatio << ",\n";
        os << "    \"charT\": " << m.config.charT << ",\n";
        os << "    \"colThrRatio\": " << m.config.colThrRatio << ",\n";
        os << "    \"gapMergeRatio\": " << m.config.gapMergeRatio << ",\n";
        os << "    \"binFreq\": " << m.config.binFreq << "\n";
        os << "  }\n";
        os << "}\n";
    }

    // 区域与字符
    //
    // 格式（行式，每行一个区域头，紧跟该区域的字符行）：
    //
    //   字符区：
    //     #region <序号> box <x> <y> <宽> <高> <字符数>
    //     <每字符一行：bbox x y w h  extend x y w h  序号>
    //
    //   图案区（闭合曲线）：
    //     #region <序号> ellipse <中心x> <中心y> <长半轴> <短半轴> <长轴角度> <字符数>
    //     （circle 同格式，长短半轴相等；字符数通常为 0）
    //
    // 图案区为什么不存矩形：轴对齐矩形存不下"半轴 + 角度"，
    // 同一个外接框能对应多个不同角度、不同半轴的椭圆，反推是多解的。
    // 而 SC 环实测长轴 -13.3°，不存角度就没法贴合。
    // 矩形由 syncRectFromEll() 在加载后现算，只用来做初步裁剪。
    {
        std::ofstream os(modelDir + "/regions.txt");
        if (!os) return false;
        os << "# regions v3\n";
        for (const auto& r : m.regions) {
            if (r.shape == RegionShape::Box) {
                os << "#region " << r.index << " box " << r.rect.x << " " << r.rect.y << " "
                   << r.rect.width << " " << r.rect.height << " " << r.chars.size() << "\n";
            } else {
                const char* shp = r.shape == RegionShape::Circle ? "circle" : "ellipse";
                os << "#region " << r.index << " " << shp << " " << r.ell.center.x << " "
                   << r.ell.center.y << " " << r.ell.a << " " << r.ell.b << " " << r.ell.angleDeg
                   << " " << r.chars.size() << "\n";
            }
            for (const auto& c : r.chars) {
                os << c.bbox.x << " " << c.bbox.y << " " << c.bbox.width << " " << c.bbox.height
                   << " " << c.extend.x << " " << c.extend.y << " " << c.extend.width << " "
                   << c.extend.height << " " << c.index << "\n";
            }
        }
    }


    // 多模板库：每字符存成一行图片（横向拼接），外加一份索引
    if (!m.charTemplates.empty()) {
        const std::string path = modelDir + "/templates.png";
        std::vector<cv::Mat> rows;
        std::ostringstream idx;
        idx << m.templateSize << "\n";
        idx << m.charTemplates.size() << "\n";
        for (const auto& list : m.charTemplates) {
            idx << list.size() << "\n";
            if (list.empty()) continue;
            std::vector<cv::Mat> rowMats;
            for (const auto& t : list) {
                cv::Mat t8;
                cv::normalize(t, t8, 0, 255, cv::NORM_MINMAX);
                t8.convertTo(t8, CV_8U);
                rowMats.push_back(t8);
            }
            cv::Mat row;
            // 注意：Mat::push_back 是纵向追加，横向拼接必须用 hconcat
            cv::hconcat(rowMats, row);
            rows.push_back(row);
        }
        if (!rows.empty()) {
            // 各行宽度不同，补零对齐后竖向拼接
            int wmax = 0;
            for (const auto& r2 : rows) wmax = std::max(wmax, r2.cols);
            cv::Mat sheet;
            for (size_t i = 0; i < rows.size(); ++i) {
                cv::Mat padded = cv::Mat::zeros(rows[i].rows, wmax, CV_8U);
                rows[i].copyTo(padded(cv::Rect(0, 0, rows[i].cols, rows[i].rows)));
                sheet.push_back(padded);
            }
            cv::imwrite(path, sheet);
            std::ofstream os(modelDir + "/templates.txt");
            if (os) os << idx.str();
        }
    }
    return true;
}
bool loadModel(const std::string& modelDir, Model& m) {
    m = Model{};
    std::error_code ec;
    if (!fs::exists(modelDir + "/regions.txt", ec)) return false;

    m.freq = cv::imread(modelDir + "/freq.png", cv::IMREAD_GRAYSCALE);
    m.ref = cv::imread(modelDir + "/ref.png", cv::IMREAD_GRAYSCALE);
    if (m.freq.empty()) return false;
    m.imageSize = m.freq.size();

    // 元信息（宽松解析：只取需要的数值）
    {
        std::ifstream is(modelDir + "/model.json");
        std::string line;
        while (std::getline(is, line)) {
            auto pick = [&](const char* key, double& out) {
                const std::string k = std::string("\"") + key + "\"";
                const size_t pos = line.find(k);
                if (pos == std::string::npos) return;
                const size_t colon = line.find(':', pos + k.size());
                if (colon == std::string::npos) return;
                out = std::atof(line.c_str() + colon + 1);
            };
            double v = 0;
            pick("view", v); if (v != 0) m.view = static_cast<int>(v);
            pick("sigma", v); if (v > 0) m.config.sigma = v;
            pick("strokeHi", v); if (v > 0) m.config.strokeHi = v;
            pick("missStrokeLo", v); if (v > 0) m.config.missStrokeLo = v;
            pick("missFreq", v); if (v > 0) m.config.missFreq = v;
            pick("charMissRatio", v); if (v > 0) m.config.charMissRatio = v;
            pick("charExtraRatio", v); if (v > 0) m.config.charExtraRatio = v;
            pick("charT", v); if (v > 0) m.config.charT = v;
        }
    }

    // 区域与字符（v2 行式格式，见 saveModel 里的说明）
    {
        std::ifstream is(modelDir + "/regions.txt");
        if (!is) return false;
        std::string line;
        RegionInfo cur;
        bool haveCur = false;
        int pendingChars = 0;

        auto flush = [&]() {
            if (haveCur) m.regions.push_back(std::move(cur));
            haveCur = false;
        };

        while (std::getline(is, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                std::istringstream ls(line);
                std::string head;
                ls >> head;
                if (head != "#region") continue;   // 文件头 "# regions vN" 之类跳过
                flush();
                cur = RegionInfo();
                int idx = 0;
                std::string shp;
                if (!(ls >> idx >> shp)) return false;
                cur.index = idx;
                if (shp == "box") {
                    cur.shape = RegionShape::Box;
                    if (!(ls >> cur.rect.x >> cur.rect.y >> cur.rect.width >> cur.rect.height >>
                          pendingChars))
                        return false;
                } else if (shp == "ellipse" || shp == "circle") {
                    cur.shape = shp == "circle" ? RegionShape::Circle : RegionShape::Ellipse;
                    // 图案区存的是椭圆参数，不是矩形
                    if (!(ls >> cur.ell.center.x >> cur.ell.center.y >> cur.ell.a >> cur.ell.b >>
                          cur.ell.angleDeg >> pendingChars))
                        return false;
                    if (cur.ell.a < cur.ell.b) std::swap(cur.ell.a, cur.ell.b);
                    cur.syncRectFromEll();   // 矩形只用来做初步裁剪
                } else {
                    return false;   // 不认识的形状：宁可报错，也别静默当成字符区
                }
                haveCur = true;
                continue;
            }
            if (!haveCur || pendingChars <= 0) continue;
            std::istringstream ls(line);
            CharBox c;
            if (!(ls >> c.bbox.x >> c.bbox.y >> c.bbox.width >> c.bbox.height >> c.extend.x >>
                  c.extend.y >> c.extend.width >> c.extend.height >> c.index))
                continue;
            cur.chars.push_back(c);
            --pendingChars;
        }
        flush();
    }

    // 多模板库（可选）：不存在时模型仍可用，只是不具备改笔画检出能力
    {
        std::ifstream is(modelDir + "/templates.txt");
        if (is) {
            size_t tsize = 0;
            size_t nCharT = 0;
            if (is >> tsize >> nCharT) {
                m.templateSize = static_cast<int>(tsize);
                cv::Mat sheet = cv::imread(modelDir + "/templates.png", cv::IMREAD_GRAYSCALE);
                if (!sheet.empty()) {
                    m.charTemplates.assign(nCharT, {});
                    int y = 0;
                    for (size_t ci = 0; ci < nCharT; ++ci) {
                        size_t n = 0;
                        if (!(is >> n)) break;
                        if (n == 0) continue;
                        if (y + m.templateSize > sheet.rows) break;
                        for (size_t k = 0; k < n; ++k) {
                            const int x0 = static_cast<int>(k) * m.templateSize;
                            if (x0 + m.templateSize > sheet.cols) break;
                            cv::Mat t = sheet(cv::Rect(x0, y, m.templateSize, m.templateSize));
                            cv::Mat tf;
                            t.convertTo(tf, CV_32F, 1.0 / 255.0);
                            m.charTemplates[ci].push_back(tf);
                        }
                        y += m.templateSize;
                    }
                }
            }
        }
    }
    return m.valid();
}

bool isModelDir(const std::string& modelDir) {
    std::error_code ec;
    return fs::exists(modelDir + "/regions.txt", ec) &&
           fs::exists(modelDir + "/freq.png", ec);
}

// ============================================================================
// 消费：模板与比对区域
// ============================================================================

cv::Mat regionTemplate(const Model& m, int regionIndex) {
    if (regionIndex < 0 || regionIndex >= static_cast<int>(m.regions.size())) return {};
    const auto& r = m.regions[static_cast<size_t>(regionIndex)];
    const cv::Rect roi = r.rect & cv::Rect(0, 0, m.freq.cols, m.freq.rows);
    if (roi.width <= 0 || roi.height <= 0) return {};
    return m.freq(roi).clone();
}

std::vector<cv::Mat> charTemplates(const Model& m, int regionIndex) {
    std::vector<cv::Mat> out;
    if (regionIndex < 0 || regionIndex >= static_cast<int>(m.regions.size())) return out;
    const auto& r = m.regions[static_cast<size_t>(regionIndex)];
    const cv::Rect img(0, 0, m.freq.cols, m.freq.rows);

    for (const auto& c : r.chars) {
        const cv::Rect box = c.bbox & img;
        if (box.width <= 0 || box.height <= 0) { out.emplace_back(); continue; }
        cv::Mat t = cv::Mat::zeros(m.freq.size(), CV_8U);
        m.freq(box).copyTo(t(box));
        out.push_back(std::move(t));
    }
    return out;
}

std::vector<CharUnit> toUnits(const Model& m, int regionIndex) {
    std::vector<CharUnit> out;
    if (regionIndex < 0 || regionIndex >= static_cast<int>(m.regions.size())) return out;
    const auto& r = m.regions[static_cast<size_t>(regionIndex)];
    const cv::Rect img(0, 0, m.freq.cols, m.freq.rows);

    int idx = 0;
    for (const auto& c : r.chars) {
        const cv::Rect box = c.bbox & img;
        if (box.width <= 0 || box.height <= 0) continue;
        cv::Rect ext = c.extend & img;
        if (ext.width <= 0 || ext.height <= 0) ext = box;

        // 扩展框至少要比字符框大一圈。
        // 脏污、划痕常紧贴字符边缘外侧，扩展太小会检不到。
        constexpr int kMinPad = 12;
        const cv::Rect minExt(std::max(0, box.x - kMinPad), std::max(0, box.y - kMinPad),
                              std::min(img.width - std::max(0, box.x - kMinPad),
                                       box.width + 2 * kMinPad),
                              std::min(img.height - std::max(0, box.y - kMinPad),
                                       box.height + 2 * kMinPad));
        ext = ext | minExt;
        ext &= img;

        CharUnit u;
        u.index = idx++;
        u.bbox = box;
        u.extend = ext;
        u.freq = cv::Mat::zeros(m.freq.size(), CV_8U);
        m.freq(box).copyTo(u.freq(box));
        out.push_back(std::move(u));
    }
    return out;
}

}  // namespace chardet
