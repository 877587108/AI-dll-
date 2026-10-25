// pattern_pipeline.cpp —— 图案检测流水线
//
// ================================================================
// 流水线的职责
// ================================================================
//
// 只有三件事：
//   ① 按顺序调用预处理、剖面、各检测器
//   ② 汇总各检测器结论（取最大 severity）
//   ③ 把结果整理成 Result
//
// 它**不含任何判据逻辑**。判据全在检测器里。
// 这条边界保证：换检测器不会影响流水线，改流水线不会影响判据。
//
// ================================================================
// 失败处理
// ================================================================
//
// 任何阶段失败都返回 valid=false 的 Result，不抛异常。
// 调用方必须区分三种情况：
//   valid=false     没验过（区域参数不可用、拉直自检失败等）
//   valid && !flagged  验过，正常
//   valid && flagged   验过，有缺陷
//
// 把"没验过"当成"合格"是这类检测系统最危险的错误，
// 所以 Result.valid 是独立字段，不能只看 flagged。
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "pattern_api.hpp"

namespace chardet {
namespace pattern {

namespace {

/// 把 cv::Range（整数角度区间）规整成"起止都合法且 min<=max"
cv::Range normalizeRange(cv::Range r) {
    int a = r.start, b = r.end;
    if (a > b) std::swap(a, b);
    return cv::Range(a, b);
}

}  // namespace

Result analyze(const cv::Mat& signal, const RegionInfo& region, const Config& cfg,
               const Params& params) {
    Result res;
    registerBuiltins();

    Input in;
    in.signal = signal;
    in.region = region;
    in.cfg = cfg;
    in.params = params;
    if (!in.valid()) return res;

    // ---- ① 预处理
    auto pre = makePreprocessor();
    if (!pre) return res;
    cv::Mat straightened;
    std::string err;
    if (!pre->run(in, res.geom, straightened, res.polar, &err)) return res;

    // ---- ② 剖面
    auto builder = makeProfileBuilder();
    if (!builder) return res;
    if (!builder->build(res.polar, res.geom, params, res.profile)) return res;

    res.valid = true;

    // ---- ③ 依次跑检测器
    //
    // 顺序即 params.detectors 的顺序，结果也按这个顺序存，
    // 便于报告里稳定显示。
    for (const auto& name : params.detectors) {
        auto det = makeDetector(name);
        if (!det) {
            Finding f;
            f.detail = "未知检测器: " + name;
            res.findings.emplace_back(name, std::move(f));
            continue;
        }
        Finding f = det->detect(in, res.geom, res.polar, res.profile);
        // 规整角度区间
        for (auto& sp : f.spansDeg) sp = normalizeRange(sp);
        res.findings.emplace_back(name, std::move(f));
    }

    // ---- ④ 汇总
    res.severity = 0.0;
    for (const auto& kv : res.findings) {
        const Finding& f = kv.second;
        res.severity = std::max(res.severity, f.severity);
        if (f.triggered && f.severity >= params.minSeverity) {
            res.flagged = true;
        }
        if (f.triggered) {
            for (const auto& p : f.at) res.breakAt.push_back(p);
        }
    }
    return res;
}

Result analyze(const cv::Mat& signal, const RegionInfo& region, const Config& cfg,
               const PatternBreakParams& legacyParams) {
    return analyze(signal, region, cfg, Params::fromLegacy(legacyParams));
}

// ---------------------------------------------------------------- 调试输出

bool dumpPolar(const PolarMap& pm, const std::string& path) {
    if (!pm.valid()) return false;
    cv::Mat vis;
    cv::normalize(pm.img, vis, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    try {
        return cv::imwrite(path, vis);
    } catch (const cv::Exception&) {
        return false;
    }
}

bool dumpProfile(const Profile& prof, const PolarMap& pm, const std::string& path) {
    if (!prof.valid() || pm.angles <= 0) return false;
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return false;
    std::fprintf(fp, "deg,ink,rel\n");
    const int N = static_cast<int>(prof.size());
    for (int i = 0; i < N; ++i) {
        std::fprintf(fp, "%.4f,%.6f,%.6f\n", pm.degOf(i), prof.ink[static_cast<size_t>(i)],
                     prof.rel[static_cast<size_t>(i)]);
    }
    std::fclose(fp);
    return true;
}

}  // namespace pattern
}  // namespace chardet
