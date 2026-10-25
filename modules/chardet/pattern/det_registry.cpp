// det_registry.cpp —— 检测器注册表
//
// ================================================================
// 加一个算法要改哪里
// ================================================================
//
// 只改这个文件的两行：
//
//   extern std::unique_ptr<IDetector> makeMyDetector();   // 已在别处声明
//   g_registry["my_detector"] = &makeMyDetector;
//
// 然后就能用 params.detectors = {"my_detector"} 跑起来。
//
// 不需要改 pattern_pipeline.cpp，不需要改 GUI，不需要改任何已有检测器。
// 这就是把检测器做成接口的全部意义。
#include "pattern_api.hpp"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace chardet {
namespace pattern {

// 各检测器的工厂函数（实现在各自的 .cpp 里）
std::unique_ptr<IDetector> makeNotchWindowDetector();
std::unique_ptr<IDetector> makeGapRunDetector();

namespace {

using Factory = std::unique_ptr<IDetector> (*)();

std::map<std::string, Factory>& registry() {
    static std::map<std::string, Factory> r;
    return r;
}

std::mutex& registryMutex() {
    static std::mutex m;
    return m;
}

}  // namespace

void registerBuiltins() {
    std::lock_guard<std::mutex> lock(registryMutex());
    auto& r = registry();
    r["notch_window"] = &makeNotchWindowDetector;
    r["gap_run"] = &makeGapRunDetector;
}

std::unique_ptr<IDetector> makeDetector(const std::string& name) {
    // 允许调用方漏调 registerBuiltins()：这里补一次，重复调用无副作用
    registerBuiltins();
    std::lock_guard<std::mutex> lock(registryMutex());
    const auto it = registry().find(name);
    if (it == registry().end()) return nullptr;
    return (it->second)();
}

std::vector<std::string> detectorNames() {
    registerBuiltins();
    std::lock_guard<std::mutex> lock(registryMutex());
    std::vector<std::string> out;
    out.reserve(registry().size());
    for (const auto& kv : registry()) out.push_back(kv.first);
    return out;
}

}  // namespace pattern
}  // namespace chardet
