#include "core/naming.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace markloc::charinsp::naming {

namespace fs = std::filesystem;

namespace {

fs::path toPath(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

std::string asUtf8(const std::u8string& s) {
    return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

std::string stemOf(const std::string& path) {
    return asUtf8(toPath(path).stem().u8string());
}

std::string nameOf(const std::string& path) {
    return asUtf8(toPath(path).filename().u8string());
}

bool hasTifSuffix(const fs::path& p) {
    const std::string ext = asUtf8(p.extension().u8string());
    return ext == ".tif" || ext == ".tiff" || ext == ".TIF" || ext == ".TIFF";
}

}  // namespace

int viewOf(const std::string& path) {
    const std::string stem = stemOf(path);
    if (stem.size() < 3) return -1;
    if (stem[stem.size() - 3] != '_') return -1;

    const std::string tail = stem.substr(stem.size() - 2);
    if (tail[0] < '0' || tail[0] > '9' || tail[1] < '0' || tail[1] > '9') return -1;
    const int v = (tail[0] - '0') * 10 + (tail[1] - '0');
    return (v >= 0 && v <= 5) ? v : -1;
}

std::string productOf(const std::string& path) {
    const std::string name = nameOf(path);
    return name.size() >= 4 ? name.substr(0, 4) : name;
}

std::string prefixOf(const std::string& path) {
    std::string stem = stemOf(path);
    if (stem.size() < 3) return {};
    stem.erase(stem.size() - 3);   // 去掉 "_NN"
    return stem;
}

std::string shotOf(const std::string& path) {
    const std::string stem = stemOf(path);
    if (stem.find("_FT") != std::string::npos) return "FT";
    if (stem.find("_RT") != std::string::npos) return "RT";
    return {};
}

std::string siblingPath(const std::string& anyView, int view) {
    const fs::path p = toPath(anyView);
    const std::string prefix = prefixOf(anyView);
    if (prefix.empty() || view < 0 || view > 5) return {};

    char buf[8];
    std::snprintf(buf, sizeof(buf), "_%02d", view);
    const std::string name = prefix + buf + asUtf8(p.extension().u8string());
    return asUtf8((p.parent_path() / toPath(name)).u8string());
}

std::vector<std::string> findSiblings(const std::string& anyView,
                                      std::array<bool, 6>& present) {
    present.fill(false);
    std::vector<std::string> out(6);
    for (int v = 0; v < 6; ++v) {
        const std::string p = siblingPath(anyView, v);
        if (p.empty()) continue;
        std::error_code ec;
        if (!fs::exists(toPath(p), ec)) continue;
        present[static_cast<size_t>(v)] = true;
        out[static_cast<size_t>(v)] = p;
    }
    return out;
}

std::vector<std::string> collectImages(const std::string& root) {
    std::vector<std::string> files;
    std::error_code ec;

    for (const char* sub : {"ok", "ng"}) {
        const fs::path d = toPath(root) / sub;
        if (!fs::exists(d, ec)) continue;
        for (const auto& e : fs::directory_iterator(d, ec)) {
            if (e.is_regular_file() && hasTifSuffix(e.path())) files.push_back(asUtf8(e.path().u8string()));
        }
    }
    if (files.empty()) {
        for (const auto& e : fs::directory_iterator(toPath(root), ec)) {
            if (e.is_regular_file() && hasTifSuffix(e.path())) files.push_back(asUtf8(e.path().u8string()));
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

}  // namespace markloc::charinsp::naming
