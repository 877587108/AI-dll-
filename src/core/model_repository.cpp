#include "core/model_repository.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <opencv2/imgcodecs.hpp>

#include "core/algorithm_internal.hpp"
#include "core/naming.hpp"

namespace markloc::algorithm {
namespace {
std::string utf8(const std::u8string& value) {
    return std::string(reinterpret_cast<const char*>(value.data()), value.size());
}

SampleLabel labelOf(const std::filesystem::path& p) {
    for (auto it = p.begin(); it != p.end(); ++it) {
        std::string s = utf8(it->u8string());
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (s == "ok") return SampleLabel::Ok;
        if (s == "ng") return SampleLabel::Ng;
    }
    return SampleLabel::Unknown;
}
}

ModelRepository::ModelRepository(RepositoryOptions options) : options_(options) {}

bool ModelRepository::readGray(const Path& path, cv::Mat& gray, std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { if (error) *error = "无法打开图片: " + utf8(path.u8string()); return false; }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.empty()) { if (error) *error = "图片为空: " + utf8(path.u8string()); return false; }
    gray = cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) { if (error) *error = "图片解码失败: " + utf8(path.u8string()); return false; }
    return true;
}

DatasetIndex ModelRepository::index(const Path& imageRoot) const {
    DatasetIndex out;
    if (!std::filesystem::exists(imageRoot)) {
        out.warnings.push_back("图片目录不存在: " + utf8(imageRoot.u8string()));
        return out;
    }
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it(imageRoot, ec), end; it != end && !ec; it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        std::string ext = utf8(it->path().extension().u8string());
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".tif" && ext != ".tiff" && ext != ".png" && ext != ".bmp") continue;
        Sample s;
        s.path = it->path();
        const std::string pathText = utf8(s.path.u8string());
        s.view = charinsp::naming::viewOf(pathText);
        s.product = charinsp::naming::productOf(pathText);
        s.label = labelOf(it->path());
        if (s.view < 0 || s.view >= charinsp::kViewCount) {
            out.warnings.push_back("无法解析视角: " + pathText);
            continue;
        }
        out.samples.push_back(std::move(s));
    }
    std::sort(out.samples.begin(), out.samples.end(), [](const Sample& a, const Sample& b) {
        return utf8(a.path.u8string()) < utf8(b.path.u8string());
    });
    return out;
}

ModelSnapshot ModelRepository::load(const Path& root, std::string* error) const {
    ModelSnapshot out;
    out.root = root;
    out.id = utf8(root.filename().u8string());
    auto data = std::make_shared<detail::ModelData>();
    std::string err;
    if (!data->model.loadFrom(utf8(root.u8string()), &err)) {
        if (error) *error = err.empty() ? "模型加载失败" : err;
        return out;
    }
    if (!options_.loadParameters) {
        // 保留模型图像，但使用库默认参数；这让离线工具可以明确选择“只读模板”。
        for (auto& params : data->model.params) params = charinsp::Params{};
    }
    if (options_.requireAllViews) {
        for (int v = 0; v < charinsp::kViewCount; ++v) if (!data->model.valid(v)) {
            if (error) *error = "模型缺少视角 v" + std::to_string(v);
            return out;
        }
    }
    out.valid = data->model.anyValid();
    if (!out.valid && error) *error = "模型没有可用视角";
    if (out.valid) out.data_ = std::move(data);
    return out;
}

} // namespace markloc::algorithm
