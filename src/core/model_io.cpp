#include "core/model_io.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace markloc::charinsp {

namespace {

// 读取用户可编辑的 rois.txt（若存在则覆盖 params.txt 里的区域定义）
// 格式：每行 `名称 启用 x y 宽 高 [检查多余笔画]`，# 开头为注释
void loadRoisFile(const std::string& dir, int view, Params& p) {
    std::ifstream f(viewDir(dir, view) + "/rois.txt");
    if (!f) return;

    std::string line;
    while (std::getline(f, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        const size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;

        std::istringstream ss(line.substr(b));
        std::string name;
        int en = 1, x = 0, y = 0, w = 0, h = 0, appendage = -1;
        if (!(ss >> name >> en >> x >> y >> w >> h)) continue;
        ss >> appendage;   // 可选

        int slot = -1;
        for (int i = 0; i < 4; ++i) {
            std::string nm = kRoiNames[i];
            while (!nm.empty() && nm.back() == ' ') nm.pop_back();
            if (name == nm) { slot = i; break; }
        }
        if (slot < 0 && name.size() == 1 && name[0] >= '0' && name[0] <= '3') {
            slot = name[0] - '0';
        }
        if (slot < 0) continue;
        if (w <= 0 || h <= 0) continue;

        p.rois[slot].rect = cv::Rect(x, y, w, h);
        p.rois[slot].enabled = en != 0;
        if (appendage >= 0) p.rois[slot].checkAppendage = appendage != 0;
    }
}

}  // namespace

std::string viewDir(const std::string& dir, int view) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "/v%02d", view);
    return dir + buf;
}

Params::Params() {
    // 四组字符的实测坐标（1688x1460 原图）
    //   logo  = SC 椭圆 + Silan（同一水平线：左侧椭圆、右侧文字）
    //   line1 = Q046R8NLP
    //   line2 = SA6JZ1F
    //   code  = 保留区域（不参与字符级比对）
    rois[0] = Roi{cv::Rect(500, 400, 660, 220), "logo ", true};
    rois[1] = Roi{cv::Rect(395, 790, 880, 215), "line1", false};
    rois[2] = Roi{cv::Rect(480, 985, 720, 200), "line2", false};
    rois[3] = Roi{cv::Rect(430, 600, 300, 200), "code ", false};
}

bool saveViewModel(const std::string& dir, int view, const ViewModel& vm, std::string* err) {
    const std::string vd = viewDir(dir, view);
    std::error_code ec;
    std::filesystem::create_directories(vd, ec);

    for (int i = 0; i < 4; ++i) {
        const std::string base = vd + "/" + kRoiNames[i];
        if (vm.rois[i].sRef.empty() || vm.rois[i].freq.empty() || vm.rois[i].tolMask.empty()) {
            if (err) *err = std::string("区域模板为空: ") + kRoiNames[i];
            return false;
        }
        cv::imwrite(base + "_ref.png", vm.rois[i].sRef);
        cv::imwrite(base + "_freq.png", vm.rois[i].freq);
        cv::imwrite(base + "_tol.png", vm.rois[i].tolMask);
    }
    // 整图频率与配准基准
    if (vm.fullOk && !vm.fullFreq.empty()) {
        cv::imwrite(vd + "/full_freq.png", vm.fullFreq);
        if (!vm.fullRef.empty()) cv::imwrite(vd + "/full_ref.png", vm.fullRef);
    }
    // 历史保留文件，无实际语义
    std::ofstream os(vd + "/code_stats.txt");
    os << "0 0 0 0\n";
    return true;
}

bool saveParams(const std::string& dir, int view, const Params& p) {
    const std::string vd = viewDir(dir, view);
    std::error_code ec;
    std::filesystem::create_directories(vd, ec);
    std::ofstream os(vd + "/params.txt");
    if (!os) return false;
    // 字段位置（按位置解析，旧文件兼容）：
    //  0 sigma  1 strokeHi  2 strokeLo  3 pKeep  4 dilateK  5 edgeTB  6 edgeLR
    //  7 scratchT  8 appendageT  9 eccMin  10 scratchMinCoverage  11 scratchMinLen
    // 12 (保留)  13 (保留)  14 enabled  15 invert  16 checkMissing  17 (保留)
    // 18 missFreq  19 missStrokeLo  20 missMinArea  21 missT
    // 22 checkChars  23 charMissRatio  24 charExtraRatio  25 charT
    os << p.sigma << " " << p.strokeHi << " " << p.strokeLo << " " << p.pKeep << " " << p.dilateK
       << " " << p.edgeTB << " " << p.edgeLR << " " << p.scratchT << " " << p.appendageT << " "
       << p.eccMin << " " << p.scratchMinCoverage << " " << p.scratchMinLen << " " << 0 << " "
       << 0 << " " << (p.enabled ? 1 : 0) << " " << (p.invert ? 1 : 0) << " "
       << (p.checkMissing ? 1 : 0) << " " << 0.55 << " " << p.missFreq << " " << p.missStrokeLo
       << " " << p.missMinArea << " " << p.missT << " " << (p.checkChars ? 1 : 0) << " "
       << p.charMissRatio << " " << p.charExtraRatio << " " << p.charT << "\n";
    for (int i = 0; i < 4; ++i) {
        os << p.rois[i].rect.x << " " << p.rois[i].rect.y << " " << p.rois[i].rect.width << " "
           << p.rois[i].rect.height << " " << (p.rois[i].enabled ? 1 : 0) << "\n";
    }
    return true;
}

bool exportRoisFile(const std::string& dir, int view, const Params& p) {
    const std::string vd = viewDir(dir, view);
    std::error_code ec;
    std::filesystem::create_directories(vd, ec);
    std::ofstream os(vd + "/rois.txt");
    if (!os) return false;

    os << "# 检测区域配置（用户可编辑）\n";
    os << "#\n";
    os << "# 格式：名称  启用  x  y  宽  高  [检查多余笔画]\n";
    os << "#   x, y, 宽, 高  —— 相对原图的像素坐标（左上角开始）\n";
    os << "#   启用          —— 1=参与检测，0=跳过\n";
    os << "#   检查多余笔画  —— 1=额外做多余笔画检测（一般只有 logo 需要）\n";
    os << "#\n";
    os << "# 修改后重新评估即可生效，不需要重新训练。\n";
    os << "# 删除本文件则回退到 params.txt 里的坐标。\n";
    os << "# ---------------------------------------------------------------\n";
    for (int i = 0; i < 4; ++i) {
        std::string nm = kRoiNames[i];
        while (!nm.empty() && nm.back() == ' ') nm.pop_back();
        const cv::Rect r = p.rois[i].rect;
        os << nm << "  " << (p.rois[i].enabled ? 1 : 0) << "  " << r.x << "  " << r.y << "  "
           << r.width << "  " << r.height << "  " << (p.rois[i].checkAppendage ? 1 : 0) << "\n";
    }
    return true;
}

bool loadParams(const std::string& dir, int view, Params& p) {
    std::ifstream f(viewDir(dir, view) + "/params.txt");
    if (!f) return false;

    // 第 1 行是标量参数。按字段数判断，避免用 operator>> 读新字段时
    // 把下一行 ROI 的 x 坐标吃进来。
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) break;
    }
    {
        std::istringstream iss(line);
        std::vector<double> vals;
        double v = 0.0;
        while (iss >> v) vals.push_back(v);
        if (vals.size() < 14) return false;   // 必需字段不足

        p.sigma = vals[0];
        p.strokeHi = vals[1];
        p.strokeLo = vals[2];
        p.pKeep = vals[3];
        p.dilateK = static_cast<int>(vals[4]);
        p.edgeTB = static_cast<int>(vals[5]);
        p.edgeLR = static_cast<int>(vals[6]);
        p.scratchT = vals[7];
        p.appendageT = vals[8];
        p.eccMin = vals[9];
        p.scratchMinCoverage = vals[10];
        p.scratchMinLen = vals[11];
        // 12、13 位为历史保留位，忽略
        if (vals.size() > 14) p.enabled = vals[14] != 0.0;
        if (vals.size() > 15) p.invert = vals[15] != 0.0;
        if (vals.size() > 16) p.checkMissing = vals[16] != 0.0;
        // 17 位历史保留
        if (vals.size() > 18) p.missFreq = vals[18];
        if (vals.size() > 19) p.missStrokeLo = vals[19];
        if (vals.size() > 20) p.missMinArea = static_cast<int>(vals[20]);
        if (vals.size() > 21) p.missT = vals[21];
        if (vals.size() > 22) p.checkChars = vals[22] != 0.0;
        if (vals.size() > 23) p.charMissRatio = vals[23];
        if (vals.size() > 24) p.charExtraRatio = vals[24];
        if (vals.size() > 25) p.charT = vals[25];
    }

    // ROI 每行一个区域。第 5 字段 enabled 可缺省，必须按行解析。
    for (int i = 0; i < 4; ++i) {
        std::string roiLine;
        while (std::getline(f, roiLine)) {
            if (roiLine.find_first_not_of(" \t\r\n") != std::string::npos) break;
        }
        std::istringstream rs(roiLine);
        int x = 0, y = 0, w = 0, h = 0;
        if (!(rs >> x >> y >> w >> h)) return false;
        p.rois[i].rect = cv::Rect(x, y, w, h);
        int en = 1;
        if (rs >> en) p.rois[i].enabled = en != 0;
    }

    // 覆盖：存在 rois.txt 时以它为准
    loadRoisFile(dir, view, p);
    return true;
}

bool Model::loadFrom(const std::string& dir, std::string* err) {
    bool any = false;
    for (int v = 0; v < kViewCount; ++v) {
        ViewModel& vm = views[v];
        vm = ViewModel{};
        const std::string vd = viewDir(dir, v);
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
            const std::string base = vd + "/" + kRoiNames[i];
            cv::Mat ref = cv::imread(base + "_ref.png", cv::IMREAD_GRAYSCALE);
            cv::Mat freq = cv::imread(base + "_freq.png", cv::IMREAD_GRAYSCALE);
            cv::Mat tol = cv::imread(base + "_tol.png", cv::IMREAD_GRAYSCALE);
            if (ref.empty() || freq.empty() || tol.empty()) {
                ok = false;
                break;
            }
            vm.rois[i].sRef = ref;
            vm.rois[i].freq = freq;
            vm.rois[i].tolMask = tol;
            vm.rois[i].ok = true;
        }
        if (!ok) continue;

        // 整图频率与配准基准
        vm.fullFreq = cv::imread(vd + "/full_freq.png", cv::IMREAD_GRAYSCALE);
        vm.fullRef = cv::imread(vd + "/full_ref.png", cv::IMREAD_GRAYSCALE);
        vm.fullOk = !vm.fullFreq.empty();

        vm.ok = true;
        loadParams(dir, v, params[static_cast<size_t>(v)]);

        // 字符级模型（由 chardet 模块自包含管理）
        // 主项目不再维护 manualBoxes，改为直接持有模块的 Model
        {
            chardet::Model cm;
            if (chardet::isModelDir(vd) && chardet::loadModel(vd, cm)) {
                vm.hasManualBoxes = cm.totalChars() > 0;
                vm.hasManualRegions = !cm.regions.empty();
                vm.chardetModel = std::make_shared<chardet::Model>(std::move(cm));
            }
        }

        any = true;
    }
    if (!any) {
        if (err) *err = "模型目录下没有有效的视角子目录（期望 v00..v05）";
        return false;
    }
    return true;
}

bool Model::valid(int view) const {
    return view >= 0 && view < kViewCount && views[view].ok;
}

bool Model::anyValid() const {
    for (const auto& v : views) {
        if (v.ok) return true;
    }
    return false;
}


}  // namespace markloc::charinsp
