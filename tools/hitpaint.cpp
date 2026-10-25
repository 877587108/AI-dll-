// 把 findScratches 的真实命中画回原图，供人眼判"这到底是不是真划痕"
//
// 为什么必须先做这一步：今晚已经出现 6 个"讲得通但被实测推翻"的根因假设
// （strokeHi 太低、归一化压低笔画、模型字形糊、HI 取值、LO=p50、模板可当掩膜）。
// 每一次都是先有推理再去找证据。这次顺序反过来：
// 先把命中的位置摆到图上，看清它压着什么，再谈原因。
//
// 输出三联：原图 | 命中线段(绿) | 信号热图叠加
//
// 用法: markloc/hitpaint <modelDir> <image> <out.png>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

namespace fs = std::filesystem;

static bool readGray(const std::string& path, cv::Mat& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::vector<unsigned char> buf;
    unsigned char tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof(tmp), fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(fp);
    if (buf.empty()) return false;
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: markloc_hitpaint <modelDir> <image> <out.png>\n");
        return 2;
    }
    chardet::Model model;
    std::string vdir = std::string(argv[1]) + "/v00";
    if (!chardet::loadModel(fs::exists(fs::path(vdir)) ? vdir : std::string(argv[1]), model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config& cfg = model.config;
    cv::Mat gray;
    if (!readGray(argv[2], gray) || gray.size() != model.freq.size()) {
        std::printf("读取失败或尺寸不符\n");
        return 1;
    }
    std::vector<chardet::CharUnit> units;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) units.push_back(std::move(x));
    }
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat r32;
    model.ref.convertTo(r32, CV_32F);
    cv::Rect inkArea;
    for (const auto& u : units) {
        const cv::Rect b = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (b.area() > 0) inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
    }
    inkArea.x = std::max(0, inkArea.x - 20);
    inkArea.y = std::max(0, inkArea.y - 20);
    inkArea.width = std::min(sig.cols - inkArea.x, inkArea.width + 40);
    inkArea.height = std::min(sig.rows - inkArea.y, inkArea.height + 40);
    cv::Mat sC, rC;
    sig(inkArea).copyTo(sC);
    r32(inkArea).copyTo(rC);
    const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
    if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
        (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
        cv::Mat al;
        cv::warpAffine(sig, al, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
        sig = al;
    }
    for (auto& u : units) {
        const cv::Rect box = u.bbox & cv::Rect(0, 0, sig.cols, sig.rows);
        if (box.width <= 4 || box.height <= 4) continue;
        constexpr int kSearch = 12;
        cv::Rect search(box.x - kSearch, box.y - kSearch, box.width + 2 * kSearch,
                        box.height + 2 * kSearch);
        search &= cv::Rect(0, 0, sig.cols, sig.rows);
        if (search.width < box.width || search.height < box.height) continue;
        cv::Mat mS, tS;
        model.freq(search).convertTo(mS, CV_32F, 1.0 / 255.0);
        sig(search).convertTo(tS, CV_32F);
        cv::Mat res;
        cv::matchTemplate(
            tS, mS(cv::Rect(box.x - search.x, box.y - search.y, box.width, box.height)), res,
            cv::TM_CCOEFF_NORMED);
        cv::Point mLoc;
        double mVal = 0;
        cv::minMaxLoc(res, nullptr, &mVal, nullptr, &mLoc);
        if (mVal < 0.30) continue;
        const cv::Rect moved(box.x + mLoc.x - kSearch, box.y + mLoc.y - kSearch, box.width,
                             box.height);
        if ((moved & cv::Rect(0, 0, sig.cols, sig.rows)) != moved) continue;
        u.bbox = moved;
    }

    markloc::ScratchParams P;
    const auto hits = markloc::findScratches(sig, model.freq, units, cfg, P);
    std::printf("命中 %zu 条\n", hits.size());

    // 左：原图 + 绿线
    cv::Mat left;
    cv::cvtColor(gray, left, cv::COLOR_GRAY2BGR);
    // 中：信号热图（放大到 0..1.5 便于看）
    cv::Mat heat;
    sig.convertTo(heat, CV_8U, 255.0 / 1.6);
    cv::applyColorMap(heat, heat, cv::COLORMAP_INFERNO);
    // 右：模型 freq
    cv::Mat mid = heat.clone();

    for (const auto& h : hits) {
        cv::line(left, h.a1, h.a2, cv::Scalar(0, 255, 0), 2);
        cv::line(mid, h.a1, h.a2, cv::Scalar(0, 255, 0), 2);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "S%d", static_cast<int>(h.composite));
        cv::putText(left, buf, cv::Point(h.a1.x + 4, h.a1.y - 4), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);
    }
    for (const auto& u : units) {
        cv::rectangle(left, u.bbox, cv::Scalar(0, 180, 255), 1);
        cv::rectangle(mid, u.bbox, cv::Scalar(0, 180, 255), 1);
    }
    cv::Mat out;
    cv::hconcat(std::vector<cv::Mat>{left, mid}, out);
    fs::create_directories(fs::path(argv[3]).parent_path());
    cv::imwrite(argv[3], out);
    std::printf("-> %s\n", argv[3]);
    std::printf("（左=原图+绿线+S值，右=信号热图+绿线）\n");
    return 0;
}
