// 命令行重训：把 strokeHi 等训练参数从"编译期默认值"变成"可以实验的变量"
//
// 背景：GUI 的训练入口（SplitVerifyWindow.cpp:439）用的是默认 Config，
// 想试不同的 strokeHi 就得改代码再重编、再点界面。实验成本太高，
// 于是每次"改一处验一处"都变成一次性大改动。
//
// 本工具把 train() 直接暴露到命令行：
//   markloc_retrain <imageDir> <outModelDir> [--strokeHi 0.20] [--sigma 15]
//                  [--regions <regions.txt>] [--view 0]
//
// 只取 <stem>_<view> 结尾的图，避免把 6 个视角混在一起。
// regions.txt 缺省时从 outModelDir 的上一级 v00 复制——训练只产出 freq/ref，
// 区域是人工划的，不能重算。
//
// 用法示例：
//   markloc_retrain image/ok model/v00 --strokeHi 0.20 --regions model/v00/regions.txt
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

namespace fs = std::filesystem;

static fs::path u8p(const std::string& s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

static bool readGray(const fs::path& path, cv::Mat& out) {
    FILE* fp = std::fopen(path.string().c_str(), "rb");
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
    if (argc < 3) {
        std::printf(
            "用法: markloc_retrain <imageDir> <outModelDir> [--strokeHi 0.20] [--sigma 15]\n"
            "                     [--view 0] [--regions path/to/regions.txt] [--binFreq 0.25]\n"
            "                     [--searchRadius 30] [--minPeak 0.60] [--noRefine 1]\n");
        return 2;
    }
    const std::string imgDir = argv[1];
    const std::string outDir = argv[2];

    chardet::Config cfg;   // 从默认值出发，只覆盖点名的项
    int view = 0;
    std::string regionsPath;
    bool noRefine = false;
    int searchRadius = 30;
    double minPeak = 0.60;

    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--strokeHi") cfg.strokeHi = std::atof(argv[i + 1]);
        else if (k == "--sigma") cfg.sigma = std::atof(argv[i + 1]);
        else if (k == "--missStrokeLo") cfg.missStrokeLo = std::atof(argv[i + 1]);
        else if (k == "--blankFreq") cfg.blankFreq = std::atof(argv[i + 1]);
        else if (k == "--binFreq") cfg.binFreq = std::atof(argv[i + 1]);
        else if (k == "--view") view = std::atoi(argv[i + 1]);
        else if (k == "--regions") regionsPath = argv[i + 1];
        else if (k == "--searchRadius") searchRadius = std::atoi(argv[i + 1]);
        else if (k == "--minPeak") minPeak = std::atof(argv[i + 1]);
        else if (k == "--noRefine") noRefine = (std::atoi(argv[i + 1]) != 0);
    }

    // 收集该视角的图
    const std::string suffix = cv::format("_%02d", view);
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(u8p(imgDir))) {
        if (!e.is_regular_file()) continue;
        const std::string stem = e.path().stem().string();
        if (stem.size() < suffix.size()) continue;
        if (stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::printf("没有找到 _%02d 视角的图\n", view);
        return 1;
    }

    chardet::TrainInput in;
    in.view = view;
    int skipped = 0;
    for (const auto& p : files) {
        cv::Mat g;
        if (!readGray(p, g) || g.empty()) { ++skipped; continue; }
        in.images.push_back(std::move(g));
    }
    std::printf("样本: %zu 张（跳过 %d）  视角 _%02d\n", in.images.size(), skipped, view);
    std::printf("参数: strokeHi=%.3f sigma=%.2f missStrokeLo=%.3f blankFreq=%.3f binFreq=%.3f\n",
                cfg.strokeHi, cfg.sigma, cfg.missStrokeLo, cfg.blankFreq, cfg.binFreq);

    chardet::TrainReport rep;
    chardet::Model m = chardet::train(in, cfg, &rep);
    if (m.freq.empty()) {
        std::printf("训练失败: %s\n", rep.message.c_str());
        return 1;
    }
    std::printf("粗训练完成: 用图 %d 张，耗时 %.0f ms\n", rep.usedImages, rep.elapsedMs);

    // ---- 精对齐（refine）
    //
    // 这一步不能省。现有 release 模型（freq.png = versions/20261021_105337_refined.png）
    // 就是 refine 的产物，不是 train 的粗模型。用粗模型做对照实验会得出
    // "重训更差"的假结论：实测 strokeHi=0.20 粗模型误报 12 张，
    // 而现有的精对齐模型只有 4 张——差别来自对齐，不是来自阈值。
    if (!noRefine) {
        // refine 需要 regions（逐字符框），先载入
        const auto loadRegionsTxt = [](const fs::path& p,
                                       std::vector<chardet::RegionInfo>& out) -> bool {
            std::ifstream is(p);
            if (!is) return false;
            size_t nRegion = 0;
            if (!(is >> nRegion)) return false;
            for (size_t i = 0; i < nRegion; ++i) {
                chardet::RegionInfo r;
                size_t nChar = 0;
                if (!(is >> r.index >> r.rect.x >> r.rect.y >> r.rect.width >> r.rect.height >>
                      nChar))
                    break;
                for (size_t k = 0; k < nChar; ++k) {
                    chardet::CharBox c;
                    if (!(is >> c.bbox.x >> c.bbox.y >> c.bbox.width >> c.bbox.height >>
                          c.extend.x >> c.extend.y >> c.extend.width >> c.extend.height >>
                          c.index))
                        break;
                    r.chars.push_back(c);
                }
                out.push_back(std::move(r));
            }
            return !out.empty();
        };
        fs::path rp0 = regionsPath.empty() ? u8p(outDir) / "regions.txt" : u8p(regionsPath);
        std::vector<chardet::RegionInfo> regs;
        if (!loadRegionsTxt(rp0, regs)) {
            std::printf("精对齐需要 regions.txt，读取失败: %s\n", rp0.string().c_str());
            return 1;
        }
        m.regions = std::move(regs);
        std::printf("精对齐: %zu 区域 %d 字符，搜索半径 %d\n", m.regions.size(), m.totalChars(),
                    searchRadius);
        chardet::RefineReport rrep;
        chardet::Model refined = chardet::refine(in.images, m, cfg, searchRadius, minPeak, &rrep);
        if (!refined.freq.empty()) {
            std::printf("精对齐完成: 成功 %d 对，跳过 %d，峰值均值 %.2f，耗时 %.0f ms\n",
                        rrep.refinedPairs, rrep.skippedWeak, rrep.meanPeak, rrep.elapsedMs);
            m = std::move(refined);
        } else {
            std::printf("精对齐失败（%s），保留粗模型\n", rrep.message.c_str());
        }
    }

    // 频率图统计：改 strokeHi 后，freq 的分布应该整体抬高
    int nz = 0, above38 = 0;
    long sum = 0;
    for (int y = 0; y < m.freq.rows; ++y)
        for (int x = 0; x < m.freq.cols; ++x) {
            const int v = m.freq.at<uchar>(y, x);
            if (v > 0) ++nz;
            if (v > 38) ++above38;
            sum += v;
        }
    const long tot = static_cast<long>(m.freq.rows) * m.freq.cols;
    std::printf("freq: 非零 %.1f%%  >38 的像素 %.2f%%  均值 %.2f\n", 100.0 * nz / tot,
                100.0 * above38 / tot, static_cast<double>(sum) / tot);

    // 区域：训练不产出，从指定路径或已存在的输出目录读取。
    // 格式见 model.cpp:507-527 —— 先读区域数，再每个区域一行
    //   index x y w h nChar
    // 后跟 nChar 行
    //   x y w h extX extY extW extH index
    const auto loadRegionsTxt = [](const fs::path& p,
                                   std::vector<chardet::RegionInfo>& out) -> bool {
        std::ifstream is(p);
        if (!is) return false;
        size_t nRegion = 0;
        if (!(is >> nRegion)) return false;
        for (size_t i = 0; i < nRegion; ++i) {
            chardet::RegionInfo r;
            size_t nChar = 0;
            if (!(is >> r.index >> r.rect.x >> r.rect.y >> r.rect.width >> r.rect.height >> nChar))
                break;
            for (size_t k = 0; k < nChar; ++k) {
                chardet::CharBox c;
                if (!(is >> c.bbox.x >> c.bbox.y >> c.bbox.width >> c.bbox.height >> c.extend.x >>
                      c.extend.y >> c.extend.width >> c.extend.height >> c.index))
                    break;
                r.chars.push_back(c);
            }
            out.push_back(std::move(r));
        }
        return !out.empty();
    };

    fs::path rp = regionsPath.empty() ? u8p(outDir) / "regions.txt" : u8p(regionsPath);
    if (fs::exists(rp)) {
        std::vector<chardet::RegionInfo> regs;
        if (loadRegionsTxt(rp, regs)) {
            m.regions = std::move(regs);
            std::printf("区域: %zu 个，字符 %d 个（来自 %s）\n", m.regions.size(), m.totalChars(),
                        rp.string().c_str());
        } else {
            std::printf("区域读取失败: %s\n", rp.string().c_str());
        }
    } else {
        std::printf("警告: 没找到 regions.txt（%s），模型将没有字符框\n", rp.string().c_str());
    }

    if (!chardet::saveModel(outDir, m)) {
        std::printf("保存失败\n");
        return 1;
    }
    std::printf("已写入 %s\n", outDir.c_str());

    // 读回验证
    chardet::Model back;
    if (!chardet::loadModel(outDir, back)) {
        std::printf("读回失败\n");
        return 1;
    }
    std::printf("读回: freq %dx%d  区域 %zu  字符 %d  strokeHi=%.3f\n", back.freq.cols,
                back.freq.rows, back.regions.size(), back.totalChars(), back.config.strokeHi);
    if (back.regions.empty()) {
        std::printf("警告: 读回后区域为空，模型不可用于检测\n");
        return 1;
    }
    return 0;
}
