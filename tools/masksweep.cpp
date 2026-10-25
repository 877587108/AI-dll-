// 全库掩膜覆盖度扫描
//
// 0109 的未覆盖率是 40%，正常图 9%。要确认这是孤例还是普遍现象，
// 必须对 ok/ 全部 84 张逐一量，并标出哪些图的未覆盖率异常高。
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "chardet.hpp"
#include "gui/scratch.hpp"

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
    if (argc < 3) {
        std::printf("用法: masksweep <modelDir> <dirOrListFile>\n");
        return 2;
    }
    chardet::Model model;
    if (!chardet::loadModel(argv[1], model)) {
        std::printf("模型载入失败\n");
        return 1;
    }
    const chardet::Config cfg = model.config;
    const std::vector<chardet::CharUnit> base = chardet::toAllUnits(model);
    markloc::ScratchParams P;

    std::vector<std::string> files;
    FILE* lf = std::fopen(argv[2], "r");
    if (lf) {
        char line[2048];
        while (std::fgets(line, sizeof(line), lf)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty()) files.push_back(s);
        }
        std::fclose(lf);
    }

    std::printf("%-52s %8s %8s\n", "image", "未覆盖%", "加粗后%");
    for (const auto& f : files) {
        cv::Mat gray;
        if (!readGray(f, gray)) continue;
        cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
        const cv::Rect band = markloc::scratchBand(sig.size(), base, P.growPx);
        const cv::Mat sBand = sig(band);
        const cv::Mat fBand = model.freq(band);
        cv::Mat weak;
        cv::compare(sBand, cfg.missStrokeLo * 0.7, weak, cv::CMP_GT);
        const int sigCnt = cv::countNonZero(weak);
        if (sigCnt == 0) continue;

        auto uncovered = [&](int dk) {
            cv::Mat ink;
            cv::compare(fBand, static_cast<int>(cfg.blankFreq * 255), ink, cv::CMP_GT);
            const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {dk, dk});
            cv::Mat dil, nm, left;
            cv::dilate(ink, dil, kd);
            cv::bitwise_not(dil, nm);
            cv::bitwise_and(weak, nm, left);
            return 100.0 * cv::countNonZero(left) / std::max(1, sigCnt);
        };
        const double u9 = uncovered(9);
        const double u41 = uncovered(41);
        // 只打出文件名尾部，路径太长看不清
        std::string shortName = f;
        const size_t slash = shortName.find_last_of("/\\");
        if (slash != std::string::npos) shortName = shortName.substr(slash + 1);
        if (shortName.size() > 50) shortName = shortName.substr(0, 50);
        std::printf("%-52s %7.1f%% %7.1f%%%s\n", shortName.c_str(), u9, u41,
                    u9 > 25.0 ? "   <== 高" : "");
    }
    return 0;
}
