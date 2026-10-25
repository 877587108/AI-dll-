// pattern_bench.cpp —— 图案检测基准台
//
// ================================================================
// 用途
// ================================================================
//
//  ① 回归验证：重构前后用同一个输入跑，数值必须一致
//  ② 横向比较：同一张图上并排看多个检测器各自的结论
//  ③ 批量评测：给定目录，统计各检测器的检出/误报
//
// ================================================================
// 用法
// ================================================================
//
//   markloc_patternbench <modelRoot> <tmpl.png> <list.txt> [选项]
//
//   选项：
//     --detectors a,b,c   只用这些检测器（默认全部已注册的）
//     --csv out.csv       逐张明细写入 CSV
//     --dump-polar DIR    把每张的极坐标图存到 DIR
//     --dump-profile DIR  把每张的剖面曲线存到 DIR
//     --legacy            同时跑老接口 analyzePatternBreak() 做对照
//
//   list.txt 每行一个图片路径（UTF-8）。
//   列表文件而不是命令行参数：Windows 命令行代码页会把中文路径破坏掉。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "pattern/pattern_api.hpp"

namespace {

bool readGray(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray b = f.readAll();
    f.close();
    if (b.isEmpty()) return false;
    const std::vector<uchar> buf(b.begin(), b.end());
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

struct Options {
    std::string modelRoot;
    std::string tmplPath;
    std::string listPath;
    std::vector<std::string> detectors;
    std::string csvPath;
    std::string dumpPolarDir;
    std::string dumpProfileDir;
    bool legacy = false;
};

void usage() {
    std::printf(
        "用法: markloc_patternbench <modelRoot> <tmpl.png> <list.txt> [选项]\n"
        "  --detectors a,b,c   只用这些检测器\n"
        "  --csv FILE          明细写 CSV\n"
        "  --dump-polar DIR    存极坐标图\n"
        "  --dump-profile DIR  存剖面曲线\n"
        "  --legacy            同时跑老接口做对照\n");
}

/// 从模型里取 logo 区域的椭圆
///
/// 用区域序号匹配：模型 regions.txt 里形状为 ellipse/circle 的区域。
bool findPatternRegion(const chardet::Model& model, chardet::RegionInfo& out) {
    for (const auto& r : model.regions) {
        if (r.isPattern()) {
            out = r;
            return true;
        }
    }
    return false;
}

/// 在整图上定位 logo 并生成该图的椭圆
///
/// 这是**唯一的**"把模型里的椭圆搬到实际图上"的实现。
/// 以前每个探针各写一份模板匹配 + 偏移，互相之间不一致，
/// 导致同一条记录在不同实验里数字对不上。
bool locatePattern(const chardet::Model& model, const cv::Mat& tmpl, const cv::Mat& gray,
                   const chardet::RegionInfo& modelRegion, chardet::RegionInfo& out,
                   cv::Point2d* mtmpl = nullptr) {
    if (tmpl.empty() || gray.empty()) return false;
    if (gray.cols < tmpl.cols || gray.rows < tmpl.rows) return false;

    cv::Mat mres;
    cv::matchTemplate(gray, tmpl, mres, cv::TM_CCOEFF_NORMED);
    cv::Point mloc;
    double mval = 0;
    cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
    if (mtmpl) *mtmpl = cv::Point2d(mloc.x, mloc.y);

    // 模板左上角在整图中的位置 = mloc。
    // 模型椭圆是**模型坐标系**里的：模型模板就是 ref，其原点与模型图一致。
    // 所以实际椭圆 = 模型椭圆 + (mloc - 模型里模板的放置位置)。
    // 模型里模板放置位置由建模时决定，这里是 0（模板即整图左上角裁出）。
    out = modelRegion;
    out.ell.center.x += static_cast<float>(mloc.x);
    out.ell.center.y += static_cast<float>(mloc.y);
    out.syncRectFromEll();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::printf("选项 %s 缺少参数\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--detectors") {
            const std::string v = need("--detectors");
            size_t s = 0;
            while (s <= v.size()) {
                const size_t e = v.find(',', s);
                const std::string one = v.substr(s, e == std::string::npos ? std::string::npos
                                                                           : e - s);
                if (!one.empty()) opt.detectors.push_back(one);
                if (e == std::string::npos) break;
                s = e + 1;
            }
        } else if (a == "--csv") {
            opt.csvPath = need("--csv");
        } else if (a == "--dump-polar") {
            opt.dumpPolarDir = need("--dump-polar");
        } else if (a == "--dump-profile") {
            opt.dumpProfileDir = need("--dump-profile");
        } else if (a == "--legacy") {
            opt.legacy = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() < 3) {
        usage();
        return 2;
    }
    opt.modelRoot = pos[0];
    opt.tmplPath = pos[1];
    opt.listPath = pos[2];

    // ---- 载入模型
    chardet::Model model;
    const QString vdir = QDir(QString::fromUtf8(opt.modelRoot.c_str()))
                             .absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(vdir.toStdString(), model) || !model.valid()) {
        std::printf("模型加载失败: %s\n", vdir.toUtf8().constData());
        return 1;
    }
    chardet::RegionInfo modelRegion;
    if (!findPatternRegion(model, modelRegion)) {
        std::printf("模型里没有椭圆/圆区域\n");
        return 1;
    }

    cv::Mat tmpl;
    if (!readGray(QString::fromUtf8(opt.tmplPath.c_str()), tmpl)) {
        std::printf("模板读取失败\n");
        return 1;
    }

    // ---- 读列表
    std::vector<std::string> files;
    {
        QFile f(QString::fromUtf8(opt.listPath.c_str()));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            std::printf("列表读取失败: %s\n", opt.listPath.c_str());
            return 1;
        }
        while (!f.atEnd()) {
            const QString line = QString::fromUtf8(f.readLine()).trimmed();
            if (!line.isEmpty()) files.push_back(line.toStdString());
        }
    }
    if (files.empty()) {
        std::printf("列表为空\n");
        return 1;
    }

    // ---- 参数
    chardet::pattern::Params params;
    if (!opt.detectors.empty()) params.detectors = opt.detectors;
    // 校验检测器名字
    const auto known = chardet::pattern::detectorNames();
    for (const auto& d : params.detectors) {
        if (std::find(known.begin(), known.end(), d) == known.end()) {
            std::printf("未知检测器: %s\n已注册: ", d.c_str());
            for (const auto& k : known) std::printf("%s ", k.c_str());
            std::printf("\n");
            return 2;
        }
    }

    std::printf("模型: %s\n", opt.modelRoot.c_str());
    std::printf("检测器: ");
    for (const auto& d : params.detectors) std::printf("%s ", d.c_str());
    std::printf("\n样本数: %zu\n\n", files.size());

    // ---- 表头
    std::printf("%-40s %8s", "file", "match");
    for (const auto& d : params.detectors) {
        std::printf(" | %-28s", d.c_str());
    }
    if (opt.legacy) std::printf(" | %-18s", "legacy score");
    std::printf("\n");
    std::printf("%s\n", std::string(40 + 9 + params.detectors.size() * 31 + 20, '-').c_str());

    FILE* csv = nullptr;
    if (!opt.csvPath.empty()) {
        csv = std::fopen(opt.csvPath.c_str(), "w");
        if (csv) {
            std::fprintf(csv, "file,match");
            for (const auto& d : params.detectors) {
                std::fprintf(csv, ",%s_sev,%s_trig,%s_score", d.c_str(), d.c_str(), d.c_str());
            }
            if (opt.legacy) std::fprintf(csv, ",legacy_score,legacy_flag");
            std::fprintf(csv, "\n");
        }
    }

    int nOk = 0, nFail = 0, nFlagged = 0;
    for (const auto& path : files) {
        const QString qp = QString::fromUtf8(path.c_str());
        cv::Mat gray;
        if (!readGray(qp, gray)) {
            std::printf("%-40s 读取失败\n", QFileInfo(qp).fileName().toUtf8().constData());
            ++nFail;
            continue;
        }
        chardet::RegionInfo region;
        cv::Point2d mt;
        if (!locatePattern(model, tmpl, gray, modelRegion, region, &mt)) {
            std::printf("%-40s 定位失败\n", QFileInfo(qp).fileName().toUtf8().constData());
            ++nFail;
            continue;
        }

        const cv::Mat sig =
            chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), model.config, false);

        // 模型椭圆在模型坐标系；模板匹配得到的偏移量补上
        // （模型椭圆按模型原图坐标记录，实际图尺寸不同时用模板位置换算）
        region.ell.center.x = modelRegion.ell.center.x + static_cast<float>(mt.x);
        region.ell.center.y = modelRegion.ell.center.y + static_cast<float>(mt.y);
        region.syncRectFromEll();

        const auto res = chardet::pattern::analyze(sig, region, model.config, params);
        const QString base = QFileInfo(qp).completeBaseName();
        const std::string name = QFileInfo(qp).fileName().toStdString();

        std::printf("%-40s %8.3f", name.substr(0, 40).c_str(), mt.x >= 0 ? mt.x : 0.0);
        if (!res.valid) {
            std::printf(" | 预处理失败\n");
            ++nFail;
            continue;
        }
        ++nOk;
        if (res.flagged) ++nFlagged;

        if (csv) std::fprintf(csv, "\"%s\",%.1f", name.c_str(), mt.x);

        for (const auto& d : params.detectors) {
            const auto* f = res.find(d);
            if (!f) {
                std::printf(" | %-28s", "(无)");
                if (csv) std::fprintf(csv, ",,,,");
                continue;
            }
            std::printf(" | %s sev=%.2f%s", f->triggered ? "NG" : "ok", f->severity,
                        f->triggered ? " *" : "  ");
            // 补齐列宽
            for (int k = 0; k < 8; ++k) std::printf(" ");
            if (csv) {
                std::fprintf(csv, ",%.4f,%d,%.4f", f->severity, f->triggered ? 1 : 0, f->score);
            }
        }

        if (opt.legacy) {
            chardet::PatternBreakParams LP;
            const auto legacy = chardet::analyzePatternBreak(sig, region, model.config, LP);
            std::printf(" | %6.0f %s", legacy.score, legacy.flagged ? "NG" : "ok");
            if (csv) std::fprintf(csv, ",%.1f,%d", legacy.score, legacy.flagged ? 1 : 0);
        }
        std::printf("\n");

        // 调试输出
        if (!opt.dumpPolarDir.empty()) {
            chardet::pattern::dumpPolar(
                res.polar,
                opt.dumpPolarDir + "/" + base.toStdString() + "_polar.png");
        }
        if (!opt.dumpProfileDir.empty()) {
            chardet::pattern::dumpProfile(
                res.profile, res.polar,
                opt.dumpProfileDir + "/" + base.toStdString() + "_profile.csv");
        }
    }
    if (csv) std::fclose(csv);

    std::printf("\n合计: 有效 %d, 失败 %d, 判NG %d\n", nOk, nFail, nFlagged);
    return 0;
}
