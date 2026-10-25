// regionfmt.cpp —— regions.txt 存盘往返验证
//
// 【为什么必须单独验这个】
// 这一轮我把 regions.txt 换成了全新的行式格式（#region <序号> <形状> ...），
// 并从 GUI 里加了"拖拽生成椭圆/圆"。
// 但"编译通过"和"存下去再读回来还是同一个东西"是两件事。
// 格式改错一样能编译通过，然后在某张图上静默地把椭圆读成矩形 ——
// 又是恒定指标。
//
// 这个工具做四件事：
//   1. 读一个模型，把每个区域的形状打出来（证明形状真的读到了）
//   2. 写到一个临时目录，再读回来，逐字段比对（证明往返无损）
//   3. 故意写一个只有椭圆没有字符的模型，确认加载后 isPattern() 为真
//   4. 确认 Box 区域的字符表没丢
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "chardet.hpp"

static const char* shapeName(chardet::RegionShape s) {
    switch (s) {
        case chardet::RegionShape::Ellipse: return "椭圆";
        case chardet::RegionShape::Circle: return "圆";
        default: return "矩形";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: markloc_regionfmt <modelDir>\n");
        return 2;
    }
    std::string dir = argv[1];
    if (dir.size() < 3 || dir.substr(dir.size() - 3) != "v00") dir += "/v00";

    chardet::Model m;
    if (!chardet::loadModel(dir, m) || !m.valid()) {
        std::printf("加载失败: %s\n", dir.c_str());
        return 1;
    }

    std::printf("=== 1. 读到的区域 ===\n");
    int nPat = 0, nBox = 0;
    for (const auto& r : m.regions) {
        if (r.isPattern()) {
            std::printf("  r%d %-4s 中心(%.1f,%.1f) 半轴%.2fx%.2f 角%.2f e=%.3f 字符%d [PATTERN]\n",
                        r.index, shapeName(r.shape), r.ell.center.x, r.ell.center.y, r.ell.a,
                        r.ell.b, r.ell.angleDeg, r.ell.eccentricity(),
                        static_cast<int>(r.chars.size()));
            ++nPat;
        } else {
            std::printf("  r%d %-4s %4dx%-4d @(%4d,%4d) 字符%2d\n", r.index, shapeName(r.shape),
                        r.rect.width, r.rect.height, r.rect.x, r.rect.y,
                        static_cast<int>(r.chars.size()));
            ++nBox;
        }
    }
    std::printf("  合计：图案区 %d 个，字符区 %d 个\n", nPat, nBox);

    // ---- 2. 往返验证
    std::printf("\n=== 2. 存盘往返 ===\n");
    chardet::Model m2;
    // 用同一份数据走一遍 save -> load
    const std::string tmp = "out/fmttest";
    m2 = m;
    if (!chardet::saveModel(tmp, m2)) {
        std::printf("  saveModel 失败\n");
        return 1;
    }
    chardet::Model m3;
    if (!chardet::loadModel(tmp, m3) || !m3.valid()) {
        std::printf("  回读失败\n");
        return 1;
    }
    bool ok = true;
    if (m3.regions.size() != m.regions.size()) {
        std::printf("  X 区域数 %d != %d\n", static_cast<int>(m3.regions.size()),
                    static_cast<int>(m.regions.size()));
        ok = false;
    } else {
        for (size_t i = 0; i < m.regions.size(); ++i) {
            const auto& a = m.regions[i];
            const auto& b = m3.regions[i];
            if (a.shape != b.shape) {
                std::printf("  X r%zu 形状 %s -> %s\n", i, shapeName(a.shape), shapeName(b.shape));
                ok = false;
            }
            if (a.rect != b.rect) {
                std::printf("  X r%zu 框 (%d,%d,%d,%d) -> (%d,%d,%d,%d)\n", i, a.rect.x, a.rect.y,
                            a.rect.width, a.rect.height, b.rect.x, b.rect.y, b.rect.width,
                            b.rect.height);
                ok = false;
            }
            if (a.chars.size() != b.chars.size()) {
                std::printf("  X r%zu 字符数 %d -> %d\n", i, static_cast<int>(a.chars.size()),
                            static_cast<int>(b.chars.size()));
                ok = false;
            } else {
                for (size_t k = 0; k < a.chars.size(); ++k) {
                    if (a.chars[k].bbox != b.chars[k].bbox || a.chars[k].index != b.chars[k].index) {
                        std::printf("  X r%zu 字符%zu 不一致\n", i, k);
                        ok = false;
                        break;
                    }
                }
            }
            std::printf("  r%zu %-4s 字符%2d  %s\n", i, shapeName(b.shape),
                        static_cast<int>(b.chars.size()), "一致");
        }
    }
    std::printf("  %s\n", ok ? "OK 往返无损" : "X 往返有损");

    // ---- 3. 图案区识别
    std::printf("\n=== 3. 图案区判定 ===\n");
    for (const auto& r : m3.regions) {
        const bool expectPattern = (r.shape != chardet::RegionShape::Box);
        if (r.isPattern() != expectPattern) {
            std::printf("  X r%d isPattern()=%d 与形状不符\n", r.index, r.isPattern());
            ok = false;
        }
    }
    if (nPat == 0) {
        std::printf("  【注意】这个模型里没有图案区，图案断点检测不会执行。\n");
    } else {
        std::printf("  %d 个图案区会被断点检测处理\n", nPat);
    }

    std::printf("\n%s\n", ok ? "全部通过" : "有失败项");
    return ok ? 0 : 1;
}
