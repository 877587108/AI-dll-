// 标注绘制的共享实现。
//
// 抽成头文件是为了让验证工具能用**同一份代码**渲染出图：如果验证工具自己
// 再抄一遍绘制逻辑，验证的就不是 GUI 真正跑的东西，早晚会分叉。
#pragma once

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace markloc {

/// 把 UTF-8 文本里非 ASCII 的部分去掉。
///
/// 这是**画进图像**的文字的最后一道保险。cv::putText 用的 Hershey 矢量字体
/// 只有 ASCII 字形：中文字符的每个字节都会被当成字形索引，画出来是一串问号。
/// 调用点本应自己传英文，但漏一次就是满屏问号，所以在最底层再挡一次。
inline std::string asciiOnly(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c < 0x80) out += static_cast<char>(c);
    }
    return out;
}

/// 中文标签 → ASCII 缩写（画进图像前必须过这一道）
///
/// cv::putText 用的是 Hershey 矢量字体，不含任何中文字形：传中文进去，
/// 每个字节都被当成一个字形索引，画出来就是一串问号。
/// 与其指望每个调用点都记得写英文，不如在这里统一兜底。
///
/// 注意不能"遇到中文就整段丢弃"：像 "缺墨 12%" 译完中文前缀后，
/// 剩下的 " 12%" 是有效数字，必须保留。
inline std::string asciiLabel(const std::string& label) {
    struct Map {
        const char* zh;
        const char* en;
    };
    static const Map kMap[] = {
        {"溢料", "BURR"},     {"污损", "DIRT"},   {"缺墨", "MISS"},
        {"多墨", "EXTRA"},    {"刻错?", "WRONG"}, {"刻错", "WRONG"},
        {"可疑", "SUSP?"},    {"缺字", "MISSC"},  {"多字", "EXTRAC"},
        {"划痕", "SCRATCH"},  {"轻微", "FAINT"},  {"明显", "OBVIOUS"},
        {"确定", "CERTAIN"},  {"字符", "CHAR"},   {"整行", "ROW"},
        {"模板", "TMPL"},     {"原图", "RAW"},    {"标注", "MARK"},
    };
    std::string out;
    std::string rest = label;
    while (!rest.empty()) {
        bool hit = false;
        for (const auto& m : kMap) {
            const size_t n = std::strlen(m.zh);
            if (rest.compare(0, n, m.zh) == 0) {
                if (!out.empty() && out.back() != ' ') out += ' ';
                out += m.en;
                rest.erase(0, n);
                hit = true;
                break;
            }
        }
        if (hit) continue;
        const unsigned char c = static_cast<unsigned char>(rest[0]);
        if (c < 0x80) {
            if (c != ' ' || (!out.empty() && out.back() != ' ')) out += static_cast<char>(c);
        }
        // 非 ASCII 字节逐个跳过：一个 UTF-8 汉字的三字节会被整体吃掉
        rest.erase(0, 1);
    }
    while (!out.empty() && out.front() == ' ') out.erase(0, 1);
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

/// 在框的右侧画一个多行文字面板。
///
/// OpenCV 的 putText 只能画 ASCII，所以中文只能靠编号/英文缩写表达。放右侧
/// 是因为缺陷本身就在框内，文字压上去会挡住判断；右边空间不够时翻到左边。
inline void drawNoteRight(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color,
                          const std::vector<std::string>& lines, int slot = 0) {
    if (lines.empty() || img.empty()) return;
    const cv::Rect rr = r & cv::Rect(0, 0, img.cols, img.rows);
    if (rr.width <= 0 || rr.height <= 0) return;

    constexpr double kScale = 0.62;   // 原 0.42：字太小，1688 宽的图上几乎看不清
    constexpr int kThick = 1;
    int baseline = 0;
    int textW = 0;
    int lineH = 0;
    // 先清洗再量尺寸：不清洗的话，量出来的是"问号串"的宽度，面板会画歪
    std::vector<std::string> safe;
    safe.reserve(lines.size());
    for (const auto& s : lines) {
        const std::string a = asciiOnly(s);
        if (a.empty()) continue;   // 整行都是中文 → 画出来是垃圾，干脆不画
        safe.push_back(a);
    }
    if (safe.empty()) return;
    for (const auto& s : safe) {
        const cv::Size ts = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, kScale, kThick, &baseline);
        textW = std::max(textW, ts.width);
        lineH = std::max(lineH, ts.height + baseline);
    }
    lineH += 2;
    constexpr int kPadX = 4, kPadY = 4;
    const int panelW = textW + kPadX * 2;
    const int panelH = lineH * static_cast<int>(safe.size()) + kPadY * 2;

    int px = rr.x + rr.width + 4;
    if (px + panelW > img.cols) px = rr.x - panelW - 4;   // 右边放不下就翻到左边
    px = std::clamp(px, 0, std::max(0, img.cols - panelW));
    int py = rr.y + rr.height / 2 - panelH / 2;
    // slot：同一位置的多条说明叠在一起会互相盖住，按顺序往下错开一行高度
    if (slot > 0) py += slot * (lineH + 4);
    py = std::clamp(py, 0, std::max(0, img.rows - panelH));

    const cv::Rect panel(px, py, panelW, panelH);
    // 白底黑字：原来用档位色的实底 + 黑字，红底黑字在暗图上对比度太低，
    // 缩略图和整窗截图里基本读不出来。改为白底 + 档位色描边，
    // 既保持"哪一档"的颜色信息，又保证文字始终可读。
    cv::rectangle(img, panel, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(img, panel, color, 2);
    int ty = py + kPadY;
    for (const auto& s : safe) {
        ty += lineH;
        cv::putText(img, s, cv::Point(px + kPadX, ty - 4), cv::FONT_HERSHEY_SIMPLEX, kScale,
                    cv::Scalar(0, 0, 0), kThick, cv::LINE_AA);
    }
    // 面板与框之间连一条细线，避免在多条缺陷时看串行
    cv::line(img, cv::Point(px < rr.x ? px + panelW : px, py + panelH / 2),
             cv::Point(px < rr.x ? rr.x : rr.x + rr.width, py + panelH / 2), color, 1, cv::LINE_AA);
}

/// 在图片上画一块**合并**的说明面板（所有划痕的说明集中在一处）。
///
/// 为什么不再每条划痕各画一块：
///   每条各画一块时，n 条划痕就有 n 块白底面板。它们都贴着各自的框找位置，
///   在 Silan 那种字挤在一起的地方就会连成一片白斑，把划痕本身和字符一起盖掉。
///   实测一张图两条划痕就足以让标注联看起来"全是看不懂的框"。
///   合并成一块、固定在图片左侧空白处，编号用 1/2/3 与框对应。
///
/// entries: 每项是一组行（第一行带编号），color 是这一组对应的档位色。
inline void drawLegendPanel(cv::Mat& img,
                            const std::vector<std::pair<std::vector<std::string>, cv::Scalar>>& entries) {
    if (entries.empty() || img.empty()) return;

    constexpr double kScale = 0.72;
    constexpr int kThick = 1;
    constexpr int kPadX = 8, kPadY = 8, kGap = 6;

    int baseline = 0;
    int textW = 0;
    int lineH = 0;
    int lineCount = 0;
    // 先清洗再量尺寸：不清洗量到的是问号串的宽度，面板会画歪
    std::vector<std::pair<std::vector<std::string>, cv::Scalar>> safe;
    safe.reserve(entries.size());
    for (const auto& e : entries) {
        std::vector<std::string> lines;
        for (const auto& s : e.first) {
            const std::string a = asciiOnly(s);
            if (!a.empty()) lines.push_back(a);
        }
        if (lines.empty()) continue;
        for (const auto& s : lines) {
            const cv::Size ts = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, kScale, kThick, &baseline);
            textW = std::max(textW, ts.width);
            lineH = std::max(lineH, ts.height + baseline);
            ++lineCount;
        }
        safe.push_back({lines, e.second});
    }
    if (safe.empty()) return;
    lineH += 3;

    const int panelW = textW + kPadX * 2;
    const int panelH = lineH * lineCount + kGap * static_cast<int>(safe.size() - 1) + kPadY * 2;
    if (panelW + 8 >= img.cols) return;   // 图太窄，放不下面板就不画，别盖住证据

    // 固定在左上角空白处。原图的芯片本体通常在中间偏左，
    // 左上角是封装边缘/背景，压住的信息量最小。
    const int px = 6;
    const int py = 6;
    const cv::Rect panel(px, py, panelW, panelH);
    cv::rectangle(img, panel, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(img, panel, cv::Scalar(60, 60, 60), 2);

    int ty = py + kPadY;
    for (size_t gi = 0; gi < safe.size(); ++gi) {
        const auto& lines = safe[gi].first;
        const cv::Scalar col = safe[gi].second;
        for (size_t li = 0; li < lines.size(); ++li) {
            ty += lineH;
            // 每组的第一行（带编号那行）用档位色，其余用黑字
            const cv::Scalar tc = li == 0 ? col : cv::Scalar(30, 30, 30);
            cv::putText(img, lines[li], cv::Point(px + kPadX, ty - 4), cv::FONT_HERSHEY_SIMPLEX,
                        kScale, tc, kThick + (li == 0 ? 1 : 0), cv::LINE_AA);
        }
        if (gi + 1 < safe.size()) ty += kGap;
    }
}

}  // namespace markloc

