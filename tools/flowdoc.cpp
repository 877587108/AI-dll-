// 流程效果图生成器 —— 用 Qt 渲染中文标注
//
// 分工：
//   图像处理部分复用 chardet（与 GUI 同一套算法），把每一步的结果存成 PNG；
//   排版与文字标注交给 QPainter（OpenCV 的 putText 不支持中文）。
//
// 用法: markloc_flowdoc <模型目录> <产品图> <输出png>

#include <QApplication>
#include <QFont>
#include <QFontMetrics>
#include <QImage>
#include <QPainter>
#include <QFile>
#include <QPainterPath>
#include <QStringList>
#include <QTextOption>

#include <algorithm>
#include <cstdio>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "common/console_utf8.hpp"

namespace {

QImage matToQImage(const cv::Mat& m) {
    if (m.empty()) return {};
    if (m.type() == CV_8UC1) {
        return QImage(m.data, m.cols, m.rows, static_cast<int>(m.step), QImage::Format_Grayscale8)
            .copy();
    }
    if (m.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(m, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                      QImage::Format_RGB888)
            .copy();
    }
    return {};
}

cv::Mat gray3(const cv::Mat& m) {
    cv::Mat o;
    if (m.channels() == 1) cv::cvtColor(m, o, cv::COLOR_GRAY2BGR);
    else o = m.clone();
    return o;
}

/// 一个步骤的描述
struct Step {
    QString no;
    QString title;
    QStringList lines;   // 前置标记：+ 用绿色、! 用红色、其余灰色
    cv::Mat before;
    cv::Mat after;
    QString beforeTag = QStringLiteral("处理前");
    QString afterTag = QStringLiteral("处理后");
};

}  // namespace

int main(int argc, char** argv) {
    markloc::enableUtf8Console();
    QApplication app(argc, argv);

    const QStringList args = QCoreApplication::arguments();
    if (args.size() < 4) {
        std::printf("usage: markloc_flowdoc <model> <img> <out.png>\n");
        return 2;
    }

    chardet::Model cm;
    if (!chardet::loadModel(args[1].toUtf8().toStdString(), cm)) {
        std::printf("load model failed\n");
        return 1;
    }
    // 用 QFile 读入再解码：cv::imread 对含中文或特殊字符的路径会失败
    cv::Mat gray;
    {
        QFile f(args[2]);
        if (f.open(QIODevice::ReadOnly)) {
            const QByteArray bytes = f.readAll();
            const std::vector<uchar> buf(bytes.begin(), bytes.end());
            gray = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
        }
    }
    if (gray.empty()) {
        std::printf("load image failed\n");
        return 1;
    }
    const chardet::Config cfg = cm.config;

    std::vector<chardet::CharUnit> all;
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        auto u = chardet::toUnits(cm, static_cast<int>(ri));
        for (auto& x : u) all.push_back(x);
    }

    const cv::Rect body(300, 225, std::min(1065, gray.cols - 300), std::min(915, gray.rows - 225));

    std::vector<Step> steps;

    // ---- 01 读入
    {
        Step s;
        s.no = QStringLiteral("01");
        s.title = QStringLiteral("读入与裁剪");
        s.lines = {QStringLiteral("+ imdecode 读入，绕开中文路径"),
                   QStringLiteral("+ 裁剪至塑封体 x300 y225 1065x915"),
                   QStringLiteral("目标：只保留与刻印相关的区域"),
                   QStringLiteral("! 引脚、基板亮银色的结构与字符检测无关，"),
                   QStringLiteral("! 全图处理会让它们主导后续配准")};
        s.before = gray;
        s.after = gray(body);
        s.beforeTag = QStringLiteral("原图 1688x1460");
        s.afterTag = QStringLiteral("裁剪后 1065x915");
        steps.push_back(s);
    }

    // ---- 02 笔画信号
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat sigVis;
    cv::normalize(sig, sigVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    {
        Step s;
        s.no = QStringLiteral("02");
        s.title = QStringLiteral("笔画信号（核心预处理）");
        s.lines = {QStringLiteral("+ GaussianBlur sigma=15 估计背景"),
                   QStringLiteral("+ 原图减背景，负值截断"),
                   QStringLiteral("+ 归一化 q99.5x0.55 + q90x0.45"),
                   QStringLiteral("目标：让刻印笔画从背景中凸显"),
                   QStringLiteral("! 原图笔画与背景仅差约 20 灰度"),
                   QStringLiteral("! sigma 必须远大于笔画宽度（约 10px）")};
        s.before = gray(body);
        s.after = sigVis(body);
        s.beforeTag = QStringLiteral("原图裁剪");
        s.afterTag = QStringLiteral("笔画信号 S");
        steps.push_back(s);
    }

    // ---- 03 二值化
    cv::Mat bin;
    cv::compare(sig, cfg.strokeHi, bin, cv::CMP_GT);
    {
        Step s;
        s.no = QStringLiteral("03");
        s.title = QStringLiteral("二值化得到墨迹掩码");
        s.lines = {QStringLiteral("+ compare(S > 0.35) 取墨迹"),
                   QStringLiteral("目标：得到“哪里有墨”的二值掩码"),
                   QStringLiteral("+ 0.35 由标定得来：该阈值下待检墨量"),
                   QStringLiteral("+ 为模型墨量的 96%，两者量纲基本一致"),
                   QStringLiteral("! 背景纹理也会越线，各判据后续做开运算净化")};
        s.before = sigVis(body);
        s.after = bin(body);
        s.beforeTag = QStringLiteral("笔画信号");
        s.afterTag = QStringLiteral("二值墨迹 S>0.35");
        steps.push_back(s);
    }

    // ---- 04 模型
    {
        Step s;
        s.no = QStringLiteral("04");
        s.title = QStringLiteral("模型频率图");
        s.lines = {QStringLiteral("+ 84 张 OK 样本逐张算信号并二值累加"),
                   QStringLiteral("+ 整图粗对齐 + 逐字符精对齐后平均"),
                   QStringLiteral("freq>0.50 视为“必然有墨”"),
                   QStringLiteral("freq<0.15 视为“应为空白”"),
                   QStringLiteral("目标：得到每个像素有墨的概率图"),
                   QStringLiteral("! 用二值累加而非强度累加，否则边缘变糊")};
        s.before = bin(body);
        s.after = cm.freq(body);
        s.beforeTag = QStringLiteral("单张二值墨迹");
        s.afterTag = QStringLiteral("模型频率图");
        steps.push_back(s);
    }

    // ---- 05 字符框
    cv::Mat boxOnModel = gray3(cm.freq);
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        cv::rectangle(boxOnModel, cm.regions[ri].rect, cv::Scalar(255, 120, 0), 2);
    }
    for (const auto& u : all) cv::rectangle(boxOnModel, u.bbox, cv::Scalar(0, 220, 0), 1);
    {
        Step s;
        s.no = QStringLiteral("05");
        s.title = QStringLiteral("字符框切分");
        s.lines = {QStringLiteral("+ 频率图二值化 (binFreq=0.25)"),
                   QStringLiteral("+ 列投影，阈值 = 峰值 x 0.03"),
                   QStringLiteral("+ 间隙 < 中位字宽 x 0.10 则合并"),
                   QStringLiteral("目标：把一行字切成单个字符框"),
                   QStringLiteral("! 合并间隙解决“i 的点”与“i 的竖”被切开"),
                   QStringLiteral("共 4 个区域、22 个字符框")};
        s.before = cm.freq(body);
        s.after = boxOnModel(body);
        s.beforeTag = QStringLiteral("模型频率图");
        s.afterTag = QStringLiteral("字符框（绿）区域框（蓝）");
        steps.push_back(s);
    }

    // ---- 06 整图配准
    cv::Mat sigReg = sig.clone();
    cv::Point2d sh(0, 0);
    if (!cm.ref.empty()) {
        cv::Mat r32, s32;
        cm.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        cv::Rect ia;
        for (const auto& u : all) ia = ia.area() > 0 ? (ia | u.bbox) : u.bbox;
        ia.x = std::max(0, ia.x - 20);
        ia.y = std::max(0, ia.y - 20);
        ia.width = std::min(sig.cols - ia.x, ia.width + 40);
        ia.height = std::min(sig.rows - ia.y, ia.height + 40);
        cv::Mat rC, sC;
        r32(ia).copyTo(rC);
        s32(ia).copyTo(sC);
        sh = cv::phaseCorrelate(sC, rC, cv::noArray());
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::warpAffine(sig, sigReg, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
        }
    }
    cv::Mat regVis;
    cv::normalize(sigReg, regVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    {
        Step s;
        s.no = QStringLiteral("06");
        s.title = QStringLiteral("整图配准");
        s.lines = {QStringLiteral("+ 相位相关 phaseCorrelate"),
                   QStringLiteral("+ 只对“字符区域”做，不用全图"),
                   QStringLiteral("+ 抛物线插值到亚像素"),
                   QStringLiteral("目标：消除整体的平移偏差"),
                   QStringLiteral("! 全图配准会被引脚高亮主导，"),
                   QStringLiteral("! 实测误差可达 8~16px，导致所有框错位")};
        s.before = sigVis(body);
        s.after = regVis(body);
        s.beforeTag = QStringLiteral("配准前");
        s.afterTag = QStringLiteral("配准后");
        steps.push_back(s);
    }

    // ---- 07 逐字符对齐
    cv::Mat aligned = gray3(gray);
    int moved = 0;
    for (auto& u : all) {
        cv::Rect box = u.bbox & cv::Rect(0, 0, sigReg.cols, sigReg.rows);
        if (box.width <= 4 || box.height <= 4) continue;
        constexpr int kS = 12;
        cv::Rect sr(box.x - kS, box.y - kS, box.width + 2 * kS, box.height + 2 * kS);
        sr &= cv::Rect(0, 0, sigReg.cols, sigReg.rows);
        if (sr.width < box.width || sr.height < box.height) continue;
        cv::Mat mS, tS;
        cm.freq(sr).convertTo(mS, CV_32F, 1.0 / 255.0);
        sigReg(sr).convertTo(tS, CV_32F);
        cv::Mat res;
        cv::matchTemplate(tS, mS(cv::Rect(box.x - sr.x, box.y - sr.y, box.width, box.height)), res,
                          cv::TM_CCOEFF_NORMED);
        cv::Point mL;
        double mv = 0;
        cv::minMaxLoc(res, nullptr, &mv, nullptr, &mL);
        if (mv < 0.30) continue;
        cv::Rect nb(box.x + mL.x - kS, box.y + mL.y - kS, box.width, box.height);
        if ((nb & cv::Rect(0, 0, sigReg.cols, sigReg.rows)) != nb) continue;
        cv::rectangle(aligned, u.bbox, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(aligned, nb, cv::Scalar(0, 220, 0), 2);
        u.bbox = nb;
        ++moved;
    }
    cv::Mat boxed = gray3(gray);
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        (void)ri;
    }
    {
        cv::Mat tmp = gray3(gray);
        for (const auto& u : cm.regions.empty() ? all : all) {
            cv::rectangle(tmp, u.bbox, cv::Scalar(0, 220, 0), 1);
        }
        Step s;
        s.no = QStringLiteral("07");
        s.title = QStringLiteral("逐字符精对齐");
        s.lines = {QStringLiteral("+ 每个字单独做模板匹配，搜索 ±12px"),
                   QStringLiteral("+ 抛物线插值到亚像素"),
                   QStringLiteral("+ 红框 = 原框，绿框 = 对齐后"),
                   QStringLiteral("目标：消除每个字 6~8px 的个体错位"),
                   QStringLiteral("! 不对齐时框内装的是“半个字 + 空白”，"),
                   QStringLiteral("! 墨量比只有 21~54%，必然误报交")};
        s.before = tmp(body);
        s.after = aligned(body);
        s.beforeTag = QStringLiteral("对齐前");
        s.afterTag = QStringLiteral("对齐后");
        steps.push_back(s);
    }

    // ---- 08 检测
    cv::Mat det = gray3(gray);
    int nm = 0, ne = 0;
    for (const auto& u : all) {
        const auto nc = chardet::diffNcc(sigReg, cm.freq, u, cfg);
        if (!nc.valid) continue;
        if (nc.missScore > 0.0) {
            cv::rectangle(det, u.bbox, cv::Scalar(0, 0, 255), 2);
            ++nm;
        }
        if (nc.extraScore > 0.0) {
            cv::rectangle(det, u.bbox, cv::Scalar(255, 120, 0), 2);
            ++ne;
        }
    }
    const auto bd = chardet::diffBlocks(sigReg, cm.freq, all, cfg, 8);
    for (const auto& b : bd.extraBlocks) {
        cv::rectangle(det, cv::Rect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0), cv::Scalar(0, 220, 220),
                      2);
    }
    {
        Step s;
        s.no = QStringLiteral("08");
        s.title = QStringLiteral("检测判定");
        s.lines = {QStringLiteral("+ 判据1 字符级：必然有墨处是否缺墨"),
                   QStringLiteral("+ 判据2 整行：字符区间是否缺字"),
                   QStringLiteral("+ 判据3 划痕：Hough 直线 + 灰度分级"),
                   QStringLiteral("+ 判据4 污损/溢料：仅限字符框内"),
                   QStringLiteral("! 任一命中即判 NG"),
                   QString(QStringLiteral("本次结果：缺墨 %1、多墨 %2、多出字符 %3")
                               .arg(nm).arg(ne).arg(bd.extraCount))};
        s.before = aligned(body);
        s.after = det(body);
        s.beforeTag = QStringLiteral("对齐后");
        s.afterTag = QStringLiteral("检测结果");
        steps.push_back(s);
    }

    // ---- 排版
    const int imgW = 420;
    const int imgH = 300;
    const int panelW = 470;
    const int rowH = imgH + 26;
    const int marginX = 16;

    QFont titleFont(QStringLiteral("Microsoft YaHei"), 11, QFont::Bold);
    QFont noFont(QStringLiteral("Consolas"), 20, QFont::Bold);
    QFont bodyFont(QStringLiteral("Microsoft YaHei"), 8);

    const int totalW = marginX * 2 + imgW * 2 + 12 + panelW;
    const int totalH = rowH * static_cast<int>(steps.size()) + 60 + marginX;

    QImage canvas(totalW, totalH, QImage::Format_RGB888);
    canvas.fill(QColor(24, 24, 28));

    QPainter p(&canvas);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::SmoothPixmapTransform, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);

    // 顶部标题
    p.setFont(QFont(QStringLiteral("Microsoft YaHei"), 13, QFont::Bold));
    p.setPen(QColor(235, 235, 240));
    p.drawText(QRect(marginX, 8, totalW - marginX * 2, 40), Qt::AlignLeft | Qt::AlignVCenter,
               QStringLiteral("芯片刻印检测流程 —— 每一步的处理方法与目标"));

    int y = 56;
    for (const Step& s : steps) {
        // 两张图
        int x = marginX;
        for (int k = 0; k < 2; ++k) {
            const cv::Mat& m = k == 0 ? s.before : s.after;
            const QString tag = k == 0 ? s.beforeTag : s.afterTag;
            if (m.empty()) continue;

            const QImage qi = matToQImage(m);
            const QRect box(x, y, imgW, imgH);
            p.setPen(QPen(QColor(70, 70, 78), 1));
            p.setBrush(QColor(14, 14, 18));
            p.drawRect(box);

            // 等比放入
            const QImage scaled = qi.scaled(imgW - 6, imgH - 6, Qt::KeepAspectRatio,
                                            Qt::SmoothTransformation);
            const int px = box.x() + (imgW - scaled.width()) / 2;
            const int py = box.y() + (imgH - scaled.height()) / 2;
            p.drawImage(px, py, scaled);

            // 小标签
            QFont tagFont(QStringLiteral("Microsoft YaHei"), 7);
            p.setFont(tagFont);
            const QFontMetrics fm(tagFont);
            const int tw = fm.horizontalAdvance(tag) + 10;
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(0, 0, 0, 170));
            p.drawRect(QRect(box.x() + 4, box.y() + 4, tw, 18));
            p.setPen(QColor(0, 220, 255));
            p.drawText(QRect(box.x() + 6, box.y() + 4, tw, 18), Qt::AlignVCenter, tag);

            x += imgW + 12;
        }

        // 右侧面板
        const QRect panel(x, y, panelW, imgH);
        p.setPen(QPen(QColor(70, 70, 78), 1));
        p.setBrush(QColor(18, 18, 22));
        p.drawRect(panel);

        p.setFont(noFont);
        p.setPen(QColor(0, 200, 255));
        p.drawText(QRect(x + 12, y + 8, 60, 30), Qt::AlignLeft | Qt::AlignVCenter, s.no);

        p.setFont(titleFont);
        p.setPen(QColor(240, 240, 245));
        p.drawText(QRect(x + 62, y + 10, panelW - 74, 26), Qt::AlignLeft | Qt::AlignVCenter,
                   s.title);

        p.setPen(QPen(QColor(60, 60, 70), 1));
        p.drawLine(x + 12, y + 42, x + panelW - 12, y + 42);

        p.setFont(bodyFont);
        int ly = y + 48;
        const QFontMetrics fm(bodyFont);
        for (const QString& raw : s.lines) {
            if (raw.isEmpty()) {
                ly += 8;
                continue;
            }
            QColor c(190, 190, 198);
            QString text = raw;
            if (text.startsWith(QLatin1Char('+'))) {
                c = QColor(130, 220, 130);
                text = text.mid(1).trimmed();
            } else if (text.startsWith(QLatin1Char('!'))) {
                c = QColor(240, 130, 130);
                text = text.mid(1).trimmed();
            }
            // 自动换行
            QTextOption opt;
            opt.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
            QRect br(x + 12, ly, panelW - 24, 200);
            QRect need = fm.boundingRect(br, Qt::TextWordWrap, text);
            p.setPen(c);
            p.drawText(br, Qt::TextWordWrap, text);
            ly += need.height() + 4;
            if (ly > y + imgH - 10) break;
        }

        // 行分隔
        p.setPen(QPen(QColor(48, 48, 56), 1));
        p.drawLine(marginX, y + imgH + 12, totalW - marginX, y + imgH + 12);

        y += rowH;
    }
    p.end();

    if (!canvas.save(args[3])) {
        std::printf("save failed\n");
        return 1;
    }
    std::printf("saved %s (%dx%d)\n", args[3].toUtf8().constData(), canvas.width(),
                canvas.height());
    return 0;
}
