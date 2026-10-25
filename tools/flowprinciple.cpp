// 原理图生成器 —— 每步：做了什么 / 实现什么 / 算法 / 算法内容
// 用 QPainter 排版，中文正常显示

#include <QApplication>
#include <QFile>
#include <QFont>
#include <QFontMetrics>
#include <QImage>
#include <QPainter>
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
    if (m.empty()) return QImage();
    if (m.type() == CV_8UC1) {
        return QImage(m.data, m.cols, m.rows, (int)m.step, QImage::Format_Grayscale8).copy();
    }
    if (m.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(m, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888).copy();
    }
    return QImage();
}

cv::Mat gray3(const cv::Mat& m) {
    cv::Mat o;
    if (m.channels() == 1) cv::cvtColor(m, o, cv::COLOR_GRAY2BGR);
    else o = m.clone();
    return o;
}

struct Card {
    QString no;
    QString title;
    QString what;     // 做了什么
    QString effect;   // 实现了什么
    QString algo;     // 算法名
    QStringList detail;  // 算法内容
    cv::Mat before;
    cv::Mat after;
};

/// 列表式卡片：用于"优化方向与建议"这类没有图像、需要多行说明的内容
struct NoteCard {
    QString no;
    QString title;
    struct Item {
        QString name;      // 项目名
        QString problem;   // 现状/问题
        QString action;    // 建议动作
        QString gain;      // 预期收益
    };
    std::vector<Item> items;
};

}  // namespace

#include <QApplication>
#include <QFile>
#include <QFont>
#include <QFontMetrics>
#include <QImage>
#include <QPainter>
#include <QStringList>
#include <QTextOption>

#include <algorithm>
#include <cstdio>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


int main(int argc, char** argv) {
    markloc::enableUtf8Console();
    QApplication app(argc, argv);
    const QStringList args = QCoreApplication::arguments();
    if (args.size() < 4) {
        std::printf("usage: markloc_flowprinciple <model> <img> <out.png>\n");
        return 2;
    }

    chardet::Model cm;
    if (!chardet::loadModel(args[1].toUtf8().toStdString(), cm)) return 1;

    cv::Mat gray;
    {
        QFile f(args[2]);
        if (f.open(QIODevice::ReadOnly)) {
            const QByteArray b = f.readAll();
            const std::vector<uchar> v(b.begin(), b.end());
            gray = cv::imdecode(v, cv::IMREAD_GRAYSCALE);
        }
    }
    if (gray.empty()) return 1;
    const chardet::Config cfg = cm.config;

    std::vector<chardet::CharUnit> all;
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        auto u = chardet::toUnits(cm, (int)ri);
        for (auto& x : u) all.push_back(x);
    }
    const cv::Rect body(300, 225, std::min(1065, gray.cols - 300), std::min(915, gray.rows - 225));

    // ---------- 计算各步图像 ----------
    cv::Mat sig = chardet::strokeSignal(gray, cv::Rect(0, 0, gray.cols, gray.rows), cfg, false);
    cv::Mat sigVis;
    cv::normalize(sig, sigVis, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat bin;
    cv::compare(sig, cfg.strokeHi, bin, cv::CMP_GT);

    cv::Mat boxOnModel = gray3(cm.freq);
    for (size_t ri = 0; ri < cm.regions.size(); ++ri)
        cv::rectangle(boxOnModel, cm.regions[ri].rect, cv::Scalar(255, 120, 0), 2);
    for (const auto& u : all) cv::rectangle(boxOnModel, u.bbox, cv::Scalar(0, 220, 0), 1);

    cv::Mat sigReg = sig.clone();
    cv::Point2d sh(0, 0);
    cv::Rect charArea;
    if (!cm.ref.empty()) {
        cv::Mat r32, s32;
        cm.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        for (const auto& u : all) charArea = charArea.area() > 0 ? (charArea | u.bbox) : u.bbox;
        charArea.x = std::max(0, charArea.x - 20);
        charArea.y = std::max(0, charArea.y - 20);
        charArea.width = std::min(sig.cols - charArea.x, charArea.width + 40);
        charArea.height = std::min(sig.rows - charArea.y, charArea.height + 40);
        cv::Mat rC, sC;
        r32(charArea).copyTo(rC);
        s32(charArea).copyTo(sC);
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

    cv::Mat alignVis = gray3(gray);
    {
        cv::Mat pre = gray3(gray);
        for (const auto& u : all) cv::rectangle(pre, u.bbox, cv::Scalar(0, 200, 0), 1);
        alignVis = pre.clone();
    }
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
        cv::rectangle(alignVis, u.bbox, cv::Scalar(0, 0, 230), 1);
        cv::rectangle(alignVis, nb, cv::Scalar(0, 220, 0), 2);
        u.bbox = nb;
        ++moved;
    }

    cv::Mat det = gray3(gray);
    int nm = 0, ne = 0;
    for (const auto& u : all) {
        const auto nc = chardet::diffNcc(sigReg, cm.freq, u, cfg);
        if (!nc.valid) continue;
        if (nc.missScore > 0.0) {
            cv::rectangle(det, u.bbox, cv::Scalar(0, 0, 230), 2);
            ++nm;
        }
        if (nc.extraScore > 0.0) {
            cv::rectangle(det, u.bbox, cv::Scalar(230, 120, 0), 2);
            ++ne;
        }
    }
    const auto bd = chardet::diffBlocks(sigReg, cm.freq, all, cfg, 8);
    for (const auto& b : bd.extraBlocks)
        cv::rectangle(det, cv::Rect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0), cv::Scalar(0, 220, 220),
                      2);

    // ---------- 卡片 ----------
    std::vector<Card> cards;

    cards.push_back({QStringLiteral("01"), QStringLiteral("读入与裁剪"),
                     QStringLiteral("读入 tif，裁到塑封体范围"),
                     QStringLiteral("后续所有运算只在有效区域进行，减少 60% 的无关像素"),
                     QStringLiteral("内存解码 + ROI 裁剪"),
                     {QStringLiteral("QFile 读入字节流，cv::imdecode 解码"),
                      QStringLiteral("裁剪 Rect(300,225,1065,915) 塑封体"),
                      QStringLiteral("! 不用 cv::imread：中文路径会失败"),
                      QStringLiteral("依据 1688x1460 原图标定边界")},
                     gray, gray(body)});

    cards.push_back({QStringLiteral("02"), QStringLiteral("笔画信号"),
                     QStringLiteral("用高斯模糊估计背景，原图减背景"),
                     QStringLiteral("刻印笔画从背景中凸显，对比度提升约 4 倍"),
                     QStringLiteral("高斯背景建模 + 双分位数归一化"),
                     {QStringLiteral("GaussianBlur(sigma=15) 得背景 bg"),
                      QStringLiteral("S = max(f - bg, 0) 负值截断"),
                      QStringLiteral("S /= q99.5*0.55 + q90*0.45"),
                      QStringLiteral("! sigma 须远大于笔画宽(10px)"),
                      QStringLiteral("! 只用 q94 会饱和；用最大值被噪点带偏")},
                     gray(body), sigVis(body)});

    cards.push_back({QStringLiteral("03"), QStringLiteral("二值化"),
                     QStringLiteral("按固定阈值把信号转成墨迹掩码"),
                     QStringLiteral("得到「哪里有墨」的二值图，供所有判据使用"),
                     QStringLiteral("全局阈值二值化"),
                     {QStringLiteral("bin = (S > 0.35)"),
                      QStringLiteral("+ 0.35 由标定得到：此时待检墨量"),
                      QStringLiteral("+ = 模型墨量的 96%，量纲一致"),
                      QStringLiteral("! 背景纹理也会越线，"),
                      QStringLiteral("! 各判据后续用开运算净化")},
                     sigVis(body), bin(body)});

    cards.push_back({QStringLiteral("04"), QStringLiteral("模型频率图"),
                     QStringLiteral("84 张 OK 样本逐张二值累加后平均"),
                     QStringLiteral("得到每个像素「有墨的概率」，比任何单张图都干净"),
                     QStringLiteral("二值累加平均（频率图）"),
                     {QStringLiteral("freq = Σ bin_i / N * 255"),
                      QStringLiteral("整图粗对齐 + 逐字符精对齐后再累加"),
                      QStringLiteral("freq>0.50 => 必然有墨"),
                      QStringLiteral("freq<0.15 => 应为空白"),
                      QStringLiteral("! 用二值而非强度：强度累加使"),
                      QStringLiteral("! 边缘 3px 变 5px，梯度 2.06 降到 1.03")},
                     bin(body), cm.freq(body)});

    cards.push_back({QStringLiteral("05"), QStringLiteral("字符切分"),
                     QStringLiteral("对频率图做列投影，按低谷切成单字"),
                     QStringLiteral("把一行字自动切成 22 个独立字符框"),
                     QStringLiteral("垂直投影 + 间隙合并"),
                     {QStringLiteral("bin = freq > 0.25*255"),
                      QStringLiteral("projX = 每列墨量之和"),
                      QStringLiteral("阈值 = max(projX) * 0.03"),
                      QStringLiteral("间隙 < 中位字宽*0.10 则合并"),
                      QStringLiteral("! 合并间隙解决「i 的点」"),
                      QStringLiteral("! 与「i 的竖」被切成两段")},
                     cm.freq(body), boxOnModel(body)});

    cards.push_back({QStringLiteral("06"), QStringLiteral("整图配准"),
                     QStringLiteral("用相位相关求整体平移量（只用字符区）"),
                     QStringLiteral("消除载物台位置差异，84 张 OK 误报从 39.3% 降到 8.3%"),
                     QStringLiteral("相位相关 phaseCorrelate"),
                     {QStringLiteral("F1=FFT(A), F2=FFT(B)"),
                      QStringLiteral("R = F1·F2* / |F1·F2*|"),
                      QStringLiteral("r = IFFT(R)，峰值位置即平移量"),
                      QStringLiteral("+ 只看相位不看幅值，对亮度免疫"),
                      QStringLiteral("! 必须只用字符区：全图被引脚主导"),
                      QStringLiteral("! 实测全图误差 8~16px，峰值仅 0.25~0.37")},
                     sigVis(body), regVis(body)});

    cv::Mat boxedBody = gray3(gray);
    cv::rectangle(boxedBody, body, cv::Scalar(255,180,0), 3);
    boxedBody = boxedBody(body).clone();
    cards.push_back({QStringLiteral("07"), QStringLiteral("逐字符对齐"),
                     QStringLiteral("每个字单独做模板匹配，微调框位"),
                     QStringLiteral("消除每字 6~8px 的个体错位，墨量比从 21~54% 恢复到 96~101%"),
                     QStringLiteral("模板匹配 + 抛物线亚像素"),
                     {QStringLiteral("搜索区 = box 外扩 12px"),
                      QStringLiteral("matchTemplate(TM_CCOEFF_NORMED)"),
                      QStringLiteral("峰值 < 0.30 则跳过该字"),
                      QStringLiteral("亚像素 = 0.5(R-L)/(2C-L-R)"),
                      QStringLiteral("+ 实测对齐后平均残余 1.0px"),
                      QStringLiteral("! 不对齐时框内装的是半个字")},
                     boxedBody, alignVis(body)});

    cards.push_back({QStringLiteral("08"), QStringLiteral("检测判定"),
                     QStringLiteral("四类判据并行，任一命中即判 NG"),
                     QString(QStringLiteral("本次检出：缺墨 %1、多墨 %2、多出字符 %3")
                                 .arg(nm).arg(ne).arg(bd.extraCount)),
                     QStringLiteral("分组预期 + 整行统计 + Hough + 连通域"),
                     {QStringLiteral("① 字符级：freq>0.5 处待检无墨"),
                      QStringLiteral("② 整行：字符区间内墨量 < 10%"),
                      QStringLiteral("③ 划痕：Hough 直线 + 灰度分级"),
                      QStringLiteral("④ 污损：空白连通块，面积>120"),
                      QStringLiteral("! 均先做开运算去边缘抖动"),
                      QStringLiteral("! 划痕范围限字符框外扩 60px 且限塑封体内")},
                     alignVis(body), det(body)});


    // ---------- NG 实例 ----------
    //
    // 单独做一张卡片，用一张已知缺陷图演示"判定为 NG 时具体发生了什么"：
    // 左图是缺陷所在的字符原样，右图把"模型认为必然有墨、而待检确实没墨"
    // 的像素高亮出来，直观说明判据命中在哪里。
    {
        cv::Mat ngBefore = gray3(gray);
        cv::Mat ngAfter = gray3(gray);

        int ngHit = 0;
        cv::Rect ngBox;
        double ngRatio = 0.0;
        for (const auto& u : all) {
            const auto nc = chardet::diffNcc(sigReg, cm.freq, u, cfg);
            if (!nc.valid || nc.missScore <= 0.0) continue;
            ++ngHit;
            if (nc.missRatio > ngRatio) {
                ngRatio = nc.missRatio;
                ngBox = u.bbox;
            }
        }

        if (ngBox.area() > 0) {
            // 左：标出缺陷字符
            cv::rectangle(ngBefore, ngBox, cv::Scalar(0, 0, 230), 3);
            // 右：在该字符内高亮缺失像素
            cv::Mat mf, tBin, sureInk, missPx, missOpen;
            cm.freq(ngBox).convertTo(mf, CV_32F, 1.0 / 255.0);
            double mMax = 0;
            cv::minMaxLoc(mf, nullptr, &mMax);
            cv::compare(mf, std::max(0.50, mMax * 0.75), sureInk, cv::CMP_GT);
            cv::compare(sigReg(ngBox), cfg.strokeHi, tBin, cv::CMP_GT);
            cv::Mat notInk;
            cv::bitwise_not(tBin, notInk);
            cv::bitwise_and(sureInk, notInk, missPx);
            cv::morphologyEx(missPx, missOpen, cv::MORPH_OPEN,
                             cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

            cv::Mat patch = ngAfter(ngBox);
            patch.setTo(cv::Scalar(0, 0, 255), missOpen);   // 缺失像素标红
            cv::Mat big;
            cv::resize(ngAfter(ngBox), big, cv::Size(), 3, 3, cv::INTER_NEAREST);
            cv::Mat canvasPatch = ngAfter.clone();
            cv::Rect dst(ngBox.x - 20, ngBox.y - 20, big.cols, big.rows);
            dst &= cv::Rect(0, 0, canvasPatch.cols, canvasPatch.rows);
            if (dst.area() > 0 && dst.area() == ngBox.area() * 9) {
                big.copyTo(canvasPatch(dst));
            }
            ngAfter = canvasPatch;
        }

        char nb[160];
        std::snprintf(nb, sizeof(nb), "缺墨字符 %d 个，最严重缺墨率 %.0f%%", ngHit, ngRatio * 100);

        cards.push_back({QStringLiteral("NG"), QStringLiteral("NG 实例"),
                         QStringLiteral("同一套判据在一张缺陷图上的实际输出"),
                         QString::fromUtf8(nb),
                         QStringLiteral("① 字符级判据（分组预期）"),
                         {QStringLiteral("模型必然有墨区 freq>0.75x峰值"),
                          QStringLiteral("待检二值化 S>0.35"),
                          QStringLiteral("缺墨像素 = 必有墨 AND 待检无墨"),
                          QStringLiteral("开运算 5x5 去边缘 1~2px 抖动"),
                          QStringLiteral("! 缺墨率 > 15% 即判 NG"),
                          QStringLiteral("右侧放大图红色 = 被判为缺失的像素")},
                         ngBefore(body), ngAfter(body)});
    }


    // ---------- 优化方向与建议 ----------
    NoteCard notes;
    notes.no = QStringLiteral("建议");
    notes.title = QStringLiteral("优化方向与建议");
    notes.items = {
        {"划痕检出率",
         "真实 NG 的划痕多数检不出。划痕判据要求线段连续覆盖 25% 且能量 > 30 万，门槛偏高",
         "把划痕检测独立成分支：先在塑封体内做全局 Hough，再按灰度分级（<0.45 浅表放行）",
         "预期真实划痕 NG 检出从 1/8 提升，同时靠灰度分级控制 OK 误报"},
        {"大框稀释缺陷",
         "logo 区是 279x177 的大框，按整框比例算，一个 15x10 的缺口只占 1~2%，任何比例判据都失效",
         "把 logo 区切成独立的图形单元（椭圆标、S、C、Silan）再各自比对",
         "预期能检出 logo 断裂缺口类缺陷"},
        {"字符框抖动",
         "即使逐字符对齐，平均仍有 1.0px 残余；笔画宽 10px 时对应约 20% 的边缘像素错位",
         "改用相位相关做逐字符对齐（对亮度免疫），或用互信息配准",
         "预期残余降到 0.3px 以内，缺墨判据的噪声底下降"},
        {"模型表示",
         "当前存的是 84 张的平均频率图。OK 样本自带 ±6px 位移、±15% 粗细波动，抖动幅度大于缺陷幅度",
         "改为每字符保存多张真实模板（如 K 个聚类中心），检测时取最近模板比对",
         "预期能真正区分“正常但偏细”与“缺一笔”，这是当前 8.3% 误报的根因"},
        {"误报来源",
         "84 张 OK 中 7 张误报。它们在像素层面确有划痕或偏细，客户判定放行",
         "把误报样本作为负样本纳入训练，或在灵敏度里按批次标定阈值",
         "预期误报降到 3% 以内"},
        {"性能",
         "单张 450ms，其中高斯模糊（sigma=15，核 91x91）占 60%",
         "用积分图或 box filter 近似高斯；或先下采样再做背景估计",
         "预期单张降到 120ms 以内，吞吐提升约 4 倍"},
        {"划痕范围",
         "划痕范围受塑封体边界限制，边界值按 1688x1460 硬编码标定",
         "改为在线检测塑封体边界（找灰度骤变处），适应不同分辨率与不同封装",
         "预期换封装或换相机后无需改代码"},
        {"缺笔画/改笔画",
         "zero_to_Q、S_to_6 这类“字还是那个字、笔画变了”的缺陷，当前检不出",
         "骨架拓扑判据（端点数/分支数变化）可区分，但需配合多模板模型才有效",
         "预期补上这一类缺陷的检出能力"},
    };

    // ---------- 排版 ----------
    const int W = 1560;
    const int cardH = 268;
    const int gap = 10;
    const int noteRowH = 66;
    const int totalH = 70 + (cardH + gap) * (int)cards.size() + 20 +
                        (int)notes.items.size() * noteRowH + 90;

    QImage canvas(W, totalH, QImage::Format_RGB888);
    canvas.fill(QColor(22, 22, 26));

    QPainter p(&canvas);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::SmoothPixmapTransform, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);

    QFont titleFont(QStringLiteral("Microsoft YaHei"), 14, QFont::Bold);
    QFont noFont(QStringLiteral("Consolas"), 17, QFont::Bold);
    QFont headFont(QStringLiteral("Microsoft YaHei"), 9, QFont::Bold);
    QFont bodyFont(QStringLiteral("Microsoft YaHei"), 8);
    QFont smallFont(QStringLiteral("Microsoft YaHei"), 7);

    p.setFont(titleFont);
    p.setPen(QColor(240, 240, 245));
    p.drawText(QRect(20, 12, W - 40, 40), Qt::AlignLeft | Qt::AlignVCenter,
               QStringLiteral("芯片刻印检测 —— 每一步的处理、目标、算法与算法内容"));

    int y = 62;
    for (const Card& c : cards) {
        const QRect outer(14, y, W - 28, cardH);
        p.setPen(QPen(QColor(64, 64, 74), 1));
        p.setBrush(QColor(30, 30, 36));
        p.drawRoundedRect(outer, 6, 6);

        // 编号 + 标题
        p.setFont(noFont);
        p.setPen(QColor(0, 200, 255));
        p.drawText(QRect(26, y + 8, 46, 30), Qt::AlignLeft | Qt::AlignVCenter, c.no);

        p.setFont(titleFont);
        p.setPen(QColor(245, 245, 250));
        p.drawText(QRect(74, y + 10, 400, 26), Qt::AlignLeft | Qt::AlignVCenter, c.title);

        // 两张图
        const int iw = 190, ih = 196;
        int ix = 26;
        for (int k = 0; k < 2; ++k) {
            const cv::Mat& m = (k == 0) ? c.before : c.after;
            if (m.empty()) continue;
            const QImage qi = matToQImage(m);
            const QRect b(ix, y + 46, iw, ih);
            p.setPen(QPen(QColor(58, 58, 66), 1));
            p.setBrush(QColor(14, 14, 18));
            p.drawRect(b);
            const QImage sc = qi.scaled(iw - 4, ih - 4, Qt::KeepAspectRatio,
                                        Qt::SmoothTransformation);
            p.drawImage(b.x() + (iw - sc.width()) / 2, b.y() + (ih - sc.height()) / 2, sc);
            p.setFont(smallFont);
            p.setPen(QColor(0, 200, 255));
            p.drawText(QRect(b.x() + 4, b.y() + 2, iw - 8, 14), Qt::AlignLeft,
                       k == 0 ? QStringLiteral("处理前") : QStringLiteral("处理后"));
            ix += iw + 8;
        }

        // 三栏说明
        const int tx = ix + 10;
        const int colW = (W - 28 - tx - 10) / 3;
        const struct {
            QString head;
            QColor color;
            QString main;
            QStringList lines;
        } cols[3] = {
            {QStringLiteral("做了什么"), QColor(120, 220, 120), c.what, {}},
            {QStringLiteral("实现了什么"), QColor(120, 200, 255), c.effect, {}},
            {QStringLiteral("用的算法"), QColor(255, 210, 120), c.algo, c.detail},
        };

        for (int k = 0; k < 3; ++k) {
            const int cx = tx + k * colW;
            p.setFont(headFont);
            p.setPen(cols[k].color);
            p.drawText(QRect(cx, y + 44, colW - 8, 18), Qt::AlignLeft | Qt::AlignVCenter,
                       cols[k].head);
            p.setPen(QPen(QColor(60, 60, 70), 1));
            p.drawLine(cx, y + 64, cx + colW - 10, y + 64);

            p.setFont(bodyFont);
            int ly = y + 70;
            QTextOption opt;
            opt.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);

            if (!cols[k].main.isEmpty()) {
                const QRect r(cx, ly, colW - 10, 300);
                const QFontMetrics fm(bodyFont);
                const QRect need = fm.boundingRect(r, Qt::TextWordWrap, cols[k].main);
                p.setPen(QColor(225, 225, 232));
                p.drawText(r, Qt::TextWordWrap, cols[k].main);
                ly += need.height() + 6;
            }

            p.setFont(smallFont);
            const QFontMetrics fm2(smallFont);
            for (const QString& raw : cols[k].lines) {
                QColor col(180, 180, 190);
                QString t = raw;
                if (t.startsWith(QLatin1Char('+'))) {
                    col = QColor(130, 220, 130);
                    t = t.mid(1).trimmed();
                } else if (t.startsWith(QLatin1Char('!'))) {
                    col = QColor(240, 130, 130);
                    t = t.mid(1).trimmed();
                }
                const QRect r(cx, ly, colW - 10, 300);
                const QRect need = fm2.boundingRect(r, Qt::TextWordWrap, t);
                p.setPen(col);
                p.drawText(r, Qt::TextWordWrap, t);
                ly += need.height() + 3;
                if (ly > y + cardH - 8) break;
            }
        }

        y += cardH + gap;
    }
    // ---- 优化方向与建议 ----
    {
        const int boxTop = y;
        const int boxH = 60 + (int)notes.items.size() * noteRowH;
        const QRect outer(14, boxTop, W - 28, boxH);
        p.setPen(QPen(QColor(200, 150, 60), 2));
        p.setBrush(QColor(34, 32, 26));
        p.drawRoundedRect(outer, 6, 6);

        p.setFont(noFont);
        p.setPen(QColor(255, 190, 80));
        p.drawText(QRect(26, boxTop + 8, 90, 30), Qt::AlignLeft | Qt::AlignVCenter,
                   notes.no);

        p.setFont(titleFont);
        p.setPen(QColor(255, 225, 160));
        p.drawText(QRect(100, boxTop + 10, 600, 26), Qt::AlignLeft | Qt::AlignVCenter,
                   notes.title);

        // 表头
        const int c0 = 26;
        const int c1 = 300;   // 项目
        const int c2 = 640;   // 现状
        const int c3 = 1050;  // 建议
        const int c4 = 1340;  // 收益
        p.setFont(headFont);
        p.setPen(QColor(255, 210, 150));
        p.drawText(QRect(c0, boxTop + 44, c1 - c0, 18), Qt::AlignLeft, QStringLiteral("项目"));
        p.drawText(QRect(c1, boxTop + 44, c2 - c1, 18), Qt::AlignLeft,
                   QStringLiteral("现状 / 问题"));
        p.drawText(QRect(c2, boxTop + 44, c3 - c2, 18), Qt::AlignLeft,
                   QStringLiteral("建议动作"));
        p.drawText(QRect(c3, boxTop + 44, c4 - c3, 18), Qt::AlignLeft,
                   QStringLiteral("预期收益"));

        int ry = boxTop + 66;
        for (size_t i = 0; i < notes.items.size(); ++i) {
            const auto& it = notes.items[i];
            if (i % 2 == 0) {
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(42, 40, 32));
                p.drawRect(QRect(c0 - 6, ry - 4, W - 28 - 2 * (c0 - 14) + 4, noteRowH - 4));
            }

            QTextOption opt;
            opt.setWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);

            p.setFont(headFont);
            p.setPen(QColor(255, 220, 140));
            p.drawText(QRect(c0, ry, c1 - c0 - 8, noteRowH), Qt::TextWordWrap, it.name);

            p.setFont(smallFont);
            p.setPen(QColor(215, 160, 160));
            p.drawText(QRect(c1, ry, c2 - c1 - 10, noteRowH), Qt::TextWordWrap, it.problem);

            p.setPen(QColor(160, 220, 160));
            p.drawText(QRect(c2, ry, c3 - c2 - 10, noteRowH), Qt::TextWordWrap, it.action);

            p.setPen(QColor(160, 200, 255));
            p.drawText(QRect(c3, ry, c4 - c3, noteRowH), Qt::TextWordWrap, it.gain);

            ry += noteRowH;
        }
    }

    p.end();

    if (!canvas.save(args[3])) return 1;
    std::printf("saved %s (%dx%d)\n", args[3].toUtf8().constData(), canvas.width(),
                canvas.height());
    return 0;
}
