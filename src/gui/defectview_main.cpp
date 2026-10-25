// 缺陷标注检查器 —— 程序入口
//
// 独立于建模工具：只读模型，不改动任何配置。
// 打开即可载入 model/vNN 与产品图，运行检测并查看标注。
//
// 另外提供无界面批量模式，用的是界面批量检测的同一份实现
// （gui/batchcore.hpp），结果写 CSV。用途是拿带真值的目录跑回归：
//
//   markloc_defectview.exe --batch <模型目录> <图片目录> [--truth ok|ng] [--out x.csv]
//
// --truth 不是检测参数，只是用来在汇总里报"误报/漏检"多少张。
// 目录真值（ok/ 与 ng/）是本项目唯一可靠的标签来源，文件名里的
// _FT_ / _RT_ 是第一次/第二次拍摄，不是 NG/OK。

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QPushButton>
#include <QStringList>
#include <QTableWidget>
#include <QTextStream>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "chardet.hpp"
#include "common/console_utf8.hpp"
#include "gui/DefectViewWindow.hpp"
#include "gui/batchcore.hpp"

namespace {

bool readGrayU8(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray bytes = f.readAll();
    f.close();
    if (bytes.isEmpty()) return false;
    const std::vector<uchar> buf(bytes.begin(), bytes.end());
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

std::string u8s(const QString& s) {
    const QByteArray b = s.toUtf8();
    return std::string(b.constData(), static_cast<size_t>(b.size()));
}

int runBatchCli(const QString& modelDir, const QString& imgDir, const QString& truth,
                const QString& outCsv, const QStringList& asserts) {
    chardet::Model model;
    const QString vdir = QDir(modelDir).absoluteFilePath(QStringLiteral("v00"));
    if (!chardet::loadModel(u8s(vdir), model) || !model.valid()) {
        std::printf("model load failed: %s\n", u8s(vdir).c_str());
        return 1;
    }
    std::vector<chardet::CharUnit> base;
    for (size_t ri = 0; ri < model.regions.size(); ++ri) {
        auto u = chardet::toUnits(model, static_cast<int>(ri));
        for (auto& x : u) base.push_back(std::move(x));
    }
    if (base.empty()) {
        std::printf("model has no characters\n");
        return 1;
    }

    // 只处理 v00 视角：其余视角是引脚与封装侧面，与刻印字符无关
    QStringList files;
    for (const QFileInfo& fi : QDir(imgDir).entryInfoList(
             {QStringLiteral("*.tif"), QStringLiteral("*.tiff"), QStringLiteral("*.png"),
              QStringLiteral("*.bmp")},
             QDir::Files, QDir::Name)) {
        if (!fi.completeBaseName().endsWith(QStringLiteral("_00"))) continue;
        files << fi.absoluteFilePath();
    }
    if (files.isEmpty()) {
        std::printf("no v00 images in %s\n", u8s(imgDir).c_str());
        return 1;
    }

    markloc::DetectSettings det;
    det.cfg = model.config;
    // 对照开关：用来把"对齐"这个因素单独关掉做实验。
    // 0101 就是靠它查出来的 —— 不开的时候 S 上到 1347，
    // 开了之后掉到 883，说明对齐把真划痕的证据削掉了。
    if (qEnvironmentVariableIsSet("MARKLOC_NO_PERCHARWARP")) det.perCharWarp = false;
    if (qEnvironmentVariableIsSet("MARKLOC_NO_ALIGN")) {
        det.perCharAlign = false;
        det.perCharWarp = false;
    }

    std::vector<markloc::BatchRow> rows;
    double totalMs = 0.0;
    for (const QString& path : files) {
        QElapsedTimer t;
        t.start();
        cv::Mat g;
        if (!readGrayU8(path, g)) continue;
        bool valid = false;
        auto row = markloc::detectOne(g, model, base, det, &valid);
        if (!valid) continue;
        row.name = u8s(QFileInfo(path).fileName());
        row.path = u8s(QFileInfo(path).absoluteFilePath());
        row.elapsedMs = static_cast<double>(t.elapsed());
        totalMs += row.elapsedMs;
        rows.push_back(row);
    }

    // ---- 汇总
    const bool hasTruth = (truth == QStringLiteral("ok") || truth == QStringLiteral("ng"));
    const bool truthNg = (truth == QStringLiteral("ng"));
    int ng = 0, fp = 0, fn = 0;
    for (const auto& r : rows) {
        if (r.ng) ++ng;
        if (!hasTruth) continue;
        if (r.ng && !truthNg) ++fp;
        if (!r.ng && truthNg) ++fn;
    }

    std::printf("%s  n=%d  NG=%d  avg=%.1fms\n", u8s(QFileInfo(imgDir).fileName()).c_str(),
                static_cast<int>(rows.size()), ng,
                rows.empty() ? 0.0 : totalMs / static_cast<double>(rows.size()));
    {
        // 分类计数：固定顺序，跑两遍好对
        static const char* kOrder[] = {"确定", "明显", "可疑", "对齐异常", "旁证", "OK"};
        std::printf("  分类:");
        for (const char* c : kOrder) {
            int cnt = 0;
            std::string cur = c;
            for (const auto& r : rows) if (r.cls == cur) ++cnt;
            std::printf(" %s=%d", c, cnt);
        }
        std::printf("\n");
    }
    if (hasTruth) {
        std::printf("  truth=%s  误报=%d  漏检=%d\n", u8s(truth).c_str(), fp, fn);
    }
    for (const auto& r : rows) {
        std::string detail;
        if (!r.rejectReason.empty()) {
            detail = "未报：" + r.rejectReason;
        } else {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "S=%d 跨%d 深%d", static_cast<int>(r.scratchS),
                          r.scratchChars, static_cast<int>(r.scratchDepth));
            detail = buf;
        }
        std::printf("  %-34s %-9s %s\n", r.name.c_str(), r.cls.c_str(), detail.c_str());
        // 图案区单独一行。
        //
        // 【为什么必须打这一行】
        // 图案区检测如果压根没跑（区域没标成椭圆、或者被跳过），
        // 输出和"跑了但没发现断口"一模一样 —— 又是一个恒定指标。
        // 所以这里连"扫了几个区域"一起打出来：区域数=0 就是没跑，不是没问题。
        if (r.patternRegions > 0 || r.patternBreaks > 0) {
            // 图案区的分数单独一行：它和划痕分不是一回事，
            // 混在同一个"严重度"里会让人误以为可以互相比较。
            std::printf("      图案区 区域%d 断口%d 最长缺口%.2f° 断点分%.0f 半径起伏%.0fpx%s\n",
                        r.patternRegions, r.patternBreaks, r.patternMaxGap, r.patternScore,
                        r.patternRadiusSwing,
                        r.patternMissing > 0 ? "（有整圈无墨）" : "");
        }
    }

    if (!outCsv.isEmpty()) {
        QFile f(outCsv);
        if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream ts(&f);
            ts.setEncoding(QStringConverter::Utf8);
            ts << "\xEF\xBB\xBF";
            ts << "file,result,class,scratch_S,scratch_band,scratch_chars,scratch_depth,"
                  "scratch_count,align_shift,missing_char,missing_ink,extra_ink,elapsed_ms,note\n";
            for (const auto& r : rows) {
                ts << QString::fromUtf8(r.name.c_str()) << ',' << (r.ng ? "NG" : "OK") << ','
                   << QString::fromUtf8(r.cls.c_str()) << ',' << static_cast<int>(r.scratchS) << ','
                   << r.scratchBand << ',' << r.scratchChars << ','
                   << static_cast<int>(r.scratchDepth) << ',' << r.scratchCount << ','
                   << r.alignMaxShift << ',' << r.rowMissing << ',' << r.miss << ',' << r.extra
                   << ',' << static_cast<int>(r.elapsedMs) << ','
                   << QString::fromUtf8(r.rejectReason.c_str()) << '\n';
            }
            std::printf("  CSV -> %s\n", u8s(outCsv).c_str());
        }
    }

    // ---- 断言模式：把"判对了没有"变成机器可判定的问题
    //
    // 为什么需要这个：
    //   没有断言时，"改完更好了"只能靠人读两行分类数字，
    //   而今晚已经反复出现"工具输出了数字、读的人读错了"（p50 当成 1.0、
    //   文件名解析错、504 张混入统计）。断言把这些错锁在工具里。
    //
    // 用法：--assert <文件子串>=OK|NG，可重复。子串按包含匹配（文件名带时间戳）。
    //   例：--assert 0109_RT=OK --assert 2224=NG
    //   任一不满足即打印 FAIL 并返回 1。
    int assertFail = 0;
    if (!asserts.isEmpty()) {
        std::printf("\n=== 断言 ===\n");
        for (const auto& a : asserts) {
            const int eq = a.indexOf(QLatin1Char('='));
            if (eq <= 0) {
                std::printf("  [FAIL] 断言格式错误: %s\n", u8s(a).c_str());
                ++assertFail;
                continue;
            }
            const QString needle = a.left(eq);
            const QString want = a.mid(eq + 1).trimmed().toUpper();
            const bool wantNg = (want == QStringLiteral("NG"));

            const markloc::BatchRow* found = nullptr;
            for (const auto& r : rows) {
                if (QString::fromUtf8(r.name.c_str()).contains(needle)) {
                    found = &r;
                    break;
                }
            }
            if (!found) {
                std::printf("  [FAIL] %-28s 找不到这个文件\n", u8s(needle).c_str());
                ++assertFail;
                continue;
            }
            const bool gotNg = found->ng;
            const bool ok = (gotNg == wantNg);
            std::printf("  [%s] %-28s 期望 %-2s 实际 %-2s  %s\n", ok ? "PASS" : "FAIL",
                        u8s(needle).c_str(), u8s(want).c_str(), gotNg ? "NG" : "OK",
                        u8s(QString::fromUtf8(found->cls.c_str())).c_str());
            if (!ok) ++assertFail;
        }
        std::printf("=== %d 条断言，%d 条不满足 ===\n", static_cast<int>(asserts.size()),
                    assertFail);
    }
    return assertFail > 0 ? 1 : 0;
}

}  // namespace

// 界面自检：把"能不能渲染"变成可以被机器判定的问题
//
//   markloc_defectview.exe --selftest-ui <图片> [--out 目录]
//
// 干的事：离屏建窗口 -> 载入图片 -> 跑一次检测 -> 量三件事：
//   1. 图像区 ImageView 里到底有没有像素（不是"控件在不在"）
//   2. 对比图 QImage 是不是 3 联、非空、非纯色
//   3. 顶层布局有没有把图像区/右栏真正纳入布局（孤儿布局检测）
//
// 这三条正是"完全空白"这个故障的判据。以前只能靠人看，现在能自动跑。
// 退出码 0 = 三条全过。
int runUiSelfTest(const QString& imgPath, const QString& outDir) {
    int fail = 0;
    auto check = [&fail](bool ok, const char* what, const std::string& detail) {
        std::printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", what, detail.c_str());
        if (!ok) ++fail;
    };

    // ---- 先验标签清洗：这是"标注框显示问号"的直接根因，不依赖窗口就能测
    //
    // cv::putText 的 Hershey 字体没有中文字形，任何非 ASCII 字节进去
    // 都会画成问号。所以只要清洗后仍含非 ASCII，就一定会出问号。
    {
        const std::vector<std::string> samples = {
            "\u7f3a\u58a8 12%",        // 缺墨 12%
            "\u591a\u58a8 34%",        // 多墨 34%
            "\u6ea2\u6599",            // 溢料
            "\u6c61\u635f",            // 污损
            "\u523b\u9519?",           // 刻错?
            "\u53ef\u7591",            // 可疑
            "\u5212\u75d5\u00b7\u786e\u5b9a",  // 划痕·确定
            "S=2224 CERTAIN",          // 纯 ASCII 应原样保留
        };
        int leaked = 0;
        std::string dump;
        for (const auto& s : samples) {
            const std::string a = markloc::asciiLabel(s);
            dump += "[" + a + "] ";
            for (unsigned char c : a) {
                if (c >= 0x80) {
                    ++leaked;
                    break;
                }
            }
        }
        char buf[512];
        std::snprintf(buf, sizeof(buf), "leaked=%d %s", leaked, dump.c_str());
        check(leaked == 0, "标注文字清洗后无中文残留", buf);
    }

    DefectViewWindow w;
    // 用接近真实用法的尺寸：太小会让分割条把下方面板压得只剩一条缝，
    // 测出来的"渲染了"却不代表实际能看清。
    w.resize(1800, 1150);
    w.show();
    // 连续 processEvents：第一次只处理 show，布局还没走完，
    // 滚动区视口此时拿不到真实尺寸。多转几圈让布局收敛，
    // 否则测出来的"适配"是假的。
    for (int i = 0; i < 5; ++i) QApplication::processEvents();

    check(w.isVisible(), "主窗口可见", w.isVisible() ? "yes" : "no");

    // (3) 孤儿布局检测：图像区必须能在窗口的 widget 树里找到，
    //     且它的几何区域非零。布局漏 addLayout 时控件虽然 new 出来了，
    //     几何是 0×0 —— 这正是上次"完全空白"的形状。
    const QPoint imgPos = w.findImageViewGlobalPos();
    const QSize imgSize = w.findImageViewSize();
    {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%dx%d", imgSize.width(), imgSize.height());
        check(imgSize.width() > 100 && imgSize.height() > 100,
              "图像区已纳入布局且有实际尺寸", buf);
    }

    // (4) 按钮文案不能被截断。
    //
    // 上一版这里没有任何校验，而截图里「灵敏度…」「批量检测…」都被压成了
    // "灵敏度..."，客户根本分不清哪个是哪个。QPushButton 被压到小于
    // sizeHint 时会自己加省略号，所以判据就是"实际宽度 >= 文本需要的宽度"。
    // 不依赖截图取色：圆角、边框、DPI 缩放都会让取色失真，直接问 Qt 最准。
    {
        int truncated = 0;
        QString worst;
        int worstSlack = 1 << 30;
        for (QPushButton* b : w.findChildren<QPushButton*>()) {
            if (!b->isVisible()) continue;
            // sizeHint 含文本与内边距，就是"不截断所需的最小宽度"
            const int need = b->sizeHint().width();
            const int have = b->width();
            const int slack = have - need;
            if (slack < 0) {
                ++truncated;
                if (slack < worstSlack) {
                    worstSlack = slack;
                    worst = QStringLiteral("%1(have=%2 need=%3)")
                                .arg(b->text())
                                .arg(have)
                                .arg(need);
                }
            }
        }
        char buf[512];
        std::snprintf(buf, sizeof(buf), "truncated=%d worst=%s", truncated,
                      worst.isEmpty() ? "none" : worst.toUtf8().constData());
        check(truncated == 0, "按钮文案均未被截断", buf);
    }

    // (1) 载入模型 + 图片 + 检测
    //
    // 顺序必须和真实使用一致：没模型时 runDetect() 会在第一个判断处弹窗返回，
    // 对比图根本不会生成——那会被误读成"渲染坏了"。
    const QString modelDir = QDir(QCoreApplication::applicationDirPath())
                                 .absoluteFilePath(QStringLiteral("model"));
    const bool modelOk = w.loadModelForTest(modelDir);
    check(modelOk, "模型载入", u8s(modelDir) + (modelOk ? "" : " (模型不在 exe 旁边的 model/)"));
    if (!modelOk) {
        std::printf("%s  失败项=%d\n", "UI SELFTEST FAIL", fail);
        return 1;
    }

    bool loaded = w.loadImageForTest(imgPath);
    QApplication::processEvents();
    check(loaded, "图片载入", u8s(QFileInfo(imgPath).fileName()));
    if (loaded) {
        w.runDetectForTest();
        QApplication::processEvents();
    }

    // 同一张图连跑两次，结论必须一致。
    //
    // 逐字符对齐会原地改写 units_ 里的字符框，第二次检测在已挪过的框上
    // 再对齐一次，偏移累加——实测第一轮 2 条划痕 S=2224、第二轮 1 条
    // S=2174。同一张图同样的参数给出不同结论，比误判更严重。
    if (loaded) {
        const QString first = w.statusTextForTest();
        w.runDetectForTest();
        QApplication::processEvents();
        const QString second = w.statusTextForTest();
        char rbuf[256];
        std::snprintf(rbuf, sizeof(rbuf), "两次结果%s", first == second ? "一致" : "不一致");
        check(first == second, "同一张图重复检测结果一致", rbuf);
    }

    // 载入图片后再量一次：图片显示出来时，它的宽度应当接近视口宽度
    // （适配 = 整张图刚好放得下），不能是 1:1 的原始尺寸 5064 宽，
    // 也不能是几个像素。这是"图片到底适配没适配"的硬判据。
    if (loaded) {
        const QSize after = w.findImageViewSize();
        const QSize vp = w.viewportSizeForTest();
        char buf[128];
        std::snprintf(buf, sizeof(buf), "image=%dx%d viewport=%dx%d", after.width(),
                      after.height(), vp.width(), vp.height());
        const bool fitsW = after.width() <= vp.width() + 4;
        const bool fitsH = after.height() <= vp.height() + 4;
        const bool usable = after.width() > 200 && after.height() > 200;
        check(fitsW && fitsH && usable, "图片按视口适配（不出界、不留白）", buf);
    }
    const QImage shotPixels = w.grabImageView();
    check(!shotPixels.isNull() && shotPixels.width() > 100,
          "图像区可抓取且尺寸合理",
          shotPixels.isNull() ? "null"
                              : (std::to_string(shotPixels.width()) + "x" +
                                 std::to_string(shotPixels.height())));

    // 状态栏必须说清 NG 的依据是划痕。
    // 以前这里显示"缺墨 0 · 多墨 0 · 多出字符 8 → NG"，
    // 把划痕算进了"多出字符"，看着像误报。
    {
        const QString st = w.statusTextForTest();
        const bool mentionsScratch = st.contains(QStringLiteral("划痕"));
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s", st.toUtf8().constData());
        check(mentionsScratch, "状态栏说明了划痕判定依据", buf);
    }

    // ---- 下方分数面板：单张检测后必须有明确呈现
    //
    // 注意这里分两种情况，都要检查，不能只查列数。
    // 以前只校验 columnCount>=8，而 0 行的空表也满足，
    // 于是"面板一片空白"这个问题被自检判成了 PASS。
    //   情况 A：报了划痕 -> 表格可见，且至少 1 行
    //   情况 B：没报划痕 -> 表格收起，空态文字可见且非空
    {
        const bool onScorePage = w.bottomPageForTest() == 0;
        const QTableWidget* t = w.scoreTableForTest();
        const QString title = w.scoreTitleForTest();
        const int rows = t ? t->rowCount() : -1;
        const bool tableShown = t && !t->isHidden();
        const QString emptyText = w.scoreEmptyTextForTest();
        const bool emptyShown = w.scoreEmptyVisibleForTest();

        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "page=%d rows=%d cols=%d tableShown=%d emptyShown=%d title=%s",
                      w.bottomPageForTest(), rows, t ? t->columnCount() : -1,
                      tableShown ? 1 : 0, emptyShown ? 1 : 0, title.toUtf8().constData());
        // 至少要有 8 列（档位/严重度/深度/跨字/位置/长度/升档/降档）
        const bool colsOk = t && t->columnCount() >= 8;
        // 两种状态必须恰好成立一种，且成立的这种要有实际内容
        const bool reportedOk = tableShown && rows >= 1;
        const bool cleanOk = !tableShown && emptyShown && !emptyText.trimmed().isEmpty();
        const bool ok = onScorePage && colsOk && !title.isEmpty() && (reportedOk || cleanOk);
        check(ok, "分数面板有明确呈现（报出则填表，未报出则给结论）", buf);
        if (!outDir.isEmpty()) {
            QDir().mkpath(outDir);
            w.grab().save(QDir(outDir).absoluteFilePath(QStringLiteral("selftest_score.png")));
        }
    }

    // ---- 批量滑动列表：灌入几条假结果，验证列表、幻灯片、切页
    //
    // 不去真跑批量（慢），只验证渲染链路：列表项数、当前下标、
    // 幻灯片图是否真的画出来了、下方容器是否切到了滑动页。
    {
        std::vector<markloc::BatchRow> fake;
        for (int i = 0; i < 3; ++i) {
            markloc::BatchRow r;
            r.name = "slide_test_" + std::to_string(i) + "_00.tif";
            r.path = u8s(QFileInfo(imgPath).absoluteFilePath());
            r.cls = "\u786e\u5b9a";
            r.ng = true;
            r.scratchCount = 1;
            r.scratchBand = 3;
            r.scratchS = 2000.0 + i * 100;
            r.scratchChars = 3;
            r.scratchDepth = 700;
            r.scratchLen = 180;
            fake.push_back(r);
        }
        w.injectBatchRowsForTest(fake);
        for (int i = 0; i < 8; ++i) QApplication::processEvents();

        const int cnt = w.batchListCountForTest();
        const int idx = w.slideIndexForTest();
        const QSize slide = w.slideViewSizeForTest();
        char buf[256];
        std::snprintf(buf, sizeof(buf), "list=%d slideIndex=%d slidePixmap=%dx%d page=%d", cnt,
                      idx, slide.width(), slide.height(), w.bottomPageForTest());
        const bool ok = cnt == 3 && idx == 0 && w.bottomPageForTest() == 1 && slide.width() > 50 &&
                        slide.height() > 20;
        check(ok, "批量滑动列表可逐项查看", buf);

        // NG/OK 必须在行首。文件名是一长串几乎相同的编号，结论摆在名字后面
        // 等于没摆——这是用户报的问题，必须由断言守住，不能靠目测。
        {
            bool allFront = true;
            QString first;
            for (int i = 0; i < cnt; ++i) {
                const QString t = w.batchListTextForTest(i);
                if (i == 0) first = t;
                if (!t.startsWith(QStringLiteral("NG")) && !t.startsWith(QStringLiteral("OK"))) {
                    allFront = false;
                }
            }
            char nbuf[512];
            std::snprintf(nbuf, sizeof(nbuf), "items=%d 首项=[%s] %s", cnt,
                          first.toUtf8().constData(), allFront ? "全部以 NG/OK 开头" : "有项不是");
            check(allFront && cnt > 0, "列表项 NG/OK 标签在最前", nbuf);
        }

        // 「打开大图对比」后列表必须还在。
        //
        // 原来 openBatchRow 走 runDetect()，runDetect 结尾会把下方切到分数页，
        // 于是从列表点进去看大图，列表就被顶掉了，没法按「下一项」继续翻。
        // 用户报的「打开大图对比后列表不见了」就是这个。
        {
            w.openRowForTest(0);
            for (int i = 0; i < 6; ++i) QApplication::processEvents();
            const int page = w.bottomPageForTest();
            const int li = w.slideIndexForTest();
            char obuf[160];
            std::snprintf(obuf, sizeof(obuf), "打开后 page=%d（1=滑动列表） slideIndex=%d", page,
                          li);
            check(page == 1, "打开大图对比后列表仍在", obuf);
        }

        // 缩放 + 拖动：这三个动作以前在滑动列表里根本不存在（只有一个静态 QLabel），
        // 所以"看不大、看不了别处"。这里把放大、拖动、复位三步都量一遍。
        {
            char zbuf[256];
            const QSize before = w.canvasSizeForTest();
            const bool dialOk = w.zoomPanForTest();
            // zoomPanForTest 内部会复位，所以这里量到的 after 是"复位后"的尺寸。
            // 判据要用 dialOk 本身（它已经在内部逐步量过放大/拖动/复位），
            // 不能拿复位后的尺寸去和放大前比——那样永远是"没变大"。
            const QSize after = w.canvasSizeForTest();
            std::snprintf(zbuf, sizeof(zbuf), "适配=%dx%d 复位后=%dx%d -> %s", before.width(),
                          before.height(), after.width(), after.height(),
                          dialOk ? "放大/拖动/复位三步均生效" : "链路不通");
            check(dialOk, "滑动列表对比图可缩放拖动", zbuf);
        }

        // 幻灯片里的档位标签必须画成英文。中文标签在 putText 下会变成问号，
        // 而问号的形状是"小圆点 + 下方小钩"，在缩略图的标签区内是可辨认的。
        // 这里不去做字形识别（不可靠），改用一个确定的判据：
        // 画出来的标签像素必须集中在图像左上角那块标题区，且该区是暗底亮字。
        {
            const QImage sp = w.grabSlidePixmapForTest();
            if (!sp.isNull() && !outDir.isEmpty()) {
                QDir().mkpath(outDir);
                sp.save(QDir(outDir).absoluteFilePath(QStringLiteral("selftest_slide_raw.png")));
            }
            // 判据：标签区里必须存在"近黑色"的描边像素。
            // 标签是用黑底+白描边写的，芯片本体不会出现这种纯黑细笔画，
            // 所以只要数到一定数量的近黑像素，就说明文字真的落上去了。
            int nearBlack = 0;
            if (!sp.isNull()) {
                const int w0 = std::min(sp.width() / 2, 260);
                const int h0 = std::min(sp.height(), 60);
                for (int y = 0; y < h0; ++y) {
                    for (int x = 0; x < w0; ++x) {
                        if (qGray(sp.pixel(x, y)) < 40) ++nearBlack;
                    }
                }
            }
            char lbuf[128];
            std::snprintf(lbuf, sizeof(lbuf), "near-black px in label area=%d", nearBlack);
            check(nearBlack > 50, "幻灯片档位标签已画出（非空白）", lbuf);
        }

        if (!outDir.isEmpty()) {
            QDir().mkpath(outDir);
            w.grab().save(QDir(outDir).absoluteFilePath(QStringLiteral("selftest_slide.png")));
        }
    }

    // (2) 对比图本身
    const QImage cmp = w.resultImageForTest();
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%dx%d", cmp.width(), cmp.height());
        check(!cmp.isNull(), "对比图非空", buf);
    }
    if (!cmp.isNull()) {
        // 3 联：宽约等于 3 倍单图、高约等于单图 + 标题条
        const double aspect = static_cast<double>(cmp.width()) / std::max(1, cmp.height());
        char buf[128];
        std::snprintf(buf, sizeof(buf), "aspect=%.2f (%dx%d)", aspect, cmp.width(), cmp.height());
        check(aspect > 2.0 && aspect < 4.5, "对比图是三联横排", buf);

        // 纯色检测：抽样统计颜色种类，纯色说明画布是空的
        int uniq = 0;
        std::vector<QRgb> seen;
        seen.reserve(4096);
        for (int y = 0; y < cmp.height(); y += 17) {
            for (int x = 0; x < cmp.width(); x += 17) {
                const QRgb v = cmp.pixel(x, y);
                bool hit = false;
                for (QRgb s : seen)
                    if (s == v) { hit = true; break; }
                if (!hit && seen.size() < 4096) {
                    seen.push_back(v);
                    ++uniq;
                }
            }
        }
        char cbuf[64];
        std::snprintf(cbuf, sizeof(cbuf), "unique colors=%d", uniq);
        check(uniq > 20, "对比图不是纯色（真的画了内容）", cbuf);
    }

    // 落盘，留证
    if (!outDir.isEmpty()) {
        QDir().mkpath(outDir);
        if (!shotPixels.isNull())
            shotPixels.save(QDir(outDir).absoluteFilePath(QStringLiteral("selftest_imageview.png")));
        if (!cmp.isNull())
            cmp.save(QDir(outDir).absoluteFilePath(QStringLiteral("selftest_compare.png")));
        // 整个窗口也抓一张：布局错位在整窗截图上一眼就看出来
        w.grab().save(QDir(outDir).absoluteFilePath(QStringLiteral("selftest_window.png")));

        // 只导出"标注结果"这一联的原尺寸图：问号问题要凑近才能看清，
        // 整窗截图缩放后细节全糊了。
        if (!cmp.isNull()) {
            // 三联图里第三联是标注结果，按宽度切开取右边 1/3
            const int third = cmp.width() / 3;
            const QImage annotated = cmp.copy(cmp.width() - third, 0, third, cmp.height());
            annotated.save(QDir(outDir).absoluteFilePath(QStringLiteral("selftest_annotated.png")));
        }
        std::printf("  -> 截图已存到 %s\n", u8s(outDir).c_str());
    }

    std::printf("%s  失败项=%d\n", fail == 0 ? "UI SELFTEST PASS" : "UI SELFTEST FAIL", fail);
    return fail == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    markloc::enableUtf8Console();

    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("缺陷标注检查器"));

    const QStringList args = QApplication::arguments();

    // 界面自检用离屏平台：不需要真实桌面，也就能在无头环境里跑
    if (args.size() >= 3 && args[1] == QStringLiteral("--selftest-ui")) {
        QApplication::setAttribute(Qt::AA_Use96Dpi);
        QString out;
        for (int i = 3; i + 1 < args.size(); i += 2)
            if (args[i] == QStringLiteral("--out")) out = args[i + 1];
        return runUiSelfTest(args[2], out);
    }

    if (args.size() >= 4 && args[1] == QStringLiteral("--batch")) {
        QString truth, out;
        QStringList asserts;
        for (int i = 4; i < args.size(); ++i) {
            if (args[i] == QStringLiteral("--truth") && i + 1 < args.size()) truth = args[++i];
            else if (args[i] == QStringLiteral("--out") && i + 1 < args.size()) out = args[++i];
            else if (args[i] == QStringLiteral("--assert") && i + 1 < args.size())
                asserts << args[++i];
        }
        return runBatchCli(args[2], args[3], truth, out, asserts);
    }

    DefectViewWindow w;
    w.show();
    return app.exec();
}
