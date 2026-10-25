#include "gui/TrainWindow.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTableWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "core/naming.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace markloc::charinsp;

namespace {

QString viewTag(int v) { return QStringLiteral("%1").arg(v, 2, 10, QLatin1Char('0')); }

bool readGray(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray bytes = f.readAll();
    f.close();
    if (bytes.isEmpty()) return false;
    const std::vector<uchar> buf(bytes.begin(), bytes.end());
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

/// 真值：父目录 ok/ng，或文件名含 _Pass/_Failed
bool inferTruth(const QString& path, bool& truthNg) {
    const QFileInfo fi(path);
    QDir up = fi.dir();
    for (int i = 0; i < 4; ++i) {
        const QString leaf = up.dirName().toLower();
        if (leaf == "ng") { truthNg = true; return true; }
        if (leaf == "ok") { truthNg = false; return true; }
        if (!up.cdUp()) break;
    }
    const QString name = fi.completeBaseName().toUpper();
    if (name.contains("_FAILED") || name.contains("_NG")) { truthNg = true; return true; }
    if (name.contains("_PASS") || name.contains("_OK")) { truthNg = false; return true; }
    return false;
}

}  // namespace

TrainWindow::TrainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("markloc — 评估（识别 + 性能）"));
    resize(1240, 820);
    buildUi();
    imageDirEdit_->setText(QDir::current().absoluteFilePath("image"));
    modelDirEdit_->setText(QDir::current().absoluteFilePath("models/char"));
    log(QStringLiteral("就绪。选好目录后点「开始评估」。训练/选区请用 markloc_train。"));
}

void TrainWindow::buildUi() {
    auto* central = new QWidget;
    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(8);

    auto* box = new QGroupBox(QStringLiteral("评估"));
    auto* form = new QFormLayout(box);
    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        imageDirEdit_ = new QLineEdit;
        imageDirEdit_->setPlaceholderText(QStringLiteral("含 ok/ 与 ng/ 的图片目录"));
        auto* btn = new QPushButton(QStringLiteral("浏览…"));
        connect(btn, &QPushButton::clicked, this, &TrainWindow::chooseImageDir);
        h->addWidget(imageDirEdit_, 1);
        h->addWidget(btn);
        form->addRow(QStringLiteral("图片目录"), row);
    }
    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        modelDirEdit_ = new QLineEdit;
        modelDirEdit_->setPlaceholderText(QStringLiteral("模型目录（含 v00..v05）"));
        auto* btn = new QPushButton(QStringLiteral("浏览…"));
        connect(btn, &QPushButton::clicked, this, &TrainWindow::chooseModelDir);
        h->addWidget(modelDirEdit_, 1);
        h->addWidget(btn);
        form->addRow(QStringLiteral("模型目录"), row);
    }
    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        recursiveCheck_ = new QCheckBox(QStringLiteral("含子文件夹"));
        recursiveCheck_->setChecked(true);
        limitSpin_ = new QSpinBox;
        limitSpin_->setRange(0, 100000);
        limitSpin_->setSpecialValueText(QStringLiteral("不限"));
        charCheck_ = new QCheckBox(QStringLiteral("启用字符级检测"));
        charCheck_->setToolTip(
            QStringLiteral("勾选后临时打开 checkChars（不改模型文件）；\n"
                           "需要模型里已有 chars_manual.txt 或 chars_N.txt"));
        h->addWidget(recursiveCheck_);
        h->addSpacing(10);
        h->addWidget(new QLabel(QStringLiteral("上限")));
        h->addWidget(limitSpin_);
        h->addSpacing(10);
        h->addWidget(charCheck_);
        h->addStretch(1);
        form->addRow(QStringLiteral("选项"), row);
    }
    root->addWidget(box);

    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        evalBtn_ = new QPushButton(QStringLiteral("开始评估"));
        evalBtn_->setMinimumHeight(42);
        QFont f = evalBtn_->font();
        f.setPointSize(f.pointSize() + 2);
        f.setBold(true);
        evalBtn_->setFont(f);
        connect(evalBtn_, &QPushButton::clicked, this, &TrainWindow::startEvaluate);
        h->addWidget(evalBtn_, 1);
        root->addWidget(row);
    }

    progress_ = new QProgressBar;
    progress_->setRange(0, 100);
    root->addWidget(progress_);

    summaryLabel_ = new QLabel(QStringLiteral("—"));
    QFont sf = summaryLabel_->font();
    sf.setPointSize(sf.pointSize() + 3);
    sf.setBold(true);
    summaryLabel_->setFont(sf);
    root->addWidget(summaryLabel_);

    auto* splitter = new QSplitter(Qt::Vertical);
    resultTable_ = new QTableWidget(0, 4);
    resultTable_->setHorizontalHeaderLabels(
        {QStringLiteral("零件"), QStringLiteral("真值"), QStringLiteral("判定"), QStringLiteral("最高分")});
    resultTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    resultTable_->verticalHeader()->setVisible(false);
    resultTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    splitter->addWidget(resultTable_);

    benchTable_ = new QTableWidget(0, 3);
    benchTable_->setHorizontalHeaderLabels({QStringLiteral("指标"), QStringLiteral("数值"),
                                            QStringLiteral("单位")});
    benchTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    benchTable_->verticalHeader()->setVisible(false);
    benchTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    splitter->addWidget(benchTable_);

    logView_ = new QPlainTextEdit;
    logView_->setReadOnly(true);
    logView_->setFont(QFont(QStringLiteral("Consolas")));
    splitter->addWidget(logView_);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);
    splitter->setStretchFactor(2, 3);
    root->addWidget(splitter, 1);

    setCentralWidget(central);
    statusLabel_ = new QLabel(QStringLiteral("就绪"));
    statusBar()->addWidget(statusLabel_, 1);
}

void TrainWindow::log(const QString& s) {
    logView_->appendPlainText(s);
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void TrainWindow::setBusy(bool busy) {
    busy_ = busy;
    evalBtn_->setEnabled(!busy);
    if (busy) QApplication::setOverrideCursor(Qt::WaitCursor);
    else QApplication::restoreOverrideCursor();
}

void TrainWindow::clearResults() {
    resultTable_->setRowCount(0);
    benchTable_->setRowCount(0);
    summaryLabel_->setText(QStringLiteral("—"));
}

void TrainWindow::setDirs(const QString& imageDir, const QString& modelDir) {
    if (!imageDir.isEmpty()) imageDirEdit_->setText(QDir::cleanPath(imageDir));
    if (!modelDir.isEmpty()) modelDirEdit_->setText(QDir::cleanPath(modelDir));
}

void TrainWindow::chooseImageDir() {
    const QString d = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择图片目录"), imageDirEdit_->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!d.isEmpty()) imageDirEdit_->setText(QDir::cleanPath(d));
}

void TrainWindow::chooseModelDir() {
    const QString d = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择模型目录"), modelDirEdit_->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!d.isEmpty()) modelDirEdit_->setText(QDir::cleanPath(d));
}

std::array<int, kViewCount> TrainWindow::collectByView(
    const QString& root, std::array<std::vector<QString>, kViewCount>& out) {
    std::array<int, kViewCount> counts{};
    for (auto& v : out) v.clear();
    const QDir dir(root);
    if (!dir.exists()) return counts;
    QStringList filters;
    filters << QStringLiteral("*.tif") << QStringLiteral("*.tiff") << QStringLiteral("*.TIF")
            << QStringLiteral("*.TIFF");
    const QDirIterator::IteratorFlags flags =
        recursiveCheck_->isChecked() ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags;
    QDirIterator it(root, filters, QDir::Files, flags);
    while (it.hasNext()) {
        const QString path = it.next();
        const int v = naming::viewOf(path.toStdString());
        if (v < 0 || v >= kViewCount) continue;
        const int limit = limitSpin_->value();
        if (limit > 0 && static_cast<int>(out[static_cast<size_t>(v)].size()) >= limit) continue;
        out[static_cast<size_t>(v)].push_back(path);
        ++counts[static_cast<size_t>(v)];
    }
    return counts;
}

void TrainWindow::startEvaluate() {
    if (busy_) return;
    const QString imageDir = imageDirEdit_->text().trimmed();
    const QString modelDir = modelDirEdit_->text().trimmed();
    if (imageDir.isEmpty() || !QDir(imageDir).exists()) {
        log(QStringLiteral("❌ 图片目录不存在"));
        return;
    }
    clearResults();
    logView_->clear();
    setBusy(true);
    evaluateAndShow(imageDir, modelDir);
    setBusy(false);
    statusLabel_->setText(QStringLiteral("评估完成"));
}

int TrainWindow::runEvaluateOnly(const QString& imageDir, const QString& modelDir) {
    clearResults();
    logView_->clear();
    setDirs(imageDir, modelDir);
    evaluateAndShow(imageDir, modelDir);
    const QString text = logView_->toPlainText();
    std::fwrite(text.toUtf8().constData(), 1, static_cast<size_t>(text.toUtf8().size()), stdout);
    std::fflush(stdout);
    return 0;
}

void TrainWindow::evaluateAndShow(const QString& imageDir, const QString& modelDir) {
    Model model;
    std::string err;
    if (!model.loadFrom(modelDir.toStdString(), &err)) {
        log(QStringLiteral("❌ 模型加载失败：%1").arg(QString::fromStdString(err)));
        return;
    }

    // 界面上临时开/关字符检测（不写回模型文件）
    const bool charOn = charCheck_ && charCheck_->isChecked();
    for (auto& p : model.params) p.checkChars = charOn;

    std::array<std::vector<QString>, kViewCount> byView;
    collectByView(imageDir, byView);

    struct ProdAgg {
        bool truthNg = false;
        bool known = false;
        bool predNg = false;
        double maxScore = 0.0;
    };
    std::map<QString, ProdAgg> agg;
    int imgTp = 0, imgFp = 0, imgTn = 0, imgFn = 0;
    int processed = 0, skipped = 0;

    BenchResult bench;
    std::vector<double> perImageMs;
    double sumDecode = 0.0, sumAnalyze = 0.0;
    int timed = 0;
    auto memMB = [](double b) { return b / (1024.0 * 1024.0); };
    auto sampleMem = [&]() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            const double w = memMB(static_cast<double>(pmc.WorkingSetSize));
            const double p = memMB(static_cast<double>(pmc.PagefileUsage));
            bench.memPeakMB = std::max(bench.memPeakMB, w);
            bench.privPeakMB = std::max(bench.privPeakMB, p);
            bench.memNowMB = w;
            bench.privNowMB = p;
        }
#endif
    };

    int total = 0;
    for (int v = 0; v < kViewCount; ++v) total += static_cast<int>(byView[static_cast<size_t>(v)].size());
    progress_->setRange(0, std::max(1, total));
    int done = 0;
    QElapsedTimer wall;
    wall.start();

    for (int v = 0; v < kViewCount; ++v) {
        if (!model.valid(v)) {
            skipped += static_cast<int>(byView[static_cast<size_t>(v)].size());
            done += static_cast<int>(byView[static_cast<size_t>(v)].size());
            progress_->setValue(done);
            continue;
        }
        for (const QString& p : byView[static_cast<size_t>(v)]) {
            cv::Mat gray;
            QElapsedTimer t;
            t.start();
            const bool decoded = readGray(p, gray);
            const double decMs = static_cast<double>(t.nsecsElapsed()) / 1e6;
            if (!decoded) {
                ++skipped;
                progress_->setValue(++done);
                continue;
            }
            t.restart();
            const Result r = analyze(gray, model.views[static_cast<size_t>(v)],
                                     model.params[static_cast<size_t>(v)], false);
            const double anaMs = static_cast<double>(t.nsecsElapsed()) / 1e6;
            sumDecode += decMs;
            sumAnalyze += anaMs;
            perImageMs.push_back(decMs + anaMs);
            ++timed;
            sampleMem();
            ++processed;

            bool truthNg = false;
            const bool hasTruth = inferTruth(p, truthNg);
            if (hasTruth) {
                if (truthNg) { if (r.ng) ++imgTp; else ++imgFn; }
                else { if (r.ng) ++imgFp; else ++imgTn; }
            }
            const QString pid = QString::fromStdString(naming::productOf(p.toStdString()));
            auto& a = agg[pid];
            if (hasTruth) { a.truthNg = truthNg; a.known = true; }
            if (r.ng) a.predNg = true;
            a.maxScore = std::max(a.maxScore, r.score);

            progress_->setValue(++done);
            if ((done % 25) == 0) QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
    }

    int pTp = 0, pFp = 0, pTn = 0, pFn = 0;
    for (const auto& [pid, a] : agg) {
        if (!a.known) continue;
        if (a.truthNg) { if (a.predNg) ++pTp; else ++pFn; }
        else { if (a.predNg) ++pFp; else ++pTn; }
    }

    log(QStringLiteral("===== 评估结果 ====="));
    log(QStringLiteral("参与判定 %1 张（跳过 %2）").arg(processed).arg(skipped));
    log(QStringLiteral("字符级检测：%1").arg(charOn ? QStringLiteral("开") : QStringLiteral("关")));
    const int imgLabeled = imgTp + imgFp + imgFn + imgTn;
    if (imgLabeled > 0) {
        log(QStringLiteral("[按图] TP=%1 FP=%2 FN=%3 TN=%4").arg(imgTp).arg(imgFp).arg(imgFn).arg(imgTn));
    } else {
        log(QStringLiteral("[按图] 无标签样本"));
    }
    const int pLabeled = pTp + pFp + pFn + pTn;
    if (pLabeled > 0) {
        log(QStringLiteral("[按零件] TP=%1 FP=%2 FN=%3 TN=%4").arg(pTp).arg(pFp).arg(pFn).arg(pTn));
    }

    // 汇总行
    if (pLabeled > 0) {
        const double acc = 100.0 * (pTp + pTn) / pLabeled;
        summaryLabel_->setText(QStringLiteral("按零件：%1 件  正确率 %2%  TP=%3 FP=%4 FN=%5 TN=%6")
                                   .arg(pLabeled)
                                   .arg(acc, 0, 'f', 2)
                                   .arg(pTp).arg(pFp).arg(pFn).arg(pTn));
        summaryLabel_->setStyleSheet(pFp == 0 && pFn == 0 ? QStringLiteral("color:#2e7d32;")
                                                         : QStringLiteral("color:#c62828;"));
    } else {
        summaryLabel_->setText(QStringLiteral("无标签样本，无法计算准确率"));
        summaryLabel_->setStyleSheet(QString());
    }

    // 逐件表
    std::vector<std::pair<QString, ProdAgg>> rows(agg.begin(), agg.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second.maxScore > b.second.maxScore; });
    resultTable_->setRowCount(static_cast<int>(rows.size()));
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& [pid, a] = rows[i];
        const int r = static_cast<int>(i);
        resultTable_->setItem(r, 0, new QTableWidgetItem(pid));
        resultTable_->setItem(r, 1, new QTableWidgetItem(
                                         a.known ? (a.truthNg ? QStringLiteral("NG") : QStringLiteral("OK"))
                                                 : QStringLiteral("—")));
        auto* pred = new QTableWidgetItem(a.predNg ? QStringLiteral("NG") : QStringLiteral("OK"));
        if (a.known) {
            pred->setForeground(a.truthNg == a.predNg ? QColor(30, 140, 60) : QColor(200, 40, 40));
        }
        resultTable_->setItem(r, 2, pred);
        resultTable_->setItem(r, 3, new QTableWidgetItem(QString::number(a.maxScore, 'f', 2)));
    }

    // 性能
    if (timed > 0) {
        const double wallMs = static_cast<double>(wall.nsecsElapsed()) / 1e6;
        std::sort(perImageMs.begin(), perImageMs.end());
        bench.valid = true;
        bench.images = timed;
        bench.parts = static_cast<int>(agg.size());
        bench.totalMs = wallMs;
        bench.decodeMs = sumDecode;
        bench.analyzeMs = sumAnalyze;
        bench.avgMs = (sumDecode + sumAnalyze) / timed;
        bench.medianMs = perImageMs[perImageMs.size() / 2];
        bench.minMs = perImageMs.front();
        bench.maxMs = perImageMs.back();
        bench.fps = wallMs > 0 ? (timed * 1000.0 / wallMs) : 0.0;
        bench.partsPerSec = (wallMs > 0 && bench.parts > 0) ? (bench.parts * 1000.0 / wallMs) : 0.0;
        showBenchmark(bench);
    }

    bool anyBad = false;
    for (const auto& [pid, a] : rows) {
        if (!a.known || a.truthNg == a.predNg) continue;
        if (!anyBad) { log(QStringLiteral("\n[误判明细]")); anyBad = true; }
        log(QStringLiteral("  %1 真值=%2 判定=%3 最高分=%4")
                .arg(pid)
                .arg(a.truthNg ? "NG" : "OK")
                .arg(a.predNg ? "NG" : "OK")
                .arg(a.maxScore, 0, 'f', 3));
    }
    if (!anyBad) log(QStringLiteral("  （无）"));
}

void TrainWindow::showBenchmark(const BenchResult& b) {
    benchTable_->setRowCount(0);
    if (!b.valid) return;
    struct Row { QString n; QString v; QString u; };
    const QVector<Row> rows = {
        {QStringLiteral("样本张数"), QString::number(b.images), QStringLiteral("张")},
        {QStringLiteral("零件数"), QString::number(b.parts), QStringLiteral("颗")},
        {QStringLiteral("总耗时"), QString::number(b.totalMs / 1000.0, 'f', 2), QStringLiteral("秒")},
        {QStringLiteral("单张平均"), QString::number(b.avgMs, 'f', 1), QStringLiteral("ms")},
        {QStringLiteral("单张中位"), QString::number(b.medianMs, 'f', 1), QStringLiteral("ms")},
        {QStringLiteral("单张最快"), QString::number(b.minMs, 'f', 1), QStringLiteral("ms")},
        {QStringLiteral("单张最慢"), QString::number(b.maxMs, 'f', 1), QStringLiteral("ms")},
        {QStringLiteral("解码累计"), QString::number(b.decodeMs, 'f', 0), QStringLiteral("ms")},
        {QStringLiteral("识别累计"), QString::number(b.analyzeMs, 'f', 0), QStringLiteral("ms")},
        {QStringLiteral("吞吐率"), QString::number(b.fps, 'f', 2), QStringLiteral("张/秒")},
        {QStringLiteral("零件速率"), QString::number(b.partsPerSec, 'f', 2), QStringLiteral("颗/秒")},
        {QStringLiteral("内存峰值"), QString::number(b.memPeakMB, 'f', 1), QStringLiteral("MB")},
        {QStringLiteral("私有峰值"), QString::number(b.privPeakMB, 'f', 1), QStringLiteral("MB")},
    };
    benchTable_->setRowCount(rows.size());
    for (int i = 0; i < rows.size(); ++i) {
        benchTable_->setItem(i, 0, new QTableWidgetItem(rows[i].n));
        benchTable_->setItem(i, 1, new QTableWidgetItem(rows[i].v));
        benchTable_->setItem(i, 2, new QTableWidgetItem(rows[i].u));
    }
    log(QStringLiteral("\n[性能] 单张 %1 ms，吞吐 %2 张/秒，零件 %3 颗/秒，内存峰值 %4 MB")
            .arg(b.avgMs, 0, 'f', 1)
            .arg(b.fps, 0, 'f', 2)
            .arg(b.partsPerSec, 0, 'f', 2)
            .arg(b.memPeakMB, 0, 'f', 1));
}
