#include "gui/TrainToolWindow.hpp"

#include "gui/CharBoxEditor.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
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
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "core/naming.hpp"

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

}  // namespace

TrainToolWindow::TrainToolWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("markloc — 模型构建（训练 / 选区 / 划字）"));
    resize(1280, 820);
    buildUi();
    imageDirEdit_->setText(QDir::current().absoluteFilePath("image"));
    modelDirEdit_->setText(QDir::current().absoluteFilePath("models/char"));
    log(QStringLiteral("就绪。① 训练模型 → ② 画选区 → ③ 划字。评估请用 markloc_gui。"));
}

void TrainToolWindow::buildUi() {
    auto* central = new QWidget;
    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(8);

    // ---------- 路径
    auto* pathBox = new QGroupBox(QStringLiteral("① 训练模型"));
    auto* form = new QFormLayout(pathBox);
    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        imageDirEdit_ = new QLineEdit;
        imageDirEdit_->setPlaceholderText(QStringLiteral("包含 OK 图片的文件夹"));
        auto* btn = new QPushButton(QStringLiteral("浏览…"));
        connect(btn, &QPushButton::clicked, this, &TrainToolWindow::chooseImageDir);
        h->addWidget(imageDirEdit_, 1);
        h->addWidget(btn);
        form->addRow(QStringLiteral("OK 样本目录"), row);
    }
    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        modelDirEdit_ = new QLineEdit;
        modelDirEdit_->setPlaceholderText(QStringLiteral("模型输出目录（含 v00..v05）"));
        auto* btn = new QPushButton(QStringLiteral("浏览…"));
        connect(btn, &QPushButton::clicked, this, &TrainToolWindow::chooseModelDir);
        h->addWidget(modelDirEdit_, 1);
        h->addWidget(btn);
        form->addRow(QStringLiteral("模型目录"), row);
    }
    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        baselineCombo_ = new QComboBox;
        baselineCombo_->setToolTip(
            QStringLiteral("继承该模型的阈值与区域开关；留空则用默认参数"));
        auto* btnRefresh = new QPushButton(QStringLiteral("刷新"));
        connect(btnRefresh, &QPushButton::clicked, this, &TrainToolWindow::refreshBaselineList);
        h->addWidget(baselineCombo_, 1);
        h->addWidget(btnRefresh);
        form->addRow(QStringLiteral("继承参数自"), row);
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
        autoCalibCheck_ = new QCheckBox(QStringLiteral("自动标定阈值"));
        autoCalibCheck_->setChecked(true);
        autoCalibCheck_->setToolTip(QStringLiteral("用 OK 样本分布反推各视角阈值"));
        h->addWidget(recursiveCheck_);
        h->addSpacing(10);
        h->addWidget(new QLabel(QStringLiteral("每视角上限")));
        h->addWidget(limitSpin_);
        h->addSpacing(10);
        h->addWidget(autoCalibCheck_);
        h->addStretch(1);
        form->addRow(QStringLiteral("选项"), row);
    }
    root->addWidget(pathBox);

    // ---------- 操作
    {
        auto* row = new QWidget;
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        trainBtn_ = new QPushButton(QStringLiteral("开始训练"));
        trainBtn_->setMinimumHeight(40);
        QFont f = trainBtn_->font();
        f.setPointSize(f.pointSize() + 2);
        f.setBold(true);
        trainBtn_->setFont(f);
        connect(trainBtn_, &QPushButton::clicked, this, &TrainToolWindow::startTrain);

        h->addWidget(new QLabel(QStringLiteral("视角:")));
        editViewCombo_ = new QComboBox;
        for (int v = 0; v < kViewCount; ++v) editViewCombo_->addItem(QStringLiteral("v%1").arg(viewTag(v)), v);
        h->addWidget(editViewCombo_);

        regionBtn_ = new QPushButton(QStringLiteral("② 选区与划字"));
        regionBtn_->setMinimumHeight(40);
        regionBtn_->setToolTip(QStringLiteral("在模型平均字形上画选区，自动切出单字符"));
        connect(regionBtn_, &QPushButton::clicked, this, &TrainToolWindow::openRegionEditor);

        auto* roiBtn = new QPushButton(QStringLiteral("导出 ROI 配置"));
        roiBtn->setMinimumHeight(40);
        roiBtn->setToolTip(
            QStringLiteral("把当前检测区域导出成可编辑的 rois.txt（改完重新评估即生效）"));
        connect(roiBtn, &QPushButton::clicked, this, &TrainToolWindow::exportRois);

        h->addWidget(trainBtn_, 2);
        h->addWidget(regionBtn_, 2);
        h->addWidget(roiBtn, 1);
        root->addWidget(row);
    }

    progress_ = new QProgressBar;
    progress_->setRange(0, 100);
    root->addWidget(progress_);

    auto* splitter = new QSplitter(Qt::Vertical);
    viewTable_ = new QTableWidget(0, 4);
    viewTable_->setHorizontalHeaderLabels({QStringLiteral("视角"), QStringLiteral("样本数"),
                                           QStringLiteral("状态"), QStringLiteral("说明")});
    viewTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    viewTable_->verticalHeader()->setVisible(false);
    viewTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    splitter->addWidget(viewTable_);

    logView_ = new QPlainTextEdit;
    logView_->setReadOnly(true);
    logView_->setFont(QFont(QStringLiteral("Consolas")));
    splitter->addWidget(logView_);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);
    root->addWidget(splitter, 1);

    setCentralWidget(central);
    statusLabel_ = new QLabel(QStringLiteral("就绪"));
    statusBar()->addWidget(statusLabel_, 1);

    refreshBaselineList();
}

void TrainToolWindow::log(const QString& s) {
    logView_->appendPlainText(s);
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void TrainToolWindow::setBusy(bool busy) {
    busy_ = busy;
    trainBtn_->setEnabled(!busy);
    regionBtn_->setEnabled(!busy);
    if (busy) QApplication::setOverrideCursor(Qt::WaitCursor);
    else QApplication::restoreOverrideCursor();
}

void TrainToolWindow::chooseImageDir() {
    const QString d = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择 OK 样本目录"), imageDirEdit_->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!d.isEmpty()) imageDirEdit_->setText(QDir::cleanPath(d));
}

void TrainToolWindow::chooseModelDir() {
    const QString d = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择模型目录"), modelDirEdit_->text(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!d.isEmpty()) modelDirEdit_->setText(QDir::cleanPath(d));
}

void TrainToolWindow::refreshBaselineList() {
    if (!baselineCombo_) return;
    const QString keep = baselineCombo_->currentData().toString();
    baselineCombo_->clear();
    baselineCombo_->addItem(QStringLiteral("（不继承，使用默认参数）"), QString());

    const QString modelsRoot = QDir::current().absoluteFilePath(QStringLiteral("models"));
    QDir root(modelsRoot);
    if (root.exists()) {
        for (const QString& sub : root.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)) {
            const QString p = root.absoluteFilePath(sub);
            if (QFileInfo::exists(p + QStringLiteral("/v00/params.txt"))) {
                baselineCombo_->addItem(sub, p);
            } else if (QFileInfo::exists(p + QStringLiteral("/char/v00/params.txt"))) {
                baselineCombo_->addItem(sub + QStringLiteral("/char"), p + QStringLiteral("/char"));
            }
        }
    }
    if (!keep.isEmpty()) {
        for (int i = 0; i < baselineCombo_->count(); ++i) {
            if (baselineCombo_->itemData(i).toString() == keep) {
                baselineCombo_->setCurrentIndex(i);
                break;
            }
        }
    }
}

std::array<int, kViewCount> TrainToolWindow::collectByView(
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

void TrainToolWindow::startTrain() {
    if (busy_) return;
    const QString imageDir = imageDirEdit_->text().trimmed();
    const QString modelDir = modelDirEdit_->text().trimmed();
    if (imageDir.isEmpty() || !QDir(imageDir).exists()) {
        log(QStringLiteral("❌ 样本目录不存在"));
        return;
    }
    if (modelDir.isEmpty()) {
        log(QStringLiteral("❌ 请选择模型目录"));
        return;
    }
    setBusy(true);
    logView_->clear();
    const QString baseline = baselineCombo_->currentData().toString();
    const bool ok = train(imageDir, modelDir, baseline);
    setBusy(false);
    statusLabel_->setText(ok ? QStringLiteral("训练完成 — 下一步：② 选区与划字")
                             : QStringLiteral("训练失败"));
}

bool TrainToolWindow::train(const QString& imageDir, const QString& modelDir,
                            const QString& baselineDir) {
    QElapsedTimer timer;
    timer.start();
    log(QStringLiteral("样本目录：%1").arg(imageDir));
    log(QStringLiteral("模型目录：%1").arg(modelDir));
    log(QStringLiteral("继承参数自：%1").arg(baselineDir.isEmpty() ? QStringLiteral("（无）") : baselineDir));

    std::array<std::vector<QString>, kViewCount> byView;
    const auto counts = collectByView(imageDir, byView);
    int total = 0;
    for (int v = 0; v < kViewCount; ++v) total += counts[static_cast<size_t>(v)];
    log(QStringLiteral("扫描到 %1 张图片").arg(total));
    if (total == 0) {
        log(QStringLiteral("❌ 没有找到符合命名约定的图片"));
        return false;
    }

    // 解码
    std::array<std::vector<cv::Mat>, kViewCount> okByView;
    std::array<int, kViewCount> decoded{};
    progress_->setRange(0, std::max(1, total));
    int done = 0;
    for (int v = 0; v < kViewCount; ++v) {
        for (const QString& p : byView[static_cast<size_t>(v)]) {
            cv::Mat img;
            if (readGray(p, img)) {
                okByView[static_cast<size_t>(v)].push_back(std::move(img));
                ++decoded[static_cast<size_t>(v)];
            }
            progress_->setValue(++done);
            if ((done % 25) == 0) QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
    }
    for (int v = 0; v < kViewCount; ++v) {
        log(QStringLiteral("  v%1：%2 张").arg(viewTag(v)).arg(decoded[static_cast<size_t>(v)]));
    }

    // 建模型
    log(QStringLiteral("\n开始建模…"));
    std::array<Params, kViewCount> ps;
    if (!baselineDir.isEmpty()) {
        int inherited = 0;
        for (int v = 0; v < kViewCount; ++v) {
            if (loadParams(baselineDir.toStdString(), v, ps[static_cast<size_t>(v)])) ++inherited;
        }
        log(QStringLiteral("已继承 %1 个视角的参数").arg(inherited));
    }
    Model model;
    std::string err;
    if (!buildModel(okByView, ps, model, &err)) {
        log(QStringLiteral("❌ 建模失败：%1").arg(QString::fromStdString(err)));
        return false;
    }

    // 自动标定
    if (autoCalibCheck_->isChecked() && baselineDir.isEmpty()) {
        log(QStringLiteral("\n标定各视角阈值…"));
        for (int v = 0; v < kViewCount; ++v) {
            if (!model.valid(v)) continue;
            std::vector<double> scores;
            for (const cv::Mat& img : okByView[static_cast<size_t>(v)]) {
                const Result r = analyze(img, model.views[static_cast<size_t>(v)],
                                         ps[static_cast<size_t>(v)], false);
                scores.push_back(r.score);
            }
            if (scores.empty()) continue;
            std::sort(scores.begin(), scores.end());
            const double okMax = scores[static_cast<size_t>(0.98 * (scores.size() - 1))];
            const double before = ps[static_cast<size_t>(v)].scratchT;
            if (okMax > 1e-6) {
                ps[static_cast<size_t>(v)].scratchT =
                    std::clamp(before * okMax / 0.85, before * 0.25, before * 40.0);
                log(QStringLiteral("  v%1：scratchT %2 → %3")
                        .arg(viewTag(v))
                        .arg(before, 0, 'f', 0)
                        .arg(ps[static_cast<size_t>(v)].scratchT, 0, 'f', 0));
            }
        }
    }

    // 保存
    const QString base = QDir(modelDir).absolutePath();
    QDir().mkpath(base);
    log(QStringLiteral("\n写入 %1").arg(base));
    int saved = 0;
    for (int v = 0; v < kViewCount; ++v) {
        if (!model.views[static_cast<size_t>(v)].ok) continue;
        if (!saveViewModel(base.toStdString(), v, model.views[static_cast<size_t>(v)], &err)) {
            log(QStringLiteral("  ❌ v%1：%2").arg(viewTag(v)).arg(QString::fromStdString(err)));
            continue;
        }
        saveParams(base.toStdString(), v, ps[static_cast<size_t>(v)]);
        ++saved;
        log(QStringLiteral("  ✅ v%1").arg(viewTag(v)));
    }

    // 表格
    viewTable_->setRowCount(kViewCount);
    for (int v = 0; v < kViewCount; ++v) {
        const bool has = model.views[static_cast<size_t>(v)].ok;
        auto set = [&](int c, const QString& s) { viewTable_->setItem(v, c, new QTableWidgetItem(s)); };
        set(0, viewTag(v));
        set(1, QString::number(decoded[static_cast<size_t>(v)]));
        set(2, has ? QStringLiteral("✅ 已建") : QStringLiteral("— 无样本"));
        set(3, has ? (ps[static_cast<size_t>(v)].invert ? QStringLiteral("暗刻印")
                                                        : QStringLiteral("亮刻印"))
                   : QStringLiteral("跳过"));
    }

    log(QStringLiteral("\n✅ 完成：%1 个视角，%2 s")
            .arg(saved)
            .arg(static_cast<double>(timer.nsecsElapsed()) / 1e9, 0, 'f', 1));
    log(QStringLiteral("下一步：点「② 选区与划字」在模型上画选区。"));
    return saved > 0;
}

void TrainToolWindow::exportRois() {
    const QString modelDir = modelDirEdit_->text().trimmed();
    if (modelDir.isEmpty() || !QDir(modelDir).exists()) {
        log(QStringLiteral("❌ 模型目录不存在"));
        return;
    }
    const int view = editViewCombo_->currentData().toInt();

    Model m;
    std::string err;
    if (!m.loadFrom(modelDir.toStdString(), &err) || !m.valid(view)) {
        log(QStringLiteral("❌ 模型目录下没有 v%1，请先训练").arg(viewTag(view)));
        return;
    }
    const Params& p = m.params[static_cast<size_t>(view)];
    if (exportRoisFile(modelDir.toStdString(), view, p)) {
        const QString path = QStringLiteral("%1/v%2/rois.txt").arg(modelDir).arg(viewTag(view));
        log(QStringLiteral("✅ 已导出 %1").arg(path));
        log(QStringLiteral("   编辑该文件后重新评估即可生效，不需要重新训练。"));
        log(QStringLiteral("   格式：名称 启用 x y 宽 高 [检查多余笔画]"));
    } else {
        log(QStringLiteral("❌ 导出失败"));
    }
}

void TrainToolWindow::openRegionEditor() {
    const QString modelDir = modelDirEdit_->text().trimmed();
    if (modelDir.isEmpty() || !QDir(modelDir).exists()) {
        log(QStringLiteral("❌ 模型目录不存在"));
        return;
    }
    const int view = editViewCombo_->currentData().toInt();
    Model m;
    std::string err;
    if (!m.loadFrom(modelDir.toStdString(), &err) || !m.valid(view)) {
        log(QStringLiteral("❌ 模型目录下没有 v%1，请先训练").arg(viewTag(view)));
        return;
    }
    RegionEditor dlg(modelDir, view, this);
    dlg.exec();
    log(QStringLiteral("选区编辑已关闭（v%1）").arg(viewTag(view)));
}

int TrainToolWindow::runHeadless(const QString& imageDir, const QString& modelDir,
                                 const QString& baselineDir) {
    logView_->clear();
    imageDirEdit_->setText(QDir::cleanPath(imageDir));
    modelDirEdit_->setText(QDir::cleanPath(modelDir));
    const bool ok = train(imageDir, modelDir, baselineDir);
    const QString text = logView_->toPlainText();
    std::fwrite(text.toUtf8().constData(), 1, static_cast<size_t>(text.toUtf8().size()), stdout);
    std::fflush(stdout);
    return ok ? 0 : 1;
}
