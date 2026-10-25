#include "gui/SplitVerifyWindow.hpp"

#include "gui/BoxCanvas.hpp"

#include <QApplication>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTextStream>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cstdio>
#include <functional>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"
#include "core/naming.hpp"

namespace {

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

QImage matToQImage(const cv::Mat& m) {
    if (m.empty() || m.type() != CV_8UC1) return {};
    return QImage(m.data, m.cols, m.rows, static_cast<int>(m.step), QImage::Format_Grayscale8)
        .copy();
}

QColor regionColor() { return QColor(70, 150, 210); }
QColor charColor() { return QColor(110, 255, 150); }

QString appDir() { return QCoreApplication::applicationDirPath(); }

QString viewTag(int v) {
    return QStringLiteral("v%1").arg(v, 2, 10, QLatin1Char('0'));
}

}  // namespace

// ---------------------------------------------------------------- 后台精对齐

/// 后台任务：跑 chardet::refine()，避免阻塞界面
///
/// 注意：这个类定义在 .cpp 中，moc 不处理，因此不能声明 Q_OBJECT。
/// 结果通过 QThread::finished 信号回到主线程（线程对象是窗口的子对象，
/// 信号天然在创建它的线程派发），不使用跨线程回调。
class RefineWorker : public QObject {
public:
    std::vector<cv::Mat> images;
    chardet::Model coarse;
    chardet::Config cfg;
    int view = 0;
    const std::atomic<bool>* cancel = nullptr;   // 指向窗口的取消标志
    std::atomic<int> done{0};                    // 已完成样本数
    std::atomic<int> total{0};                   // 总样本数

    chardet::Model result;
    chardet::RefineReport report;

    void run() {
        std::function<void(int, int)> prog = [this](int d, int t) {
            done.store(d);
            total.store(t);
        };
        result = chardet::refine(images, coarse, cfg, 20, 0.60, &report, cancel, &prog);
        done.store(total.load());
    }
};

// ---------------------------------------------------------------- 构造

SplitVerifyWindow::SplitVerifyWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("六视角模型图与选区配置"));
    resize(1560, 980);
    rootDir_ = QDir(appDir()).absoluteFilePath(QStringLiteral("model"));
    buildUi();
    tryLoadExistingModels();
    updateStatus();
}

SplitVerifyWindow::~SplitVerifyWindow() {
    // 若精对齐仍在进行中就被关闭，必须先把工作线程停住再销毁。
    // 否则线程会在对象销毁后继续访问它们 → 崩溃。
    if (workerThread_) {
        // 置取消标志：refine() 每处理完一张样本会检查它，提前返回
        cancelFlag_.store(true);
        workerThread_->quit();
        if (!workerThread_->wait(5000)) {
            workerThread_->terminate();
            workerThread_->wait(2000);
        }
        delete workerThread_;
        workerThread_ = nullptr;
    }
    delete worker_;
    worker_ = nullptr;
}

int SplitVerifyWindow::currentView() const {
    if (!viewCombo_ || viewCombo_->currentIndex() < 0) return 0;
    return viewCombo_->currentData().toInt();
}

QString SplitVerifyWindow::viewDir(int v) const {
    return QDir(rootDir_).absoluteFilePath(viewTag(v));
}

QString SplitVerifyWindow::modelPath(int v) const {
    return QDir(viewDir(v)).absoluteFilePath(QStringLiteral("model.png"));
}

QString SplitVerifyWindow::configPath() const {
    // 共享选区配置：放在根目录，不属于任何视角
    return QDir(rootDir_).absoluteFilePath(QStringLiteral("regions.conf"));
}

QString SplitVerifyWindow::versionsDir(int v) const {
    return QDir(viewDir(v)).absoluteFilePath(QStringLiteral("versions"));
}

// ---------------------------------------------------------------- UI

void SplitVerifyWindow::buildUi() {
    auto* central = new QWidget;
    auto* root = new QVBoxLayout(central);

    // ---- 根目录 + 视角选择
    {
        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel(QStringLiteral("模型根目录：")));
        dirEdit_ = new QLineEdit(rootDir_);
        dirEdit_->setToolTip(QStringLiteral("其下会有 v00..v05 六个子目录"));
        connect(dirEdit_, &QLineEdit::editingFinished, this, [this]() {
            rootDir_ = dirEdit_->text().trimmed();
            tryLoadExistingModels();
            updateStatus();
        });
        row->addWidget(dirEdit_, 1);

        auto* browse = new QPushButton(QStringLiteral("浏览…"));
        connect(browse, &QPushButton::clicked, this, [this]() {
            const QString d = QFileDialog::getExistingDirectory(
                this, QStringLiteral("选择模型根目录"), rootDir_);
            if (!d.isEmpty()) {
                rootDir_ = d;
                dirEdit_->setText(d);
                tryLoadExistingModels();
                updateStatus();
            }
        });
        row->addWidget(browse);
        root->addLayout(row);
    }

    // ---- 视角切换
    {
        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel(QStringLiteral("当前视角：")));
        viewCombo_ = new QComboBox;
        for (int v = 0; v < kViews; ++v) {
            viewCombo_->addItem(viewTag(v), v);
        }
        viewCombo_->setMinimumWidth(140);
        connect(viewCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                &SplitVerifyWindow::onViewChanged);
        row->addWidget(viewCombo_);

        auto* sampleBtn = new QPushButton(QStringLiteral("① 选择样本文件夹，训练六个视角…"));
        sampleBtn->setMinimumHeight(36);
        connect(sampleBtn, &QPushButton::clicked, this, &SplitVerifyWindow::chooseSampleDir);
        row->addWidget(sampleBtn, 3);

        auto* openBtn = new QPushButton(QStringLiteral("打开单张模型图…"));
        connect(openBtn, &QPushButton::clicked, this, &SplitVerifyWindow::openModelImage);
        row->addWidget(openBtn);

        root->addLayout(row);
    }

    // ---- 历史版本
    {
        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel(QStringLiteral("历史版本：")));
        versionCombo_ = new QComboBox;
        versionCombo_->setMinimumWidth(430);
        row->addWidget(versionCombo_, 1);

        auto* loadBtn = new QPushButton(QStringLiteral("载入选中版本"));
        connect(loadBtn, &QPushButton::clicked, this, &SplitVerifyWindow::loadSelectedVersion);
        row->addWidget(loadBtn);
        root->addLayout(row);
    }

    infoLabel_ = new QLabel(QStringLiteral("尚未载入模型图"));
    infoLabel_->setStyleSheet(QStringLiteral("color:#666;"));
    root->addWidget(infoLabel_);

    // ---- 主体
    auto* body = new QHBoxLayout;
    canvas_ = new BoxCanvas;
    canvas_->setDrawEnabled(true);
    connect(canvas_, &BoxCanvas::boxCreated, this, &SplitVerifyWindow::onBoxCreated);
    connect(canvas_, &BoxCanvas::selectionChanged, this, &SplitVerifyWindow::onSelectionChanged);
    connect(canvas_, &BoxCanvas::boxDeleteRequested, this, &SplitVerifyWindow::onDeleteRequested);
    body->addWidget(canvas_, 1);

    auto* right = new QWidget;
    right->setMinimumWidth(320);
    right->setMaximumWidth(370);
    auto* rl = new QVBoxLayout(right);
    rl->setContentsMargins(0, 0, 0, 0);

    auto* pbox = new QGroupBox(QStringLiteral("切分参数"));
    auto* pf = new QFormLayout(pbox);

    colThrSpin_ = new QSpinBox;
    colThrSpin_->setRange(1, 90);
    colThrSpin_->setValue(3);
    colThrSpin_->setSuffix(QStringLiteral(" %"));
    colThrSpin_->setToolTip(
        QStringLiteral("列投影阈值。\n细笔画（1、L 的竖）被切碎 → 调低\n相邻字粘连 → 调高"));
    pf->addRow(QStringLiteral("列阈值"), colThrSpin_);

    gapSpin_ = new QSpinBox;
    gapSpin_->setRange(0, 60);
    gapSpin_->setValue(10);
    gapSpin_->setSuffix(QStringLiteral(" %"));
    gapSpin_->setToolTip(QStringLiteral("间隙合并。\n一个字被切成两半 → 调大"));
    pf->addRow(QStringLiteral("间隙合并"), gapSpin_);

    auto* resplitBtn = new QPushButton(QStringLiteral("按新参数重新切分"));
    connect(resplitBtn, &QPushButton::clicked, this, &SplitVerifyWindow::resplitAll);
    pf->addRow(resplitBtn);

    rl->addWidget(pbox);

    // ---- 去黑点（互斥按钮）
    auto* dbox = new QGroupBox(QStringLiteral("模型图去黑点"));
    auto* dl = new QVBoxLayout(dbox);

    auto makeBtn = [&](const QString& text, const QString& tip, int mode) {
        auto* b = new QPushButton(text);
        b->setCheckable(true);
        b->setToolTip(tip);
        connect(b, &QPushButton::clicked, this, [this, mode]() { setDenoiseMode(mode); });
        dl->addWidget(b);
        return b;
    };

    fillHoleBtn_ = makeBtn(
        QStringLiteral("填笔画黑点"),
        QStringLiteral("在已划出的区域内填平笔画内部的黑色凹陷（孔洞填充）。\n"
                       "只填被笔画完全包围的封闭小孔，笔画之间的缝隙不动，\n"
                       "因此不会造成笔画粘连。适合精对齐之前使用。"),
        2);
    rawBtn_ = makeBtn(QStringLiteral("还原原图"),
                      QStringLiteral("不做任何处理，显示模型图原始状态。"), 0);
    rawBtn_->setChecked(true);

    auto* srow = new QHBoxLayout;
    srow->addWidget(new QLabel(QStringLiteral("孔洞面积门限")));
    denoiseStrengthSpin_ = new QSpinBox;
    denoiseStrengthSpin_->setRange(100, 5000);
    denoiseStrengthSpin_->setSingleStep(100);
    denoiseStrengthSpin_->setValue(500);
    denoiseStrengthSpin_->setSuffix(QStringLiteral(" px"));
    denoiseStrengthSpin_->setToolTip(
        QStringLiteral("小于此面积的孔洞才会被填。\n\n"
                       "实测（v00 型号行）：\n"
                       "  真黑点： 1 ~ 137 px\n"
                       "  字符内环与结构空白： 738 px 以上（8 的上环 = 1102）\n\n"
                       "因此 500 是安全分界：大于所有真黑点(<200)，小于所有内环(>700)。\n"
                       "若发现内环被误填，调小；若发现大黑点未处理，调大。"));
    srow->addWidget(denoiseStrengthSpin_, 1);
    dl->addLayout(srow);

    rl->addWidget(dbox);
    rl->addWidget(new QLabel(QStringLiteral("已划出的区域（六视角共用）")));
    list_ = new QListWidget;
    rl->addWidget(list_, 1);
    body->addWidget(right);
    root->addLayout(body, 1);

    // ---- 进度条（精对齐时显示，平时隐藏）
    {
        auto* prow = new QHBoxLayout;
        progressBar_ = new QProgressBar;
        progressBar_->setRange(0, 100);
        progressBar_->setValue(0);
        progressBar_->setTextVisible(true);
        progressBar_->setFormat(QStringLiteral("%p%"));
        progressBar_->setMinimumHeight(20);
        progressBar_->setVisible(false);
        prow->addWidget(progressBar_, 1);
        root->addLayout(prow);
    }

    // ---- 底部
    {
        auto* row = new QHBoxLayout;
        hintLabel_ = new QLabel(
            QStringLiteral("在模型图上按住左键拖拽，框住一行文字 → 松开即自动切出单个字符"));
        hintLabel_->setStyleSheet(QStringLiteral("color:#888;"));
        row->addWidget(hintLabel_, 1);

        auto* delBtn = new QPushButton(QStringLiteral("删除选中"));
        connect(delBtn, &QPushButton::clicked, this, &SplitVerifyWindow::deleteSelected);
        row->addWidget(delBtn);

        auto* clrBtn = new QPushButton(QStringLiteral("清空全部区域"));
        connect(clrBtn, &QPushButton::clicked, this, &SplitVerifyWindow::clearAll);
        row->addWidget(clrBtn);

        auto* saveBtn = new QPushButton(QStringLiteral("③ 保存选区配置"));
        saveBtn->setMinimumWidth(150);
        saveBtn->setDefault(true);
        connect(saveBtn, &QPushButton::clicked, this, &SplitVerifyWindow::saveConfig);
        row->addWidget(saveBtn);

        auto* refineBtn = new QPushButton(QStringLiteral("④ 精对齐重建"));
        refineBtn->setMinimumWidth(120);
        refineBtn->setToolTip(QStringLiteral("用当前视角划好的字符框重建模型图"));
        connect(refineBtn, &QPushButton::clicked, this, &SplitVerifyWindow::refineModel);
        row->addWidget(refineBtn);

        auto* detectBtn = new QPushButton(QStringLiteral("⑤ 检测产品"));
        detectBtn->setMinimumWidth(110);
        detectBtn->setToolTip(
            QStringLiteral("载入一张产品图，与当前视角的精修模型比对，\n"
                           "逐字符检查 漏字 / 漏墨 / 多墨。"));
        connect(detectBtn, &QPushButton::clicked, this, &SplitVerifyWindow::detectProduct);
        row->addWidget(detectBtn);

        auto* closeBtn = new QPushButton(QStringLiteral("关闭"));
        connect(closeBtn, &QPushButton::clicked, this, &QWidget::close);
        row->addWidget(closeBtn);
        root->addLayout(row);
    }

    setCentralWidget(central);
}

// ---------------------------------------------------------------- ① 训练六个视角

void SplitVerifyWindow::chooseSampleDir() {
    const QString start = QDir(appDir()).absoluteFilePath(QStringLiteral(".."));
    const QString dir = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择装有多张 OK 产品图的文件夹（含 _00.._05 各视角）"), start);
    if (dir.isEmpty()) return;
    sampleDir_ = dir;

    const QDir d(dir);
    const QStringList all = d.entryList(
        {QStringLiteral("*.tif"), QStringLiteral("*.tiff"), QStringLiteral("*.png"),
         QStringLiteral("*.bmp"), QStringLiteral("*.jpg"), QStringLiteral("*.jpeg")},
        QDir::Files, QDir::Name);
    if (all.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("没有图片"),
                             QStringLiteral("该文件夹里没有找到图片。"));
        return;
    }

    // 按视角分组
    std::array<QStringList, kViews> groups;
    int unrecognized = 0;
    for (const QString& f : all) {
        const int v = markloc::charinsp::naming::viewOf(f.toStdString());
        if (v >= 0 && v < kViews) groups[static_cast<size_t>(v)].append(f);
        else ++unrecognized;
    }

    QStringList avail;
    for (int v = 0; v < kViews; ++v) {
        if (!groups[static_cast<size_t>(v)].isEmpty()) {
            avail << QStringLiteral("%1(%2张)").arg(viewTag(v)).arg(groups[static_cast<size_t>(v)].size());
        }
    }
    if (avail.isEmpty()) {
        QMessageBox::warning(
            this, QStringLiteral("无法识别视角"),
            QStringLiteral("文件名里没有 _00.._05 的视角标记。\n"
                           "共 %1 个文件无法归类。").arg(unrecognized));
        return;
    }

    if (QMessageBox::question(
            this, QStringLiteral("确认训练"),
            QStringLiteral("将训练以下视角：\n\n%1\n\n每个视角会生成一张独立模型图。\n继续？")
                .arg(avail.join(QStringLiteral("\n")))) != QMessageBox::Yes) {
        return;
    }

    // 逐个视角训练
    int ok = 0, failed = 0;
    for (int v = 0; v < kViews; ++v) {
        const QStringList& files = groups[static_cast<size_t>(v)];
        if (files.isEmpty()) continue;

        hintLabel_->setText(QStringLiteral("正在训练 %1（%2 张）…").arg(viewTag(v)).arg(files.size()));
        hintLabel_->setStyleSheet(QStringLiteral("color:#1565c0; font-weight:bold;"));
        QApplication::processEvents();

        chardet::TrainInput in;
        in.view = v;
        for (const QString& f : files) {
            cv::Mat g;
            if (readGray(d.absoluteFilePath(f), g)) in.images.push_back(std::move(g));
        }
        if (in.images.empty()) { ++failed; continue; }

        chardet::Config cfg;
        chardet::TrainReport rep;
        chardet::Model m = chardet::train(in, cfg, &rep);
        if (m.freq.empty()) { ++failed; continue; }

        ViewState& st = views_[static_cast<size_t>(v)];
        st.model = m.freq.clone();
        st.raw = st.model.clone();
        st.ref = m.ref.clone();
        st.trained = true;

        // 落盘
        QDir().mkpath(viewDir(v));
        cv::imwrite(modelPath(v).toStdString(), st.model);
        if (!st.ref.empty()) {
            cv::imwrite(QDir(viewDir(v)).absoluteFilePath(QStringLiteral("model_ref.png")).toStdString(),
                        st.ref);
        }
        saveVersion(v, QStringLiteral("初版"),
                    QStringLiteral("%1 张样本").arg(rep.usedImages));
        ++ok;
    }

    hintLabel_->setStyleSheet(QStringLiteral("color:#888;"));
    hintLabel_->setText(QStringLiteral("训练完成：成功 %1 个视角，失败 %2 个").arg(ok).arg(failed));

    // 切到第一个有结果的视角
    for (int v = 0; v < kViews; ++v) {
        if (views_[static_cast<size_t>(v)].trained) {
            viewCombo_->setCurrentIndex(v);
            break;
        }
    }
    onViewChanged(viewCombo_->currentIndex());

    QMessageBox::information(
        this, QStringLiteral("训练完成"),
        QStringLiteral("成功 %1 个视角，失败 %2 个。\n\n"
                       "现在可以切换视角，分别在各自的模型图上框选文字。")
            .arg(ok)
            .arg(failed));
}

// ---------------------------------------------------------------- 视角切换

void SplitVerifyWindow::onViewChanged(int index) {
    if (index < 0) return;
    cur_ = viewCombo_->currentData().toInt();
    syncCanvasFromView();
    refreshVersionList();
    updateStatus();
}

void SplitVerifyWindow::syncCanvasFromView() {
    const ViewState& st = views_[static_cast<size_t>(cur_)];
    if (st.model.empty()) {
        canvas_->clearImage();
        canvas_->setDrawEnabled(false);
        qimg_ = QImage();
    } else {
        qimg_ = matToQImage(st.model);
        canvas_->setImage(qimg_);
        canvas_->fitToWindow();
        // 选区只在 v00 上定义；其他视角只读显示，避免同一份选区被不同视角改乱
        canvas_->setDrawEnabled(cur_ == 0);
    }
    refreshCanvas();
    refreshList();
}

// ---------------------------------------------------------------- 载入

void SplitVerifyWindow::tryLoadExistingModels() {
    for (int v = 0; v < kViews; ++v) {
        ViewState& st = views_[static_cast<size_t>(v)];
        st = ViewState{};
        const QString mp = modelPath(v);
        if (!QFile::exists(mp)) continue;

        cv::Mat g;
        if (!readGray(mp, g)) continue;
        st.model = g;
        st.raw = g.clone();
        st.trained = true;

        const QString rf = QDir(viewDir(v)).absoluteFilePath(QStringLiteral("model_ref.png"));
        if (QFile::exists(rf)) readGray(rf, st.ref);
    }

    // 载入共享选区配置（只读一次）
    regions_.clear();
    QFile f(configPath());
    if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream ts(&f);
        int nRegion = 0;
        ts >> nRegion;
        for (int i = 0; i < nRegion && !ts.atEnd(); ++i) {
            int x = 0, y = 0, w = 0, h = 0, nChar = 0;
            ts >> x >> y >> w >> h >> nChar;
            Region r;
            r.rect = cv::Rect(x, y, w, h);
            for (int c = 0; c < nChar; ++c) {
                int bx = 0, by = 0, bw = 0, bh = 0;
                ts >> bx >> by >> bw >> bh;
                chardet::CharBox cb;
                cb.bbox = cv::Rect(bx, by, bw, bh);
                cb.extend = cv::Rect(bx - 4, by - 4, bw + 8, bh + 8);
                cb.index = c;
                r.chars.push_back(cb);
            }
            regions_.push_back(std::move(r));
        }
    }

    syncCanvasFromView();
    refreshVersionList();
}

void SplitVerifyWindow::openModelImage() {
    const QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择模型图（会载入到当前视角）"), rootDir_,
        QStringLiteral("图像 (*.png *.tif *.tiff *.bmp);;所有文件 (*.*)"));
    if (path.isEmpty()) return;

    cv::Mat g;
    if (!readGray(path, g)) {
        QMessageBox::warning(this, QStringLiteral("打不开"), QStringLiteral("无法读取该图片。"));
        return;
    }
    ViewState& st = views_[static_cast<size_t>(cur_)];
    st.model = g;
    st.raw = g.clone();
    st.trained = true;
    // 不清空共享选区：选区对所有视角通用，换模型图不影响它
    syncCanvasFromView();
    updateStatus();
}

// ---------------------------------------------------------------- 状态

void SplitVerifyWindow::updateStatus() {
    const ViewState& st = views_[static_cast<size_t>(cur_)];
    if (st.model.empty()) {
        // 统计有多少视角可用
        int trained = 0;
        for (const auto& v : views_) if (v.trained) ++trained;
        infoLabel_->setText(QStringLiteral("%1 尚未训练。已有 %2 个视角可用。")
                                .arg(viewTag(cur_)).arg(trained));
        infoLabel_->setStyleSheet(QStringLiteral("color:#c62828;"));
        return;
    }
    int total = 0;
    for (const auto& r : regions_) total += static_cast<int>(r.chars.size());

    const QString editable = (cur_ == 0) ? QStringLiteral("可编辑选区")
                                         : QStringLiteral("只读（选区在 v00 上编辑）");
    infoLabel_->setText(QStringLiteral("%1   模型图 %2×%3    %4 个区域，共 %5 个字符    [%6]")
                            .arg(viewTag(cur_))
                            .arg(st.model.cols)
                            .arg(st.model.rows)
                            .arg(regions_.size())
                            .arg(total)
                            .arg(editable));
    infoLabel_->setStyleSheet(cur_ == 0 ? QStringLiteral("color:#333;")
                                        : QStringLiteral("color:#1565c0;"));
}

// ---------------------------------------------------------------- 切分

void SplitVerifyWindow::splitRegion(int idx) {
    ViewState& st = views_[static_cast<size_t>(cur_)];
    if (idx < 0 || idx >= static_cast<int>(regions_.size())) return;
    if (st.model.empty()) return;

    auto& r = regions_[static_cast<size_t>(idx)];
    const cv::Rect roi = r.rect & cv::Rect(0, 0, st.model.cols, st.model.rows);
    r.chars.clear();
    if (roi.width <= 0 || roi.height <= 0) return;

    chardet::Config cfg;
    cfg.colThrRatio = colThrSpin_->value() / 100.0;
    cfg.gapMergeRatio = gapSpin_->value() / 100.0;

    const auto units = chardet::split(st.model(roi), cfg);
    int ci = 0;
    for (const auto& u : units) {
        chardet::CharBox cb;
        cb.bbox = u.bbox + cv::Point(roi.x, roi.y);
        cb.extend = u.extend + cv::Point(roi.x, roi.y);
        cb.index = ci++;
        r.chars.push_back(cb);
    }
}

void SplitVerifyWindow::onBoxCreated(const QRectF& r) {
    if (cur_ != 0) {
        hintLabel_->setText(
            QStringLiteral("选区只在 v00 上定义。请切到 v00 画框，其余视角自动共用。"));
        return;
    }
    ViewState& st = views_[static_cast<size_t>(cur_)];
    if (st.model.empty()) {
        hintLabel_->setText(QStringLiteral("当前视角还没有模型图，请先训练"));
        return;
    }
    const QRectF clipped = r.intersected(QRectF(0, 0, st.model.cols, st.model.rows));
    if (clipped.width() < 12 || clipped.height() < 12) return;

    Region reg;
    reg.rect = cv::Rect(static_cast<int>(clipped.x()), static_cast<int>(clipped.y()),
                        static_cast<int>(clipped.width()), static_cast<int>(clipped.height()));
    regions_.push_back(reg);
    splitRegion(static_cast<int>(regions_.size()) - 1);

    const int n = static_cast<int>(regions_.back().chars.size());
    if (n == 0) {
        hintLabel_->setText(QStringLiteral("框里没找到文字，试试框大一点或调低列阈值"));
    } else if (n == 1) {
        hintLabel_->setText(QStringLiteral("只切出 1 个字符。若这行有多个字，把框拉长"));
    } else {
        hintLabel_->setText(QStringLiteral("%1：切出 %2 个字符").arg(viewTag(cur_)).arg(n));
    }
    refreshCanvas();
    refreshList();
    updateStatus();
}

void SplitVerifyWindow::resplitAll() {
    ViewState& st = views_[static_cast<size_t>(cur_)];
    if (regions_.empty()) {
        hintLabel_->setText(QStringLiteral("当前视角还没有框选区域"));
        return;
    }
    int total = 0;
    for (int i = 0; i < static_cast<int>(regions_.size()); ++i) {
        splitRegion(i);
        total += static_cast<int>(regions_[static_cast<size_t>(i)].chars.size());
    }
    refreshCanvas();
    refreshList();
    updateStatus();
    hintLabel_->setText(QStringLiteral("%1：列阈值 %2%、间隙 %3% → 共 %4 个字符")
                            .arg(viewTag(cur_))
                            .arg(colThrSpin_->value())
                            .arg(gapSpin_->value())
                            .arg(total));
}

void SplitVerifyWindow::refreshCanvas() {
    QList<EditableBox> disp;
    const ViewState& st = views_[static_cast<size_t>(cur_)];
    int id = 1;
    for (size_t ri = 0; ri < regions_.size(); ++ri) {
        const auto& r = regions_[ri];
        EditableBox rb;
        rb.id = id++;
        rb.rect = QRectF(r.rect.x, r.rect.y, r.rect.width, r.rect.height);
        rb.color = regionColor();
        rb.label = QStringLiteral("区%1").arg(ri + 1);
        disp.append(rb);
        for (size_t ci = 0; ci < r.chars.size(); ++ci) {
            const auto& c = r.chars[ci];
            EditableBox cb;
            cb.id = id++;
            cb.rect = QRectF(c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height);
            cb.color = charColor();
            cb.label = QString::number(ci + 1);
            disp.append(cb);
        }
    }
    canvas_->setBoxes(disp);
}

void SplitVerifyWindow::refreshList() {
    if (!list_) return;
    list_->clear();
    const ViewState& st = views_[static_cast<size_t>(cur_)];
    for (size_t ri = 0; ri < regions_.size(); ++ri) {
        const auto& r = regions_[ri];
        auto* item = new QListWidgetItem(QStringLiteral("区域%1：%2 个字符  (%3,%4) %5×%6")
                                             .arg(ri + 1)
                                             .arg(r.chars.size())
                                             .arg(r.rect.x).arg(r.rect.y)
                                             .arg(r.rect.width).arg(r.rect.height));
        item->setForeground(regionColor());
        list_->addItem(item);
    }
    if (list_->count() == 0) {
        auto* it = new QListWidgetItem(QStringLiteral("（本视角还没有区域）"));
        it->setFlags(Qt::NoItemFlags);
        list_->addItem(it);
    }
}

void SplitVerifyWindow::onSelectionChanged(int id) { (void)id; }

// ---------------------------------------------------------------- 去黑点

cv::Mat& SplitVerifyWindow::rawModel(int v) {
    ViewState& st = views_[static_cast<size_t>(v)];
    if (st.raw.empty() && !st.model.empty()) st.raw = st.model.clone();
    return st.raw;
}

void SplitVerifyWindow::setDenoiseMode(int mode) {
    ViewState& st = views_[static_cast<size_t>(cur_)];
    if (st.model.empty() && st.raw.empty()) {
        hintLabel_->setText(QStringLiteral("当前视角没有模型图"));
        return;
    }

    const cv::Mat base = rawModel(cur_).clone();
    if (base.empty()) {
        hintLabel_->setText(QStringLiteral("找不到原始模型图"));
        return;
    }

    denoiseMode_ = mode;
    fillHoleBtn_->setChecked(mode == 2);
    rawBtn_->setChecked(mode == 0);

    if (mode == 0) {
        st.model = base.clone();
        syncCanvasFromView();
        updateStatus();
        hintLabel_->setText(QStringLiteral("%1：已还原原始模型图").arg(viewTag(cur_)));
        return;
    }

    // ---- 填笔画黑点（孔洞填充）
    //
    // 只填"被笔画完全包围的封闭小孔"：
    //   黑点是封闭凹陷 → 会被填
    //   笔画之间的缝隙是开放的 → 不受影响，因此不粘连
    // 填充值取周围笔画的高分位(90%)，不用均值——均值会留下灰环。
    // 笔画判定阈值。
    // 取 100 而非 128：边界黑点周围的笔画很窄，阈值高时包围圈不闭合、
    // 黑点被判成"开放区域"而漏掉。降到 100 后包围圈更易闭合，
    // 实测可多识别 2~3 个边界黑点，且不会把背景误判成笔画。
    constexpr int inkThr = 100;
    const int maxHoleArea = denoiseStrengthSpin_->value();

    cv::Mat out = base.clone();
    int regionCount = 0;
    long long filledPx = 0;

    for (const auto& r : regions_) {
        const cv::Rect roi = r.rect & cv::Rect(0, 0, base.cols, base.rows);
        if (roi.width <= 2 || roi.height <= 2) continue;

        cv::Mat patch = base(roi).clone();

        cv::Mat ink;
        cv::compare(patch, inkThr, ink, cv::CMP_GT);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(ink.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat filled = cv::Mat::zeros(patch.size(), CV_8U);
        cv::drawContours(filled, contours, -1, cv::Scalar(255), cv::FILLED);

        cv::Mat holes;
        cv::subtract(filled, ink, holes);

        cv::Mat labels, stats, cents;
        const int n = cv::connectedComponentsWithStats(holes, labels, stats, cents, 8, CV_32S);

        cv::Mat dst = out(roi);
        for (int i = 1; i < n; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area <= 0 || area > maxHoleArea) continue;   // 内环等大孔不填

            const int x0 = stats.at<int>(i, cv::CC_STAT_LEFT);
            const int y0 = stats.at<int>(i, cv::CC_STAT_TOP);
            const int hw = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int hh = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            const int pad = 3;
            const int ex = std::max(0, x0 - pad);
            const int ey = std::max(0, y0 - pad);
            const cv::Rect expand(ex, ey, std::min(patch.cols - ex, hw + 2 * pad),
                                  std::min(patch.rows - ey, hh + 2 * pad));

            std::vector<int> ringVals;
            for (int y = expand.y; y < expand.y + expand.height; ++y) {
                const uchar* prow = patch.ptr<uchar>(y);
                const int* lrow = labels.ptr<int>(y);
                for (int x = expand.x; x < expand.x + expand.width; ++x) {
                    if (lrow[x] == i) continue;
                    // 只取"真笔画"的灰度（>=200），不用 inkThr：
                    // inkThr 已降到 100，用它取样会把过渡带混进来，拉低填充值
                    if (prow[x] < 200) continue;
                    ringVals.push_back(prow[x]);
                }
            }
            int fillVal = 240;
            if (!ringVals.empty()) {
                std::sort(ringVals.begin(), ringVals.end());
                const size_t idx = (ringVals.size() * 9) / 10;
                fillVal = ringVals[std::min(idx, ringVals.size() - 1)];
                fillVal = std::max(fillVal, 200);
            }

            for (int y = y0; y < y0 + hh; ++y) {
                uchar* drow = dst.ptr<uchar>(y);
                const int* lrow = labels.ptr<int>(y);
                for (int x = x0; x < x0 + hw; ++x) {
                    if (lrow[x] == i) {
                        drow[x] = static_cast<uchar>(fillVal);
                    }
                }
            }
            // ② 再处理孔洞外紧邻的一圈过渡带（pad 像素内、灰度 < 200）
            //    只在本孔洞外接框内操作，不会波及字符内环
            for (int y = y0; y < y0 + hh; ++y) {
                uchar* drow = dst.ptr<uchar>(y);
                const int* lrow = labels.ptr<int>(y);
                for (int x = x0; x < x0 + hw; ++x) {
                    if (lrow[x] == i) continue;                 // 已处理
                    if (patch.at<uchar>(y, x) >= 200) continue; // 已是笔画色
                    // 检查 pad 邻域内是否有本孔洞的像素
                    bool near = false;
                    for (int dy = -pad; dy <= pad && !near; ++dy) {
                        const int ny = y + dy;
                        if (ny < 0 || ny >= patch.rows) continue;
                        const int* nrow = labels.ptr<int>(ny);
                        for (int dx = -pad; dx <= pad; ++dx) {
                            const int nx = x + dx;
                            if (nx < 0 || nx >= patch.cols) continue;
                            if (nrow[nx] == i) { near = true; break; }
                        }
                    }
                    if (near) drow[x] = static_cast<uchar>(fillVal);
                }
            }
            filledPx += area;
        }
        ++regionCount;
    }

    st.model = out;
    syncCanvasFromView();
    updateStatus();

    if (regionCount == 0) {
        hintLabel_->setText(QStringLiteral("⚠ 未划选区，未做处理。请先在 v00 上框选文字区域。"));
    } else {
        hintLabel_->setText(QStringLiteral("%1：孔洞填充完成，共填 %2 px（门限 %3 px）")
                                .arg(viewTag(cur_))
                                .arg(filledPx)
                                .arg(maxHoleArea));
    }
}


void SplitVerifyWindow::onDeleteRequested(int id) {
    if (cur_ != 0) {
        hintLabel_->setText(QStringLiteral("选区只在 v00 上编辑。请切到 v00 再删。"));
        return;
    }
    int cursor = 1;
    for (size_t ri = 0; ri < regions_.size(); ++ri) {
        if (id == cursor) {
            regions_.erase(regions_.begin() + ri);
            refreshCanvas();
            refreshList();
            updateStatus();
            return;
        }
        ++cursor;
        cursor += static_cast<int>(regions_[ri].chars.size());
    }
}

void SplitVerifyWindow::deleteSelected() {
    const int id = canvas_->selectedId();
    if (id <= 0) {
        hintLabel_->setText(QStringLiteral("请先在图上点选一个区域框"));
        return;
    }
    onDeleteRequested(id);
}

void SplitVerifyWindow::clearAll() {
    if (cur_ != 0) {
        hintLabel_->setText(QStringLiteral("选区只在 v00 上编辑。请切到 v00 再清空。"));
        return;
    }
    if (regions_.empty()) return;
    if (QMessageBox::question(this, QStringLiteral("确认"),
                              QStringLiteral("清空全部选区？六个视角都会受影响。")) !=
        QMessageBox::Yes) {
        return;
    }
    regions_.clear();
    refreshCanvas();
    refreshList();
    updateStatus();
}

// ---------------------------------------------------------------- 版本

void SplitVerifyWindow::saveVersion(int v, const QString& kind, const QString& note) {
    const ViewState& st = views_[static_cast<size_t>(v)];
    if (st.model.empty()) return;

    QDir().mkpath(versionsDir(v));

    const QString stamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const QString base = kind == QStringLiteral("精对齐") ? QStringLiteral("refined")
                                                         : QStringLiteral("coarse");
    QString stem = QStringLiteral("%1_%2").arg(stamp, base);
    QString path = QDir(versionsDir(v)).absoluteFilePath(stem + QStringLiteral(".png"));
    int n = 2;
    while (QFile::exists(path)) {
        path = QDir(versionsDir(v))
                   .absoluteFilePath(QStringLiteral("%1_%2.png").arg(stem).arg(n++));
    }

    if (!cv::imwrite(path.toStdString(), st.model)) return;
    if (!st.ref.empty()) {
        const QString rp = path.left(path.size() - 4) + QStringLiteral("_ref.png");
        cv::imwrite(rp.toStdString(), st.ref);
    }

    QFile f(QDir(versionsDir(v)).absoluteFilePath(QStringLiteral("versions.txt")));
    if (f.open(QIODevice::Append | QIODevice::Text)) {
        QTextStream ts(&f);
        ts << QDateTime::currentDateTime().toString(Qt::ISODate) << "\t" << kind << "\t"
           << QFileInfo(path).fileName() << "\t" << note << "\n";
        f.close();
    }
    refreshVersionList();
}

void SplitVerifyWindow::refreshVersionList() {
    if (!versionCombo_) return;
    const QString keep = versionCombo_->currentData().toString();
    versionCombo_->clear();

    const QString idx = QDir(versionsDir(cur_)).absoluteFilePath(QStringLiteral("versions.txt"));
    QFile f(idx);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        versionCombo_->addItem(QStringLiteral("（本视角暂无历史版本）"), QString());
        versionCombo_->setEnabled(false);
        return;
    }
    QTextStream ts(&f);
    while (!ts.atEnd()) {
        const QString line = ts.readLine();
        if (line.trimmed().isEmpty()) continue;
        const QStringList parts = line.split(QLatin1Char('\t'));
        if (parts.size() < 3) continue;
        const QString full = QDir(versionsDir(cur_)).absoluteFilePath(parts[2]);
        if (!QFile::exists(full)) continue;

        QString text = QStringLiteral("%1  %2").arg(parts[0], parts[1]);
        if (parts.size() > 3 && !parts[3].isEmpty()) {
            text += QStringLiteral("  —  %1").arg(parts[3]);
        }
        versionCombo_->addItem(text, full);
    }
    f.close();

    if (versionCombo_->count() == 0) {
        versionCombo_->addItem(QStringLiteral("（本视角暂无历史版本）"), QString());
        versionCombo_->setEnabled(false);
    } else {
        versionCombo_->setEnabled(true);
        if (!keep.isEmpty()) {
            for (int i = 0; i < versionCombo_->count(); ++i) {
                if (versionCombo_->itemData(i).toString() == keep) {
                    versionCombo_->setCurrentIndex(i);
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- ⑤ 检测产品

void SplitVerifyWindow::detectProduct() {
    const ViewState& st = views_[static_cast<size_t>(cur_)];
    if (st.model.empty()) {
        QMessageBox::warning(this, QStringLiteral("无模型"), QStringLiteral("当前视角没有模型图。"));
        return;
    }
    if (regions_.empty()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("请先在 v00 上框选文字区域。"));
        return;
    }

    const QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择要检测的产品图"), rootDir_,
        QStringLiteral("图像 (*.tif *.tiff *.png *.bmp *.jpg);;所有文件 (*.*)"));
    if (path.isEmpty()) return;

    cv::Mat gray;
    if (!readGray(path, gray)) {
        QMessageBox::warning(this, QStringLiteral("打不开"), QStringLiteral("无法读取该图片。"));
        return;
    }
    if (gray.size() != st.model.size()) {
        QMessageBox::warning(
            this, QStringLiteral("尺寸不符"),
            QStringLiteral("产品图 %1×%2 与模型图 %3×%4 尺寸不同。")
                .arg(gray.cols).arg(gray.rows).arg(st.model.cols).arg(st.model.rows));
        return;
    }

    // 1) 产品图 → 笔画信号
    chardet::Config cfg;
    const cv::Rect full(0, 0, gray.cols, gray.rows);
    cv::Mat sig = chardet::strokeSignal(gray, full, cfg, false);

    // 2) 与模型图配准（模型图是频率图，需转成同一量纲）
    if (!st.ref.empty()) {
        cv::Mat r32, s32;
        st.ref.convertTo(r32, CV_32F);
        sig.convertTo(s32, CV_32F);
        const cv::Point2d sh = cv::phaseCorrelate(s32, r32, cv::noArray());
        if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
            (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
            cv::warpAffine(sig, sig, M, sig.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
        }
    }

    // 3) 构造 chardet 模型并逐字符比对
    chardet::Model cm;
    cm.freq = st.model.clone();
    cm.config = cfg;
    cm.imageSize = st.model.size();
    for (size_t ri = 0; ri < regions_.size(); ++ri) {
        chardet::RegionInfo info;
        info.index = static_cast<int>(ri);
        info.rect = regions_[ri].rect;
        info.chars = regions_[ri].chars;
        cm.regions.push_back(std::move(info));
    }

    // 收集所有字符结果
    struct Hit {
        int region, index;
        cv::Rect rect;
        double score;
        bool missingChar;
        std::string label;
    };
    std::vector<Hit> hits;

    int totalChars = 0;
    for (size_t ri = 0; ri < cm.regions.size(); ++ri) {
        const auto units = chardet::toUnits(cm, static_cast<int>(ri));
        const auto matches = chardet::match(sig, units, cfg);
        totalChars += static_cast<int>(matches.size());
        for (const auto& mc : matches) {
            if (mc.score <= 0.0) continue;
            hits.push_back({static_cast<int>(ri), mc.index, mc.bbox, mc.score, mc.missingChar,
                            mc.label});
        }
    }

    // 4) 生成可视化
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto& h : hits) {
        cv::rectangle(vis, h.rect, cv::Scalar(60, 60, 230), 2);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "%.1f", h.score);
        cv::putText(vis, tag, cv::Point(h.rect.x, std::max(14, h.rect.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(60, 60, 230), 1, cv::LINE_AA);
    }
    const QString visPath = QDir(rootDir_).absoluteFilePath(QStringLiteral("_detect.png"));
    cv::imwrite(visPath.toStdString(), vis);

    // 5) 结果
    std::sort(hits.begin(), hits.end(),
              [](const Hit& a, const Hit& b) { return a.score > b.score; });

    int nMissing = 0, nMissInk = 0, nExtra = 0;
    for (const auto& h : hits) {
        if (h.missingChar) ++nMissing;
        else if (h.label.rfind("缺墨", 0) == 0) ++nMissInk;
        else if (h.label.rfind("多墨", 0) == 0) ++nExtra;
    }

    const bool ng = !hits.empty() && hits.front().score >= 1.0;

    QString msg;
    msg += QStringLiteral("产品：%1\n").arg(QFileInfo(path).fileName());
    msg += QStringLiteral("视角：%1    字符总数：%2\n\n").arg(viewTag(cur_)).arg(totalChars);
    msg += QStringLiteral("判定：%1\n\n").arg(ng ? QStringLiteral("NG") : QStringLiteral("OK"));
    msg += QStringLiteral("漏字 %1 处 ／ 漏墨 %2 处 ／ 多墨 %3 处\n")
               .arg(nMissing).arg(nMissInk).arg(nExtra);

    if (!hits.empty()) {
        msg += QStringLiteral("\n问题字符（按分数排序，前 10）：\n");
        for (size_t i = 0; i < hits.size() && i < 10; ++i) {
            msg += QStringLiteral("  区域%1 字符#%2  分数 %.2f  %3\n")
                       .arg(hits[i].region + 1)
                       .arg(hits[i].index + 1)
                       .arg(hits[i].score)
                       .arg(QString::fromStdString(hits[i].label));
        }
    } else {
        msg += QStringLiteral("\n未发现异常字符。");
    }
    msg += QStringLiteral("\n\n标注图：%1").arg(visPath);

    QMessageBox::information(this, QStringLiteral("检测结果"), msg);
    hintLabel_->setText(QStringLiteral("%1 检测完成：%2 处异常，判定 %3")
                            .arg(viewTag(cur_))
                            .arg(hits.size())
                            .arg(ng ? QStringLiteral("NG") : QStringLiteral("OK")));
}

void SplitVerifyWindow::loadSelectedVersion() {
    const QString path = versionCombo_->currentData().toString();
    if (path.isEmpty() || !QFile::exists(path)) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("还没有可载入的历史版本。"));
        return;
    }
    cv::Mat g;
    if (!readGray(path, g)) {
        QMessageBox::warning(this, QStringLiteral("载入失败"),
                             QStringLiteral("无法读取 %1").arg(path));
        return;
    }
    ViewState& st = views_[static_cast<size_t>(cur_)];
    st.model = g;
    st.raw = g.clone();
    st.trained = true;

    const QString rp = path.left(path.size() - 4) + QStringLiteral("_ref.png");
    if (QFile::exists(rp)) {
        cv::Mat r;
        if (readGray(rp, r)) st.ref = r;
    }

    syncCanvasFromView();
    updateStatus();
    hintLabel_->setText(QStringLiteral("%1 已载入版本：%2")
                            .arg(viewTag(cur_))
                            .arg(QFileInfo(path).fileName()));
}

// ---------------------------------------------------------------- ③ 保存

void SplitVerifyWindow::saveConfig() {
    ViewState& st = views_[static_cast<size_t>(cur_)];
    if (st.model.empty()) {
        QMessageBox::warning(this, QStringLiteral("无模型图"),
                             QStringLiteral("请先训练出 %1 的模型图。").arg(viewTag(cur_)));
        return;
    }
    if (regions_.empty()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("先框选并切分区域，再保存。"));
        return;
    }

    QDir().mkpath(viewDir(cur_));
    cv::imwrite(modelPath(cur_).toStdString(), st.model);
    if (!st.ref.empty()) {
        cv::imwrite(
            QDir(viewDir(cur_)).absoluteFilePath(QStringLiteral("model_ref.png")).toStdString(),
            st.ref);
    }

    QFile f(configPath());
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("保存失败"),
                             QStringLiteral("无法写入 %1").arg(configPath()));
        return;
    }
    QTextStream ts(&f);
    ts << "# 选区配置 — 六视角共用\n";
    ts << "# 格式：区域数 / 每区域一行(x y w h 字符数) / 每字符一行(x y w h)\n";
    ts << regions_.size() << "\n";
    for (const auto& r : regions_) {
        ts << r.rect.x << " " << r.rect.y << " " << r.rect.width << " " << r.rect.height << " "
           << r.chars.size() << "\n";
        for (const auto& c : r.chars) {
            ts << c.bbox.x << " " << c.bbox.y << " " << c.bbox.width << " " << c.bbox.height
               << "\n";
        }
    }
    f.close();

    // 同时写出 chardet 模块格式，供检测程序直接读取。
    //
    // 两套格式并存：
    //   · model.png + regions.conf 是 GUI 自己的显示/编辑格式
    //   · chardet 模型目录（model.json + freq.png + regions.txt）供检测使用
    // 之前只写前者，导致精修后的模型检测程序读不到。
    saveChardetModel(cur_);

    int total = 0;
    for (const auto& r : regions_) total += static_cast<int>(r.chars.size());
    hintLabel_->setText(
        QStringLiteral("选区已保存（六视角共用）：%1 个区域，%2 个字符").arg(regions_.size()).arg(total));
}

// ---------------------------------------------------------------- chardet 格式导出

bool SplitVerifyWindow::saveChardetModel(int v) {
    const ViewState& st = views_[static_cast<size_t>(v)];
    if (st.model.empty()) return false;

    // chardet 模型目录 = 同一视角目录（vNN/）
    // 布局：model.json + freq.png + ref.png + regions.txt
    chardet::Model cm;
    cm.version = 1;
    cm.view = v;
    cm.imageSize = st.model.size();
    cm.config = chardet::Config();
    cm.freq = st.model.clone();
    if (!st.ref.empty()) cm.ref = st.ref.clone();

    // 选区 → chardet 的 RegionInfo
    for (size_t ri = 0; ri < regions_.size(); ++ri) {
        chardet::RegionInfo info;
        info.index = static_cast<int>(ri);
        info.rect = regions_[ri].rect;
        info.chars = regions_[ri].chars;
        cm.regions.push_back(std::move(info));
    }
    if (cm.regions.empty()) return false;

    const std::string dir = viewDir(v).toStdString();
    if (!chardet::saveModel(dir, cm)) return false;

    // saveModel 只写 freq.png / regions.txt / model.json，
    // 另存一份 ref.png 供配准使用
    if (!st.ref.empty()) {
        cv::imwrite(QDir(viewDir(v)).absoluteFilePath(QStringLiteral("ref.png")).toStdString(),
                    st.ref);
    }
    return true;
}

// ---------------------------------------------------------------- ④ 精对齐
void SplitVerifyWindow::refineModel() {
    if (refining_) {
        QMessageBox::information(this, QStringLiteral("请稍候"),
                                 QStringLiteral("精对齐正在进行中。"));
        return;
    }
    const ViewState& st = views_[static_cast<size_t>(cur_)];
    if (st.model.empty()) {
        QMessageBox::warning(this, QStringLiteral("无模型图"),
                             QStringLiteral("请先训练出 %1 的模型图。").arg(viewTag(cur_)));
        return;
    }
    if (regions_.empty()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("请先在模型图上框选文字区域。"));
        return;
    }
    if (sampleDir_.isEmpty() || !QDir(sampleDir_).exists()) {
        QMessageBox::warning(
            this, QStringLiteral("缺少样本"),
            QStringLiteral("精对齐需要读取训练用的样本。\n\n"
                           "请先点「① 选择样本文件夹」重新走一遍，再点本按钮。"));
        return;
    }

    // 只读本视角的样本
    const QDir d(sampleDir_);
    const QStringList all = d.entryList(
        {QStringLiteral("*.tif"), QStringLiteral("*.tiff"), QStringLiteral("*.png"),
         QStringLiteral("*.bmp"), QStringLiteral("*.jpg"), QStringLiteral("*.jpeg")},
        QDir::Files, QDir::Name);

    std::vector<cv::Mat> images;
    for (const QString& f : all) {
        if (markloc::charinsp::naming::viewOf(f.toStdString()) != cur_) continue;
        cv::Mat g;
        if (readGray(d.absoluteFilePath(f), g)) images.push_back(std::move(g));
    }
    if (images.empty()) {
        QMessageBox::warning(this, QStringLiteral("没有样本"),
                             QStringLiteral("样本文件夹里没有 %1 的图片。").arg(viewTag(cur_)));
        return;
    }

    chardet::Model coarse;
    coarse.freq = st.model.clone();
    coarse.ref = st.ref.clone();
    coarse.config = chardet::Config();
    coarse.imageSize = st.model.size();
    for (size_t ri = 0; ri < regions_.size(); ++ri) {
        chardet::RegionInfo info;
        info.index = static_cast<int>(ri);
        info.rect = regions_[ri].rect;
        info.chars = regions_[ri].chars;
        coarse.regions.push_back(std::move(info));
    }

    refining_ = true;
    auto* wk = new RefineWorker;
    worker_ = wk;
    wk->images = std::move(images);
    wk->coarse = std::move(coarse);
    wk->cfg = chardet::Config();
    wk->view = cur_;
    cancelFlag_.store(false);       // 每次启动前复位
    wk->cancel = &cancelFlag_;

    // QThread 不设 parent：生命周期由我们在 onRefineDone 里全权管理。
    // 若设了 parent(this)，deleteLater() 删除后，窗口析构时父对象还会再删一次
    // → 重复释放崩溃。
    workerThread_ = new QThread;
    wk->moveToThread(workerThread_);

    // run() 结束后必须显式 quit()，否则线程会停在事件循环里，finished 永不发出。
    //
    // 这里踩过坑：只连 started → run()，没人在结束后让线程退出，
    // QThread::run() 默认 enters its own event loop，于是：
    //   进度条停在 100%（worker 已把 done 设满），但 onRefineDone 永不被调用，
    //   界面看起来"卡死"。
    // 正确做法：在 started 的槽里跑完计算后立即 quit()。
    QThread* th = workerThread_;
    connect(workerThread_, &QThread::started, wk, [wk, th]() {
        wk->run();
        th->quit();
    });

    connect(workerThread_, &QThread::finished, this, &SplitVerifyWindow::onRefineDone);

    hintLabel_->setText(QStringLiteral("%1 精对齐进行中（%2 张 × %3 字符），请稍候…")
                            .arg(viewTag(cur_))
                            .arg(wk->images.size())
                            .arg(wk->coarse.totalChars()));
    hintLabel_->setStyleSheet(QStringLiteral("color:#1565c0; font-weight:bold;"));
    setCursor(Qt::BusyCursor);

    // 进度轮询：每 500ms 读一次 worker 的进度，更新进度条
    if (progressBar_) {
        progressBar_->setValue(0);
        progressBar_->setFormat(QStringLiteral("0 / %1 张   0%").arg(wk->images.size()));
        progressBar_->setVisible(true);
    }
    if (progressTimer_) progressTimer_->deleteLater();
    progressTimer_ = new QTimer(this);
    connect(progressTimer_, &QTimer::timeout, this, [this]() {
        if (!worker_) return;
        auto* w = static_cast<RefineWorker*>(worker_);
        const int d = w->done.load();
        const int t = w->total.load();
        if (t <= 0) return;
        const int pct = static_cast<int>(100.0 * d / t);

        if (progressBar_) {
            progressBar_->setVisible(true);
            progressBar_->setValue(pct);
            progressBar_->setFormat(QStringLiteral("%1 / %2 张   %p%").arg(d).arg(t));
        }
        hintLabel_->setText(QStringLiteral("%1 精对齐进行中…").arg(viewTag(w->view)));
    });
    progressTimer_->start(500);

    workerThread_->start();
}

void SplitVerifyWindow::onRefineDone() {
    if (!worker_) return;
    auto* w = static_cast<RefineWorker*>(worker_);
    const chardet::RefineReport rep = w->report;
    cv::Mat newFreq = w->result.freq.clone();
    cv::Mat newRef = w->result.ref.clone();
    const int v = w->view;

    refining_ = false;
    unsetCursor();
    hintLabel_->setStyleSheet(QStringLiteral("color:#888;"));

    // 停止进度轮询
    if (progressTimer_) {
        progressTimer_->stop();
        progressTimer_->deleteLater();
        progressTimer_ = nullptr;
    }
    if (progressBar_) {
        progressBar_->setValue(100);
        progressBar_->setVisible(false);
    }

    // 线程清理
    //
    // 本函数由 QThread::finished 触发，此时 run() 已返回、线程已结束，
    // 因此可以安全地释放对象。
    //
    // 早期实现用 wait(3000) 后无条件 deleteLater()：
    //   精对齐实测 16~60 秒，3 秒必然超时，
    //   销毁仍在运行的 QThread 会崩溃。现改为由 finished 驱动。
    if (worker_) {
        worker_->deleteLater();
        worker_ = nullptr;
    }
    if (workerThread_) {
        workerThread_->deleteLater();
        workerThread_ = nullptr;
    }

    if (!rep.ok) {
        hintLabel_->setText(QStringLiteral("精对齐失败"));
        QMessageBox::critical(this, QStringLiteral("精对齐失败"),
                              QString::fromStdString(rep.message));
        return;
    }

    ViewState& st = views_[static_cast<size_t>(v)];
    st.model = std::move(newFreq);
    st.raw = st.model.clone();
    if (!newRef.empty()) st.ref = std::move(newRef);

    QDir().mkpath(viewDir(v));
    cv::imwrite(modelPath(v).toStdString(), st.model);
    if (!st.ref.empty()) {
        cv::imwrite(QDir(viewDir(v)).absoluteFilePath(QStringLiteral("model_ref.png")).toStdString(),
                    st.ref);
    }
    saveVersion(v, QStringLiteral("精对齐"),
                QStringLiteral("成功 %1 / 跳过 %2，峰值 %3")
                    .arg(rep.refinedPairs)
                    .arg(rep.skippedWeak)
                    .arg(rep.meanPeak, 0, 'f', 2));

    // 精修后的模型立即导出为 chardet 格式，检测程序可直接使用
    saveChardetModel(v);

    if (v == cur_) syncCanvasFromView();
    updateStatus();

    hintLabel_->setText(QStringLiteral("%1 精对齐完成：成功 %2 对，跳过 %3 对")
                            .arg(viewTag(v))
                            .arg(rep.refinedPairs)
                            .arg(rep.skippedWeak));

    QMessageBox::information(
        this, QStringLiteral("精对齐完成"),
        QStringLiteral("%1\n\n样本 %2 张，字符框 %3 个\n"
                       "成功对齐 %4 对，跳过弱匹配 %5 对\n"
                       "匹配峰值：均值 %6")
            .arg(viewTag(v))
            .arg(rep.usedImages)
            .arg(rep.totalChars)
            .arg(rep.refinedPairs)
            .arg(rep.skippedWeak)
            .arg(rep.meanPeak, 0, 'f', 3));
}
