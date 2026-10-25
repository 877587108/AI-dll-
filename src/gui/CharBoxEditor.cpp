#include "gui/CharBoxEditor.hpp"

#include "gui/BoxCanvas.hpp"

#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

QColor regionColor() { return QColor(70, 150, 210); }
QColor charColor() { return QColor(110, 255, 150); }

QImage matToQImage(const cv::Mat& m) {
    if (m.empty() || m.type() != CV_8UC1) return {};
    return QImage(m.data, m.cols, m.rows, static_cast<int>(m.step), QImage::Format_Grayscale8)
        .copy();
}

}  // namespace

RegionEditor::RegionEditor(const QString& modelDir, int view, QWidget* parent)
    : QDialog(parent), modelDir_(modelDir), view_(view) {
    setWindowTitle(QStringLiteral("选区与划字 — v%1").arg(view, 2, 10, QLatin1Char('0')));
    resize(1500, 950);
    buildUi();
    reload();
}

void RegionEditor::buildUi() {
    auto* root = new QVBoxLayout(this);

    infoLabel_ = new QLabel(QStringLiteral("—"));
    root->addWidget(infoLabel_);

    auto* body = new QHBoxLayout;
    canvas_ = new BoxCanvas;
    canvas_->setDrawEnabled(true);
    connect(canvas_, &BoxCanvas::boxCreated, this, &RegionEditor::onBoxCreated);
    connect(canvas_, &BoxCanvas::selectionChanged, this, &RegionEditor::onSelectionChanged);
    connect(canvas_, &BoxCanvas::boxDeleteRequested, this, &RegionEditor::onDeleteRequested);
    body->addWidget(canvas_, 1);

    auto* right = new QWidget;
    right->setMinimumWidth(310);
    right->setMaximumWidth(360);
    auto* rl = new QVBoxLayout(right);
    rl->setContentsMargins(0, 0, 0, 0);

    auto* pbox = new QGroupBox(QStringLiteral("切分参数"));
    auto* pf = new QFormLayout(pbox);
    colThrSpin_ = new QSpinBox;
    colThrSpin_->setRange(1, 90);
    colThrSpin_->setValue(3);
    colThrSpin_->setSuffix(QStringLiteral(" %"));
    colThrSpin_->setToolTip(QStringLiteral("调低→细笔画不断；调高→少粘连"));
    pf->addRow(QStringLiteral("列阈值"), colThrSpin_);

    gapRatioSpin_ = new QSpinBox;
    gapRatioSpin_->setRange(0, 60);
    gapRatioSpin_->setValue(10);
    gapRatioSpin_->setSuffix(QStringLiteral(" %"));
    gapRatioSpin_->setToolTip(QStringLiteral("调大→更多字被合并"));
    pf->addRow(QStringLiteral("间隙合并"), gapRatioSpin_);
    rl->addWidget(pbox);

    rl->addWidget(new QLabel(QStringLiteral("区域与字符")));
    list_ = new QListWidget;
    rl->addWidget(list_, 1);

    // 图形选择：决定"下一次拖拽生成什么"。已有的区域不受影响。
    //
    // 为什么要分开：字符区和图案区的判据完全不同，
    // 字符区问"每个字的墨量对不对"，图案区只问"这条闭合曲线有没有断"。
    // 在选框这一步就把两者分开，后面才不会用错判据。
    {
        auto* shapeBox = new QGroupBox(QStringLiteral("新区域图形"));
        auto* sf = new QVBoxLayout(shapeBox);
        shapeRect_ = new QRadioButton(QStringLiteral("矩形 — 字符区（切字比对）"));
        shapeEllipse_ = new QRadioButton(QStringLiteral("椭圆 — 图案区（查环断裂）"));
        shapeCircle_ = new QRadioButton(QStringLiteral("圆 — 图案区（查环断裂）"));
        shapeRect_->setChecked(true);
        sf->addWidget(shapeRect_);
        sf->addWidget(shapeEllipse_);
        sf->addWidget(shapeCircle_);

        // 长轴角度。椭圆环在实物上多半是斜的：
        // SC 这个 logo 的长轴约 -13°，不设角度的话椭圆和环对不上，
        // 沿周长采样会一路擦边，量出来的环半径起伏能到几十像素，
        // 而墨线本身只有 5~8px 宽 —— 那个尺度下断点判据不成立。
        //
        // 这个值同时用于：① 新建椭圆时作为初始角度
        //               ② 选中已有图案区后改它，立刻按新角度重画并回写模型
        {
            auto* arow = new QHBoxLayout;
            arow->addWidget(new QLabel(QStringLiteral("长轴角度")));
            angleSpin_ = new QDoubleSpinBox;
            angleSpin_->setRange(-180.0, 180.0);
            angleSpin_->setDecimals(1);
            angleSpin_->setSingleStep(0.5);
            angleSpin_->setSuffix(QStringLiteral(" °"));
            angleSpin_->setToolTip(
                QStringLiteral("椭圆长轴相对水平方向的角度，逆时针为正。\n"
                               "选中已画的椭圆后改这里，会立刻按新角度重画。"));
            arow->addWidget(angleSpin_, 1);
            sf->addLayout(arow);
            connect(angleSpin_, &QDoubleSpinBox::valueChanged, this,
                    &RegionEditor::onAngleEdited);
        }

        auto sync = [this]() {
            if (!canvas_) return;
            canvas_->setNewShape(selectedShape());
        };
        connect(shapeRect_, &QRadioButton::toggled, this, sync);
        connect(shapeEllipse_, &QRadioButton::toggled, this, sync);
        connect(shapeCircle_, &QRadioButton::toggled, this, sync);
        rl->addWidget(shapeBox);
    }

    body->addWidget(right);
    root->addLayout(body, 1);

    {
        auto* row = new QHBoxLayout;
        hintLabel_ = new QLabel(QStringLiteral("在频率图上拖拽框选文字 → 「选区内切分」→ 保存"));
        hintLabel_->setStyleSheet(QStringLiteral("color:#888;"));
        row->addWidget(hintLabel_, 1);

        auto* splitBtn = new QPushButton(QStringLiteral("选区内切分"));
        splitBtn->setMinimumWidth(120);
        connect(splitBtn, &QPushButton::clicked, this, &RegionEditor::doSplit);
        row->addWidget(splitBtn);

        auto* delBtn = new QPushButton(QStringLiteral("删除选中"));
        connect(delBtn, &QPushButton::clicked, this, &RegionEditor::deleteCurrent);
        row->addWidget(delBtn);

        auto* clrBtn = new QPushButton(QStringLiteral("全部清空"));
        connect(clrBtn, &QPushButton::clicked, this, &RegionEditor::clearAll);
        row->addWidget(clrBtn);

        auto* saveBtn = new QPushButton(QStringLiteral("保存模型"));
        saveBtn->setMinimumWidth(110);
        saveBtn->setDefault(true);
        connect(saveBtn, &QPushButton::clicked, this, &RegionEditor::save);
        row->addWidget(saveBtn);

        auto* closeBtn = new QPushButton(QStringLiteral("关闭"));
        connect(closeBtn, &QPushButton::clicked, this, &QDialog::reject);
        row->addWidget(closeBtn);
        root->addLayout(row);
    }
}

void RegionEditor::reload() {
    chardet::Model m;
    if (chardet::loadModel(modelDir_.toStdString(), m)) {
        model_ = std::move(m);
    }
    if (!model_.freq.empty()) {
        refQImg_ = matToQImage(model_.freq);
        canvas_->setImage(refQImg_);
        canvas_->fitToWindow();
        infoLabel_->setText(QStringLiteral("频率图 %1×%2   区域 %3 个   字符 %4 个")
                                .arg(model_.freq.cols)
                                .arg(model_.freq.rows)
                                .arg(model_.regions.size())
                                .arg(model_.totalChars()));
        infoLabel_->setStyleSheet(QString());
    } else {
        infoLabel_->setText(QStringLiteral("⚠ 该视角还没有 chardet 模型，请先训练"));
        infoLabel_->setStyleSheet(QStringLiteral("color:#c62828;"));
    }
    refreshCanvas();
    refreshList();
}

void RegionEditor::refreshCanvas() {
    QList<EditableBox> disp;
    items_.clear();

    for (size_t ri = 0; ri < model_.regions.size(); ++ri) {
        const auto& r = model_.regions[ri];
        EditableBox rb;
        rb.id = static_cast<int>(items_.size()) + 1;
        rb.rect = QRectF(r.rect.x, r.rect.y, r.rect.width, r.rect.height);
        rb.color = regionColor();
        rb.label = QStringLiteral("区%1").arg(ri);
        // 形状和角度要带过来，否则重新打开后斜椭圆又显示成正的，
        // 用户会以为自己画的角度丢了
        if (r.shape == chardet::RegionShape::Ellipse) rb.shape = BoxShape::Ellipse;
        else if (r.shape == chardet::RegionShape::Circle) rb.shape = BoxShape::Circle;
        else rb.shape = BoxShape::Rect;
        rb.angleDeg = r.ell.angleDeg;
        disp.append(rb);
        items_.push_back({Kind::Region, static_cast<int>(ri), -1});

        for (size_t ci = 0; ci < r.chars.size(); ++ci) {
            const auto& c = r.chars[ci];
            EditableBox cb;
            cb.id = static_cast<int>(items_.size()) + 1;
            cb.rect = QRectF(c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height);
            cb.color = charColor();
            cb.label = QString::number(ci);
            disp.append(cb);
            items_.push_back({Kind::CharBox, static_cast<int>(ri), static_cast<int>(ci)});
        }
    }
    canvas_->setBoxes(disp);
}

void RegionEditor::refreshList() {
    list_->clear();
    for (size_t ri = 0; ri < model_.regions.size(); ++ri) {
        const auto& r = model_.regions[ri];
        // 图案区和字符区的判据不同，列表里必须一眼分得清，
        // 否则用户不知道这个框被当成"一排字"还是"一个环"
        QString text;
        if (r.shape == chardet::RegionShape::Ellipse || r.shape == chardet::RegionShape::Circle) {
            // 打离心率：这个数说明椭圆有多扁，是"框和环贴不贴"的第一手依据。
            // e 太大而半径起伏也大，就说明框没画准。
            text = QStringLiteral("◯ 区域%1 %2 中心(%3,%4) 半轴%5x%6 角%7° e=%8")
                       .arg(ri)
                       .arg(r.shape == chardet::RegionShape::Circle ? QStringLiteral("圆")
                                                                    : QStringLiteral("椭圆"))
                       .arg(r.ell.center.x, 0, 'f', 0)
                       .arg(r.ell.center.y, 0, 'f', 0)
                       .arg(r.ell.a, 0, 'f', 1)
                       .arg(r.ell.b, 0, 'f', 1)
                       .arg(r.ell.angleDeg, 0, 'f', 1)
                       .arg(r.ell.eccentricity(), 0, 'f', 3);
        } else {
            text = QStringLiteral("■ 区域%1 矩形 (%2,%3) %4×%5 — %6 字符")
                       .arg(ri)
                       .arg(r.rect.x).arg(r.rect.y)
                       .arg(r.rect.width).arg(r.rect.height)
                       .arg(r.chars.size());
        }
        auto* h = new QListWidgetItem(text);
        h->setForeground(regionColor());
        h->setFlags(Qt::NoItemFlags);
        list_->addItem(h);
    }
    if (list_->count() == 0) {
        list_->addItem(QStringLiteral("（还没有区域，先在图上拖拽框选）"));
    }
}

BoxShape RegionEditor::selectedShape() const {
    if (shapeEllipse_ && shapeEllipse_->isChecked()) return BoxShape::Ellipse;
    if (shapeCircle_ && shapeCircle_->isChecked()) return BoxShape::Circle;
    return BoxShape::Rect;
}

void RegionEditor::onAngleEdited(double deg) {
    // 只对"当前选中的那个图案区"生效。
    // 没选中、或者选中的是字符区，就只是记下这个值给下一次新建用。
    const int sel = canvas_ ? canvas_->selectedId() : -1;
    if (sel <= 0 || sel > static_cast<int>(items_.size())) return;
    const CanvasItem& it = items_[sel - 1];
    if (it.kind != Kind::Region) return;
    if (it.idx < 0 || it.idx >= static_cast<int>(model_.regions.size())) return;
    auto& r = model_.regions[static_cast<size_t>(it.idx)];
    if (!r.isPattern()) return;

    r.ell.angleDeg = static_cast<float>(deg);
    r.syncRectFromEll();
    refreshCanvas();
    refreshList();
    hintLabel_->setText(QStringLiteral("区域%1 长轴角度改为 %2°（离心率 %3）")
                            .arg(it.idx)
                            .arg(deg, 0, 'f', 1)
                            .arg(r.ell.eccentricity(), 0, 'f', 3));
}

void RegionEditor::onBoxCreated(const QRectF& r) {
    if (model_.freq.empty()) return;
    QRectF rel = r.intersected(QRectF(0, 0, model_.freq.cols, model_.freq.rows));
    if (rel.width() < 8 || rel.height() < 8) return;

    const BoxShape shape = selectedShape();

    chardet::RegionInfo ri;
    ri.index = static_cast<int>(model_.regions.size());

    // 图案区（椭圆/圆）
    //
    // 【关键：这里不切字】
    // 椭圆环里虽然也可能有字（SC 中间就有），但那些字是 logo 的一部分，
    // 不是要逐字比对的刻印字符。更重要的：切字会把环切成一堆碎片，
    // 让后面既量不出环的断口，又多出一堆假字符框
    // （v00 原来那个 279x177 的"字符"就是这么来的）。
    // 图案区只记几何形状，字符表留空，判据走 analyzePatternBreak。
    //
    // 拖出来的矩形当作椭圆的外接框：中心 = 矩形中心，
    // 长半轴 = 长边一半，短半轴 = 短边一半，长轴角度 = 0（框是正的）。
    // 斜着的环（比如 SC 长轴 -13°）需要再单独调角度，
    // 这里先按正椭圆建立，角度可以在区域列表上后续修正。
    if (shape != BoxShape::Rect) {
        QRectF box = rel;
        if (shape == BoxShape::Circle) {
            // 圆用正方形外接框，保证是正圆而不是椭圆
            const double d = std::min(box.width(), box.height());
            const QPointF c = box.center();
            box = QRectF(c.x() - d / 2, c.y() - d / 2, d, d);
            box = box.intersected(QRectF(0, 0, model_.freq.cols, model_.freq.rows));
        }
        ri.shape = (shape == BoxShape::Circle) ? chardet::RegionShape::Circle
                                              : chardet::RegionShape::Ellipse;
        ri.ell.center = cv::Point2f(static_cast<float>(box.center().x()),
                                    static_cast<float>(box.center().y()));
        ri.ell.a = static_cast<float>(std::max(box.width(), box.height()) / 2.0);
        ri.ell.b = static_cast<float>(std::min(box.width(), box.height()) / 2.0);
        // 角度取界面上的设置值：拖出来的矩形是正的，但实物上的环多半是斜的
        ri.ell.angleDeg = angleSpin_ ? static_cast<float>(angleSpin_->value()) : 0.0f;
        ri.syncRectFromEll();
        model_.regions.push_back(std::move(ri));
        const auto& back = model_.regions.back();
        hintLabel_->setText(QStringLiteral("已添加%1 中心(%2,%3) 半轴%4x%5 离心率%6")
                                .arg(shape == BoxShape::Circle ? QStringLiteral("圆")
                                                               : QStringLiteral("椭圆"))
                                .arg(back.ell.center.x, 0, 'f', 0)
                                .arg(back.ell.center.y, 0, 'f', 0)
                                .arg(back.ell.a, 0, 'f', 1)
                                .arg(back.ell.b, 0, 'f', 1)
                                .arg(back.ell.eccentricity(), 0, 'f', 3));
        refreshCanvas();
        refreshList();
        return;
    }

    ri.rect = cv::Rect(static_cast<int>(rel.x()), static_cast<int>(rel.y()),
                       static_cast<int>(rel.width()), static_cast<int>(rel.height()));

    // 立即在该 ROI 内切分，让用户马上看到结果
    chardet::Config tuning = model_.config;
    tuning.colThrRatio = colThrSpin_->value() / 100.0;
    tuning.gapMergeRatio = gapRatioSpin_->value() / 100.0;
    const cv::Mat sub = model_.freq(ri.rect);
    const auto units = chardet::split(sub, tuning);
    int ci = 0;
    for (const auto& u : units) {
        chardet::CharBox cb;
        cb.bbox = u.bbox + cv::Point(ri.rect.x, ri.rect.y);
        cb.extend = u.extend + cv::Point(ri.rect.x, ri.rect.y);
        cb.index = ci++;
        ri.chars.push_back(cb);
    }

    model_.regions.push_back(std::move(ri));

    hintLabel_->setText(QStringLiteral("已添加 ROI，切出 %1 个字符（共 %2 个区域）")
                            .arg(model_.regions.back().charCount())
                            .arg(model_.regions.size()));
    refreshCanvas();
    refreshList();
}

void RegionEditor::onSelectionChanged(int id) {
    if (id <= 0 || id > static_cast<int>(items_.size())) return;
    const auto& it = items_[static_cast<size_t>(id - 1)];
    if (it.kind == Kind::Region) {
        const auto& r = model_.regions[static_cast<size_t>(it.idx)];
        if (r.isPattern()) {
            // 图案区：把它的角度和离心率显示出来，并同步到角度输入框。
            // 阻塞信号，否则 setValue 会触发 onAngleEdited 把刚读出来的值又写回去。
            if (angleSpin_) {
                const QSignalBlocker block(angleSpin_);
                angleSpin_->setValue(static_cast<double>(r.ell.angleDeg));
            }
            hintLabel_->setText(QStringLiteral("区域%1 图案区 中心(%2,%3) 半轴%4×%5 角%6° e=%7")
                                    .arg(it.idx)
                                    .arg(r.ell.center.x, 0, 'f', 1)
                                    .arg(r.ell.center.y, 0, 'f', 1)
                                    .arg(r.ell.a, 0, 'f', 2)
                                    .arg(r.ell.b, 0, 'f', 2)
                                    .arg(r.ell.angleDeg, 0, 'f', 1)
                                    .arg(r.ell.eccentricity(), 0, 'f', 3));
        } else {
            hintLabel_->setText(QStringLiteral("区域 %1 (%2,%3) %4×%5")
                                    .arg(it.idx).arg(r.rect.x).arg(r.rect.y)
                                    .arg(r.rect.width).arg(r.rect.height));
        }
    } else {
        const auto& r = model_.regions[static_cast<size_t>(it.idx)];
        if (it.subIdx >= 0 && it.subIdx < static_cast<int>(r.chars.size())) {
            const auto& c = r.chars[static_cast<size_t>(it.subIdx)];
            hintLabel_->setText(QStringLiteral("区域%1 字符#%2 (%3,%4) %5×%6")
                                    .arg(it.idx).arg(it.subIdx)
                                    .arg(c.bbox.x).arg(c.bbox.y)
                                    .arg(c.bbox.width).arg(c.bbox.height));
        }
    }
}

void RegionEditor::onDeleteRequested(int id) {
    if (id <= 0 || id > static_cast<int>(items_.size())) return;
    const auto& it = items_[static_cast<size_t>(id - 1)];
    if (it.kind == Kind::Region) {
        if (it.idx >= 0 && it.idx < static_cast<int>(model_.regions.size())) {
            model_.regions.erase(model_.regions.begin() + it.idx);
        }
    } else {
        auto& r = model_.regions[static_cast<size_t>(it.idx)];
        if (it.subIdx >= 0 && it.subIdx < static_cast<int>(r.chars.size())) {
            r.chars.erase(r.chars.begin() + it.subIdx);
        }
    }
    refreshCanvas();
    refreshList();
}

void RegionEditor::deleteCurrent() {
    const int id = canvas_->selectedId();
    if (id <= 0) {
        hintLabel_->setText(QStringLiteral("请先在图上点选一个框"));
        return;
    }
    onDeleteRequested(id);
}

void RegionEditor::clearAll() {
    if (QMessageBox::question(this, QStringLiteral("确认"),
                              QStringLiteral("清空所有区域？")) != QMessageBox::Yes) {
        return;
    }
    model_.regions.clear();
    refreshCanvas();
    refreshList();
}

void RegionEditor::doSplit() {
    if (model_.regions.empty()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("请先在图上拖拽画选区"));
        return;
    }
    if (model_.freq.empty()) return;

    chardet::Config tuning = model_.config;
    tuning.colThrRatio = colThrSpin_->value() / 100.0;
    tuning.gapMergeRatio = gapRatioSpin_->value() / 100.0;

    int total = 0;
    for (auto& r : model_.regions) {
        const cv::Rect roi = r.rect & cv::Rect(0, 0, model_.freq.cols, model_.freq.rows);
        if (roi.width <= 0 || roi.height <= 0) { r.chars.clear(); continue; }
        const cv::Mat sub = model_.freq(roi);
        const auto units = chardet::split(sub, tuning);
        r.chars.clear();
        int ci = 0;
        for (const auto& u : units) {
            chardet::CharBox cb;
            cb.bbox = u.bbox + cv::Point(roi.x, roi.y);
            cb.extend = u.extend + cv::Point(roi.x, roi.y);
            cb.index = ci++;
            r.chars.push_back(cb);
            ++total;
        }
    }

    refreshCanvas();
    refreshList();
    hintLabel_->setText(QStringLiteral("切分完成：%1 区域 → %2 字符（列阈值 %3%, 间隙 %4%）")
                            .arg(model_.regions.size())
                            .arg(total)
                            .arg(colThrSpin_->value())
                            .arg(gapRatioSpin_->value()));
}

void RegionEditor::save() {
    if (model_.freq.empty()) {
        QMessageBox::warning(this, QStringLiteral("无模型"), QStringLiteral("没有可保存的模型"));
        return;
    }
    if (chardet::saveModel(modelDir_.toStdString(), model_)) {
        QMessageBox::information(this, QStringLiteral("已保存"),
                                 QStringLiteral("已写入 %1\n区域 %2 个，字符 %3 个")
                                     .arg(modelDir_)
                                     .arg(model_.regions.size())
                                     .arg(model_.totalChars()));
    } else {
        QMessageBox::warning(this, QStringLiteral("保存失败"), QStringLiteral("无法写入模型目录"));
    }
}
