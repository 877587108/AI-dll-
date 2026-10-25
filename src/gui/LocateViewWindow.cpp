// LocateViewWindow.cpp —— 定位算法可视化调试平台
#include "LocateViewWindow.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <QApplication>
#include <QBoxLayout>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QWheelEvent>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "chardet.hpp"

namespace locateview {

namespace {

// UTF-8 ↔ QString 的安全转换（Windows 上 toStdString 是 GBK，会乱码）
QString q8(const std::string& s) { return QString::fromUtf8(s.c_str(), static_cast<int>(s.size())); }
std::string u8(const QString& s) {
    const QByteArray b = s.toUtf8();
    return std::string(b.constData(), static_cast<size_t>(b.size()));
}

// 读灰度图：cv::imread 在 Windows 上按 ANSI 解释中文路径会失败，
// 统一用 QFile + imdecode
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

}  // namespace

// ================================================================ ImageHoverWidget

ImageHoverWidget::ImageHoverWidget(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);   // 无按键也能收到 mouseMoveEvent
    setMinimumSize(300, 300);
}

void ImageHoverWidget::setSource(const cv::Mat& img) {
    imgW_ = img.cols;
    imgH_ = img.rows;
    image_ = matToQImage(img);
    zoom_ = 1.0;
    update();
}

void ImageHoverWidget::setZoom(double z) {
    zoom_ = std::clamp(z, 0.02, 40.0);
    update();
}

void ImageHoverWidget::fitToView() {
    if (image_.isNull()) return;
    const double zw = static_cast<double>(width()) / imgW_;
    const double zh = static_cast<double>(height()) / imgH_;
    zoom_ = std::clamp(std::min(zw, zh), 0.02, 40.0);
    update();
}

QPoint ImageHoverWidget::widgetToImage(const QPoint& w) const {
    if (image_.isNull()) return {-1, -1};
    const double dispW = imgW_ * zoom_;
    const double dispH = imgH_ * zoom_;
    const double offX = (width() - dispW) * 0.5;
    const double offY = (height() - dispH) * 0.5;
    const double ix = (w.x() - offX) / zoom_;
    const double iy = (w.y() - offY) / zoom_;
    if (ix < 0 || iy < 0 || ix >= imgW_ || iy >= imgH_) return {-1, -1};
    return {static_cast<int>(ix), static_cast<int>(iy)};
}

void ImageHoverWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(30, 30, 30));
    if (image_.isNull()) {
        p.setPen(QColor(150, 150, 150));
        p.drawText(rect(), Qt::AlignCenter, QStringLiteral("未加载图像"));
        return;
    }

    const double dispW = imgW_ * zoom_;
    const double dispH = imgH_ * zoom_;
    const double offX = (width() - dispW) * 0.5;
    const double offY = (height() - dispH) * 0.5;

    display_ = image_.scaled(static_cast<int>(dispW), static_cast<int>(dispH),
                             Qt::KeepAspectRatio, Qt::SmoothTransformation);
    p.drawImage(static_cast<int>(offX), static_cast<int>(offY), display_);

    drawOverlays(p, offX, offY);
}

void ImageHoverWidget::mouseMoveEvent(QMouseEvent* e) {
    const QPoint im = widgetToImage(e->pos());
    emit mouseMovedImage(im.x(), im.y());
    emit mouseMovedGlobal(mapToGlobal(e->pos()));
    update();
}

void ImageHoverWidget::leaveEvent(QEvent*) {
    emit mouseMovedImage(-1, -1);
    update();
}

void ImageHoverWidget::wheelEvent(QWheelEvent* e) {
    const double factor = e->angleDelta().y() > 0 ? 1.25 : 0.8;
    setZoom(zoom_ * factor);
    e->accept();
}

// ================================================================ DetectImageWidget

DetectImageWidget::DetectImageWidget(QWidget* parent) : ImageHoverWidget(parent) {}

void DetectImageWidget::setLocateResult(const LocateResult* res, int boxCount) {
    res_ = res;
    boxCount_ = boxCount;
    update();
}

void DetectImageWidget::drawOverlays(QPainter& p, double offX, double offY) {
    if (!res_ || !res_->chars || boxCount_ <= 0) return;
    for (int i = 0; i < boxCount_; ++i) {
        const LocatedChar& lc = res_->chars[i];
        const QColor col = lc.accepted ? QColor(0, 220, 0) : QColor(255, 0, 0);
        p.setPen(QPen(col, 1));
        p.setBrush(Qt::NoBrush);
        const QRectF r(offX + lc.x * zoom_, offY + lc.y * zoom_, lc.w * zoom_, lc.h * zoom_);
        p.drawRect(r);
        p.drawText(static_cast<int>(r.left()), static_cast<int>(r.top() - 2),
                   QString::number(lc.index));
    }
}

// ================================================================ TemplateImageWidget

TemplateImageWidget::TemplateImageWidget(QWidget* parent) : ImageHoverWidget(parent) {}

void TemplateImageWidget::setModelBoxes(const std::vector<ModelBox>& boxes) {
    boxes_ = boxes;
    update();
}

void TemplateImageWidget::drawOverlays(QPainter& p, double offX, double offY) {
    for (const auto& b : boxes_) {
        p.setPen(QPen(QColor(0, 220, 0), 1));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(offX + b.x * zoom_, offY + b.y * zoom_, b.w * zoom_, b.h * zoom_));
        p.drawText(static_cast<int>(offX + b.x * zoom_),
                   static_cast<int>(offY + b.y * zoom_ - 2), QString::number(b.index));
    }
}

// ================================================================ PreviewPopup

PreviewPopup::PreviewPopup(QWidget* parent)
    : QWidget(parent, Qt::ToolTip | Qt::FramelessWindowHint) {
    setAttribute(Qt::WA_TranslucentBackground);
    setFixedSize(displaySize_ + 8, displaySize_ + 8);
    hide();
}

void PreviewPopup::updatePreview(const cv::Mat& tmpl, int cx, int cy, int size) {
    size_ = std::max(3, size);
    cellPx_ = displaySize_ / size_;
    if (cellPx_ < 1) cellPx_ = 1;
    const int disp = cellPx_ * size_;

    // 窗口严格等于 size_×size_，中心对齐 cx,cy。
    // 注意：x1 = x0 + size_，不能写 cx + half + 1 —— 那会得到 size_+1 宽，
    // 超出 patch 尺寸导致 copyTo 越界崩溃。
    const int x0 = cx - size_ / 2;
    const int y0 = cy - size_ / 2;
    const int x1 = x0 + size_;
    const int y1 = y0 + size_;

    // 裁模板区域，越界部分填 0
    cv::Mat patch = cv::Mat::zeros(size_, size_, CV_8UC1);
    const int sx = std::max(0, x0), sy = std::max(0, y0);
    const int ex = std::min(tmpl.cols, x1), ey = std::min(tmpl.rows, y1);
    if (ex > sx && ey > sy) {
        tmpl(cv::Rect(sx, sy, ex - sx, ey - sy)).copyTo(
            patch(cv::Rect(sx - x0, sy - y0, ex - sx, ey - sy)));
    }

    preview_ = matToQImage(patch).scaled(disp, disp, Qt::KeepAspectRatio,
                                         Qt::FastTransformation);
    setFixedSize(disp + 8, disp + 8);
    show();
    update();
}

void PreviewPopup::clear() {
    hide();
}

void PreviewPopup::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(0, 0, 0, 220));
    const int ox = 4, oy = 4;
    if (!preview_.isNull()) {
        p.drawImage(ox, oy, preview_);
    }
    // 中心 3×3 提示框
    const int half3 = 3 * cellPx_;
    const int cx = ox + cellPx_ * (size_ / 2);
    const int cy = oy + cellPx_ * (size_ / 2);
    p.setPen(QPen(QColor(255, 60, 60), 1.5));
    p.setBrush(Qt::NoBrush);
    p.drawRect(cx, cy, half3, half3);
    // 外圈
    p.setPen(QPen(QColor(255, 255, 255, 120), 1));
    p.drawRect(ox - 1, oy - 1, cellPx_ * size_ + 1, cellPx_ * size_ + 1);
}

// ================================================================ LocateViewWindow

LocateViewWindow::LocateViewWindow(QWidget* parent) : QWidget(parent) {
    setWindowTitle(QStringLiteral("定位算法调试平台"));
    resize(1500, 950);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(6, 6, 6, 6);
    root->setSpacing(6);

    // ================= 顶部工具栏（横排） =================
    auto* toolbar = new QWidget(this);
    auto* toolbarLay = new QHBoxLayout(toolbar);
    toolbarLay->setContentsMargins(0, 0, 0, 0);

    pluginEdit_ = new QLineEdit(toolbar);
    modelEdit_ = new QLineEdit(toolbar);
    imageEdit_ = new QLineEdit(toolbar);
    auto* loadPluginBtn = new QPushButton(QStringLiteral("插件..."), toolbar);
    auto* loadModelBtn = new QPushButton(QStringLiteral("模型..."), toolbar);
    auto* loadImageBtn = new QPushButton(QStringLiteral("图..."), toolbar);
    runBtn_ = new QPushButton(QStringLiteral("跑定位"), toolbar);
    previewSizeSpin_ = new QSpinBox(toolbar);
    previewSizeSpin_->setRange(5, 200);
    previewSizeSpin_->setValue(30);
    previewSizeSpin_->setSuffix(QStringLiteral(" px"));

    toolbarLay->addWidget(new QLabel(QStringLiteral("插件"), toolbar));
    toolbarLay->addWidget(pluginEdit_, 1);
    toolbarLay->addWidget(loadPluginBtn);
    toolbarLay->addWidget(new QLabel(QStringLiteral("模型"), toolbar));
    toolbarLay->addWidget(modelEdit_, 1);
    toolbarLay->addWidget(loadModelBtn);
    toolbarLay->addWidget(new QLabel(QStringLiteral("图"), toolbar));
    toolbarLay->addWidget(imageEdit_, 1);
    toolbarLay->addWidget(loadImageBtn);
    toolbarLay->addWidget(runBtn_);
    toolbarLay->addWidget(new QLabel(QStringLiteral("预览窗"), toolbar));
    toolbarLay->addWidget(previewSizeSpin_);
    root->addWidget(toolbar);

    // ================= 中间：两图并排 =================
    detectView_ = new DetectImageWidget(this);
    auto* detectScroll = new QScrollArea(this);
    detectScroll->setWidgetResizable(true);
    detectScroll->setWidget(detectView_);

    tmplView_ = new TemplateImageWidget(this);
    tmplScroll_ = new QScrollArea(this);
    tmplScroll_->setWidgetResizable(true);
    tmplScroll_->setWidget(tmplView_);

    // 每个图带一个标题条
    auto* detectBox = new QWidget(this);
    auto* detectBoxLay = new QVBoxLayout(detectBox);
    detectBoxLay->setContentsMargins(0, 0, 0, 0);
    detectBoxLay->setSpacing(2);
    detectBoxLay->addWidget(new QLabel(QStringLiteral("检测图（绿框=定位OK，红框=失败）"), detectBox));
    detectBoxLay->addWidget(detectScroll, 1);

    auto* tmplBox = new QWidget(this);
    auto* tmplBoxLay = new QVBoxLayout(tmplBox);
    tmplBoxLay->setContentsMargins(0, 0, 0, 0);
    tmplBoxLay->setSpacing(2);
    tmplBoxLay->addWidget(new QLabel(QStringLiteral("模板图（字符框）"), tmplBox));
    tmplBoxLay->addWidget(tmplScroll_, 1);

    auto* imageSplitter = new QSplitter(Qt::Horizontal, this);
    imageSplitter->addWidget(detectBox);
    imageSplitter->addWidget(tmplBox);
    imageSplitter->setStretchFactor(0, 3);
    imageSplitter->setStretchFactor(1, 2);
    root->addWidget(imageSplitter, 1);

    // ================= 底部：结果表 + 状态栏 =================
    resultTable_ = new QTableWidget(0, 6, this);
    resultTable_->setHorizontalHeaderLabels(
        {QStringLiteral("序号"), QStringLiteral("定位"), QStringLiteral("置信度"),
         QStringLiteral("x"), QStringLiteral("y"), QStringLiteral("dx,dy")});
    resultTable_->horizontalHeader()->setStretchLastSection(true);
    resultTable_->setMaximumHeight(220);
    root->addWidget(resultTable_);

    statusLabel_ = new QLabel(QStringLiteral("就绪"), this);
    root->addWidget(statusLabel_);

    // 预览浮窗（挂在主窗口下，手动定位）
    popup_ = new PreviewPopup(this);

    // 信号
    connect(loadPluginBtn, &QPushButton::clicked, this, &LocateViewWindow::onLoadPlugin);
    connect(loadModelBtn, &QPushButton::clicked, this, &LocateViewWindow::onLoadModel);
    connect(loadImageBtn, &QPushButton::clicked, this, &LocateViewWindow::onLoadImage);
    connect(runBtn_, &QPushButton::clicked, this, &LocateViewWindow::onRunLocate);
    connect(previewSizeSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &LocateViewWindow::onPreviewSizeChanged);
    connect(detectView_, &DetectImageWidget::mouseMovedImage, this,
            &LocateViewWindow::onDetectMouseMoved);
    connect(detectView_, &DetectImageWidget::mouseMovedGlobal, this,
            &LocateViewWindow::onDetectMouseGlobal);
    connect(tmplView_, &TemplateImageWidget::mouseMovedImage, this,
            &LocateViewWindow::onTemplateMouseMoved);
    connect(tmplView_, &TemplateImageWidget::mouseMovedGlobal, this,
            &LocateViewWindow::onTemplateMouseGlobal);

    // 默认参数
    pluginEdit_->setText(QStringLiteral("build/Release/plugins/locate_ncc.dll"));
    modelEdit_->setText(QStringLiteral("release_defectview/model/v00"));
}

void LocateViewWindow::updateStatus(const QString& msg) {
    statusLabel_->setText(msg);
}

bool LocateViewWindow::loadModelBoxes(const std::string& modelDir, std::string* err) {
    const QString dir = q8(modelDir);

    // 读频率图（模板图），同时得到模型尺寸
    cv::Mat freq;
    if (!readGray(QDir(dir).absoluteFilePath("freq.png"), freq) &&
        !readGray(QDir(dir).absoluteFilePath("ref.png"), freq)) {
        if (err) *err = "读不到 freq.png / ref.png";
        return false;
    }
    modelFreq_ = freq;
    modelW_ = freq.cols;
    modelH_ = freq.rows;

    // 读 regions.txt
    modelBoxes_.clear();
    QFile f(QDir(dir).absoluteFilePath("regions.txt"));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (err) *err = "读不到 regions.txt";
        return false;
    }
    int idx = 0;
    while (!f.atEnd()) {
        const QString line = QString::fromUtf8(f.readLine()).trimmed();
        if (line.isEmpty() || line.startsWith('#')) continue;
        const auto parts = line.split(' ', Qt::SkipEmptyParts);
        if (parts.size() == 1 || parts.size() == 6) continue;   // 区域数 / 区域行
        if (parts.size() >= 9) {
            ModelBox b;
            b.x = parts[0].toInt();
            b.y = parts[1].toInt();
            b.w = parts[2].toInt();
            b.h = parts[3].toInt();
            b.index = idx++;
            if (b.w > 3 && b.h > 3) modelBoxes_.push_back(b);
        }
    }
    if (modelBoxes_.empty()) {
        if (err) *err = "regions.txt 里没有字符框";
        return false;
    }
    return true;
}

void LocateViewWindow::onLoadPlugin() {
    const QString p = QFileDialog::getOpenFileName(this, QStringLiteral("选择插件 DLL"), {},
                                                   QStringLiteral("DLL (*.dll)"));
    if (p.isEmpty()) return;
    pluginEdit_->setText(p);
}

void LocateViewWindow::onLoadModel() {
    const QString d = QFileDialog::getExistingDirectory(this, QStringLiteral("选择模型目录"));
    if (d.isEmpty()) return;
    modelEdit_->setText(d);
    reloadAll();
}

void LocateViewWindow::onLoadImage() {
    const QString p = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择检测图"), {},
        QStringLiteral("图像 (*.tif *.tiff *.bmp *.png *.jpg *.jpeg)"));
    if (p.isEmpty()) return;
    imageEdit_->setText(p);
}

void LocateViewWindow::onPreviewSizeChanged(int n) {
    // 预览窗大小由 previewSizeSpin_ 直接驱动，这里只需触发重绘（无需额外动作）
    (void)n;
}

void LocateViewWindow::reloadAll() {
    // 加载插件
    pluginPath_ = u8(pluginEdit_->text());
    std::string err;
    if (!plugin_.load(pluginPath_, &err)) {
        updateStatus(q8("插件加载失败: " + err));
        return;
    }

    // 加载模型
    modelDir_ = u8(modelEdit_->text());
    if (!loadModelBoxes(modelDir_, &err)) {
        updateStatus(q8("模型加载失败: " + err));
        return;
    }

    // 显示模板图（对比度增强后再显示，否则频率图 94% 像素接近 0，几乎全黑）
    // 注意：只增强显示用的副本，modelFreq_ 保持原始值 —— 预览窗裁的是原始像素，
    // 不能拿增强后的图去裁，否则显示的和实际不一致。
    cv::Mat tmplVis;
    {
        // 取 0~p98 分位做拉伸上限，避免被极少数亮点拉爆对比度
        cv::Mat sorted;
        cv::sort(modelFreq_.reshape(1, 1), sorted, cv::SORT_EVERY_ROW);
        const int total = sorted.cols;
        const double p98 = sorted.at<uchar>(0, static_cast<int>(total * 0.98));
        const double cap = std::max(40.0, p98);
        modelFreq_.convertTo(tmplVis, -1, 255.0 / cap, 0.0);
        cv::threshold(tmplVis, tmplVis, 255.0, 255.0, cv::THRESH_TRUNC);
    }
    tmplView_->setSource(tmplVis);
    tmplView_->setModelBoxes(modelBoxes_);
    tmplView_->fitToView();

    const std::string pname = plugin_.api()->name ? plugin_.api()->name : "?";
    updateStatus(q8("已加载插件 " + pname + "，模型 " + std::to_string(modelW_) + "x" +
                    std::to_string(modelH_) + "，" + std::to_string(modelBoxes_.size()) +
                    " 字符框"));
}

void LocateViewWindow::onRunLocate() {
    reloadAll();
    if (!plugin_.loaded()) return;

    // 读检测图
    const QString ip = imageEdit_->text();
    if (!readGray(ip, detectGray_)) {
        updateStatus(QStringLiteral("检测图读取失败: ") + ip);
        return;
    }

    // 跑定位
    locate_api_t* api = plugin_.api();
    if (api->init && !api->init(modelDir_.c_str())) {
        updateStatus(QStringLiteral("插件 init 失败"));
        return;
    }

    // 笔画信号
    chardet::Config cfg;
    signal_ = chardet::strokeSignal(detectGray_, cv::Rect(0, 0, detectGray_.cols, detectGray_.rows),
                                    cfg, false);

    // 构造 TemplateBox 数组
    std::vector<TemplateBox> tbs;
    tbs.reserve(modelBoxes_.size());
    for (const auto& b : modelBoxes_) {
        TemplateBox tb;
        tb.x = b.x;
        tb.y = b.y;
        tb.w = b.w;
        tb.h = b.h;
        tb.index = b.index;
        tbs.push_back(tb);
    }

    result_ = api->detect(&signal_, detectGray_.cols, detectGray_.rows, modelW_, modelH_,
                          tbs.data(), static_cast<int>(tbs.size()));

    hasResult_ = result_.chars && result_.charCount > 0;
    if (!hasResult_) {
        updateStatus(q8("定位无结果: " + std::string(result_.message ? result_.message : "")));
        return;
    }

    // 记录整图仿射（用于预览映射）
    scaleX_ = result_.scaleX > 0 ? result_.scaleX : 1.0f;
    scaleY_ = result_.scaleY > 0 ? result_.scaleY : 1.0f;
    offsetX_ = result_.offsetX;
    offsetY_ = result_.offsetY;

    // 显示检测图 + 定位框
    detectView_->setSource(detectGray_);
    detectView_->setLocateResult(&result_, result_.charCount);

    updateResultTable();
    updateStatus(q8("定位完成，整图可信=" + std::string(result_.globalOk ? "是" : "否")));
}

void LocateViewWindow::updateResultTable() {
    if (!hasResult_) return;
    resultTable_->setRowCount(result_.charCount);
    for (int i = 0; i < result_.charCount; ++i) {
        const LocatedChar& lc = result_.chars[i];
        // 与模板框（按整图尺度换算的参照位置）的偏差
        double ex = 0, ey = 0;
        if (i < static_cast<int>(modelBoxes_.size())) {
            const ModelBox& mb = modelBoxes_[static_cast<size_t>(i)];
            ex = mb.x * scaleX_;
            ey = mb.y * scaleY_;
        }
        const double dx = lc.x - ex;
        const double dy = lc.y - ey;

        resultTable_->setItem(i, 0, new QTableWidgetItem(QString::number(lc.index)));
        resultTable_->setItem(i, 1,
                              new QTableWidgetItem(lc.accepted ? QStringLiteral("OK")
                                                               : QStringLiteral("FAIL")));
        resultTable_->setItem(i, 2, new QTableWidgetItem(QString::number(lc.confidence, 'f', 3)));
        resultTable_->setItem(i, 3, new QTableWidgetItem(QString::number(lc.x)));
        resultTable_->setItem(i, 4, new QTableWidgetItem(QString::number(lc.y)));
        resultTable_->setItem(
            i, 5,
            new QTableWidgetItem(QStringLiteral("%1,%2").arg(dx, 0, 'f', 1).arg(dy, 0, 'f', 1)));
    }
}

void LocateViewWindow::onDetectMouseMoved(int x, int y) {
    if (x < 0 || y < 0 || modelFreq_.empty()) {
        popup_->clear();
        return;
    }
    // 整图仿射映射：检测图坐标 → 模板图坐标
    const int tx = static_cast<int>(std::lround((x - offsetX_) / scaleX_));
    const int ty = static_cast<int>(std::lround((y - offsetY_) / scaleY_));
    popup_->updatePreview(modelFreq_, tx, ty, previewSizeSpin_->value());
}

void LocateViewWindow::onDetectMouseGlobal(const QPoint& g) {
    // 浮窗定位到鼠标右下侧
    popup_->move(g + QPoint(20, 20));
}

void LocateViewWindow::onTemplateMouseMoved(int x, int y) {
    if (x < 0 || y < 0 || detectGray_.empty()) {
        popup_->clear();
        return;
    }
    // 反向映射：模板图坐标 → 检测图坐标（正向的逆）
    // 正向：检测 = 模板 × scale + offset
    // 反向：检测 = 模板 × scale + offset（直接代入模板坐标即可）
    const int dx = static_cast<int>(std::lround(x * scaleX_ + offsetX_));
    const int dy = static_cast<int>(std::lround(y * scaleY_ + offsetY_));
    popup_->updatePreview(detectGray_, dx, dy, previewSizeSpin_->value());
}

void LocateViewWindow::onTemplateMouseGlobal(const QPoint& g) {
    popup_->move(g + QPoint(20, 20));
}

}  // namespace locateview
