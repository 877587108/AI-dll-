#include "gui/DefectViewWindow.hpp"

#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QElapsedTimer>
#include <QCoreApplication>
#include <QCheckBox>
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
#include <QRegularExpression>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QScrollBar>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QStringList>
#include <QTableWidget>
#include <QTabWidget>
#include <QTextEdit>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "chardet.hpp"
#include "gui/annotate.hpp"
#include "gui/scratch.hpp"

namespace {

QString appDir() { return QCoreApplication::applicationDirPath(); }

QString viewTag(int v) { return QStringLiteral("v%1").arg(v, 2, 10, QLatin1Char('0')); }

/// std::string(UTF-8) → QString
///
/// 批量结果的 name/path/cls 用 std::string 存（--batch 模式也用同一份结构），
/// 两者之间只走 UTF-8，绝不走 toStdString()：Windows 上那是本地编码(GBK)，
/// 遇到中文目录会直接乱码。
QString q8(const std::string& s) { return QString::fromUtf8(s.c_str(), static_cast<int>(s.size())); }
std::string u8(const QString& s) {
    const QByteArray b = s.toUtf8();
    return std::string(b.constData(), static_cast<size_t>(b.size()));
}

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

/// 画带标签的框
///
/// 标签一律经 markloc::asciiLabel 转换（实现在 annotate.hpp）：
/// putText 只认 ASCII，中文字节会被画成问号。
void drawTag(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, const std::string& label,
             int thickness = 2) {
    const cv::Rect rr = r & cv::Rect(0, 0, img.cols, img.rows);
    if (rr.width <= 0 || rr.height <= 0) return;
    cv::rectangle(img, rr, color, thickness);
    const std::string safe = markloc::asciiLabel(label);
    if (safe.empty()) return;

    int baseline = 0;
    const double scale = 0.45;
    const cv::Size ts = cv::getTextSize(safe, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    const int ty = std::max(ts.height + 2, rr.y - 2);
    const cv::Rect bg(rr.x, ty - ts.height - 2, ts.width + 6, ts.height + baseline + 4);
    const cv::Rect bgc = bg & cv::Rect(0, 0, img.cols, img.rows);
    if (bgc.width > 0 && bgc.height > 0) cv::rectangle(img, bgc, color, cv::FILLED);
    cv::putText(img, safe, cv::Point(rr.x + 3, ty), cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}
/// 在框的右侧画一个多行文字面板（OpenCV 只能画 ASCII，所以用编号/数字表达）
///
/// 实现在 annotate.hpp，验证工具用的是同一份代码，避免两处分叉。
using markloc::drawNoteRight;

// moldingArea 跟着划痕判据一起搬到了 gui/scratch.hpp（markloc::moldingArea）。

}  // namespace

// ---------------------------------------------------------------- ImageView

ImageView::ImageView(QWidget* parent) : QLabel(parent) {
    setAlignment(Qt::AlignCenter);
    setMinimumSize(200, 200);
}

void ImageView::setImage(const QImage& img) {
    src_ = img;
    userZoomed_ = false;
    // 新图一律先按视口适配再画。
    //
    // 不能等 resize 事件来触发：视口尺寸没变时根本不会发 resize，
    // 缩放就会永远停在 1:1 —— 三联图 5064 宽塞进 1507 宽的视口，
    // 只能看到最左边一联的一角。
    if (!src_.isNull()) zoom_ = std::clamp(computeFitZoom(), 0.05, 20.0);
    apply();
}

void ImageView::setZoom(double z) {
    // 上下限不能是固定的 0.05/20 这种绝对倍数。
    //
    // 大图适配后的 zoom 可能只有 0.08，用户点三次放大就到 0.15，被 clamp 到
    // 0.05 之上倒无所谓；但反过来，小图适配后 zoom 已是 2.0，上限 20 又太松。
    // 真正要保证的是"相对适配倍率"在一个合理区间：最小到看不清，最大 20 倍。
    const double fit = (src_.isNull() || src_.width() <= 0) ? 1.0 : computeFitZoom();
    const double lo = std::max(0.02, fit / 8.0);
    const double hi = std::max(1.0, fit * 20.0);
    userZoomed_ = true;   // 手动缩放过，resizeEvent 不再抢着 fit
    zoom_ = std::clamp(z, lo, hi);
    apply();
}

void ImageView::fitToWindow() {
    if (src_.isNull()) return;
    // 用滚动区视口的尺寸，不用 parentWidget() 也不用自己。
    //
    // 三个坑，全部踩过：
    //   1. parentWidget() 在布局首次生效前可能是 1×1，缩放被夹到 0.05 下限；
    //   2. 用自己的尺寸会形成反馈环：apply() 里 setFixedSize 改尺寸 ->
    //      触发 resizeEvent -> 又 fit -> 越缩越小；
    //   3. 用整窗高度而非视口高度，会让图片只占上方一小条、下面一大片空白。
    zoom_ = std::clamp(computeFitZoom(), 0.05, 20.0);
    apply();
}

/// 按视口算"整张图刚好放得下"的缩放；拿不到有效视口时返回 1.0
double ImageView::computeFitZoom() const {
    int availW = 0, availH = 0;
    // 注意：scroll_ 在 setWidget 之后会成为本控件的 parentWidget，
    // 但 setWidgetResizable(false) + 布局未收敛时，直接问它可能仍是
    // 0×0 或 1×1。所以除了 parentWidget，还要沿祖先链找真正的 QScrollArea。
    QWidget* w = parentWidget();
    QScrollArea* sa = nullptr;
    for (int depth = 0; w && depth < 4; ++depth, w = w->parentWidget()) {
        if (auto* cand = qobject_cast<QScrollArea*>(w)) {
            sa = cand;
            break;
        }
    }
    if (sa) {
        availW = sa->viewport()->width();
        availH = sa->viewport()->height();
        // 视口尺寸含滚动条/边框，留 20px 余量
        availW -= 20;
        availH -= 20;
    }
    // 视口还没布局好（窗口未显示）时先按 1:1，等 resize 再调整
    if (availW < 50 || availH < 50) return 1.0;
    const double zw = static_cast<double>(availW) / src_.width();
    const double zh = static_cast<double>(availH) / src_.height();
    return std::min(zw, zh);
}

void ImageView::apply() {
    if (src_.isNull()) {
        // 没有图时不要留纯空白：显示一句提示，把"没选图"和"渲染坏了"分开
        clear();
        setText(placeholder_);
        setFixedSize(std::max(200, parentWidget() ? parentWidget()->width() : 400),
                     std::max(200, parentWidget() ? parentWidget()->height() : 400));
        return;
    }
    const int w = std::max(1, static_cast<int>(src_.width() * zoom_));
    const int h = std::max(1, static_cast<int>(src_.height() * zoom_));
    const QPixmap pm = QPixmap::fromImage(src_.scaled(
        w, h, Qt::KeepAspectRatio,
        zoom_ > 1.0 ? Qt::FastTransformation : Qt::SmoothTransformation));
    setPixmap(pm);
    // 内容尺寸 = 图片尺寸。
    //
    // 这里必须 setFixedSize 而不是 resize：滚动区在 widgetResizable(false)
    // 且内容比视口小时，会把 widget 拉伸到视口大小——图片就会被拉变形、
    // 或者与背景混在一起看不出边界。固定尺寸能让滚动区老老实实按图片大小
    // 计算滚动范围，缩放后才有滚动条。
    setFixedSize(pm.size());
}

void ImageView::setPlaceholder(const QString& text) {
    placeholder_ = text;
    if (src_.isNull()) apply();
}

/// 内容尺寸变化：不做任何自动 fit
///
/// apply() 里 setFixedSize 会触发本函数。若在这里再 fit，就是
/// "改尺寸 -> fit -> 又改尺寸" 的反馈环，越缩越小直到下限。
/// 真正的适配时机在滚动区视口变化（见 buildUi 里的 viewport 事件过滤）。
void ImageView::resizeEvent(QResizeEvent* e) {
    QLabel::resizeEvent(e);
}

void ImageView::wheelEvent(QWheelEvent* e) {
    const double step = (e->angleDelta().y() > 0) ? 1.15 : 1.0 / 1.15;
    setZoom(zoom_ * step);
    e->accept();
}

/// 拖动：把图片在滚动区里平移
///
/// 直接改两个滚动条的值，而不是移动控件自身——滚动区已经知道内容多大，
/// 自己算偏移会和 setFixedSize 打架。
void ImageView::panBy(int dx, int dy) {
    QWidget* w = parentWidget();
    QScrollArea* sa = qobject_cast<QScrollArea*>(w);
    for (int depth = 0; !sa && w && depth < 4; ++depth, w = w->parentWidget()) {
        sa = qobject_cast<QScrollArea*>(w);
    }
    if (!sa) return;
    if (auto* hb = sa->horizontalScrollBar()) hb->setValue(hb->value() + dx);
    if (auto* vb = sa->verticalScrollBar()) vb->setValue(vb->value() + dy);
}

void ImageView::mousePressEvent(QMouseEvent* e) {
    // 中键或右键拖动。左键留给 QLabel 自己的文本选择，抢了会反选一片高亮。
    if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) {
        QWidget* w = parentWidget();
        QScrollArea* sa = qobject_cast<QScrollArea*>(w);
        for (int depth = 0; !sa && w && depth < 4; ++depth, w = w->parentWidget()) {
            sa = qobject_cast<QScrollArea*>(w);
        }
        if (sa) {
            panning_ = true;
            panAnchor_ = e->globalPosition().toPoint();
            panScrollAnchor_ = QPoint(sa->horizontalScrollBar()->value(),
                                      sa->verticalScrollBar()->value());
            setCursor(Qt::ClosedHandCursor);
            e->accept();
            return;
        }
    }
    QLabel::mousePressEvent(e);
}

void ImageView::mouseMoveEvent(QMouseEvent* e) {
    if (panning_) {
        // 反向：按住往右拖，画面跟着往右走，所以滚动条往回退
        const QPoint d = e->globalPosition().toPoint() - panAnchor_;
        QWidget* w = parentWidget();
        QScrollArea* sa = qobject_cast<QScrollArea*>(w);
        for (int depth = 0; !sa && w && depth < 4; ++depth, w = w->parentWidget()) {
            sa = qobject_cast<QScrollArea*>(w);
        }
        if (sa) {
            sa->horizontalScrollBar()->setValue(panScrollAnchor_.x() - d.x());
            sa->verticalScrollBar()->setValue(panScrollAnchor_.y() - d.y());
        }
        e->accept();
        return;
    }
    QLabel::mouseMoveEvent(e);
}

void ImageView::mouseReleaseEvent(QMouseEvent* e) {
    if (panning_) {
        panning_ = false;
        unsetCursor();
        e->accept();
        return;
    }
    QLabel::mouseReleaseEvent(e);
}

// ---------------------------------------------------------------- 构造

DefectViewWindow::DefectViewWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("缺陷标注检查器"));
    resize(1500, 950);
    modelDir_ = QDir(appDir()).absoluteFilePath(QStringLiteral("model"));
    buildUi();

    // 启动时自动载入默认视角的模型。
    // 之前漏了这步：界面建好但模型从未载入，用户看到的就是"找不到 chardet 文件"。
    modelEdit_->setText(modelDir_);
    loadModel();
    updateStatus();
}

void DefectViewWindow::buildUi() {
    auto* central = new QWidget;
    auto* root = new QVBoxLayout(central);

    // ---- 模型目录
    {
        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel(QStringLiteral("模型目录：")));
        modelEdit_ = new QLineEdit(modelDir_);
        row->addWidget(modelEdit_, 1);

        auto* b = new QPushButton(QStringLiteral("浏览…"));
        b->setMinimumWidth(76);
        connect(b, &QPushButton::clicked, this, &DefectViewWindow::chooseModelDir);
        row->addWidget(b);

        auto* lb = new QPushButton(QStringLiteral("载入模型"));
        lb->setMinimumWidth(88);
        connect(lb, &QPushButton::clicked, this, [this]() {
            modelDir_ = modelEdit_->text().trimmed();
            loadModel();
        });
        row->addWidget(lb);
        // 视角固定为 v00，不提供切换，放在这里当作一行状态说明。
        auto* viewLbl = new QLabel(QStringLiteral("视角：v00（固定）"));
        viewLbl->setStyleSheet(QStringLiteral("color:#666; padding-left:12px;"));
        viewLbl->setToolTip(QStringLiteral("本项目只在 v00 视角上检字符："
                                           "刻印与 logo 在这个视角成像最稳定，"
                                           "其余视角主要看到引脚与封装侧面。"));
        row->addWidget(viewLbl);
        root->addLayout(row);
    }

    // ---- 视角 + 图片 + 检测
    {
        auto* row = new QHBoxLayout;
        // 固定使用 v00 视角，不再提供切换。
        // 本项目的刻印字符与 logo 在 v00 上可见且成像最稳定，
        // 其余视角主要看到引脚与封装侧面，对字符检测没有贡献。
        //
        // 视角与"产品图"合并到同一行时，1924 窗宽下这一行要塞
        // 2 个 label + 1 个输入框 + 5 个按钮，按钮会被压成省略号。
        // 把视角挪到上一行（模型目录那行尾部），这一行只留输入框和按钮。
        row->addWidget(new QLabel(QStringLiteral("产品图：")));
        imageEdit_ = new QLineEdit;
        imageEdit_->setPlaceholderText(QStringLiteral("选择要检查的图片…"));
        // 输入框是这一行唯一可以伸缩的东西，但没有它按钮又会被拉长，
        // 所以给它一个下限，剩下的空间全给按钮固定宽度。
        imageEdit_->setMinimumWidth(240);
        row->addWidget(imageEdit_, 1);

        auto* ib = new QPushButton(QStringLiteral("选择图片…"));
        ib->setMinimumWidth(104);
        connect(ib, &QPushButton::clicked, this, &DefectViewWindow::chooseImage);
        row->addWidget(ib);

        auto* db = new QPushButton(QStringLiteral("开始检测"));
        db->setMinimumWidth(110);
        db->setDefault(true);
        connect(db, &QPushButton::clicked, this, &DefectViewWindow::runDetect);
        row->addWidget(db);
        auto* sb = new QPushButton(QStringLiteral("灵敏度…"));
        // 不设最小宽度时，一旦路径长到顶格，Qt 会把这几个按钮压到只剩
        // "灵敏度..." 的省略号，客户根本点不清。给足文字宽度。
        sb->setMinimumWidth(104);
        sb->setToolTip(QStringLiteral("调整各检测项的判定门槛"));
        connect(sb, &QPushButton::clicked, this, &DefectViewWindow::openSensitivity);
        row->addWidget(sb);

        batchBtn_ = new QPushButton(QStringLiteral("批量检测…"));
        batchBtn_->setMinimumWidth(112);
        batchBtn_->setToolTip(QStringLiteral("选一个文件夹，逐张检测并输出 CSV 汇总"));
        connect(batchBtn_, &QPushButton::clicked, this, &DefectViewWindow::runBatch);
        row->addWidget(batchBtn_);

        cancelBtn_ = new QPushButton(QStringLiteral("取消"));
        cancelBtn_->setMinimumWidth(72);
        cancelBtn_->setVisible(false);
        connect(cancelBtn_, &QPushButton::clicked, this, &DefectViewWindow::cancelBatch);
        row->addWidget(cancelBtn_);
        root->addLayout(row);
    }

    // ---- 主体：图 + 右侧结果
    auto* body = new QHBoxLayout;

    scroll_ = new QScrollArea;
    // 必须关掉 widgetResizable：开了之后滚动区会强行把内容缩放到容器大小，
    // 图片永远"刚好塞满"，滚动条不会出现，放大后也无法拖动查看细节。
    scroll_->setWidgetResizable(false);
    scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scroll_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    // 背景用中性浅灰而不是 Dark。
    //
    // 三联对比图是 3.4:1 的宽图，视口接近方形时按 min(zw,zh) 适配，
    // 横向必然剩一块。Dark 角色让这块剩白看起来像"没画完的洞"，
    // 实测右半边 740px 全黑，客户第一眼就以为界面坏了。
    // 换成浅灰后它就是一块正常的画布留白；再配合居中对齐，
    // 余量分到左右两边，而不是全堆在右边。
    scroll_->setBackgroundRole(QPalette::Mid);
    imgView_ = new ImageView;
    imgView_->setPlaceholder(
        QStringLiteral("尚未选择产品图\n\n点上方「选择图片…」载入一张 1688×1460 的 v00 图，\n"
                       "再点「开始检测」即可看到标注与对比图。"));
    scroll_->setWidget(imgView_);
    // 视口尺寸一变（窗口缩放、右侧面板变宽）就重新适配。
    //
    // 事件过滤器挂在 viewport 上而不是 ImageView 上：
    // ImageView 自己的 resize 由 setFixedSize 触发，跟着它走会成反馈环。
    scroll_->viewport()->installEventFilter(this);

    // ---- 左列 = 上（图片）+ 下（结果面板），中间可拖
    //
    // 原来图片直接占满左列，而对比图是 3.4:1 的宽图，在近方形的区域里
    // 纵向必然留一大片空白。那块空白现在放两样东西，用 QStackedWidget 切换：
    //   页 0 分数面板：单张检测报出的每条划痕的分值、深度、跨字，
    //                以及"切点调到多少会让它升档/降档"
    //   页 1 批量列表：批量检测时逐项翻看的滑动列表
    // 用 QSplitter 而不是固定高度：客户可以自己拖，图看得多还是数看得多由他定。
    bottomStack_ = new QStackedWidget;
    {
        // 页 0：分数面板
        auto* page = new QWidget;
        auto* pv = new QVBoxLayout(page);
        pv->setContentsMargins(4, 4, 4, 4);
        pv->setSpacing(4);
        scoreTitle_ = new QLabel(QStringLiteral("缺陷分数（检测后显示）"));
        scoreTitle_->setStyleSheet(QStringLiteral("font-weight:bold; color:#333;"));
        pv->addWidget(scoreTitle_);

        // 没有划痕时的空态。
        //
        // 以前这种情况下表格照样铺出来：8 列表头 + 一整片空白行，
        // 占了 240px 面板的绝大部分，看着像"数据没加载出来"。
        // 现在 0 行就把表格收起来，只留一句明确结论。
        scoreEmpty_ = new QLabel;
        scoreEmpty_->setWordWrap(true);
        scoreEmpty_->setStyleSheet(QStringLiteral("color:#35a055; padding:12px;"));
        pv->addWidget(scoreEmpty_, 1);

        scoreTable_ = new QTableWidget;
        // 档位、严重度、深度、跨字、位置、长度、升档切点、降档切点
        scoreTable_->setColumnCount(8);
        scoreTable_->setHorizontalHeaderLabels(
            {QStringLiteral("档位"), QStringLiteral("严重度S"), QStringLiteral("深度"),
             QStringLiteral("跨字"), QStringLiteral("位置"), QStringLiteral("长度"),
             QStringLiteral("升档切点"), QStringLiteral("降档切点")});
        scoreTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        scoreTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
        scoreTable_->setSelectionMode(QAbstractItemView::SingleSelection);
        scoreTable_->verticalHeader()->setVisible(false);
        scoreTable_->verticalHeader()->setDefaultSectionSize(22);
        scoreTable_->setToolTip(
            QStringLiteral("按严重度从高到低排。\n"
                           "「升档切点」= 把对应档位的切点压到这个值，本条升一档；\n"
                           "「降档切点」= 把对应档位的切点抬到这个值，本条降一档。\n"
                           "两个数都能直接填进「灵敏度…」，不用反推。"));
        scoreTable_->horizontalHeader()->setStretchLastSection(false);
        pv->addWidget(scoreTable_, 1);
        // 初始就是空态
        scoreTable_->setVisible(false);
        scoreEmpty_->setText(QStringLiteral("尚未检测。\n选图后点「开始检测」，"
                                            "报出的划痕会在这里逐条列出档位与切点。"));
        bottomStack_->addWidget(page);
    }
    {
        // 页 1：批量滑动列表
        auto* page = new QWidget;
        auto* ph = new QHBoxLayout(page);
        ph->setContentsMargins(4, 4, 4, 4);
        ph->setSpacing(6);

        batchList_ = new QListWidget;
        batchList_->setMinimumWidth(180);
        batchList_->setMaximumWidth(260);
        batchList_->setToolTip(QStringLiteral("点任意一项 → 右侧显示该图的检测结果"));
        connect(batchList_, &QListWidget::currentRowChanged, this,
                [this](int i) { showSlide(i); });
        connect(batchList_, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* it) {
            Q_UNUSED(it);
            // 双击 = 进单张页看大图对比
            if (slideIndex_ >= 0 && slideIndex_ < static_cast<int>(batchRows_.size())) {
                openBatchRow(slideIndex_);
            }
        });
        ph->addWidget(batchList_, 0);

        // 列表下方的整批汇总。
        //
        // 列表只有三五项时下面空掉一大块（实测 3 项列表占满 420px 高度，
        // 下方全是白的）。把整批结论钉在那儿：多少 NG、各档位各几个。
        // 客户跑完批量第一问就是"这批多少 NG"，不该让他去翻 CSV。
        batchFoot_ = new QLabel;
        batchFoot_->setWordWrap(true);
        batchFoot_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        batchFoot_->setStyleSheet(
            QStringLiteral("color:#444; font-size:12px; padding:8px; background:#f6f8fa;"));
        batchFoot_->setVisible(false);

        auto* listCol = new QWidget;
        auto* lc = new QVBoxLayout(listCol);
        lc->setContentsMargins(0, 0, 0, 0);
        lc->setSpacing(4);
        lc->addWidget(batchList_, 1);
        lc->addWidget(batchFoot_, 0);
        ph->addWidget(listCol, 0);

        auto* rightPane = new QWidget;
        auto* rp = new QVBoxLayout(rightPane);
        rp->setContentsMargins(0, 0, 0, 0);
        rp->setSpacing(4);
        slideInfo_ = new QLabel(QStringLiteral("批量检测后在这里逐项查看"));
        slideInfo_->setStyleSheet(QStringLiteral("color:#333;"));
        slideInfo_->setWordWrap(true);
        rp->addWidget(slideInfo_);
        slideView_ = new ImageView;
        slideView_->setPlaceholder(QStringLiteral("批量检测后在这里逐项查看"));
        slideScroll_ = new QScrollArea;
        slideScroll_->setWidgetResizable(false);
        slideScroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        slideScroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        slideScroll_->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
        // 与上面 scroll_ 同理：批量页的对比图也只有 3.4:1，
        // 面板偏方形时右边会剩一大块，Dark 背景会显得像渲染失败。
        //
        // 对齐用居中而不是左上：左对齐时所有余量都堆在右边，实测 x>840
        // 到 1920 整整 1080px 全空，看着像右半边没渲染。居中之后余量
        // 分到两侧，视觉上就是"图放在画布中间"，不再是缺陷。
        slideScroll_->setBackgroundRole(QPalette::Mid);
        slideScroll_->setMinimumHeight(260);
        slideScroll_->setWidget(slideView_);
        slideScroll_->viewport()->installEventFilter(this);
        rp->addWidget(slideScroll_, 1);
        {
            auto* nav = new QHBoxLayout;
            auto* prev = new QPushButton(QStringLiteral("◀ 上一项"));
            connect(prev, &QPushButton::clicked, this, &DefectViewWindow::slidePrev);
            nav->addWidget(prev);
            auto* next = new QPushButton(QStringLiteral("下一项 ▶"));
            connect(next, &QPushButton::clicked, this, &DefectViewWindow::slideNext);
            nav->addWidget(next);

            auto* fitBtn = new QPushButton(QStringLiteral("适配窗口"));
            fitBtn->setToolTip(QStringLiteral("把对比图缩回刚好放得下（也可按 0 键）"));
            connect(fitBtn, &QPushButton::clicked, this, [this]() {
                if (slideView_) slideView_->fitToWindow();
            });
            nav->addWidget(fitBtn);
            auto* zo = new QPushButton(QStringLiteral("放大"));
            zo->setMaximumWidth(56);
            connect(zo, &QPushButton::clicked, this, [this]() {
                if (slideView_) slideView_->setZoom(slideView_->zoom() * 1.25);
            });
            nav->addWidget(zo);
            auto* zi = new QPushButton(QStringLiteral("缩小"));
            zi->setMaximumWidth(56);
            connect(zi, &QPushButton::clicked, this, [this]() {
                if (slideView_) slideView_->setZoom(slideView_->zoom() / 1.25);
            });
            nav->addWidget(zi);
            auto* hint = new QLabel(QStringLiteral("滚轮缩放 · 按住中键拖动"));
            hint->setStyleSheet(QStringLiteral("color:#777;"));
            nav->addWidget(hint);

            nav->addStretch(1);
            auto* openBtn = new QPushButton(QStringLiteral("打开大图对比"));
            connect(openBtn, &QPushButton::clicked, this, [this]() {
                if (slideIndex_ >= 0 && slideIndex_ < static_cast<int>(batchRows_.size())) {
                    openBatchRow(slideIndex_);
                }
            });
            nav->addWidget(openBtn);
            rp->addLayout(nav);
        }
        ph->addWidget(rightPane, 1);
        bottomStack_->addWidget(page);
    }

    auto* leftSplit = new QSplitter(Qt::Vertical);
    leftSplit->addWidget(scroll_);
    leftSplit->addWidget(bottomStack_);
    // 图片占大头：图是看缺陷的，面板是看分数的，别把图挤没了。
    // 两个都不许拖到 0，否则又回到"看不见图"的老问题。
    leftSplit->setStretchFactor(0, 3);
    leftSplit->setStretchFactor(1, 2);
    leftSplit->setChildrenCollapsible(false);
    scroll_->setMinimumHeight(200);
    bottomStack_->setMinimumHeight(140);
    leftSplit->setSizes({620, 240});
    leftSplit_ = leftSplit;
    body->addWidget(leftSplit, 1);

    auto* right = new QWidget;
    right->setMinimumWidth(340);
    right->setMaximumWidth(420);
    auto* rl = new QVBoxLayout(right);
    rl->setContentsMargins(0, 0, 0, 0);

    auto* vbox = new QGroupBox(QStringLiteral("显示"));
    auto* vf = new QFormLayout(vbox);
    {
        auto* vc = new QComboBox;
        vc->addItem(QStringLiteral("标注结果"), 2);
        vc->addItem(QStringLiteral("待检原图"), 1);
        vc->addItem(QStringLiteral("模型模板"), 0);
        vc->setCurrentIndex(0);
        connect(vc, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int i) {
            resultView_ = viewCombo_ ? (3 - 1 - (2 - i)) : 2;   // 与下拉项 data 对齐
            resultView_ = 2 - i;   // 0→标注, 1→原图, 2→模型
        });
        vf->addRow(QStringLiteral("查看"), vc);
    }
    showOkChk_ = new QCheckBox(QStringLiteral("显示正常字符"));
    vf->addRow(showOkChk_);

    charLevelChk_ = new QCheckBox(QStringLiteral("启用字符级判据"));
    charLevelChk_->setChecked(true);
    charLevelChk_->setToolTip(QStringLiteral(
        "检测缺笔画 / 多笔画 / 改笔画。\\n"
        "注意：划痕、脏污落在字符框内时会被误判为缺陷。\\n"
        "实测 84 张 OK 样本中有 33 张因此误报，故默认关闭。"));
    vf->addRow(charLevelChk_);
    {
        auto* zrow = new QHBoxLayout;
        auto* zi = new QPushButton(QStringLiteral("放大"));
        connect(zi, &QPushButton::clicked, this, &DefectViewWindow::zoomIn);
        zrow->addWidget(zi);
        auto* zo = new QPushButton(QStringLiteral("缩小"));
        connect(zo, &QPushButton::clicked, this, &DefectViewWindow::zoomOut);
        zrow->addWidget(zo);
        auto* zf = new QPushButton(QStringLiteral("适应"));
        connect(zf, &QPushButton::clicked, this, &DefectViewWindow::fitView);
        zrow->addWidget(zf);
        vf->addRow(zrow);
    }
    {
        auto* zrow = new QHBoxLayout;
        zrow->addWidget(new QLabel(QStringLiteral("缩放")));
        zoomSpin_ = new QSpinBox;
        zoomSpin_->setRange(5, 400);
        zoomSpin_->setValue(100);
        zoomSpin_->setSuffix(QStringLiteral(" %"));
        connect(zoomSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [this](int v) { imgView_->setZoom(v / 100.0); });
        zrow->addWidget(zoomSpin_, 1);
        vf->addRow(zrow);
    }
    rl->addWidget(vbox);

    rl->addWidget(new QLabel(QStringLiteral("检出项")));
    list_ = new QListWidget;
    connect(list_, &QListWidget::itemClicked, this, [this](QListWidgetItem* it) {
        Q_UNUSED(it);
        resultView_ = 0;   // 点列表自动切到标注视图需同步下拉，这里只切内部状态
    });
    tabs_ = new QTabWidget;

    // 页 1：当前图片的检出项
    list_ = new QListWidget;
    connect(list_, &QListWidget::itemClicked, this, [this](QListWidgetItem* it) {
        Q_UNUSED(it);
        resultView_ = 0;
    });
    tabs_->addTab(list_, QStringLiteral("检出项"));

    // 页 2：批量检测结果表
    {
        auto* page = new QWidget;
        auto* pv = new QVBoxLayout(page);
        pv->setContentsMargins(0, 0, 0, 0);
        pv->setSpacing(4);

        // 分类筛选条：往左收窄，右边显示计数
        auto* frow = new QHBoxLayout;
        frow->addWidget(new QLabel(QStringLiteral("分类")));
        batchFilter_ = new QComboBox;
        batchFilter_->addItem(QStringLiteral("全部"), QString());
        // 固定六档，顺序与填表一致
        for (const char* c : {"确定", "明显", "可疑", "对齐异常", "旁证", "OK"}) {
            const QString s = QString::fromUtf8(c);
            batchFilter_->addItem(s, s);
        }
        connect(batchFilter_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this](int) { applyBatchFilter(); });
        frow->addWidget(batchFilter_);
        batchSummary_ = new QLabel;
        batchSummary_->setStyleSheet(QStringLiteral("color:#555;"));
        frow->addWidget(batchSummary_, 1);
        pv->addLayout(frow);

        batchTable_ = new QTableWidget;
        // 分类、文件、划痕严重度、跨字数、深度、对齐位移、旁证三项、耗时、未报原因
        batchTable_->setColumnCount(11);
        batchTable_->setHorizontalHeaderLabels(
            {QStringLiteral("分类"), QStringLiteral("文件"), QStringLiteral("划痕S"),
             QStringLiteral("跨字"), QStringLiteral("深度"), QStringLiteral("对齐位移"),
             QStringLiteral("缺字"), QStringLiteral("缺墨"), QStringLiteral("多墨"),
             QStringLiteral("ms"), QStringLiteral("未报原因")});
        batchTable_->setToolTip(
            QStringLiteral("「未报原因」写的是这一张卡在哪道门槛上。\n"
                           "空白 = 已报出划痕。"));
        batchTable_->horizontalHeader()->setStretchLastSection(true);
        batchTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        batchTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
        batchTable_->setSelectionMode(QAbstractItemView::SingleSelection);
        batchTable_->verticalHeader()->setVisible(false);
        batchTable_->verticalHeader()->setDefaultSectionSize(20);
        batchTable_->setToolTip(QStringLiteral("双击任意一行 → 载入该图并显示对比图"));
        // 双击或回车都打开：鼠标顺手，键盘也能用
        connect(batchTable_, &QTableWidget::cellDoubleClicked, this,
                [this](int row, int) { openBatchRow(row); });
        connect(batchTable_, &QTableWidget::cellActivated, this,
                [this](int row, int) { openBatchRow(row); });
        pv->addWidget(batchTable_, 1);

        tabs_->addTab(page, QStringLiteral("批量结果"));
    }

    // 页 3：参数与性能
    statsView_ = new QTextEdit;
    statsView_->setReadOnly(true);
    statsView_->setStyleSheet(QStringLiteral("font-family:Consolas,monospace; font-size:12px;"));
    tabs_->addTab(statsView_, QStringLiteral("参数/性能"));

    rl->addWidget(tabs_, 1);

    // 左侧图 + 右侧结果，作为主体塞进根布局。
    //
    // 这一步原来漏了：body 建好了、scroll_ 和 right 都加进去了，
    // 但 body 本身从没 addLayout 到 root —— 于是整个图片区和右侧面板
    // 都不在窗口里，界面只剩上面两行输入框和底栏，看起来就是"完全空白"。
    // 布局是右值引用语义，漏一次就是静悄悄地整块不显示，编译器不报错。
    root->addLayout(body, 1);

    // 批量检测进度条（平时隐藏）
    progressBar_ = new QProgressBar;
    progressBar_->setTextVisible(true);
    progressBar_->setMinimumHeight(20);
    progressBar_->setVisible(false);
    root->addWidget(progressBar_);

    status_ = new QLabel(QStringLiteral("就绪"));
    status_->setStyleSheet(QStringLiteral("color:#666;"));
    root->addWidget(status_);

    setCentralWidget(central);
}

// ---------------------------------------------------------------- 槽

void DefectViewWindow::chooseModelDir() {
    const QString d = QFileDialog::getExistingDirectory(this, QStringLiteral("选择模型目录"),
                                                        modelDir_);
    if (d.isEmpty()) return;
    modelDir_ = d;
    modelEdit_->setText(d);
    loadModel();
}

void DefectViewWindow::chooseImage() {
    const QString start = QDir(appDir()).absoluteFilePath(QStringLiteral(".."));
    const QString p = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择产品图"), start,
        QStringLiteral("图像 (*.tif *.tiff *.png *.bmp *.jpg);;所有文件 (*.*)"));
    if (p.isEmpty()) return;
    imageEdit_->setText(p);
    imagePath_ = p;

    if (!readGray(p, gray_)) {
        QMessageBox::warning(this, QStringLiteral("读取失败"), QStringLiteral("无法读取该图片。"));
        return;
    }
    cv::cvtColor(gray_, rawBgr_, cv::COLOR_GRAY2BGR);
    resultBgr_ = rawBgr_.clone();
    showImage(resultBgr_);
    updateStatus();
}

void DefectViewWindow::onViewChanged(int) {
    loadModel();
}

void DefectViewWindow::loadModel() {
    constexpr int v = 0;   // 固定 v00
    const QString dir = QDir(modelDir_).absoluteFilePath(viewTag(v));

    model_ = chardet::Model{};
    units_.clear();
    modelReady_ = false;

    // 用 UTF-8 转换：toStdString() 在 Windows 上走本地编码（GBK），
    // 而 chardet 内部用 std::filesystem 拼接路径，遇到中文目录名会失败。
    if (!chardet::loadModel(dir.toUtf8().toStdString(), model_)) {
        // 明确指出缺什么文件，而不是笼统说"找不到"
        QStringList missing;
        const QStringList need = {QStringLiteral("freq.png"), QStringLiteral("regions.txt")};
        for (const QString& f : need) {
            if (!QFile::exists(QDir(dir).absoluteFilePath(f))) missing << f;
        }
        QString msg = QStringLiteral("%1：模型加载失败 — %2").arg(viewTag(v), dir);
        if (missing.isEmpty()) {
            msg += QStringLiteral("（文件存在但无法解析，可能已损坏）");
        } else {
            msg += QStringLiteral("  缺少：%1").arg(missing.join(QStringLiteral(", ")));
        }
        status_->setText(msg);
        status_->setToolTip(dir);
        status_->setStyleSheet(QStringLiteral("color:#c62828;"));
        return;
    }
    for (size_t ri = 0; ri < model_.regions.size(); ++ri) {
        auto u = chardet::toUnits(model_, static_cast<int>(ri));
        for (auto& x : u) units_.push_back(std::move(x));
    }
    modelReady_ = !units_.empty();
    status_->setStyleSheet(QStringLiteral("color:#333;"));
    status_->setText(QStringLiteral("%1：模型 %2 个字符已载入")
                         .arg(viewTag(v))
                         .arg(units_.size()));
}

/// 把界面上的灵敏度设置原样搬进划痕参数
///
/// 一一对应，不做任何换算或兜底。参数该是多少就是多少——
/// 这里的职责只是"抄过去"，任何自动修正都是偷偷改客户的标定。
markloc::ScratchParams DefectViewWindow::scratchParams() const {
    markloc::ScratchParams p;
    p.enabled = sens_.checkScratch;
    p.minLen = sens_.scratchMinLen;
    p.minCov = sens_.scratchMinCov;
    p.bridge = sens_.scratchBridge;
    p.minEnergy = sens_.scratchMinEnergy;
    p.minComposite = sens_.scratchMinComposite;
    p.cutCertain = sens_.scratchCutCertain;
    p.cutObvious = sens_.scratchCutObvious;
    p.cutSuspect = sens_.scratchCutSuspect;
    p.depthReach = sens_.scratchDepthReach;
    return p;
}

// ---- 自检接口：把"界面到底渲染了没有"变成可量的数
//
// 为什么需要：上一次"完全空白"是因为 buildUi() 建了 body 布局却没 addLayout，
// 编译器不报错、控件也确实 new 出来了，只是几何区域是 0×0。
// 所以判据不能是"控件存不存在"，必须是"尺寸是不是正的"。

/// 设模型目录并载入（自检用，跳过选目录对话框）
bool DefectViewWindow::loadModelForTest(const QString& dir) {
    modelDir_ = dir;
    if (modelEdit_) modelEdit_->setText(dir);
    loadModel();
    return modelReady_;
}

/// 视口尺寸变了就重新适配图片缩放
///
/// 挂在滚动区 viewport 上，因为那才是"图片实际可见的区域"。
/// ImageView 自己的 resize 不能用（setFixedSize 会触发它形成反馈环）。
bool DefectViewWindow::eventFilter(QObject* obj, QEvent* ev) {
    if (scroll_ && obj == scroll_->viewport() && ev->type() == QEvent::Resize) {
        if (imgView_ && !imgView_->imageIsEmpty() && !imgView_->manualZoom()) {
            imgView_->fitToWindow();
        }
    }
    // 滑动列表的对比图同理：面板被分割条拖动改变大小时要重新适配，
    // 否则图会停在旧尺寸、在视口里偏到一边
    if (slideScroll_ && obj == slideScroll_->viewport() && ev->type() == QEvent::Resize) {
        if (slideView_ && !slideView_->imageIsEmpty() && !slideView_->manualZoom()) {
            slideView_->fitToWindow();
        }
    }
    return QMainWindow::eventFilter(obj, ev);
}

QSize DefectViewWindow::findImageViewSize() const {
    if (!imgView_ || imgView_->isHidden()) return QSize(0, 0);
    return imgView_->size();
}

void DefectViewWindow::injectBatchRowsForTest(const std::vector<BatchRow>& rows) {
    batchRows_ = rows;
    fillBatchTable();
    fillBatchList();
    if (bottomStack_) bottomStack_->setCurrentIndex(1);
}

int DefectViewWindow::bottomPageForTest() const {
    return bottomStack_ ? bottomStack_->currentIndex() : -1;
}

int DefectViewWindow::batchListCountForTest() const {
    return batchList_ ? batchList_->count() : 0;
}

QImage DefectViewWindow::grabSlidePixmapForTest() const {
    return currentSlideImage_;
}

QString DefectViewWindow::batchListTextForTest(int i) const {
    if (!batchList_ || i < 0 || i >= batchList_->count()) return QString();
    QListWidgetItem* it = batchList_->item(i);
    return it ? it->text() : QString();
}

QSize DefectViewWindow::canvasSizeForTest() const {
    // 量的是 ImageView 的固定尺寸——它等于"图片缩放后的像素大小"，
    // 也就是滚动区的滚动范围。缩放是否真的生效，看这个数最直接。
    if (!slideView_) return QSize(0, 0);
    return slideView_->size();
}

QImage DefectViewWindow::slideViewImageForTest() const {
    if (!slideView_) return QImage();
    const QPixmap pm = slideView_->pixmap();
    return pm.isNull() ? QImage() : pm.toImage();
}

/// 缩放/拖动链路自检：量缩放前后画布尺寸、拖动前后滚动条位置、复位是否回到适配
bool DefectViewWindow::zoomPanForTest() {
    if (!slideView_ || !slideScroll_) return false;
    if (slideView_->imageIsEmpty()) return false;

    const QSize fitSize = slideView_->size();
    const double fitZoom = slideView_->zoom();
    if (fitSize.width() <= 0) return false;

    // 1) 放大：画布必须变大，且出现滚动条（否则"放大了但看不到别处"）
    //
    // 这里不插 processEvents：待处理的 resize 事件会让视口重新布局，
    // 若它恰好落在 fit 路径上，会把刚设的缩放冲掉。那是另一个 bug，
    // 不该被本测试的时序掩盖——先量 setZoom 的直接效果。
    slideView_->setZoom(fitZoom * 6.0);
    const QSize bigSize = slideView_->size();
    if (bigSize.width() <= fitSize.width()) return false;
    QScrollBar* hb = slideScroll_->horizontalScrollBar();
    QScrollBar* vb = slideScroll_->verticalScrollBar();
    const bool scrollable = hb->maximum() > 0 || vb->maximum() > 0;
    if (!scrollable) return false;

    // 2) 拖动：把滚动条挪到中间，量值确实变了
    hb->setValue(hb->maximum() / 2);
    vb->setValue(vb->maximum() / 2);
    QApplication::processEvents();
    const bool panned = hb->value() > 0 || vb->value() > 0;
    if (!panned) return false;

    // 3) 复位：回到适配缩放，画布必须明显小于放大态
    //
    // 这里不能拿"和最初完全一样"当判据：拖动后滚动条位置变了，视口可用
    // 范围跟着变，重新适配得到的倍率本来就可能差一点点（404 vs 390）。
    // 真正要保证的是"复位确实把图缩回去了"，不是像素级还原。
    slideView_->fitToWindow();
    QApplication::processEvents();
    const QSize backSize = slideView_->size();
    const bool restored = backSize.width() < bigSize.width() / 2;
    return restored;
}

QSize DefectViewWindow::slideViewSizeForTest() const {
    if (!slideScroll_ || slideScroll_->isHidden()) return QSize(0, 0);
    if (slideView_ && !slideView_->pixmap().isNull()) return slideView_->pixmap().size();
    return slideScroll_->viewport()->size();
}

QString DefectViewWindow::statusTextForTest() const {
    return status_ ? status_->text() : QString();
}

QSize DefectViewWindow::viewportSizeForTest() const {
    if (!scroll_) return QSize(0, 0);
    return scroll_->viewport()->size();
}

QPoint DefectViewWindow::findImageViewGlobalPos() const {
    if (!imgView_) return QPoint(0, 0);
    return imgView_->mapToGlobal(QPoint(0, 0));
}

QImage DefectViewWindow::grabImageView() const {
    if (!imgView_ || imgView_->isHidden() || imgView_->size().isEmpty()) return QImage();
    return imgView_->grab().toImage();
}

QImage DefectViewWindow::resultImageForTest() const {
    if (resultBgr_.empty()) return QImage();
    // copy：切断对 cv::Mat 内存的引用，避免 Mat 释放后 QImage 悬空
    return matToQImage(resultBgr_).copy();
}

/// 读入一张图，供批量结果双击跳转用
bool DefectViewWindow::loadImageFile(const QString& path) {
    if (!readGray(path, gray_)) return false;
    cv::cvtColor(gray_, rawBgr_, cv::COLOR_GRAY2BGR);
    resultBgr_ = rawBgr_.clone();
    return true;
}

void DefectViewWindow::runDetect() {
    if (!modelReady_) {
        QMessageBox::warning(this, QStringLiteral("未载入模型"),
                             QStringLiteral("请先选择模型目录并载入。"));
        return;
    }
    if (gray_.empty()) {
        QMessageBox::information(this, QStringLiteral("提示"),
                                 QStringLiteral("请先选择一张产品图。"));
        return;
    }
    if (gray_.size() != model_.freq.size()) {
        QMessageBox::warning(
            this, QStringLiteral("尺寸不符"),
            QStringLiteral("产品图 %1×%2 与模型 %3×%4 尺寸不同。")
                .arg(gray_.cols).arg(gray_.rows).arg(model_.freq.cols).arg(model_.freq.rows));
        return;
    }

    // 检测范围：严格限制在字符框内
    //
    // 溢料、污损、划痕都不检查字符框之外的区域。
    // 理由：框外属于背景（引脚、封装边、基板），那里的痕迹与字符质量无关，
    // 而且会把引脚的高亮、封装的划痕误报成缺陷。

    // 判断矩形是否完全落在某个字符框内
    const auto insideAnyChar = [this](const cv::Rect& r) -> bool {
        for (const auto& u : units_) {
            const cv::Rect b = u.bbox;
            if (r.x >= b.x && r.y >= b.y && r.x + r.width <= b.x + b.width &&
                r.y + r.height <= b.y + b.height) {
                return true;
            }
        }
        return false;
    };

    const chardet::Config cfg = model_.config;

    // 每轮从模型重建，禁止上次检测的位移累积。
    if (model_.hasRegions()) units_ = chardet::toAllUnits(model_);
    cv::Mat alignedBgr = rawBgr_.clone();
    // 配准位移每轮清零：不清的话，这一帧若没触发配准，
    // 图案区会沿用上一张图的位移，位置整个错掉。
    lastAlignX_ = 0.0;
    lastAlignY_ = 0.0;
    // 笔画信号 + 配准
    signal_ = chardet::strokeSignal(gray_, cv::Rect(0, 0, gray_.cols, gray_.rows), cfg, false);
    {
        cv::Mat r32, s32;
        model_.ref.convertTo(r32, CV_32F);
        signal_.convertTo(s32, CV_32F);
        // 只用字符区域做相位相关，不用整幅图。
        //
        // 整幅图里引脚、封装边缘的高对比结构远强于刻印字符，
        // 相位相关会被它们主导，给出错误的平移量。
        // 实测三张误报件的对比：
        //   0007_RT  全图 (+0.1,+2.3)  仅字符区 (+8.0,+4.1)
        //   0014_RT  全图 (+2.5,+2.2)  仅字符区 (+1.4,+18.0)
        //   0030_RT  全图 (+1.8,+2.2)  仅字符区 (+0.4,+14.4)
        // 差异达 8~16px，足以让所有字符框整体错位、全部误报缺墨。
        // 正常件两种方式结果一致（差 <0.5px），说明字符区配准是可靠的。
        cv::Rect inkArea;
        for (const auto& u : units_) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, signal_.cols, signal_.rows);
            if (b.width <= 0 || b.height <= 0) continue;
            inkArea = inkArea.area() > 0 ? (inkArea | b) : b;
        }
        if (inkArea.area() > 0) {
            inkArea.x = std::max(0, inkArea.x - 20);
            inkArea.y = std::max(0, inkArea.y - 20);
            inkArea.width = std::min(signal_.cols - inkArea.x, inkArea.width + 40);
            inkArea.height = std::min(signal_.rows - inkArea.y, inkArea.height + 40);

            cv::Mat rC, sC;
            r32(inkArea).copyTo(rC);
            s32(inkArea).copyTo(sC);
            const cv::Point2d sh = cv::phaseCorrelate(sC, rC, cv::noArray());
            // 60px 这个门限是实测出来的失效边界，不能随便放宽。
            //
            // 用一张合格件整幅平移后重测，位移 <= 56px 时零误报，
            // 到 60px 相位相关拒绝修正，立刻崩成 50 条误报（S=3260）。
            // 把门限放宽到 200 只多撑到 68px，再往后逐字匹配的 ±12px
            // 搜索窗接不住，残差一路涨到 200px 以上。
            //
            // 曾试过用垂直/水平投影替换这里来扩范围，结论是不行：
            // 在 84 张真实合格件上直接对比框墨覆盖率，
            //   不对齐 89.9% | 相位相关 95.9% | 投影对齐 83.3%
            //   胜场：相位 81、投影 0、持平 3
            // 投影比不对齐还差，原因是参照帧是训练集平均帧，行结构本身
            // 就是糊的，压到 1D 再把各行混在一起，位置信息进一步丢失。
            // 分析工具见 tools/aligncmp.cpp（markloc_aligncmp）。
            if (std::abs(sh.x) < 60 && std::abs(sh.y) < 60 &&
                (std::abs(sh.x) > 0.01 || std::abs(sh.y) > 0.01)) {
                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -sh.x, 0, 1, -sh.y);
                // 源与目标不能是同一个 Mat（未定义行为），先算到临时块再换手
                cv::Mat aligned;
                cv::warpAffine(signal_, aligned, M, signal_.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                signal_ = aligned;
                cv::Mat alignedView;
                cv::warpAffine(alignedBgr, alignedView, M, alignedBgr.size(),
                               cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
                alignedBgr = alignedView;
                // 记下这次配准挪了多少，图案区要跟着挪。
                // 存的原始 sh；用的时候取负（见下方图案区那段说明）。
                lastAlignX_ = sh.x;
                lastAlignY_ = sh.y;
            }
        }
    }


    // 与批量入口共用定位与重采样；框和模型始终处于同一坐标系。
    if (sens_.perCharAlign) {
        const auto locations = markloc::locateCharacters(signal_, model_.freq, units_);
        signal_ = markloc::alignCharacterSignal(signal_, locations, &alignedBgr);
    }

    hits_.clear();
    scratchHits_.clear();
    cv::Mat vis = alignedBgr.clone();

    const bool showOk = showOkChk_ && showOkChk_->isChecked();

    for (const auto& u : units_) {
        // 字符级判据（勾选"启用字符级判据"才跑）
        //
        // 默认关闭：实测 84 张 OK 有 39.3% 被它误报，
        // 因为划痕/脏污落在字符框内时无法与笔画缺陷区分。
        // 默认只跑整行墨块判据（84 张 OK 零误报）。
        if (!sens_.checkCharLevel) { continue; }
        const auto pc = chardet::diffNcc(signal_, model_.freq, u, cfg);
        if (!pc.valid) continue;

        const bool isMiss = pc.missScore > 0.0;
        const bool isExtra = pc.extraScore > 0.0;

        if (!isMiss && !isExtra) {
            if (showOk) drawTag(vis, u.bbox, cv::Scalar(80, 200, 80), "", 1);
            continue;
        }

        char lbl[96];
        if (isMiss) {
            std::snprintf(lbl, sizeof(lbl), "缺墨 %.0f%%", pc.missScore * 100);
            bool drew = false;
            for (const auto& sx : pc.missSpanX) {
                drawTag(vis, cv::Rect(u.bbox.x + sx.start, u.bbox.y, sx.end - sx.start,
                                      u.bbox.height),
                        cv::Scalar(60, 60, 230), drew ? "" : lbl);
                drew = true;
            }
            for (const auto& sy : pc.missSpanY) {
                drawTag(vis, cv::Rect(u.bbox.x, u.bbox.y + sy.start, u.bbox.width,
                                      sy.end - sy.start),
                        cv::Scalar(60, 60, 230), drew ? "" : lbl);
                drew = true;
            }
            if (!drew) drawTag(vis, u.bbox, cv::Scalar(60, 60, 230), lbl);

            Hit h;
            h.kind = QStringLiteral("缺墨");
            h.score = pc.missScore;
            h.detail = QStringLiteral("缺墨 %.0f%%").arg(pc.missScore * 100);
            h.where = QStringLiteral("(%1,%2) %3×%4")
                          .arg(u.bbox.x).arg(u.bbox.y).arg(u.bbox.width).arg(u.bbox.height);
            hits_.push_back(h);
        }
        if (isExtra) {
            std::snprintf(lbl, sizeof(lbl), "多墨 %.0f%%", pc.extraScore * 100);
            bool drew = false;
            for (const auto& sx : pc.extraSpanX) {
                drawTag(vis, cv::Rect(u.bbox.x + sx.start, u.bbox.y, sx.end - sx.start,
                                      u.bbox.height),
                        cv::Scalar(230, 150, 60), drew ? "" : lbl);
                drew = true;
            }
            if (!drew) drawTag(vis, u.bbox, cv::Scalar(230, 150, 60), lbl);

            Hit h;
            h.kind = QStringLiteral("多墨");
            h.score = pc.extraScore;
            h.detail = QStringLiteral("多墨 %.0f%%").arg(pc.extraScore * 100);
            h.where = QStringLiteral("(%1,%2) %3×%4")
                          .arg(u.bbox.x).arg(u.bbox.y).arg(u.bbox.width).arg(u.bbox.height);
            hits_.push_back(h);
        }
    }

    // 整字数量校验（仅逐字符，不扫描框外）
    //
    // 按要求不检测字符框外区域，因此不再扫描"字符区间之外的新墨块"。
    // 保留的是逐字符区间核对：某个字符位置完全没有墨 → 少字。
    // 注意：这只能发现"少字"，无法发现"行尾多一个字"（那属于框外检测）。
    int rowExtra = 0, rowMissing = 0;
    const auto bd = chardet::diffBlocks(signal_, model_.freq, units_, cfg, 8);
    if (bd.valid && sens_.checkRowCount) {
        for (const auto& rc : bd.rows) {
            rowMissing += rc.missingChars;
            if (rc.missingChars == 0) continue;

            Hit h;
            h.kind = QStringLiteral("整字数");
            h.score = 1.5;
            h.detail = QStringLiteral("本应%1字 实有%2字  缺%3字")
                           .arg(rc.expected)
                           .arg(rc.found)
                           .arg(rc.missingChars);
            h.where = QStringLiteral("行 y=%1..%2").arg(rc.y0).arg(rc.y1);
            hits_.push_back(h);
        }
    }
    // ---- 划痕检测（客户确认的唯一缺陷类型）
    //
    // 实现搬到 src/gui/scratch.hpp。原因是批量检测要跑同一份判据：
    // 原来这段只存在于 runDetect()，runBatch() 压根没调，于是批量页把
    // 真划痕件全判成 OK，与单张页自相矛盾。抽成函数后两边共用，
    // 参数照抄界面上的灵敏度设置，数值一个不动。
    if (sens_.checkScratch) {
        const markloc::ScratchParams sp = scratchParams();
        const auto scratches = markloc::findScratches(signal_, model_.freq, units_, cfg, sp);
        // 留一份结构化结果给下方的分数面板：面板要显示每条划痕的
        // 深度/跨字/严重度，以及"切点调到多少能让它升档/降档"。
        // hits_ 里只有拼好的字符串，反解不可靠，所以这里单独存一遍。
        scratchHits_ = scratches;
        for (size_t si = 0; si < scratches.size(); ++si) {
            const auto& sh = scratches[si];
            markloc::drawScratch(vis, sh, sp, static_cast<int>(si));

            Hit h;
            h.kind = QStringLiteral("划痕·") + QString::fromUtf8(markloc::bandName(sh.band));
            h.score = sh.composite;
            h.detail = QStringLiteral("长%1px 深%2 宽%3 跨%4字 严重度%5")
                           .arg(static_cast<int>(sh.len))
                           .arg(QString::number(sh.depth, 'f', 0))
                           .arg(QString::number(sh.widthPx, 'f', 1))
                           .arg(sh.charsHit)
                           .arg(QString::number(sh.composite, 'f', 0));
            h.where = QStringLiteral("(%1,%2)->(%3,%4)")
                          .arg(sh.a1.x)
                          .arg(sh.a1.y)
                          .arg(sh.a2.x)
                          .arg(sh.a2.y);
            hits_.push_back(h);
        }
        // 所有划痕的说明合并成一块图例，画在图片空白处。
        //
        // 不能混在上面那个循环里画：每条各占一块白底面板，多一条就多一块，
        // 在字符密集处连成一片白斑，把划痕本身盖住——用户看到的
        // "许多意义不明的框"就是这些叠在一起的说明面板。
        if (!scratches.empty()) {
            std::vector<std::pair<std::vector<std::string>, cv::Scalar>> legend;
            legend.reserve(scratches.size());
            for (size_t si = 0; si < scratches.size(); ++si) {
                legend.push_back({markloc::scratchLegendLines(scratches[si], sp,
                                                              static_cast<int>(si)),
                                  markloc::bandColor(std::clamp(scratches[si].band, 0, 3))});
            }
            markloc::drawLegendPanel(vis, legend);
        }
    }

    // ---- 图案区断点检测（logo 椭圆环有没有被切断）
    //
    // 与划痕并列的第二套判据，不是它的变体：
    //   划痕问"字的墨量对不对"，图案区只问"这条闭合曲线有没有断"。
    // 两者量纲不同、门槛不同，分数各算各的，任一达标就判 NG。
    //
    // 为什么不复用划痕判据：logo 环在一个大框里只占一条 5~8px 宽的线，
    // 框内九成是背景，"框内墨量"被稀释得什么都测不出来。
    // 早先那个 279x177 的假字符框就是这么来的 —— 它既不是一个字，
    // 也量不出环的断口。
    patternResults_.clear();
    if (sens_.checkPattern) {
        for (const auto& rg : model_.regions) {
            if (!rg.isPattern()) continue;

            // 【椭圆不需要跟着平移 —— 这一步曾经写错，造成 0095_RT 硬误报】
            //
            // 上面配准用的矩阵是 M = [1,0,-sh.x; 0,1,-sh.y] 配 WARP_INVERSE_MAP。
            // WARP_INVERSE_MAP 的语义是 dst(x,y) = src(M·(x,y))，也就是用 M 把
            // **目标坐标映射回源坐标**。代入 M 得 dst(x,y) = src(x-sh.x, y-sh.y)，
            // 即图像内容整体移动了 **+sh**。
            //
            // 而配准的目的正是"把这张图挪回模型的基准位置"。挪完之后，
            // 模型里的坐标就重新对上了当前这张图，椭圆**保持原位**即可。
            //
            // 之前这里给椭圆加了 -sh，等于把"已经校正好的偏移"又反着加了一遍：
            // 实测 0095_RT 上覆盖率从 0.9944 掉到 0.9444，
            // 完好的环被报成"断口 4 处、缺口 8.5°、断点分 850"，是一次硬误报。
            //
            // 验证方法：在配准后的图上分别用 原位 / +sh / -sh 三个位置量覆盖率，
            // 原位 0.9944 最高，另两个 0.9264/0.9444 都更差。
            const chardet::RegionInfo& moved = rg;
            (void)lastAlignX_;
            (void)lastAlignY_;

            const auto pb = chardet::analyzePatternBreak(signal_, moved, cfg, sens_.pattern);
            if (!pb.valid) continue;
            patternResults_.push_back(pb);

            // 把椭圆画出来：用户要能看见"检测器到底在看哪一圈"，
            // 否则覆盖率和断口数只是两个看不懂的数字。
            {
                const cv::Scalar col = pb.flagged      ? cv::Scalar(0, 0, 255)
                                       : !pb.alignOk   ? cv::Scalar(0, 165, 255)
                                                       : cv::Scalar(80, 200, 80);
                cv::RotatedRect rr(
                    cv::Point2f(static_cast<float>(moved.ell.center.x),
                                static_cast<float>(moved.ell.center.y)),
                    cv::Size2f(static_cast<float>(moved.ell.a * 2),
                               static_cast<float>(moved.ell.b * 2)),
                    static_cast<float>(moved.ell.angleDeg));
                cv::ellipse(vis, rr, col, 2, cv::LINE_AA);

                // 断口位置逐个标出来
                for (const auto& pt : pb.breakAt) {
                    cv::circle(vis, cv::Point(cvRound(pt.x), cvRound(pt.y)), 6,
                               cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                }

                Hit h;
                if (!pb.alignOk) {
                    // 没对准 = 没验过，不能写成"正常"。
                    h.kind = QStringLiteral("图案区未对准");
                    h.detail = QStringLiteral("半径起伏%1px 超过墨线宽度，结论作废")
                                   .arg(QString::number(pb.radiusSwing, 'f', 1));
                } else if (pb.flagged) {
                    h.kind = QStringLiteral("图案断·") +
                             QString::fromUtf8(chardet::patternBandName(pb.band));
                    h.detail = QStringLiteral("断口%1处 最长缺口%2° 断点分%3 覆盖%4")
                                   .arg(pb.breaks)
                                   .arg(QString::number(pb.maxBreakAngle, 'f', 2))
                                   .arg(QString::number(pb.score, 'f', 0))
                                   .arg(QString::number(pb.coverage, 'f', 3));
                } else {
                    h.kind = QStringLiteral("图案区正常");
                    h.detail = QStringLiteral("缺口%1处 最长%2° 覆盖%3")
                                   .arg(pb.gapCount)
                                   .arg(QString::number(pb.maxGapAngle, 'f', 2))
                                   .arg(QString::number(pb.coverage, 'f', 3));
                }
                h.score = pb.score;
                h.where = QStringLiteral("中心(%1,%2) 半轴%3x%4")
                              .arg(QString::number(moved.ell.center.x, 'f', 0))
                              .arg(QString::number(moved.ell.center.y, 'f', 0))
                              .arg(QString::number(moved.ell.a, 'f', 1))
                              .arg(QString::number(moved.ell.b, 'f', 1));
                // 未对准和正常都不该混进缺陷列表，但仍要让人看见
                if (pb.flagged || !pb.alignOk) hits_.push_back(h);
            }
        }
    }

    // ---- 溢料检测：细长多余笔画（PCA 长宽比判别）
    //
    // 与划痕的区别：溢料是"多出来的一撮墨"，形状细长但比划痕粗、比划痕短；
    // 划痕是细直线。这里用主成分分析的长宽比 > 2.2 来识别细长形状。
    if (sens_.checkAppendage) {
        cv::Mat tBin, mBin;
        cv::compare(signal_, cfg.strokeHi, tBin, cv::CMP_GT);
        cv::compare(model_.freq, static_cast<int>(cfg.blankFreq * 255), mBin, cv::CMP_GT);

        // 模型空白处多出来的墨
        cv::Mat notM;
        cv::bitwise_not(mBin, notM);
        cv::Mat extra;
        cv::bitwise_and(tBin, notM, extra);

        // 排除笔画邻域：溢料必须是"离笔画有距离"的独立墨迹。
        //
        // 不做这步会得到大量假溢料——实测 0110 单张图检出 64 个候选，
        // 全是笔画边缘的残差碎片。原因是笔画的边缘过渡带恰好同时满足
        // "模板频率低（边缘处频率回落）"和"待检信号高（边缘处仍有墨）"，
        // 于是每个字的轮廓都被切成碎块误报。
        {
            const cv::Mat kDil = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
            cv::Mat inkDil;
            cv::dilate(mBin, inkDil, kDil);   // 笔画外扩 4px
            cv::Mat notNear;
            cv::bitwise_not(inkDil, notNear);
            cv::bitwise_and(extra, notNear, extra);
        }
        const cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
        cv::morphologyEx(extra, extra, cv::MORPH_OPEN, k3);

        cv::Mat labels, stats, cents;
        const int n = cv::connectedComponentsWithStats(extra, labels, stats, cents, 8, CV_32S);
        for (int k = 1; k < n; ++k) {
            const int area = stats.at<int>(k, cv::CC_STAT_AREA);
            if (area < sens_.appendMinArea || area > sens_.appendMaxArea) continue;

            const cv::Rect rc(stats.at<int>(k, cv::CC_STAT_LEFT),
                              stats.at<int>(k, cv::CC_STAT_TOP),
                              stats.at<int>(k, cv::CC_STAT_WIDTH),
                              stats.at<int>(k, cv::CC_STAT_HEIGHT));

            // PCA 求长宽比
            std::vector<cv::Point> pts;
            pts.reserve(static_cast<size_t>(area));
            for (int y = rc.y; y < rc.y + rc.height; ++y) {
                const int* row = labels.ptr<int>(y);
                for (int x = rc.x; x < rc.x + rc.width; ++x) {
                    if (row[x] == k) pts.emplace_back(x, y);
                }
            }
            if (pts.size() < 80) continue;

            cv::Mat data(static_cast<int>(pts.size()), 2, CV_32F);
            for (size_t j = 0; j < pts.size(); ++j) {
                data.at<float>(static_cast<int>(j), 0) = static_cast<float>(pts[j].x);
                data.at<float>(static_cast<int>(j), 1) = static_cast<float>(pts[j].y);
            }
            cv::Mat mean, covar;
            cv::calcCovarMatrix(data, covar, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS);
            cv::Mat evals;
            cv::eigen(covar, evals);
            if (evals.rows < 2) continue;

            const double l1 = std::max(evals.at<double>(0), 1e-6);
            const double l2 = std::max(evals.at<double>(1), 1e-6);
            const double elong = std::sqrt(l1 / l2);
            if (elong <= sens_.appendMinElong) continue;

            // 排除细直线（那是划痕）：溢料的长宽比通常在 2.2~6 之间，
            // 且宽度明显大于划痕
            const double width = std::min(rc.width, rc.height);
            if (width < 4) continue;

            // 只检查字形附近：水平 10px、垂直 5px 以内
            // 只检查字符框内
            if (!insideAnyChar(rc)) continue;
            drawTag(vis, rc, cv::Scalar(0, 140, 255), "溢料", 2);

            Hit h;
            h.kind = QStringLiteral("溢料");
            h.score = area;
            h.detail = QStringLiteral("%1px 长宽比%2").arg(area).arg(elong, 0, 'f', 1);
            h.where = QStringLiteral("(%1,%2) %3×%4").arg(rc.x).arg(rc.y).arg(rc.width).arg(rc.height);
            hits_.push_back(h);
        }
    }

    // ---- 污损检测：模型空白处出现成块的墨
    //
    // 与划痕的区别：污损是面状（块），划痕是线状。
    // 只统计远离笔画的位置，避免把笔画边缘的刻印抖动算进来。
    if (sens_.checkDirt) {
        cv::Mat tBin, mBin;
        cv::compare(signal_, cfg.strokeHi, tBin, cv::CMP_GT);
        cv::compare(model_.freq, static_cast<int>(cfg.blankFreq * 255), mBin, cv::CMP_GT);

        const cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15});
        cv::Mat inkDil, farFromInk;
        cv::dilate(mBin, inkDil, kd);
        cv::bitwise_not(inkDil, farFromInk);

        cv::Mat dirt;
        cv::bitwise_and(tBin, farFromInk, dirt);
        const cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
        cv::morphologyEx(dirt, dirt, cv::MORPH_OPEN, k5);

        cv::Mat labels, stats, cents;
        const int n = cv::connectedComponentsWithStats(dirt, labels, stats, cents, 8, CV_32S);
        for (int i = 1; i < n; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            // （面积门槛在取得 rect 后结合距离一起判断）

            const cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                             stats.at<int>(i, cv::CC_STAT_TOP),
                             stats.at<int>(i, cv::CC_STAT_WIDTH),
                             stats.at<int>(i, cv::CC_STAT_HEIGHT));
            const double elong = static_cast<double>(std::max(r.width, r.height)) /
                                 std::max(1, std::min(r.width, r.height));
            if (elong >= 3.0) continue;   // 细长的归划痕

            // 只检查字形附近：水平 10px、垂直 5px 以内
            // 只检查字符框内
            if (!insideAnyChar(r)) continue;
            if (area < sens_.dirtMinArea) continue;

            drawTag(vis, r, cv::Scalar(255, 0, 255), "污损", 2);

            Hit h;
            h.kind = QStringLiteral("污损");
            h.score = area / 100.0;
            h.detail = QStringLiteral("%1px").arg(area);
            h.where = QStringLiteral("(%1,%2) %3×%4").arg(r.x).arg(r.y).arg(r.width).arg(r.height);
            hits_.push_back(h);
        }
    }

    // ---- 局部缺墨检测（大框专用）
    //
    // 解决"logo 大框稀释缺陷"的问题。
    //
    // 实测案例：0110 的椭圆标左上角断了一点（约 15×10px 缺口）。
    // 按整框比例算，缺口只占 logo 墨量的 1~2%，任何"整框缺墨率"判据都检不出；
    // 但按局部小格判断，那一格就几乎全空，立刻暴露。
    //
    // 做法：把每个字符框（尤其是大框）划分为约 30×30 的小格，
    //       逐格比较模型墨量与待检墨量，某格缺墨过半即报。
    if (sens_.checkLocalMiss) {
        cv::Mat tBin, mBin;
        cv::compare(signal_, cfg.strokeHi, tBin, cv::CMP_GT);
        cv::compare(model_.freq, static_cast<int>(cfg.missFreq * 255), mBin, cv::CMP_GT);

        constexpr int kCell = 30;   // 小格边长

        for (const auto& u : units_) {
            const cv::Rect b = u.bbox & cv::Rect(0, 0, signal_.cols, signal_.rows);
            if (b.width < kCell * 2 || b.height < kCell * 2) continue;   // 小字符走字符级判据

            for (int y = b.y; y < b.y + b.height; y += kCell) {
                for (int x = b.x; x < b.x + b.width; x += kCell) {
                    const cv::Rect cell(x, y, std::min(kCell, b.x + b.width - x),
                                        std::min(kCell, b.y + b.height - y));
                    if (cell.width < 8 || cell.height < 8) continue;

                    const int mInk = cv::countNonZero(mBin(cell));
                    if (mInk < 60) continue;   // 该格模型本来就没墨，不判

                    const int tInk = cv::countNonZero(tBin(cell));
                    const double ratio = static_cast<double>(tInk) / mInk;
                    if (ratio > sens_.localMissRatio) continue;   // 墨还在

                    // 缺口所在的格子，向外扩一点便于看清
                    cv::Rect show = cell;
                    show.x -= 4;
                    show.y -= 4;
                    show.width += 8;
                    show.height += 8;
                    drawTag(vis, show, cv::Scalar(0, 0, 255), "缺墨", 2);

                    Hit h;
                    h.kind = QStringLiteral("缺墨");
                    h.score = 1.0 - ratio;
                    h.detail = QStringLiteral("局部缺墨 %1%")
                                   .arg(static_cast<int>((1.0 - ratio) * 100));
                    h.where = QStringLiteral("(%1,%2) %3×%4")
                                  .arg(cell.x).arg(cell.y).arg(cell.width).arg(cell.height);
                    hits_.push_back(h);
                }
            }
        }
    }

    // ---- 多模板双指标（可疑位置提示）
    //
    // 与其它判据的关系：其它判据答"这颗件是不是 NG"，
    // 这一条额外回答"哪个字符最可疑"，即使整体判 OK 也要标出来供人工复核。
    //
    // 注意：默认只做提示、不判 NG（multiDecideNg=false）。原因见头文件注释：
    // 真实 OK 与真实 NG 的比值分布完全重叠，强行判定只会造成误报。
    if (sens_.checkMultiTmpl && !model_.charTemplates.empty()) {
        std::vector<chardet::CharUnit> allU = units_;
        const auto mr = chardet::diffMultiTemplate(signal_, model_, allU);

        if (mr.valid) {
            const double ratio = mr.secondDiff > 1e-6 ? mr.maxDiff / mr.secondDiff : 99.0;

            // 逐字符标注
            //   红框「刻错?」：仅在显式允许判 NG 且比值、绝对差异都超门槛时出现
            //   黄框「可疑」：d_c 超过提示门槛（即使整体 OK 也标出）
            for (size_t i = 0; i < allU.size() && i < mr.perChar.size(); ++i) {
                const double d = mr.perChar[i];
                if (d <= 0.0) continue;

                const bool isWorst = (d >= mr.maxDiff - 1e-6);
                const bool isNg =
                    sens_.multiDecideNg && isWorst && ratio >= sens_.multiRatio &&
                    mr.maxDiff >= sens_.multiWarn;
                const bool isWarn = d >= sens_.multiWarn;

                if (!isNg && !isWarn) continue;

                const cv::Rect r = allU[i].bbox;
                if (isNg) {
                    drawTag(vis, r, cv::Scalar(0, 0, 255), "刻错?", 2);
                } else {
                    drawTag(vis, r, cv::Scalar(0, 215, 255), "可疑", 2);
                }

                Hit h;
                h.kind = isNg ? QStringLiteral("缺笔画") : QStringLiteral("可疑");
                h.score = d;
                h.detail = QStringLiteral("与模板差异 %1%2")
                               .arg(d, 0, 'f', 3)
                               .arg(isWorst ? QStringLiteral("（本件最可疑字）") : QString());
                h.where = QStringLiteral("(%1,%2) %3×%4")
                              .arg(r.x).arg(r.y).arg(r.width).arg(r.height);
                hits_.push_back(h);
            }

            multiMax_ = mr.maxDiff;
            multiSecond_ = mr.secondDiff;
            multiRatio_ = ratio;

            // 整体判定：仅在显式开启时才允许判 NG
            if (sens_.multiDecideNg && ratio >= sens_.multiRatio &&
                mr.maxDiff >= sens_.multiWarn) {
                Hit h;
                h.kind = QStringLiteral("缺笔画");
                h.score = ratio;
                h.detail = QStringLiteral("max/second = %1/%2 = %3（一枝独秀）")
                               .arg(mr.maxDiff, 0, 'f', 3)
                               .arg(mr.secondDiff, 0, 'f', 3)
                               .arg(ratio, 0, 'f', 1);
                h.where = QStringLiteral("整幅");
                hits_.push_back(h);
            }
        }
    }

    // 生成对比图：模型 | 待检原图 | 标注结果
    //
    // 三栏并排，便于人工对照"模型应该长什么样 / 实际长什么样 / 差异标在哪"。
    // 同时写入磁盘，方便存档与反馈。
    {
        cv::Mat modelBgr;
        cv::cvtColor(model_.freq, modelBgr, cv::COLOR_GRAY2BGR);

        const std::vector<cv::Mat> panels = {modelBgr, rawBgr_, vis};
        cv::Mat cmp;
        cv::hconcat(panels, cmp);

        // 顶部标题条
        constexpr int barH = 28;
        cv::Mat titled;
        cv::copyMakeBorder(cmp, titled, barH, 0, 0, 0, cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));
        const int w = modelBgr.cols;
        auto putTitle = [&](int col, const std::string& text, const cv::Scalar& color) {
            cv::putText(titled, text, cv::Point(col + 8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color,
                        1, cv::LINE_AA);
        };
        putTitle(0, "MODEL (template)", cv::Scalar(0, 255, 255));
        putTitle(w, "TEST (raw)", cv::Scalar(0, 255, 255));
        // NG 只由划痕决定。缺字/缺墨/多墨是旁证，不能单独判 NG——
        // 这与批量检测（batchcore.hpp）的判定口径必须一致，否则
        // 同一张图在两个页面得出相反结论。
        int nScratch = 0;
        double bestS = 0.0;
        for (const auto& h : hits_) {
            if (!h.kind.startsWith(QStringLiteral("划痕"))) continue;
            ++nScratch;
            if (h.score > bestS) bestS = h.score;
        }
        const bool ng = nScratch > 0;
        // putText 只认 ASCII（Hershey 字体没有中文字形，中文会画成方块/问号），
        // 所以标题一律用英文，中文解释放在状态栏和右侧列表里。
        const std::string title =
            ng ? ("RESULT: scratch x" + std::to_string(nScratch) + " S=" +
                  std::to_string(static_cast<int>(bestS)) + " -> NG")
               : std::string("RESULT: no scratch -> OK");
        putTitle(2 * w, title, ng ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));

        resultBgr_ = titled;

        // 存盘：写到系统临时目录，**不写产品图旁边**。
        //
        // 原来是 QDir(fi.absolutePath())，即"与产品图同目录"。这在生产上
        // 有两个坏处：
        //   1. 检测会在被检图片的目录里留下一堆 _compare.png，客户拿去
        //      打包/归档时会连结果图一起拷走，数据集被污染；
        //   2. 目录只读或没有写权限时（放在共享盘、光盘、程序目录下）
        //      imwrite 直接失败，对比图就存不出来。
        // 自检里这条路径就撞上过 permission denied。
        // 对比图只是给人看的中间产物，放临时目录即可，界面上照样能看、
        // 照样能另存。
        const QFileInfo fi(imagePath_);
        const QString cacheDir = QDir(QDir::tempPath()).absoluteFilePath(QStringLiteral("markloc"));
        QDir().mkpath(cacheDir);
        comparePath_ = QDir(cacheDir)
                           .absoluteFilePath(fi.completeBaseName() + QStringLiteral("_compare.png"));
        if (!cv::imwrite(comparePath_.toUtf8().toStdString(), resultBgr_)) {
            comparePath_.clear();
        }
    }

    showImage(resultBgr_);
    refreshList();
    updateStatus();
    // 分数面板切到前台：单张检测后最该看的"报了几分、调到多少会变档"
    fillScoreTable();
    if (bottomStack_) bottomStack_->setCurrentIndex(0);
}

void DefectViewWindow::refreshList() {
    list_->clear();
    std::sort(hits_.begin(), hits_.end(),
              [](const Hit& a, const Hit& b) { return a.score > b.score; });

    if (hits_.empty()) {
        auto* it = new QListWidgetItem(QStringLiteral("未检出缺陷"));
        it->setForeground(QColor(60, 140, 60));
        list_->addItem(it);
        return;
    }
    for (const auto& h : hits_) {
        auto* it = new QListWidgetItem(QStringLiteral("【%1】%2").arg(h.kind, h.detail));
        if (h.kind == QStringLiteral("缺墨")) it->setForeground(QColor(200, 60, 60));
        else if (h.kind == QStringLiteral("多墨")) it->setForeground(QColor(200, 130, 40));
        else it->setForeground(QColor(160, 160, 0));
        it->setToolTip(h.where);
        list_->addItem(it);
    }
}

void DefectViewWindow::showImage(const cv::Mat& bgr) {
    const QImage q = matToQImage(bgr);
    if (q.isNull()) return;
    imgView_->setImage(q);
    imgView_->fitToWindow();
    if (zoomSpin_) {
        zoomSpin_->blockSignals(true);
        zoomSpin_->setValue(static_cast<int>(imgView_->zoom() * 100));
        zoomSpin_->blockSignals(false);
    }
}

// ---------------------------------------------------------------- Batch detection

void DefectViewWindow::cancelBatch() {
    batchCancel_.store(true);
    if (cancelBtn_) cancelBtn_->setEnabled(false);
    status_->setText(QStringLiteral("正在取消…（当前这张跑完即停）"));
}

void DefectViewWindow::runBatch() {
    if (batching_) return;
    if (!modelReady_) {
        QMessageBox::warning(this, QStringLiteral("未载入模型"),
                             QStringLiteral("请先选择模型目录并载入。"));
        return;
    }

    const QString dir = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择要批量检测的图片文件夹"), QDir(appDir()).absoluteFilePath(QStringLiteral("..")));
    if (dir.isEmpty()) return;

    const QStringList filters = {QStringLiteral("*.tif"), QStringLiteral("*.tiff"),
                                 QStringLiteral("*.png"), QStringLiteral("*.bmp"),
                                 QStringLiteral("*.jpg")};
    QStringList files;
    for (const QString& f : filters) {
        for (const QFileInfo& fi : QDir(dir).entryInfoList({f}, QDir::Files, QDir::Name)) {
            // 只处理 v00 视角的图片：文件名以 _00 结尾（如 xxx_00.tif）。
            // 其余视角看到的是引脚与封装侧面，与刻印字符检测无关。
            const QString base = fi.completeBaseName();
            if (!base.endsWith(QStringLiteral("_00"))) continue;
            files << fi.absoluteFilePath();
        }
    }
    // 批量检测会在同一视角下逐张跑，要求尺寸一致
    if (gray_.empty()) {
        cv::Mat probe;
        if (!readGray(files.first(), probe)) {
            QMessageBox::warning(this, QStringLiteral("读取失败"),
                                 QStringLiteral("无法读取第一张图片。"));
            return;
        }
        if (probe.size() != model_.freq.size()) {
            QMessageBox::warning(this, QStringLiteral("尺寸不符"),
                                 QStringLiteral("图片 %1×%2 与模型 %3×%4 尺寸不同。")
                                     .arg(probe.cols).arg(probe.rows)
                                     .arg(model_.freq.cols).arg(model_.freq.rows));
            return;
        }
    }

    batching_ = true;
    batchCancel_.store(false);
    if (batchBtn_) batchBtn_->setEnabled(false);
    if (cancelBtn_) {
        cancelBtn_->setEnabled(true);
        cancelBtn_->setVisible(true);
    }
    if (progressBar_) {
        progressBar_->setRange(0, files.size());
        progressBar_->setValue(0);
        progressBar_->setFormat(QStringLiteral("%v / %m"));
        progressBar_->setVisible(true);
    }

    if (units_.empty()) {
        QMessageBox::warning(this, QStringLiteral("模型为空"), QStringLiteral("模型里没有字符。"));
        batching_ = false;
        return;
    }

    batchRows_.clear();
    batchRows_.reserve(static_cast<size_t>(files.size()));
    batchTotalMs_ = 0.0;
    batchDir_ = dir;
    // 设置只取一次：批量跑到一半用户改参数会让整批不可比
    markloc::DetectSettings det;
    det.cfg = model_.config;
    det.perCharAlign = sens_.perCharAlign;
    det.checkRowCount = sens_.checkRowCount;
    det.checkCharLevel = sens_.checkCharLevel;
    det.checkScratch = sens_.checkScratch;
    det.scratch = scratchParams();
    int doneCount = 0;
    // 每张的检测部分单独计时
    for (const QString& path : files) {
        if (batchCancel_.load()) break;

        QElapsedTimer tElapsed;
        tElapsed.start();

        cv::Mat g;
        if (!readGray(path, g) || g.size() != model_.freq.size()) {
            ++doneCount;
            if (progressBar_) progressBar_->setValue(doneCount);
            QCoreApplication::processEvents();
            continue;
        }
        // ---- 单张检测：与单文件「开始检测」共用同一份实现（gui/batchcore.hpp）
        bool valid = false;
        BatchRow row = markloc::detectOne(g, model_, units_, det, &valid);
        if (!valid) {
            ++doneCount;
            if (progressBar_) progressBar_->setValue(doneCount);
            QCoreApplication::processEvents();
            continue;
        }
        row.name = u8(QFileInfo(path).fileName());
        row.path = u8(QFileInfo(path).absoluteFilePath());

        row.elapsedMs = tElapsed.elapsed();
        batchTotalMs_ += row.elapsedMs;
        batchRows_.push_back(row);

        ++doneCount;
        if (progressBar_) progressBar_->setValue(doneCount);
        QCoreApplication::processEvents();
    }

    // 结果先进表格：即使 CSV 写不进去，界面上也能看到
    fillBatchTable();
    // 同时灌进下方滑动列表，并切到那一页——批量跑完最想看的是"逐项翻一遍"
    fillBatchList();
    if (bottomStack_) bottomStack_->setCurrentIndex(1);
    if (tabs_) tabs_->setCurrentIndex(1);

    // 写 CSV：优先写到图片目录，失败则退到程序目录
    int ngCount = 0;
    for (const auto& r : batchRows_) {
        if (r.ng) ++ngCount;
    }

    QString outPath = QDir(dir).absoluteFilePath(QStringLiteral("batch_result.csv"));
    bool ok = false;
    for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
        if (attempt == 1) {
            outPath = QDir(appDir()).absoluteFilePath(QStringLiteral("batch_result.csv"));
        }
        QFile f(outPath);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) continue;

        QTextStream ts(&f);
        ts.setEncoding(QStringConverter::Utf8);
        ts << "\xEF\xBB\xBF";   // BOM，Excel 打开不乱码
        // 划痕列放在最前：它是判定依据，其他三列只是旁证
        ts << "file,result,class,scratch_S,scratch_band,scratch_chars,scratch_depth,"
              "scratch_count,missing_char,missing_ink,extra_ink,elapsed_ms\n";
        for (const auto& r : batchRows_) {
            ts << q8(r.name) << ',' << (r.ng ? "NG" : "OK") << ',' << q8(r.cls) << ','
               << static_cast<int>(r.scratchS) << ',' << r.scratchBand << ',' << r.scratchChars << ','
               << static_cast<int>(r.scratchDepth) << ',' << r.scratchCount << ',' << r.rowMissing
               << ',' << r.miss << ',' << r.extra << ',' << static_cast<int>(r.elapsedMs) << '\n';
        }
        f.close();
        ok = (f.error() == QFile::NoError);
    }

    QString msg = QStringLiteral("共 %1 张，NG %2 张，OK %3 张。\n平均 %4 ms/张，总计 %5 s。")
                      .arg(batchRows_.size())
                      .arg(ngCount)
                      .arg(static_cast<int>(batchRows_.size()) - ngCount)
                      .arg(batchRows_.empty() ? 0.0 : batchTotalMs_ / batchRows_.size(), 0, 'f', 1)
                      .arg(batchTotalMs_ / 1000.0, 0, 'f', 2);
    {
        QStringList parts;
        for (const QString& c : batchClasses_) {
            int cnt = 0;
            for (const auto& r : batchRows_) if (q8(r.cls) == c) ++cnt;
            parts << QStringLiteral("%1 %2").arg(c).arg(cnt);
        }
        msg += QStringLiteral("\n分类：%1").arg(parts.join(QStringLiteral(" · ")));
    }
    if (ok) {
        msg += QStringLiteral("\n\nCSV 已保存到：\n%1").arg(outPath);
    } else {
        msg += QStringLiteral("\n\nCSV 写入失败（可能是文件夹只读）。\n结果已在右侧「批量结果」页显示。");
    }
    QMessageBox::information(this, QStringLiteral("批量检测完成"), msg);

    batching_ = false;
    if (batchBtn_) batchBtn_->setEnabled(true);
    if (cancelBtn_) cancelBtn_->setVisible(false);
    if (progressBar_) progressBar_->setVisible(false);
    updateStatus();
}

/// 把批量结果灌进下方滑动列表
///
/// 与表格是同一份数据（batchRows_）的两种看法：
///   表格 = 横向对比，一屏看几十行，适合找规律
///   滑动列表 = 纵向逐项，一次看一项的图与分，适合"过一遍确认"
/// 列表项文字带上档位和严重度，不用点开就知道该看哪几张。
void DefectViewWindow::fillBatchList() {
    if (!batchList_) return;
    batchList_->clear();
    slideIndex_ = -1;
    for (const auto& r : batchRows_) {
        // NG/OK 打在行首，并配底色。
        //
        // 原来是「文件名   分类 S=值」——文件名在本项目里是一长串
        // "0026_FT_Failed [2026-1010-024655-034] (字符) _00.tif"，
        // 一眼扫过去全一样，NG 和 OK 反而要靠颜色才能分辨。判定结论是
        // 这一列最重要的信息，必须排在名字前面。
        const QString verdict = r.ng ? QStringLiteral("NG") : QStringLiteral("OK");
        QString tail;
        if (r.scratchCount > 0) {
            tail = QStringLiteral("%1  S=%2  %3")
                       .arg(q8(r.cls))
                       .arg(static_cast<int>(r.scratchS))
                       .arg(q8(r.name));
        } else {
            tail = QStringLiteral("%1        %2").arg(q8(r.cls), q8(r.name));
        }
        auto* it = new QListWidgetItem(verdict + QStringLiteral("  ") + tail);

        // 结论色优先于档位色：是否 NG 是客户最终要的结论
        if (r.ng) {
            it->setForeground(QColor(200, 30, 30));
            it->setBackground(QColor(255, 236, 236));
        } else {
            it->setForeground(QColor(35, 120, 60));
            it->setBackground(QColor(238, 250, 240));
        }
        // 字体加粗 NG：列表里一眼定位红行
        QFont f = batchList_->font();
        f.setBold(r.ng);
        it->setFont(f);
        it->setToolTip(QStringLiteral("%1\n结论：%2\n分类：%3")
                           .arg(q8(r.name), verdict, q8(r.cls)));
        batchList_->addItem(it);
    }
    // 整批汇总：列表下方那块空白用起来。
    //
    // 档位分布单独列出来，是因为处理方式不同：确定的一律判退，
    // 可疑的要人工复看，旁证只是线索。只报总数等于没说。
    if (batchFoot_) {
        if (batchRows_.empty()) {
            batchFoot_->setVisible(false);
        } else {
            int ng = 0;
            std::map<QString, int> bands;
            for (const auto& r : batchRows_) {
                if (r.ng) ++ng;
                bands[q8(r.cls)] += 1;
            }
            QString txt = QStringLiteral("整批 %1 张\nNG %2 · OK %3")
                              .arg(static_cast<int>(batchRows_.size()))
                              .arg(ng)
                              .arg(static_cast<int>(batchRows_.size()) - ng);
            for (const auto& kv : bands)
                txt += QStringLiteral("\n%1 %2").arg(kv.first).arg(kv.second);
            batchFoot_->setText(txt);
            batchFoot_->setVisible(true);
        }
    }

    if (!batchRows_.empty()) {
        // 把下方撑到能看清对比图的高度。
        //
        // 分数面板只要两百多像素就够（就几行数字），滑动列表要的是"看图"，
        // 同样 240px 高度里再扣掉说明和按钮，图片只剩一条缝——这正是
        // 「点开了也看不清」的原因之一。批量跑完就默认让列表占大头，
        // 客户仍可以拖分割条自己调。
        if (leftSplit_) {
            const int total = leftSplit_->height() > 300 ? leftSplit_->height() : 860;
            leftSplit_->setSizes({std::max(220, total * 2 / 5), total * 3 / 5});
        }
        batchList_->setCurrentRow(0);
        showSlide(0);
    }
}

void DefectViewWindow::fillBatchTable() {
    if (!batchTable_) return;

    // 分类是固定的六档，顺序即严重度顺序。用固定表而不是"出现过的分类"，
    // 是为了让筛选下拉框在每批之间保持一致——跑两次批量，下拉框的项会跳位
    // 是最容易被误操作的地方。
    //
    // 只有前两档（确定/明显）算"划痕实症"；可疑档是过了报告门槛、
    // 落在客户定的切点之下的那些，要不要算 NG 由切点决定，界面不改判定。
    batchClasses_ = {QStringLiteral("确定"), QStringLiteral("明显"), QStringLiteral("可疑"),
                     QStringLiteral("对齐异常"), QStringLiteral("旁证"), QStringLiteral("OK")};

    const auto classColor = [](const QString& c) -> QColor {
        if (c == QStringLiteral("确定")) return QColor(200, 40, 40);
        if (c == QStringLiteral("明显")) return QColor(215, 110, 0);
        if (c == QStringLiteral("可疑")) return QColor(160, 140, 0);
        if (c == QStringLiteral("对齐异常")) return QColor(150, 60, 160);
        if (c == QStringLiteral("旁证")) return QColor(70, 90, 160);
        return QColor(40, 140, 40);
    };

    batchTable_->setRowCount(static_cast<int>(batchRows_.size()));
    for (int i = 0; i < static_cast<int>(batchRows_.size()); ++i) {
        const auto& r = batchRows_[static_cast<size_t>(i)];
        const QColor col = classColor(q8(r.cls));
        const auto setCell = [&](int c, const QString& text) {
            auto* it = new QTableWidgetItem(text);
            it->setForeground(c == 0 ? col : QColor(60, 60, 60));
            batchTable_->setItem(i, c, it);
        };
        setCell(0, q8(r.cls));
        setCell(1, q8(r.name));
        setCell(2, r.scratchBand >= 0 ? QString::number(static_cast<int>(r.scratchS))
                                      : QStringLiteral("-"));
        setCell(3, r.scratchBand >= 0 ? QString::number(r.scratchChars) : QStringLiteral("-"));
        setCell(4, r.scratchBand >= 0 ? QString::number(static_cast<int>(r.scratchDepth))
                                      : QStringLiteral("-"));
        // 对齐位移 + 它的取证人（哪个字、往哪边挪了多少）。
        // 位移贴到搜索边界说明这一张的框是被硬拽过去的，结果不能当"查过了没问题"。
        setCell(5, QStringLiteral("%1 %2").arg(r.alignMaxShift).arg(q8(r.alignEvidence)));
        setCell(6, QString::number(r.rowMissing));
        setCell(7, QString::number(r.miss));
        setCell(8, QString::number(r.extra));
        setCell(9, QString::number(static_cast<int>(r.elapsedMs)));
        setCell(10, q8(r.rejectReason));
    }
    batchTable_->resizeColumnsToContents();
    // 文件名最长、别的列都是短数字：让文件名列吃掉剩余宽度，
    // 窄窗口下不至于被压成省略号
    batchTable_->horizontalHeader()->setStretchLastSection(false);
    batchTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    applyBatchFilter();

    // 统计页
    if (!statsView_) return;
    int ngCount = 0;
    for (const auto& r : batchRows_) if (r.ng) ++ngCount;
    const int n = static_cast<int>(batchRows_.size());

    QString s;
    s += QStringLiteral("=== 批量检测汇总 ===\n");
    s += QStringLiteral("样本数   : %1\n").arg(n);
    // 判定口径写死在这里，免得看报告的人自己猜"NG 到底算的哪一条"
    s += QStringLiteral("NG       : %1  （划痕过报告门槛 %2 张即算）\n")
             .arg(ngCount)
             .arg(static_cast<int>(sens_.scratchMinComposite));
    s += QStringLiteral("OK       : %1\n").arg(n - ngCount);
    s += QStringLiteral("NG 率    : %1%\n")
             .arg(n > 0 ? QString::number(100.0 * ngCount / n, 'f', 1) : QStringLiteral("0"));
    s += QStringLiteral("\n--- 分类 ---\n");
    for (const QString& c : batchClasses_) {
        int cnt = 0;
        for (const auto& r : batchRows_) if (q8(r.cls) == c) ++cnt;
        s += QStringLiteral("%1 : %2\n").arg(c, -6).arg(cnt);
    }
    s += QStringLiteral("\n判定依据：只有「划痕」一条（客户确认的唯一缺陷类型）。\n"
                        "  确定/明显/可疑 三档都算 NG，差别只在严重度；\n"
                        "  切到哪一档才算 NG 由「灵敏度」里的四档切点决定，界面不改判定。\n"
                        "  缺字/缺墨/多墨 只作旁证，不参与 NG 判定。\n"
                        "  对齐异常 = 逐字符对齐位移贴到搜索半径，且同时报告了墨量差异；\n"
                        "  这类结果说明框没对准字形，不能当「查过了没问题」，需要人工看。\n");
    s += QStringLiteral("总耗时   : %1 s\n").arg(batchTotalMs_ / 1000.0, 0, 'f', 2);
    s += QStringLiteral("平均单张 : %1 ms\n")
             .arg(n > 0 ? batchTotalMs_ / n : 0.0, 0, 'f', 1);
    if (n > 0) {
        s += QStringLiteral("吞吐     : %1 张/秒\n").arg(1000.0 * n / std::max(1.0, batchTotalMs_), 0,
                                                     'f', 2);
    }

    s += QStringLiteral("\n=== 当前参数 ===\n");
    s += QStringLiteral("模型视角 : %1\n").arg(viewCombo_ ? viewTag(viewCombo_->currentData().toInt())
                                                       : QStringLiteral("-"));
    s += QStringLiteral("字符数   : %1\n").arg(units_.size());
    s += QStringLiteral("逐字符对齐 : %1\n").arg(sens_.perCharAlign ? QStringLiteral("开")
                                                                  : QStringLiteral("关"));
    s += QStringLiteral("整字数量   : %1\n").arg(sens_.checkRowCount ? QStringLiteral("开")
                                                                   : QStringLiteral("关"));
    s += QStringLiteral("字符级判据 : %1  阈值 %2%\n")
             .arg(sens_.checkCharLevel ? QStringLiteral("开") : QStringLiteral("关"))
             .arg(15);
    s += QStringLiteral("划痕       : %1  长度≥%2px 覆盖≥%3% 能量≥%4\n")
             .arg(sens_.checkScratch ? QStringLiteral("开") : QStringLiteral("关"))
             .arg(sens_.scratchMinLen)
             .arg(sens_.scratchMinCov * 100)
             .arg(sens_.scratchMinEnergy);
    s += QStringLiteral("污损       : %1  面积≥%2px（仅字符框内）\n")
             .arg(sens_.checkDirt ? QStringLiteral("开") : QStringLiteral("关"))
             .arg(sens_.dirtMinArea);
    s += QStringLiteral("溢料       : %1  面积%2~%3 长宽比≥%4（仅字符框内）\n")
             .arg(sens_.checkAppendage ? QStringLiteral("开") : QStringLiteral("关"))
             .arg(sens_.appendMinArea)
             .arg(sens_.appendMaxArea)
             .arg(sens_.appendMinElong, 0, 'f', 1);
    s += QStringLiteral("局部缺墨   : %1\n").arg(sens_.checkLocalMiss ? QStringLiteral("开")
                                                                    : QStringLiteral("关"));
    s += QStringLiteral("多模板提示 : %1  判定 %2  maxDiff≥%3\n")
             .arg(sens_.checkMultiTmpl ? QStringLiteral("开") : QStringLiteral("关"))
             .arg(sens_.multiDecideNg ? QStringLiteral("参与") : QStringLiteral("仅提示"))
             .arg(sens_.multiWarn, 0, 'f', 3);
    if (multiSecond_ > 0.0) {
        s += QStringLiteral("  最近一张：max %1 / second %2 / 比值 %3\n")
                 .arg(multiMax_, 0, 'f', 3)
                 .arg(multiSecond_, 0, 'f', 3)
                 .arg(multiRatio_, 0, 'f', 2);
    }

    if (n > 0) {
        s += QStringLiteral("\n=== 最慢的 5 张 ===\n");
        std::vector<BatchRow> sorted = batchRows_;
        std::sort(sorted.begin(), sorted.end(),
                  [](const BatchRow& a, const BatchRow& b) { return a.elapsedMs > b.elapsedMs; });
        for (size_t i = 0; i < sorted.size() && i < 5; ++i) {
            s += QStringLiteral("  %1  %2 ms\n")
                     .arg(q8(sorted[i].name))
                     .arg(sorted[i].elapsedMs, 0, 'f', 0);
        }
    }
    statsView_->setPlainText(s);
}

/// 按分类筛选批量结果表
///
/// 把上一次单张检测的划痕填进下方分数面板
///
/// 表格按严重度从高到低排，和 hits_ 的排序口径一致。
/// 每一行都给出"升档切点 / 降档切点"，客户照着填「灵敏度…」即可，
/// 不需要自己反推 S 和档位的关系。
void DefectViewWindow::fillScoreTable() {
    if (!scoreTable_ || !scoreTitle_) return;
    scoreTable_->setRowCount(0);

    // 图案区结论先算出来：它和划痕是并列的两套判据，
    // "划痕为空"不等于"本张没问题"，标题不能只报划痕。
    const bool hasPattern = !patternResults_.empty();
    bool patternBad = false;
    for (const auto& pb : patternResults_) {
        if (pb.flagged || !pb.alignOk) patternBad = true;
    }

    if (scratchHits_.empty() && !patternBad) {
        scoreTitle_->setText(hasPattern ? QStringLiteral("缺陷分数：划痕 0 条，图案区无断口")
                                        : QStringLiteral("缺陷分数：本张未报出划痕"));
        scoreTitle_->setStyleSheet(QStringLiteral("font-weight:bold; color:#2e7d32;"));
        // 收起空表格，改显示明确结论。
        //
        // 空表格铺出来是 8 列表头 + 零行，占满整块面板却什么也没说，
        // 客户会以为"分数没算出来"。这里把结论和"为什么是 OK"讲清楚。
        scoreTable_->setVisible(false);
        if (scoreEmpty_) {
            scoreEmpty_->setText(
                hasPattern ? QStringLiteral("划痕 0 条，图案区未见断口 → 判 OK\n\n"
                                            "两套判据都已跑完：划痕查字的墨迹，"
                                            "图案区查闭合曲线有没有被切断。\n"
                                            "若你认为该判 NG，点「灵敏度…」把门槛调低后重测。")
                           : QStringLiteral("未报出划痕 → 判 OK\n\n"
                                            "字符识别、缺字、缺墨、多墨均已检查，未发现异常。\n"
                                            "若你认为该判 NG，点「灵敏度…」把门槛调低后重测。"));
            scoreEmpty_->setVisible(true);
        }
        return;
    }

    if (scratchHits_.empty()) {
        // 只有图案区报了：标题要点明是另一套判据报的，不能显示成"划痕 0 条"
        scoreTitle_->setText(QStringLiteral("缺陷分数：图案区断口 → 判 NG"));
        scoreTitle_->setStyleSheet(QStringLiteral("font-weight:bold; color:#c62828;"));
    }

    scoreTable_->setVisible(true);
    if (scoreEmpty_) scoreEmpty_->setVisible(false);

    // 按严重度降序，和图上标注的顺序一致
    auto rows = scratchHits_;
    std::sort(rows.begin(), rows.end(),
              [](const markloc::ScratchHit& a, const markloc::ScratchHit& b) {
                  return a.composite > b.composite;
              });

    const markloc::ScratchParams P = scratchParams();
    scoreTable_->setRowCount(static_cast<int>(rows.size()));
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& h = rows[i];
        const int r = static_cast<int>(i);
        auto setCell = [&](int col, const QString& text, const QColor& fg = QColor()) {
            auto* it = new QTableWidgetItem(text);
            if (fg.isValid()) it->setForeground(fg);
            it->setTextAlignment(Qt::AlignCenter);
            scoreTable_->setItem(r, col, it);
        };

        const int b = std::clamp(h.band, 0, 3);
        auto bandQColor = [](int band) {
            switch (band) {
                case 3: return QColor(198, 40, 40);    // 确定
                case 2: return QColor(239, 108, 0);    // 明显
                case 1: return QColor(245, 178, 0);    // 可疑
                default: return QColor(130, 130, 130); // 轻微
            }
        };
        setCell(0, QString::fromUtf8(markloc::bandName(b)), bandQColor(b));
        setCell(1, QString::number(static_cast<int>(h.composite)));
        setCell(2, QString::number(static_cast<int>(h.depth)));
        setCell(3, QString::number(h.charsHit));
        setCell(4, QStringLiteral("(%1,%2)").arg(h.a1.x).arg(h.a1.y));
        setCell(5, QStringLiteral("%1px").arg(static_cast<int>(h.len)));

        const markloc::ScratchEdges e = markloc::edgesOf(P, h);
        if (e.upCut >= 0.0) {
            setCell(6, QStringLiteral("%1（→%2）")
                           .arg(static_cast<int>(e.upCut))
                           .arg(QString::fromUtf8(markloc::bandName(b + 1))));
        } else {
            setCell(6, QStringLiteral("已是最高档"), QColor(120, 120, 120));
        }
        if (e.dnCut >= 0.0) {
            setCell(7, QStringLiteral("%1（→%2）")
                           .arg(static_cast<int>(e.dnCut))
                           .arg(QString::fromUtf8(markloc::bandName(b - 1))));
        } else {
            setCell(7, QStringLiteral("已是最低档"), QColor(120, 120, 120));
        }
    }
    // 图案区追加在划痕之后，用"图案"两字标出这是另一套判据。
    //
    // 两套分数的单位根本不同（深度×跨字数 vs 缺口角度×100），
    // 所以绝不能排进同一个降序里比大小 —— 那会暗示它们可以相加。
    for (const auto& pb : patternResults_) {
        if (!pb.flagged && pb.alignOk) continue;
        const int r = scoreTable_->rowCount();
        scoreTable_->insertRow(r);
        auto setPCell = [&](int col, const QString& text, const QColor& fg = QColor()) {
            auto* it = new QTableWidgetItem(text);
            if (fg.isValid()) it->setForeground(fg);
            it->setTextAlignment(Qt::AlignCenter);
            scoreTable_->setItem(r, col, it);
        };

        if (!pb.alignOk) {
            setPCell(0, QStringLiteral("图案·未对准"), QColor(239, 108, 0));
            setPCell(1, QString::number(static_cast<int>(pb.score)));
            setPCell(4, QStringLiteral("基准椭圆"));
            setPCell(5, QStringLiteral("起伏%1px").arg(QString::number(pb.radiusSwing, 'f', 1)));
            setPCell(6, QStringLiteral("先对准再判"), QColor(120, 120, 120));
            setPCell(7, QStringLiteral("—"), QColor(120, 120, 120));
            continue;
        }

        setPCell(0, QStringLiteral("图案·") + QString::fromUtf8(chardet::patternBandName(pb.band)),
                 QColor(198, 40, 40));
        setPCell(1, QString::number(static_cast<int>(pb.score)));
        // 图案区的"深度/跨字数"两列没有对应含义，留空而不是填 0，
        // 填 0 会让人以为量过、值为零。
        setPCell(2, QStringLiteral("—"), QColor(160, 160, 160));
        setPCell(3, QStringLiteral("—"), QColor(160, 160, 160));
        setPCell(4, QStringLiteral("断口%1处").arg(pb.breaks));
        setPCell(5, QStringLiteral("缺口%1°").arg(QString::number(pb.maxBreakAngle, 'f', 2)));
        // 降档切点：断点分只要降到这个值以下就不报了
        const double need = std::max(0.0, static_cast<double>(sens_.pattern.minScore));
        if (pb.score >= need) {
            setPCell(7, QStringLiteral("%1（→不报）").arg(QString::number(need, 'f', 0)));
        } else {
            setPCell(7, QStringLiteral("已是不报"), QColor(120, 120, 120));
        }
        setPCell(6, QStringLiteral("—"), QColor(160, 160, 160));
    }
    scoreTable_->resizeColumnsToContents();

    if (rows.empty()) return;  // 仅图案区报出时，标题已在上方设好
    char buf[160];
    std::snprintf(buf, sizeof(buf), "缺陷分数：%d 处划痕，最严重 S=%d（%s）", static_cast<int>(rows.size()),
                  static_cast<int>(rows.front().composite),
                  markloc::bandName(std::clamp(rows.front().band, 0, 3)));
    scoreTitle_->setText(QString::fromUtf8(buf));
    scoreTitle_->setStyleSheet(QStringLiteral("font-weight:bold; color:#c62828;"));
}

/// 幻灯片：上翻一项
void DefectViewWindow::slidePrev() {
    if (batchRows_.empty()) return;
    showSlide(std::max(0, slideIndex_ - 1));
}

/// 幻灯片：下翻一项
void DefectViewWindow::slideNext() {
    if (batchRows_.empty()) return;
    showSlide(std::min(static_cast<int>(batchRows_.size()) - 1, slideIndex_ + 1));
}

/// 显示第 i 项：左侧列表跟随、右侧对比图与文字同步
void DefectViewWindow::showSlide(int i) {
    if (!batchList_ || !slideView_ || !slideInfo_) return;
    if (batchRows_.empty()) {
        slideIndex_ = -1;
        slideView_->setImage(QImage());
        slideInfo_->setText(QStringLiteral("批量检测后在这里逐项查看"));
        return;
    }
    i = std::clamp(i, 0, static_cast<int>(batchRows_.size()) - 1);
    slideIndex_ = i;
    // 列表与幻灯片的选中状态保持一致，避免两处高亮互相矛盾
    if (batchList_->currentRow() != i) {
        const QSignalBlocker blk(batchList_);
        batchList_->setCurrentRow(i);
    }
    const auto& r = batchRows_[static_cast<size_t>(i)];

    QString info = QStringLiteral("%1   [%2 · %3]   (%4/%5)")
                       .arg(q8(r.name), r.ng ? QStringLiteral("NG") : QStringLiteral("OK"),
                            q8(r.cls))
                       .arg(i + 1)
                       .arg(static_cast<int>(batchRows_.size()));
    if (r.scratchCount > 0) {
        info += QStringLiteral("\n划痕 %1 处，最严重 S=%2，跨 %3 字，深度 %4，长度 %5px")
                    .arg(r.scratchCount)
                    .arg(static_cast<int>(r.scratchS))
                    .arg(static_cast<int>(r.scratchChars))
                    .arg(static_cast<int>(r.scratchDepth))
                    .arg(static_cast<int>(r.scratchLen));
    } else {
        info += QStringLiteral("\n未报出划痕");
        if (!r.rejectReason.empty()) info += QStringLiteral("：") + q8(r.rejectReason);
    }
    slideInfo_->setText(info);

    const QPixmap pm = makeSlidePixmap(i);
    if (pm.isNull()) {
        slideView_->setImage(QImage());
        slideView_->setPlaceholder(QStringLiteral("（该图无法生成对比图）"));
    } else {
        // 交给 ImageView 按视口适配。它会自己算 fit 缩放，
        // 后续滚轮/按钮放大缩小时也只是改同一张源图的缩放倍数，
        // 所以放大之后能看到原图细节，而不是把缩略图拉糊。
        slideView_->setPlaceholder(QString());
        slideView_->setImage(pm.toImage());
    }
}

/// 生成某一项的对比图
///
/// 出完整分辨率（原图尺寸），由 ImageView 负责缩放。
/// 以前在这里先缩到 640px 宽再交给 QLabel，后面怎么放大都只有 640px 的信息。
QPixmap DefectViewWindow::makeSlidePixmap(int row) {
    if (row < 0 || row >= static_cast<int>(batchRows_.size())) return QPixmap();
    const auto& r = batchRows_[static_cast<size_t>(row)];

    cv::Mat g;
    if (!readGray(q8(r.path), g)) return QPixmap();
    cv::Mat bgr;
    cv::cvtColor(g, bgr, cv::COLOR_GRAY2BGR);

    // 边框与标签按原图尺寸画：缩放由视图负责，文字跟着一起缩就不会糊
    if (r.scratchBand >= 0) {
        const cv::Scalar col = markloc::bandColor(r.scratchBand);
        cv::rectangle(bgr, cv::Rect(0, 0, bgr.cols, bgr.rows), col, 6);
    }
    // putText 只认 ASCII，中文会画成方块，这里用英文档位标签
    const std::string tag =
        r.scratchBand >= 0 ? (std::string(markloc::bandTag(r.scratchBand)) + "  S=" +
                              std::to_string(static_cast<int>(r.scratchS)))
                           : std::string("no scratch");
    const cv::Scalar tcol = r.scratchBand >= 0 ? markloc::bandColor(r.scratchBand)
                                               : cv::Scalar(120, 200, 120);
    // 加黑底再写字：浅色字压在亮塑封体上会看不清
    const double sc = std::max(1.4, bgr.cols / 700.0);
    const cv::Point org(static_cast<int>(12 * sc / 1.4) + 8,
                        static_cast<int>(34 * sc / 1.4) + 10);
    cv::putText(bgr, tag, org, cv::FONT_HERSHEY_SIMPLEX, sc, cv::Scalar(0, 0, 0),
                static_cast<int>(5 * sc / 1.4) + 2, cv::LINE_AA);
    cv::putText(bgr, tag, org, cv::FONT_HERSHEY_SIMPLEX, sc, tcol,
                static_cast<int>(2 * sc / 1.4) + 1, cv::LINE_AA);

    currentSlideImage_ = matToQImage(bgr).copy();
    return QPixmap::fromImage(currentSlideImage_);
}

/// 用隐藏行而不是重建表格：行号与 batchRows_ 严格对应，
/// 双击拿到的 row 就是真正的下标，不会因为筛选而错位。
void DefectViewWindow::applyBatchFilter() {
    if (!batchTable_ || !batchFilter_) return;
    const QString want = batchFilter_->currentData().toString();

    int shown = 0;
    for (int i = 0; i < batchTable_->rowCount(); ++i) {
        const bool inRange = i < static_cast<int>(batchRows_.size());
        const QString cls = inRange ? q8(batchRows_[static_cast<size_t>(i)].cls) : QString();
        const bool vis = want.isEmpty() || cls == want;
        batchTable_->setRowHidden(i, !vis);
        if (vis) ++shown;
    }

    if (!batchSummary_) return;
    QStringList parts;
    int ngCount = 0;
    for (const QString& c : batchClasses_) {
        int cnt = 0;
        for (const auto& r : batchRows_) {
            if (q8(r.cls) != c) continue;
            ++cnt;
            if (r.ng) ++ngCount;
        }
        if (cnt > 0) parts << QStringLiteral("%1 %2").arg(c).arg(cnt);
    }
    if (batchRows_.empty()) {
        batchSummary_->setText(QStringLiteral("还没有批量结果"));
        return;
    }
    batchSummary_->setText(QStringLiteral("共 %1 张 · NG %2 · 显示 %3 · %4")
                               .arg(batchRows_.size())
                               .arg(ngCount)
                               .arg(shown)
                               .arg(parts.join(QStringLiteral("  "))));
}

/// 双击批量结果的一行：载入该图、跑同一套检测、左侧直接给出对比图
void DefectViewWindow::openBatchRow(int row) {
    if (row < 0 || row >= static_cast<int>(batchRows_.size())) return;
    if (!modelReady_) {
        QMessageBox::warning(this, QStringLiteral("未载入模型"),
                             QStringLiteral("请先选择模型目录并载入。"));
        return;
    }

    const BatchRow r = batchRows_[static_cast<size_t>(row)];
    if (!loadImageFile(q8(r.path))) {
        QMessageBox::warning(this, QStringLiteral("读取失败"),
                             QStringLiteral("无法读取：\n%1").arg(q8(r.path)));
        return;
    }
    imagePath_ = q8(r.path);
    imageEdit_->setText(q8(r.path));

    // 走的是和「开始检测」完全相同的流程，所以左侧显示的就是对比图
    // （模型 | 原图 | 标注），不会出现"批量说 NG、点开却是 OK"。
    runDetect();

    // runDetect() 结尾会把下方切到分数页（单张检测就该看分数）。
    // 但从列表点进来的场景不一样：用户是在逐项翻列表，期望列表还在，
    // 按一下「下一项」就能继续看。切走等于把人踢出了流程——
    // 这是用户报的「打开大图对比后列表不见了」。
    // 所以在批量浏览进来时，把下方切回滑动列表页并恢复面板高度。
    if (!batchRows_.empty() && bottomStack_ && slideIndex_ >= 0) {
        bottomStack_->setCurrentIndex(1);
        if (leftSplit_) {
            const int total = leftSplit_->height() > 300 ? leftSplit_->height() : 860;
            leftSplit_->setSizes({std::max(220, total * 2 / 5), total * 3 / 5});
        }
    }
    if (tabs_) tabs_->setCurrentIndex(0);   // 切到检出项，划痕排在最前

    const QString s = QStringLiteral("%1  ·  %2  ·  划痕严重度 %3（跨 %4 字，深 %5）  ·  %6 ms")
                          .arg(q8(r.name))
                          .arg(q8(r.cls))
                          .arg(static_cast<int>(r.scratchS))
                          .arg(r.scratchChars)
                          .arg(static_cast<int>(r.scratchDepth))
                          .arg(static_cast<int>(r.elapsedMs));
    status_->setText(s);
    status_->setStyleSheet(r.ng ? QStringLiteral("color:#c62828; font-weight:bold;")
                                : QStringLiteral("color:#2e7d32; font-weight:bold;"));
}

void DefectViewWindow::updateStatus() {
    // 分类统计。划痕必须单独拎出来：
    // 它是客户确认的唯一缺陷类型，也是唯一会判 NG 的项；
    // 其他项（缺字/缺墨/多墨）只是旁证，即使有也不能当成判 NG 的理由。
    // 以前这里没有划痕分支，划痕被算进 else 的"多出字符"，
    // 于是状态栏显示"缺字 0 · 多墨 0 · 多出字符 8 → NG"，
    // 看不出 NG 到底是因为划痕——这是误导。
    int scratch = 0, miss = 0, extra = 0, blk = 0;
    double bestS = 0.0;
    QString bestBand;
    for (const auto& h : hits_) {
        if (h.kind.startsWith(QStringLiteral("划痕"))) {
            ++scratch;
            if (h.score > bestS) {
                bestS = h.score;
                bestBand = h.kind.mid(3);   // "划痕·确定" -> "确定"
            }
        } else if (h.kind == QStringLiteral("缺墨")) {
            ++miss;
        } else if (h.kind == QStringLiteral("多墨")) {
            ++extra;
        } else {
            ++blk;
        }
    }
    const bool ng = scratch > 0;
    if (gray_.empty()) {
        status_->setText(QStringLiteral("就绪：选择模型与产品图后点「开始检测」"));
        status_->setStyleSheet(QStringLiteral("color:#666;"));
        return;
    }

    QString msg;
    if (scratch > 0) {
        msg = QStringLiteral("%1   划痕 %2 处（最严重：%3，严重度 %4）   →   NG")
                  .arg(QFileInfo(imagePath_).fileName())
                  .arg(scratch)
                  .arg(bestBand)
                  .arg(static_cast<int>(bestS));
    } else {
        msg = QStringLiteral("%1   未发现划痕   →   OK").arg(QFileInfo(imagePath_).fileName());
    }
    // 旁证项单独一行，避免与判定依据混在一起被误读
    if (miss + extra + blk > 0) {
        msg += QStringLiteral("      旁证：缺墨 %1 · 多墨 %2 · 其他 %3（仅参考，不判 NG）")
                   .arg(miss)
                   .arg(extra)
                   .arg(blk);
    }
    if (!comparePath_.isEmpty()) {
        msg += QStringLiteral("     对比图：%1").arg(QFileInfo(comparePath_).fileName());
        status_->setToolTip(comparePath_);
    }
    status_->setText(msg);
    status_->setStyleSheet(ng ? QStringLiteral("color:#c62828; font-weight:bold;")
                              : QStringLiteral("color:#2e7d32; font-weight:bold;"));
}

void DefectViewWindow::zoomIn() {
    imgView_->setZoom(imgView_->zoom() * 1.25);
    if (zoomSpin_) zoomSpin_->setValue(static_cast<int>(imgView_->zoom() * 100));
}

void DefectViewWindow::zoomOut() {
    imgView_->setZoom(imgView_->zoom() / 1.25);
    if (zoomSpin_) zoomSpin_->setValue(static_cast<int>(imgView_->zoom() * 100));
}

// ---------------------------------------------------------------- 灵敏度持久化

/// 配置文件路径：与模型同目录，便于随模型一起分发
static QString sensitivityPath(const QString& modelDir) {
    return QDir(modelDir).absoluteFilePath(QStringLiteral("detect_config.json"));
}

void DefectViewWindow::saveSensitivity() {
    const QString path = sensitivityPath(modelDir_);
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        status_->setText(QStringLiteral("参数保存失败：无法写入 %1").arg(path));
        status_->setStyleSheet(QStringLiteral("color:#c62828;"));
        return;
    }

    QTextStream ts(&f);
    ts.setEncoding(QStringConverter::Utf8);
    ts << "{\n";
    ts << "  \"rowCount\": " << (sens_.checkRowCount ? "true" : "false") << ",\n";
    ts << "  \"scratch\": " << (sens_.checkScratch ? "true" : "false") << ",\n";
    ts << "  \"scratchBridge\": " << sens_.scratchBridge << ",\n";
    ts << "  \"scratchMinLen\": " << sens_.scratchMinLen << ",\n";
    ts << "  \"scratchMinCov\": " << sens_.scratchMinCov << ",\n";
    ts << "  \"scratchMinEnergy\": " << sens_.scratchMinEnergy << ",\n";
    ts << "  \"scratchMinComposite\": " << sens_.scratchMinComposite << ",\n";
    ts << "  \"scratchCutCertain\": " << sens_.scratchCutCertain << ",\n";
    ts << "  \"scratchCutObvious\": " << sens_.scratchCutObvious << ",\n";
    ts << "  \"scratchCutSuspect\": " << sens_.scratchCutSuspect << ",\n";
    ts << "  \"scratchDepthReach\": " << sens_.scratchDepthReach << ",\n";
    ts << "  \"scratchMinGray\": " << sens_.scratchMinGray << ",\n";
    ts << "  \"dirt\": " << (sens_.checkDirt ? "true" : "false") << ",\n";
    ts << "  \"dirtMinArea\": " << sens_.dirtMinArea << ",\n";
    ts << "  \"appendage\": " << (sens_.checkAppendage ? "true" : "false") << ",\n";
    ts << "  \"appendMinArea\": " << sens_.appendMinArea << ",\n";
    ts << "  \"appendMaxArea\": " << sens_.appendMaxArea << ",\n";
    ts << "  \"appendMinElong\": " << sens_.appendMinElong << ",\n";
    ts << "  \"perCharAlign\": " << (sens_.perCharAlign ? "true" : "false") << ",\n";
    ts << "  \"localMiss\": " << (sens_.checkLocalMiss ? "true" : "false") << ",\n";
    ts << "  \"localMissRatio\": " << sens_.localMissRatio << ",\n";
    ts << "  \"charLevel\": " << (sens_.checkCharLevel ? "true" : "false") << ",\n";
    ts << "  \"multiTmpl\": " << (sens_.checkMultiTmpl ? "true" : "false") << ",\n";
    ts << "  \"multiDecideNg\": " << (sens_.multiDecideNg ? "true" : "false") << ",\n";
    ts << "  \"multiRatio\": " << sens_.multiRatio << ",\n";
    ts << "  \"multiWarn\": " << sens_.multiWarn << "\n";
    ts << "}\n";
    f.close();

    status_->setText(QStringLiteral("灵敏度已保存：%1").arg(QFileInfo(path).fileName()));
    status_->setStyleSheet(QStringLiteral("color:#2e7d32;"));
}

void DefectViewWindow::loadSensitivity() {
    const QString path = sensitivityPath(modelDir_);
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;   // 没有配置就用默认值

    const QString text = QString::fromUtf8(f.readAll());
    f.close();

    // 简易解析：逐行找 "key": value
    const auto readNum = [&text](const QString& key, double def) -> double {
        const QRegularExpression re(QStringLiteral("\"%1\"\\s*:\\s*(-?[0-9.]+)").arg(key));
        const auto m = re.match(text);
        return m.hasMatch() ? m.captured(1).toDouble() : def;
    };
    const auto readBool = [&text](const QString& key, bool def) -> bool {
        const QRegularExpression re(QStringLiteral("\"%1\"\\s*:\\s*(true|false)").arg(key));
        const auto m = re.match(text);
        if (!m.hasMatch()) return def;
        return m.captured(1) == QStringLiteral("true");
    };

    const Sensitivity d;
    sens_.checkRowCount = readBool(QStringLiteral("rowCount"), d.checkRowCount);
    sens_.checkScratch = readBool(QStringLiteral("scratch"), d.checkScratch);
    sens_.scratchBridge =
        static_cast<int>(readNum(QStringLiteral("scratchBridge"), d.scratchBridge));
    sens_.scratchMinLen = readNum(QStringLiteral("scratchMinLen"), d.scratchMinLen);
    sens_.scratchMinCov = readNum(QStringLiteral("scratchMinCov"), d.scratchMinCov);
    sens_.scratchMinEnergy = readNum(QStringLiteral("scratchMinEnergy"), d.scratchMinEnergy);
    sens_.scratchMinGray = readNum(QStringLiteral("scratchMinGray"), d.scratchMinGray);
    sens_.scratchMinComposite =
        readNum(QStringLiteral("scratchMinComposite"), d.scratchMinComposite);
    sens_.scratchCutCertain = readNum(QStringLiteral("scratchCutCertain"), d.scratchCutCertain);
    sens_.scratchCutObvious = readNum(QStringLiteral("scratchCutObvious"), d.scratchCutObvious);
    sens_.scratchCutSuspect = readNum(QStringLiteral("scratchCutSuspect"), d.scratchCutSuspect);
    sens_.scratchDepthReach =
        static_cast<int>(readNum(QStringLiteral("scratchDepthReach"), d.scratchDepthReach));
    sens_.checkDirt = readBool(QStringLiteral("dirt"), d.checkDirt);
    sens_.dirtMinArea = static_cast<int>(readNum(QStringLiteral("dirtMinArea"), d.dirtMinArea));
    sens_.checkAppendage = readBool(QStringLiteral("appendage"), d.checkAppendage);
    sens_.appendMinArea = static_cast<int>(readNum(QStringLiteral("appendMinArea"), d.appendMinArea));
    sens_.appendMaxArea = static_cast<int>(readNum(QStringLiteral("appendMaxArea"), d.appendMaxArea));
    sens_.appendMinElong = readNum(QStringLiteral("appendMinElong"), d.appendMinElong);
    sens_.perCharAlign = readBool(QStringLiteral("perCharAlign"), d.perCharAlign);
    sens_.checkLocalMiss = readBool(QStringLiteral("localMiss"), d.checkLocalMiss);
    sens_.localMissRatio = readNum(QStringLiteral("localMissRatio"), d.localMissRatio);
    sens_.checkCharLevel = readBool(QStringLiteral("charLevel"), d.checkCharLevel);
    sens_.checkMultiTmpl = readBool(QStringLiteral("multiTmpl"), d.checkMultiTmpl);
    sens_.multiDecideNg = readBool(QStringLiteral("multiDecideNg"), d.multiDecideNg);
    sens_.multiRatio = readNum(QStringLiteral("multiRatio"), d.multiRatio);
    sens_.multiWarn = readNum(QStringLiteral("multiWarn"), d.multiWarn);

    if (charLevelChk_) charLevelChk_->setChecked(sens_.checkCharLevel);
}

void DefectViewWindow::openSensitivity() {
    QDialog dlg(this);
    dlg.setWindowTitle(QStringLiteral("检测灵敏度设置"));
    auto* root = new QVBoxLayout(&dlg);

    auto* tip = new QLabel(QStringLiteral(
        "数值越小 = 判得越严（检出多、误报也多）；数值越大 = 判得越宽（误报少、可能漏检）。\n"
        "这批 OK 样本自带刻印抖动与轻微划痕，具体切点建议用客户的真实 NG 样本标定。"));
    tip->setWordWrap(true);
    tip->setStyleSheet(QStringLiteral("color:#666; padding:4px;"));
    root->addWidget(tip);

    // ---- 各检测项
    auto* g1 = new QGroupBox(QStringLiteral("整字数量校验（最可靠，建议保持开启）"));
    auto* f1 = new QFormLayout(g1);
    auto* rowChk = new QCheckBox(QStringLiteral("启用"));
    rowChk->setChecked(sens_.checkRowCount);
    f1->addRow(rowChk);
    root->addWidget(g1);

    auto* g2 = new QGroupBox(QStringLiteral("划痕"));
    auto* f2 = new QFormLayout(g2);
    auto* scrChk = new QCheckBox(QStringLiteral("启用"));
    scrChk->setChecked(sens_.checkScratch);
    f2->addRow(scrChk);
    auto* scrBridge = new QSpinBox;
    scrBridge->setRange(0, 101);
    scrBridge->setSingleStep(2);
    scrBridge->setValue(sens_.scratchBridge);
    scrBridge->setSuffix(QStringLiteral(" px"));
    scrBridge->setToolTip(QStringLiteral(
        "划痕跨字符重连核大小。减去模型笔画会把一条跨字划痕切成碎片，\n"
        "碎片各自都过不了长度门槛导致漏检。此值把断口接回去。\n"
        "0 = 关闭。"));
    f2->addRow(QStringLiteral("跨字符重连"), scrBridge);
    auto* scrLen = new QSpinBox;
    scrLen->setRange(20, 1000);
    scrLen->setValue(static_cast<int>(sens_.scratchMinLen));
    scrLen->setSuffix(QStringLiteral(" px"));
    f2->addRow(QStringLiteral("最短长度"), scrLen);
    auto* scrCov = new QSpinBox;
    scrCov->setRange(5, 95);
    scrCov->setValue(static_cast<int>(sens_.scratchMinCov * 100));
    scrCov->setSuffix(QStringLiteral(" %"));
    f2->addRow(QStringLiteral("最短覆盖率"), scrCov);
    auto* scrEng = new QSpinBox;
    scrEng->setRange(0, 2000000);
    scrEng->setSingleStep(10000);
    scrEng->setValue(static_cast<int>(sens_.scratchMinEnergy));
    scrEng->setToolTip(QStringLiteral(
        "仅为去噪门槛，默认 0（不筛）。\n"
        "原值 30 万是按 84 张 OK 标的，实测会让 8 张真 NG 全部漏检：\n"
        "真划痕能量只有 1.7 万~24.4 万，标定集里没出现过。\n"
        "判别职责已交给下面的「严重度门槛」。"));
    f2->addRow(QStringLiteral("能量门槛（去噪）"), scrEng);
    auto* scrGray = new QSpinBox;
    scrGray->setRange(10, 95);
    scrGray->setValue(static_cast<int>(sens_.scratchMinGray * 100));
    scrGray->setSuffix(QStringLiteral(" %"));
    f2->addRow(QStringLiteral("灰度门槛（已停用）"), scrGray);

    // 严重度 = 深度 × 跨越字数。这一项是现在的判定主门槛。
    auto* scrComp = new QSpinBox;
    scrComp->setRange(0, 5000);
    scrComp->setSingleStep(50);
    scrComp->setValue(static_cast<int>(sens_.scratchMinComposite));
    scrComp->setToolTip(QStringLiteral(
        "严重度门槛 = 深度 × 跨越字数，低于此值不报告。\n"
        "\n"
        "实测全量（84 张 OK + 7 张 NG，均为 v00）：\n"
        "  NG  2289 2256 2209 2040 1321 1219（0110 只有 73）\n"
        "  OK  1430 1182 1080 873 856 847 …\n"
        "  前三张分别是 0109_RT / 0109_FT / 0101_FT。\n"
        "  其中 0101 客户判定确实是缺陷（正确检出），0109 两张是真误报。\n"
        "\n"
        "  注意 0112_RT(1219) 比 0109_RT(1430) 还低，两者是真的重叠，\n"
        "  没有任何门槛能同时做到不漏 0112 又不报 0109。0109 那两道痕\n"
        "  深度 477/591、跨 3 个字，和真划痕同量级，是痕迹本身像。\n"
        "\n"
        "  1000 → 真 NG 6/6 全报，OK 误报 2 张（0109 的两次拍摄）← 默认\n"
        "  1500 → 零误报，但 0112 两张漏检，真 NG 只剩 4/6\n"
        "\n"
        "切点默认 >=1900 确定 / >=1300 明显 / >=1000 可疑，可在下面逐档调整。"));
    f2->addRow(QStringLiteral("严重度门槛"), scrComp);

    auto* scrCut3 = new QSpinBox;
    scrCut3->setRange(0, 20000);
    scrCut3->setSingleStep(100);
    scrCut3->setValue(static_cast<int>(sens_.scratchCutCertain));
    scrCut3->setToolTip(QStringLiteral("严重度 ≥ 此值 = 确定 NG。界面上每条缺陷旁会写出调到多少能升级。"));
    f2->addRow(QStringLiteral("切点·确定"), scrCut3);
    auto* scrCut2 = new QSpinBox;
    scrCut2->setRange(0, 20000);
    scrCut2->setSingleStep(100);
    scrCut2->setValue(static_cast<int>(sens_.scratchCutObvious));
    scrCut2->setToolTip(QStringLiteral("严重度 ≥ 此值 = 明显。"));
    f2->addRow(QStringLiteral("切点·明显"), scrCut2);
    auto* scrCut1 = new QSpinBox;
    scrCut1->setRange(0, 20000);
    scrCut1->setSingleStep(50);
    scrCut1->setValue(static_cast<int>(sens_.scratchCutSuspect));
    scrCut1->setToolTip(QStringLiteral("严重度 ≥ 此值 = 可疑；低于则只是轻微。"));
    f2->addRow(QStringLiteral("切点·可疑"), scrCut1);
    auto* scrReach = new QSpinBox;
    scrReach->setRange(6, 30);
    scrReach->setValue(sens_.scratchDepthReach);
    scrReach->setSuffix(QStringLiteral(" px"));
    scrReach->setToolTip(QStringLiteral("深度测量的垂直剖面半长。要大于划痕宽度、小于字宽。"));
    f2->addRow(QStringLiteral("深度剖面半长"), scrReach);
    root->addWidget(g2);

    auto* g3 = new QGroupBox(QStringLiteral("污损"));
    auto* f3 = new QFormLayout(g3);
    auto* dirtChk = new QCheckBox(QStringLiteral("启用"));
    dirtChk->setChecked(sens_.checkDirt);
    f3->addRow(dirtChk);
    auto* dirtArea = new QSpinBox;
    dirtArea->setRange(20, 5000);
    dirtArea->setValue(sens_.dirtMinArea);
    dirtArea->setSuffix(QStringLiteral(" px"));
    f3->addRow(QStringLiteral("最小面积"), dirtArea);
    root->addWidget(g3);

    auto* g4 = new QGroupBox(QStringLiteral("溢料"));
    auto* f4 = new QFormLayout(g4);
    auto* appChk = new QCheckBox(QStringLiteral("启用"));
    appChk->setChecked(sens_.checkAppendage);
    f4->addRow(appChk);
    auto* appMin = new QSpinBox;
    appMin->setRange(20, 2000);
    appMin->setValue(sens_.appendMinArea);
    f4->addRow(QStringLiteral("最小面积"), appMin);
    auto* appMax = new QSpinBox;
    appMax->setRange(500, 20000);
    appMax->setValue(sens_.appendMaxArea);
    f4->addRow(QStringLiteral("最大面积"), appMax);
    auto* appElong = new QSpinBox;
    appElong->setRange(10, 100);
    appElong->setValue(static_cast<int>(sens_.appendMinElong * 10));
    appElong->setSuffix(QStringLiteral(" /10"));
    f4->addRow(QStringLiteral("最小长宽比"), appElong);
    root->addWidget(g4);

    auto* g5 = new QGroupBox(QStringLiteral("字符级判据（缺墨/多墨）"));
    auto* f5 = new QFormLayout(g5);
    auto* charChk = new QCheckBox(QStringLiteral("启用"));
    charChk->setChecked(sens_.checkCharLevel);
    f5->addRow(charChk);
    auto* warn = new QLabel(QStringLiteral("注意：实测 84 张 OK 样本中有 33 张因此误报，\n"
                                           "原因是背景划痕落在字符框内时无法与笔画缺陷区分。"));
    warn->setStyleSheet(QStringLiteral("color:#c62828;"));
    f5->addRow(warn);
    root->addWidget(g5);

    auto* btns = new QHBoxLayout;
    auto* resetBtn = new QPushButton(QStringLiteral("恢复默认"));
    btns->addWidget(resetBtn);
    btns->addStretch();
    auto* okBtn = new QPushButton(QStringLiteral("确定"));
    okBtn->setDefault(true);
    btns->addWidget(okBtn);
    auto* cancelBtn = new QPushButton(QStringLiteral("取消"));
    btns->addWidget(cancelBtn);
    root->addLayout(btns);

    connect(okBtn, &QPushButton::clicked, &dlg, &QDialog::accept);
    connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    connect(resetBtn, &QPushButton::clicked, &dlg, [&]() {
        const Sensitivity d;
        rowChk->setChecked(d.checkRowCount);
        scrChk->setChecked(d.checkScratch);
        scrBridge->setValue(d.scratchBridge);
        scrLen->setValue(static_cast<int>(d.scratchMinLen));
        scrCov->setValue(static_cast<int>(d.scratchMinCov * 100));
        scrEng->setValue(static_cast<int>(d.scratchMinEnergy));
        scrGray->setValue(static_cast<int>(d.scratchMinGray * 100));
        scrComp->setValue(static_cast<int>(d.scratchMinComposite));
        scrCut3->setValue(static_cast<int>(d.scratchCutCertain));
        scrCut2->setValue(static_cast<int>(d.scratchCutObvious));
        scrCut1->setValue(static_cast<int>(d.scratchCutSuspect));
        scrReach->setValue(d.scratchDepthReach);
        dirtChk->setChecked(d.checkDirt);
        dirtArea->setValue(d.dirtMinArea);
        appChk->setChecked(d.checkAppendage);
        appMin->setValue(d.appendMinArea);
        appMax->setValue(d.appendMaxArea);
        appElong->setValue(static_cast<int>(d.appendMinElong * 10));
        charChk->setChecked(d.checkCharLevel);
    });

    if (dlg.exec() != QDialog::Accepted) return;

    sens_.checkRowCount = rowChk->isChecked();
    sens_.checkScratch = scrChk->isChecked();
    sens_.scratchBridge = scrBridge->value();
    sens_.scratchMinLen = scrLen->value();
    sens_.scratchMinCov = scrCov->value() / 100.0;
    sens_.scratchMinEnergy = scrEng->value();
    sens_.scratchMinGray = scrGray->value() / 100.0;
    sens_.scratchMinComposite = scrComp->value();
    sens_.scratchCutCertain = scrCut3->value();
    sens_.scratchCutObvious = scrCut2->value();
    sens_.scratchCutSuspect = scrCut1->value();
    sens_.scratchDepthReach = scrReach->value();
    sens_.checkDirt = dirtChk->isChecked();
    sens_.dirtMinArea = dirtArea->value();
    sens_.checkAppendage = appChk->isChecked();
    sens_.appendMinArea = appMin->value();
    sens_.appendMaxArea = appMax->value();
    sens_.appendMinElong = appElong->value() / 10.0;
    sens_.checkCharLevel = charChk->isChecked();

    // 同步到界面上的复选框
    if (charLevelChk_) charLevelChk_->setChecked(sens_.checkCharLevel);

    status_->setText(QStringLiteral("灵敏度已更新，重新点「开始检测」生效"));
}

void DefectViewWindow::fitView() {
    imgView_->fitToWindow();
    if (zoomSpin_) zoomSpin_->setValue(static_cast<int>(imgView_->zoom() * 100));
}
