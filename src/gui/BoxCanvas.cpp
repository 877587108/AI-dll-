#include "gui/BoxCanvas.hpp"

#include <QGraphicsEllipseItem>
#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include <QGraphicsSimpleTextItem>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QScrollBar>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace {

constexpr double kMinBoxPx = 6.0;   // 小于此尺寸不认为是有效框

}  // namespace

BoxCanvas::BoxCanvas(QWidget* parent) : QGraphicsView(parent) {
    scene_ = new QGraphicsScene(this);
    setScene(scene_);
    setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    setBackgroundBrush(QColor(34, 36, 40));
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setFocusPolicy(Qt::StrongFocus);
    // 拖拽模式设成 NoDrag：我们要自己处理左键拖拽画框，
    // 平移改用中键或按住空格（这里简化为中键）。
    setDragMode(QGraphicsView::NoDrag);
}

void BoxCanvas::setImage(const QImage& img) {
    image_ = img;
    boxes_.clear();
    selectedId_ = -1;
    nextId_ = 1;
    rebuildScene();
}

void BoxCanvas::clearImage() {
    image_ = QImage();
    boxes_.clear();
    selectedId_ = -1;
    rebuildScene();
}

void BoxCanvas::setBoxes(const QList<EditableBox>& boxes) {
    boxes_ = boxes;
    // 保证 id 不冲突
    int mx = 0;
    for (const auto& b : boxes_) mx = std::max(mx, b.id);
    nextId_ = mx + 1;
    selectedId_ = -1;
    rebuildScene();
}

int BoxCanvas::addBox(const QRectF& r, const QString& label, const QColor& col,
                      BoxShape shape) {
    EditableBox b;
    b.id = nextId_++;
    b.rect = r;
    b.label = label;
    b.color = col;
    b.shape = shape;
    boxes_.append(b);
    rebuildScene();
    return b.id;
}

void BoxCanvas::removeBox(int id) {
    for (int i = 0; i < boxes_.size(); ++i) {
        if (boxes_[i].id == id) {
            boxes_.removeAt(i);
            if (selectedId_ == id) selectedId_ = -1;
            rebuildScene();
            emit selectionChanged(selectedId_);
            return;
        }
    }
}

void BoxCanvas::clearBoxes() {
    boxes_.clear();
    selectedId_ = -1;
    rebuildScene();
    emit selectionChanged(-1);
}

int BoxCanvas::selectedId() const { return selectedId_; }

void BoxCanvas::rebuildScene() {
    scene_->clear();
    pixItem_ = nullptr;

    if (!image_.isNull()) {
        pixItem_ = scene_->addPixmap(QPixmap::fromImage(image_));
        scene_->setSceneRect(pixItem_->boundingRect());
    } else {
        scene_->setSceneRect(QRectF(0, 0, 1, 1));
    }

    // 画框：半透明填充 + 彩色边 + 左上角标签。
    // 椭圆/圆画成曲线而不是矩形 —— 一眼能看出这个框包的是"一个环"，
    // 而不是"一排字"，两者用的判据完全不同。
    for (const EditableBox& b : boxes_) {
        const bool sel = (b.id == selectedId_);
        QPen pen(b.color, sel ? 3.0 : 2.0);
        if (sel) pen.setStyle(Qt::SolidLine);
        const QBrush fill(QColor(b.color.red(), b.color.green(), b.color.blue(),
                                 sel ? 60 : 28));
        QGraphicsItem* item = nullptr;
        if (b.shape == BoxShape::Ellipse || b.shape == BoxShape::Circle) {
            // 用 QPainterPath 画椭圆，支持任意角度。
            // 不能用 addEllipse 再 setRotation —— 那是绕图元原点转，
            // 得先把原点挪到椭圆中心，容易算错。
            QRectF er = b.rect;
            if (b.shape == BoxShape::Circle) {
                const double d = std::min(er.width(), er.height());
                const QPointF c = er.center();
                er = QRectF(c.x() - d / 2, c.y() - d / 2, d, d);
            }
            const QPointF c = er.center();
            const double ra = er.width() / 2.0, rb = er.height() / 2.0;
            const double kPi = 3.14159265358979323846;   // 不用 M_PI：MSVC 默认不定义它
            const double ang = b.angleDeg * kPi / 180.0;
            QPainterPath path;
            const int N = 96;
            for (int i = 0; i <= N; ++i) {
                const double th = 2.0 * kPi * i / N;
                const double u = ra * std::cos(th), v = rb * std::sin(th);
                const double px = c.x() + u * std::cos(ang) - v * std::sin(ang);
                const double py = c.y() + u * std::sin(ang) + v * std::cos(ang);
                if (i == 0) path.moveTo(px, py);
                else path.lineTo(px, py);
            }
            path.closeSubpath();
            item = scene_->addPath(path, pen, fill);
        } else {
            item = scene_->addRect(b.rect, pen, fill);
        }
        if (item) item->setZValue(10);
        if (!b.label.isEmpty()) {
            auto* txt = scene_->addSimpleText(b.label);
            txt->setBrush(b.color);
            QFont f = txt->font();
            f.setPointSize(9);
            f.setBold(true);
            txt->setFont(f);
            txt->setPos(b.rect.x() + 2, std::max(0.0, b.rect.y() - 16));
            txt->setZValue(11);
        }
    }

    // 正在拖拽的临时框：按当前图形画，松手前就能看见会生成什么
    if (drawing_) {
        QRectF r = QRectF(drawStart_, drawCur_).normalized();
        const QPen tmpPen(QColor(255, 220, 0), 2, Qt::DashLine);
        if (newShape_ == BoxShape::Ellipse || newShape_ == BoxShape::Circle) {
            if (newShape_ == BoxShape::Circle) {
                const double d = std::min(r.width(), r.height());
                const QPointF c = r.center();
                r = QRectF(c.x() - d / 2, c.y() - d / 2, d, d);
            }
            auto* tmp = scene_->addEllipse(r, tmpPen);
            if (tmp) tmp->setZValue(20);
        } else {
            auto* tmp = scene_->addRect(r, tmpPen);
            if (tmp) tmp->setZValue(20);
        }
    }
}

QRectF BoxCanvas::toImageRect(const QRectF& sceneRect) const {
    if (image_.isNull()) return QRectF();
    // scene 坐标在 pixmap 放置后与图像坐标一致（场景原点在图像左上角）
    QRectF r = sceneRect;
    r = r.intersected(QRectF(0, 0, image_.width(), image_.height()));
    return r;
}

int BoxCanvas::hitTest(const QPointF& pt) const {
    // 从后往前找（后画的在上层）
    for (int i = boxes_.size() - 1; i >= 0; --i) {
        if (boxes_[i].rect.contains(pt)) return boxes_[i].id;
    }
    return -1;
}

void BoxCanvas::mousePressEvent(QMouseEvent* e) {
    if (image_.isNull()) {
        QGraphicsView::mousePressEvent(e);
        return;
    }
    const QPointF sp = mapToScene(e->pos());
    const QPointF ip = sp;

    if (e->button() == Qt::MiddleButton) {
        // 中键拖拽平移
        QGraphicsView::mousePressEvent(e);
        return;
    }

    if (e->button() == Qt::LeftButton) {
        // 先判断是否点中了已有框 → 选中
        const int hit = hitTest(ip);
        if (hit >= 0) {
            selectedId_ = hit;
            for (auto& b : boxes_) b.selected = (b.id == hit);
            rebuildScene();
            emit selectionChanged(hit);
            setFocus();
            return;
        }
        // 未点中：若允许画框，开始拖拽
        if (drawEnabled_) {
            drawing_ = true;
            drawStart_ = ip;
            drawCur_ = ip;
            selectedId_ = -1;
            rebuildScene();
            emit selectionChanged(-1);
        }
        setFocus();
        return;
    }
    QGraphicsView::mousePressEvent(e);
}

void BoxCanvas::mouseMoveEvent(QMouseEvent* e) {
    if (drawing_) {
        drawCur_ = mapToScene(e->pos());
        rebuildScene();
        return;
    }
    QGraphicsView::mouseMoveEvent(e);
}

void BoxCanvas::mouseReleaseEvent(QMouseEvent* e) {
    if (drawing_ && e->button() == Qt::LeftButton) {
        drawing_ = false;
        QRectF r = QRectF(drawStart_, drawCur_).normalized();
        r = toImageRect(r);
        drawStart_ = drawCur_ = QPointF();
        rebuildScene();
        if (r.width() >= kMinBoxPx && r.height() >= kMinBoxPx) {
            emit boxCreated(r);
        }
        return;
    }
    QGraphicsView::mouseReleaseEvent(e);
}

void BoxCanvas::wheelEvent(QWheelEvent* e) {
    if (image_.isNull()) {
        QGraphicsView::wheelEvent(e);
        return;
    }
    const double factor = (e->angleDelta().y() > 0) ? 1.15 : 1.0 / 1.15;
    const double next = zoom_ * factor;
    if (next < 0.05 || next > 40.0) return;
    zoom_ = next;
    scale(factor, factor);
}

void BoxCanvas::keyPressEvent(QKeyEvent* e) {
    if ((e->key() == Qt::Key_Delete || e->key() == Qt::Key_Backspace) && selectedId_ >= 0) {
        emit boxDeleteRequested(selectedId_);
        return;
    }
    QGraphicsView::keyPressEvent(e);
}

void BoxCanvas::zoomIn() {
    if (image_.isNull()) return;
    zoom_ *= 1.25;
    scale(1.25, 1.25);
}

void BoxCanvas::zoomOut() {
    if (image_.isNull()) return;
    zoom_ /= 1.25;
    scale(1.0 / 1.25, 1.0 / 1.25);
}

void BoxCanvas::fitToWindow() {
    if (image_.isNull() || !pixItem_) return;
    fitInView(pixItem_, Qt::KeepAspectRatio);
    zoom_ = transform().m11();
}
