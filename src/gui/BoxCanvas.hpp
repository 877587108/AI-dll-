#pragma once

// 可交互画框控件：显示一张图，支持鼠标拖拽画矩形框、选中、删除
//
// 设计要点：
//   · 坐标统一用「图像像素坐标」，缩放/平移只影响显示，不影响数据
//   · 每个框带一个 id 和类别（颜色区分），便于按区域归属
//   · 不直接持有业务逻辑，只负责画和改，业务由外部回调处理

#include <QGraphicsPixmapItem>
#include <QGraphicsView>
#include <QImage>
#include <QList>
#include <QString>

class QGraphicsScene;
class QGraphicsRectItem;
/// 框的图形
///
/// 字符区和图案区用不同的图形和不同的判据：
///   矩形   字符区：切字、逐字符比对墨量
///   椭圆   图案区（如 logo 外环）：只查这条闭合曲线有没有断
///   圆     图案区（定位点、圆形标记）：同上
/// 显示上椭圆/圆画成曲线，这样一眼能看出"这个框包的是环，不是一排字"。
enum class BoxShape { Rect = 0, Ellipse = 1, Circle = 2 };

/// 一个可编辑的框
struct EditableBox {
    int id = 0;
    QRectF rect;        // 图像坐标（椭圆/圆是外接框）
    QString label;      // 显示文字（如 "0"、"line1#2"）
    QColor color;       // 边框颜色
    bool selected = false;
    BoxShape shape = BoxShape::Rect;
    // 椭圆/圆的长轴角度（度，逆时针为正）。
    // 斜着的环（SC 长轴约 -13°）必须能显示成斜椭圆，
    // 否则用户看不到自己画的椭圆和环对不对得上。
    double angleDeg = 0.0;
};

class BoxCanvas : public QGraphicsView {
    Q_OBJECT
public:
    explicit BoxCanvas(QWidget* parent = nullptr);

    void setImage(const QImage& img);
    void clearImage();

    void setBoxes(const QList<EditableBox>& boxes);
    const QList<EditableBox>& boxes() const { return boxes_; }

    /// 新增/删除后由外部决定是否 keep，返回新框 id
    int addBox(const QRectF& r, const QString& label, const QColor& col,
               BoxShape shape = BoxShape::Rect);
    void removeBox(int id);
    void clearBoxes();

    /// 拖拽时生成哪种图形。只影响新建的框，不影响已有的。
    void setNewShape(BoxShape s) { newShape_ = s; }
    BoxShape newShape() const { return newShape_; }

    /// 当前选中框（没有返回 -1）
    int selectedId() const;

    void setDrawEnabled(bool on) { drawEnabled_ = on; }
    bool drawEnabled() const { return drawEnabled_; }

    /// 缩放控制
    void zoomIn();
    void zoomOut();
    void fitToWindow();

signals:
    /// 用户画完一个新框（图像坐标）。shape 是拖拽时选的图形。
    void boxCreated(const QRectF& r);
    /// 选中项变化
    void selectionChanged(int id);
    /// 用户按 Delete 删除选中框
    void boxDeleteRequested(int id);

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private:
    void rebuildScene();
    QRectF toImageRect(const QRectF& sceneRect) const;
    int hitTest(const QPointF& imagePt) const;

    QGraphicsScene* scene_ = nullptr;
    QGraphicsPixmapItem* pixItem_ = nullptr;
    QImage image_;
    QList<EditableBox> boxes_;
    int nextId_ = 1;
    int selectedId_ = -1;

    bool drawEnabled_ = true;
    bool drawing_ = false;
    QPointF drawStart_;
    QPointF drawCur_;
    BoxShape newShape_ = BoxShape::Rect;   // 拖拽时生成哪种图形

    double zoom_ = 1.0;
};
