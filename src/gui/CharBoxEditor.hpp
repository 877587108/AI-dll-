#pragma once

// 选区编辑对话框（消费 chardet 模块的自包含模型）
//
// 工作流：
//   1. 模块训练产出模型（频率图 + 自动切分的区域）
//   2. 本对话框加载该模型，显示频率图
//   3. 用户可增删区域、重新切分字符
//   4. 保存即写回模块模型目录

#include <QDialog>
#include <QString>

#include <vector>

#include <opencv2/core.hpp>

#include "chardet.hpp"
#include "gui/BoxCanvas.hpp"  // BoxShape is part of the editor's public state

class BoxCanvas;
class QLabel;
class QDoubleSpinBox;
class QListWidget;
class QPushButton;
class QRadioButton;
class QSpinBox;

class RegionEditor : public QDialog {
    Q_OBJECT
public:
    explicit RegionEditor(const QString& modelDir, int view, QWidget* parent = nullptr);

private slots:
    void onBoxCreated(const QRectF& r);
    void onSelectionChanged(int id);
    void onDeleteRequested(int id);
    void deleteCurrent();
    void clearAll();
    void doSplit();
    void save();
    void reload();
    /// 角度输入框被改动：若当前选中的是一个图案区，立刻回写并重画
    void onAngleEdited(double deg);

private:
    enum class Kind { Region = 0, CharBox = 1 };

    void buildUi();
    void refreshCanvas();
    void refreshList();
    /// 当前选中的图形（决定下一次拖拽生成什么）
    BoxShape selectedShape() const;

    QString modelDir_;
    int view_ = 0;
    QImage refQImg_;

    chardet::Model model_;

    BoxCanvas* canvas_ = nullptr;
    QListWidget* list_ = nullptr;
    QLabel* infoLabel_ = nullptr;
    QLabel* hintLabel_ = nullptr;
    QSpinBox* colThrSpin_ = nullptr;
    QSpinBox* gapRatioSpin_ = nullptr;
    QRadioButton* shapeRect_ = nullptr;
    QRadioButton* shapeEllipse_ = nullptr;
    QRadioButton* shapeCircle_ = nullptr;
    QDoubleSpinBox* angleSpin_ = nullptr;

    struct CanvasItem {
        Kind kind = Kind::Region;
        int idx = -1;      // 区域下标
        int subIdx = -1;   // 字符下标（Kind::CharBox 时有效）
    };
    std::vector<CanvasItem> items_;
};
