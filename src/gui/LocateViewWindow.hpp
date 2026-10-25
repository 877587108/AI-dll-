#pragma once

// LocateViewWindow —— 定位算法可视化调试平台
//
// 职责（只渲染与交互，不重算定位）：
//   1. 加载定位插件 DLL + 模型 + 检测图，跑一次定位
//   2. 左侧显示检测图（叠加定位框），右侧显示模板图（叠加字符框）
//   3. 鼠标悬停在检测图上时，浮出预览小窗，显示该点对应模板图的
//      N×N 区域（默认 30×30，可调），中心 3×3 画提示框
//   4. 参数面板：可改插件参数并重跑
//   5. 结果列表：每个字符的定位框、置信度、偏差
//
// 依赖：Qt Widgets + OpenCV + locate_plugin（不依赖 markloc_core 的检测逻辑）

#include <memory>
#include <string>
#include <vector>

#include <QWidget>

#include <opencv2/core.hpp>

#include "locate_plugin/locate_plugin.hpp"
#include "locate_plugin/plugin_loader.hpp"

class QLabel;
class QScrollArea;
class QLineEdit;
class QSpinBox;
class QTableWidget;
class QPushButton;
class QSplitter;

namespace locateview {

// 从模型 regions.txt 读出的字符框（模型坐标）
struct ModelBox {
    int x = 0, y = 0, w = 0, h = 0;
    int index = 0;
};

/// 可悬停的图像视图基类：显示一张图 + 缩放 + 捕获鼠标移动发信号
///
/// 子类只需重写 drawOverlays() 画各自的框（定位框 / 字符框）。
/// 鼠标、缩放、坐标换算、信号都在基类里，避免两份实现分叉。
class ImageHoverWidget : public QWidget {
    Q_OBJECT
public:
    explicit ImageHoverWidget(QWidget* parent = nullptr);

    /// 设置底层图（灰度或 BGR），内部转成 QImage
    void setSource(const cv::Mat& img);

    double zoom() const { return zoom_; }
    void setZoom(double z);
    void fitToView();

    /// 当前显示的图像尺寸（原图像素）
    int imageW() const { return imgW_; }
    int imageH() const { return imgH_; }

signals:
    /// 鼠标在原图上的像素坐标（-1,-1 表示移出图外）
    void mouseMovedImage(int x, int y);
    /// 鼠标在屏幕上的全局坐标（用于定位浮窗）
    void mouseMovedGlobal(const QPoint& g);

protected:
    void paintEvent(QPaintEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void leaveEvent(QEvent*) override;
    void wheelEvent(QWheelEvent*) override;

    /// 子类在这里画覆盖层（框、标签等）。
    /// offX/offY 是原图左上角在 widget 上的位置，zoom 是当前缩放。
    virtual void drawOverlays(QPainter& p, double offX, double offY) = 0;

    /// 从 widget 坐标换算到原图坐标（图外返回 -1,-1）
    QPoint widgetToImage(const QPoint& w) const;

    /// 当前缩放（子类 drawOverlays 用）
    double zoom_ = 1.0;
    /// 图像尺寸（原图像素）
    int imgW_ = 0, imgH_ = 0;

private:
    QImage image_;        // 底层原图
    QImage display_;      // 缩放后的显示图
};

/// 检测图视图：画定位框
class DetectImageWidget : public ImageHoverWidget {
    Q_OBJECT
public:
    explicit DetectImageWidget(QWidget* parent = nullptr);

    void setLocateResult(const LocateResult* res, int boxCount);

protected:
    void drawOverlays(QPainter& p, double offX, double offY) override;

private:
    const LocateResult* res_ = nullptr;
    int boxCount_ = 0;
};

/// 模板图视图：画字符框
class TemplateImageWidget : public ImageHoverWidget {
    Q_OBJECT
public:
    explicit TemplateImageWidget(QWidget* parent = nullptr);

    void setModelBoxes(const std::vector<ModelBox>& boxes);

protected:
    void drawOverlays(QPainter& p, double offX, double offY) override;

private:
    std::vector<ModelBox> boxes_;
};

/// 预览浮窗：显示模板图对应点的 N×N 区域 + 3×3 中心提示框
class PreviewPopup : public QWidget {
    Q_OBJECT
public:
    explicit PreviewPopup(QWidget* parent = nullptr);

    /// 更新预览：center 是模板图坐标，tmpl 是模板灰度图
    void updatePreview(const cv::Mat& tmpl, int cx, int cy, int size);
    /// 清空（鼠标移出图时）
    void clear();

protected:
    void paintEvent(QPaintEvent*) override;

private:
    QImage preview_;     // 放大后的预览图
    int size_ = 30;
    int displaySize_ = 180;   // 预览图在屏幕上的边长
    int cellPx_ = 0;          // 每个源像素占的屏幕像素
};

/// 主窗口
class LocateViewWindow : public QWidget {
    Q_OBJECT
public:
    explicit LocateViewWindow(QWidget* parent = nullptr);

private slots:
    void onLoadPlugin();
    void onLoadModel();
    void onLoadImage();
    void onRunLocate();
    void onPreviewSizeChanged(int n);
    void onDetectMouseMoved(int x, int y);
    void onDetectMouseGlobal(const QPoint& g);
    void onTemplateMouseMoved(int x, int y);
    void onTemplateMouseGlobal(const QPoint& g);

private:
    void reloadAll();
    bool loadModelBoxes(const std::string& modelDir, std::string* err);
    void updateResultTable();
    void updateStatus(const QString& msg);

    // 插件与模型
    locate_plugin::Plugin plugin_;
    std::string pluginPath_;
    std::string modelDir_;
    cv::Mat modelFreq_;                  // 模型频率图（模板）
    std::vector<ModelBox> modelBoxes_;
    int modelW_ = 0, modelH_ = 0;

    // 检测图与结果
    cv::Mat detectGray_;                 // 检测图灰度
    cv::Mat signal_;                     // 笔画信号（跑定位用）
    LocateResult result_;
    bool hasResult_ = false;
    float scaleX_ = 1.0f, scaleY_ = 1.0f;   // 整图仿射（用于预览映射）
    float offsetX_ = 0.0f, offsetY_ = 0.0f;

    // 控件
    DetectImageWidget* detectView_ = nullptr;
    TemplateImageWidget* tmplView_ = nullptr;
    QScrollArea* tmplScroll_ = nullptr;
    PreviewPopup* popup_ = nullptr;
    QSpinBox* previewSizeSpin_ = nullptr;
    QTableWidget* resultTable_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QLineEdit* pluginEdit_ = nullptr;
    QLineEdit* modelEdit_ = nullptr;
    QLineEdit* imageEdit_ = nullptr;
    QPushButton* runBtn_ = nullptr;
};

}  // namespace locateview
