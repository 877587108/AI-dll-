#pragma once

// 六视角模型图与选区配置工具
//
// 流程：
//   ① 选 OK 样本文件夹 → 按文件名末尾 _00.._05 分成六个视角，
//      各自训练出一张「模型图」（平均字形）
//   ② 在任一视角的模型图上框选文字行 → 自动切出单个字符
//   ③ 保存：选区配置对**六个视角通用**
//   ④ 精对齐：按视角各自重建模型图（模板需来自本视角）
//
// 为什么选区可以通用：
//   实测六个视角的芯片位置由同一治具决定，平移 < 1px；
//   刻印是物理过程，字在各视角中位置相同。
//   因此一份坐标可套用到全部视角。
//
// 目录约定：
//   model/regions.conf          ★ 共享选区配置（六视角通用）
//   model/v00/model.png         视角 0 的模型图
//   model/v00/model_ref.png     配准基准
//   model/v00/versions/         视角 0 的历史版本
//   model/v01/ … v05/           同上

#include <QImage>
#include <QMainWindow>
#include <QString>

#include <array>
#include <atomic>
#include <vector>

#include <opencv2/core.hpp>

#include "chardet.hpp"

class BoxCanvas;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSpinBox;
class QComboBox;
class QThread;
class QTimer;
class QProgressBar;

class SplitVerifyWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit SplitVerifyWindow(QWidget* parent = nullptr);
    ~SplitVerifyWindow() override;

private slots:
    void chooseSampleDir();
    void onViewChanged(int index);
    void openModelImage();
    void onBoxCreated(const QRectF& r);
    void onSelectionChanged(int id);
    void onDeleteRequested(int id);
    void resplitAll();
    void deleteSelected();
    void clearAll();
    void saveConfig();
    void refineModel();
    void onRefineDone();
    void loadSelectedVersion();
    /// 切换去黑点方式：0=原图 1=中值滤波 2=开运算
    void setDenoiseMode(int mode);
    /// ⑤ 检测：载入产品图，与精修模型比对（漏字/漏墨/多墨）
    void detectProduct();

private:
    static constexpr int kViews = 6;

    /// 单个视角的模型状态
    struct ViewState {
        cv::Mat model;                          // 当前显示的模型图（可能已滤波）
        cv::Mat raw;                            // 未滤波的原始模型图
        cv::Mat ref;                            // 配准基准
        bool trained = false;
    };

    /// 一个选区及其字符（六视角共享）
    struct Region {
        cv::Rect rect;
        std::vector<chardet::CharBox> chars;
    };

    void buildUi();
    void refreshCanvas();
    void refreshList();
    void updateStatus();
    void splitRegion(int idx);
    void tryLoadExistingModels();
    QString viewDir(int v) const;
    /// 共享选区配置路径（根目录下，不属于任何视角）
    QString configPath() const;
    QString modelPath(int v) const;
    QString versionsDir(int v) const;
    void saveVersion(int v, const QString& kind, const QString& note);
    /// 把当前视角的模型图与选区写成 chardet 模块格式（供检测读取）
    bool saveChardetModel(int v);
    void refreshVersionList();
    void syncCanvasFromView();
    int currentView() const;

    /// 当前视角模型图的"原图"（未经滤波），用于每次重新应用滤波
    cv::Mat& rawModel(int v);
    /// 按当前滤波设置处理模型图并刷新显示
    void applyFilterToCurrent();

    std::array<ViewState, kViews> views_;
    std::vector<Region> regions_;      // ★ 六视角共享的选区
    int cur_ = 0;
    QImage qimg_;
    QString rootDir_;       // 根目录（其下是 v00..v05）
    QString sampleDir_;

    BoxCanvas* canvas_ = nullptr;
    QLineEdit* dirEdit_ = nullptr;
    QComboBox* viewCombo_ = nullptr;
    QComboBox* versionCombo_ = nullptr;
    QListWidget* list_ = nullptr;
    QLabel* infoLabel_ = nullptr;
    QLabel* hintLabel_ = nullptr;
    QSpinBox* colThrSpin_ = nullptr;
    QSpinBox* gapSpin_ = nullptr;
    // 模型图去黑点：两个互斥按钮
    QPushButton* fillHoleBtn_ = nullptr;   // 填笔画黑点（孔洞填充）
    QPushButton* rawBtn_ = nullptr;        // 还原
    QSpinBox* denoiseStrengthSpin_ = nullptr;   // 孔洞面积门限
    /// 0=原图 2=填笔画黑点
    int denoiseMode_ = 0;

    // 后台精对齐
    QObject* worker_ = nullptr;
    QThread* workerThread_ = nullptr;
    bool refining_ = false;
    std::atomic<bool> cancelFlag_{false};   // 精对齐中断标志
    QTimer* progressTimer_ = nullptr;       // 精对齐进度轮询
    QProgressBar* progressBar_ = nullptr;    // 精对齐进度条
};
