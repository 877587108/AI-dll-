#pragma once

// 模型构建工具（独立 GUI）
//
// 职责只有三件事，与评估完全分开：
//   1. 模型训练：从 OK 样本统计平均字形
//   2. 选区编辑：在模型平均字形上画选区
//   3. 字符划分：在选区内切分出单字符，可手动增删
//
// 评估/性能测试不在本窗口，用 markloc_gui 单独跑。
// 这样拆分的好处：调模型时不会误触评估，跑产线时也不需要训练界面。

#include <QMainWindow>
#include <QString>
#include <QStringList>

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include "core/char_core.hpp"

class QLineEdit;
class QProgressBar;
class QPushButton;
class QPlainTextEdit;
class QTableWidget;
class QLabel;
class QCheckBox;
class QSpinBox;
class QComboBox;

class TrainToolWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit TrainToolWindow(QWidget* parent = nullptr);

    /// 无界面训练：供命令行/自动化调用
    int runHeadless(const QString& imageDir, const QString& modelDir,
                    const QString& baselineDir);

private slots:
    void chooseImageDir();
    void chooseModelDir();
    void refreshBaselineList();
    void startTrain();
    void openRegionEditor();
    void exportRois();

private:
    void buildUi();
    void log(const QString& s);
    void setBusy(bool busy);

    std::array<int, markloc::charinsp::kViewCount> collectByView(
        const QString& root, std::array<std::vector<QString>, markloc::charinsp::kViewCount>& out);
    bool train(const QString& imageDir, const QString& modelDir, const QString& baselineDir);

    QLineEdit* imageDirEdit_ = nullptr;
    QLineEdit* modelDirEdit_ = nullptr;
    QComboBox* baselineCombo_ = nullptr;
    QComboBox* editViewCombo_ = nullptr;
    QSpinBox* limitSpin_ = nullptr;
    QCheckBox* recursiveCheck_ = nullptr;
    QCheckBox* autoCalibCheck_ = nullptr;
    QPushButton* trainBtn_ = nullptr;
    QPushButton* regionBtn_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QPlainTextEdit* logView_ = nullptr;
    QTableWidget* viewTable_ = nullptr;

    bool busy_ = false;
};
