#pragma once

// 评估窗口（独立 GUI）
//
// 只做一件事：用模型对图片目录跑识别，输出准确率 + 性能指标。
//
// 训练、选区、划字不在这里 —— 那些在 markloc_train 里。
// 这样拆分的好处：跑产线时不需要训练代码路径，界面也不容易误操作。

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

class TrainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit TrainWindow(QWidget* parent = nullptr);

    void setDirs(const QString& imageDir, const QString& modelDir);
    /// 无界面评估（含性能统计）
    int runEvaluateOnly(const QString& imageDir, const QString& modelDir);

private slots:
    void chooseImageDir();
    void chooseModelDir();
    void startEvaluate();

private:
    /// 性能统计结果（评估时顺带采集）
    struct BenchResult {
        bool valid = false;
        int images = 0;
        int parts = 0;
        double totalMs = 0.0;
        double decodeMs = 0.0;
        double analyzeMs = 0.0;
        double avgMs = 0.0;
        double medianMs = 0.0;
        double minMs = 0.0;
        double maxMs = 0.0;
        double fps = 0.0;
        double partsPerSec = 0.0;
        double memNowMB = 0.0;
        double memPeakMB = 0.0;
        double privNowMB = 0.0;
        double privPeakMB = 0.0;
    };

    void buildUi();
    void log(const QString& s);
    void setBusy(bool busy);
    void clearResults();

    std::array<int, markloc::charinsp::kViewCount> collectByView(
        const QString& root, std::array<std::vector<QString>, markloc::charinsp::kViewCount>& out);
    void evaluateAndShow(const QString& imageDir, const QString& modelDir);
    void showBenchmark(const BenchResult& b);

    QLineEdit* imageDirEdit_ = nullptr;
    QLineEdit* modelDirEdit_ = nullptr;
    QSpinBox* limitSpin_ = nullptr;
    QCheckBox* recursiveCheck_ = nullptr;
    QCheckBox* charCheck_ = nullptr;
    QPushButton* evalBtn_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QPlainTextEdit* logView_ = nullptr;
    QTableWidget* resultTable_ = nullptr;
    QTableWidget* benchTable_ = nullptr;
    QLabel* summaryLabel_ = nullptr;

    bool busy_ = false;
};
