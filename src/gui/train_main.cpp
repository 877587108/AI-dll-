// 模型构建工具入口：训练 / 选区 / 划字
// 与评估程序 markloc_gui 完全独立

#include "gui/TrainToolWindow.hpp"

#include <QApplication>
#include <QStringList>

#include <cstdio>

#include "common/console_utf8.hpp"

int main(int argc, char** argv) {
    markloc::enableUtf8Console();
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("markloc_train"));
    QApplication::setApplicationDisplayName(QStringLiteral("markloc 模型构建"));

    const QStringList args = QApplication::arguments();

    // 无界面训练：--train <样本目录> <模型目录> [--baseline <基线目录>]
    if (args.size() >= 4 && args[1] == QStringLiteral("--train")) {
        const QString imageDir = args[2];
        const QString modelDir = args[3];
        QString baseline;
        for (int i = 4; i + 1 < args.size(); ++i) {
            if (args[i] == QStringLiteral("--baseline")) baseline = args[i + 1];
        }
        TrainToolWindow w;
        return w.runHeadless(imageDir, modelDir, baseline);
    }

    TrainToolWindow w;
    w.show();
    return app.exec();
}
