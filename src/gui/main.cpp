// 评估程序入口（识别 + 性能）
// 训练 / 选区 / 划字在独立的 markloc_train 里

#include "gui/TrainWindow.hpp"

#include <QApplication>
#include <QStringList>

#include <cstdio>

#include "common/console_utf8.hpp"

namespace {

void usage() {
    std::printf(
        "用法:\n"
        "  markloc_gui.exe                            打开评估界面\n"
        "  markloc_gui.exe --eval <图片目录> <模型目录>\n"
        "      无界面评估：准确率 + 性能（速度/内存/吞吐/零件速率）\n"
        "\n"
        "模型训练与选区划字请用 markloc_train.exe\n");
}

}  // namespace

int main(int argc, char** argv) {
    markloc::enableUtf8Console();
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("markloc_gui"));
    QApplication::setApplicationDisplayName(QStringLiteral("markloc 评估"));

    const QStringList args = QApplication::arguments();

    if (args.size() >= 4 && args[1] == QStringLiteral("--eval")) {
        TrainWindow w;
        return w.runEvaluateOnly(args[2], args[3]);
    }

    if (args.size() >= 2 && args[1].startsWith(QStringLiteral("--"))
        && args[1] != QStringLiteral("--help")) {
        usage();
        return 2;
    }

    TrainWindow w;
    w.show();
    return app.exec();
}
