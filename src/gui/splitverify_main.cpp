// 选区切分验证工具 —— 程序入口
//
// 给客户使用的独立小程序：打开一张芯片图，框选文字行，看切分是否准确。
// 不依赖训练好的模型，不需要任何前置配置。

#include <QApplication>

#include "common/console_utf8.hpp"
#include "gui/SplitVerifyWindow.hpp"

int main(int argc, char** argv) {
    markloc::enableUtf8Console();

    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("选区切分验证"));

    SplitVerifyWindow w;
    w.show();
    return app.exec();
}
