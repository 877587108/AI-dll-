// locateview_main.cpp —— 定位算法调试平台入口
#include <QApplication>

#include "LocateViewWindow.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    locateview::LocateViewWindow w;
    w.show();
    return app.exec();
}
