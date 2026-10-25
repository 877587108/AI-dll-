// scale_probe.cpp —— 验证跨分辨率下的几何比例是否恒定
//
// 【要回答的问题】
// unset/ngv2 是 1496×1496，image/ok 是 1688×1460。
// 如果 logo 椭圆的长半轴 a 占图宽的比例在两批里相同，
// 说明两批物理分辨率一致，"按比例换算坐标"的路线成立。
// 如果比例不同，说明两批的镜头/距离/分辨率不同，
// 按比例换算是错的，必须先解决分辨率标定问题。
//
// 【为什么这是致命风险】
// 部分1的整条路线（定位 → 按比例换算阈值 → 检测）
// 都建立在"同一颗芯片在不同图上几何比例恒定"这个假设上。
// 这个假设不成立，阈值按比例缩放就是无意义的。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <QFile>
#include <QFileInfo>
#include <QString>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static bool readGray(const QString& path, cv::Mat& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray b = f.readAll();
    f.close();
    if (b.isEmpty()) return false;
    const std::vector<uchar> buf(b.begin(), b.end());
    out = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: markloc_scaleprobe <tmpl.png> <list.txt>\n");
        std::printf("  量每张图上 logo 椭圆的 a 值，输出 a/图宽、a/图高\n");
        return 2;
    }
    cv::Mat tmpl;
    if (!readGray(QString::fromUtf8(argv[1]), tmpl)) {
        std::printf("模板读取失败\n");
        return 1;
    }

    std::vector<std::string> files;
    {
        QFile f(QString::fromUtf8(argv[2]));
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            std::printf("列表读取失败\n");
            return 1;
        }
        while (!f.atEnd()) {
            const QString line = QString::fromUtf8(f.readLine()).trimmed();
            if (!line.isEmpty()) files.push_back(line.toStdString());
        }
    }

    // 模型里的 logo 椭圆（来自 release_defectview/model/v00 的 regions）
    const double modelA = 134.25, modelB = 77.05;

    std::printf("%-42s %6s %6s %8s %8s %8s %8s\n", "file", "W", "H", "x", "y", "x/W", "y/H");
    std::printf("%s\n", std::string(100, '-').c_str());

    for (const auto& p : files) {
        cv::Mat g;
        if (!readGray(QString::fromUtf8(p.c_str()), g)) {
            std::printf("%-42s 读取失败\n", QFileInfo(QString::fromUtf8(p.c_str()))
                                                 .fileName().toStdString().substr(0, 40).c_str());
            continue;
        }
        if (g.cols < tmpl.cols || g.rows < tmpl.rows) {
            std::printf("%-42s 图比模板小\n", QFileInfo(QString::fromUtf8(p.c_str()))
                                                  .fileName().toStdString().substr(0, 40).c_str());
            continue;
        }
        cv::Mat mres;
        cv::matchTemplate(g, tmpl, mres, cv::TM_CCOEFF_NORMED);
        cv::Point mloc;
        double mval = 0;
        cv::minMaxLoc(mres, nullptr, &mval, nullptr, &mloc);
        // 模板匹配得到的偏移 → 实际图上 logo 椭圆中心
        const double cx = mloc.x + 127.8, cy = mloc.y + 116.0;
        const QString fn = QFileInfo(QString::fromUtf8(p.c_str())).fileName();
        std::printf("%-42s %6d %6d %8.1f %8.1f %8.4f %8.4f  ncc=%.3f\n",
                    fn.toStdString().substr(0, 40).c_str(), g.cols, g.rows, cx, cy,
                    cx / g.cols, cy / g.rows, mval);
        (void)modelA;
        (void)modelB;
    }
    return 0;
}
