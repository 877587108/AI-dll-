// cropcmp.cpp —— 两图差异区域放大对比
//
// re_ok/1_00 与 ngv2/1_00 只差 (514,374,30,22) 这一小块。
// 若这就是断口，那它给了断口的精确坐标，比从连通块反推可靠。
//
// 出图：左 = NG 该区域（放大 8 倍），右 = re_ok 同区域（放大 8 倍），
//       中间标出差异包围盒。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static cv::Mat readAny(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::vector<uchar> buf;
    uchar tmp[65536];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(f);
    if (buf.empty()) return {};
    return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("用法: markloc_cropcmp <a> <b> <cx> <cy> [--half 60] [--zoom 8] [--out x.png]\n");
        return 2;
    }
    const cv::Mat a = readAny(argv[1]);
    const cv::Mat b = readAny(argv[2]);
    if (a.empty() || b.empty()) { std::printf("读图失败\n"); return 1; }
    const int cx = std::atoi(argv[3]), cy = std::atoi(argv[4]);
    int half = 60, zoom = 8;
    std::string out = "out/trace/cropcmp.png";
    for (int i = 5; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--half") half = std::atoi(argv[i + 1]);
        else if (k == "--zoom") zoom = std::atoi(argv[i + 1]);
        else if (k == "--out") out = argv[i + 1];
    }
    cv::Rect r(cx - half, cy - half, 2 * half, 2 * half);
    r &= cv::Rect(0, 0, a.cols, a.rows);
    cv::Mat ca = a(r).clone(), cb = b(r).clone();
    cv::Mat za, zb;
    cv::resize(ca, za, cv::Size(), zoom, zoom, cv::INTER_NEAREST);
    cv::resize(cb, zb, cv::Size(), zoom, zoom, cv::INTER_NEAREST);
    cv::Mat za3, zb3;
    cv::cvtColor(za, za3, cv::COLOR_GRAY2BGR);
    cv::cvtColor(zb, zb3, cv::COLOR_GRAY2BGR);
    // 标出中心
    cv::drawMarker(za3, cv::Point(za3.cols / 2, za3.rows / 2), cv::Scalar(0, 0, 255),
                   cv::MARKER_CROSS, 20, 1);
    cv::drawMarker(zb3, cv::Point(zb3.cols / 2, zb3.rows / 2), cv::Scalar(0, 0, 255),
                   cv::MARKER_CROSS, 20, 1);
    auto lab = [](cv::Mat& m, const char* t) {
        cv::Mat o(m.rows + 24, m.cols, CV_8UC3, cv::Scalar(25, 25, 25));
        m.copyTo(o(cv::Rect(0, 24, m.cols, m.rows)));
        cv::putText(o, t, cv::Point(5, 17), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
        m = o;
    };
    lab(za3, "NG  (ngv2/1_00)");
    lab(zb3, "OK  (re_ok/1_00)");
    cv::Mat outm;
    cv::hconcat(za3, zb3, outm);
    cv::imwrite(out, outm);
    std::printf("已写 %s (%dx%d)，区域 (%d,%d,%d,%d)，放大 %dx\n", out.c_str(), outm.cols,
                outm.rows, r.x, r.y, r.width, r.height, zoom);
    return 0;
}
