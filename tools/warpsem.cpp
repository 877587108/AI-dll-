// warpAffine 方向语义最小复现
//
// 这个仓库里 warpAffine 被同时用于"制造平移"和"修正平移"，
// 两处方向必须相反。搞反了不会报错，只会让结果整体偏一倍位移，
// 而现象看起来像是"对齐算法不行"。先用 3x3 图确认到底哪个是哪个。
#include <cstdio>
#include <opencv2/imgproc.hpp>

static void dump(const char* tag, const cv::Mat& m) {
    std::printf("%s: ", tag);
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.cols; ++x) std::printf("%3.0f", m.at<float>(y, x));
    std::printf("\n");
}

int main() {
    // 只有一个亮点在 (0,0)
    cv::Mat src = cv::Mat::zeros(3, 3, CV_32F);
    src.at<float>(0, 0) = 9.f;
    dump("原图", src);

    // 正方向：+1 x 和 +1 y
    const cv::Mat Mfwd = (cv::Mat_<double>(2, 3) << 1, 0, 1, 0, 1, 1);
    cv::Mat a, b;
    cv::warpAffine(src, a, Mfwd, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::warpAffine(src, b, Mfwd, src.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, 0);
    dump("M=+1 无INVERSE", a);   // 期望亮点在 (1,1)
    dump("M=+1 有INVERSE", b);   // 期望亮点在 (2,2) 或原样

    // 反方向：-1
    const cv::Mat Minv = (cv::Mat_<double>(2, 3) << 1, 0, -1, 0, 1, -1);
    cv::Mat c, d;
    cv::warpAffine(src, c, Minv, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::warpAffine(src, d, Minv, src.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, 0);
    dump("M=-1 无INVERSE", c);
    dump("M=-1 有INVERSE", d);
    return 0;
}
