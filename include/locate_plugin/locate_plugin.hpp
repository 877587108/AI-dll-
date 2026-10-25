#pragma once

// locate_plugin.hpp —— 定位算法的 DLL 注入接口
//
// ================================================================
// 这个接口要解决的问题
// ================================================================
//
// 现在定位算法写死在 markloc 内部（`char_alignment.hpp` 的 locateCharacters），
// 换一个定位算法必须重新编译整个程序。而定位恰恰是最需要反复试的部分：
//
//   · 模板匹配 NCC     —— 快，但对旋转和尺度敏感
//   · 特征点匹配       —— 抗旋转，但字符笔画细，特征点不稳
//   · 相位相关         —— 能算整体平移，但无法逐字
//   · 光流             —— 能算局部形变
//   · 深度学习特征     —— 最稳，但部署重
//
// 这些算法需要能在**同一批图上横向比较**，比较的依据是同一个指标：
// 「定位结果与模板字符框的偏差」。所以需要一个插件机制。
//
// ================================================================
// 调用方与插件的关系
// ================================================================
//
//   宿主（markloc_locatebench）
//     ├─ 读模型：拿到模板字符框（模型坐标）
//     ├─ 读待测图：转成笔画信号
//     ├─ 加载 DLL ──► 插件实现 ILocator
//     ├─ 调用 locate()
//     └─ 评估：把结果与模板框比较，算偏差、统计、画图
//
// 插件的职责只有一件：**给定图和模板框，说出每个框在图上的实际位置**。
// 它不做评估、不画图、不读文件。
//
// ================================================================
// 插件必须做什么（契约）
// ================================================================
//
// 1. 实现 ILocator 的全部纯虚函数
// 2. 导出下面两个 C 函数：
//
//      extern "C" __declspec(dllexport) locate_api_t* locate_create();
//      extern "C" __declspec(dllexport) void         locate_destroy(locate_api_t*);
//
//    用 C 链接是为了避免不同编译器/标准库的 ABI 问题。
//
// 3. 绝不修改输入（const），绝不抛异常到宿主（自己 catch 后返回空）
// 4. 返回的每个 LocatedChar 必须与输入的 templateBoxes 一一对应（顺序相同）
//
// ================================================================
// 坐标系约定
// ================================================================
//
//   templateBoxes  —— **模型坐标**（模型图尺寸下的像素）
//   imageBoxes     —— **待测图坐标**（待测图尺寸下的像素）
//
//   两图尺寸可能不同（模型 1688×1460，待测 1496×1496）。
//   插件负责处理这个差异，不能假定两图同尺寸。

#include <cstdint>

#ifdef _WIN32
#define LOCATE_EXPORT extern "C" __declspec(dllexport)
#else
#define LOCATE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

/// 接口版本。宿主加载时校验，不匹配则拒绝。
/// 改任何结构体布局都必须加一。
#define LOCATE_PLUGIN_API_VERSION 1

/// 一个字符框的定位结果
struct LocatedChar {
    /// 字符序号，与输入 templateBoxes 的下标一致
    int index = -1;

    /// 定位到的框（待测图坐标）
    /// 未定位成功时保留模板框经缩放换算后的近似位置，accepted 置 false
    int x = 0, y = 0, w = 0, h = 0;

    /// 置信度。含义由插件自定（NCC 分数、内点比例等），宿主要求落在 [0,1]。
    /// 宿主只做排序和显示，不拿它做阈值判断 —— 阈值由 config 里的参数决定。
    float confidence = 0.0f;

    /// 该字符是否定位成功。false 时上面的框坐标仅供参考，评估会跳过。
    bool accepted = false;
};

/// 整体定位结果
struct LocateResult {
    /// 整图的尺度/平移估计（可选输出，用于诊断）
    /// scaleX = 待测图相对模型图的横向缩放，scaleY 同理
    float scaleX = 1.0f;
    float scaleY = 1.0f;

    /// 整图偏移（待测图坐标 - 模型坐标×scale），用于诊断
    float offsetX = 0.0f;
    float offsetY = 0.0f;

    /// 每个字符的结果，顺序与输入 templateBoxes 一致
    const LocatedChar* chars = nullptr;
    int charCount = 0;

    /// 算法自评：整图定位是否可信
    /// 宿主会显示这个值，但**不会**据此跳过评估 ——
    /// "没定位好"也要如实评估出来，而不是藏起来（见 docs/csv-schema.md §3）
    bool globalOk = false;

    /// 失败或说明信息，UTF-8，可空
    const char* message = nullptr;
};

/// 输入几何：一个模板字符框
struct TemplateBox {
    int x, y, w, h;
    int index = 0;
};

/// 可调参数：宿主从命令行传入 key=value，插件自己解析
struct LocateParam {
    const char* key = nullptr;
    double value = 0.0;
};

/// 定位插件的全部接口
///
/// **注意生命周期**：
///   · 宿主调用 detect() 期间，输入指针有效
///   · detect() 返回的 LocateResult 里的 chars 指针
///     必须一直有效到宿主调用 release() 或下次 detect()
///   · 插件内部负责这块内存
struct locate_api_t {
    int apiVersion = LOCATE_PLUGIN_API_VERSION;

    /// 插件名字，显示用
    const char* name = nullptr;

    /// 一次性初始化。返回 false 表示不可用（缺依赖文件等）。
    bool (*init)(const char* modelDir);

    /// 设置参数。宿主在 init 之后、detect 之前调用。
    /// 未知 key 应当被忽略（便于宿主传通用参数）。
    void (*setParams)(const LocateParam* params, int count);

    /// 列出自带参数的默认值和说明，供 --list-params 显示
    /// 返回的字符串数组由插件持有，长度为 *count
    const char** (*listParams)(int* count);

    /// 执行定位
    ///
    /// @param signal        待测图的笔画信号（CV_32F，整图尺寸）
    ///                      void* 是为了不把 OpenCV 类型暴露到 ABI 边界
    ///                      插件内部 static_cast 成 const cv::Mat*
    /// @param imageW/H      待测图尺寸（冗余，便于不依赖 cv::Mat 的实现）
    /// @param modelW/H      模型图尺寸
    /// @param templateBoxes 模板字符框（模型坐标），长度 boxCount
    /// @return 定位结果
    LocateResult (*detect)(const void* signal, int imageW, int imageH, int modelW, int modelH,
                           const TemplateBox* templateBoxes, int boxCount);

    /// 释放 detect() 返回结果里的内存
    void (*release)(LocateResult* result);

    /// 插件卸载前清理
    void (*shutdown)();
};
