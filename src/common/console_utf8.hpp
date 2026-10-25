#pragma once

// 控制台 UTF-8 输出支持
// Windows 控制台默认代码页常为 936(GBK)，而源码以 UTF-8 编译（/utf-8），
// 直接输出会造成中文乱码。程序启动时把控制台输入/输出代码页切到 65001(UTF-8)。

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
    // 优先使用 UTF-8 代码页
namespace markloc {

inline void enableUtf8Console() {
#ifdef _WIN32
    // 优先使用 UTF-8 代码页
    if (SetConsoleOutputCP(CP_UTF8) == 0) {
        // 极旧系统可能不支持，忽略失败
    }
    SetConsoleCP(CP_UTF8);
#endif
}

}  // namespace markloc
