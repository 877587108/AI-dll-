#pragma once

// plugin_loader.hpp —— 定位插件 DLL 的加载与封装
//
// 供宿主（markloc_locatebench / markloc_locateview）复用。
// 封装了 Windows LoadLibrary 的 UTF-8 路径处理、导出函数查找、版本校验、
// 以及卸载时的清理顺序。
//
// 用法：
//   locate_plugin::Plugin p;
//   std::string err;
//   if (!p.load("path/to/locate_ncc.dll", &err)) { ... }
//   locate_api_t* api = p.api();
//   // ... 调用 api->init / setParams / detect ...
//   // p 析构时自动 unload

#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "locate_plugin.hpp"

namespace locate_plugin {

class Plugin {
public:
    Plugin() = default;
    Plugin(const Plugin&) = delete;
    Plugin& operator=(const Plugin&) = delete;
    Plugin(Plugin&& other) noexcept { *this = std::move(other); }
    Plugin& operator=(Plugin&& other) noexcept {
        if (this != &other) {
            unload();
            hmod_ = other.hmod_;
            destroy_ = other.destroy_;
            api_ = other.api_;
            other.hmod_ = nullptr;
            other.destroy_ = nullptr;
            other.api_ = nullptr;
        }
        return *this;
    }

    ~Plugin() { unload(); }

    /// 加载插件 DLL。
    /// 成功后 api() 可用；失败时 err 写原因，返回 false。
    bool load(const std::string& path, std::string* err = nullptr) {
        unload();
#ifdef _WIN32
        // UTF-8 → 宽字符，避免中文路径问题
        const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wn <= 0) {
            setErr(err, "路径转宽字符失败");
            return false;
        }
        std::wstring wpath(static_cast<size_t>(wn), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wn);

        hmod_ = LoadLibraryW(wpath.c_str());
        if (!hmod_) {
            const DWORD e = GetLastError();
            setErr(err, "LoadLibrary 失败，错误码 " + std::to_string(e));
            return false;
        }
        using CreateFn = locate_api_t* (*)();
        using DestroyFn = void (*)(locate_api_t*);
        auto create = reinterpret_cast<CreateFn>(GetProcAddress(hmod_, "locate_create"));
        destroy_ = reinterpret_cast<DestroyFn>(GetProcAddress(hmod_, "locate_destroy"));
        if (!create) {
            setErr(err, "DLL 里找不到 locate_create");
            unload();
            return false;
        }
        api_ = create();
        if (!api_) {
            setErr(err, "locate_create 返回空");
            unload();
            return false;
        }
        if (api_->apiVersion != LOCATE_PLUGIN_API_VERSION) {
            setErr(err, "接口版本不匹配：插件 " + std::to_string(api_->apiVersion) +
                            "，宿主 " + std::to_string(LOCATE_PLUGIN_API_VERSION));
            unload();
            return false;
        }
        return true;
#else
        (void)path;
        setErr(err, "仅支持 Windows");
        return false;
#endif
    }

    void unload() {
        if (api_ && api_->shutdown) api_->shutdown();
        if (api_ && destroy_) destroy_(api_);
        api_ = nullptr;
        destroy_ = nullptr;
#ifdef _WIN32
        if (hmod_) {
            FreeLibrary(hmod_);
            hmod_ = nullptr;
        }
#endif
    }

    bool loaded() const { return api_ != nullptr; }
    locate_api_t* api() const { return api_; }

private:
    static void setErr(std::string* err, const std::string& msg) {
        if (err) *err = msg;
    }

#ifdef _WIN32
    HMODULE hmod_ = nullptr;
#endif
    void (*destroy_)(locate_api_t*) = nullptr;
    locate_api_t* api_ = nullptr;
};

}  // namespace locate_plugin
