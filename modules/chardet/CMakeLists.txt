cmake_minimum_required(VERSION 3.20)
project(chardet VERSION 0.1.0 LANGUAGES CXX)

# 字符级检测模块 —— 独立构建
#
# 这个模块从 markloc 主项目分离出来，只依赖 OpenCV。
# 可以单独构建为静态库，供任何项目链接：
#
#   add_subdirectory(modules/chardet)
#   target_link_libraries(your_target PRIVATE chardet)
#
# 单独构建：
#   cmake -S modules/chardet -B modules/chardet/build -DOpenCV_ROOT=D:/opencv

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_ROOT "D:/opencv" CACHE PATH "OpenCV install root")
if (NOT OpenCV_DIR)
    set(OpenCV_DIR "${OpenCV_ROOT}/build/x64/vc16/lib")
endif()
find_package(OpenCV REQUIRED)

add_library(chardet STATIC
    split.cpp
    match.cpp
    blocks.cpp
    skeleton.cpp
    multitmpl.cpp
    align_char.cpp
    model.cpp
    pattern_break.cpp
    pattern/pattern_types.cpp
    pattern/preprocessor.cpp
    pattern/profile_builder.cpp
    pattern/pattern_pipeline.cpp
    pattern/det_notch_window.cpp
    pattern/det_gap_run.cpp
    pattern/det_registry.cpp
)

# 输出到与主项目一致的目录，避免多配置下的路径不一致
set_target_properties(chardet PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")
target_include_directories(chardet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(chardet PUBLIC ${OpenCV_LIBS})

if (MSVC)
    target_compile_options(chardet PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
    set_property(TARGET chardet PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
