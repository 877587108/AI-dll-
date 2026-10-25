@echo off
rem ================================================================
rem build_plugin.bat - compile a locate plugin .cpp into a DLL
rem
rem usage: build_plugin.bat <source.cpp> [output.dll]
rem default output: out\dev_plugin.dll
rem
rem uses cl.exe directly (no CMake), for fast AI iteration loop
rem ================================================================
setlocal

if "%~1"=="" (
    echo usage: build_plugin.bat ^<source.cpp^> [output.dll]
    exit /b 1
)

set "SRC=%~1"
if "%~2"=="" (
    set "OUT=out\dev_plugin.dll"
) else (
    set "OUT=%~2"
)

set "ROOT=%~dp0.."
set "OPENCV=D:\opencv"

echo [build] src: %SRC%
echo [build] out: %OUT%

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [error] vcvars64.bat failed
    exit /b 1
)

cl.exe /nologo /LD /EHsc /std:c++20 /utf-8 /O2 /I "%ROOT%\include" /I "%OPENCV%\build\include" "%SRC%" /link /LIBPATH:"%OPENCV%\build\x64\vc16\lib" opencv_world4100.lib /OUT:"%OUT%"

if errorlevel 1 (
    echo.
    echo [error] compile failed
    exit /b 1
)

echo.
echo [ok] compiled: %OUT%
exit /b 0
