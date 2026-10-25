@echo off
rem ================================================================
rem eval_plugin.bat - run locate_bench on a compiled plugin DLL
rem usage: eval_plugin.bat ^<plugin.dll^> ^<modelDir^> ^<list.txt^> [outDir]
rem list.txt: one image path per line (UTF-8)
rem ================================================================
setlocal

if "%~3"=="" (
    echo usage: eval_plugin.bat ^<plugin.dll^> ^<modelDir^> ^<list.txt^> [outDir]
    exit /b 1
)

set "PLUGIN=%~1"
set "MODEL=%~2"
set "LIST=%~3"
if "%~4"=="" (
    set "OUT=out\eval"
) else (
    set "OUT=%~4"
)

echo [eval] plugin: %PLUGIN%
echo [eval] model : %MODEL%
echo [eval] list  : %LIST%
echo [eval] out   : %OUT%
echo.

rem bench does not auto-create dir; fopen silently fails without it
if not exist "%OUT%" mkdir "%OUT%"

build\Release\markloc_locatebench.exe --plugin "%PLUGIN%" --model "%MODEL%" --list "%LIST%" --out "%OUT%"

exit /b %errorlevel%
