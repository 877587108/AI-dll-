@echo off
rem ===================================================================
rem  diagnose_0109.cmd  —  0109_RT 误报的唯一判定回路
rem
rem  一个命令，几十秒，红了就说明 bug 还在。
rem  断言来自用户确认的判定（见 docs/scratch-detection-decisions.md D2）：
rem    0109_RT / 0109_FT 必须 OK
rem    0101_FT 必须 NG          （用户认定 ok 目录里标错的真缺陷）
rem    NG 侧 4 张真缺陷不得回归
rem
rem  用法:  diagnose_0109.cmd                    用交付模型
rem         diagnose_0109.cmd <modelDir>         用指定模型
rem  注意:  模型目录传 release_defectview\model（加载器自己拼 v00），
rem         传 ...\model\v00 会变成 v00\v00 而失败。
rem ===================================================================
setlocal
set MODEL=%~1
if "%MODEL%"=="" set MODEL=release_defectview\model
set FAIL=0

echo === 模型: %MODEL% ===
echo --- ok 侧 ---
build\Release\markloc_defectview.exe --batch "%MODEL%" image\ok ^
  --assert 0109_RT=OK --assert 0109_FT=OK --assert 0101_FT=NG | findstr /C:"[FAIL]" /C:"[PASS]" /C:"条断言"
if errorlevel 1 set FAIL=1

echo --- ng 侧 ---
build\Release\markloc_defectview.exe --batch "%MODEL%" image\ng ^
  --assert 2224=NG --assert 2287=NG --assert 2426=NG --assert 2735=NG | findstr /C:"[FAIL]" /C:"[PASS]" /C:"条断言"
if errorlevel 1 set FAIL=1

echo.
if %FAIL%==0 (echo === 全绿 ===) else (echo === 红了：bug 还在 ===)
endlocal & exit /b %FAIL%
