#!/usr/bin/env bash
# diagnose_0109.sh —— 0109_RT 误报的唯一判定回路（bash 版，供 agent 直接用）
#
# 为什么要有这个脚本：这一轮我推了 8 个假设、8 个都被实测推翻，
# 根因是我先讲故事再找证据。这个回路把"讲"换成"跑"：
# 一个命令、几十秒、退出码即结论。
#
#   退出码 0  -> 全绿，bug 修好了
#   退出码 1  -> 红了，bug 还在
#
# 断言来源（用户亲自确认过，不是我的推断）：
#   docs/scratch-detection-decisions.md  D2 0109_RT 是误报 / 0101 是真缺陷
#
# 用法: ./diagnose_0109.sh                  用交付模型 release_defectview/model
#       ./diagnose_0109.sh out/r020_refine  用别的模型对照
set -u
MODEL="${1:-release_defectview/model}"
EXE=./build/Release/markloc_defectview.exe
FAIL=0

echo "=== 模型: $MODEL ==="

echo "--- ok 侧 ---"
# 0109_RT / 0109_FT 必须 OK；0101_FT 必须 NG（真缺陷）
OUT=$("$EXE" --batch "$MODEL" image/ok \
      --assert 0109_RT=OK --assert 0109_FT=OK --assert 0101_FT=NG 2>&1)
echo "$OUT" | sed -n '/=== 断言 ===/,$p'
echo "$OUT" | grep -q '\[FAIL\]' && FAIL=1

# 【护栏：断言块必须真的出现】
# 这一条是补上来的。模型加载失败时程序直接返回，断言块根本不打印，
# 而上面那些 grep 什么都匹配不到，脚本就会喊"全绿"——假绿灯。
# 实测踩过一次：改了 regions.txt 格式后模型读不了，
# 脚本照样报"全绿：0109_RT OK..."。所以必须显式要求断言块存在。
if ! echo "$OUT" | grep -q '=== 断言 ==='; then
    echo "  [FAIL] ok 侧没有产生断言输出 —— 模型加载失败或程序异常退出"
    echo "$OUT" | head -5 | sed 's/^/     /'
    FAIL=1
fi

echo "--- ng 侧 ---"
# 4 张真缺陷不得回归。断言按文件名子串匹配（2224/2287/2426/2735 是严重度，不是号）
OUT=$("$EXE" --batch "$MODEL" image/ng \
      --assert 0026_FT=NG --assert 0026_RT=NG \
      --assert 0083_FT=NG --assert 0083_RT=NG 2>&1)
echo "$OUT" | sed -n '/=== 断言 ===/,$p'
echo "$OUT" | grep -q '\[FAIL\]' && FAIL=1
if ! echo "$OUT" | grep -q '=== 断言 ==='; then
    echo "  [FAIL] ng 侧没有产生断言输出 —— 模型加载失败或程序异常退出"
    echo "$OUT" | head -5 | sed 's/^/     /'
    FAIL=1
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "=== 全绿：0109_RT OK, 0109_FT OK, 0101_FT NG, NG 侧 4 张全守 ==="
else
    echo "=== 红了：bug 还在 ==="
fi

# ---- 副作用护栏：不参与判定，但每次都要看一眼 ----
# ok 侧总误报数。现在是 4。修 0109 不能靠"把所有图都压低"，
# 那会让这个数掉下去、同时把真缺陷也一起放过。
echo
echo "--- 护栏（不判定，只用来看） ---"
"$EXE" --batch "$MODEL" image/ok  --truth ok 2>&1 | grep -E "^  truth="
"$EXE" --batch "$MODEL" image/ng  --truth ng 2>&1 | grep -E "^  truth="
exit "$FAIL"
