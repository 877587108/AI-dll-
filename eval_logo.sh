#!/usr/bin/env bash
# eval_logo.sh —— logo 图案区判据的量化评估
#
# 每次改完算法都跑这个，输出一组可横向对比的数字。
#
# 三组数据：
#   ngv2    7 个不同样本（断字_* 与 1_* 是同一文件，不重复计），期望全部检出
#   unset   158 张正常料，期望零误报
#   ok      83 张旧规格正常料，期望零误报
#
# 同时跑挖洞活性：往一张正常图上挖已知角度的洞，看检测器响应曲线。
# 活性不成立的话，"零误报"只是"什么都没测出来"，不能算通过。
set -u
cd "$(dirname "$0")"

BIN=./build/Release
MODEL=release_defectview/model
TMPL=out/tmpl_logo.png
PRE="${1:-0}"
NGDEG="${2:-1.0}"

echo "=========================================================="
echo " logo 判据评估   preGrayThr=$PRE   ngDeg=$NGDEG"
echo "=========================================================="

# ---- ngv2：期望 7/7 检出
echo
echo "[1] ngv2  7 样本（期望全部检出）"
$BIN/markloc_unsetlogo.exe $MODEL image/ngv2 $TMPL --pre "$PRE" --ngdeg "$NGDEG" 2>/dev/null \
  | grep -E '^[0-9]' | awk -F'  +' '{print $1, $NF, $(NF-1)}' | head -20

# ---- unset：期望零误报
echo
echo "[2] unset 158 张正常料（期望零误报）"
$BIN/markloc_unsetlogo.exe $MODEL image/unset $TMPL --pre "$PRE" --ngdeg "$NGDEG" 2>/dev/null \
  | tail -2

echo
echo "[3] 旧规格 83 张（字符区回归，见 diagnose_0109.sh）"
./diagnose_0109.sh 2>/dev/null | tail -8

echo
echo "[4] 挖洞活性（8deg 应报 NG）"
for d in 0 2 4 6 8 12; do
  printf "  %3ddeg: " $d
  $BIN/markloc_panelhole.exe $MODEL "image/ok/0003_RT_Pass [2026-1010-151442-995] _00.tif" /tmp/h.png $d 2>/dev/null \
    | grep -oE '断点分[0-9]+ 档位[^ ]+ 判定[^ ]+'
done
