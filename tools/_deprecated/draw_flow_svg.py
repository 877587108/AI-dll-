# -*- coding: utf-8 -*-
"""绘制整条检测流水线的总览示意图（SVG + PNG，中文字体已转路径，任何机器都能看）。

布局对齐 models/char/vNN/params.txt 的实际处理顺序：
    ROI → 去背景 → ECC → 安全区 → 可疑 → 三类检查 → 打分 → ECC 门控 → 取最大 → 1.0
用法：python tools/draw_flow_svg.py
输出：out/algo_fig/00_flow.svg、out/algo_fig/00_flow.png
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "algo_fig")

BG = "#0e1117"
BOX = "#1b2130"
BOX2 = "#18232b"
EDGE = "#4a5670"
TXT = "#e8eef8"
SUB = "#93a4bd"
ACCENT = {"step": "#63b3ff", "input": "#ffd166", "out": "#ff6b6b", "ok": "#51cf8a",
          "warn": "#ffa94d"}

FONT_PATH = next((p for p in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                              r"C:\Windows\Fonts\simsun.ttc"] if os.path.exists(p)), None)
if FONT_PATH:
    font_manager.fontManager.addfont(FONT_PATH)
    FP = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = [FP.get_name()]
else:
    FP = None


def box(ax, x, y, w, h, title, lines=(), color=ACCENT["step"], fs=10.5, ls="round"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"{ls},pad=0,rounding_size=0.08",
                                fc=BOX, ec=color, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top", color=TXT, fontsize=fs,
            fontweight="bold", zorder=3)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.44 - i * 0.21, t, ha="center", va="top", color=SUB,
                fontsize=fs - 2.2, zorder=3)


def arrow(ax, p1, p2, color=EDGE, rad=0.0, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, color=color,
                                 lw=lw, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.2, 15.2), dpi=110, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 15.2)
    ax.axis("off")

    ax.text(0.25, 14.95, "刻印缺陷检测 · 整条流水线", color=TXT, fontsize=23, fontweight="bold",
            va="top")
    ax.text(0.25, 14.50, "一张图 1688×1460（实拍尺寸）→ 四个 ROI → 六视角各算一遍 → 取最大值与 1.0 比较",
            color=SUB, fontsize=12, va="top")

    # ---- 输入
    box(ax, 0.25, 12.55, 3.30, 1.30, "输入 · 一张灰度图",
        ["1920×1200（描述）/ 1688×1460（实拍）", "同零件共 6 个视角 _00.._05"], ACCENT["input"], 10)
    # ---- ① 去背景
    box(ax, 3.85, 12.40, 4.20, 1.60, "① 去背景 + 归一化  strokeSignal()",
        ["S = (I − Gaussian(I, σ=15)) / q94", "暗刻印先取反 · 只留正残差 · 截到 0"],
        ACCENT["step"], 11)
    # ---- ② ECC
    box(ax, 8.35, 12.40, 3.55, 1.60, "② 对齐  estimateWarp()",
        ["半分辨率 findTransformECC", "平移+旋转 → 得到 ECC 质量分"], ACCENT["step"], 11)
    # ---- ③ 掩膜
    box(ax, 12.20, 12.40, 3.75, 1.60, "③ 安全区  tolMask",
        ["allowed = NOT tolMask", "usable = 去掉上下25/左右10px"], ACCENT["step"], 11)

    # ---- ④ 三类检查（并行）
    ax.add_patch(FancyBboxPatch((3.85, 9.25), 12.10, 2.45, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=BOX2, ec=EDGE, lw=1.2, ls="--", zorder=1.5))
    ax.text(3.98, 11.50, "④ 可疑 = (S > 阈值) ∧ 非安全区 ∧ 非边缘　→　三类并行检查", color=TXT,
            fontsize=12, fontweight="bold", va="top", zorder=3)
    box(ax, 4.05, 9.40, 3.80, 1.70, "A 多余笔画（仅 logo）",
        ["连通域 → PCA 伸长比 > 2.2", "原始量 = 面积 × 平均 S"], ACCENT["warn"], 10)
    box(ax, 8.00, 9.40, 3.85, 1.70, "B 划痕（全部区域）",
        ["Hough 直线 → 密度 ≥ 0.25", "长度 ≥ 160px 才计分 · 原始量 = 能量×长度"], ACCENT["warn"], 10)
    box(ax, 12.00, 9.40, 3.95, 1.70, "C 笔画粗细（仅 code）",
        ["距离变换 → 笔画宽度中位数", "超粗像素占比 → z-score"], ACCENT["warn"], 10)

    # ---- ⑤ 打分
    box(ax, 3.85, 7.05, 12.10, 1.55, "⑤ 各自除以归一化分母 → 变成可比的“分”",
        ["A: 面积×强度 / appendageT(210)　·　B: 能量×长度 / scratchT(逐视角)　·　C: (z − 死区) / 1.5",
         "区域分 = 三项最大　（当前 params.txt：checkCode=0 → C 关闭；checkMissing=0 → 缺墨分支关闭）"],
        ACCENT["step"], 11.5)

    # ---- ⑥ ECC 门控
    box(ax, 3.85, 5.05, 3.10, 1.45, "⑥ ECC 门控",
        ["ecc ≥ eccMin(0.60)", "code 区 ≥ codeEccMin(0.55)", "不达标 → 区域分清零"],
        ACCENT["warn"], 10.5)
    # ---- ⑦ 三层 max
    box(ax, 7.35, 5.05, 3.35, 1.45, "⑦ 视角分",
        ["= max(四个区域分)", "v03/v04 的 logo 区关闭，跳过", "视角开关 enabled=0 不参与"], ACCENT["step"],
        10.5)
    box(ax, 11.10, 5.05, 4.85, 1.45, "⑧ 零件分",
        ["= max(六个视角分)　没有投票、没有平均", "一个视角一个区域报 → 整颗 NG"], ACCENT["step"], 10.5)

    # ---- 最终判定
    ax.add_patch(FancyBboxPatch((3.85, 3.05), 12.10, 1.35, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=BOX, ec=ACCENT["out"], lw=1.8, zorder=2))
    ax.text(9.90, 4.20, "⑨ 最终判定：零件分 ≥ 1.0 ?", ha="center", va="top", color=TXT,
            fontsize=13.5, fontweight="bold", zorder=3)
    ax.text(9.90, 3.80, "kNgThreshold = 1.0（全局常量，不随视角改变）", ha="center", va="top",
            color=SUB, fontsize=10.5, zorder=3)
    box(ax, 4.60, 2.05, 3.10, 0.72, "NG 不良", (), ACCENT["out"], 13)
    box(ax, 11.60, 2.05, 3.10, 0.72, "OK 合格", (), ACCENT["ok"], 13)
    arrow(ax, (7.6, 3.05), (6.15, 2.70), ACCENT["out"])
    arrow(ax, (12.2, 3.05), (13.65, 2.70), ACCENT["ok"])
    ax.text(6.45, 2.80, "是", color=ACCENT["out"], fontsize=11, fontweight="bold")
    ax.text(12.90, 2.80, "否", color=ACCENT["ok"], fontsize=11, fontweight="bold")

    # 主干箭头
    arrow(ax, (3.55, 13.20), (3.85, 13.20))
    arrow(ax, (8.05, 13.20), (8.35, 13.20))
    arrow(ax, (11.90, 13.20), (12.20, 13.20))
    arrow(ax, (5.95, 12.40), (5.95, 11.70), ACCENT["step"])
    arrow(ax, (9.92, 12.40), (9.92, 11.70), ACCENT["step"])
    arrow(ax, (13.97, 12.40), (13.97, 11.70), ACCENT["step"])
    for xa in (5.95, 9.92, 13.97):
        arrow(ax, (xa, 9.40), (xa, 8.75))
    arrow(ax, (9.90, 7.05), (9.90, 6.50))
    arrow(ax, (6.95, 5.77), (7.35, 5.77))
    arrow(ax, (10.70, 5.77), (11.10, 5.77))
    arrow(ax, (13.52, 5.05), (13.52, 4.40), ACCENT["out"])

    # 图例
    ax.text(0.25, 1.45, "记法：橙色字 = 会让分数归零或已关闭的开关　；"
                        "所有阈值与分母都随模型保存在 models/char/vNN/params.txt 里，逐视角独立",
            color=SUB, fontsize=10.5, va="top")
    ax.text(0.25, 1.00, "设计取向：三层都取最大值 = 宁可错杀。"
                        "代价是 OK 样本余量薄（0116 实测 0.927，离 1.0 只剩 0.073）。",
            color=ACCENT["warn"], fontsize=10.5, va="top")

    svg = os.path.join(OUT, "00_flow.svg")
    png = os.path.join(OUT, "00_flow.png")
    fig.savefig(svg, facecolor=fig.get_facecolor(), bbox_inches="tight")
    import time
    for _ in range(6):
        try:
            fig.savefig(png, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
            break
        except OSError:        # 杀软偶发占用
            time.sleep(0.4)
    print("wrote", os.path.relpath(svg, ROOT), os.path.relpath(png, ROOT))


if __name__ == "__main__":
    sys.exit(main())
