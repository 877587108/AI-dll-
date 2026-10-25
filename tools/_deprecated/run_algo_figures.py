# -*- coding: utf-8 -*-
"""
把「刻印缺陷检测」整条流水线逐步跑一遍，并把每一步的中间结果渲染成图。

用途：解释性文档配图。
  - 所有中间量都用真实数据算出（不是手绘示意）
  - 算法步骤逐行对齐 src/core/*.cpp，超出的部分只是为了画图而写的辅助代码
  - 第 6/7 步的分数由本脚本用 Python 复刻 C++ 打分逻辑得到（未调用 C++ 库）

用法（在仓库根目录）：
    python tools/run_algo_figures.py
输出：
    out/algo_fig/*.png        各步骤的截图
    out/algo_fig/figures.json 每张图的说明与实测数值（文档直接引用）

依赖：opencv-python / numpy / matplotlib（仅工具链，非产品依赖）
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "algo_fig")

# ---------------------------------------------------------------------------
# 0. 复刻 Params（src/core/char_core.hpp 默认值），随后用 models/char/vNN/params.txt 覆盖
# ---------------------------------------------------------------------------
ROI_NAMES = ["logo", "line1", "line2", "code"]
ROI_DEFAULT = [  # Params::Params()
    (480, 325, 700, 235),
    (390, 780, 870, 180),
    (490, 945, 660, 175),
    (430, 550, 320, 260),
]

SCALAR_NAMES = [
    "sigma", "strokeHi", "strokeLo", "pKeep", "dilateK", "edgeTB", "edgeLR", "scratchT",
    "appendageT", "eccMin", "scratchMinCoverage", "scratchMinLen", "codeZNg", "checkCode",
    "enabled", "invert", "checkMissing", "codeEccMin", "missFreq", "missStrokeLo", "missMinArea",
    "missT", "checkChars", "charMissRatio", "charExtraRatio", "charT",
]
INT_KEYS = {"dilateK", "edgeTB", "edgeLR", "missMinArea"}
BOOL_KEYS = {"checkCode", "enabled", "invert", "checkMissing", "checkChars"}


@dataclass
class Params:
    sigma: float = 15.0
    strokeHi: float = 0.35
    strokeLo: float = 0.20
    pKeep: float = 0.20
    dilateK: int = 9
    edgeTB: int = 25
    edgeLR: int = 10
    scratchT: float = 22000.0
    appendageT: float = 210.0
    eccMin: float = 0.60
    scratchMinCoverage: float = 0.25
    scratchMinLen: float = 160.0
    codeZNg: float = 3.5
    checkCode: bool = False
    enabled: bool = True
    invert: bool = False
    checkMissing: bool = True
    codeEccMin: float = 0.55
    missFreq: float = 0.50
    missStrokeLo: float = 0.15
    missMinArea: int = 30
    missT: float = 900.0
    checkChars: bool = True
    charMissRatio: float = 0.15
    charExtraRatio: float = 0.12
    charT: float = 1.0
    rois: list = field(default_factory=lambda: [list(r) + [True] for r in ROI_DEFAULT])
    source: str = "defaults"


def load_params(model_dir: str, view: int) -> Params:
    """对齐 loadParams()：第 1 行标量（按字段数解析），随后 4 行 ROI。"""
    p = Params()
    path = os.path.join(model_dir, f"v{view:02d}", "params.txt")
    if not os.path.exists(path):
        return p
    lines = open(path, encoding="utf-8").read().splitlines()
    toks = [float(t) for t in lines[0].split()]
    for name, v in zip(SCALAR_NAMES, toks):
        if name in INT_KEYS:
            setattr(p, name, int(v))
        elif name in BOOL_KEYS:
            setattr(p, name, bool(v))
        else:
            setattr(p, name, float(v))
    for i in range(4):
        parts = lines[1 + i].split()
        x, y, w, h = [int(float(t)) for t in parts[:4]]
        en = len(parts) > 4 and int(parts[4]) != 0
        p.rois[i] = [x, y, w, h, en]
    p.source = os.path.relpath(path, ROOT)
    return p


# ---------------------------------------------------------------------------
# 1. 图像 IO（中文路径）
# ---------------------------------------------------------------------------
def imread_unicode(path: str, flags=cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def model_file(model_dir: str, view: int, roi_name: str, suffix: str) -> str:
    """模型文件名在磁盘上有两种写法：'logo _ref.png'（kRoiNames 末尾带空格）
    与 'line1_ref.png'（无空格），两种都试。"""
    for cand in (f"{roi_name} _{suffix}.png", f"{roi_name}_{suffix}.png"):
        path = os.path.join(model_dir, f"v{view:02d}", cand)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(os.path.join(model_dir, f"v{view:02d}", f"{roi_name}_{suffix}.png"))


def pic(files: list, pid: str, view: int) -> str:
    """按命名约定找某零件某视角的图。"""
    for f in files:
        name = os.path.basename(f)
        if name.startswith(pid) and name[:-4].endswith(f"_{view:02d}"):
            return f
    raise FileNotFoundError(f"{pid} view {view}")


# ---------------------------------------------------------------------------
# 2. 算法复刻（逐函数对齐 C++）
# ---------------------------------------------------------------------------
def stroke_signal(gray: np.ndarray, rect, p: Params, return_bg=False):
    """strokeSignal()：原图 − 高斯模糊 → 极性 → 除以 q94 → 只留正残差。"""
    x, y, w, h = rect[:4]
    patch = gray[y:y + h, x:x + w].astype(np.float32)
    bg = cv2.GaussianBlur(patch, (0, 0), p.sigma)
    d = patch - bg
    if p.invert:
        d = -d
    # 512 桶直方图（四舍五入后 +256），取累计到 94% 的桶
    b = np.clip(np.rint(d).astype(np.int32) + 256, 0, 511)
    hist = np.bincount(b.ravel(), minlength=512)
    acc = np.cumsum(hist)
    idx = int(np.searchsorted(acc, int(d.size * 0.94)))
    q94 = float(idx - 256)
    scale = max(8.0, q94)
    s = d / scale
    s = np.where(s > 0, s, 0.0).astype(np.float32)
    if return_bg:
        return s, patch, bg, d, q94
    return s


def estimate_warp(s_img: np.ndarray, s_ref: np.ndarray):
    """estimateWarp()：半分辨率 findTransformECC（欧氏）→ 平移量 ×2 还原。"""
    a = cv2.resize(s_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    b = cv2.resize(s_ref, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    a8 = np.clip(a * 80.0, 0, 255).astype(np.uint8)
    b8 = np.clip(b * 80.0, 0, 255).astype(np.uint8)
    m = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    try:
        ecc, m = cv2.findTransformECC(b8, a8, m, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return None, -1.0
    m = m.copy()
    m[0, 2] *= 2.0
    m[1, 2] *= 2.0
    return m, float(ecc)


def apply_warp(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """applyWarp()：WARP_INVERSE_MAP + BORDER_REPLICATE。"""
    return cv2.warpAffine(s, m, (s.shape[1], s.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def usable_mask(shape, p: Params) -> np.ndarray:
    u = np.full(shape, 255, np.uint8)
    h, w = shape
    if h > 2 * p.edgeTB and w > 2 * p.edgeLR:
        u[:p.edgeTB, :] = 0
        u[h - p.edgeTB:, :] = 0
        u[:, :p.edgeLR] = 0
        u[:, w - p.edgeLR:] = 0
    return u


def score_appendage(aligned: np.ndarray, extra: np.ndarray):
    """scoreAppendage()：面积>=15 且 PCA sqrt(L1/L2)>2.2 的连通域，取 面积×平均S 最大。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(extra, connectivity=8)
    best, details, comps = 0.0, [], []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < 15:
            continue
        ys, xs = np.nonzero(labels == k)
        data = np.stack([xs, ys], axis=1).astype(np.float32)
        ev = np.linalg.eigvalsh(np.cov(data, rowvar=False))
        l1, l2 = max(float(ev[-1]), 1e-6), max(float(ev[0]), 1e-6)
        elong = math.sqrt(l1 / l2)
        mean_s = float(aligned[ys, xs].mean())
        rec = dict(area=area, elong=elong, peak=mean_s, accepted=elong > 2.2,
                   x=int(xs.min()), y=int(ys.min()))
        if elong > 2.2:
            best = max(best, area * mean_s)
        details.append(rec)
        comps.append((k, area, elong))
    return best, details, comps


def score_scratch(aligned: np.ndarray, weak: np.ndarray, p: Params):
    """scoreScratch()：HoughLinesP → 沿线采样（±4px 能量、±1px 密度）→ 长度门槛。"""
    lines = cv2.HoughLinesP(weak, 1, math.pi / 180.0, 20, 30.0, 60.0)
    BAND = 9
    best, severe = 0.0, 0.0
    rec = dict(coverage=0.0, length=0.0, meanS=0.0, n_lines=0, n_dense=0)
    kept = []
    if lines is None:
        return severe, best, rec, kept
    rec["n_lines"] = int(len(lines))
    for ln in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in ln]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1.0:
            continue
        steps = max(2, int(length))
        energy, covered = 0.0, 0
        for t in range(steps + 1):
            r = t / steps
            x = int(x1 + (x2 - x1) * r)
            yy_c = int(y1 + (y2 - y1) * r)
            hit = False
            for dy in range(-(BAND // 2), BAND // 2 + 1):
                yy = yy_c + dy
                if yy < 0 or yy >= weak.shape[0]:
                    continue
                xx = min(max(x, 0), weak.shape[1] - 1)
                if weak[yy, xx] == 0:
                    continue
                energy += max(float(aligned[yy, xx]) - p.strokeLo, 0.0)
                if -1 <= dy <= 1:
                    hit = True
            if hit:
                covered += 1
        coverage = covered / (steps + 1)
        if coverage < p.scratchMinCoverage:
            continue
        rec["n_dense"] += 1
        sc = energy * length
        if length >= p.scratchMinLen:
            severe = max(severe, sc)
            kept.append((x1, y1, x2, y2, length, coverage, sc))
        if sc > best:
            best = sc
            rec.update(coverage=coverage, length=length,
                       meanS=energy / max(1.0, float(covered)))
    kept.sort(key=lambda t: -t[6])
    return severe, best, rec, kept[:3]


def measure_code(s_img: np.ndarray, p: Params):
    """measureCode()：二值化+开运算 → 距离变换 → 宽度中位数 → 超粗像素占比。"""
    binary = (s_img > p.strokeHi).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fg = int(cv2.countNonZero(binary))
    out = dict(ink=fg / float(s_img.size), fg=fg, widthMedian=0.0, widthMax=0.0,
               thickRatio=0.0, thickMask=None)
    if fg < 60:
        return out
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    widths = dist[binary > 0] * 2.0
    med = float(np.median(widths))
    w_thr = max(1.6 * max(1.0, med), max(1.0, med) + 3.0)
    thick = ((binary > 0) & (dist * 2.0 > w_thr)).astype(np.uint8) * 255
    out.update(widthMedian=med, widthMax=float(widths.max()),
               thickRatio=float(np.count_nonzero(thick)) / fg, thickMask=thick)
    return out


# ---------------------------------------------------------------------------
# 3. 画图小工具
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def setup_cjk_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def txt(img, s, org, scale=0.6, color=(240, 240, 240), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def rrect(img, x, y, w, h, color, th=2):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, th, cv2.LINE_AA)


def norm_u8(a: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    a = a.astype(np.float32)
    v1, v2 = np.percentile(a, [lo, hi])
    if v2 <= v1:
        v2 = v1 + 1e-6
    return np.clip((a - v1) / (v2 - v1) * 255.0, 0, 255).astype(np.uint8)


def colorize(s: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(norm_u8(s), cv2.COLORMAP_JET)


def roi_tile_with_outside(gray, rect, margin=26, scale=1.0):
    """ROI 连同周围一圈背景一起截出来，便于看清框的位置。"""
    x, y, w, h = rect[:4]
    H, W = gray.shape
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
    tile = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return tile, ((x - x0) * scale, (y - y0) * scale)


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color_bgr, alpha=0.55) -> np.ndarray:
    out = bgr.copy()
    mask = mask.astype(np.uint8)
    if out.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    sel = mask > 0
    col = np.zeros_like(out)
    col[:] = color_bgr
    out[sel] = (out[sel] * (1 - alpha) + col[sel] * alpha).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 4. matplotlib 版式：纸带式排版（每张图占满自己的格子，同一行可以放不同尺寸的图）
# ---------------------------------------------------------------------------
class Container:
    """一块排版区域：宽度 = 整图宽度，高度按内容自动累加（单位：英寸）。

    坐标系统一为「左上角原点、英寸」。面板按原始宽高比摆放，不会被拉伸，
    因此同一行里宽扁的 ROI 和竖长的剖面图可以并排，而不互相撑高。
    """

    PAD = 0.28     # 面板间距
    CAP = 0.26     # 每个面板下方的题注高度

    def __init__(self, width: float, pad_top: float = 0.0, bg="#101216"):
        self.width = width
        self.items = []          # (x, y, w, h, img, caption, sub)
        self.h = pad_top
        self.pad_top = pad_top
        self.bg = bg

    def image(self, img, caption="", sub="", new_row=False):
        """整行宽度；max_h 防止超高图把整张画布撑爆。"""
        margin = 0.012 * self.width
        avail = self.width - 2 * margin
        ih, iw = img.shape[:2]
        w = avail
        h = w * ih / iw
        max_h = 2.6
        if h > max_h:                      # 太宽的图限高，居中，并标注缩放
            h = max_h
            w = h * iw / ih
        if new_row:
            self.h += self.PAD
        if self.items:
            self.h += self.CAP * 0.6
        x = margin + (avail - w) / 2
        self.items.append((x, self.h, w, h, img, caption, sub))
        self.h += h + self.CAP
        return self

    def row(self, specs, pad_before=True, stretch=False):
        """specs: [(img, caption, sub), ...] 等宽分格，行高由最高的图决定，图按原始宽高比居中。
        stretch=True 时按行内最高图高度撑满（适合一批同尺寸的 ROI）。"""
        margin = 0.012 * self.width
        avail = self.width - 2 * margin
        n = len(specs)
        slot = (avail - self.PAD * (n - 1)) / n
        if self.items and pad_before:
            self.h += self.PAD
        hs = [min(slot * img.shape[0] / img.shape[1], 5.6) for img, _, _ in specs]
        if n == 1:
            hs[0] *= 0.86          # 单图独占一行，不要太抢眼
        h = max(hs)
        if stretch:
            hs = [h] * n
        y0 = self.h
        x = margin
        for (img, cap, sub), hi in zip(specs, hs):
            w = hi * img.shape[1] / img.shape[0]
            self.items.append((x + (slot - w) / 2, y0, w, hi, img, cap, sub))
            x += slot + self.PAD
        self.h += h + (self.CAP if any(s[1] for s in specs) else 0)
        return self

    def figure(self, title: str, subtitle: str, bottom: str = "") -> plt.Figure:
        head = 1.30 if title else 0.0
        bot = 0.30 if bottom else 0.12
        if bottom:
            n = max(1, bottom.count("\n") + 1)
            bot = 0.22 + 0.22 * n
        fig_h = head + self.h + bot
        fig = plt.figure(figsize=(self.width, fig_h), dpi=110, facecolor=self.bg)
        if title:
            fig.text(0.012, 1 - 0.12 / fig_h, title, fontsize=19, fontweight="bold",
                     color="#f2f4f8", va="top")
            fig.text(0.012, 1 - 0.58 / fig_h, subtitle, fontsize=11.5, color="#9fb0c8", va="top")
        for (x, y, w, h, img, cap, sub) in self.items:
            ax = fig.add_axes([x / self.width, 1 - (head + y + h) / fig_h,
                               w / self.width, h / fig_h])
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#39414f")
            if cap:
                ax.text(0.0, -0.035, cap, transform=ax.transAxes, fontsize=10.5,
                        color="#e6ecf5", va="top", ha="left", fontweight="bold")
            if sub:
                ax.text(1.0, -0.035, sub, transform=ax.transAxes, fontsize=9.5,
                        color="#8fa3bd", va="top", ha="right")
        if bottom:
            fig.text(0.012, 0.012, bottom, fontsize=10.3, color="#c9d6e6", va="bottom",
                     linespacing=1.55)
        return fig


class Canvas:
    """保留旧接口，内部用 Container 排版。"""

    def __init__(self, title: str, subtitle: str, ncols: int = 3, width: float = 13.5):
        self.title, self.subtitle = title, subtitle
        self.ncols = ncols
        self.width = width
        self.c = Container(width)

    def add(self, img_bgr, caption: str = "", sub: str = "", span: int = 1):
        if span >= self.ncols:
            self.c.image(img_bgr, caption, sub)
        else:
            self.c.row([(img_bgr, caption, sub)])

    def figure(self, row_h: float = 2.6, bottom: str = "") -> plt.Figure:
        return self.c.figure(self.title, self.subtitle, bottom)


def save_png(fig: plt.Figure, path: str) -> None:
    """matplotlib/PIL 打开文件偶尔会被杀软占用（WindowsError 22），重试几次。"""
    import time
    last = None
    for _ in range(6):
        try:
            fig.savefig(path, facecolor=fig.get_facecolor())
            return
        except OSError as e:          # noqa: PERF203
            last = e
            time.sleep(0.4)
    raise last


def save(fig: plt.Figure, name: str, registry: list, desc: str, extra: dict | None = None):
    path = os.path.join(OUT, name)
    save_png(fig, path)
    plt.close(fig)
    item = {"file": name, "desc": desc}
    if extra:
        item["data"] = extra
    registry.append(item)
    print("  wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------------------
# 5. 各步骤
# ---------------------------------------------------------------------------
def step1_rois(gray, p: Params, reg, view, pid):
    H, W = gray.shape
    sc = 0.55
    dim = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), None, fx=sc, fy=sc,
                     interpolation=cv2.INTER_AREA)
    colors = [(60, 220, 255), (80, 255, 120), (255, 200, 80), (255, 120, 220)]
    for i, (x, y, w, h, en) in enumerate(p.rois):
        col = colors[i] if en else (110, 110, 110)
        cv2.rectangle(dim, (int(x * sc), int(y * sc)), (int((x + w) * sc), int((y + h) * sc)),
                      col, 2)
        label = ROI_NAMES[i] + ("off" if not en else "")
        cv2.putText(dim, label, (int(x * sc) + 4, int(y * sc) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    txt(dim, f"{pid}  v{view:02d}   {W}x{H}", (10, 26), 0.62, (255, 255, 255), 2)

    c = Container(13.5, pad_top=0.0)
    c.row([(dim, "整图 + 四个 ROI", "灰 = 本视角关闭"),
           (roi_tile(gray, p.rois[0], colors[0]), "logo (480,325) 700x235", "厂商标识")])
    c.row([(roi_tile(gray, p.rois[1], colors[1]), "line1 (390,780) 870x180 型号行", "启用"),
           (roi_tile(gray, p.rois[2], colors[2]), "line2 (490,945) 660x175 批号行", "启用"),
           (roi_tile(gray, p.rois[3], colors[3]), "code (430,550) 320x260 工号", "启用")])
    fig = c.figure(f"第 1 步 · 只看刻字的地方（四个 ROI，视角 v{view:02d}）",
                   "框外的一切（引脚、塑封体边缘、台面）根本不进算法——不排除会被疯狂误报",
                   bottom=(f"尺寸提醒：描述里写 1920×1200，但 params.txt 的 ROI 与 image/ 下实拍图都是 "
                           f"{W}×{H}。ROI 是绝对像素坐标，换分辨率必须重新标定，否则框会整体偏移。"))
    save(fig, "01_rois.png", reg, "四个 ROI 在整图中的位置与各自放大图",
         {"w": int(W), "h": int(H), "pid": pid,
          "rois": {ROI_NAMES[i]: p.rois[i] for i in range(4)}, "params": p.source})


def roi_tile(gray, roi, color, margin=26, scale=1.0):
    tile, off = roi_tile_with_outside(gray, roi, margin=margin, scale=scale)
    x, y, w, h = roi[:4]
    rrect(tile, int(off[0]), int(off[1]), int(w * scale), int(h * scale), color, 2)
    return tile


def profile_png(d: np.ndarray, q94: float, col_x: int) -> np.ndarray:
    """残差剖面 + 直方图，渲染成图片后放进排版流。"""
    col = d[:, col_x]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 2.5), dpi=110, facecolor="#101216",
                            gridspec_kw={"width_ratios": [1, 1.6]})
    for ax in axs:
        ax.set_facecolor("#161a20")
        ax.tick_params(colors="#9fb0c8", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#4a5568")
    axs[0].plot(col, np.arange(len(col))[::-1], color="#7aa7ff", lw=1.1)
    axs[0].axvline(0, color="#5a6577", lw=0.8)
    axs[0].set_title("竖切一刀：残差沿 y 的剖面", color="#e6ecf5", fontsize=10.5)
    axs[0].set_xlim(-max(12, np.abs(col).max() * 1.1), max(12, np.abs(col).max() * 1.1))
    axs[0].invert_yaxis()
    lo, hi = np.percentile(d, [0.2, 99.8])
    axs[1].hist(d.ravel(), bins=140, range=(lo, hi), color="#5b8def", log=True)
    axs[1].axvline(q94, color="#ffb547", lw=2)
    axs[1].axvline(0, color="#5a6577", lw=0.8)
    axs[1].text(q94, axs[1].get_ylim()[1] * 0.3, f" q94={q94:.0f}", color="#ffb547", fontsize=9.5)
    axs[1].set_title("残差直方图（对数纵轴）：刻度 = 94% 位置", color="#e6ecf5", fontsize=10.5)
    fig.tight_layout()
    path = os.path.join(OUT, "02b_histogram.png")
    save_png(fig, path)
    plt.close(fig)
    return cv2.imread(path)


def step2_signal(gray, p: Params, reg, view, ridx=1):
    x, y, w, h, _ = p.rois[ridx]
    s, patch, bg, d, q94 = stroke_signal(gray, [x, y, w, h], p, return_bg=True)
    c = Container(13.5)
    c.row([(norm_u8(patch, 0.5, 99.5), "① 原图 ROI", f"{w}x{h} px")], pad_before=False)
    c.row([(norm_u8(bg, 0.5, 99.5), "② 模糊图 = 这一带的平均亮度", f"σ = {p.sigma:g}"),
           (norm_u8(d), "③ 原图 − 模糊图 = 残差 d", "明暗渐变没了，只剩细笔画")])
    c.row([(colorize(s), "④ 归一化后的笔画信号 S", f"q94={q94:.0f} → scale={max(8.0, q94):.0f}"),
           (profile_png(d, q94, w // 2), "⑤ 残差剖面（左）与直方图（右）", "尺子来自本图自身")],
          pad_before=False)
    bottom = ("为什么要先减再除：\n"
              "· 直接找“亮的像素”会把整片亮区域都选中；减去模糊图只留下比周围突出的细笔画。\n"
              "· 取反是为了暗刻印（凹字）：否则算法只认亮字，暗视角全部失效。\n"
              "· 除以 94 分位 = 用本图自己的残差分布当尺子；不同打光下笔画都落在 1.0 附近，"
              "六个视角才能共用 strokeHi=0.35 / strokeLo=0.20。")
    fig = c.figure(f"第 2 步 · 抹掉背景，只留“突出来的东西”（{ROI_NAMES[ridx]} 区）",
                   f"原图 − 高斯模糊(σ={p.sigma:g}) = 残差；暗刻印取反；除以 94 分位统一刻度", bottom)
    save(fig, "02_debackground.png", reg, "去背景：原图→模糊图→残差→归一化笔画信号",
         {"roi": ROI_NAMES[ridx], "sigma": p.sigma, "q94": q94, "scale": max(8.0, q94),
          "invert": p.invert, "S_max": float(s.max()), "d_min": float(d.min()),
          "d_max": float(d.max())})
    reg.append({"file": "02b_histogram.png", "desc": "残差剖面与直方图、94 分位刻度",
                "data": {"q94": q94}})


def step3_align(s_cur, ref8, p, reg, view, pid):
    ref = ref8.astype(np.float32) / 80.0
    m, ecc = estimate_warp(s_cur, ref)
    if m is None:
        m = np.eye(2, 3, dtype=np.float32)
    aligned = apply_warp(s_cur, m)
    # 人为制造放置偏移，证明 ECC 能把偏移吃回去
    shifted = np.roll(np.roll(s_cur, 7, axis=0), -5, axis=1)
    m2, ecc2 = estimate_warp(shifted, ref)
    sh_aligned = apply_warp(shifted, m2) if m2 is not None else shifted

    diff_raw = np.abs(shifted - ref)
    diff_al = np.abs(sh_aligned - ref)
    side = np.hstack([norm_u8(diff_raw, 99, 100), np.full((diff_raw.shape[0], 6), 90, np.uint8),
                      norm_u8(diff_al, 99, 100)])
    gate = p.eccMin
    c = Container(13.5)
    c.row([(colorize(ref), "模型基准 sRef", "建模时第一张 OK 样本"),
           (colorize(s_cur), f"当前图 S（{pid}）", "放置位置有抖动"),
           (colorize(aligned), "对齐后的 S", f"ECC = {ecc:.3f}")], pad_before=False)
    c.image(side, "差异图（左：未对齐　右：对齐后）", "暗 = 没有差异")
    tx, ty = float(m[0, 2]), float(m[1, 2])
    bottom = (f"实现：缩到半分辨率做 findTransformECC（MOTION_EUCLIDEAN = 平移 + 旋转，最多 100 次迭代），"
              f"再把平移量 ×2 还原。本例相对基准 dx={tx:+.2f} dy={ty:+.2f}px，ECC={ecc:.3f}，"
              f"门槛 eccMin={gate:.2f}（门控统一用 eccMin）"
              f" → {'通过，区域分保留' if ecc >= gate else '不通过，该区域分数清零'}。\n"
              f"未对齐时平均差异 {diff_raw.mean():.3f}，对齐后 {diff_al.mean():.3f}。")
    save(c.figure(f"第 3 步 · 把图对齐到模型的位置（视角 v{view:02d}）",
                  "ECC 不只是参考数字，它是否决开关：对不上时，所有差异都是错位造成的幻觉", bottom),
         "03_align.png", reg, "ECC 配准与对齐前后差异对比",
         {"ecc": ecc, "ecc_after_shift": ecc2, "dx": tx, "dy": ty, "eccMin": gate,
          "codeEccMin": p.codeEccMin,
          "mean_diff_unaligned": float(diff_raw.mean()),
          "mean_diff_aligned": float(diff_al.mean())})


def step4_masks(s_aligned, freq8, tol8, p, reg, roi_name):
    allowed = ~(tol8 > 127)
    usable = usable_mask(s_aligned.shape, p) > 0
    strong = s_aligned > p.strokeHi
    suspicious = strong & allowed & usable
    keep = (freq8.astype(np.float32) / 255.0 > p.pKeep).astype(np.uint8) * 255
    base = cv2.cvtColor(norm_u8(s_aligned), cv2.COLOR_GRAY2BGR)
    vis = overlay_mask(base, tol8 > 127, (60, 200, 60), 0.35)
    vis[~usable] = (vis[~usable] * 0.22).astype(np.uint8)
    c = Container(13.5)
    c.row([(freq8, "① 频率图 freq", "该像素在正常样本里有笔画的概率"),
           (keep, "② freq > pKeep", f"pKeep={p.pKeep:g}（>20% 才留下）"),
           (tol8, "③ 容差图 = ② 膨胀", f"椭圆核 {p.dilateK}×{p.dilateK}")], pad_before=False)
    c.row([(colorize(s_aligned), "④ 对齐后的 S", f"强 {p.strokeHi:g} / 弱 {p.strokeLo:g}"),
           (overlay_mask(base, suspicious, (0, 0, 255), 0.9),
            "⑤ 可疑 = 强 ∧ 非安全区 ∧ 非边缘", f"{int(suspicious.sum())} px"),
           (vis, "绿 = 安全区　暗 = 边缘屏蔽", f"上下 {p.edgeTB}px / 左右 {p.edgeLR}px")])
    bottom = ("这一步排除了绝大多数误报：字本身落在安全区里，不会被当成缺陷；"
              "ROI 边缘一圈亮度突变天生强烈，直接屏蔽。只有红色像素进入第 5 步分类。\n"
              "代价：安全区越宽越不容易误报，也越容易漏掉“贴着字边”的真缺陷——"
              "pKeep 与 dilateK 调的就是这个平衡。")
    save(c.figure(f"第 4 步 · 划出“安全区”，只找安全区以外的东西（{roi_name} 区）",
                  "频率图 → 阈值 → 膨胀 = 容差图；容差区里再强的残差也不算缺陷", bottom),
         "04_masks.png", reg, "频率图→容差图→安全区→可疑像素",
         {"roi": roi_name, "pKeep": p.pKeep, "dilateK": p.dilateK,
          "suspicious_px": int(suspicious.sum()),
          "tol_fraction": float(np.mean(tol8 > 127)), "freq_mean": float(freq8.mean())})


def step5_draw(s_aligned, tol8, p, reg, roi_name):
    allowed = ~(tol8 > 127)
    usable = usable_mask(s_aligned.shape, p) > 0
    extra = ((s_aligned > p.strokeHi) & allowed & usable).astype(np.uint8) * 255
    weak = ((s_aligned > p.strokeLo) & allowed & usable).astype(np.uint8) * 255
    app, details, comps = score_appendage(s_aligned, extra)
    severe, best, rec, lines = score_scratch(s_aligned, weak, p)
    base = cv2.cvtColor(norm_u8(s_aligned), cv2.COLOR_GRAY2BGR)

    vis_app = base.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(extra, connectivity=8)
    for (k, area, elong) in comps[:12]:
        col = (0, 90, 255) if elong > 2.2 else (190, 190, 70)
        vis_app = overlay_mask(vis_app, (labels == k).astype(np.uint8) * 255, col, 0.85)
        x, y = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
        cv2.putText(vis_app, f"{'!' if elong > 2.2 else 'o'} {area}/{elong:.1f}",
                    (x, max(11, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    vis_scr = base.copy()
    for (x1, y1, x2, y2, ln, cov, sc) in lines:
        cv2.line(vis_scr, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_scr, f"len={ln:.0f} cov={cov:.2f}",
                    (min(x1, x2), max(12, min(y1, y2) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    if not lines:
        txt(vis_scr, f"none passes len>={p.scratchMinLen:g}", (8, 22), 0.5, (200, 200, 200), 1)

    cm = measure_code(s_aligned, p)
    vis_code = base.copy()
    if cm["thickMask"] is not None:
        vis_code = overlay_mask(vis_code, cm["thickMask"], (0, 0, 255), 0.9)

    board = np.full((base.shape[0], base.shape[1], 3), 22, np.uint8)
    rows = sorted(details, key=lambda t: -t["area"])[:8]
    txt(board, f"appendage raw = {app:.1f}", (10, 26), 0.55, (120, 200, 255), 1)
    for i, d in enumerate(rows):
        s = (f"area={d['area']:>4}  elong={d['elong']:5.2f}  peak={d['peak']:4.2f}  "
             f"{'-> 多余笔画' if d['accepted'] else '-> 丢弃(圆形)'}")
        txt(board, s, (10, 52 + i * 20), 0.42, (225, 225, 225), 1)
    txt(board, f"Hough 候选 {rec['n_lines']} 条 / 密度合格 {rec['n_dense']} 条 / "
               f"最长合格 {rec['length']:.0f}px",
        (10, 52 + 8 * 20 + 12), 0.42, (255, 220, 120), 1)

    c = Container(13.5)
    c.row([(overlay_mask(base, extra, (0, 0, 255), 0.8), "① 多余笔画：连通域形状",
            "伸长比>2.2 = 笔画；圆疙瘩 = 污点"),
           (vis_scr, "② 划痕：Hough 直线 + 密度 + 长度", f"len ≥ {p.scratchMinLen:g}px 才计分"),
           (vis_code, "③ 笔画粗细：距离变换量宽度", f"超粗像素占 {cm['thickRatio'] * 100:.1f}%")],
          pad_before=False)
    c.row([(vis_app, "① 放大：每个连通域标 面积/伸长比", "! = 细长(过)　o = 圆形(丢)"),
           (board, "原始量一览", "下一步各自除以归一化分母")])
    bottom = ("怎么读这三张图：\n"
              "① 多余笔画只在 logo 区做——其他区域多出来的墨通常表现为划痕，形状判据会把划痕也判成“笔画”。\n"
              "② 划痕一律记录，只有长度达标的才进判定。实测严重划痕“很长但很淡”，小瑕疵“很短但很亮”，"
              "用亮度当门槛会漏掉真缺陷。\n"
              "③ 距离变换 2×dist = 局部宽度："
              "中位数 = 这颗芯片的正常粗细，超粗像素占比异常高 = 糊成一团。")
    save(c.figure("第 5 步 · 把可疑的东西分成三类（三个并行检查）",
                  "形状、长度、粗细——各自回答“这是不是真缺陷”", bottom),
         "05_classify.png", reg, "三类检查：多余笔画 / 划痕 / 笔画粗细",
         {"roi": roi_name, "appendage_raw": app, "scratch_best": best,
          "scratch_severe": severe, "scratch_cov": rec["coverage"], "scratch_len": rec["length"],
          "hough_lines": rec["n_lines"], "hough_dense": rec["n_dense"],
          "code_thickRatio": cm["thickRatio"], "code_widthMedian": cm["widthMedian"],
          "code_ink": cm["ink"]})
    return dict(appendage=app, severe=severe, best=best)


def step6_table(region_table, reg, scratchT_all, scratchT_default, pid):
    rows = len(region_table)
    board = np.full((120 + rows * 46, 1230, 3), 22, np.uint8)
    hdr = ["区域", "ECC", "门控", "多余笔画 / appendageT", "划痕 / scratchT", "笔画粗细", "=> 区域分"]
    xs = [10, 120, 240, 360, 690, 920, 1060]
    for x, h in zip(xs, hdr):
        txt(board, h, (x, 30), 0.46, (150, 200, 255), 1)
    cv2.line(board, (10, 40), (1220, 40), (70, 80, 95), 1, cv2.LINE_AA)
    for i, r in enumerate(region_table):
        y = 74 + i * 46
        gate_txt = r["why"] or ("通过" if r["reliable"] else "清零")
        vals = [r["name"] + ("" if r["enabled"] else " (off)"),
                f"{r['ecc']:.3f}" if r["enabled"] else "-",
                gate_txt, r["app_text"], r["scratch_text"], r["code_text"], f"{r['score']:.3f}"]
        for j, (x, v) in enumerate(zip(xs, vals)):
            col = (240, 240, 240)
            if j == 2 and r["enabled"] and not r["reliable"]:
                col = (90, 90, 255)
            if j == 6 and r["enabled"]:
                col = (90, 90, 255) if r["score"] >= 1.0 else (120, 220, 120)
            txt(board, v, (x, y), 0.46, col, 1)
        cv2.line(board, (10, y + 14), (1220, y + 14), (48, 54, 64), 1, cv2.LINE_AA)
    vs = max(r["score"] for r in region_table)
    txt(board, f"视角分 = 四个区域分取最大 = {vs:.3f}", (10, 74 + rows * 46 + 26), 0.56,
        (255, 205, 120), 2)

    c = Container(13.5)
    c.image(board, "区域分台账（依次：归一化 → ECC 门控 → 取最大）")
    fig = c.figure(f"第 6 步 · 算分：各自除以归一化分母，再取最大（{pid}，v00）",
                   "量纲不同的原始量没法比较；除以该视角专属的分母，全部变成 1.0 附近可比的分数",
                   "分母逐视角标定：scratchT 在六个视角间差约 5 倍（v03=90708、v05=19462），"
                   "所以“1.0 分”在不同视角代表的绝对残差量不同——视角分只在本视角内有可比性。")
    save(fig, "06_score_table.png", reg, "区域分：归一化、ECC 门控、取最大",
         {"region_scores": {r["name"]: r["score"] for r in region_table},
          "view_score": vs, "scratchT_all_views": scratchT_all,
          "scratchT_default": scratchT_default})

    cats = ["v%02d" % i for i in range(6)]
    vals = [scratchT_all[k] for k in cats]
    fig, ax = plt.subplots(figsize=(11, 2.8), dpi=110, facecolor="#101216")
    ax.set_facecolor("#161a20")
    bars = ax.bar(cats, vals, color="#3f7fd0")
    ax.axhline(scratchT_default, color="#e05a5a", ls="--", lw=1.4)
    ax.text(-0.45, scratchT_default * 0.86, f"代码默认 {scratchT_default:,.0f}",
            color="#e05a5a", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:,.0f}", ha="center",
                color="#e6ecf5", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) * 0.6, max(vals) * 2.2)
    ax.set_ylabel("scratchT（对数）", color="#9fb0c8")
    ax.tick_params(colors="#9fb0c8")
    ax.grid(axis="y", alpha=0.15)
    ax.set_title("划痕归一化分母逐视角标定：六视角差约 %.0f 倍" % (max(vals) / max(min(vals), 1)),
                 color="#e6ecf5", fontsize=11.5)
    for sp in ax.spines.values():
        sp.set_color("#4a5568")
    fig.tight_layout()
    path = os.path.join(OUT, "06b_scratchT.png")
    save_png(fig, path)
    plt.close(fig)
    reg.append({"file": "06b_scratchT.png", "desc": "各视角 scratchT 对比",
                "data": {"scratchT_all_views": scratchT_all}})
    print("  wrote", os.path.relpath(path, ROOT))


def step7_product(per_view_scores, reg, worst_text, pid):
    sc = per_view_scores
    fig, ax = plt.subplots(figsize=(11, 3.3), dpi=110, facecolor="#101216")
    ax.set_facecolor("#161a20")
    bars = ax.bar([f"v{v:02d}" for v in range(len(sc))], sc,
                  color=["#e05a5a" if s >= 1.0 else "#3f7fd0" for s in sc])
    ax.axhline(1.0, color="#ffb547", lw=2, ls="--")
    ax.text(len(sc) - 0.5, 1.03, "NG 门槛 1.0", color="#ffb547", ha="right", fontsize=10)
    for b, s in zip(bars, sc):
        ax.text(b.get_x() + b.get_width() / 2, s + max(sc) * 0.02, f"{s:.2f}", ha="center",
                color="#e6ecf5", fontsize=10)
    ax.set_ylabel("视角分", color="#9fb0c8")
    ax.set_ylim(0, max(1.35, max(sc) * 1.25))
    ax.tick_params(colors="#9fb0c8")
    ax.grid(axis="y", alpha=0.15)
    ax.set_title(f"{pid}：六个视角各自独立打分 → 零件分 = 最大值 = {max(sc):.3f}", color="#e6ecf5",
                 fontsize=11.5)
    for s in ax.spines.values():
        s.set_color("#4a5568")
    fig.tight_layout()
    path = os.path.join(OUT, "07_product_scores.png")
    save_png(fig, path)
    plt.close(fig)
    reg.append({"file": "07_product_scores.png", "desc": "六视角分数与最终判定",
                "data": {"view_scores": sc, "product_score": max(sc),
                         "verdict": "NG" if max(sc) >= 1.0 else "OK", "worst": worst_text}})
    print("  wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------------------
# 5b. 笔画粗细演示（该分支在当前模型里 checkCode=0 关闭，单独跑演示图）
# ---------------------------------------------------------------------------
def step5b_code(gray, pv, p_model, model_dir, v, reg, pid):
    x, y, w, h, _ = pv.rois[3]
    s = stroke_signal(gray, [x, y, w, h], pv)
    cm = measure_code(s, pv)
    binary = (s > pv.strokeHi).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    widths = dist[binary > 0] * 2.0
    med = cm["widthMedian"]
    w_thr = max(1.6 * max(1.0, med), max(1.0, med) + 3.0)

    c = Container(13.5)
    vis = cv2.cvtColor(norm_u8(s), cv2.COLOR_GRAY2BGR)
    if cm["thickMask"] is not None:
        vis = overlay_mask(vis, cm["thickMask"], (0, 0, 255), 0.9)
    c.row([(vis, "code 区 S + 超粗像素（红）", f"宽度中位数 {med:.1f}px，阈值 {w_thr:.1f}px"),
           (binary, "二值化 + 3×3 开运算", f"笔画像素 {cm['fg']} px"),
           (cv2.applyColorMap(norm_u8(dist, 99.9, 100), cv2.COLORMAP_INFERNO),
            "距离变换（越亮离边界越远）", "2 × dist = 局部笔画宽度")],
          pad_before=False)

    fig, ax = plt.subplots(figsize=(9.0, 2.2), dpi=110, facecolor="#101216")
    ax.set_facecolor("#161a20")
    ax.hist(widths, bins=60, color="#5b8def")
    ax.axvline(med, color="#ffb547", lw=2)
    ax.axvline(w_thr, color="#e05a5a", lw=2, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.92, f" 中位数 {med:.1f}", color="#ffb547", fontsize=9.5)
    ax.text(w_thr, ax.get_ylim()[1] * 0.75, f" 超粗阈值 {w_thr:.1f}", color="#e05a5a", fontsize=9.5)
    ax.set_xlabel("局部笔画宽度 (px)", color="#9fb0c8")
    ax.tick_params(colors="#9fb0c8", labelsize=8.5)
    ax.grid(axis="y", alpha=0.15)
    ax.set_title("笔画宽度分布：中位数 = 这颗芯片的“正常粗细”", color="#e6ecf5", fontsize=10.5)
    for sp in ax.spines.values():
        sp.set_color("#4a5568")
    fig.tight_layout()
    tmp = os.path.join(OUT, "_codewidth.png")
    save_png(fig, tmp)
    plt.close(fig)
    c.image(cv2.imread(tmp), "宽度直方图", "红色阈值右侧 = 超粗像素")
    os.remove(tmp)

    cs = os.path.join(model_dir, f"v{v:02d}", "code_stats.txt")
    mean = sd = None
    if os.path.exists(cs):
        vals = [float(t) for t in open(cs, encoding="utf-8").read().split()]
        if len(vals) == 4:
            mean, sd = vals[2], vals[3]
    z = (cm["thickRatio"] - mean) / max(sd, 0.005) if mean is not None else float("nan")
    z_dead = max(0.5, pv.codeZNg - 1.5)
    sc = max(0.0, z - z_dead) / 1.5
    bottom = (f"thickRatio = 超粗像素 / 笔画像素 = {cm['thickRatio'] * 100:.1f}%\n"
              f"模型 code_stats.txt：thickMean={mean:.4f}，thickStd={sd:.4f} → z={z:.2f}；"
              f"死区 zDead=max(0.5, {pv.codeZNg:g}−1.5)={z_dead:.1f} → 崩缺分={sc:.2f}\n"
              f"注意：当前 params.txt 里 checkCode={1 if pv.checkCode else 0}，本分支"
              f"{'启用' if pv.checkCode else '关闭'}；此处用默认规则单独演示（PID {pid}，v{v:02d}）。")
    save(c.figure("第 5 步附 · 笔画粗细：不比模板，只比笔画粗细", "笔画粗细可度量的区域", bottom),
         "05b_code.png", reg, "笔画粗细：距离变换宽度与 z-score",
         {"thickRatio": cm["thickRatio"], "widthMedian": med, "widthThr": w_thr,
          "z": z, "zDead": z_dead, "score": sc, "checkCode": pv.checkCode})
    return sc


def step7_views(live: dict, reg, pid: str, disabled: dict):
    """同一颗芯片 6 个视角各自能算出的分（只看判定链里生效的区域）。"""
    views = sorted(live)
    sc = [live[v] for v in views]
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=110, facecolor="#101216")
    ax.set_facecolor("#161a20")
    bars = ax.bar([f"v{v:02d}" for v in views], sc,
                  color=["#e05a5a" if s >= 1.0 else "#3f7fd0" for s in sc])
    ax.axhline(1.0, color="#ffb547", lw=2, ls="--")
    ax.text(len(sc) - 0.5, 1.03, "NG 门槛 1.0", color="#ffb547", ha="right", fontsize=10)
    top = max(1.3, max(sc) * 1.3)
    for b, s in zip(bars, sc):
        ax.text(b.get_x() + b.get_width() / 2, min(s + top * 0.02, top * 0.96), f"{s:.2f}",
                ha="center", color="#e6ecf5", fontsize=10)
    ax.set_ylabel("视角分", color="#9fb0c8")
    ax.set_ylim(0, top)
    ax.tick_params(colors="#9fb0c8")
    ax.grid(axis="y", alpha=0.15)
    ax.set_title(f"{pid}：六个视角各自独立打分 → 零件分 = 最大值 = {max(sc):.3f}"
                 f"（{'NG' if max(sc) >= 1.0 else 'OK'}）", color="#e6ecf5", fontsize=11.5)
    for s in ax.spines.values():
        s.set_color("#4a5568")
    off = "、".join(sorted(disabled)) if disabled else "无"
    fig.text(0.01, 0.012, f"enabled=0、直接跳过的区域：{off}"
                          "　　与 C++ 评估 out/baseline_3view.txt 中 v00/v02/v05 的数值一致",
             color="#8fa3bd", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(OUT, "07_product_scores.png")
    save_png(fig, path)
    plt.close(fig)
    reg.append({"file": "07_product_scores.png", "desc": "六视角分数与最终判定",
                "data": {"view_scores": live, "product_score": max(sc),
                         "verdict": "NG" if max(sc) >= 1.0 else "OK",
                         "disabled_rois": sorted(disabled)}})
    print("  wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    print("CJK font:", setup_cjk_font())
    MODEL = os.path.join(ROOT, "models", "char")
    files = sorted(glob.glob(os.path.join(ROOT, "image", "ng", "*.tif")) +
                   glob.glob(os.path.join(ROOT, "image", "ok", "*.tif")))
    VIEW, PID_NG, PID_OK = 0, "0110", "0116"
    reg = []
    gray = imread_unicode(pic(files, PID_NG, VIEW))
    p0 = load_params(MODEL, VIEW)
    print(f"params {p0.source}: sigma={p0.sigma} scratchT={p0.scratchT} invert={p0.invert}")

    step1_rois(gray, p0, reg, VIEW, PID_NG)
    step2_signal(gray, p0, reg, VIEW, ridx=1)

    per_view_scores = [0.0] * 6
    region_table = None
    scratchT_all = {}
    code_present = True          # code_stats.txt 是否存在（缺失时要回避该分支）
    enabled_off = {}             # 被关掉的区域
    for v in range(6):
        pv = load_params(MODEL, v)
        scratchT_all[f"v{v:02d}"] = round(pv.scratchT, 1)
        g = imread_unicode(pic(files, PID_NG, v))
        cs = os.path.join(MODEL, f"v{v:02d}", "code_stats.txt")
        code_mean = code_sd = None
        if os.path.exists(cs):
            vals = [float(t) for t in open(cs, encoding="utf-8").read().split()]
            if len(vals) == 4:
                code_mean, code_sd = vals[2], vals[3]
        table = []
        for i in range(4):
            x, y, w, h, en = pv.rois[i]
            row = dict(name=ROI_NAMES[i], enabled=en, ecc=-1.0, reliable=False,
                       app_text="-", scratch_text="-", code_text="关闭",
                       score=0.0, severe_raw=0.0, present=True, why="")
            if en:
                ref8 = imread_unicode(model_file(MODEL, v, ROI_NAMES[i], "ref"))
                tol8 = imread_unicode(model_file(MODEL, v, ROI_NAMES[i], "tol"))
                s = stroke_signal(g, [x, y, w, h], pv)
                m, ecc = estimate_warp(s, ref8.astype(np.float32) / 80.0)
                aligned = apply_warp(s, m) if m is not None else s
                row["ecc"] = ecc
                gate_v = pv.codeEccMin if i == 3 else pv.eccMin
                row["why"] = "通过" if ecc >= gate_v else "清零"
                if aligned.shape == tol8.shape:
                    allowed = ~(tol8 > 127)
                    usable = usable_mask(aligned.shape, pv) > 0
                    extra = ((aligned > pv.strokeHi) & allowed & usable).astype(np.uint8) * 255
                    weak = ((aligned > pv.strokeLo) & allowed & usable).astype(np.uint8) * 255
                    app = score_appendage(aligned, extra)[0] if i == 0 else 0.0
                    severe = score_scratch(aligned, weak, pv)[0]
                    row["severe_raw"] = severe
                    region_score = severe / pv.scratchT
                    row["scratch_text"] = f"{severe:.0f} / {pv.scratchT:.0f} = {severe / pv.scratchT:.2f}"
                    if i == 0:
                        row["app_text"] = f"{app:.0f} / {pv.appendageT:.0f} = {app / pv.appendageT:.2f}"
                        region_score = max(region_score, app / pv.appendageT)
                    else:
                        row["app_text"] = "—（仅 logo）"
                    row["code_text"] = "—"
                    if i == 3:
                        if pv.checkCode:
                            cm = measure_code(s, pv)
                            if code_mean is not None:
                                z = (cm["thickRatio"] - code_mean) / max(code_sd, 0.005)
                                z_dead = max(0.5, pv.codeZNg - 1.5)
                                row["code_text"] = f"z={z:.2f} → {max(0.0, z - z_dead) / 1.5:.2f}"
                                code_present = True
                            else:
                                row["code_text"] = "0（缺 code_stats）"
                                code_present = False
                        else:
                            row["code_text"] = "0（checkCode=0）"
                    gate = pv.codeEccMin if i == 3 else pv.eccMin
                    row["reliable"] = ecc >= gate
                    if not row["reliable"]:
                        region_score = 0.0
                        row["why"] = "清零"
                    row["score"] = region_score
            else:
                row["why"] = "该视角下此区域关闭"
                enabled_off[f"v{v:02d}/{ROI_NAMES[i]}"] = True
            table.append(row)
        per_view_scores[v] = max(r["score"] for r in table)
        if v == VIEW:
            region_table = table
            x, y, w, h, _ = pv.rois[1]
            s1 = stroke_signal(g, [x, y, w, h], pv)
            step3_align(s1, imread_unicode(model_file(MODEL, v, "line1", "ref")), pv, reg, v, PID_NG)
            x, y, w, h, _ = pv.rois[0]
            s0 = stroke_signal(g, [x, y, w, h], pv)
            m0, _ = estimate_warp(s0, imread_unicode(model_file(MODEL, v, "logo", "ref")).astype(np.float32) / 80.0)
            al0 = apply_warp(s0, m0) if m0 is not None else s0
            f0 = imread_unicode(model_file(MODEL, v, "logo", "freq"))
            t0 = imread_unicode(model_file(MODEL, v, "logo", "tol"))
            step4_masks(al0, f0, t0, pv, reg, "logo")
            step5_draw(al0, t0, pv, reg, "logo")
            pv_code = Params(**{f: getattr(pv, f) for f in pv.__dataclass_fields__})
            pv_code.checkCode = True          # 仅演示
            pv_code.rois = [list(r) for r in pv.rois]
            step5b_code(g, pv_code, pv, MODEL, v, reg, PID_NG)

    step6_table(region_table, reg, scratchT_all, 22000.0, PID_NG)
    worst = max(region_table, key=lambda r: r["score"])["name"]
    step7_product(per_view_scores, reg, f"v00/{worst}", PID_NG)          # 每个视角的判定（六视角图）

    # ---- 同一颗芯片各视角实际能算出分的情况（只看判定链里真正生效的区域）
    live = {}
    for v in range(6):
        pv = load_params(MODEL, v)
        g = imread_unicode(pic(files, PID_NG, v))
        cs = os.path.join(MODEL, f"v{v:02d}", "code_stats.txt")
        code_mean = code_sd = None
        if os.path.exists(cs):
            vals = [float(t) for t in open(cs, encoding="utf-8").read().split()]
            if len(vals) == 4:
                code_mean, code_sd = vals[2], vals[3]
        best = 0.0
        for i in range(4):
            x, y, w, h, en = pv.rois[i]
            if not en:
                continue
            ref8 = imread_unicode(model_file(MODEL, v, ROI_NAMES[i], "ref"))
            tol8 = imread_unicode(model_file(MODEL, v, ROI_NAMES[i], "tol"))
            s = stroke_signal(g, [x, y, w, h], pv)
            m, ecc = estimate_warp(s, ref8.astype(np.float32) / 80.0)
            aligned = apply_warp(s, m) if m is not None else s
            allowed = ~(tol8 > 127)
            usable = usable_mask(aligned.shape, pv) > 0
            extra = ((aligned > pv.strokeHi) & allowed & usable).astype(np.uint8) * 255
            weak = ((aligned > pv.strokeLo) & allowed & usable).astype(np.uint8) * 255
            sc = score_scratch(aligned, weak, pv)[0] / pv.scratchT
            if i == 0:
                sc = max(sc, score_appendage(aligned, extra)[0] / pv.appendageT)
            if i == 3 and pv.checkCode and code_mean is not None:
                cm = measure_code(s, pv)
                z = (cm["thickRatio"] - code_mean) / max(code_sd, 0.005)
                sc = max(sc, max(0.0, z - max(0.5, pv.codeZNg - 1.5)) / 1.5)
            gate = pv.codeEccMin if i == 3 else pv.eccMin
            if ecc < gate:
                sc = 0.0
            best = max(best, sc)
        live[v] = best
    step7_views(live, reg, PID_NG, enabled_off)

    meta = dict(view=VIEW, pid_ng=PID_NG, pid_ok=PID_OK,
                per_view_scores=per_view_scores, live_view_scores=live,
                region_table=region_table, scratchT_all_views=scratchT_all,
                disabled_rois=list(enabled_off))
    with open(os.path.join(OUT, "figures.json"), "w", encoding="utf-8") as f:
        json.dump({"figures": reg, "meta": meta}, f, ensure_ascii=False, indent=2)
    print("done:", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    sys.exit(main())
