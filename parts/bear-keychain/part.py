"""bear-keychain: 黒地に白いクマの顔のアイコン絵 (source.png) をキーホルダーにする。

絵は白と黒の 2 階調だけなので、1色刷りでは高さ 2 段で描き分ける:
白い所 (鼻づらと目) と黒い所 (地、鼻、口の線) のどちらかを RELIEF だけ高くする。
"""

from pathlib import Path

import cv2
import numpy as np
from build123d import *

from v3d import fit, trace

SOURCE = Path(__file__).with_name("source.png")
PREVIEW_COLOR = "#c9ccd1"     # 確認用の絵の色 (仮の色)

SHAPE = "circle"      # 板の形: "circle" (丸) / "square" (角丸の四角)
RAISED = True         # True: 白い所を盛り上げる / False: 白い所をくぼませる
CROP = 460            # 絵のうち板に載せる範囲 (中央の正方形の一辺、元の絵の画素)。460 で絵の全体

# ---- 寸法 (mm) ----
SIZE = 53.0           # 板の直径 (四角なら一辺)。hood-keychain とそろえた
BASE = 1.8            # 低い所の厚み
RELIEF = 1.2          # 高い所と低い所の段差
MIN_LINE = 1.0        # 口の線の最小幅 (絵のままだと 0.5 mm ほどで細すぎるので太らせる)
CORNER = 4.0          # 四角の角の丸み
EDGE = 1.0            # 白い所が板の縁にこれ以上近づかないこと
RING_OUTER = 8.5      # 金具を通す輪の外径
RING_HOLE = 3.5       # 輪の穴の仕上がり径 (ボールチェーン・二重リング用)
RING_OVERLAP = 2.0    # 輪を板にどれだけ食い込ませるか

# ---- 絵の読み取り ----
UPSAMPLE = 4          # 輪郭を滑らかにするため、拡大してから処理する
PAD_MM = 10.0         # 輪がはみ出せるよう、絵の周りに足す余白

PRINT = {"copies": {"bear-keychain": 2}}     # 2 個並べて刷る (持ち主の注文)


def _read():
    """絵を拡大して、白い所のマスクにする。"""
    gray = cv2.imread(str(SOURCE), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    return np.where(gray > 128, 255, 0).astype(np.uint8)


def _layout(shape: str = SHAPE, crop: float = CROP):
    """板と白い所の領域 (マスク)、寸法の換算を決める。"""
    white = _read()
    n = white.shape[0]
    side = crop * UPSAMPLE                 # 板に載せる範囲の一辺 (拡大後の画素)
    mm = SIZE / side
    px = 1 / mm

    # 顔の中の黒 (鼻と口) = 白に囲まれていて絵の端に触れない黒。細い口の線だけ太らせる
    ink = np.zeros_like(white)
    for region in trace.enclosed_regions(white):
        ink |= region
    assert ink.any(), "鼻と口が見つからない"
    ink = trace.smooth(trace.thicken_thin_lines(trace.smooth(ink, 0.08 * px), MIN_LINE * px), 0.08 * px)
    white = trace.smooth(white, 0.08 * px) & ~ink

    # 板 (余白なしの座標)。絵の中央に合わせる
    plate = np.zeros_like(white)
    c, half = n // 2, int(round(side / 2))
    if shape == "circle":
        cv2.circle(plate, (c, c), half, 255, cv2.FILLED)
    elif shape == "square":
        r = int(round(CORNER * px))
        lo, hi = c - half, c + half - 1
        cv2.rectangle(plate, (lo + r, lo), (hi - r, hi), 255, cv2.FILLED)
        cv2.rectangle(plate, (lo, lo + r), (hi, hi - r), 255, cv2.FILLED)
        for x in (lo + r, hi - r):
            for y in (lo + r, hi - r):
                cv2.circle(plate, (x, y), r, 255, cv2.FILLED)
    else:
        raise ValueError(f"SHAPE が不正: {shape}")
    assert not (white & ~trace.shrink(plate, EDGE * px)).any(), "白い所が板の縁に近すぎる (CROP が小さすぎる)"

    pad = int(round(PAD_MM * px))
    plate, white = np.pad(plate, pad), np.pad(white, pad)

    # 金具を通す輪: 板の上辺の中央 (絵が左右対称なので、中央で真っすぐ吊れる)
    r = RING_OUTER / 2 * px
    ring_x = plate.shape[1] // 2
    ring_y = int(round(np.nonzero(plate[:, ring_x])[0].min() - r + RING_OVERLAP * px))
    cv2.circle(plate, (ring_x, ring_y), int(round(r)), 255, cv2.FILLED)
    plate = trace.fill_holes(trace.components(trace.smooth(plate, 0.3 * px))[0])

    ys, xs = np.nonzero(plate)
    center = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    return {
        "plate": plate,
        "white": white,
        "mm": mm,
        "center": center,
        "ring": ((ring_x - center[0]) * mm, (center[1] - ring_y) * mm),
    }


def _prism(mask, layout, z0: float, z1: float):
    shapes = trace.faces(mask, layout["mm"], layout["center"], detail_mm=0.015)
    return extrude(Sketch() + shapes, amount=z1 - z0).move(Location((0, 0, z0)))


def build():
    """絵の面を上にして寝かせて刷る (裏は平ら)。"""
    L = _layout()
    high = L["white"] if RAISED else L["plate"] & ~L["white"]
    part = _prism(L["plate"], L, 0, BASE) + _prism(high, L, BASE, BASE + RELIEF)
    hole = Cylinder(fit.hole(RING_HOLE) / 2, BASE + RELIEF, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return part - hole.move(Location((*L["ring"], 0)))
