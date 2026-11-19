"""hood-keychain: フードをかぶった子のアイコン絵 (source.png) をキーホルダーにする。

1色でしか刷れないので、cat-keychain と同じく黒い線を深い溝にして描く。
色の塗り分け (フード・髪・肌・紫のメッシュ) は再現しない (持ち主の判断)。
口の中だけは、輪郭の溝より浅く一段下げる (cat-keychain で決めた作り方)。
"""

from pathlib import Path

import cv2
import numpy as np
from build123d import *

from v3d import fit, trace

SOURCE = Path(__file__).with_name("source.png")
PREVIEW_COLOR = "#46bccf"     # 確認用の絵の色 (絵の水色に寄せた仮の色)

# 板の形: "square" (角丸の四角、絵がまるごと入る) / "circle" (丸、四隅は切れる) /
#         "outline" (背景を切り落として、キャラの輪郭どおり)
SHAPE = "circle"

# ---- 寸法 (mm) ----
ART = 50.0            # 絵の一辺。板は縁の分だけこれより大きい
THICK = 2.6           # 板の厚み
FLOOR = 0.4           # 線の溝の底に残す薄皮 (2 層)
MOUTH_LEVEL = 1.3     # 口の中の高さ (板の表面より下、溝の底より上)
MIN_LINE = 1.0        # 溝の幅 (絵の線は 0.6 mm 相当しかないので、ここまで太らせる)
MIN_WEB = 0.8         # 溝と溝の間に残す壁の最小幅。これより細い壁は溝にまとめる
MIN_ISLAND = 1.5      # 溝に囲まれた島の最小面積 (mm²)。これより小さい島は溝にまとめる
RIM = 1.6             # 絵の外側に付ける縁 (四角・丸)
CORNER = 4.0          # 四角の角の丸み
OUTLINE_RIM = 1.4     # 輪郭の外側に付ける縁 (輪郭どおり)
RING_OUTER = 8.5      # 金具を通す輪の外径
RING_HOLE = 3.5       # 輪の穴の仕上がり径 (ボールチェーン・二重リング用)
RING_OVERLAP = 2.0    # 輪を板にどれだけ食い込ませるか

# ---- 絵の読み取り ----
UPSAMPLE = 4          # 輪郭を滑らかにするため、拡大してから処理する
PAD_MM = 10.0         # 縁と輪がはみ出せるよう、絵の周りに足す余白

PRINT = {"copies": {"hood-keychain": 2}}     # 2 個並べて刷る (持ち主の注文)


def _read():
    """絵を拡大して、黒い線・赤 (口の中)・白っぽい所 (背景と髪と肌) のマスクに分ける。"""
    img = cv2.imread(str(SOURCE))
    img = cv2.resize(img, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))

    def mask(condition):
        return np.where(condition, 255, 0).astype(np.uint8)

    return mask(v < 100), mask(((h < 12) | (h > 168)) & (s > 100) & (v > 150)), mask((s < 60) & (v > 200))


def _touches_border(mask) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _layout(shape: str = SHAPE):
    """板・溝・口の領域 (マスク) と、寸法の換算を決める。"""
    black, red, pale = _read()
    n = black.shape[0]
    mm = ART / n                           # 拡大後の 1 px が何 mm か
    px = 1 / mm

    ink = trace.smooth(black, 0.1 * px)
    ink = trace.smooth(trace.thicken_evenly(ink, MIN_LINE * px), 0.1 * px)
    groove = trace.smooth(trace.close_gaps(ink, MIN_WEB / 2 * px), 0.08 * px)
    for island in trace.components(cv2.bitwise_not(groove)):
        if (island > 0).sum() * mm * mm < MIN_ISLAND:
            groove |= island

    # 口の中 = 溝で囲まれた領域のうち、赤く塗られているもの
    mouth = np.zeros_like(groove)
    for region in trace.enclosed_regions(groove):
        if (red[region > 0] > 0).mean() > 0.5:
            mouth |= region
    assert mouth.any(), "口が見つからない"

    # 絵のうち板に載せる範囲 (art) を決める。座標はまだ余白なし
    if shape == "square":
        r = int(round(CORNER * px))
        art = np.zeros_like(groove)
        cv2.rectangle(art, (r, 0), (n - 1 - r, n - 1), 255, cv2.FILLED)
        cv2.rectangle(art, (0, r), (n - 1, n - 1 - r), 255, cv2.FILLED)
        for cx in (r, n - 1 - r):
            for cy in (r, n - 1 - r):
                cv2.circle(art, (cx, cy), r, 255, cv2.FILLED)
        rim = RIM
    elif shape == "circle":
        art = np.zeros_like(groove)
        cv2.circle(art, (n // 2, n // 2), n // 2, 255, cv2.FILLED)
        rim = RIM
    elif shape == "outline":
        # 背景 = 絵の端に触れている白い領域 (髪も白いが、線で囲まれていて端に触れない)
        background = np.zeros_like(groove)
        for region in trace.components(cv2.bitwise_not(groove)):
            if _touches_border(region) and (pale[region > 0] > 0).mean() > 0.8:
                background |= region
        assert background.any(), "背景が見つからない"
        art = trace.fill_holes(trace.components(cv2.bitwise_not(background))[0])
        # 背景との境の線は溝として残す。その外側 (背景側) にはみ出した分だけ切る
        rim = OUTLINE_RIM
    else:
        raise ValueError(f"SHAPE が不正: {shape}")

    pad = int(round(PAD_MM * px))
    art, groove, mouth = (np.pad(m, pad) for m in (art, groove & art, mouth & art))
    plate = trace.grow(art, rim * px)

    # 金具を通す輪: 板の上辺の中央 (絵が左右ほぼ対称なので、中央で真っすぐ吊れる)
    r = RING_OUTER / 2 * px
    ring_x = art.shape[1] // 2
    ring_y = int(round(np.nonzero(plate[:, ring_x])[0].min() - r + RING_OVERLAP * px))
    cv2.circle(plate, (ring_x, ring_y), int(round(r)), 255, cv2.FILLED)
    plate = trace.fill_holes(trace.components(trace.smooth(plate, 0.3 * px))[0])
    assert not (groove & ~trace.shrink(plate, 1.0 * px)).any(), "溝が板の縁に近すぎる"

    ys, xs = np.nonzero(plate)
    center = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    by, bx = np.nonzero(plate & ~groove)
    return {
        "plate": plate,
        "groove": groove,
        "top": plate & ~groove & ~mouth,   # 板の表面まである所
        "mouth": mouth,
        "mm": mm,
        "center": center,
        "ring": ((ring_x - center[0]) * mm, (center[1] - ring_y) * mm),
        "balance": ((bx.mean() - center[0]) * mm, (center[1] - by.mean()) * mm),
    }


def _prism(mask, layout, z0: float, z1: float):
    shapes = trace.faces(mask, layout["mm"], layout["center"], detail_mm=0.015)
    return extrude(Sketch() + shapes, amount=z1 - z0).move(Location((0, 0, z0)))


def build():
    """絵の面を上にして寝かせて刷る (裏は平ら)。"""
    L = _layout()
    part = _prism(L["plate"], L, 0, FLOOR)
    part += _prism(L["top"], L, FLOOR, THICK)
    part += _prism(L["mouth"], L, FLOOR, MOUTH_LEVEL)
    hole = Cylinder(fit.hole(RING_HOLE) / 2, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return part - hole.move(Location((*L["ring"], 0)))
