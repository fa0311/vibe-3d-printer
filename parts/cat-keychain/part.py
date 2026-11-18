"""cat-keychain: 持ち主が描いた猫の絵 (source.jpg) をキーホルダーにする。

1色でしか刷れないので、色の代わりに深さで描き分ける (5案を見比べて持ち主が選んだ形):
- 板: 絵の線を一回り太らせた輪郭どおりの形 (アクリルキーホルダーのような「ダイカット」)
- 黒い線: 深い溝にする。裏に薄皮 FLOOR だけ残す (完全に抜くと、耳やひげの先で板が
  幅 1 mm ほどの橋だけでつながる形になって折れやすい)
- 口の中 (赤い所): 輪郭の溝より浅く、一段だけ下げる。輪郭と中身の区別を深さで出す
"""

from pathlib import Path

import cv2
import numpy as np
from build123d import *

from v3d import fit, trace

SOURCE = Path(__file__).with_name("source.jpg")
PREVIEW_COLOR = "#78c85a"     # 確認用の絵の色 (絵の緑に寄せた仮の色)

# ---- 寸法 (mm) ----
WIDTH = 55.0          # 絵の線の横幅 (ひげの先から先まで)。板は縁の分だけこれより大きい
THICK = 2.6           # 板の厚み
FLOOR = 0.4           # 線の溝の底に残す薄皮 (2 層)
MOUTH_LEVEL = 1.3     # 口の中の高さ (板の表面より下、溝の底より上)
MIN_LINE = 1.0        # 溝の最小幅 (眼鏡の細い線はこれに届くよう太らせる)
MIN_WEB = 0.8         # 溝と溝の間に残す壁の最小幅。これより細い壁は溝にまとめる
MARGIN = 2.0          # 線の外側に付ける板の縁
RING_OUTER = 8.5      # 金具を通す輪の外径
RING_HOLE = 3.5       # 輪の穴の仕上がり径 (ボールチェーン・二重リング用)
RING_OVERLAP = 1.4    # 輪を板にどれだけ食い込ませるか

# ---- 絵の読み取り (座標は元の絵の画素) ----
UPSAMPLE = 4                  # 輪郭を滑らかにするため、拡大してから処理する
PAD = 65                      # 板が絵の端で切れないよう、周りに足す余白
EAR_GAP_X = (118, 190)        # 両耳の間 (輪が耳の線を隠さずに置ける範囲) の X 座標
# 絵では足の間が開いていて「顔の内側」が閉じていない。板で埋める範囲を決めるためだけに、
# 左足の付け根 → (下にふくらませて) → 右足の付け根を仮の線で結ぶ。この線自体は彫らない。
CHIN = [(117, 283), (160, 322), (216, 304)]

PRINT = {}


def _read():
    """絵を拡大して、黒い線と赤のマスクに分ける。"""
    img = cv2.imread(str(SOURCE))
    img = cv2.resize(img, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))

    def mask(condition):
        return np.pad(np.where(condition, 255, 0).astype(np.uint8), PAD * UPSAMPLE)

    black = mask(v < 90)
    red = mask(((h < 12) | (h > 168)) & (s > 120) & (v > 120))
    return black, red


def _at(points) -> np.ndarray:
    """元の絵の座標を、拡大・余白つきの座標に直す。"""
    return (np.asarray(points, dtype=float) + PAD) * UPSAMPLE


def _layout():
    """板・溝・口の領域 (マスク) と、寸法の換算を決める。"""
    black, red = _read()

    # 絵の端にある描き損じの点を捨てる
    lines = np.zeros_like(black)
    for blob in trace.components(black, min_area_px=20 * UPSAMPLE**2):
        lines |= blob

    xs = np.nonzero(lines)[1]
    mm = WIDTH / (xs.max() - xs.min())     # 拡大後の 1 px が何 mm か
    px = 1 / mm

    ink = trace.smooth(lines, 0.1 * px)
    ink = trace.smooth(trace.thicken_thin_lines(ink, MIN_LINE * px), 0.1 * px)
    # 溝。間の壁が細すぎる所は一つの溝にまとめる
    groove = trace.smooth(trace.close_gaps(ink, MIN_WEB / 2 * px), 0.08 * px)

    # 口の中 = 溝で囲まれた領域のうち、赤く塗られているもの
    mouth = np.zeros_like(groove)
    for region in trace.enclosed_regions(groove):
        if (red[region > 0] > 0).mean() > 0.5:
            mouth |= region
    assert mouth.any(), "口が見つからない"

    # 顔の内側 = 溝 (と仮のあごの線) で囲まれた領域。ここも板で埋める
    p0, p1, p2 = _at(CHIN)
    t = np.linspace(0, 1, 40)[:, None]
    chin = ((1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2).round().astype(np.int32)
    for end in (chin[0], chin[-1]):
        assert groove[end[1], end[0]], "あごの仮線の端が黒い線に乗っていない"
    closed = groove.copy()
    cv2.polylines(closed, [chin], False, 255, thickness=3)
    face = np.zeros_like(groove)
    for region in trace.enclosed_regions(closed):
        face |= region

    plate = trace.grow(groove | face, MARGIN * px)
    plate = trace.fill_holes(trace.close_gaps(plate, 2.5 * px))

    # 金具を通す輪: 両耳の間で、できるだけ重心の真上
    r = RING_OUTER / 2 * px
    lo, hi = _at(EAR_GAP_X)
    balance_x = float(np.nonzero(plate & ~groove)[1].mean())
    ring_x = int(round(np.clip(balance_x, lo + r, hi - r)))
    ring_y = int(round(np.nonzero(plate[:, ring_x])[0].min() - r + RING_OVERLAP * px))
    cv2.circle(plate, (ring_x, ring_y), int(round(r)), 255, cv2.FILLED)
    plate = trace.fill_holes(trace.components(trace.smooth(plate, 0.3 * px))[0])
    assert not (groove & ~trace.shrink(plate, 1.0 * px)).any(), "溝が板の縁に近すぎる"

    ys, xs = np.nonzero(plate)
    center = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    by, bx = np.nonzero(plate & ~groove)
    return {
        "plate": plate,
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
