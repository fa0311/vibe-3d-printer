"""dog-keychain: 犬耳の子の絵 (source.png、持ち主の SNS アイコンの元絵) をキーホルダーにする。

1色でしか刷れないので、線を深い溝にして描く (cat-keychain、hood-keychain と同じ)。
この絵は線が茶色で細く、塗りに濃淡があるので、「黒い画素」ではなく「周りより暗い細い線」を拾う。
板に載せるのは SNS のアイコンと同じ範囲 (顔のアップ)。全身の案は持ち主に「不要」と言われた。
"""

from pathlib import Path

import cv2
import numpy as np
from build123d import *

from v3d import fit, trace

SOURCE = Path(__file__).with_name("source.png")     # 500×500 px、背景は透明
PREVIEW_COLOR = "#d9b77e"     # 確認用の絵の色 (髪の色に寄せた仮の色)

# 瞳の描き方: "solid" (まるごとくぼませて、白い光だけ残す) / "outline" (輪郭の線だけ) /
#             "pupil" (輪郭の線 + 真ん中の瞳)
EYES = "pupil"

# ---- 寸法 (mm) ----
ICON_BOX = (149, 119, 183)    # アイコンの切り出し位置: 左上の x, y と一辺 (元の絵の画素)
ART = 50.0            # 絵の直径。板は縁の分だけこれより大きい (hood / bear とそろえた)
THICK = 2.6           # 板の厚み
FLOOR = 0.4           # 溝の底に残す薄皮 (2 層)
MIN_LINE = 1.0        # 溝の幅 (絵の線は 0.7 mm 相当なので、ここまで太らせる)
MIN_WEB = 0.8         # 溝と溝の間に残す壁の最小幅。これより細い壁は溝にまとめる
MIN_ISLAND = 1.5      # 溝に囲まれた島の最小面積 (mm²)。これより小さい島は溝にまとめる
RIM = 1.6             # 絵の外側に付ける縁
RING_OUTER = 8.5      # 金具を通す輪の外径
RING_HOLE = 3.5       # 輪の穴の仕上がり径 (ボールチェーン・二重リング用)
RING_OVERLAP = 2.0    # 輪を板にどれだけ食い込ませるか

# 瞳
EYE_RING = 0.9        # 瞳の輪郭の溝の幅 ("outline" / "pupil")
EYE_LASH = 0.7        # 上まつ毛の分、輪郭の上側をこれだけ太くする
EYE_SPARK = 0.3       # 白い光を、絵よりこれだけ大きくする (小さいと刷れない)
LASH_ZONE = 4.0       # 瞳のふちからこの距離 (元の絵の画素) までの細いまつ毛は描かない

# ---- 絵の読み取り ----
UPSAMPLE = 8          # 輪郭を滑らかにするため、拡大してから処理する
MARGIN = 6            # 切り出しの外側に付けて読む余白 (元の絵の画素)
PAD_MM = 10.0         # 縁と輪がはみ出せるよう、絵の周りに足す余白

PRINT = {}


def _mask(condition) -> np.ndarray:
    return np.where(condition, 255, 0).astype(np.uint8)


def _keep_big(mask, min_area_px):
    out = np.zeros_like(mask)
    for piece in trace.components(mask, min_area_px=min_area_px):
        out |= piece
    return out


def _read(box, up):
    """絵の box=(x0, y0, x1, y1) を up 倍に拡大して、色 (背景は白) とシルエットを返す。"""
    rgba = cv2.imread(str(SOURCE), cv2.IMREAD_UNCHANGED)
    x0, y0, x1, y1 = box
    rgba = rgba[y0:y1, x0:x1]
    alpha = rgba[..., 3]
    bgr = rgba[..., :3].copy()
    bgr[alpha < 128] = 255

    def big(image):
        return cv2.resize(image, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)

    return big(bgr), _mask(big(alpha) > 128)


def _trace(bgr, sil, up):
    """線と瞳を拾う。up は元の絵の 1 px が何 px か。"""
    # 線 = 周りより暗い細い所 (ブラックハット)。線が茶色でも、塗りが何色でも拾える
    light = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0], (0, 0), 0.35 * up)
    lines = _mask(cv2.morphologyEx(light, cv2.MORPH_BLACKHAT, trace.disk(5 * up)) > 18) & sil

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))

    # 骨 (黄色) の上の文字 "DOGTOY" は、アイコンの範囲だと上半分しか入らないので描かない
    yellow = _mask((h > 20) & (h < 36) & (s > 60) & (v > 200))
    bone = trace.fill_holes(trace.close_gaps(trace.components(yellow)[0], 4 * up))
    lines &= ~trace.shrink(bone, 4 * up)

    # 瞳 = 灰色〜黒の塊。黒い芯のまわりの明るい三日月と、下の輪郭線までを 1 つに塗りつぶす
    gray = _mask(((s < 60) & (v < 205)) | (v < 80)) & sil
    cores = trace.components(trace.drop_thin(gray, 2.5 * up))[:2]
    assert len(cores) == 2, "瞳が 2 つ見つからない"
    eyes, sparks = [], []
    for core in cores:
        zone = trace.grow(core, 6 * up)
        blob = trace.fill_holes(trace.close_gaps((gray | lines) & zone, 1.5 * up))
        blob = trace.drop_thin(blob & trace.shrink(zone, 1.5 * up), 2.5 * up)
        eye = trace.components(blob)[0]
        eyes.append(eye)
        # 白い光 = 瞳の内側の白い塊のうち、いちばん大きいもの (下の細い三日月やふちの肌色は除く)
        white = trace.drop_thin(_mask(v > 225) & trace.shrink(eye, 2 * up), 1.2 * up)
        sparks.append(trace.components(white)[:1])
    return {"lines": lines, "eyes": eyes, "sparks": sparks}


def _eyes(traced, mm):
    """瞳の溝と、瞳の領域を返す。描き方は EYES で選ぶ。"""
    px = 1 / mm
    groove = np.zeros_like(traced["eyes"][0])
    area = np.zeros_like(groove)
    for eye, spark in zip(traced["eyes"], traced["sparks"]):
        eye = trace.smooth(trace.grow(eye, 0.25 * px), 0.2 * px)
        light = np.zeros_like(eye)
        for piece in spark:
            light |= trace.smooth(trace.grow(piece, EYE_SPARK * px), 0.1 * px)
        if EYES == "solid":
            g = eye & ~light
        elif EYES in ("outline", "pupil"):
            inner = trace.shrink(eye, EYE_RING * px)
            inner &= np.roll(inner, int(EYE_LASH * px), axis=0)      # 上はまつ毛の分だけ太い溝にする
            g = eye & ~inner
            if EYES == "pupil":
                # 瞳は、上まつ毛からぶら下がる縦長の丸。幅を狭くして、溝として暗く見えるようにする
                ys, xs = np.nonzero(inner)
                a, b = 0.24 * (xs.max() - xs.min()), 0.40 * (ys.max() - ys.min())
                center = (int((xs.min() + xs.max()) / 2), int(ys.min() + b - 0.3 * px))
                pupil = np.zeros_like(eye)
                cv2.ellipse(pupil, center, (int(a), int(b)), 0, 0, 360, 255, cv2.FILLED)
                g |= pupil & eye & ~light
        else:
            raise ValueError(f"EYES が不正: {EYES}")
        groove |= g
        area |= eye
    return groove, area


def _merge_small_islands(groove, mm, min_island):
    for island in trace.components(cv2.bitwise_not(groove)):
        if (island > 0).sum() * mm * mm < min_island:
            groove |= island
    return groove


def _add_ring(plate, mm, ring_x):
    """板の上辺に金具を通す輪を足す。輪の中心 (画素) も返す。"""
    px = 1 / mm
    r = RING_OUTER / 2 * px
    ring_y = int(round(np.nonzero(plate[:, ring_x])[0].min() - r + RING_OVERLAP * px))
    cv2.circle(plate, (ring_x, ring_y), int(round(r)), 255, cv2.FILLED)
    return trace.fill_holes(trace.components(trace.smooth(plate, 0.3 * px))[0]), (ring_x, ring_y)


def _layout():
    """板・溝の領域 (マスク) と、寸法の換算を決める。"""
    x, y, side = ICON_BOX
    up = UPSAMPLE
    bgr, sil = _read((x - MARGIN, y - MARGIN, x + side + MARGIN, y + side + MARGIN), up)
    n = bgr.shape[0]
    mm = ART / (side * up)                 # 拡大後の 1 px が何 mm か
    px = 1 / mm
    traced = _trace(bgr, sil, up)

    lines = traced["lines"]
    for eye in traced["eyes"]:
        lines &= ~trace.grow(eye, LASH_ZONE * up)      # 細いまつ毛は、太らせるとつぶれて汚れになる
    lines = _keep_big(lines, 30 * up * up)             # 髪の白い光のふちなど、線でない粒を捨てる
    groove = trace.smooth(trace.thicken_evenly(trace.smooth(lines, 0.1 * px), MIN_LINE * px), 0.15 * px)
    groove = trace.smooth(trace.close_gaps(groove, MIN_WEB / 2 * px), 0.08 * px)
    groove = _merge_small_islands(groove, mm, MIN_ISLAND)
    eye_groove, eye_area = _eyes(traced, mm)
    groove = (groove & ~trace.grow(eye_area, 0.1 * px)) | eye_groove

    art = np.zeros_like(groove)
    cv2.circle(art, (n // 2, n // 2), side * up // 2, 255, cv2.FILLED)      # 丸で切り出す (四隅は切れる)
    pad = int(round(PAD_MM * px))
    art, groove = (np.pad(m, pad) for m in (art, groove & art))
    plate, ring = _add_ring(trace.grow(art, RIM * px), mm, art.shape[1] // 2)

    # 角だけで触れ合う所があると立体が壊れるので、最後に輪郭をならす
    groove = trace.smooth(groove & plate, 0.04 * px)
    assert not (groove & ~trace.shrink(plate, 1.0 * px)).any(), "溝が板の縁に近すぎる"
    ys, xs = np.nonzero(plate)
    center = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    return {
        "plate": plate,
        "groove": groove,
        "top": plate & ~groove,            # 板の表面まである所
        "mm": mm,
        "center": center,
        "ring": ((ring[0] - center[0]) * mm, (center[1] - ring[1]) * mm),
    }


def _prism(mask, layout, z0: float, z1: float):
    shapes = trace.faces(mask, layout["mm"], layout["center"], detail_mm=0.015)
    return extrude(Sketch() + shapes, amount=z1 - z0).move(Location((0, 0, z0)))


def build():
    """絵の面を上にして寝かせて刷る (裏は平ら)。"""
    L = _layout()
    part = _prism(L["plate"], L, 0, FLOOR) + _prism(L["top"], L, FLOOR, THICK)
    hole = Cylinder(fit.hole(RING_HOLE) / 2, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return part - hole.move(Location((*L["ring"], 0)))
