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

# 瞳の描き方 (どれも輪郭の輪 + 三日月形の上まつ毛 + 白い光):
#   "manga":  上まつ毛の下の影と真ん中の瞳をくぼませ、下に明るい所を残す (元の絵の濃淡の配分)
#   "iris":   瞳全体をくぼませ、下に細い三日月だけ残す
#   "pupil":  真ん中の細い瞳だけくぼませる / "outline": 輪郭だけ
EYES = "manga"
# ほっぺたの描き方: "lines" (斜めの短い線 3 本) / "dimple" (浅い楕円のくぼみ) / "none" (描かない)
CHEEKS = "lines"

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

# 瞳。絵をなぞると輪郭がでこぼこするので、位置と大きさだけ絵から取って、楕円で描き直す
EYE_RING = 0.9        # 瞳の輪郭の溝の幅
EYE_LASH = 1.5        # 上まつ毛 (三日月形の溝) のいちばん太い所
EYE_WING = 1.0        # 上まつ毛の両端を、瞳の幅よりこれだけ外へ出す (LASH_ENDS = False のとき)
LASH_ENDS = True      # True: 上まつ毛の両端 (目尻で下へ回り込む跳ね) を絵からなぞる / False: 左右対称の三日月
LASH_MIN = 0.9        # なぞったまつ毛の最小幅 (目尻の細い跳ねを、ここまで太らせる)
EYE_PUPIL = (0.36, 0.50)      # 真ん中の瞳の半径 (瞳全体の半径に対する割合: 横, 縦)。絵では幅の 38%
EYE_SHADE = 0.8       # "manga": 上まつ毛の下の影の厚み (瞳の縦の半径に対する割合)
EYE_GLINT = 1.3       # "iris": 下に残す三日月のいちばん太い所
EYE_SPARK = (0.85, 0.55)      # 白い光 (楕円) の半径: 横, 縦。位置は絵のとおり
LASH_ZONE = 4.0       # 瞳のふちからこの距離 (元の絵の画素) までの細いまつ毛は描かない

# ほっぺた
CHEEK_LINE = (0.8, 2.8)       # 斜めの線の幅と長さ
CHEEK_PITCH = 1.7     # 線と線の間隔
CHEEK_SLANT = 65      # 線の傾き (度。90 で垂直)
CHEEK_DEPTH = 0.4     # 浅いくぼみの深さ ("dimple")

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

    # まつ毛と瞳の黒 = 暗くて色味の無い所 (髪や顔の線は茶色なので入らない)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(int)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    black = _mask((v < 120) & (chroma < 14)) & sil
    solid = _mask((v < 170) & (chroma < 12)) & sil       # 瞳の中の灰色まで含めた、目のひとかたまり

    # 瞳 = 灰色〜黒の塊。黒い芯のまわりの明るい三日月と、下の輪郭線までを 1 つに塗りつぶす
    gray = _mask(((s < 60) & (v < 205)) | (v < 80)) & sil
    cores = trace.components(trace.drop_thin(gray, 2.5 * up))[:2]
    assert len(cores) == 2, "瞳が 2 つ見つからない"
    eyes = []
    for core in cores:
        zone = trace.grow(core, 6 * up)
        blob = trace.fill_holes(trace.close_gaps((gray | lines) & zone, 1.5 * up))
        blob = trace.components(trace.drop_thin(blob & trace.shrink(zone, 1.5 * up), 2.5 * up))[0]
        # 白い光 = 瞳の内側の白い塊のうち、いちばん大きいもの (下の細い三日月やふちの肌色は除く)
        white = trace.components(trace.drop_thin(_mask(v > 225) & trace.shrink(blob, 2 * up), 1.2 * up))[0]
        wy, wx = np.nonzero(white)

        # 瞳の大きさ: 上はまつ毛の上端、下は瞳の下端。幅は、まつ毛の両端や目尻の跳ねを含めないよう、
        # 真ん中の列を含むひと続きの幅を、まつ毛より下の行で測る
        whole = np.zeros_like(solid)
        for piece in trace.components(solid & trace.grow(core, 12 * up)):
            if (piece & core).any():
                whole |= piece
        whole = trace.fill_holes(whole)
        column = int(np.nonzero(core)[1].mean())
        rows = np.nonzero(whole[:, column])[0]
        top, bottom = rows.min(), rows.max()
        runs = []
        for row in range(int(top + 0.35 * (bottom - top)), int(top + 0.65 * (bottom - top)), max(up // 2, 1)):
            filled = whole[row] > 0
            left = right = column
            while left > 0 and filled[left - 1]:
                left -= 1
            while right < len(filled) - 1 and filled[right + 1]:
                right += 1
            runs.append((right - left, (left + right) / 2))
        width, middle = sorted(runs)[int(0.8 * len(runs))]
        eyes.append({
            "blob": blob | whole,
            "spark": (wx.mean(), wy.mean()),
            "center": (middle, (top + bottom) / 2),
            "radius": (width / 2, (bottom - top) / 2),
        })

    # ほっぺた = 肌より赤みの強い所 (線は無く、ぼかした桃色の楕円)
    skin = (s < 45) & (v > 235)
    pink = _mask((lab[..., 1] - np.median(lab[..., 1][skin]) > 9) & (lab[..., 0] > 200) & (lab[..., 2] < 150))
    cheeks = []
    for blob in trace.components(trace.drop_thin(pink, 2 * up))[:2]:
        ys, xs = np.nonzero(blob)
        cheeks.append({
            "center": ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2),
            "radius": ((xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2),
        })
    assert len(cheeks) == 2, "ほっぺたが 2 つ見つからない"
    return {"lines": lines, "eyes": eyes, "cheeks": cheeks, "black": black, "up": up}


def _ellipse(shape, center, radius):
    out = np.zeros(shape, np.uint8)
    cv2.ellipse(out, (int(round(center[0])), int(round(center[1]))),
                (max(int(round(radius[0])), 1), max(int(round(radius[1])), 1)), 0, 0, 360, 255, cv2.FILLED)
    return out


def _eyes(traced, mm):
    """瞳の溝と、瞳の領域 (なぞった線を消す範囲) を返す。描き方は EYES で選ぶ。

    位置と大きさは絵のとおり。形は楕円で描き直す (絵をなぞると輪郭がでこぼこする)。
    元の絵の瞳は、上がまつ毛の影で黒く、真ん中に黒い瞳 (幅の 38%) があり、下に明るい三日月がある。
    """
    px = 1 / mm
    shape = traced["eyes"][0]["blob"].shape
    groove = np.zeros(shape, np.uint8)
    area = np.zeros(shape, np.uint8)
    for e in traced["eyes"]:
        cx, cy = e["center"]
        a, b = e["radius"]
        outer = _ellipse(shape, (cx, cy), (a, b))
        inner_radius = (a - EYE_RING * px, b - EYE_RING * px)
        inner = _ellipse(shape, (cx, cy), inner_radius)
        if LASH_ENDS:
            # 絵からなぞったまつ毛の両端を、瞳の楕円に溶接する (0.6 mm より狭いすき間は埋まる)
            joined = outer | _lash_ends(traced, outer, (cx, cy), b, px)
            lash = trace.close_gaps(joined, 0.3 * px)
            # てっぺんは、まつ毛と楕円の間にできる浅い切れ込みも埋めて、なめらかな 1 本の弧にする
            top = trace.close_gaps(joined, 0.8 * px)
            top[int(cy - 0.6 * b):] = 0
            lash = (lash | top) & ~outer
        else:
            lash_radius = (a + EYE_WING * px, 0.55 * b)
            lash_y = cy - b + lash_radius[1]
            lash = _ellipse(shape, (cx, lash_y), lash_radius) & ~_ellipse(shape, (cx, lash_y + EYE_LASH * px), lash_radius)

        # 白い光は絵の位置に置く。輪郭の溝にかかるなら、かからなくなるまで瞳の中心へ寄せる
        room = trace.shrink(inner, 0.2 * px)
        sx, sy = e["spark"]
        for step in np.linspace(0, 1, 41):
            spark = _ellipse(shape, (sx + (cx - sx) * step, sy + (cy - sy) * step), (EYE_SPARK[0] * px, EYE_SPARK[1] * px))
            if not (spark & ~room).any():
                break

        g = (outer & ~inner) | lash
        if EYES == "manga":
            shade = inner & ~_ellipse(shape, (cx, cy + EYE_SHADE * b), inner_radius)
            pupil = _ellipse(shape, (cx, cy + 0.1 * b), (EYE_PUPIL[0] * a, EYE_PUPIL[1] * b))
            g |= (shade | pupil) & ~spark
        elif EYES == "iris":
            glint = inner & ~_ellipse(shape, (cx, cy - EYE_GLINT * px), inner_radius)
            g |= inner & ~glint & ~spark
        elif EYES == "pupil":
            g |= _ellipse(shape, (cx, cy - 0.2 * b), (EYE_PUPIL[0] * a, EYE_PUPIL[1] * b)) & ~spark
        elif EYES != "outline":
            raise ValueError(f"EYES が不正: {EYES}")
        groove |= g
        area |= outer | lash | e["blob"]
    return trace.smooth(groove, 0.1 * px), area


def _lash_ends(traced, outer, center, b, px):
    """上まつ毛のうち、瞳の楕円の外へはっきりはみ出す部分 (両端と、目尻で下へ回り込む跳ね) を絵からなぞる。

    楕円の内側の上の方は、輪郭の輪と影でどのみち溝になるので、外側だけ拾えば足りる。
    楕円のすぐ外 (0.4 mm 以内) は拾わない: そこは瞳の輪郭のずれで、拾うと輪郭がでこぼこする。
    """
    up = traced["up"]
    cx, cy = center
    near = trace.grow(outer, 14 * up)
    # まつ毛とつながっている黒だけ (瞳の上の方の 1 点を含む塊)
    seed = np.zeros_like(outer)
    cv2.circle(seed, (int(cx), int(cy - 0.5 * b)), int(2 * up), 255, cv2.FILLED)
    dark = np.zeros_like(outer)
    for piece in trace.components(traced["black"] & near):
        if (piece & seed).any():
            dark |= piece
    ends = _keep_big(dark & ~trace.grow(outer, 0.4 * px), (0.6 * px) ** 2)
    return trace.smooth(trace.thicken_thin_lines(trace.smooth(ends, 0.15 * px), LASH_MIN * px), 0.15 * px)


def _cheeks(traced, mm):
    """ほっぺたの (溝, 浅いくぼみ) を返す。描き方は CHEEKS で選ぶ。"""
    px = 1 / mm
    shape = traced["eyes"][0]["blob"].shape
    groove = np.zeros(shape, np.uint8)
    dimple = np.zeros(shape, np.uint8)
    for cheek in traced["cheeks"]:
        cx, cy = cheek["center"]
        if CHEEKS == "lines":
            width, length = CHEEK_LINE
            dx = np.cos(np.radians(CHEEK_SLANT)) * (length - width) / 2 * px
            dy = np.sin(np.radians(CHEEK_SLANT)) * (length - width) / 2 * px
            for k in (-1, 0, 1):
                x = cx + k * CHEEK_PITCH * px
                cv2.line(groove, (int(round(x - dx)), int(round(cy + dy))), (int(round(x + dx)), int(round(cy - dy))),
                         255, int(round(width * px)), cv2.LINE_8)
        elif CHEEKS == "dimple":
            dimple |= _ellipse(shape, (cx, cy), cheek["radius"])
        elif CHEEKS != "none":
            raise ValueError(f"CHEEKS が不正: {CHEEKS}")
    return groove, dimple


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
    eye_groove, eye_area = _eyes(traced, mm)
    lines &= ~trace.grow(eye_area, LASH_ZONE * up)     # 細いまつ毛は、太らせるとつぶれて汚れになる
    lines = _keep_big(lines, 30 * up * up)             # 髪の白い光のふちなど、線でない粒を捨てる
    groove = trace.smooth(trace.thicken_evenly(trace.smooth(lines, 0.1 * px), MIN_LINE * px), 0.15 * px)
    groove = trace.smooth(trace.close_gaps(groove, MIN_WEB / 2 * px), 0.08 * px)
    groove = _merge_small_islands(groove, mm, MIN_ISLAND)
    cheek_groove, dimple = _cheeks(traced, mm)
    groove = (groove & ~trace.grow(eye_area, 0.1 * px)) | eye_groove | cheek_groove

    art = np.zeros_like(groove)
    cv2.circle(art, (n // 2, n // 2), side * up // 2, 255, cv2.FILLED)      # 丸で切り出す (四隅は切れる)
    pad = int(round(PAD_MM * px))
    art, groove, dimple = (np.pad(m, pad) for m in (art, groove & art, dimple & art))
    plate, ring = _add_ring(trace.grow(art, RIM * px), mm, art.shape[1] // 2)

    # 角だけで触れ合う所があると立体が壊れるので、最後に輪郭をならす
    groove = trace.smooth(groove & plate, 0.04 * px)
    assert not (groove & ~trace.shrink(plate, 1.0 * px)).any(), "溝が板の縁に近すぎる"
    ys, xs = np.nonzero(plate)
    center = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    return {
        "plate": plate,
        "groove": groove,
        "dimple": dimple & ~groove,        # 浅くくぼませる所 (ほっぺた)
        "top": plate & ~groove & ~dimple,  # 板の表面まである所
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
    if L["dimple"].any():
        part += _prism(L["dimple"], L, FLOOR, THICK - CHEEK_DEPTH)
    hole = Cylinder(fit.hole(RING_HOLE) / 2, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return part - hole.move(Location((*L["ring"], 0)))
