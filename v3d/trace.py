"""絵 (画像) の塗り分けを、立体にできる輪郭に変える道具。

流れ: 画像を色で塗り分けたマスク (0/255 の配列) にする → ここの関数で太らせたり滑らかにしたりする
→ `faces()` で build123d の面にして extrude する。長さの引数はすべて画素数 (px)。
"""

from __future__ import annotations

import cv2
import numpy as np


def disk(radius_px: float) -> np.ndarray:
    r = max(int(round(radius_px)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def grow(mask: np.ndarray, radius_px: float) -> np.ndarray:
    return cv2.dilate(mask, disk(radius_px))


def shrink(mask: np.ndarray, radius_px: float) -> np.ndarray:
    return cv2.erode(mask, disk(radius_px))


def close_gaps(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """radius より狭いすき間・入り江を埋める。"""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(radius_px))


def drop_thin(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """幅が 2×radius より細い部分を取り除く。"""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(radius_px))


def smooth(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """輪郭のギザギザをならす (ぼかしてから二値に戻す)。"""
    blurred = cv2.GaussianBlur(mask, (0, 0), sigma_px)
    return np.where(blurred >= 128, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """囲まれた穴を塗りつぶす。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def components(mask: np.ndarray, min_area_px: float = 0) -> list[np.ndarray]:
    """つながった塊ごとのマスク (大きい順)。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [
        np.where(labels == i, 255, 0).astype(np.uint8)
        for i in order
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px
    ]


def enclosed_regions(lines: np.ndarray) -> list[np.ndarray]:
    """線で完全に囲まれた内側の領域 (口の中、レンズの中など) のマスク。"""
    outside_and_inside = cv2.bitwise_not(lines)
    regions = []
    for region in components(outside_and_inside):
        touches_border = (
            region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
        )
        if not touches_border:
            regions.append(region)
    return regions


def thicken_thin_lines(lines: np.ndarray, min_width_px: float) -> np.ndarray:
    """min_width より細い線だけを太らせる (もともと太い線はそのまま)。"""
    thick = drop_thin(lines, min_width_px / 2)
    # 太い線の縁のでこぼこは「細い線」ではないので、太い部分のすぐそばは除外する
    thin = cv2.bitwise_and(lines, cv2.bitwise_not(grow(thick, min_width_px / 3)))
    strokes = np.zeros_like(lines)
    for piece in components(thin, min_area_px=min_width_px**2 / 2):
        strokes |= piece
    return cv2.bitwise_or(lines, grow(strokes, min_width_px / 4))


def line_width(lines: np.ndarray) -> float:
    """線の太さの中央値 (px)。線の芯 (縁からいちばん遠い所) での太さを集めて測る。"""
    depth = cv2.distanceTransform(lines, cv2.DIST_L2, 5)
    core = (depth > 0) & (depth >= cv2.dilate(depth, np.ones((3, 3), np.uint8)) - 1e-6)
    return 2 * float(np.median(depth[core]))


def thicken_evenly(lines: np.ndarray, min_width_px: float) -> np.ndarray:
    """線全体を同じだけ太らせて、太さの中央値を min_width に届かせる。

    太さのそろった線画向き。thicken_thin_lines は細い線だけを選んで太らせるので、
    全部の線が細い絵に使うと、線の交わる所や先端だけ太ってこぶになる。
    """
    extra = (min_width_px - line_width(lines)) / 2
    return grow(lines, extra) if extra >= 0.5 else lines


def faces(mask: np.ndarray, mm_per_px: float, center_px: tuple[float, float], detail_mm: float = 0.04):
    """マスクの輪郭を build123d の面にする。画像は Y が下向きなので上下を反転する。"""
    from build123d import Face, Wire

    def wire(contour: np.ndarray, counterclockwise: bool):
        simplified = cv2.approxPolyDP(contour, detail_mm / mm_per_px, closed=True).reshape(-1, 2)
        if len(simplified) < 3:
            return None
        pts = [((x - center_px[0]) * mm_per_px, (center_px[1] - y) * mm_per_px) for x, y in simplified]
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1])) / 2
        if abs(area) < 0.05:
            return None
        if (area > 0) != counterclockwise:
            pts.reverse()
        return Wire.make_polygon(pts, close=True)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:      # 穴は親の面と一緒に処理する
            continue
        outer = wire(contour, counterclockwise=True)
        if outer is None:
            continue
        holes = []
        child = hierarchy[0][i][2]
        while child != -1:
            hole = wire(contours[child], counterclockwise=False)
            if hole is not None:
                holes.append(hole)
            child = hierarchy[0][child][0]
        out.append(Face(outer, holes))
    return out


def shaded(height: np.ndarray, mm_per_px: float, color_bgr=(90, 200, 120), background=(255, 255, 255)) -> np.ndarray:
    """高さの地図 (mm、0 は「何も無い」) から、1色で刷ったときの見え方に近い絵を作る。

    左上から光を当てたときの明暗・落ちる影・溝の奥の暗さを付ける。立体を作る前に、
    線を盛るか彫るかなどの案を素早く見比べるためのもの (実物の質感までは再現しない)。
    """
    h = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 0.05 / mm_per_px)
    material = height > 0

    gy, gx = np.gradient(h, mm_per_px)
    light = np.array([-0.5, -0.5, 0.707])                      # 画像の左上・斜め 45°
    normal_len = np.sqrt(gx**2 + gy**2 + 1)
    lambert = np.clip((-gx * light[0] - gy * light[1] + light[2]) / normal_len, 0, 1)

    # 落ちる影: 光の来る方向に、自分より「距離ぶん以上」高い所があれば影
    shadow = np.zeros(h.shape, bool)
    step = 2
    for k in range(1, int(4.0 / (step * 1.414 * mm_per_px)) + 1):
        toward_light = np.roll(h, (k * step, k * step), axis=(0, 1))
        shadow |= toward_light - k * step * 1.414 * mm_per_px > h + 0.05
    # 影と下の「溝の暗さ」は、境目を少しぼかす (ぼかさないと、広いくぼみの中が縞模様に見える)
    shadow = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), 0.12 / mm_per_px)

    # 溝の奥ほど暗い。壁から 1.2 mm より離れた所 (広いくぼみの真ん中) は暗くならない
    surroundings = cv2.dilate(h, disk(1.2 / mm_per_px))
    occlusion = 0.45 * np.clip((surroundings - h) / 2.0, 0, 1)
    occlusion = cv2.GaussianBlur(occlusion, (0, 0), 0.2 / mm_per_px)

    brightness = (0.42 + 0.58 * lambert / light[2]) * (1 - occlusion) * (1 - 0.28 * shadow)
    out = np.empty((*h.shape, 3), np.float32)
    out[:] = background
    # 地面に落ちる影 (穴の中や外周で、板が浮いて見えるように)
    out[~material] *= 1 - 0.14 * shadow[~material, None]
    out[material] = np.clip(brightness[material, None] * np.array(color_bgr, np.float32), 0, 255)
    return out.astype(np.uint8)
