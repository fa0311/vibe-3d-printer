"""suica-bunny: Suica の新しいキャラクター (B案の紫のうさぎ) の置物。個人で飾るためのファンメイド。

どこから見ても丸い立体なので、build123d ではなく v3d/blob.py (管や柱をなめらかにつなぐ道具) で作る。
形は、持ち主がくれた設定画の三面図 (正面・横) に輪郭を重ねて合わせた。顔は公式の正面の絵から測った。
寸法はすべて「耳の上から足の裏までの高さ H」に対する割合で持ち、H を変えれば全体が同じ形で伸び縮みする。

向き: 正面は -y、右手側 (向かって左) は -x、上は +z。立てたまま刷る (足の裏がベッド)。
"""

import cv2
import numpy as np

from v3d.blob import Clay, Column, Stencil, Tube

PREVIEW_COLOR = "#8b3dff"     # 確認用の絵の色 (設定画のメインカラー)

H = 80.0              # 耳の上から足の裏までの高さ (mm)
VOXEL = 0.25          # 形を計算する格子の細かさ (mm)。小さいほどなめらかだが、時間がかかる

# ---- 体 (H に対する割合) ----
# 胴体と頭はひとつながり。おなか (z = 0.34) がいちばん太く、胸から上はほぼ同じ太さの筒で、てっぺんは丸い。
# 横から見ると正面より薄い (depth < width)。いちばん下の点は形には効かず、底の方の幅と奥行きの比を決めるだけ
BODY = dict(heights=[0.24, 0.34, 0.42, 0.50, 0.60, 0.70], radii=[0.080, 0.185, 0.172, 0.151, 0.142, 0.140],
            width=[0.90, 1.03, 1.06, 1.10, 1.13, 1.12], depth=[0.93, 1.00, 0.97, 0.95, 0.94, 0.95],
            lean=[0.0, -0.006, -0.003, 0.0, 0.004, 0.012])     # おなかは少し前へ、頭は少し後ろへ

# 耳: 頭のてっぺんの細めの付け根から立ち上がり、外へ広がりながら前へ倒れる太い房 (設定画の横向きの絵のとおり)。
# 前へ倒してあるおかげで、耳の下の支えを腕や手をよけてベッドからまっすぐ立てられる (下の PRINT を参照)
EAR = dict(points=[(0.045, 0.000, 0.770), (0.072, -0.012, 0.840), (0.105, -0.045, 0.892),
                   (0.170, -0.075, 0.916), (0.240, -0.135, 0.900), (0.262, -0.165, 0.885)],
           radii=[0.058, 0.062, 0.076, 0.084, 0.082, 0.078],
           squash=[(1, 0.80, 1), (1, 0.80, 1), (1, 0.85, 1), (1, 0.95, 1), (1, 1, 1), (1, 1, 1)])

# 腕: 肩 (胴体の中) から外へ開きながら下がり、先 (手) ほど太い。手は胴体から離れてぶら下がる
ARM = dict(points=[(0.115, 0.0, 0.555), (0.155, 0.0, 0.500), (0.213, 0.0, 0.400), (0.252, 0.0, 0.300),
                   (0.262, 0.0, 0.230)],
           radii=[0.045, 0.048, 0.068, 0.083, 0.084])
THUMB = dict(points=[(0.192, -0.040, 0.258)] * 2, radii=[0.026, 0.026])      # 手の内側の小さなこぶ (親指)

# 足: 胴体の下の短い脚と、前後に長い足先。左右の足先は真ん中で触れ合う。
# 足の裏は平らに切って、ベッドと机に接する面にする
LEG = dict(points=[(0.058, -0.010, 0.270), (0.060, -0.015, 0.100)], radii=[0.060, 0.060], squash=(1, 1.35, 1))
FOOT = dict(points=[(0.100, -0.083, 0.055), (0.100, -0.003, 0.055)], radii=[0.100, 0.100], squash=(1, 1, 0.78))

# つなぎ目を丸める幅。小さいほど、境目の筋がはっきり残る
BLEND = dict(ear=0.030, arm=0.028, thumb=0.012, leg=0.030, foot=0.025)

# ---- 顔 (公式の正面の絵から測った。x は中心から、z は足の裏から。H に対する割合) ----
FACE_PATCH_LOBE = ((0.058, 0.705), 0.040)             # 白い所の、目のまわりの丸いふくらみ (左右): 中心と半径
FACE_PATCH_CHIN = ((0.0, 0.668), (0.082, 0.056))      # 白い所の下半分: 楕円の中心と半径 (横, 縦)
EYE = ((0.060, 0.706), 0.0145)                        # 目 (左右): 中心と半径
NOSE = ((0.0, 0.699), (0.0195, 0.0106))               # 鼻: 楕円
MOUTH = dict(corner=0.066, corner_z=0.6775, bottom=0.0402, top_z=0.6641)   # 口: 三日月 (下は楕円の弧、上は放物線)
# 1 色なので、白い所は「縁取りの溝」で、目・鼻・口は「深い穴」で描く (狭くて深い所は影で黒く見える)
FACE_LINE = (0.9, 1.0)        # 白い所の縁取りの溝: 幅, 深さ (mm)
FEATURE_DEPTH = 2.2           # 目・鼻・口の深さ (mm)
FACE_PX = 0.02        # 顔の模様を描く細かさ (mm / 画素)

# 立てたまま刷る。耳の下・手の下・股の下は宙に浮くので、支え (サポート) が要る。
PRINT = {
    "process": "0.16mm Optimal",          # 丸い面の段々を目立たなくするため、標準より薄い層にする
    "settings": {
        "enable_support": 1,
        # 支えは「ベッドからまっすぐ立つ柱」にする。枝分かれする支え (tree) も試したが、耳の下まで 60 mm を
        # 登る枝が太さ 1〜2 mm の針のようになったり、途中の層が抜けたりして、頼りなかった (README を参照)
        "support_type": "normal(auto)",
        "support_on_build_plate_only": 1,     # 腕や肩の上には支えを乗せない (乗せると、見える所に跡が残る)
        "brim_type": "outer_only",            # 足の裏だけで立つので、はがれ止めのつば (幅 5 mm) を付ける
        "seam_position": "back",              # 層の継ぎ目の筋を背中側に寄せる
    },
}


def _mirror(spec: dict, **extra) -> list[Tube]:
    """左右の対になった管 (x の符号だけ逆)。"""
    out = []
    for sign in (1, -1):
        points = [(sign * x, y, z) for x, y, z in spec["points"]]
        out.append(Tube(np.array(points) * H, np.array(spec["radii"]) * H, squash=spec.get("squash", (1, 1, 1)), **extra))
    return out


def _face():
    """顔の模様 (白い所, 目鼻口) を、正面から見た絵として描く。"""
    x_half, z_lo, z_hi = 0.115, 0.600, 0.755
    w, h = int(2 * x_half * H / FACE_PX), int((z_hi - z_lo) * H / FACE_PX)
    center = (w / 2, h + z_lo * H / FACE_PX)          # 画像のこの画素が (x=0, z=0)

    def px(x, z):
        return int(round(center[0] + x * H / FACE_PX)), int(round(center[1] - z * H / FACE_PX))

    def size(a, b=None):
        return (int(round(a * H / FACE_PX)), int(round((b if b is not None else a) * H / FACE_PX)))

    patch = np.zeros((h, w), np.uint8)
    (lx, lz), lr = FACE_PATCH_LOBE
    for sign in (1, -1):
        cv2.circle(patch, px(sign * lx, lz), size(lr)[0], 255, cv2.FILLED)
    cv2.ellipse(patch, px(*FACE_PATCH_CHIN[0]), size(*FACE_PATCH_CHIN[1]), 0, 0, 360, 255, cv2.FILLED)
    # ふくらみと下半分のつなぎ目 (へこんだ角) を丸める
    r = size(0.012)[0]
    patch = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))

    marks = np.zeros((h, w), np.uint8)
    (ex, ez), er = EYE
    for sign in (1, -1):
        cv2.circle(marks, px(sign * ex, ez), size(er)[0], 255, cv2.FILLED)
    cv2.ellipse(marks, px(*NOSE[0]), size(*NOSE[1]), 0, 0, 360, 255, cv2.FILLED)
    m = MOUTH
    xs = np.linspace(-m["corner"], m["corner"], 81)
    lower = m["corner_z"] - m["bottom"] * np.sqrt(np.clip(1 - (xs / m["corner"]) ** 2, 0, 1))
    upper = m["top_z"] + (m["corner_z"] - m["top_z"]) * (xs / m["corner"]) ** 2
    outline = [px(x, z) for x, z in zip(xs, upper)] + [px(x, z) for x, z in zip(xs[::-1], lower[::-1])]
    cv2.fillPoly(marks, [np.array(outline, np.int32)], 255)

    # 白い所は、縁に沿った幅 FACE_LINE[0] の溝として描く
    r = max(int(round(FACE_LINE[0] / 2 / FACE_PX)), 1)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    ring = cv2.dilate(patch, disk) & ~cv2.erode(patch, disk)

    at = (0.0, 0.0)
    return Stencil(ring, FACE_PX, center, at), Stencil(marks, FACE_PX, center, at), (x_half * H, z_lo * H, z_hi * H)


def sculpt(voxel: float = VOXEL) -> Clay:
    """形を作る (メッシュにする手前まで)。輪郭を設定画と見比べるときにも使う。"""
    clay = Clay(bounds=((-0.38 * H, 0.38 * H), (-0.25 * H, 0.24 * H), (-0.03 * H, 1.02 * H)), voxel=voxel)
    scale = lambda values: np.array(values) * H  # noqa: E731
    clay.add(Column(scale(BODY["heights"]), scale(BODY["radii"]), BODY["width"], BODY["depth"], scale(BODY["lean"])))
    for name, spec in (("leg", LEG), ("foot", FOOT), ("arm", ARM), ("ear", EAR), ("thumb", THUMB)):
        for tube in _mirror(spec):
            clay.add(tube, blend=BLEND[name] * H)

    # 顔: 正面 (y < 0) の表面にだけ押し込む
    ring, marks, (x_half, z_lo, z_hi) = _face()
    front = -0.02 * H

    def on_front(stencil):
        return lambda x, y, z: np.minimum(stencil.inside(x, z), front - y)

    # 穴の天井は 45° の斜面にする (水平だと、スライサーが顔の穴の中に支えを立ててしまう)
    clay.press([(on_front(ring), ring.roofed(FACE_LINE[1])), (on_front(marks), marks.roofed(FEATURE_DEPTH))],
               lo=(-x_half, -0.25 * H, z_lo), hi=(x_half, front, z_hi))

    clay.cut_below(0.0)
    return clay


def build():
    """立てたまま刷る向き (足の裏がベッド)。"""
    return sculpt().mesh()
