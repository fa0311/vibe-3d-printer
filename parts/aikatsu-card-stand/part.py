"""aikatsu-card-stand: アイカツ!のバスボム (びっくらたまご) から出るカード型チャームを置く台。

アニメのフィッティングルームでカードをセットする、金色の紋章の形にする: 四つ角の星形の枠の中に、
上・左・中央・右の 4 か所に縦のカード、下に横向きのカード (学生証の位置) が十字に並ぶ
(持ち主がくれた場面の絵による)。下のカードは、チェーンを通す出っ張りが左に来る向きで置く。

机に寝かせて置く。カードは浅いくぼみに載り、半分 (1 mm) だけ頭が出るので、つまんで取れる。
"""

import cv2
import numpy as np
from build123d import *

from v3d import fit, trace

PREVIEW_COLOR = "#d4b24a"     # 確認用の絵の色 (金色のつもりの仮の色)

# ---- チャームの実測 (mm、持ち主がノギスの外側ジョウで測った) ----
CARD_W = 21.9         # カード本体の幅
CARD_H = 30.1         # カード本体の高さ
CARD_T = 2.0          # 厚み
TAB_W = 6.4           # ボールチェーンを通す出っ張りの幅
TAB_H = 34.7 - 30.1   # 出っ張りの高さ (出っ張り込みの全高 34.7)

# ---- 星の形 ----
# 中心から輪郭までの距離。先端 (0°) から、隣の先端との真ん中のくびれ (45°) まで 2.5° おき。
# アニメの場面の絵から紋章の内側の輪郭をなぞり、上下左右と斜めの対称をならして測った。
# 最初は「カードを囲みやすい、ふくらんだひし形」にして、持ち主に「もっと綺麗な星型」と言われた。
STAR = [0.998, 0.982, 0.951, 0.916, 0.884, 0.852, 0.819, 0.788, 0.757, 0.728,
        0.700, 0.679, 0.660, 0.642, 0.625, 0.614, 0.603, 0.595, 0.593]

# ---- 並べ方 (mm) ----
GAP_X = 8.0           # 左右のカードと中央のカードの間
GAP_Y = 4.0           # 中央のカードの出っ張りの先と、上のカードの間
ID_GAP = 7.0          # 中央のカードと、下の横向きのカードの間
WALL = 2.0            # くぼみと星の輪郭の間に最低限残す幅。星の大きさは、これを満たす最小にする

# ---- 台の寸法 (mm) ----
# すき間は「ゆるい」。大きい部品で刷り直しが重いので、確実に入る側に倒した (見た目の差は 0.1 mm)
CLEAR = fit.CLEAR_LOOSE
TAB_SIDE = 1.3        # 出っ張りの左右に空けるすき間 (実物は付け根が丸く広がって 8 mm ほどある)
RECESS = 1.0          # くぼみの深さ。カードの厚みの半分で、残り半分が頭を出す
CORNER = 0.6          # くぼみの角の丸み (実物の角の丸みは 0.7 mm ほど。それより小さくしておく)
ICON_DEPTH = 0.6      # くぼみの底に彫る絵 (何のカードを置くかの印) の深さ
PANEL_Z = 2.6         # パネル (カードを置く面) の高さ
FRAME_RISE = 3.0      # 枠がパネルより高い分
FRAME = 6.0           # 枠の幅
BEVEL = 1.4           # 枠の上の角の面取り (内側・外側とも 45°)
FOOT = 0.4            # 底面の外周の面取り

SCALE = 20            # 輪郭を作るときの細かさ (1 mm を何画素にするか)

PRINT = {}


def _slots():
    """くぼみの位置と大きさ。原点は中央のカードの中心、y は上が正。"""
    w, h = CARD_W + 2 * CLEAR, CARD_H + 2 * CLEAR
    tab_w, tab_h = TAB_W + 2 * TAB_SIDE, TAB_H + CLEAR
    side_x = w + GAP_X
    top_y = h / 2 + tab_h + GAP_Y + h / 2
    return {
        "card": (w, h), "tab": (tab_w, tab_h),
        # 名前: (中心 x, 中心 y, 回す角度)。アニメの並び: 上=アクセサリー、左=トップス、中央=ボトムス、右=シューズ、
        # 下=学生証。下だけは横向きで、出っ張りが左に来る (反時計回りに 90°)
        "cards": {
            "accessory": (0.0, top_y, 0), "tops": (-side_x, 0.0, 0), "bottoms": (0.0, 0.0, 0), "shoes": (side_x, 0.0, 0),
            "id": (0.0, -(h / 2 + ID_GAP + w / 2), 90),
        },
    }


def _star(radius: float, n: int = 1440) -> np.ndarray:
    """先端までの距離が radius の星の輪郭 (多角形)。先端は右・上・左・下。"""
    angle = np.linspace(0, 360, n, endpoint=False)
    folded = angle % 90
    folded = np.where(folded > 45, 90 - folded, folded)
    r = radius * np.interp(folded, np.arange(0, 45.01, 2.5), STAR)
    return np.stack([r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))], axis=1)


def _corners(s):
    """くぼみの角 (星の中に収まっているか確かめる点)。"""
    w, h = s["card"]
    tab_w, tab_h = s["tab"]
    one = [(sx * w / 2, sy * h / 2) for sx in (-1, 1) for sy in (-1, 1)] + [(sx * tab_w / 2, h / 2 + tab_h) for sx in (-1, 1)]
    points = []
    for cx, cy, angle in s["cards"].values():
        c, n = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        points += [(cx + c * x - n * y, cy + n * x + c * y) for x, y in one]
    return points


def _radius() -> float:
    """くぼみが全部、輪郭から WALL 以上離れて収まる、いちばん小さい星の大きさ。"""
    points = _corners(_slots())
    for radius in np.arange(40.0, 90.0, 0.5):
        star = _star(radius).astype(np.float32)
        if min(cv2.pointPolygonTest(star, (float(x), float(y)), True) for x, y in points) >= WALL:
            return float(radius)
    raise AssertionError("くぼみが星に収まらない")


class _Outline:
    """星の輪郭を外へ広げた形を、画素の地図で作る (広げる計算は、多角形より地図の方が破綻しない)。"""

    def __init__(self, radius: float):
        self.radius = radius
        # 枠の外側の先端は尖らせる。星の辺 (先端から 7.5°〜15° の所) をまっすぐ延ばした線で作るので、
        # その線の傾きと、線どうしが交わる点 (星の本当の先端は少し丸いので、それより少し先) を求めておく
        a, b = (radius * STAR[i] * np.array([np.cos(np.radians(2.5 * i)), np.sin(np.radians(2.5 * i))]) for i in (3, 6))
        self.half_angle = np.arctan2(b[1] - a[1], a[0] - b[0])
        self.apex = a[0] + a[1] / np.tan(self.half_angle)
        reach = self.apex + FRAME / np.sin(self.half_angle)
        self.size = int(np.ceil((reach + 3) * 2 * SCALE))
        self.center = (self.size / 2, self.size / 2)
        panel = np.zeros((self.size, self.size), np.uint8)
        cv2.fillPoly(panel, [self._px(_star(radius))], 255)
        self.distance = cv2.distanceTransform(255 - panel, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / SCALE

    def _px(self, points) -> np.ndarray:
        points = np.asarray(points, float)
        return np.round(np.stack([self.center[0] + points[:, 0] * SCALE, self.center[1] - points[:, 1] * SCALE], axis=1)).astype(np.int32)

    def mask(self, d: float, pointed: bool) -> np.ndarray:
        """星を外へ d だけ広げた形。pointed なら 4 つの先端を尖らせる (そのままだと半径 d で丸くなる)。"""
        out = np.where(self.distance <= d, 255, 0).astype(np.uint8)
        if pointed and d > 0:
            tip = self.apex + d / np.sin(self.half_angle)
            back = 24.0
            side = (tip - back * np.cos(self.half_angle), back * np.sin(self.half_angle))
            wedge = np.array([(tip, 0), side, (0.85 * self.radius, 0), (side[0], -side[1])])
            for quarter in range(4):
                c, s = np.cos(quarter * np.pi / 2), np.sin(quarter * np.pi / 2)
                cv2.fillPoly(out, [self._px(wedge @ np.array([[c, s], [-s, c]]))], 255)
            out = trace.smooth(out, 0.15 * SCALE)        # 尖らせた所と丸い所のつなぎ目をならす
        return out

    def faces(self, mask: np.ndarray):
        # 細かさは 1 画素ぶん (0.05 mm)。これより細かくすると画素の階段を全部拾って、立体化が終わらなくなる
        return trace.faces(mask, 1 / SCALE, self.center, detail_mm=1 / SCALE)

    def extent(self, mask: np.ndarray) -> float:
        ys, xs = np.nonzero(mask)
        return max(xs.max() - xs.min(), ys.max() - ys.min()) / SCALE


def _slab(shapes, z0: float, z1: float):
    return extrude(Sketch() + shapes, amount=z1 - z0).move(Location((0, 0, z0)))


def _face(points) -> Face:
    """多角形の面。点の並びは必ず反時計回りにそろえる: 時計回りだと面が裏返り、押し出しが下へ向かって、
    彫るはずの印が底の中の見えない空洞になる (靴の印で実際に起きた)。"""
    points = [(float(x), float(y)) for x, y in points]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    if area < 0:
        points.reverse()
    return Face(Wire.make_polygon([Vector(x, y) for x, y in points], close=True))


def _card_recess(s):
    """カード 1 枚分のくぼみの形 (本体 + 上の出っ張り)。"""
    w, h = s["card"]
    tab_w, tab_h = s["tab"]
    # 出っ張りの分は、本体に 2 mm 食い込ませて 1 つの形にする
    return RectangleRounded(w, h, CORNER) + Pos(0, h / 2 + (tab_h - 2) / 2) * RectangleRounded(tab_w, tab_h + 2, 1.0)


# くぼみの底に彫る印 (アニメの紋章の絵に合わせた、ごく簡単な絵)。原点が中心、12 mm 角くらい
def _five_point_star(outer: float = 6.0, inner: float = 2.6):
    angles = np.pi / 2 + np.arange(10) * np.pi / 5
    radius = np.where(np.arange(10) % 2 == 0, outer, inner)
    return _face(np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1))


def _ring():
    """下のくぼみ (学生証の位置) の印: 宝石の付いた指輪。"""
    return Pos(0, -1.4) * ((Circle(4.6) - Circle(3.1)) + _face([(0, 7.6), (-2.2, 5.6), (0, 3.6), (2.2, 5.6)]))


ICONS = {
    "accessory": _five_point_star,
    "tops": lambda: _face([(-2, 5.5), (-6.5, 3.6), (-5.2, 0.6), (-3.6, 1.4), (-3.6, -5.5), (3.6, -5.5), (3.6, 1.4),
                           (5.2, 0.6), (6.5, 3.6), (2, 5.5), (0, 4.0)]),
    "bottoms": lambda: _face([(-3, 5.5), (-3, 4.0), (-6, -5.5), (6, -5.5), (3, 4.0), (3, 5.5)]),
    "shoes": lambda: _face([(-0.5, 6), (3, 6), (3, -5.5), (-5.5, -5.5), (-5.5, -3.2), (-3.5, -1.6), (-0.5, -1.2)]),
    "id": _ring,
}


def build():
    """表を上にして寝かせて刷る (裏は平ら、サポートなし)。"""
    s = _slots()
    outline = _Outline(_radius())
    top = PANEL_Z + FRAME_RISE
    layer = fit.LAYER
    outer = outline.mask(FRAME, pointed=True)
    assert outline.extent(outer) <= min(fit.BED[:2]) - 10, f"ベッドに載らない: {outline.extent(outer):.1f} mm"

    # 台。底は外周を面取りする (1層目が広がっても角が立たない)。
    # 斜面はどれも 1 層ごとの段で作る (プリンタは 1 層ずつ積むので、刷り上がりは斜面と同じ)
    part = _slab(outline.faces(outer), FOOT, PANEL_Z)
    for k in range(int(round(FOOT / layer))):
        foot = outline.mask(FRAME - FOOT + layer * (k + 0.5), pointed=True)
        part += _slab(outline.faces(foot), layer * k, layer * (k + 1))

    # 枠: まっすぐ立ち上げて、上の角は内側・外側とも 45° に落とす
    panel = outline.mask(0, pointed=False)
    part += _slab(outline.faces(outer & ~panel), PANEL_Z, top - BEVEL)
    for k in range(int(round(BEVEL / layer))):
        inset = layer * (k + 0.5)
        ring = outline.mask(FRAME - inset, pointed=True) & ~outline.mask(inset, pointed=False)
        part += _slab(outline.faces(ring), top - BEVEL + layer * k, top - BEVEL + layer * (k + 1))

    # カードのくぼみと、その底の印
    floor = PANEL_Z - RECESS
    recess = _card_recess(s)
    for name, (x, y, angle) in s["cards"].items():
        # Pos(...) * 形 は移した写しを返す (.move() は元の形そのものを動かすので、使い回す形には使えない)
        part -= _slab(Pos(x, y) * Rot(0, 0, angle) * recess, floor, top)
        part -= _slab(Pos(x, y) * ICONS[name](), floor - ICON_DEPTH, floor + 0.1)
    return part
