"""どこから見ても丸い立体 (フィギュア、マスコット) を作る道具。

build123d は板・箱・穴のような形は得意だが、粘土でこねたような丸い形は作れない。
ここでは、管や柱の「表面までの距離」を細かい格子の上で計算し、それらをなめらかにつないでから、
表面を三角形の網 (メッシュ) として取り出す。part.py の build() は、その網をそのまま返せばよい。

    clay = Clay(bounds=((-30, 30), (-20, 20), (-1, 82)), voxel=0.25)
    clay.add(Column(...))                 # 最初の塊
    clay.add(Tube(...), blend=2.5)        # つなぎ目を幅 2.5 mm で丸めてくっつける
    clay.press([(pattern, 0.6)], window)  # 表面に沿って、模様を深さ 0.6 mm だけ押し込む
    clay.cut_below(0.0)                   # 底を平らに切る (ベッドに接する面)
    mesh = clay.mesh()

長さはすべて mm。距離は「外が正、中が負」。使用例: parts/suica-bunny。
"""

from __future__ import annotations

import numpy as np

FAR = 1.0e3


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """2 つの形の和。k > 0 なら、つなぎ目を幅 k くらいで丸める。"""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _resample(points, radii, squash, steps: int):
    """制御点をなめらかな曲線でつなぎ、細かい折れ線にする。太さとつぶし具合も一緒に補間する。"""
    from scipy.interpolate import CubicSpline, PchipInterpolator

    points = np.asarray(points, float)
    radii = np.asarray(radii, float)
    squash = np.asarray(squash, float)
    if squash.ndim == 1:
        squash = np.tile(squash, (len(points), 1))
    if len(points) == 2:
        t = np.linspace(0, 1, 2)
        return points, radii, squash
    chord = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    t = np.linspace(0, chord[-1], (len(points) - 1) * steps + 1)
    # 太さは PCHIP (制御点の間で行き過ぎない)、中心線は 3 次スプライン
    return CubicSpline(chord, points)(t), PchipInterpolator(chord, radii)(t), PchipInterpolator(chord, squash)(t)


class Tube:
    """曲がった管。太さは場所ごとに変えられる (腕、耳、足など)。両端は丸い。

    points: 中心線が通る点 [(x, y, z), ...] / radii: 各点での太さ (半径)
    squash: 断面のつぶし方 (x, y, z の倍率)。(1, 0.7, 1) なら奥行きだけ 0.7 倍の、平たい管になる。
            点ごとに変えたいときは、点の数だけ並べる。
    """

    def __init__(self, points, radii, squash=(1.0, 1.0, 1.0), steps: int = 8):
        self.points, self.radii, self.squash = _resample(points, radii, squash, steps)

    def bounds(self):
        reach = self.radii[:, None] * np.maximum(self.squash, 1.0)
        return (self.points - reach).min(axis=0), (self.points + reach).max(axis=0)

    def distance(self, x, y, z):
        best = None
        for i in range(len(self.points) - 1):
            s = (self.squash[i] + self.squash[i + 1]) / 2
            a, b = self.points[i] / s, self.points[i + 1] / s
            ra, rb = self.radii[i], self.radii[i + 1]
            ba = (b - a).astype(np.float32)
            px, py, pz = x / s[0] - a[0], y / s[1] - a[1], z / s[2] - a[2]
            t = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / max(float(ba @ ba), 1e-9), 0.0, 1.0)
            d = np.sqrt((px - ba[0] * t) ** 2 + (py - ba[1] * t) ** 2 + (pz - ba[2] * t) ** 2) - (ra + (rb - ra) * t)
            # つぶした空間での距離なので、いちばん縮めた向きの倍率で mm に近づける
            d *= float(s.min())
            best = d if best is None else np.minimum(best, d)
        return best.astype(np.float32)


class Column:
    """縦に立った柱 (胴体など)。高さごとに太さ・幅と奥行きの比・前後の位置を変えられる。上下は丸い。

    heights: 中心線上の高さ [z, ...] / radii: その高さでの太さ (半径の基準)
    width, depth: 断面の幅・奥行きの倍率 (高さごと)。1 と 1 なら円、1.1 と 0.9 なら横長の楕円
    lean: 断面の中心を前後 (y) にずらす量 (高さごと)
    """

    def __init__(self, heights, radii, width=1.0, depth=1.0, lean=0.0, steps: int = 8):
        from scipy.interpolate import PchipInterpolator

        heights = np.asarray(heights, float)
        full = lambda v: np.broadcast_to(np.asarray(v, float), heights.shape)  # noqa: E731
        self.heights, self.radii = heights, np.asarray(radii, float)
        self._width, self._depth, self._lean = (PchipInterpolator(heights, full(v)) for v in (width, depth, lean))
        self._axis = Tube([(0.0, 0.0, h) for h in heights], radii, steps=steps)
        self._reach = float(self.radii.max() * max(np.max(full(width)), np.max(full(depth))) + np.abs(full(lean)).max())

    def bounds(self):
        lo, hi = self.heights[0] - self.radii[0], self.heights[-1] + self.radii[-1]
        return np.array([-self._reach, -self._reach, lo]), np.array([self._reach, self._reach, hi])

    def distance(self, x, y, z):
        h = np.clip(z, self.heights[0], self.heights[-1])
        width, depth, lean = (f(h).astype(np.float32) for f in (self._width, self._depth, self._lean))
        return self._axis.distance(x / width, (y - lean) / depth, z) * np.minimum(np.minimum(width, depth), 1.0)


class Stencil:
    """正面から見た模様 (顔など)。mask (0/255 の画像) の白い所が模様。

    inside(x, z) は「模様の内側へどれだけ入っているか」(mm、内側が正) を返す。
    center_px: 画像のどの画素が、立体の (x0, z0) に当たるか。画像は上が +z。
    """

    def __init__(self, mask: np.ndarray, mm_per_px: float, center_px, at=(0.0, 0.0)):
        import cv2

        inside = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.signed = ((inside - outside) * mm_per_px).astype(np.float32)
        self.mm, self.center, self.at = mm_per_px, center_px, at
        # 模様の中の各点から、真上にある模様の縁までの距離 (天井を斜めにするのに使う)
        room = np.zeros(mask.shape, np.float32)
        run = np.zeros(mask.shape[1], np.float32)
        for row in range(mask.shape[0]):
            run = (run + 1) * (mask[row] > 0)
            room[row] = run
        self.room = room * mm_per_px

    def bounds_xz(self):
        h, w = self.signed.shape
        return ((self.at[0] - self.center[0] * self.mm, self.at[0] + (w - self.center[0]) * self.mm),
                (self.at[1] - (h - self.center[1]) * self.mm, self.at[1] + self.center[1] * self.mm))

    def _sample(self, image, x, z, border: float):
        import cv2

        u = np.broadcast_to(self.center[0] + (x - self.at[0]) / self.mm, np.broadcast(x, z).shape).astype(np.float32)
        v = np.broadcast_to(self.center[1] - (z - self.at[1]) / self.mm, u.shape).astype(np.float32)
        flat = cv2.remap(image, u.reshape(1, -1), v.reshape(1, -1), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        return flat.reshape(u.shape)

    def inside(self, x, z):
        return self._sample(self.signed, x, z, -FAR)

    def roofed(self, depth: float, slope: float = 1.0):
        """深さを決める関数を返す。穴の天井が斜め (slope = 1 で 45°) になるよう、上の縁に近いほど浅くする。

        立てて刷る物の横向きの穴は、天井が水平だと宙に浮いて、スライサーが穴の中に支えを立ててしまう。
        天井を 45° の斜面にすれば支えなしで刷れる。斜面は下を向くので、上から見下ろすと穴は深いままに見える。
        """
        return lambda x, y, z: np.minimum(depth, slope * self._sample(self.room, x, z, 0.0))


class Clay:
    """格子の上で形を足したり削ったりして、最後にメッシュにする。"""

    def __init__(self, bounds, voxel: float = 0.25):
        self.voxel = voxel
        # 格子点が z = 0 などの切りのよい面に乗らないよう、半目盛りずらす (平らに切った面がきれいに出る)
        self.axes = [np.arange(np.floor(lo / voxel) + 0.5, np.ceil(hi / voxel) + 0.5) * voxel for lo, hi in bounds]
        self.axes = [a.astype(np.float32) for a in self.axes]
        self.field = np.full([len(a) for a in self.axes], FAR, np.float32)

    def _window(self, lo, hi, margin: float):
        slices = []
        for axis, a, b in zip(self.axes, lo, hi):
            i = int(np.searchsorted(axis, a - margin))
            j = int(np.searchsorted(axis, b + margin))
            slices.append(slice(max(i - 1, 0), min(j + 1, len(axis))))
        return tuple(slices)

    def grid(self, window=None):
        """格子点の座標 (x, y, z)。掛け合わせると 3 次元に広がる形で返す。"""
        window = window or (slice(None),) * 3
        x, y, z = (a[s] for a, s in zip(self.axes, window))
        return x[:, None, None], y[None, :, None], z[None, None, :]

    def add(self, shape, blend: float = 0.0):
        """形を足す。blend > 0 なら、つなぎ目をその幅くらいで丸める。"""
        w = self._window(*shape.bounds(), margin=blend + 2 * self.voxel)
        self.field[w] = _smooth_min(self.field[w], shape.distance(*self.grid(w)), blend)

    def carve(self, shape, blend: float = 0.0):
        """形を削り取る。"""
        w = self._window(*shape.bounds(), margin=blend + 2 * self.voxel)
        self.field[w] = -_smooth_min(-self.field[w], shape.distance(*self.grid(w)), blend)

    def press(self, patterns, lo, hi):
        """表面に沿って模様を押し込む。patterns = [(inside(x, y, z), 深さ), ...]。深さは数か、(x, y, z) の関数。

        どの模様も、押し込む前の表面から測った深さになる (深い模様を浅い模様の中に重ねてよい)。
        lo, hi: 模様のある範囲 (x, y, z の下限・上限)。
        """
        w = self._window(lo, hi, margin=2 * self.voxel)
        before = self.field[w].copy()
        x, y, z = self.grid(w)
        for inside, depth in patterns:
            if callable(depth):               # 場所ごとに深さを変える (Stencil.roofed など)
                depth = depth(x, y, z)
            self.field[w] = np.maximum(self.field[w], np.minimum(inside(x, y, z), before + depth))

    def cut_below(self, z0: float):
        """z0 より下を切り落とす (底を平らにする)。"""
        self.field = np.maximum(self.field, z0 - self.axes[2][None, None, :])

    def silhouette(self, axis: int) -> np.ndarray:
        """axis の向きから見た影 (0/255)。axis=1 なら正面、axis=0 なら横。行が z (上が先頭)。"""
        shadow = (self.field < 0).any(axis=axis)
        return np.where(shadow.T[::-1], 255, 0).astype(np.uint8)

    def mesh(self):
        """表面を三角形の網にして返す (trimesh.Trimesh)。"""
        import pyvista as pv
        import trimesh

        grid = pv.ImageData(dimensions=self.field.shape, spacing=(self.voxel,) * 3,
                            origin=tuple(float(a[0]) for a in self.axes))
        grid.point_data["d"] = self.field.ravel(order="F")
        surface = grid.contour([0.0], scalars="d", method="flying_edges").triangulate().clean()
        mesh = trimesh.Trimesh(surface.points, surface.faces.reshape(-1, 4)[:, 1:], process=True)
        mesh.fix_normals()
        return mesh
