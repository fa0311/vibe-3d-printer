"""STL から確認用の画像を描く (画面なしで動く)。"""

from __future__ import annotations

import os
from pathlib import Path

# デスクトップにウィンドウを出さず GPU/ソフトウェアで直接描く。vtk の import より前に設定する。
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")

import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True

BODY = "#f0a04b"
EDGE = "#2b2118"
BACKGROUND = "#f3f4f6"
GAP = 8.0  # 複数ピースを並べるときの間隔 (mm)


def layout(stls: list[Path]) -> list[pv.PolyData]:
    """ピースを X 方向に並べて読み込む (1個ならそのまま)。"""
    meshes, x = [], 0.0
    for path in stls:
        m = pv.read(str(path))
        b = m.bounds
        if len(stls) > 1:
            m = m.translate((x - b[0], -(b[2] + b[3]) / 2, -b[4]), inplace=False)
            x += (b[1] - b[0]) + GAP
        meshes.append(m)
    if len(stls) > 1:
        shift = -(x - GAP) / 2
        meshes = [m.translate((shift, 0, 0), inplace=False) for m in meshes]
    return meshes


def _add(pl: pv.Plotter, meshes: list[pv.PolyData], line_width: float = 1.6) -> None:
    for m in meshes:
        pl.add_mesh(
            m,
            color=BODY,
            smooth_shading=True,
            split_sharp_edges=True,
            feature_angle=30,
            ambient=0.25,
            diffuse=0.75,
            specular=0.15,
        )
        edges = m.extract_feature_edges(
            feature_angle=30, boundary_edges=False, non_manifold_edges=False, manifold_edges=False
        )
        if edges.n_cells:
            pl.add_mesh(edges, color=EDGE, line_width=line_width, lighting=False)


def _aim(pl: pv.Plotter, direction: tuple[float, float, float], up=(0, 0, 1), zoom: float = 1.0) -> None:
    """direction = 被写体から見たカメラの方向。"""
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.position = tuple(np.asarray(direction, dtype=float))
    pl.camera.up = up
    pl.reset_camera()
    pl.camera.zoom(zoom)


ISO = (1.0, -1.25, 0.9)
ISO_UNDER = (-1.0, 1.25, -0.9)


def hero(stls: list[Path], out: Path, size: tuple[int, int] = (1200, 900)) -> Path:
    """斜め上から見た1枚絵。"""
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background(BACKGROUND)
    _add(pl, layout(stls), line_width=2.0)
    pl.enable_anti_aliasing("ssaa")
    _aim(pl, ISO, zoom=1.15)
    pl.screenshot(str(out))
    pl.close()
    return out


def sheet(stls: list[Path], out: Path, size: tuple[int, int] = (1600, 1200)) -> Path:
    """4方向の確認図: 斜め上 / 斜め下(裏面) / 真上 / 正面。下2つは目盛り付き (mm)。"""
    meshes = layout(stls)
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=list(size), border=True, border_color="#c8ccd2")
    pl.set_background(BACKGROUND)

    def panel(row: int, col: int, label: str) -> None:
        pl.subplot(row, col)
        _add(pl, meshes)
        pl.add_text(label, position="upper_left", font_size=11, color="#333333")

    panel(0, 0, "ISO")
    _aim(pl, ISO)

    panel(0, 1, "UNDERSIDE")
    _aim(pl, ISO_UNDER)

    ruler = dict(
        grid=False, location="outer", ticks="outside", font_size=11, color="#444444",
        fmt="%.0f", n_xlabels=5, n_ylabels=5, n_zlabels=5, padding=0.0,
    )
    panel(1, 0, "TOP (mm)")
    pl.enable_parallel_projection()
    pl.show_bounds(xtitle="X", ytitle="Y", show_zaxis=False, **ruler)
    _aim(pl, (0, 0, 1), up=(0, 1, 0), zoom=1.05)

    panel(1, 1, "FRONT (mm)")
    pl.enable_parallel_projection()
    pl.show_bounds(xtitle="X", ztitle="Z", show_yaxis=False, **ruler)
    _aim(pl, (0, -1, 0), zoom=1.05)

    pl.enable_anti_aliasing("ssaa")
    pl.screenshot(str(out))
    pl.close()
    return out


def thumbnail(stls: list[Path], out: Path, px: int) -> Path:
    """.gcode.3mf に埋め込むサムネイル (背景透過・正方形)。"""
    pl = pv.Plotter(off_screen=True, window_size=[px, px])
    _add(pl, layout(stls), line_width=1.0)
    pl.enable_anti_aliasing("ssaa")
    _aim(pl, ISO, zoom=1.1)
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    return out
