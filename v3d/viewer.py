"""ブラウザで回して見られる 3D ビューア (外部ライブラリ無しの単一 HTML) を書き出す。"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from . import render

_TEMPLATE = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def write(stls: list[Path], out: Path, title: str) -> Path:
    import pyvista as pv

    def dims(path: Path) -> str:
        b = pv.read(str(path)).bounds
        return " × ".join(f"{b[i + 1] - b[i]:.1f}" for i in (0, 2, 4))

    if len(stls) == 1:
        caption = f"外寸 {dims(stls[0])} mm"
    else:
        caption = " / ".join(f"{p.stem} {dims(p)}" for p in stls) + " mm"
    meshes = render.layout(stls)
    merged = meshes[0] if len(meshes) == 1 else pv.merge(meshes, merge_points=False)
    surf = merged.compute_normals(
        cell_normals=False, point_normals=True, split_vertices=True, feature_angle=30,
        auto_orient_normals=False, consistent_normals=False,
    )
    edges = merged.extract_feature_edges(
        feature_angle=30, boundary_edges=False, non_manifold_edges=False, manifold_edges=False
    )

    lo = np.asarray(merged.bounds[0::2], dtype=np.float64)
    hi = np.asarray(merged.bounds[1::2], dtype=np.float64)
    size = np.maximum(hi - lo, 1e-6)

    def quantize(points: np.ndarray) -> np.ndarray:
        return np.round((points - lo) / size * 65535).astype("<u2")

    normals = np.zeros((surf.n_points, 4), dtype=np.int8)
    normals[:, :3] = np.round(np.clip(surf.point_data["Normals"], -1, 1) * 127)
    line_points = (
        edges.points[edges.lines.reshape(-1, 3)[:, 1:].ravel()] if edges.n_cells else np.zeros((0, 3))
    )

    data = {
        "title": title,
        "caption": caption,
        "min": lo.tolist(),
        "size": size.tolist(),
        "pos": _b64(quantize(surf.points)),
        "nrm": _b64(normals),
        "idx": _b64(surf.faces.reshape(-1, 4)[:, 1:].astype("<u4")),
        "lines": _b64(quantize(line_points)),
    }
    out.write_text(_TEMPLATE.replace("/*DATA*/null", json.dumps(data)), encoding="utf-8")
    return out
