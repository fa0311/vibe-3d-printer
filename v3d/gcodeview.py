"""スライス結果 (.gcode.3mf) の中身を読んで、本体と支え (サポート) がどこに刷られるかを確かめる。

Bambu Studio のプレビュー画面は視点を変えにくいので、G コードを直接読んで絵にする。
支えを使う部品では、**支えがベッドから途切れずに、支える相手まで届いているか** をここで必ず見る。
枝分かれする支え (tree) は、背の高い所へ登る枝が針のように細くなったり、途中の層が抜けたりすることがある
(parts/suica-bunny で実際に起きた)。
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

_WORD = {c: re.compile(rf" {c}(-?[\d.]+)") for c in "XYZE"}


def segments(gcode_3mf: Path) -> dict[str, np.ndarray]:
    """刷る線を種類ごとに返す。各行は (z, x0, y0, x1, y1)。種類は "model" / "support" / "brim"。"""
    archive = zipfile.ZipFile(gcode_3mf)
    name = next(n for n in archive.namelist() if n.endswith(".gcode"))
    out: dict[str, list] = {"model": [], "support": [], "brim": []}
    feature, z, x, y = "", 0.0, 0.0, 0.0
    for line in archive.read(name).decode("utf-8", "ignore").splitlines():
        if line.startswith("; FEATURE: "):
            feature = line[11:].strip()
            continue
        if not line.startswith(("G1 ", "G0 ", "G2 ", "G3 ")):
            continue
        word = {c: p.search(line) for c, p in _WORD.items()}
        if word["Z"]:
            z = float(word["Z"].group(1))
        nx = float(word["X"].group(1)) if word["X"] else x
        ny = float(word["Y"].group(1)) if word["Y"] else y
        extruding = word["E"] and float(word["E"].group(1)) > 0 and (word["X"] or word["Y"])
        if extruding and feature and feature != "Custom":
            kind = "support" if "upport" in feature else "brim" if feature in ("Brim", "Skirt") else "model"
            out[kind].append((z, x, y, nx, ny))
        x, y = nx, ny
    return {k: np.array(v, float).reshape(-1, 5) for k, v in out.items()}


def support_profile(seg: dict[str, np.ndarray], band: float = 4.0) -> list[tuple[float, int, float, float]]:
    """支えの量を高さの帯ごとに: (帯の下端, 支えのある層の数, 1 層あたりの線の長さの最小, 中央値)。

    最小が数 mm しか無い帯は、支えが針のように細い。層の数が周りより少ない帯は、途中が抜けている。
    """
    s = seg["support"]
    if len(s) == 0:
        return []
    length = np.hypot(s[:, 3] - s[:, 1], s[:, 4] - s[:, 2])
    rows = []
    for lo in np.arange(0.0, s[:, 0].max(), band):
        pick = (s[:, 0] >= lo) & (s[:, 0] < lo + band)
        levels = np.unique(s[pick, 0])
        per_layer = [length[pick & (s[:, 0] == lv)].sum() for lv in levels]
        rows.append((float(lo), len(levels), float(min(per_layer, default=0.0)), float(np.median(per_layer or [0.0]))))
    return rows


def report(seg: dict[str, np.ndarray]) -> str:
    rows = support_profile(seg)
    if not rows:
        return "  支え: なし"
    top = seg["support"][:, 0].max()
    # いちばん下 (1 層目) といちばん上 (支えの先端) の帯は、もともと量が少ないので見ない
    thin = [f"z {lo:.0f}〜{lo + 4:.0f} mm ({low:.0f} mm)" for lo, n, low, mid in rows[1:-1] if n and low < 10]
    lines = [f"  支え: ベッドから高さ {top:.1f} mm まで。1 層あたりの線の長さは中央値 {np.median([r[3] for r in rows]):.0f} mm"]
    if thin:
        lines.append("    ! 支えが細い (1 層あたり 10 mm 未満の) 帯: " + ", ".join(thin[:8]) + (" ほか" if len(thin) > 8 else ""))
    return "\n".join(lines)


def plot(seg: dict[str, np.ndarray], out: Path, px_per_mm: float = 9.0) -> Path:
    """正面 / 横 / 真上 から見た、本体 (紫) と支え (橙) の線。"""
    import cv2

    model = seg["model"]
    everything = np.concatenate([v for v in seg.values() if len(v)])
    cx = (model[:, [1, 3]].min() + model[:, [1, 3]].max()) / 2
    cy = (model[:, [2, 4]].min() + model[:, [2, 4]].max()) / 2
    half = max(np.abs(everything[:, [1, 3]] - cx).max(), np.abs(everything[:, [2, 4]] - cy).max()) + 4
    height = everything[:, 0].max() + 6
    s = px_per_mm

    def panel(label, project, w_mm, h_mm, ground):
        w, h = int(w_mm * s), int(h_mm * s)
        img = np.full((h, w, 3), 255, np.uint8)
        for kind, color in (("model", (240, 175, 205)), ("brim", (200, 200, 200)), ("support", (0, 130, 255))):
            a = seg[kind]
            if len(a) == 0:
                continue
            u0, v0 = project(a[:, 1], a[:, 2], a[:, 0])
            u1, v1 = project(a[:, 3], a[:, 4], a[:, 0])
            lines = np.stack([w / 2 + u0 * s, ground(h) - v0 * s, w / 2 + u1 * s, ground(h) - v1 * s], axis=1)
            cv2.polylines(img, list(np.round(lines).astype(np.int32).reshape(-1, 2, 2)), False, color, 1)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return img

    from_floor = lambda h: h - 12  # noqa: E731
    from_middle = lambda h: h / 2  # noqa: E731
    front = panel("front", lambda x, y, z: (x - cx, z), 2 * half, height, from_floor)
    side = panel("side (front on the left)", lambda x, y, z: (y - cy, z), 2 * half, height, from_floor)
    top = panel("top (front at the bottom)", lambda x, y, z: (x - cx, y - cy), 2 * half, 2 * half, from_middle)
    tall = max(i.shape[0] for i in (front, side, top))
    pad = [np.pad(i, ((tall - i.shape[0], 0), (0, 0), (0, 0)), constant_values=255) for i in (front, side, top)]
    cv2.imwrite(str(out), np.hstack(pad))
    return out
