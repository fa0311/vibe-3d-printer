"""parts/<名前>/part.py を実行して、検査済みの STL/STEP を out/ に書き出す。

part.py の約束:
    def build():           -> 形状1個、または {"ピース名": 形状} (印刷する向きで作る。Z+ が上)
                              形状は build123d の立体か、三角形の網 (trimesh.Trimesh。v3d/blob.py で作る丸い形)
    PRINT = {...}          -> スライス設定 (省略可。v3d/slicer.py 参照)
    def assembly():        -> 組み立てた状態の形状 (省略可。確認用の絵にだけ使う)
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from . import fit
from .paths import REPO, part_dir

# 垂直からこの角度を超えて張り出す面はサポート無しでは荒れる。
# 45° ちょうどの面取りは問題なく刷れるので、判定には少し余裕を持たせる。
OVERHANG_DEG = 45.0
_OVERHANG_MARGIN_DEG = 3.0


@dataclass
class Piece:
    name: str
    stl: Path
    step: Path | None          # 三角形の網から作ったピースには STEP が無い
    size: tuple[float, float, float]
    volume_mm3: float
    bed_contact_mm2: float
    overhang_mm2: float
    ceiling_mm2: float
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Built:
    name: str
    dir: Path
    out: Path
    pieces: list[Piece]
    print_settings: dict
    assembly_stl: Path | None
    preview_color: str | None = None

    @property
    def ok(self) -> bool:
        return not any(p.problems for p in self.pieces)


def load(name: str) -> ModuleType:
    path = part_dir(name) / "part.py"
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(f"v3d_part_{name.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "build", None)):
        raise AttributeError(f"{path} に build() がありません")
    return mod


def _is_mesh(shape) -> bool:
    import trimesh

    return isinstance(shape, trimesh.Trimesh)


def _export(shape, stl: Path, step: Path | None = None) -> None:
    if _is_mesh(shape):
        shape.export(str(stl))
        return
    from build123d import export_step, export_stl

    export_stl(shape, str(stl), tolerance=0.01, angular_tolerance=0.1)
    if step is not None:
        export_step(shape, str(step))


def _seat_on_bed(shape):
    """XY は原点中心、底面は Z=0 に置く。"""
    if _is_mesh(shape):
        lo, hi = shape.bounds
        seated = shape.copy()
        seated.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
        return seated
    from build123d import Location

    bb = shape.bounding_box()
    c = bb.center()
    return shape.moved(Location((-c.X, -c.Y, -bb.min.Z)))


def _measure(shape) -> tuple[tuple[float, float, float], float]:
    """(外寸, 体積 mm³)。"""
    if _is_mesh(shape):
        return tuple(float(v) for v in shape.extents), float(shape.volume)
    bb = shape.bounding_box()
    return (bb.size.X, bb.size.Y, bb.size.Z), float(shape.volume)


def _inspect(piece: Piece, shape) -> None:
    import trimesh

    if _is_mesh(shape):
        solids = shape.body_count
    else:
        if not shape.is_valid:
            piece.problems.append("形状が壊れています (is_valid=False)")
        solids = len(shape.solids())
    if solids == 0:
        piece.problems.append("ソリッドがありません (面だけの形状)")
    elif solids > 1:
        piece.warnings.append(
            f"{solids} 個の離れた塊でできています。意図していなければピースを分けるか、つなげること"
        )

    for axis, size, limit in zip("XYZ", piece.size, fit.BED):
        if size > limit:
            piece.problems.append(f"{axis} が {size:.1f} mm で、造形範囲 {limit:.0f} mm を超えています")

    mesh = trimesh.load_mesh(piece.stl)
    if not mesh.is_watertight:
        piece.problems.append("メッシュが閉じていません (穴あき)")
    elif piece.volume_mm3 > 0 and abs(mesh.volume - piece.volume_mm3) / piece.volume_mm3 > 0.01:
        piece.problems.append(
            f"メッシュの体積 {mesh.volume:.0f} mm³ が形状の体積 {piece.volume_mm3:.0f} mm³ と合いません"
        )

    nz = mesh.face_normals[:, 2]
    area = mesh.area_faces
    z = mesh.triangles_center[:, 2]
    on_bed = z < 0.02
    down = nz < -math.sin(math.radians(OVERHANG_DEG + _OVERHANG_MARGIN_DEG))
    flat = nz < -0.999
    piece.bed_contact_mm2 = float(area[on_bed & flat].sum())
    piece.ceiling_mm2 = float(area[~on_bed & flat].sum())
    piece.overhang_mm2 = float(area[~on_bed & down & ~flat].sum())

    if piece.bed_contact_mm2 < 1.0:
        piece.problems.append("ベッドに接する平らな面がありません。向きを見直すこと")
    else:
        footprint = math.sqrt(piece.bed_contact_mm2)
        if piece.size[2] > 6 * footprint:
            piece.warnings.append(
                f"接地面 {piece.bed_contact_mm2:.0f} mm² に対して高さ {piece.size[2]:.0f} mm。"
                "倒れやすいのでブリムを付けるか寝かせること"
            )
    if piece.overhang_mm2 > 20:
        piece.warnings.append(
            f"{OVERHANG_DEG:.0f}° を超える斜めの張り出しが {piece.overhang_mm2:.0f} mm² あります。"
            "向きを変える・面取りにする・サポートを付ける のどれかを検討"
        )
    if piece.ceiling_mm2 > 20:
        piece.warnings.append(
            f"宙に浮いた水平な下面が {piece.ceiling_mm2:.0f} mm² あります。"
            "短いブリッジなら可。長い場合はサポートが必要"
        )


def build(name: str) -> Built:
    mod = load(name)
    pdir = part_dir(name)
    out = pdir / "out"
    out.mkdir(exist_ok=True)
    for old in list(out.glob("*.stl")) + list(out.glob("*.step")):
        old.unlink()

    result = mod.build()
    shapes = result if isinstance(result, dict) else {name: result}
    if not shapes:
        raise ValueError("build() が空です")

    pieces = []
    for pname, shape in shapes.items():
        shape = _seat_on_bed(shape)
        size, volume = _measure(shape)
        piece = Piece(
            name=pname,
            stl=out / f"{pname}.stl",
            step=None if _is_mesh(shape) else out / f"{pname}.step",
            size=size,
            volume_mm3=volume,
            bed_contact_mm2=0.0,
            overhang_mm2=0.0,
            ceiling_mm2=0.0,
        )
        _export(shape, piece.stl, piece.step)
        _inspect(piece, shape)
        pieces.append(piece)

    assembly_stl = None
    if callable(getattr(mod, "assembly", None)):
        assembly_stl = out / "_assembly.stl"
        _export(mod.assembly(), assembly_stl)

    built = Built(
        name=name,
        dir=pdir,
        out=out,
        pieces=pieces,
        print_settings=dict(getattr(mod, "PRINT", {})),
        assembly_stl=assembly_stl,
        preview_color=getattr(mod, "PREVIEW_COLOR", None),
    )
    (out / "build.json").write_text(
        json.dumps(
            {
                "part": name,
                "ok": built.ok,
                "pieces": [
                    {
                        "name": p.name,
                        "size_mm": [round(v, 3) for v in p.size],
                        "volume_cm3": round(p.volume_mm3 / 1000, 3),
                        "bed_contact_mm2": round(p.bed_contact_mm2, 1),
                        "overhang_mm2": round(p.overhang_mm2, 1),
                        "ceiling_mm2": round(p.ceiling_mm2, 1),
                        "problems": p.problems,
                        "warnings": p.warnings,
                    }
                    for p in pieces
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return built


def report(built: Built) -> str:
    lines = [f"パーツ: {built.name}  ({'OK' if built.ok else '問題あり'})"]
    for p in built.pieces:
        x, y, z = p.size
        lines.append(
            f"  [{p.name}] {x:.2f} × {y:.2f} × {z:.2f} mm, 体積 {p.volume_mm3 / 1000:.2f} cm³, "
            f"接地 {p.bed_contact_mm2:.0f} mm²"
        )
        lines += [f"    ✗ {m}" for m in p.problems]
        lines += [f"    ! {m}" for m in p.warnings]
    return "\n".join(lines)
