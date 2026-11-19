"""使い方: uv run python -m v3d <コマンド> ...   (リポジトリのルートで実行)"""

from __future__ import annotations

import argparse
import shutil
import sys

from . import config
from .paths import PARTS, REPO

TEMPLATE = '''"""{name}

何のための部品か / どこに付くか をここに書く。
寸法の根拠 (ノギスの実測値) は README.md に残す。
"""

from build123d import *

from v3d import fit

# ---- 寸法 (mm) ----
WIDTH = 40.0
DEPTH = 25.0
HEIGHT = 10.0

PRINT = {{}}


def build():
    """印刷する向きで作る (Z+ が上、底面がベッド)。"""
    with BuildPart() as p:
        Box(WIDTH, DEPTH, HEIGHT)
    return p.part
'''


def cmd_new(args) -> int:
    d = PARTS / args.part
    if d.exists():
        print(f"parts/{args.part} は既にあります", file=sys.stderr)
        return 1
    d.mkdir(parents=True)
    (d / "part.py").write_text(TEMPLATE.format(name=args.part), encoding="utf-8")
    (d / "README.md").write_text(
        f"# {args.part}\n\n## 目的\n\n## 実測値\n\n## 印刷の記録\n", encoding="utf-8"
    )
    print(f"作成: {d.relative_to(REPO)}/part.py")
    return 0


def _build(part: str):
    from . import model, render, viewer

    built = model.build(part)
    stls = [p.stl for p in built.pieces]
    render.hero(stls, built.out / "preview.png")
    render.sheet(stls, built.out / "views.png")
    tint = {"color": built.preview_color} if built.preview_color else {}
    render.flat(stls, built.out / "flat.png", **tint)
    render.turnaround(stls, built.out / "turnaround.png", **tint)
    if built.assembly_stl:
        render.hero([built.assembly_stl], built.out / "assembly.png")
    viewer.write(built.assembly_stl and [built.assembly_stl] or stls, built.out / "viewer.html", built.name)
    return built


def cmd_build(args) -> int:
    from . import model

    built = _build(args.part)
    print(model.report(built))
    print(f"  画像: {built.out / 'preview.png'}")
    print(f"        {built.out / 'views.png'}")
    print(f"        {built.out / 'flat.png'}  (真上から、陰影つき)")
    print(f"        {built.out / 'turnaround.png'}  (正面・斜め・横・背面)")
    if built.assembly_stl:
        print(f"        {built.out / 'assembly.png'}")
    print(f"  3Dビューア: {built.out / 'viewer.html'}")
    return 0 if built.ok else 1


def cmd_slice(args) -> int:
    from . import model, slicer

    built = _build(args.part)
    print(model.report(built))
    try:
        sliced = slicer.slice_part(built, process=args.process, filament=args.filament)
    except slicer.SliceError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    print(slicer.report(sliced))
    if sliced.settings.get("enable_support"):
        from . import gcodeview

        seg = gcodeview.segments(sliced.file)
        print(gcodeview.report(seg))
        print(f"    図: {gcodeview.plot(seg, built.out / 'supports.png')}  (本体と支えの位置)")
    return 0


def cmd_profiles(args) -> int:
    from . import profiles

    suffix = config.spec().profile_suffix
    for name in profiles.names(args.kind):
        if name.endswith(suffix) and (not args.query or args.query.lower() in name.lower()):
            print(name.removesuffix(suffix).strip())
    return 0


def cmd_status(args) -> int:
    from . import printer

    try:
        print(printer.describe(printer.status()))
    except (printer.PrinterError, FileNotFoundError, ValueError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    return 0


def cmd_print(args) -> int:
    from . import printer

    out = PARTS / args.part / "out"
    file = out / f"{args.part}.gcode.3mf"
    if not file.exists():
        print(f"{file} がありません。先に slice すること", file=sys.stderr)
        return 1
    # 直した後にスライスし忘れて、古い形を印刷してしまうのを防ぐ
    sources = [PARTS / args.part / "part.py", REPO / "printer.toml", *(REPO / "v3d").glob("*.py")]
    newer = [p.relative_to(REPO) for p in sources if p.stat().st_mtime > file.stat().st_mtime]
    if newer:
        print(f"スライス後に変更されたファイルがあります: {', '.join(map(str, newer))}\n"
              "slice し直してから印刷すること", file=sys.stderr)
        return 1
    try:
        printer.send_and_start(file, bed_leveling=not args.no_leveling, ams_slot=args.ams_slot)
    except (printer.PrinterError, FileNotFoundError, ValueError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    return 0


def cmd_control(args) -> int:
    from . import printer

    try:
        printer.control(args.cmd)
        print(printer.describe(printer.status()))
    except (printer.PrinterError, FileNotFoundError, ValueError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    return 0


def cmd_clean(args) -> int:
    for out in PARTS.glob("*/out"):
        shutil.rmtree(out)
        print(f"削除: {out.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v3d", description="モデリングから印刷まで")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="パーツの雛形を作る")
    p.add_argument("part")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("build", help="モデルを生成・検査して画像を描く")
    p.add_argument("part")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("slice", help="build してからスライスする (.gcode.3mf を作る)")
    p.add_argument("part")
    p.add_argument("--process", help="例: '0.16mm Optimal'")
    p.add_argument("--filament", help="例: 'Bambu PETG HF'")
    p.set_defaults(fn=cmd_slice)

    p = sub.add_parser("profiles", help="この機種で使えるプリセット名の一覧")
    p.add_argument("kind", choices=["process", "filament"])
    p.add_argument("query", nargs="?")
    p.set_defaults(fn=cmd_profiles)

    p = sub.add_parser("status", help="プリンタの状態を見る")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("print", help="スライス済みファイルを送って印刷を始める")
    p.add_argument("part")
    p.add_argument("--no-leveling", action="store_true", help="ベッドレベリングを省く")
    p.add_argument("--ams-slot", type=int, help="使う AMS スロット (1 始まり)。省略時は最初に入っているもの")
    p.set_defaults(fn=cmd_print)

    for name, help_ in (("pause", "一時停止"), ("resume", "再開"), ("stop", "中止")):
        p = sub.add_parser(name, help=f"印刷を{help_}")
        p.set_defaults(fn=cmd_control)

    p = sub.add_parser("clean", help="生成物 (parts/*/out) を消す")
    p.set_defaults(fn=cmd_clean)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
