"""fit-test: この機体の「はめあい」と「穴径」の実力を測るゲージ。

板に、同じ 10 mm 角のペグ用の角穴を すき間違いで 5 個、丸穴を 5 個あける。
印刷後に「どの角穴にどう入るか」と「丸穴・ペグの実測値」を聞き、v3d/fit.py を更新する。
ここは基準なので fit.py の補正値は使わず、呼び寸法そのままで作る。
"""

from build123d import *

# ---- 寸法 (mm) ----
PEG = 10.0                                   # ペグの一辺 (呼び寸法)
PEG_LENGTH = 25.0
CLEARANCES = [0.05, 0.10, 0.15, 0.20, 0.30]  # 角穴の片側すき間。刻印は 1/100 mm 単位
HOLES = [3.0, 4.0, 5.0, 6.0, 8.0]            # 丸穴の直径 (呼び寸法)

PLATE = (80.0, 40.0, 5.0)
PITCH = 15.0
ROW_SOCKET, ROW_SOCKET_LABEL, ROW_HOLE, ROW_HOLE_LABEL = 12.0, 2.5, -7.5, -16.0
TEXT_SIZE = 4.5
TEXT_DEPTH = 0.4
LEAD_IN = 0.5                                # 入口の面取り。1層目の広がり(エレファントフット)も逃がす


def _xs(n: int) -> list[float]:
    return [(i - (n - 1) / 2) * PITCH for i in range(n)]


def _plate():
    length, width, thick = PLATE
    with BuildPart() as p:
        Box(length, width, thick, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), 3)

        for x, c in zip(_xs(len(CLEARANCES)), CLEARANCES):
            with Locations((x, ROW_SOCKET, 0)):
                Box(PEG + 2 * c, PEG + 2 * c, thick, align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT)
        for x, d in zip(_xs(len(HOLES)), HOLES):
            with Locations((x, ROW_HOLE, 0)):
                Cylinder(d / 2, thick, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)

        # 穴の上下の縁を面取り (上面・下面にある内側の輪郭だけを選ぶ)
        rims = [
            e for face in p.faces().filter_by(Axis.Z) for wire in face.inner_wires() for e in wire.edges()
        ]
        chamfer(rims, LEAD_IN)

        labels = [(f"{round(c * 100):02d}", x, ROW_SOCKET_LABEL) for x, c in zip(_xs(len(CLEARANCES)), CLEARANCES)]
        labels += [(f"{d:g}", x, ROW_HOLE_LABEL) for x, d in zip(_xs(len(HOLES)), HOLES)]
        with BuildSketch(Plane.XY.offset(thick)):
            for text, x, y in labels:
                with Locations((x, y)):
                    Text(text, font_size=TEXT_SIZE, font="DejaVu Sans", font_style=FontStyle.BOLD)
        extrude(amount=-TEXT_DEPTH, mode=Mode.SUBTRACT)
    return p.part


def _peg():
    with BuildPart() as p:
        Box(PEG, PEG, PEG_LENGTH, align=(Align.CENTER, Align.CENTER, Align.MIN))
        chamfer(p.edges().group_by(Axis.Z)[0] + p.edges().group_by(Axis.Z)[-1], 0.8)
    return p.part


def build():
    """板は寝かせて、ペグは立てて印刷する (どちらも はめあい面が XY 方向に出る向き)。"""
    return {"plate": _plate(), "peg": _peg()}
