"""nfc-test: 印刷の途中で NFC のチップを埋め込めるかを確かめる、丸い札 (そのままキーホルダーにもなる)。

板の中にチップが入る空洞を作っておき、空洞の天井を刷る直前でプリンタを一時停止させる (PRINT の pause_at)。
止まっている間に持ち主がチップを置いて再開すると、その上に天井が刷られて、チップが中に閉じ込められる。
チップの裏には貼り付け用のテープが付いているので、空洞の底に貼る (刷っている間に動かず、振っても音がしない)。

確かめたいこと: 止まるか / チップがすっと入るか / ノズルがチップに当たらないか / 閉じたあとも読めるか /
振ってもカタカタ言わないか。
"""

from build123d import *

from v3d import fit

# ---- チップ (持ち主の実測。README.md) ----
CHIP_D = 25.0         # 直径
CHIP_T = 0.9          # 厚み (裏のテープのはくり紙を含む。はがすと少し薄くなる)

# ---- 札 (mm) ----
DIAMETER = 32.0       # 札の直径 (空洞のまわりに 3 mm 強の壁が残る)
BOTTOM = 0.6          # 空洞の下の板 (3 層)
ROOF = 0.8            # 空洞の上の板 (4 層。1 層目は宙に渡して刷るので、表面まで 4 層とる)
# 空洞の深さ: チップより少し深くして、チップが縁から飛び出さないようにする (飛び出すとノズルが当たる)。
# 層 (0.2 mm) の境目でしか止められないので、チップの厚み以上で、いちばん近い境目にする。
# はくり紙をはがしたチップは 0.8 mm 前後のはずなので、上に 0.1〜0.2 mm のすき間が残る。
# 天井の 1 層目は宙に渡して刷られ、少し垂れてチップに乗る
POCKET_DEPTH = 1.0
# 空洞の直径: 熱いベッドの上で指でぽとりと置けるように、ゆるめにする
POCKET_D = fit.hole(CHIP_D) + 2 * fit.CLEAR_LOOSE
EDGE = 0.4            # 外周の面取り

RING_OUTER = 8.5      # 金具を通す輪の外径 (ほかのキーホルダーと同じ)
RING_HOLE = 3.5       # 輪の穴の仕上がり径
RING_OVERLAP = 2.0    # 輪を札にどれだけ食い込ませるか

THICK = BOTTOM + POCKET_DEPTH + ROOF

assert POCKET_DEPTH >= CHIP_T, "空洞がチップより浅い (ノズルがチップに当たる)"
assert abs(POCKET_DEPTH / fit.LAYER - round(POCKET_DEPTH / fit.LAYER)) < 1e-6, "空洞の深さは層の倍数にする"

PRINT = {
    # 空洞の天井を刷る直前 (チップが入る高さまで刷り終えたところ) で一時停止する
    "pause_at": [BOTTOM + POCKET_DEPTH],
}


def build():
    """寝かせて刷る (どちらの面も平ら)。"""
    ring_y = DIAMETER / 2 + RING_OUTER / 2 - RING_OVERLAP
    outline = Circle(DIAMETER / 2) + Pos(0, ring_y) * Circle(RING_OUTER / 2)
    # 外周の面取りは、層の厚みごとに輪郭を 1 段ずつ内へ寄せた板を重ねて作る (刷れば面取りと同じ形になる。
    # 輪と札のつなぎ目はへこんだ角なので、chamfer() では作れなかった)
    steps = round(EDGE / fit.LAYER)
    body = Pos(0, 0, EDGE) * extrude(outline, amount=THICK - 2 * EDGE)
    for i in range(steps):
        slab = extrude(offset(outline, -(steps - i) * fit.LAYER), amount=fit.LAYER)
        body += Pos(0, 0, i * fit.LAYER) * slab                            # 下の面取り
        body += Pos(0, 0, THICK - (i + 1) * fit.LAYER) * slab              # 上の面取り
    pocket = Pos(0, 0, BOTTOM) * Cylinder(POCKET_D / 2, POCKET_DEPTH, align=(Align.CENTER, Align.CENTER, Align.MIN))
    hole = Pos(0, ring_y) * Cylinder(fit.hole(RING_HOLE) / 2, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return body - pocket - hole
