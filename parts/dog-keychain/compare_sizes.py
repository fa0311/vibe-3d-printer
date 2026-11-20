"""犬耳のキーホルダーの大きさ違いを、同じ縮尺の「刷り上がりの絵」で並べる。

使い方 (リポジトリのルートで): PYTHONPATH=. uv run python parts/dog-keychain/compare_sizes.py [出力 PNG]
既定の出力は parts/dog-keychain/out/sizes.png。1 個あたりの時間は、2026-11-21 にスライスして測った値を書いてある。
"""
import importlib.util
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from v3d import fit, trace

spec = importlib.util.spec_from_file_location("dog", "parts/dog-keychain/part.py")
dog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dog)

MMPX = 0.06
VARIANTS = [   # 時間はスライスで測った 1 個あたりの値 (チップ入り、準備の時間込み)
    ("今の大きさ", "直径 53 mm\n0.4 mm ノズル・1 個 30 分", dog.Size()),
    ("A", "直径 40 mm\n今のノズル・1 個 21 分", dog.Size.with_plate(40, 0.4)),
    ("B", "直径 30 mm\n今のノズル・1 個 14 分", dog.Size.with_plate(30, 0.4)),
    ("C", "直径 30 mm\n0.2 mm ノズル・1 個 36 分", dog.Size.with_plate(30, 0.2)),
]
color = tuple(int(dog.PREVIEW_COLOR[i:i + 2], 16) for i in (5, 3, 1))     # BGR


def picture(size):
    L = dog._layout(size)
    h = np.zeros(L["plate"].shape, np.float32)
    h[L["plate"] > 0] = dog.FLOOR
    h[L["dimple"] > 0] = dog.THICK - dog.CHEEK_DEPTH
    h[L["top"] > 0] = dog.THICK
    cx, cy = L["center"]
    rx, ry = L["ring"]
    cv2.circle(h, (int(round(cx + rx / L["mm"])), int(round(cy - ry / L["mm"]))),
               int(round(fit.hole(dog.RING_HOLE) / 2 / L["mm"])), 0.0, cv2.FILLED)
    f = L["mm"] / MMPX
    h = cv2.resize(h, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    image = trace.shaded(h, MMPX, color_bgr=color)
    ys, xs = np.nonzero(h > 0)
    m = int(2 / MMPX)
    return image[ys.min() - m: ys.max() + m, xs.min() - m: xs.max() + m]


font_path = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans CJK JP:bold"], capture_output=True, text=True).stdout
font = lambda s: ImageFont.truetype(font_path, s)  # noqa: E731
pics = [picture(size) for _, _, size in VARIANTS]
gap, top, bottom = int(6 / MMPX), 230, 90
W = sum(p.shape[1] for p in pics) + gap * (len(pics) + 1)
H = max(p.shape[0] for p in pics) + top + bottom
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
x = gap
for (name, note, size), p in zip(VARIANTS, pics):
    y = top + (H - top - bottom - p.shape[0])
    canvas.paste(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)), (x, y))
    cx = x + p.shape[1] // 2
    draw.text((cx, 16), name, font=font(64), fill="black", anchor="ma")
    draw.multiline_text((cx, 100), note, font=font(44), fill=(60, 60, 60), anchor="ma", align="center", spacing=10)
    print(f"{name}: 板 {size.plate:.1f} mm, 絵 {size.art:.1f} mm, 溝の幅 {dog.MIN_LINE * size.stroke:.2f} mm, 縁 {size.rim:.2f} mm")
    x += p.shape[1] + gap
bar = int(10 / MMPX)
draw.rectangle([gap, H - 50, gap + bar, H - 40], fill="black")
draw.text((gap + bar + 16, H - 66), "10 mm (同じ縮尺で並べてある)", font=font(36), fill="black")
canvas.save(sys.argv[1] if len(sys.argv) > 1 else "parts/dog-keychain/out/sizes.png")
