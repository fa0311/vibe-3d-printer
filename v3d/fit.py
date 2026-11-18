"""はめあい・穴径の補正値。パーツはここの値を使い、数値を直書きしない。

FDM は「穴は小さく、外形はわずかに大きく」出る。下の値は A1 mini + PLA の一般的な出発点で、
`parts/fit-test` を印刷して実測したら CALIBRATED を True にして値を更新する
(手順と記録は parts/fit-test/README.md)。
"""

from __future__ import annotations

CALIBRATED = False

# --- 穴 ---
HOLE_COMP = 0.20      # 丸穴の直径に足す量。垂直(Z軸方向)の穴向け

# --- 2部品のすき間 (片側あたり) ---
CLEAR_PRESS = 0.05    # 圧入: 叩き込む / 外れない
CLEAR_SNUG = 0.10     # しっくり: 手で押し込める、ガタなし
CLEAR_SLIDE = 0.20    # すべり: 抜き差しする蓋・引き出し
CLEAR_LOOSE = 0.35    # ゆるい: 確実に入ればよい

# --- 印刷の都合 ---
NOZZLE = 0.4
LAYER = 0.2
MIN_WALL = 1.2        # 強度が要る壁の最小厚 (ノズル幅の3倍)
BED = (180.0, 180.0, 180.0)


def hole(d: float) -> float:
    """実寸 d の軸・ねじが通る丸穴のモデル上の直径。"""
    return d + HOLE_COMP


# メートルねじ: (通し穴, 直接ねじ込む下穴, キャップボルト頭径, 頭高さ, ナット二面幅, ナット厚)
_SCREWS = {
    "M2":   (2.4, 1.7, 3.8, 2.0, 4.0, 1.6),
    "M2.5": (2.9, 2.1, 4.5, 2.5, 5.0, 2.0),
    "M3":   (3.4, 2.6, 5.5, 3.0, 5.5, 2.4),
    "M4":   (4.5, 3.5, 7.0, 4.0, 7.0, 3.2),
    "M5":   (5.5, 4.4, 8.5, 5.0, 8.0, 4.7),
    "M6":   (6.6, 5.2, 10.0, 6.0, 10.0, 5.2),
}


def screw(size: str) -> dict[str, float]:
    """ねじ周りのモデル寸法 (補正込み)。size は "M3" など。"""
    through, tap, head_d, head_h, nut_af, nut_h = _SCREWS[size]
    return {
        "through": through + HOLE_COMP,          # 通し穴の直径
        "tap": tap + HOLE_COMP,                  # 樹脂に直接ねじ込む下穴の直径
        "head_d": head_d + 2 * CLEAR_LOOSE,      # 頭を沈めるザグリの直径
        "head_h": head_h + 0.2,                  # ザグリの深さ
        "nut_af": nut_af + 2 * CLEAR_SNUG,       # ナットポケットの二面幅
        "nut_h": nut_h + 0.3,                    # ナットポケットの深さ
    }
