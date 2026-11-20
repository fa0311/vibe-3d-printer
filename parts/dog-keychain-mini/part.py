"""dog-keychain-mini: 犬耳の子のキーホルダーの小さい版 (板の直径 30 mm、NFC チップ入り)。

絵の描き方とチップの入れ方は parts/dog-keychain と同じ。そのファイルの make() を、小さい Size で呼ぶ。
今のノズル (0.4 mm) のまま 30 mm にすると、線を 1.0 mm より細くできず瞳がつぶれる。そこで 0.2 mm ノズルで刷り、
線も絵と一緒に細くする (溝の幅 0.55 mm)。持ち主が 3 つの候補からこれを選んだ (README.md)。
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("dog_keychain", Path(__file__).parents[1] / "dog-keychain" / "part.py")
dog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dog)

SIZE = dog.Size.with_plate(30.0, nozzle=0.2)       # チップ (直径 25 mm) が入る下限に近い大きさ
PREVIEW_COLOR = dog.PREVIEW_COLOR

PRINT = {
    "nozzle": SIZE.nozzle,                          # 0.4 mm ノズルの設定ではスライスさせない (細い線が刷れない)
    "process": "0.10mm Standard",                   # 0.2 mm ノズルの標準 (層 0.1 mm)
    "pause_at": [dog.CHIP_BOTTOM + dog.CHIP_POCKET],    # チップを置くための一時停止 (高さ 1.4 mm)
}


def build():
    """絵の面を上にして寝かせて刷る (裏は平ら)。"""
    return dog.make(SIZE)
