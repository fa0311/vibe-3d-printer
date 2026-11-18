"""Bambu Studio のシステムプリセットを「完全展開」する。

Bambu Studio の CLI は `inherits` / `include` を解決しない。継承元を持つ JSON を
そのまま渡すと、足りないキーが汎用デフォルト値 (開始Gコードが `G28` だけ、ノズル 200℃、
ベッド 35℃ …) で埋まり、A1 mini では使えないGコードが黙って出力される。
そのため GUI と同じ順序 (継承元 → include → 自分自身) でマージした JSON を
`<resources>/profiles/BBL/{machine,process,filament}_full/` に書き出し、CLI にはそれを渡す。
このディレクトリ配置は CLI 自身が `printer_model_id` を引くのに参照する場所でもある。
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from .paths import bambu_studio_home

# 展開後に持ち越さないキー (プリセットの系譜を表すだけで、設定値ではない)
_LINEAGE_KEYS = {"inherits", "include", "instantiation", "description", "renamed_from"}
# include テンプレートから取り込まないキー
_META_KEYS = _LINEAGE_KEYS | {"type", "name", "from", "setting_id", "filament_id"}

_KINDS = {
    "machine": "machine_list",
    "process": "process_list",
    "filament": "filament_list",
}


def profiles_dir() -> Path:
    return bambu_studio_home() / "resources" / "profiles" / "BBL"


def full_dir(kind: str) -> Path:
    return profiles_dir() / f"{kind}_full"


def _file_name(name: str) -> str:
    # "Bambu Support For PA/PET" のように "/" を含むプリセット名がある
    return name.replace("/", "_") + ".json"


@cache
def _vendor() -> dict:
    return json.loads((profiles_dir().parent / "BBL.json").read_text(encoding="utf-8"))


@cache
def _index(kind: str) -> dict[str, Path]:
    return {e["name"]: profiles_dir() / e["sub_path"] for e in _vendor()[_KINDS[kind]]}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(kind: str, name: str, stack: tuple[str, ...] = ()) -> dict:
    if name in stack:
        raise ValueError(f"プリセットの継承が循環しています: {' -> '.join(stack + (name,))}")
    index = _index(kind)
    if name not in index:
        raise KeyError(f"{kind} プリセット '{name}' が見つかりません")
    src = _load(index[name])

    merged: dict = {}
    parent = src.get("inherits")
    if parent:
        merged.update(_resolve(kind, parent, stack + (name,)))

    includes = src.get("include") or []
    if isinstance(includes, str):
        includes = [includes]
    for inc in includes:
        if inc not in index:
            raise KeyError(f"{kind} プリセット '{name}' の include '{inc}' が見つかりません")
        merged.update({k: v for k, v in _load(index[inc]).items() if k not in _META_KEYS})

    merged.update(src)
    return merged


def flatten(kind: str, name: str) -> dict:
    """継承を解決した完全なプリセットを返す。"""
    full = {k: v for k, v in _resolve(kind, name).items() if k not in _LINEAGE_KEYS}
    full["type"] = kind
    full["name"] = name
    full["from"] = "system"
    return full


def names(kind: str, *, instantiable_only: bool = True) -> list[str]:
    out = []
    for name, path in _index(kind).items():
        if instantiable_only and _load(path).get("instantiation") != "true":
            continue
        out.append(name)
    return sorted(out)


def write_all() -> dict[str, int]:
    """全プリセットを展開して *_full/ に書き出す。セットアップ時に一度実行する。"""
    counts = {}
    for kind in _KINDS:
        out = full_dir(kind)
        out.mkdir(exist_ok=True)
        n = 0
        for name in names(kind):
            (out / _file_name(name)).write_text(
                json.dumps(flatten(kind, name), ensure_ascii=False, indent=1), encoding="utf-8"
            )
            n += 1
        counts[kind] = n
    # 機種定義 (model_id を持つ)。CLI が machine_full/<printer_model>.json を読みに来る。
    for e in _vendor()["machine_model_list"]:
        src = profiles_dir() / e["sub_path"]
        (full_dir("machine") / _file_name(e["name"])).write_bytes(src.read_bytes())
    return counts


def full_path(kind: str, name: str) -> Path:
    path = full_dir(kind) / _file_name(name)
    if not path.exists():
        raise FileNotFoundError(
            f"展開済みプリセットがありません: {path}\n`scripts/setup.sh` を実行してください。"
        )
    return path
