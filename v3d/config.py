"""printer.toml (機種・既定の設定) と printer.local.toml (接続情報・秘密) を読む。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache

from .paths import LOCAL_CONFIG, REPO


@dataclass(frozen=True)
class PrinterSpec:
    machine: str          # Bambu Studio の機種プリセット名
    profile_suffix: str   # プロセス/フィラメントのプリセット名の末尾 (例: "@BBL A1M")
    plate: str            # ビルドプレートの種類
    process: str          # 既定のプロセス
    filament: str         # 既定のフィラメント


@dataclass(frozen=True)
class Connection:
    ip: str
    serial: str
    access_code: str


@cache
def spec() -> PrinterSpec:
    data = tomllib.loads((REPO / "printer.toml").read_text(encoding="utf-8"))
    p, d = data["printer"], data["defaults"]
    return PrinterSpec(
        machine=p["machine"],
        profile_suffix=p["profile_suffix"],
        plate=p["plate"],
        process=d["process"],
        filament=d["filament"],
    )


def connection() -> Connection:
    if not LOCAL_CONFIG.exists():
        raise FileNotFoundError(
            f"{LOCAL_CONFIG.name} がありません。printer.local.example.toml をコピーして、"
            "プリンタの IP・シリアル番号・アクセスコードを書いてください。"
        )
    c = tomllib.loads(LOCAL_CONFIG.read_text(encoding="utf-8")).get("connection", {})
    missing = [k for k in ("ip", "serial", "access_code") if not str(c.get(k, "")).strip()]
    if missing:
        raise ValueError(f"{LOCAL_CONFIG.name} の [connection] に未記入の項目があります: {missing}")
    return Connection(ip=str(c["ip"]), serial=str(c["serial"]), access_code=str(c["access_code"]))
