from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARTS = REPO / "parts"
LOCAL_CONFIG = REPO / "printer.local.toml"


def bambu_studio_home() -> Path:
    """展開済み Bambu Studio (AppImage の中身) の場所。"""
    home = Path(
        os.environ.get("V3D_BAMBU_STUDIO", "~/.local/opt/bambustudio/squashfs-root")
    ).expanduser()
    if not (home / "AppRun").exists():
        raise FileNotFoundError(
            f"Bambu Studio が見つかりません: {home}\n`scripts/setup.sh` を実行してください。"
        )
    return home


def part_dir(name: str) -> Path:
    d = PARTS / name
    if not (d / "part.py").exists():
        known = sorted(p.parent.name for p in PARTS.glob("*/part.py"))
        raise FileNotFoundError(f"parts/{name}/part.py がありません。既存のパーツ: {known}")
    return d
