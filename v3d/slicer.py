"""Bambu Studio の CLI でスライスして、プリンタに送れる .gcode.3mf を作る。

part.py の PRINT で指定できるもの (すべて省略可):
    PRINT = {
        "process": "0.20mm Standard",        # 積層ピッチ等のプリセット (v3d profiles process で一覧)
        "filament": "Bambu PLA Basic",       # フィラメントのプリセット (v3d profiles filament で一覧)
        "settings": {"enable_support": 1},   # プロセス設定の上書き
        "filament_settings": {...},          # フィラメント設定の上書き
        "copies": {"ピース名": 2},            # 同じピースを複数個並べる
        "pause_at": [1.0],                   # この高さ (mm) まで刷り終えたら一時停止する (物を埋め込むとき)
        "nozzle": 0.2,                       # この部品を刷ってよいノズルの太さの上限 (細い線を 0.2 mm 用に描いた物)
    }

一時停止 (pause_at) の仕組み: 機種の「層が変わるたびに実行する G コード」に、決まった層でだけ一時停止の命令
(機種プリセットの machine_pause_gcode。A1 mini は M400 U1) を出す条件文を足している。Bambu Studio の画面で
「Add Pause」をしたときと同じ命令が、同じ場所 (その層を刷り始める直前) に入る。本体の画面か Bambu Handy で
再開するまで待つ。

その場所では、ノズルが刷ったばかりの面の高さにいる。止まっている間に頭 (ノズル) がその上に居座ると、物を置けず、
熱いノズルが部品に触れたままになる。ファームウェアが自分でよけるかは確かめていないので、止める前にこちらで
ノズルを持ち上げ、機種の終了 G コードと同じ「よける位置」(造形範囲の左の外、ベッドは手前) へ動かす。
再開すると、スライサーの「次の層へ移る動き」で刷る位置の上へ戻り、刷る高さまで下りてから刷り始める
(そうなっていることも、スライス結果で確かめる)。
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config, profiles, render
from .model import Built
from .paths import bambu_studio_home


class SliceError(RuntimeError):
    pass


@dataclass
class Sliced:
    file: Path
    seconds: float
    grams: float
    meters: float
    layers: int
    machine: str
    process: str
    filament: str
    plate: str
    settings: dict
    filament_settings: dict
    copies: dict
    warnings: list[str]
    pauses: list[dict]          # 一時停止: {"z": 止まる高さ, "layer": 次に刷る層 (1 始まり), "minute": 開始からの目安}

    @property
    def duration(self) -> str:
        m = round(self.seconds / 60)
        return f"{m // 60}時間{m % 60:02d}分" if m >= 60 else f"{m}分"


def resolve_profile(kind: str, name: str) -> str:
    """"0.20mm Standard" のような短い名前を、この機種用の正式なプリセット名にする。"""
    known = profiles.names(kind)
    suffix = config.spec().profile_suffix
    for candidate in (name, f"{name} {suffix}"):
        if candidate in known:
            return candidate
    ours = [n for n in known if n.endswith(suffix)]
    near = difflib.get_close_matches(f"{name} {suffix}", ours, n=5, cutoff=0.4)
    raise SliceError(f"{kind} プリセット '{name}' がありません。近いもの: {near}")


def _with_overrides(base: dict, overrides: dict, per_filament: bool) -> dict:
    """プリセットに上書きを適用する。

    プリセットは既定値と同じ設定を省略しているので、base に無いキーも正当でありうる。
    綴りの誤りはスライス後の _verify (Gコードに反映されたかの確認) で捕まえる。
    """
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, bool):
            value = int(value)
        current = base.get(key)
        if isinstance(value, (list, tuple)):
            out[key] = [str(v) for v in value]
        elif isinstance(current, list):
            out[key] = [str(value)] * max(len(current), 1)
        elif current is None and per_filament:
            out[key] = [str(value)]
        else:
            out[key] = str(value)
    return out


def _gcode_config(gcode: str) -> dict[str, str]:
    block = gcode.split("; CONFIG_BLOCK_START", 1)[1].split("; CONFIG_BLOCK_END", 1)[0]
    return dict(m.groups() for m in re.finditer(r"^; (\S+) = (.*)$", block, re.M))


def _same(expected, actual: str) -> bool:
    """設定値の比較。Gコード側は配列をカンマ区切り、文字列を引用符付きで書く。"""
    if isinstance(expected, list):
        expected = ",".join(expected)

    def norm(s: str) -> str:
        return str(s).strip().strip('"').replace("\\n", "\n").strip()

    a, b = norm(expected), norm(actual)
    if a == b:
        return True
    try:
        return [float(x.rstrip("%")) for x in a.split(",")] == [float(x.rstrip("%")) for x in b.split(",")]
    except ValueError:
        return False


def _verify(gcode: str, machine: dict, process: dict, filament: dict, plate: str,
            settings: dict, filament_settings: dict) -> None:
    """意図したプリセットが本当にGコードに反映されたかを確かめる。

    CLI は読めなかった設定を黙って汎用デフォルトに置き換えるので、成功終了だけでは信用できない。
    """
    cfg = _gcode_config(gcode)
    expect = {
        "printer_model": machine["printer_model"],
        "printer_settings_id": machine["name"],
        "print_settings_id": process["name"],
        "filament_settings_id": filament["name"],
        "curr_bed_type": plate,
        "nozzle_temperature": filament["nozzle_temperature"],
        "layer_height": process["layer_height"],
    }
    expect |= {k: process[k] for k in settings}
    expect |= {k: filament[k] for k in filament_settings}
    wrong = []
    for k, v in expect.items():
        if k not in cfg:
            near = difflib.get_close_matches(k, cfg.keys(), n=4)
            wrong.append(f"{k}: そのような設定はありません。近いもの: {near}")
        elif not _same(v, cfg[k]):
            wrong.append(f"{k}: 期待 {v!r} / 実際 {cfg[k]!r}")
    start = next(ln for ln in machine["machine_start_gcode"].splitlines() if ln.strip())
    if start not in gcode.split("; EXECUTABLE_BLOCK_START", 1)[-1]:
        wrong.append(f"機種の開始Gコード ({start.strip()}) が出力に含まれていません")
    if wrong:
        raise SliceError("スライス結果が指定した設定と一致しません:\n  " + "\n  ".join(wrong))


PAUSE_MARK = "; v3d: pause here"
# 止まっている間に頭をよける位置。A1 mini の終了 G コードが、刷り終えたあとに頭をよける動きと同じ:
# 左端 (X=0) を経て造形範囲の左の外 (X=-13) へ。ベッドは手前 (Y=180) へ出す。先にノズルを持ち上げる
PARK_MODEL = "Bambu Lab A1 mini"
PARK_LIFT = 20.0      # 持ち上げる高さ (mm)
PARK_MOVES = ("G1 X0 F18000", "G1 X-13.0 F3000", "G1 Y180 F3600")


def _pause_layers(heights: list[float], process: dict) -> dict[int, float]:
    """止まる高さ → その直後に刷る層の番号 (0 始まり。layer_change_gcode の layer_num と同じ数え方)。"""
    first, step = float(process["initial_layer_print_height"]), float(process["layer_height"])
    out = {}
    for z in heights:
        done = round((z - first) / step) + 1          # ここまでに刷り終わっている層の数
        if done < 1 or abs(first + (done - 1) * step - z) > 0.005:
            below = first + max(int((z - first) // step), 0) * step
            raise SliceError(f"pause_at {z} mm は層の境目ではありません。近い境目: {below:.2f} / {below + step:.2f} mm "
                             f"(1 層目 {first} mm、以降 {step} mm ごと)")
        out[done] = float(z)
    return out


def _pause_block(machine: dict, z: float) -> list[str]:
    """高さ z まで刷ったところで止める命令: 持ち上げる → よける → 止まる。"""
    lift = f"G1 Z{min(z + PARK_LIFT, float(machine['printable_height'])):.2f} F600"
    return [PAUSE_MARK, lift, *PARK_MOVES, machine["machine_pause_gcode"].strip()]


def _with_pauses(machine: dict, layers: dict[int, float]) -> dict:
    if not layers:
        return machine
    if machine["printer_model"] != PARK_MODEL:
        raise SliceError(f"pause_at の頭のよけ方は {PARK_MODEL} 用です (この機種: {machine['printer_model']})")
    blocks = ["\n".join([f"{{if layer_num == {n}}}", *_pause_block(machine, z), "{endif}"]) for n, z in sorted(layers.items())]
    return machine | {"layer_change_gcode": machine["layer_change_gcode"].rstrip("\n") + "\n" + "\n".join(blocks)}


def _height_when_printing_resumes(lines: list[str]) -> float | None:
    """再開後、最初に (横へ動きながら) 押し出す時点で、ノズルに指示されている高さ。"""
    z = None
    for line in lines:
        if not line.startswith(("G0 ", "G1 ", "G2 ", "G3 ")):
            continue
        if m := re.search(r" Z(-?[\d.]+)", line):
            z = float(m.group(1))
        e = re.search(r" E(-?[\d.]+)", line)
        if e and float(e.group(1)) > 0 and re.search(r" [XY]-?[\d.]", line):
            return z
    return None


def _verify_pauses(gcode: str, layers: dict[int, float], machine: dict, total_minutes: float) -> list[dict]:
    """一時停止が、頼んだ高さを刷り終えた直後に、頼んだ数だけ入っているかを G コードで確かめる。

    止まる前に頭をよけること、再開後は次の層の高さまで下りてから刷り始めることも確かめる。
    """
    pause = machine["machine_pause_gcode"].strip()
    found, top, height, layer, remaining = [], 0.0, 0.0, 0, None
    lines = gcode.split("; EXECUTABLE_BLOCK_START", 1)[-1].splitlines()
    for i, line in enumerate(lines):
        if line.startswith("; Z_HEIGHT: "):
            top, layer = float(line[12:]), layer + 1
        elif line.startswith("; LAYER_HEIGHT: "):
            height = float(line[16:])
        elif line.startswith("M73 P") and (m := re.search(r" R(\d+)", line)):
            remaining = int(m.group(1))
        elif line.strip() == PAUSE_MARK:
            block = _pause_block(machine, top - height)
            written = [ln.strip() for ln in lines[i: i + len(block)]]
            if written != block:
                raise SliceError(f"一時停止の命令が意図と違います:\n  意図 {block}\n  実際 {written}")
            resume = _height_when_printing_resumes(lines[i + len(block):])
            if resume is None or abs(resume - top) > 1e-3:
                raise SliceError(f"再開後、刷る高さ ({top} mm) に下りないまま刷り始めます (指示されている高さ: {resume})")
            minute = None if remaining is None else max(round(total_minutes - remaining), 0)
            found.append({"z": round(top - height, 3), "layer": layer, "minute": minute})
    want = sorted(layers.values())
    if [f["z"] for f in found] != [round(z, 3) for z in want]:
        raise SliceError(f"一時停止の位置が指定と違います: 指定 {want} mm / 実際 {[f['z'] for f in found]} mm")
    stray = sum(1 for ln in lines if ln.strip() == pause) - len(found)
    if stray:
        raise SliceError(f"指定していない一時停止 ({pause}) が {stray} 個入っています")
    return found


def slice_part(built: Built, *, process: str | None = None, filament: str | None = None) -> Sliced:
    if not built.ok:
        raise SliceError("モデルに問題があるためスライスしません (v3d build の出力を参照)")

    spec = config.spec()
    ps = built.print_settings
    unknown = set(ps) - {"process", "filament", "settings", "filament_settings", "copies", "pause_at", "nozzle"}
    if unknown:
        raise SliceError(f"PRINT に未知のキーがあります: {sorted(unknown)}")
    settings = dict(ps.get("settings", {}))
    filament_settings = dict(ps.get("filament_settings", {}))
    copies = dict(ps.get("copies", {}))
    piece_names = [p.name for p in built.pieces]
    if set(copies) - set(piece_names):
        raise SliceError(f"copies に存在しないピース名があります: {sorted(set(copies) - set(piece_names))}")

    machine = profiles.flatten("machine", spec.machine)
    nozzle = float(machine["nozzle_diameter"][0])
    if "nozzle" in ps and nozzle > float(ps["nozzle"]) + 1e-6:
        raise SliceError(f"この部品は {ps['nozzle']} mm 以下のノズル用です (細い線が刷れない)。printer.toml のノズルは "
                         f"{nozzle} mm。持ち主がノズルを替えたか確かめて、printer.toml を合わせること (CLAUDE.md の「ノズル」)")
    process_cfg = _with_overrides(
        profiles.flatten("process", resolve_profile("process", process or ps.get("process", spec.process))),
        settings, per_filament=False,
    )
    filament_cfg = _with_overrides(
        profiles.flatten("filament", resolve_profile("filament", filament or ps.get("filament", spec.filament))),
        filament_settings, per_filament=True,
    )
    pause_layers = _pause_layers([float(z) for z in ps.get("pause_at", [])], process_cfg)
    machine = _with_pauses(machine, pause_layers)
    for cfg in (process_cfg, filament_cfg):
        compatible = cfg.get("compatible_printers") or []
        if compatible and spec.machine not in compatible:
            raise SliceError(f"プリセット '{cfg['name']}' は {spec.machine} に対応していません")

    target = f"{built.name}.gcode.3mf"
    # 失敗したときに前回の (古い形の) 印刷データが残って、それを印刷してしまわないように先に消す
    (built.out / target).unlink(missing_ok=True)
    (built.out / "slice.json").unlink(missing_ok=True)

    work = built.out / "slice"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    paths = {}
    for kind, cfg in (("machine", machine), ("process", process_cfg), ("filament", filament_cfg)):
        paths[kind] = work / f"{kind}.json"
        paths[kind].write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    cmd = [
        str(bambu_studio_home() / "AppRun"),
        "--debug", "2",
        "--orient", "0",       # 向きは part.py で決める
        "--arrange", "1",
        "--curr-bed-type", spec.plate,
        "--load-settings", f"{paths['machine']};{paths['process']}",
        "--load-filaments", str(paths["filament"]),
        "--slice", "0",
        # 形状を含めない「スライス済みファイル」として出す。こうしないと Bambu Studio の画面が
        # 編集用プロジェクトとして開き、こちらで確認済みのGコードではなく再スライスを求めてくる。
        "--min-save",
        "--outputdir", str(work),
        "--export-3mf", target,
    ]
    if copies:
        cmd += ["--clone-objects", ",".join(str(int(copies.get(n, 1))) for n in piece_names)]
    cmd += [str(p.stl) for p in built.pieces]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    log = proc.stdout + proc.stderr
    (work / "slicer.log").write_text(log, encoding="utf-8")
    result_file = work / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    if proc.returncode != 0 or result.get("return_code") != 0 or not (work / target).exists():
        reason = result.get("error_string") or "原因不明"
        tail = "\n".join(ln for ln in log.splitlines() if "[error]" in ln)[-1500:]
        raise SliceError(f"スライス失敗 (終了コード {proc.returncode}): {reason}\n{tail}")

    plates = result["sliced_plates"]
    if len(plates) != 1:
        raise SliceError(f"{len(plates)} プレートに分かれました。1プレートに収まるよう分割すること")
    plate = plates[0]

    with zipfile.ZipFile(work / target) as z:
        gcode = z.read("Metadata/plate_1.gcode").decode("utf-8", "replace")
    _verify(gcode, machine, process_cfg, filament_cfg, spec.plate, settings, filament_settings)
    pauses = _verify_pauses(gcode, pause_layers, machine, float(plate["total_predication"]) / 60)

    header = gcode.split("; HEADER_BLOCK_END", 1)[0]
    meters = float(re.search(r"total filament length \[mm\] : ([\d.,]+)", header).group(1).split(",")[0]) / 1000
    layers = int(re.search(r"total layer number: (\d+)", header).group(1))

    # CLI はこの環境ではサムネイルを描けない (参照だけが 3MF に残る) ので、こちらで描いて埋める
    stls = [p.stl for p in built.pieces]
    with zipfile.ZipFile(work / target, "a", zipfile.ZIP_DEFLATED) as z:
        have = set(z.namelist())
        for name, px in (("Metadata/plate_1.png", 512), ("Metadata/plate_1_small.png", 128)):
            if name not in have:
                z.write(render.thumbnail(stls, work / Path(name).name, px), name)

    final = built.out / target
    shutil.move(work / target, final)
    warnings = [w for w in [plate.get("warning_message", "")] if w]
    sliced = Sliced(
        file=final,
        seconds=float(plate["total_predication"]),
        grams=sum(f["total_used_g"] for f in plate["filaments"]),
        meters=meters,
        layers=layers,
        machine=machine["name"],
        process=process_cfg["name"],
        filament=filament_cfg["name"],
        plate=spec.plate,
        settings=settings,
        filament_settings=filament_settings,
        copies=copies,
        warnings=warnings,
        pauses=pauses,
    )
    (built.out / "slice.json").write_text(
        json.dumps(asdict(sliced) | {"file": final.name}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sliced


def report(s: Sliced) -> str:
    lines = [
        f"スライス完了: {s.file}",
        f"  時間 {s.duration} / フィラメント {s.grams:.1f} g ({s.meters:.2f} m) / {s.layers} 層",
        f"  {s.machine} / {s.process} / {s.filament} / {s.plate}",
    ]
    if s.settings or s.filament_settings:
        lines.append(f"  上書き: {s.settings | s.filament_settings}")
    for p in s.pauses:
        when = "" if p["minute"] is None else f"、開始から約 {p['minute']} 分後"
        lines.append(f"  一時停止: 高さ {p['z']:.2f} mm まで刷ったところ ({p['layer']} 層目の前{when})。再開は本体か Bambu Handy で")
    lines += [f"  ! {w}" for w in s.warnings]
    return "\n".join(lines)
