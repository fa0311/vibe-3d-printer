"""Bambu プリンタと LAN で直接やりとりする (状態取得・ファイル送信・印刷開始/停止)。

経路: MQTT over TLS (8883) で指示と状態、暗黙的 FTPS (990) でファイル転送。
ユーザー名は常に "bblp"、パスワードはプリンタ画面に出るアクセスコード。
2025年以降のファームウェアでは、Bambu 純正以外からの印刷開始は
「LAN Only Mode + Developer Mode」を有効にしたプリンタでしか受け付けられない。
"""

from __future__ import annotations

import ftplib
import json
import socket
import ssl
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import paho.mqtt.client as mqtt

from . import config

MQTT_PORT = 8883
FTPS_PORT = 990
START_TIMEOUT = 60   # 印刷開始の指示から動き出すまで待つ秒数

IDLE_STATES = {"IDLE", "FINISH", "FAILED"}
STATE_JA = {
    "IDLE": "待機中", "PREPARE": "準備中", "RUNNING": "印刷中", "PAUSE": "一時停止中",
    "FINISH": "完了", "FAILED": "失敗/中止", "SLICING": "スライス中",
}
DEVELOPER_MODE_HELP = (
    "プリンタが外部からの指示を拒否しました。プリンタ本体の画面で\n"
    "  設定 > LAN Only Mode を ON → 同じ画面の Developer Mode を ON\n"
    "にしてください (アクセスコードが変わったら printer.local.toml も更新)。"
)


class PrinterError(RuntimeError):
    pass


def _tls() -> ssl.SSLContext:
    # プリンタは Bambu の独自 CA で署名した証明書を出す (公的に検証できない)。
    # 相手は自宅 LAN 内の機器で、認証はアクセスコードで行う。
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class _Session:
    def __init__(self, conn: config.Connection):
        self.conn = conn
        self.state: dict = {}
        self.replies: list[dict] = []
        self._got_status = threading.Event()
        self._connected = threading.Event()
        self._failure: str | None = None
        self._seq = int(time.time()) % 100000
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"v3d-{uuid.uuid4().hex[:8]}"
        )
        self.client.username_pw_set("bblp", conn.access_code)
        self.client.tls_set_context(_tls())
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            self._failure = f"MQTT 接続を拒否されました ({reason_code})。アクセスコードを確認してください"
        else:
            client.subscribe(f"device/{self.conn.serial}/report")
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        report = payload.get("print")
        if not isinstance(report, dict):
            return
        if report.get("command") == "push_status":
            self.state.update(report)
            if "gcode_state" in self.state:
                self._got_status.set()
        else:
            self.replies.append(report)

    def open(self) -> None:
        try:
            self.client.connect(self.conn.ip, MQTT_PORT, keepalive=60)
        except (OSError, socket.timeout) as e:
            raise PrinterError(
                f"プリンタ {self.conn.ip}:{MQTT_PORT} に届きません ({e})。"
                "IP アドレスと、この PC がプリンタと同じネットワークにいるかを確認してください"
            ) from e
        self.client.loop_start()
        if not self._connected.wait(15):
            raise PrinterError("プリンタとの MQTT 接続がタイムアウトしました")
        if self._failure:
            raise PrinterError(self._failure)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, body: dict) -> None:
        self._seq += 1
        for section in body.values():
            section["sequence_id"] = str(self._seq)
        info = self.client.publish(f"device/{self.conn.serial}/request", json.dumps(body), qos=1)
        info.wait_for_publish(10)

    def refresh(self, timeout: float = 15) -> dict:
        self._got_status.clear()
        self.publish({"pushing": {"command": "pushall", "version": 1, "push_target": 1}})
        if not self._got_status.wait(timeout):
            raise PrinterError(
                "プリンタから状態が返ってきません。シリアル番号が正しいか確認してください"
            )
        return dict(self.state)


@contextmanager
def _session():
    s = _Session(config.connection())
    s.open()
    try:
        yield s
    finally:
        s.close()


def status() -> dict:
    with _session() as s:
        return s.refresh()


def _errors(st: dict) -> list[str]:
    out = []
    if st.get("print_error"):
        out.append(f"print_error {int(st['print_error']):08X}")
    for h in st.get("hms") or []:
        code = f"{int(h.get('attr', 0)):08X}{int(h.get('code', 0)):08X}"
        out.append("HMS_" + "_".join(code[i:i + 4] for i in range(0, 16, 4)))
    return out


def _ams_trays(st: dict) -> list[dict]:
    """フィラメントが入っている AMS スロットの一覧 (id は 0 始まりの通し番号)。"""
    trays = []
    for unit in (st.get("ams") or {}).get("ams") or []:
        for tray in unit.get("tray") or []:
            if tray.get("tray_type"):
                trays.append({
                    "id": int(unit.get("id", 0)) * 4 + int(tray.get("id", 0)),
                    "type": tray.get("tray_type"),
                    "color": str(tray.get("tray_color", ""))[:6],
                    "name": tray.get("tray_sub_brands") or tray.get("tray_type"),
                })
    return trays


def describe(st: dict) -> str:
    state = st.get("gcode_state", "?")
    lines = [f"状態: {STATE_JA.get(state, state)} ({state})"]
    if state in ("RUNNING", "PAUSE", "PREPARE"):
        lines[0] += (
            f"  {st.get('mc_percent', '?')}%  残り {st.get('mc_remaining_time', '?')} 分"
            f"  層 {st.get('layer_num', '?')}/{st.get('total_layer_num', '?')}"
        )
    if st.get("subtask_name"):
        lines.append(f"ジョブ: {st['subtask_name']}")
    lines.append(
        f"ノズル {st.get('nozzle_temper', 0):.0f}/{st.get('nozzle_target_temper', 0):.0f}℃  "
        f"ベッド {st.get('bed_temper', 0):.0f}/{st.get('bed_target_temper', 0):.0f}℃"
    )
    trays = _ams_trays(st)
    if trays:
        lines.append("AMS: " + ", ".join(f"スロット{t['id'] + 1} {t['name']} #{t['color']}" for t in trays))
    else:
        ext = st.get("vt_tray") or {}
        lines.append(f"AMS: なし (外部スプール: {ext.get('tray_type') or '種類未設定'})")
    errs = _errors(st)
    lines.append("エラー: " + (", ".join(errs) if errs else "なし"))
    return "\n".join(lines)


class _ImplicitFTPS(ftplib.FTP_TLS):
    """接続直後から TLS で話す FTPS (標準ライブラリは明示的 FTPS しか扱えない)。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sock = None

    @property
    def sock(self):
        return self._sock

    @sock.setter
    def sock(self, value):
        if value is not None and not isinstance(value, ssl.SSLSocket):
            value = self.context.wrap_socket(value)
        self._sock = value

    def ntransfercmd(self, cmd, rest=None):
        # プリンタはデータ接続に制御接続と同じ TLS セッションの再利用を要求する
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:
            conn = self.context.wrap_socket(conn, server_hostname=self.host, session=self.sock.session)
        return conn, size

    def storbinary(self, cmd, fp, blocksize=65536, callback=None, rest=None):
        self.voidcmd("TYPE I")
        conn = self.transfercmd(cmd, rest)
        try:
            while buf := fp.read(blocksize):
                conn.sendall(buf)
            # TLS を正しく終了してから閉じる。これを省くと、サーバから届いて未読のままの
            # TLS レコードがあるとき close() が RST になり、送信の末尾が相手に届かず欠ける。
            # 終了応答を返さないサーバで固まらないよう、待つのは数秒まで。
            conn.settimeout(5)
            try:
                conn.unwrap()
            except (OSError, ValueError):
                pass
        finally:
            conn.close()
        return self.voidresp()


def upload(file: Path, remote_name: str) -> None:
    conn = config.connection()
    ftp = _ImplicitFTPS(context=_tls())
    try:
        ftp.connect(conn.ip, FTPS_PORT, timeout=30)
        ftp.login("bblp", conn.access_code)
        ftp.prot_p()
        with file.open("rb") as fh:
            ftp.storbinary(f"STOR /{remote_name}", fh)
        try:
            remote_size = ftp.size(f"/{remote_name}")
        except ftplib.Error:
            remote_size = None   # SIZE に対応していないサーバ
        if remote_size is not None and remote_size != file.stat().st_size:
            raise PrinterError(
                f"送信したファイルが欠けています (送信 {file.stat().st_size} バイト / "
                f"プリンタ側 {remote_size} バイト)"
            )
    except (OSError, ftplib.Error) as e:
        raise PrinterError(
            f"ファイル送信に失敗しました ({e})。microSD カードがプリンタに入っているか確認してください"
        ) from e
    finally:
        try:
            ftp.quit()
        except (OSError, ftplib.Error, AttributeError):
            pass


def send_and_start(file: Path, *, bed_leveling: bool = True, ams_slot: int | None = None) -> None:
    """ファイルを送って印刷を始め、動き出すところまで見届ける。

    呼ぶ前に人が確認すること: ベッドの上が空 / プレートが載っている / フィラメントが装填済み。
    """
    remote = file.name
    with _session() as s:
        st = s.refresh()
        state = st.get("gcode_state")
        if state not in IDLE_STATES:
            raise PrinterError(f"プリンタが空いていません ({STATE_JA.get(state, state)})\n{describe(st)}")
        if st.get("sdcard") is False:
            raise PrinterError("microSD カードが入っていません (LAN 経由の印刷に必要)")

        trays = _ams_trays(st)
        if ams_slot is not None:
            if ams_slot - 1 not in {t["id"] for t in trays}:
                raise PrinterError(f"AMS スロット {ams_slot} にフィラメントがありません")
            tray = ams_slot - 1
        else:
            tray = trays[0]["id"] if trays else None
        source = f"AMS スロット{tray + 1}" if tray is not None else "外部スプール"

        print(f"送信中: {remote} ({file.stat().st_size / 1e6:.1f} MB)")
        upload(file, remote)

        s.replies.clear()
        s.publish({
            "print": {
                "command": "project_file",
                "param": "Metadata/plate_1.gcode",
                "project_id": "0", "profile_id": "0", "task_id": "0", "subtask_id": "0",
                "subtask_name": remote.removesuffix(".gcode.3mf"),
                "file": remote,
                "url": f"ftp:///{remote}",
                "md5": "",
                "timelapse": False,
                "bed_type": "auto",
                "bed_leveling": bed_leveling,
                "bed_levelling": bed_leveling,
                "flow_cali": True,
                "vibration_cali": False,
                "layer_inspect": False,
                "use_ams": tray is not None,
                "ams_mapping": [tray if tray is not None else 0],
            }
        })
        print(f"印刷開始を指示しました (フィラメント: {source})。反応を待っています…")

        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            time.sleep(2)
            for r in s.replies:
                if r.get("command") == "project_file" and str(r.get("result", "")).lower() not in ("", "success"):
                    raise PrinterError(f"印刷開始に失敗: {r.get('reason') or r}\n{DEVELOPER_MODE_HELP}")
            now = s.state.get("gcode_state")
            if _errors(s.state) and now not in ("PREPARE", "RUNNING"):
                raise PrinterError(f"プリンタがエラーを報告しました\n{describe(s.state)}\n{DEVELOPER_MODE_HELP}")
            if now in ("PREPARE", "RUNNING") and s.state.get("subtask_name", remote).startswith(
                remote.removesuffix(".gcode.3mf")
            ):
                print("印刷が始まりました。\n" + describe(s.state))
                return
        raise PrinterError(
            f"{START_TIMEOUT} 秒待っても印刷が始まりません。\n{describe(s.state)}\n{DEVELOPER_MODE_HELP}"
        )


def control(command: str) -> None:
    if command not in ("pause", "resume", "stop"):
        raise ValueError(command)
    with _session() as s:
        s.refresh()
        s.publish({"print": {"command": command, "param": ""}})
        time.sleep(2)
