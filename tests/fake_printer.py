"""模擬 Bambu プリンタ: TLS 付き MQTT ブローカー (最小実装) + 暗黙的 FTPS サーバ。

実機と同じ手順 (bblp/アクセスコードで認証 → pushall → FTPS で送信 → project_file) を
v3d/printer.py が正しく踏めるかを確かめるためのもの。実機の癖までは再現しない。
"""

from __future__ import annotations

import json
import socket
import ssl
import subprocess
import threading
from pathlib import Path

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import TLS_FTPHandler
from pyftpdlib.servers import FTPServer

SERIAL = "FAKE0123456789"
ACCESS_CODE = "12345678"


def make_cert(directory: Path) -> tuple[Path, Path]:
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-subj", "/CN=fake-printer", "-keyout", str(key), "-out", str(cert)],
        check=True, capture_output=True,
    )
    return cert, key


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n % 128, n // 128
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _string(b: bytes, at: int) -> tuple[bytes, int]:
    n = int.from_bytes(b[at:at + 2], "big")
    return b[at + 2:at + 2 + n], at + 2 + n


class FakePrinter:
    def __init__(self, tmp: Path):
        self.sdcard = tmp / "sdcard"
        self.sdcard.mkdir()
        self.cert, self.key = make_cert(tmp)
        self.mqtt_port = _free_port()
        self.ftps_port = _free_port()
        self.developer_mode = True
        self.state = {
            "command": "push_status", "gcode_state": "IDLE", "sdcard": True,
            "nozzle_temper": 25.0, "nozzle_target_temper": 0.0,
            "bed_temper": 24.0, "bed_target_temper": 0.0, "print_error": 0, "hms": [],
        }
        self.received: list[dict] = []   # 受け取った指示 (pushall 以外)
        self._stop = threading.Event()
        self._start_mqtt()
        self._start_ftps()

    # ---- MQTT ----
    def _start_mqtt(self) -> None:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert, self.key)
        self._listener = socket.create_server(("127.0.0.1", self.mqtt_port))
        self._listener.settimeout(0.2)

        def accept_loop():
            while not self._stop.is_set():
                try:
                    raw, _ = self._listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                try:
                    conn = ctx.wrap_socket(raw, server_side=True)
                except (ssl.SSLError, OSError):
                    continue
                threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()

    @staticmethod
    def _read(conn, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError
            buf += chunk
        return buf

    def _packet(self, conn) -> tuple[int, bytes]:
        head = self._read(conn, 1)[0]
        length, shift = 0, 0
        while True:
            byte = self._read(conn, 1)[0]
            length |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        return head, self._read(conn, length)

    def _report(self, conn, body: dict) -> None:
        topic = f"device/{SERIAL}/report".encode()
        payload = len(topic).to_bytes(2, "big") + topic + json.dumps({"print": body}).encode()
        conn.sendall(b"\x30" + _varint(len(payload)) + payload)

    def _serve(self, conn) -> None:
        try:
            while not self._stop.is_set():
                head, body = self._packet(conn)
                kind = head >> 4
                if kind == 1:      # CONNECT
                    _, at = _string(body, 0)
                    flags = body[at + 1]
                    _, at = _string(body, at + 4)          # client id
                    user, at = _string(body, at) if flags & 0x80 else (b"", at)
                    password, at = _string(body, at) if flags & 0x40 else (b"", at)
                    ok = user == b"bblp" and password == ACCESS_CODE.encode()
                    conn.sendall(bytes([0x20, 2, 0, 0 if ok else 5]))
                    if not ok:
                        return
                elif kind == 8:    # SUBSCRIBE
                    conn.sendall(bytes([0x90, 3]) + body[:2] + b"\x00")
                elif kind == 3:    # PUBLISH
                    qos = (head >> 1) & 3
                    topic, at = _string(body, 0)
                    if qos:
                        conn.sendall(bytes([0x40, 2]) + body[at:at + 2])
                        at += 2
                    if topic == f"device/{SERIAL}/request".encode():
                        self._handle(conn, json.loads(body[at:]))
                elif kind == 12:   # PINGREQ
                    conn.sendall(b"\xd0\x00")
                elif kind == 14:   # DISCONNECT
                    return
        except (ConnectionError, OSError, ssl.SSLError):
            pass
        finally:
            conn.close()

    def _handle(self, conn, msg: dict) -> None:
        if "pushing" in msg:
            self._report(conn, self.state)
            return
        cmd = msg.get("print", {})
        self.received.append(cmd)
        name = cmd.get("command")
        if name == "project_file":
            reply = {"command": "project_file", "sequence_id": cmd.get("sequence_id")}
            file = self.sdcard / cmd.get("url", "").removeprefix("ftp:///")
            if not self.developer_mode:
                self._report(conn, reply | {"result": "fail", "reason": "mqtt message verify failed"})
            elif not file.is_file():
                self._report(conn, reply | {"result": "fail", "reason": "file not found"})
            else:
                self.state |= {"gcode_state": "PREPARE", "subtask_name": cmd.get("subtask_name", "")}
                self._report(conn, reply | {"result": "success"})
                self._report(conn, self.state)
        elif name in ("pause", "resume", "stop"):
            self.state["gcode_state"] = {"pause": "PAUSE", "resume": "RUNNING", "stop": "FAILED"}[name]
            self._report(conn, self.state)

    # ---- FTPS (接続直後から TLS) ----
    def _start_ftps(self) -> None:
        authorizer = DummyAuthorizer()
        authorizer.add_user("bblp", ACCESS_CODE, str(self.sdcard), perm="elradfmwMT")

        class Handler(TLS_FTPHandler):
            ssl_context = None

            def handle(self):
                self.secure_connection(self.ssl_context)

            def handle_ssl_established(self):
                TLS_FTPHandler.handle(self)

        Handler.certfile = str(self.cert)
        Handler.keyfile = str(self.key)
        Handler.authorizer = authorizer
        Handler.tls_data_required = True
        self._ftp = FTPServer(("127.0.0.1", self.ftps_port), Handler)
        threading.Thread(target=self._ftp.serve_forever, kwargs={"handle_exit": False}, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        self._listener.close()
        self._ftp.close_all()
