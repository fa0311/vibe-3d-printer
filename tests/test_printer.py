"""v3d/printer.py を模擬プリンタ相手に動かす。実行: uv run pytest"""

from __future__ import annotations

import pytest

from v3d import config, printer

from .fake_printer import ACCESS_CODE, SERIAL, FakePrinter


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = FakePrinter(tmp_path)
    monkeypatch.setattr(printer, "MQTT_PORT", f.mqtt_port)
    monkeypatch.setattr(printer, "FTPS_PORT", f.ftps_port)
    monkeypatch.setattr(printer, "START_TIMEOUT", 8)
    monkeypatch.setattr(
        config, "connection", lambda: config.Connection("127.0.0.1", SERIAL, ACCESS_CODE)
    )
    yield f
    f.close()


@pytest.fixture
def job(tmp_path):
    file = tmp_path / "widget.gcode.3mf"
    file.write_bytes(bytes(range(256)) * 4000)   # 約 1 MB
    return file


def test_status(fake):
    st = printer.status()
    assert st["gcode_state"] == "IDLE"
    assert "待機中" in printer.describe(st)


def test_wrong_access_code(fake, monkeypatch):
    monkeypatch.setattr(config, "connection", lambda: config.Connection("127.0.0.1", SERIAL, "wrong"))
    with pytest.raises(printer.PrinterError, match="アクセスコード"):
        printer.status()


def test_unreachable(fake, monkeypatch):
    fake.close()
    with pytest.raises(printer.PrinterError, match="届きません"):
        printer.status()


def test_send_and_start(fake, job):
    printer.send_and_start(job)
    assert (fake.sdcard / job.name).read_bytes() == job.read_bytes()
    (cmd,) = fake.received
    assert cmd["command"] == "project_file"
    assert cmd["url"] == f"ftp:///{job.name}"
    assert cmd["param"] == "Metadata/plate_1.gcode"
    assert cmd["subtask_name"] == "widget"
    assert cmd["use_ams"] is False and cmd["bed_leveling"] is True
    assert fake.state["gcode_state"] == "PREPARE"


def test_uses_loaded_ams_slot(fake, job):
    fake.state["ams"] = {"ams": [{"id": "0", "tray": [
        {"id": "0"}, {"id": "1", "tray_type": "PLA", "tray_color": "FFFFFFFF"},
    ]}]}
    printer.send_and_start(job)
    assert fake.received[0]["use_ams"] is True
    assert fake.received[0]["ams_mapping"] == [1]


def test_refused_without_developer_mode(fake, job):
    fake.developer_mode = False
    with pytest.raises(printer.PrinterError, match="Developer Mode"):
        printer.send_and_start(job)


def test_busy_printer_is_left_alone(fake, job):
    fake.state["gcode_state"] = "RUNNING"
    with pytest.raises(printer.PrinterError, match="空いていません"):
        printer.send_and_start(job)
    assert not (fake.sdcard / job.name).exists()
    assert fake.received == []


def test_pause(fake):
    printer.control("pause")
    assert fake.received[0]["command"] == "pause"
    assert fake.state["gcode_state"] == "PAUSE"
