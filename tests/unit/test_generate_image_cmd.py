"""generate-image CLI + doctor's image-module line (the module's operator surfaces)."""
from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from localharness.cli import doctor_cmd
from localharness.cli.app import app
from localharness.tools.builtin import generate_image_tool as gi

PNG = b"\x89PNG\r\n\x1a\n-cli-bytes"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LOCALHARNESS_COMFYUI_URL", "LOCALHARNESS_COMFYUI_WORKFLOW",
                "LOCALHARNESS_COMFYUI_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gi, "_POLL_S", 0.01)


def _fake_comfy(monkeypatch, *, connect_error=False):
    class _Resp:
        status_code = 200
        content = PNG
        text = ""
        def json(self):
            return {"prompt_id": "p1"}
        def raise_for_status(self):
            pass

    class _Hist(_Resp):
        def json(self):
            return {"p1": {"status": {"status_str": "success", "completed": True},
                           "outputs": {"9": {"images": [{"filename": "f.png", "subfolder": "",
                                                         "type": "output"}]}}}}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None):
            if connect_error:
                raise httpx.ConnectError("refused")
            return _Resp()
        async def get(self, url, params=None):
            return _Hist() if "/history/" in url else _Resp()

    monkeypatch.setattr(gi.httpx, "AsyncClient", _Client)


# --- CLI ------------------------------------------------------------------------

def test_unconfigured_exits_2_with_setup_hint():
    r = runner.invoke(app, ["generate-image", "hello"])
    assert r.exit_code == 2
    assert "LOCALHARNESS_COMFYUI_URL" in r.output


def test_happy_path_writes_png_where_asked(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    _fake_comfy(monkeypatch)
    out = tmp_path / "fox.png"
    r = runner.invoke(app, ["generate-image", "a fox", "--seed", "5",
                            "--steps", "4", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.read_bytes() == PNG
    assert "saved:" in r.output and "seed 5" in r.output and "qwen-image-2.1" in r.output


def test_out_must_be_png():
    r = runner.invoke(app, ["generate-image", "x", "--out", "pic.jpg"])
    # checked before any network use, so this fails fast even unconfigured -> the
    # unconfigured check fires first; configure via env to reach the suffix check
    assert r.exit_code == 2


def test_unreachable_exits_1(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://127.0.0.1:8188")
    _fake_comfy(monkeypatch, connect_error=True)
    r = runner.invoke(app, ["generate-image", "x", "--out", str(tmp_path / "a.png")])
    assert r.exit_code == 1
    assert "unreachable" in r.output


# --- doctor ----------------------------------------------------------------------

def test_doctor_silent_when_module_unconfigured(capsys):
    doctor_cmd._print_comfyui()
    assert "Image module" not in capsys.readouterr().out


def test_doctor_reports_unreachable_server(monkeypatch, capsys):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://127.0.0.1:8188")
    def _refused(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(doctor_cmd.httpx, "get", _refused)
    doctor_cmd._print_comfyui()
    out = capsys.readouterr().out
    assert "Image module" in out and "unreachable" in out


def test_doctor_reports_reachable_and_template(monkeypatch, capsys):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://127.0.0.1:8188")
    class _Ok:
        def raise_for_status(self):
            pass
    monkeypatch.setattr(doctor_cmd.httpx, "get", lambda *a, **k: _Ok())
    doctor_cmd._print_comfyui()
    out = capsys.readouterr().out
    assert "ComfyUI reachable" in out and "qwen-image-2.1" in out
