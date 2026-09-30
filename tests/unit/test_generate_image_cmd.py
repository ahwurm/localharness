"""`localharness generate-image`: the image plugin's command, mounted lazily from its manifest,
reading image.* from the same resolved plugin config a session reads (never the environment).

Hermetic: `components_home` is the global config dir; each case writes its overrides.yaml; cwd is a
tmp project with no .localharness. ComfyUI is faked at `generate_image_tool._TRANSPORT`."""
from __future__ import annotations

import os
import re
import subprocess
import sys

import httpx
import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.tools.builtin import generate_image_tool as gi

PNG = b"\x89PNG\r\n\x1a\n-cli-bytes"
GIF = b"GIF89a-not-a-png"
URL = "http://comfy.test"
runner = CliRunner()


@pytest.fixture(autouse=True)
def home(components_home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    monkeypatch.setattr(gi, "_POLL_S", 0.01)
    return components_home


def _image(home, **cfg) -> None:
    body = ", ".join(f"{k}: {v}" for k, v in {"enabled": "true", **cfg}.items())
    (home / "overrides.yaml").write_text(f"image: {{{body}}}\n", encoding="utf-8")


def _fake_comfy(monkeypatch, *, connect_error=False, view=PNG) -> list[str]:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(f"{req.method} {req.url.path}")
        if connect_error:
            raise httpx.ConnectError("refused", request=req)
        if req.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if req.url.path == "/history/p1":
            return httpx.Response(200, json={"p1": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"9": {"images": [{"filename": "f.png", "subfolder": "", "type": "output"}]}}}})
        if req.url.path == "/view":
            return httpx.Response(200, content=view)
        return httpx.Response(404)

    monkeypatch.setattr(gi, "_TRANSPORT", httpx.MockTransport(handler))
    return seen


def test_on_but_not_set_up_exits_2_naming_the_set_command(home):
    _image(home)
    r = runner.invoke(app, ["generate-image", "hello"])
    assert r.exit_code == 2, r.output
    assert "image generation is not set up" in r.output
    assert "localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188" in r.output


def test_configured_it_writes_the_png_where_asked(home, tmp_path, monkeypatch):
    _image(home, comfyui_url=URL)
    _fake_comfy(monkeypatch)
    out = tmp_path / "cat.png"
    r = runner.invoke(app, ["generate-image", "a cat", "--out", str(out), "--seed", "5",
                            "--width", "992", "--height", "512"])
    assert r.exit_code == 0, r.output
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for bit in ("saved:", "seed 5", "992x512", "template qwen-image-2.1"):
        assert bit in r.output


def test_a_non_png_out_exits_2_before_any_request(home, tmp_path, monkeypatch):
    _image(home, comfyui_url=URL)
    seen = _fake_comfy(monkeypatch)
    r = runner.invoke(app, ["generate-image", "a cat", "--out", str(tmp_path / "cat.jpg")])
    assert r.exit_code == 2, r.output
    assert "--out must end in .png" in r.output
    assert seen == []


def test_comfyui_down_exits_1(home, tmp_path, monkeypatch):
    _image(home, comfyui_url=URL)
    _fake_comfy(monkeypatch, connect_error=True)
    r = runner.invoke(app, ["generate-image", "x", "--out", str(tmp_path / "a.png")])
    assert r.exit_code == 1, r.output
    assert f"ComfyUI unreachable at {URL}" in r.output


def test_without_out_it_writes_one_stamped_png_in_the_current_directory(home, monkeypatch):
    _image(home, comfyui_url=URL)
    _fake_comfy(monkeypatch)
    before = set(os.listdir("."))
    r = runner.invoke(app, ["generate-image", "x"])
    assert r.exit_code == 0, r.output
    new = set(os.listdir(".")) - before
    assert len(new) == 1
    assert re.fullmatch(r"image-\d{8}-\d{6}-[0-9a-f]{6}\.png", new.pop())


def test_a_file_that_is_not_a_png_is_refused_and_not_written(home, tmp_path, monkeypatch):
    _image(home, comfyui_url=URL)
    _fake_comfy(monkeypatch, view=GIF)
    out = tmp_path / "cat.png"
    r = runner.invoke(app, ["generate-image", "x", "--out", str(out)])
    assert r.exit_code == 1, r.output
    assert "not a PNG" in r.output
    assert not out.exists()


def test_with_image_off_there_is_no_command():
    help_ = runner.invoke(app, ["--help"], env={"COLUMNS": "400"})
    assert help_.exit_code == 0, help_.output
    assert "generate-image" not in help_.output
    assert runner.invoke(app, ["generate-image", "x"]).exit_code == 2


def test_help_does_not_import_the_command_module(home, tmp_path):
    _image(home, comfyui_url=URL)
    code = ("import sys; from typer.testing import CliRunner; from localharness.cli.app import app; "
            "r = CliRunner().invoke(app, ['--help'], env={'COLUMNS': '400'}); "
            "print('generate-image' in r.output, 'localharness.cli.generate_image_cmd' in sys.modules)")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALHARNESS_")}
    env.update(LOCALHARNESS_DIR=str(home), HOME=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=tmp_path / "proj", env=env).stdout.split()
    assert out == ["True", "False"]
