"""The image plugin's roadmap criteria 1-3, composed on the surfaces the rig can reach.

Criteria, verbatim (ROADMAP, the image plugin phase, as amended):

    1. **Off by default.** On a fresh install, `localharness start` registers no `generate_image`,
       `doctor` prints `image: off — turn on: localharness plugins enable image` and no ComfyUI
       check, `plugins list` shows image off with its enable command, and `--help` lists no
       `generate-image`.
    2. **Enabled but unconfigured, and machine-only.** After `plugins enable image` with no address,
       start warns `plugin image: unconfigured — set image.comfyui_url` and registers nothing, and
       doctor says the same; a project folder cannot set `image.comfyui_url` (its value is dropped
       with a warning naming the file) and the machine's value serves.
    3. **A picture reaches the phone through the generic artifact route.** With image on and a
       ComfyUI answering, one turn's `generate_image` call yields `Observation.artifact`; GET
       `/api/artifacts/image/<id>` serves those bytes; the page renders `<img class=genimg>` from
       that route; `localharness generate-image` writes a PNG; doctor reports reachable and
       unreachable.

REAL: every CLI command (the Typer app through CliRunner); plugin discovery, the resolver and
lifecycle, the settings layers and their machine-only rule; the session, `_start_async` run in web
mode (a WebChannel handed in, the one mode that binds the artifact route); the loop, the tool
registry, the permission gate in its default mode; the image tool writing through core's
`write_artifact`; the web channel's own queued wire frame; the web server over ASGI (no socket); the
page's reducer run verbatim under node.

STUBBED: ComfyUI (an `httpx.MockTransport` at `generate_image_tool._TRANSPORT`, the one seam the
tool, the command and doctor share — no socket is ever opened to it); the LLM probe, the tokenizer
and the REPL's read loop (`_stub_start_boundaries`, `drive` below performs the loop's steps); the
model's two replies (`LLMClient.stream_complete`). The provider points at the loopback discard port
and every address a session dials is recorded and asserted.

NOT proven here: the owner's phone rendering the picture (device row, owner's checkpoint); a real
ComfyUI; a real model.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import socket

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import WebServer
from localharness.cli.app import app
from localharness.cli.slash_commands import set_plugin_rows
from localharness.core.bus import EventBus
from localharness.core.events import ARTIFACT_ID_RE
from localharness.tools.builtin import generate_image_tool
from tests.conftest import FakeLLMResponse, FakeToolCall
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _DISCARD_URL, _offline_provider
from tests.unit.channels.test_web_artifacts import IMMUTABLE, PNG
from tests.unit.channels.test_web_server import BEARER, TOKEN
from tests.unit.channels.test_web_reference_page import FIND_IMAGES, HELLO, PAGE, _drive
from tests.unit.test_image_plugin import _full
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries
from tests.unit.test_start_plugins import _record_loop
from tests.unit.test_workspace_state_landing import _boom, _hermetic

OFF_ROW = "off — turn on: localharness plugins enable image"
UNCONFIGURED = "plugin image: unconfigured — set image.comfyui_url"
NEXT_STEP = ("next step — give it the ComfyUI address: "
             "localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188")
DROPPED = ": only the global config may set it"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _process_state():
    yield
    set_plugin_rows(())


def _invoke(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"`localharness {' '.join(args)}` raised {result.exception!r}\n{result.output}")
    return result


def _machine(tmp_path, monkeypatch, fake_home, *, project: bool = False):
    """A hermetic machine: fake HOME holding the global layer, an offline provider, cwd outside any
    project (or inside a git project when `project`). Returns (global_dir, cwd)."""
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, fake_home, home)
    _stub_start_boundaries(global_dir, monkeypatch, real_plugins=True)
    _offline_provider(global_dir)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    cwd = home / "proj" if project else tmp_path / "elsewhere"
    (cwd / ".git").mkdir(parents=True) if project else cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)
    return global_dir, cwd


def _fake_comfy(monkeypatch, *, host: str = "comfy.test", down: bool = False) -> list[str]:
    """ComfyUI answering only for `host`; any other host fails the test. Returns requests seen."""
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.host == host, f"a request went to {req.url} — only {host} may be asked"
        seen.append(f"{req.method} {req.url.path}")
        if down:
            raise httpx.ConnectError("refused", request=req)
        path = req.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={})
        if path.startswith("/object_info/"):
            cls = path.removeprefix("/object_info/")
            return httpx.Response(200, json={cls: _full().get(cls, {"input": {"required": {}}})})
        if path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if path == "/history/p1":
            return httpx.Response(200, json={"p1": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"459": {"images": [{"filename": "f.png", "subfolder": "",
                                                "type": "output"}]}}}})
        if path == "/view":
            return httpx.Response(200, content=PNG)
        return httpx.Response(404)

    monkeypatch.setattr(generate_image_tool, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(generate_image_tool, "_POLL_S", 0.01)
    return seen


def _record_dials(monkeypatch) -> list:
    real = socket.socket.connect
    dialed: list = []

    def connect(self, address):
        dialed.append(address)
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return dialed


def _session(monkeypatch, *, turn: str | None = None, tool_args: dict | None = None) -> dict:
    """One web-mode session through `_start_async`. With `turn`, the model calls generate_image
    with `tool_args` and then says "Done."; `drive` dispatches the line, keeps the frames the channel
    queued for an attached phone, and GETs the artifact over ASGI. Returns what it observed."""
    out: dict = {"printed": _capture_start_console(monkeypatch), "loops": _record_loop(monkeypatch),
                 "calls": [], "dialed": _record_dials(monkeypatch)}

    async def model(self, messages, tools=None, on_token=None, **_):
        out["calls"].append([t.name for t in tools or ()])
        if len(out["calls"]) == 1:
            return FakeLLMResponse(tool_calls=[FakeToolCall(id="c1", name="generate_image",
                                                            arguments=tool_args or {})]), None
        return FakeLLMResponse(content="Done."), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)

    async def drive(self):  # stands in for OrchestratorREPL.run
        await self._channel.start()
        try:
            if turn is None:
                return
            phone = self._channel.attach_client()
            out["answer"] = await asyncio.wait_for(await self._dispatch_input(turn), 30)
            frames = []
            while not phone.queue.empty():
                frames.append(phone.queue.get_nowait())
            out["frames"] = frames
            obs = [json.loads(p) for t, _, p in frames if t == "Observation"]
            out["observations"] = obs
            art = next((o["artifact"] for o in obs if o.get("artifact")), None)
            if art:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(
                        app=WebServer(self._channel, token=TOKEN).app), base_url="http://web.test") as c:
                    out["get"] = await c.get(f"/api/artifacts/image/{art['id']}", headers=BEARER)
        finally:
            await self._channel.stop()

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)

    async def run() -> None:
        from localharness.cli.start_cmd import _start_async
        await _start_async(None, False, False, None, channel_mode="web",
                           web_channel=WebChannel(bus=EventBus(), config={}))

    asyncio.run(run())
    (loop,) = out["loops"]
    out["tools"] = {n for scope in loop["tool_registry"]._tools.values() for n in scope}
    addresses = {a[:2] for a in out["dialed"] if isinstance(a, tuple)}
    assert addresses <= {("127.0.0.1", 9)}, f"the session dialed {addresses}, not only {_DISCARD_URL}"
    return out


# --- criterion 1 ---------------------------------------------------------------------------------


def test_image_is_off_by_default(tmp_path, monkeypatch, fake_home):
    _machine(tmp_path, monkeypatch, fake_home)
    listed = _invoke("plugins", "list").output
    (row,) = [line for line in listed.splitlines() if re.match(r"\s*image\s", line)]
    assert "makes pictures with ComfyUI" in row and OFF_ROW in row, row
    assert "generate-image" not in _invoke("--help").output
    doctor = _invoke("doctor").output
    assert f"image: {OFF_ROW}" in doctor, doctor
    assert "ComfyUI" not in doctor, doctor
    s = _session(monkeypatch)
    assert "generate_image" not in s["tools"], sorted(s["tools"])
    assert not any("plugin image:" in line for line in s["printed"]), s["printed"]


# --- criterion 2 ---------------------------------------------------------------------------------


def test_enabled_but_unconfigured_warns_and_registers_nothing(tmp_path, monkeypatch, fake_home):
    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    enabled = _invoke("plugins", "enable", "image")
    assert enabled.exit_code == 0, enabled.output
    assert NEXT_STEP in enabled.output, enabled.output
    assert yaml.safe_load((global_dir / "overrides.yaml").read_text()) == {"image": {"enabled": True}}
    s = _session(monkeypatch)
    assert any(UNCONFIGURED in line for line in s["printed"]), s["printed"]
    assert "generate_image" not in s["tools"], sorted(s["tools"])
    doctor = _invoke("doctor").output
    assert "image: unconfigured — set image.comfyui_url" in doctor, doctor


def test_a_project_cannot_set_the_comfyui_address(tmp_path, monkeypatch, fake_home):
    global_dir, proj = _machine(tmp_path, monkeypatch, fake_home, project=True)
    (global_dir / "overrides.yaml").write_text(
        "image: {enabled: true, comfyui_url: http://global.test}\n", encoding="utf-8")
    assert _invoke("init", "--workspace").exit_code == 0
    ws_overrides = proj / ".localharness" / "overrides.yaml"
    ws_overrides.write_text("image: {comfyui_url: http://project.test}\n", encoding="utf-8")
    seen = _fake_comfy(monkeypatch, host="global.test")
    doctor = _invoke("doctor").output
    flat = " ".join(doctor.split())  # doctor soft-wraps; the path may break across lines
    assert re.search(r"ignoring image\.comfyui_url in \S*overrides\.yaml" + re.escape(DROPPED), flat), doctor
    dropped = re.search(r"ignoring image\.comfyui_url in (\S+)" + re.escape(DROPPED), flat).group(1)
    assert dropped.endswith(".localharness/overrides.yaml") and str(proj.name) in dropped, dropped
    assert "ComfyUI reachable at http://global.test" in doctor, doctor
    assert seen and "GET /system_stats" in seen
    info = _invoke("plugins", "info", "image", "--json")
    assert info.exit_code == 0, info.output
    settings = {s["path"]: s for s in json.loads(info.stdout)["settings"]}
    assert settings["image.comfyui_url"]["current_value"] == "http://global.test", settings
