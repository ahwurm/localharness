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
mode (a MobileChannel handed in, the one mode that binds the artifact route); the loop, the tool
registry, the permission gate in its default mode; the image tool writing through core's
`write_artifact`; the mobile channel's own queued wire frame; the web server over ASGI (no socket); the
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

from localharness.channels.mobile.channel import MobileChannel
from localharness.channels.mobile.server import MobileServer
from localharness.cli.app import app
from localharness.cli.slash_commands import set_plugin_rows
from localharness.core.bus import EventBus
from localharness.core.events import ARTIFACT_ID_RE
from localharness.tools.builtin import generate_image_tool
from tests.conftest import FakeLLMResponse, FakeToolCall
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _DISCARD_URL, _offline_provider
from tests.unit.channels.test_mobile_artifacts import IMMUTABLE, PNG
from tests.unit.channels.test_mobile_server import BEARER, TOKEN
from tests.unit.channels.test_mobile_reference_page import FIND_IMAGES, HELLO, PAGE, _drive
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
                        app=MobileServer(self._channel, token=TOKEN).app), base_url="http://web.test") as c:
                    out["get"] = await c.get(f"/api/artifacts/image/{art['id']}", headers=BEARER)
        finally:
            await self._channel.stop()

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)

    async def run() -> None:
        from localharness.cli.start_cmd import _start_async
        await _start_async(None, False, False, None, channel_mode="mobile",
                           mobile_channel=MobileChannel(bus=EventBus(), config={}))

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


# --- criterion 3 (the rig side; the phone itself is the owner's checkpoint) ----------------------


def test_image_on_end_to_end(tmp_path, monkeypatch, fake_home):
    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    enabled = _invoke("plugins", "enable", "image", "--set", "comfyui_url=http://comfy.test")
    assert enabled.exit_code == 0, enabled.output
    seen = _fake_comfy(monkeypatch)
    s = _session(monkeypatch, turn="Draw a red square.",
                 tool_args={"prompt": "a red square", "width": 512, "height": 512, "steps": 4, "seed": 7})
    assert "generate_image" in s["tools"] and "generate_image" in s["calls"][0]
    assert len(s["calls"]) == 2 and s["answer"] == "Done."
    assert seen[0] == "POST /prompt" and "GET /view" in seen
    # a. exactly one Observation for the tool, carrying a core-minted image artifact
    (obs,) = [o for o in s["observations"] if o.get("tool_name") == "generate_image"]
    art = obs["artifact"]
    assert (art["plugin"], art["kind"], art["mime"]) == ("image", "image", "image/png"), art
    assert ARTIFACT_ID_RE.fullmatch(art["id"]), art
    assert f"artifact {art['id']}" in obs["output"], obs["output"]
    # b. the file is in the core-computed root under the session's state dir (the global dir here)
    png = global_dir / "artifacts" / "image" / f"{art['id']}.png"
    assert png.read_bytes() == PNG, f"no PNG at {png}"
    # c. the generic route serves exactly those bytes, immutable
    got = s["get"]
    assert got.status_code == 200, got.status_code
    assert (got.content, got.headers["cache-control"]) == (PNG, IMMUTABLE)
    # d. the page, fed the frame the channel queued for the phone, draws one img from that route
    if shutil.which("node") is None:
        pytest.skip("no JS engine on this box: the page half of this test cannot run")
    drawn = _drive(PAGE.read_text(encoding="utf-8"),
                   HELLO + f"onEvent(\"Observation\", {json.dumps(obs)});" + FIND_IMAGES, tmp_path)
    assert [(d["cls"], d["src"]) for d in drawn] == [("genimg", f"/api/artifacts/image/{art['id']}")], drawn
    # e. offline: _session asserted every dial went to the discard port; ComfyUI was MockTransport


def test_generate_image_command_writes_a_png(tmp_path, monkeypatch, fake_home):
    _machine(tmp_path, monkeypatch, fake_home)
    assert _invoke("plugins", "enable", "image", "--set", "comfyui_url=http://comfy.test").exit_code == 0
    seen = _fake_comfy(monkeypatch)
    out = tmp_path / "sq.png"
    ran = _invoke("generate-image", "a red square", "--out", str(out))
    assert ran.exit_code == 0, ran.output
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "POST /prompt" in seen


def test_doctor_reports_reachable_and_unreachable(tmp_path, monkeypatch, fake_home):
    from localharness.cli import doctor_cmd

    _machine(tmp_path, monkeypatch, fake_home)
    assert _invoke("plugins", "enable", "image", "--set", "comfyui_url=http://comfy.test").exit_code == 0
    recorded: list[str] = []
    real = doctor_cmd._summarize_and_exit

    def spy(failures: list[str]) -> None:
        recorded[:] = failures
        real(failures)

    monkeypatch.setattr(doctor_cmd, "_summarize_and_exit", spy)
    _fake_comfy(monkeypatch)
    up = _invoke("doctor").output
    assert "✓ image: ComfyUI reachable at http://comfy.test (template: qwen-image-2.1)" in up, up
    assert "plugin-image" not in recorded, recorded
    _fake_comfy(monkeypatch, down=True)
    down = _invoke("doctor").output
    assert "image: ComfyUI unreachable at http://comfy.test" in down, down
    assert "plugin-image" in recorded, recorded
