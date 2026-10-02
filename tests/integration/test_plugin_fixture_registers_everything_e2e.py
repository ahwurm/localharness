"""Roadmap criterion 1 of the plugin substrate — the fixture registers everything — in ONE composed test.

Criterion 1, verbatim (ROADMAP.md, Phase 44):

    **The fixture registers everything** (§8 "Fixture registers everything"): with
    `examples/plugin-template/` installed by the dev extra and enabled, `localharness start` names it
    on the banner's plugins line; its declared tool is callable in a turn, registered under its bare
    name at global scope with `source_plugin` set; its command appears in `--help` and runs without
    the plugin runtime being imported until invoked; `doctor` prints its check in a plugins section;
    a GET on `/api/artifacts/<plugin>/<id>` serves the PNG its tool wrote into the core-computed root
    `<state dir>/artifacts/<plugin>/`; `components list` shows its config keys with layer provenance
    suffixed `(plugin: <name>)` and `plugins info <name>` prints the same dot-paths; `init` and
    `init --workspace` scaffold `plugins/` with a README naming the two ways to add a plugin and
    pointing at the example — all with zero edits to any file on CORE-02's core list.

The plugin arrives the way a stranger's would: the example is its own distribution
(`localharness-plugin-example` 0.1.0, installed into this venv by the dev dependency group, 44-08),
found through the UNMOCKED `importlib.metadata` entry point, and switched on with the shipped
`localharness plugins enable example`. One user in one project, in order: init, list, enable,
--help, a session with a turn, the phone, the command, doctor, components, info. Every step asserts
on what the command actually produced (exit code, output, files).

REAL: every CLI command (the Typer app through CliRunner); discovery (unmocked importlib.metadata),
the resolver, the plan, the lifecycle, the tool registry, the root capability floor, the permission
gate in its default `auto` mode; the session, `_start_async`, run the way `localharness web` runs it
(a WebChannel handed in) because that is the one mode that binds the artifact route — the banner,
the turn and the slash dispatch are the same code in `localharness start`; the REPL's own
`_dispatch_input` taking a typed line and a slash command the phone POSTed; the example's tool
writing its PNG through core's `write_artifact`; the web server's routes over ASGI (no socket) on the
channel the session bound; memory (SQLite) and the session row.

STUBBED: the LLM probe, the tokenizer and the REPL's read loop
(`tests/unit/test_start_cmd.py::_stub_start_boundaries` with `real_plugins=True`, so discovery stays
real; the tokenizer extended for a turn by 44-03's helper; `drive` below performs the loop's steps);
init's provider detection, for the global `init` only; and the model's two replies
(`LLMClient.stream_complete`: call `example_swatch`, then "Done."). The provider's base_url is the
loopback discard port and every address the session dials is recorded and asserted, so nothing can
reach a model: offline by construction, and checked.

The swatch's artifact also reaches Observation.artifact through the loop, the same contract the
image plugin uses; the page rendering it is proven for the image plugin in test_image_plugin_e2e.py.

NOT proven here: a turn on a real model; the terminal channel's own rendering of these lines.
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import socket
import sys
from collections import Counter
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml
from prompt_toolkit.document import Document
from typer.testing import CliRunner

import localharness
from localharness.channels.terminal import SlashCommandCompleter
from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import WebServer
from localharness.cli.app import app
from localharness.cli.slash_commands import find_row, set_plugin_rows
from localharness.cli.theme import entity
from localharness.core.bus import EventBus
from localharness.core.events import ARTIFACT_ID_RE
from localharness.core.events import Observation as ObservationEvent
from tests.conftest import FakeLLMResponse, FakeToolCall
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _DISCARD_URL, _offline_provider
from tests.unit.channels.test_web_server import BEARER, TOKEN
from tests.unit.test_init_cmd import _make_capability_result, _make_detector_result
from tests.unit.test_start_cmd import _capture_start_console, _read_sessions, _stub_start_boundaries
from tests.unit.test_start_plugins import _record_loop
from tests.unit.test_workspace_state_landing import _boom, _hermetic

PKG = "localharness_plugin_example"
SENTINEL_ENV = "LOCALHARNESS_EXAMPLE_PLUGIN_SENTINEL"
SWATCH_REPLY = "example plugin: swatches render in #4a90d9 at 8px"
ART_ID = re.compile(r"art-\d{8}-\d{6}-[0-9a-f]{6}")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMMUTABLE = "private, max-age=31536000, immutable"
# What the example defines. No core module may name any of it (the criterion's "zero edits").
EXAMPLE_NAMES = (PKG, "localharness-plugin-example", "example_swatch", SENTINEL_ENV)
runner = CliRunner()


def example_modules() -> set[str]:
    return {m for m in sys.modules if m == PKG or m.startswith(PKG + ".")}


def forget_example(sentinel: Path) -> None:
    """Module-level code runs once per process, so an import is observable only from a clean slate:
    every example module leaves sys.modules and the sentinel file goes. Both signals are asserted —
    the sentinel alone could be absent because an earlier test already imported the module."""
    for name in example_modules():
        del sys.modules[name]
    sentinel.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _process_state():
    """The slash table and sys.modules are process-wide: what this test adds leaves with it."""
    before = {m: sys.modules[m] for m in example_modules()}
    yield
    set_plugin_rows(())
    for name in example_modules():
        del sys.modules[name]
    sys.modules.update(before)


def _invoke(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"`localharness {' '.join(args)}` raised {result.exception!r}\n{result.output}")
    return result


def _record_dials(monkeypatch) -> list:
    """Every address a socket in this process dials — the session's offline proof, measured."""
    real = socket.socket.connect
    dialed: list = []

    def connect(self, address):
        dialed.append(address)
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return dialed


def test_fixture_registers_everything(tmp_path, monkeypatch, fake_home):
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, fake_home, home)
    _stub_start_boundaries(global_dir, monkeypatch, real_plugins=True)  # discovery stays REAL
    _offline_provider(global_dir)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    sentinel = tmp_path / "example-imported"
    monkeypatch.setenv(SENTINEL_ENV, str(sentinel))
    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)  # an in-project workspace never asks
    installed = [(e.name, e.value, e.dist.name, e.dist.version)
                 for e in entry_points(group="localharness.plugins")]
    assert ("example", f"{PKG}:ExamplePlugin", "localharness-plugin-example", "0.1.0") in installed, (
        f"the example plugin is not installed in this venv ({installed}) — `uv sync --extra dev` "
        "installs it (44-08). This test fails rather than skips.")

    # --- init and init --workspace scaffold plugins/ with its README (ENAB-05) -------------------
    assert _invoke("init", "--workspace").exit_code == 0
    ws = (proj / ".localharness").resolve()
    text = (ws / "plugins" / "README.md").read_text(encoding="utf-8")
    for phrase in ("never loads plugin code from a project folder", "`localharness.plugins` entry point",
                   "plugins/<name>/", "examples/plugin-template"):
        assert phrase in text, f"the workspace plugins README lacks {phrase!r}"
    machine = tmp_path / "fresh-machine"  # a SEPARATE dir: init writes config.yaml, the session's is ours
    client = MagicMock()
    client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    with monkeypatch.context() as m:  # test_init_cmd.py's hermetic provider detection
        m.setattr("localharness.cli.init_cmd.detect_provider", AsyncMock(return_value=_make_detector_result()))
        m.setattr("localharness.cli.init_cmd.LLMClient", MagicMock(return_value=client))
        m.setattr("localharness.cli.init_cmd._detect_max_model_len", lambda *_: None)
        m.setattr("localharness.cli.init_cmd._identify_endpoint_provider", lambda *_: "unknown")
        assert _invoke("init", "--config-dir", str(machine), "--force").exit_code == 0
    text = (machine / "plugins" / "README.md").read_text(encoding="utf-8")
    for phrase in ("`localharness.plugins` entry point", "plugins/<name>/", "examples/plugin-template"):
        assert phrase in text, f"the machine plugins README lacks {phrase!r}"

    # --- plugins list: installed, available, the exact enable command; nothing imported ----------
    forget_example(sentinel)
    listed = _invoke("plugins", "list", "--json")
    assert listed.exit_code == 0, listed.output
    (row,) = [r for r in json.loads(listed.stdout) if r["name"] == "example"]
    assert (row["state_kind"], row["from"], row["enable_command"]) == (
        "available", "pip: localharness-plugin-example 0.1.0", "localharness plugins enable example")
    assert not sentinel.exists() and not example_modules(), "listing an available plugin imported it"

    # --- plugins enable: the machine's overrides.yaml, never config.yaml -------------------------
    config_bytes = (global_dir / "config.yaml").read_bytes()
    enabled = _invoke("plugins", "enable", "example")
    assert enabled.exit_code == 0, enabled.output
    assert yaml.safe_load((global_dir / "overrides.yaml").read_text()) == {"example": {"enabled": True}}
    assert (global_dir / "config.yaml").read_bytes() == config_bytes, "enable rewrote config.yaml"

    # --- --help lists its command from the manifest; the command's module is not imported --------
    helped = _invoke("--help")
    assert helped.exit_code == 0
    assert re.search(r"│ example\s+Show what the example plugin does\.", helped.output), helped.output
    # 44-15 decision 1 (owner read): an ENABLED plugin's package and class load for the manifest.
    assert example_modules() == {PKG, f"{PKG}.plugin"}, f"--help imported {sorted(example_modules())}"

    # --- one session: the banner, a turn that calls the tool, the slash row, the phone -----------
    forget_example(sentinel)
    printed = _capture_start_console(monkeypatch)
    loops = _record_loop(monkeypatch)
    calls: list[dict] = []

    async def model(self, messages, tools=None, on_token=None, **_):
        calls.append({"messages": [dict(m) for m in messages], "tools": [t.name for t in tools or ()]})
        if len(calls) == 1:
            return FakeLLMResponse(tool_calls=[FakeToolCall(id="c1", name="example_swatch")]), None
        return FakeLLMResponse(content="Done."), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)
    seen: dict = {}

    async def drive(self):  # stands in for OrchestratorREPL.run: its steps, in its order
        await self._channel.start()
        try:
            seen["slash_module_before"] = f"{PKG}.slash" in sys.modules
            row = find_row("/example")
            seen["row_plugin"] = row.plugin if row is not None else None
            phone_stream = self._channel.attach_client()  # the frames a phone would receive
            # Bounded: in `auto` the gate runs an undeclared-family tool unasked; if that ever
            # changes, the turn would wait on a phone that never answers instead of failing.
            seen["answer"] = await asyncio.wait_for(await self._dispatch_input("Render a swatch."), 30)
            frames = []
            while not phone_stream.queue.empty():
                frames.append(phone_stream.queue.get_nowait())
            self._channel.detach_client(phone_stream)
            seen["observations"] = [json.loads(p) for t, _, p in frames if t == "Observation"]
            replies: list[str] = []
            real_send = self._channel.send_message

            async def record(content, agent_id=None, metadata=None):
                replies.append(content)
                await real_send(content, agent_id=agent_id, metadata=metadata)

            self._channel.send_message = record
            tool_text = next(str(m.get("content")) for m in calls[-1]["messages"] if m.get("role") == "tool")
            seen["tool_text"] = tool_text
            art = ART_ID.search(tool_text)
            seen["art_id"] = art.group(0) if art else None
            async with httpx.AsyncClient(transport=httpx.ASGITransport(
                    app=WebServer(self._channel, token=TOKEN).app), base_url="http://web.test") as phone:
                queued = await phone.post(f"/api/sessions/{self._channel.session_id}/command",
                                          json={"text": "/example"}, headers=BEARER)
                seen["queued"] = queued.json()
                await self._dispatch_input(await self._channel.read_input())  # the loop's one step
                await self._dispatch_input("/help")
                seen["menu"] = (await phone.get("/api/protocol", headers=BEARER)).json()["commands"]
                seen["get"] = await phone.get(f"/api/artifacts/example/{seen['art_id']}", headers=BEARER)
            seen["replies"] = replies
            seen["completions"] = [c.text for c in SlashCommandCompleter().get_completions(
                Document("/exa"), None)]
        finally:
            await self._channel.stop()

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)
    dialed = _record_dials(monkeypatch)

    async def session() -> None:
        from localharness.cli.start_cmd import _start_async
        await _start_async(None, False, False, None, channel_mode="web",
                           web_channel=WebChannel(bus=EventBus(), config={}))

    asyncio.run(session())

    # The banner: one plugin, named on its own line under the summary; nothing to warn about it.
    i = next(n for n, line in enumerate(printed) if "startup)" in line)
    assert entity("tool", "3 plugins") in printed[i]  # web (46-02) and memory (47) are bundled and on by default
    assert printed[i + 1] == "  " + entity("tool", "Plugins: web, memory, example"), printed[i:i + 2]  # web (46-02) and memory (47) are bundled and on by default
    assert "plugin example" not in printed[i], f"a startup warning about the example: {printed[i]}"
    assert not any("available, not enabled" in line for line in printed), "an enabled plugin was hinted"
    assert sentinel.exists() and PKG in sys.modules, "an enabled plugin was not imported by start"
    # Registered bare, at global scope, with its provenance; the root holds it and the model saw it.
    (loop,) = loops
    registry = loop["tool_registry"]
    assert "example_swatch" in registry._tools["global"]
    schema = registry.schema_of("example_swatch")
    assert (schema.name, schema.scope, schema.source_plugin) == ("example_swatch", "global", "example")
    assert "example_swatch" in calls[0]["tools"], "the root agent did not offer the tool to the model"
    # Callable in a turn: the tool ran and its result reached the model's second request.
    assert len(calls) == 2 and seen["answer"] == "Done."
    assert re.search(r"Rendered a 8x8 #4a90d9 swatch: artifact art-", seen["tool_text"]), seen["tool_text"]
    # The PNG is in the core-computed root <state dir>/artifacts/example/ — written out literally, so a
    # root that lost its plugin segment cannot pass by being computed the same wrong way here.
    png = ws / "artifacts" / "example" / f"{seen['art_id']}.png"
    assert png.is_file() and png.read_bytes().startswith(PNG_SIGNATURE), f"no PNG at {png}"
    # The same artifact rides the Observation the phone receives (the image plugin's contract).
    (obs,) = [ObservationEvent.model_validate(o) for o in seen["observations"]
              if o.get("tool_name") == "example_swatch"]
    assert obs.artifact is not None and obs.artifact.plugin == "example" and ARTIFACT_ID_RE.fullmatch(obs.artifact.id)
    assert obs.artifact.id == seen["art_id"], (obs.artifact, seen["art_id"])
    # The result the model's second request carried names that very file, so the model can tell the
    # user where the swatch is (QA-05). Compared resolved: `ws` is, the session's state dir need not be.
    saved = re.search(r"saved to (\S+\.png)", seen["tool_text"])
    assert saved is not None, f"the tool's result names no saved file: {seen['tool_text']}"
    said = Path(saved.group(1))
    assert said.is_absolute() and said.resolve() == png.resolve() and said.is_file(), (said, png)
    assert not (global_dir / "artifacts").exists(), "a workspace session's artifact landed globally"
    # The phone: the route serves exactly those bytes from the root THIS session bound.
    got = seen["get"]
    assert got.status_code == 200, f"GET /api/artifacts/example/{seen['art_id']}: {got.status_code}"
    assert (got.headers["content-type"], got.headers["cache-control"]) == ("image/png", IMMUTABLE)
    assert got.content == png.read_bytes()
    # The slash row: in the one table for the session, dispatched by the REPL's own dispatcher from a
    # command the phone POSTed, imported on first use; listed by /help, the completer and the menu.
    assert seen["row_plugin"] == "example" and seen["slash_module_before"] is False
    assert seen["queued"] == {"status": "queued", "command": "/example"}
    swatch_reply, help_reply = seen["replies"]
    assert swatch_reply == SWATCH_REPLY
    assert "/example" in help_reply and "Show the example plugin's swatch settings" in help_reply
    assert {"name": "/example", "description": "Show the example plugin's swatch settings"} in seen["menu"]
    assert seen["completions"] == ["/example"]
    assert find_row("/example") is None, "the plugin's row outlived its session"
    rows = _read_sessions(ws)
    assert len(rows) == 1 and rows[0][3] == "complete"
    addresses = {a[:2] for a in dialed if isinstance(a, tuple)}
    assert addresses <= {("127.0.0.1", 9)}, f"the session dialed {addresses}, not only {_DISCARD_URL}"

    # --- the command runs, and only now is its module imported ------------------------------------
    assert f"{PKG}.cli" not in sys.modules
    ran = _invoke("example")
    assert ran.exit_code == 0, ran.output
    assert "example plugin 0.1.0" in ran.output
    assert f"{PKG}.cli" in sys.modules

    # --- doctor: its check, in the plugins section, where the session put its artifacts ----------
    doctor = _invoke("doctor").output
    assert "✓ Plugins: web, memory, example" in doctor, doctor  # web (46-02) and memory (47) are bundled and on by default
    assert (f"✓ example: swatches render in #4a90d9; artifacts go to {ws / 'artifacts' / 'example'}"
            in doctor), doctor
    assert not [line for line in doctor.splitlines() if line.startswith("✗") and "example" in line]

    # --- components list and plugins info: the same keys, the same provenance ----------------------
    listed = _invoke("components", "list", "--json")
    assert listed.exit_code == 0, listed.output
    owned = {r["path"]: r["layer"] for r in json.loads(listed.stdout) if r["plugin"] == "example"}
    assert owned == {"example.enabled": "global-overrides", "example.color": "default",
                     "agent.example.size": "default"}
    table = _invoke("components", "list").output
    for path, layer in owned.items():
        assert any(path in line and f"{layer} (plugin: example)" in line for line in table.splitlines()), (
            f"{path} is not shown as `{layer} (plugin: example)`:\n{table}")
    info = _invoke("plugins", "info", "example", "--json")
    assert info.exit_code == 0, info.output
    assert {s["path"]: s["layer"] for s in json.loads(info.stdout)["settings"]} == owned

    # --- zero core edits: no core module names anything the example defines -------------------------
    src = Path(localharness.__file__).parent
    files = sorted(src.rglob("*.py"))
    assert len(files) > 100, f"premise: the scan found only {len(files)} core files under {src}"
    knows = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                named = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                knows += [(path, n) for n in named if n == PKG or n.startswith(PKG + ".")]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in ("example", "/example") or any(n in node.value for n in EXAMPLE_NAMES):
                    knows.append((path, node.value[:60]))
    assert not knows, f"core knows the example exists: {knows}"
    # The template is named in core only as prose telling an author what to copy: the two init
    # READMEs (two lines each), `plugins list`'s nothing-installed hint (44-15) and the plugin API's
    # module docstring (44-04). None of them is code that reads, finds or loads it.
    pointers = Counter(p.relative_to(src).as_posix() for p in files
                       for line in p.read_text(encoding="utf-8").splitlines() if "plugin-template" in line)
    assert pointers == {"cli/init_cmd.py": 4, "cli/plugins_cmd.py": 1, "plugins/api.py": 1}, pointers
