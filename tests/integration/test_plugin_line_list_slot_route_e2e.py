"""Roadmap criterion 5 of the plugin substrate — the line, the one list, the empty slot, the route — in
ONE composed test.

Criterion 5, verbatim (ROADMAP.md, Phase 44):

    **The line, the one list, the empty slot, the route** (§8 "Import direction", "Artifact route
    security"): the CORE-02 test fails on an injected `from localharness.memory import …` in
    `agent/loop.py` and passes without it, classifies `tools/builtin/generate_image_tool.py` and
    `memory_tools.py` as plugin files, and carries the named burn-down list (D5); `BUILTIN_PLUGINS`
    exists and the banner, `plugins list`, `doctor`, `components` and the loader read it — no second
    list anywhere; a load plan with two `kind == "memory"` plugins is refused, and with none the slot
    is empty (no memory section from the slot, guardrails present); a workspace-layer value for a
    global-only plugin field is dropped with a warning and the global value stands; on the generic
    artifact route an unauthenticated GET is refused, an off plugin 404s without touching any root, a
    plugin returning a non-core root has its route refused, a mime off the allowlist is 415, and a
    valid id carries the immutable-cache header.

Criteria 1 and 2 prove the substrate for a stranger's plugin; this one proves the structure the four
conversions will fill: where the line is, which list they append to, the seat memory will take, and
the one route every producing plugin is served through. Five parts, in the criterion's order.

1. The line. REAL: CORE-02's own checker (tests/unit/test_import_direction.py — imported, one checker,
   not a copy) over the real source tree; the injected import is text appended in memory, never
   written to agent/loop.py.
2. The one list. REAL: an AST scan of every core file; then `BUILTIN_PLUGINS` swapped to one bundled
   plugin (the one edit a conversion makes) and every reader asked through its shipped surface — the
   loader (`ConfigLoader.load_harness`), a real `_start_async` session's banner and registry,
   `plugins list`, `doctor`, `components list`, `--help` and the mounted command (the Typer app via
   CliRunner). STUBBED, as in criteria 1-3: the model probe, the tokenizer and the REPL's read loop
   (`_stub_start_boundaries`), the provider pointed at the loopback discard port (every dial is
   recorded and asserted); entry-point discovery is reduced to its real folder scan, so the venv's
   installed example plugin (criteria 1 and 2's subject) does not enter this test.
3. The empty slot. REAL: resolve() → the pure load plan → the lifecycle → the MemorySlot → AgentLoop's
   prompt assembly, with a global GUARDRAILS.md. STUBBED: the model (it records each system prompt).
4. Global-only. REAL: resolve() over a real ConfigLoader holding the global layer and a workspace
   layer — `merge_plugin_layers` does the narrowing.
5. The route. REAL: resolve() → the lifecycle deciding which artifact roots to accept, the web server's
   routes over ASGI (no socket) with exactly those roots, files written by core's `write_artifact`.

NOT proven here: the phone rendering an artifact (`Observation.artifact` is the image conversion's,
Phase 45); a real memory plugin in the slot (Phase 47); `--channel` validation reading the list
(CORE-03's last reader, the web conversion). Each conversion edits part 1's per-plugin line when it
empties its plugin's burn-down entries.
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import sys
import textwrap

import pytest
from pydantic import BaseModel, Field

from localharness.channels.web import auth
from localharness.channels.web import server as server_mod
from localharness.cli.slash_commands import set_plugin_rows
from localharness.cli.theme import entity
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.core.artifacts import artifact_root, mint_artifact_id, write_artifact
from localharness.core.bus import EventBus
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, CliDescriptor, Plugin, PluginManifest, PluginPaths
from localharness.plugins.lifecycle import start_plugins
from localharness.plugins.resolve import resolve
from localharness.tools.base import Tool, ToolResult, ToolSchema
from localharness.tools.registry import ToolRegistry
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_plugin_fixture_registers_everything_e2e import _invoke, _record_dials
from tests.integration.test_workspace_cli_surface_e2e import _DISCARD_URL, _offline_provider
from tests.unit.channels.test_web_artifacts import GIF, IMMUTABLE, PNG
from tests.unit.channels.test_web_server import BEARER, _stack
from tests.unit.test_import_direction import BURN_DOWN, SRC, classify, edges_of, scan, violations
from tests.unit.test_memory_slot import _loop, _memory_plugin
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries
from tests.unit.test_start_plugins import _record_loop
from tests.unit.test_workspace_state_landing import _boom, _hermetic

_REAL_DISCOVER = discovery.discover
CLI_MODULE = "lh_criterion5_cli"
MEMORY_SECTION = "## Division Context\nD"  # what the test memory plugin's context() would add


class _ProbeTool(Tool):
    """Declares all four axes, so the root agent keeps it."""

    def info(self) -> ToolSchema:
        return ToolSchema(name="fakebundled_probe", description="Probe.", parameters={},
                          ingest="none", host="safe", result_origin="trusted")

    async def _execute(self, **_) -> ToolResult:
        return self.ok("probed")


class _FakeSettings(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)  # an endpoint: machine-level only (ENAB-02)
    label: str = ""                                        # an ordinary setting: layered


class FakeBundled(Plugin):
    """Stands in for the first bundled conversion."""

    manifest = PluginManifest(name="fakebundled", version="0.1.0", kind="tools", cli=(
        CliDescriptor(name="fakecmd", help="Run the fake bundled command.", target=f"{CLI_MODULE}:app"),))
    ConfigModel = _FakeSettings

    async def tools(self, ctx):
        return [_ProbeTool()]


class Kept(Plugin):
    """Writes its artifacts where core says."""

    manifest = PluginManifest(name="kept", version="0.1.0", kind="tools")
    wants_artifacts = True


class Strays(Plugin):
    """Names an artifact root of its own."""

    manifest = PluginManifest(name="strays", version="0.1.0", kind="tools")
    wants_artifacts = True

    def artifact_root(self, ctx):
        return ctx.paths.state_dir / "elsewhere"


class Dormant(Plugin):
    """A bundled plugin that is off."""

    manifest = PluginManifest(name="dormant", version="0.1.0", kind="tools", enabled_by_default=False)
    wants_artifacts = True


@pytest.fixture(autouse=True)
def _process_state():
    """The slash table and sys.modules are process-wide: what this test adds leaves with it."""
    sys.modules.pop(CLI_MODULE, None)
    yield
    set_plugin_rows(())
    sys.modules.pop(CLI_MODULE, None)


def _started(resolution, state_dir):
    return asyncio.run(start_plugins(resolution, bus=EventBus(), registry=ToolRegistry(), hooks=None,
                                     llm=None, paths=PluginPaths(state_dir, None, state_dir)))


def test_the_line_the_one_list_the_empty_slot_the_route(tmp_path, monkeypatch, fake_home):
    from localharness.cli.start_cmd import _start_async

    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, fake_home, home)
    _stub_start_boundaries(global_dir, monkeypatch, real_plugins=True)
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])  # no entry points
    _offline_provider(global_dir)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    with (global_dir / "config.yaml").open("a", encoding="utf-8") as cfg:
        cfg.write("fakebundled:\n  url: http://global.example\n")
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / f"{CLI_MODULE}.py").write_text(textwrap.dedent('''\
        import typer

        app = typer.Typer()


        @app.command()
        def run():
            """Say that the bundled plugin's command ran."""
            typer.echo("fakebundled command ran")
        '''), encoding="utf-8")
    monkeypatch.syspath_prepend(str(mods))
    (home / "work").mkdir()
    monkeypatch.chdir(home / "work")  # under $HOME, in no project: no workspace layer
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)  # nothing here may ask

    # --- 1. the line: CORE-02's checker, on the real tree and on one injected import ----------------
    assert violations(scan()) == set(), "a core module imports a plugin module off the burn-down list"
    loop_text = (SRC / "agent" / "loop.py").read_text(encoding="utf-8")
    assert edges_of("agent/loop.py", loop_text) == set()
    injected = edges_of("agent/loop.py", loop_text + "\nfrom localharness.memory import MemoryStore\n")
    assert violations(injected) == {("agent/loop.py", "memory/__init__.py")}
    assert classify("tools/builtin/generate_image_tool.py") == "plugin"
    assert classify("tools/builtin/memory_tools.py") == "plugin"
    # web left this map in 46-04: its last burn-down entries (doctor's listener lines) are gone.
    # memory left it in 48-05: the bench builds memory through the plugin lifecycle.
    # dispatch left it in 49-06: Discord is the dispatch plugin's adapter, built by the generic branch.
    assert not any("memory" in pair[1] for pair in BURN_DOWN)
    assert not any("discord" in pair[1] or "dispatch" in pair[1] for pair in BURN_DOWN)
    owners = {"autoresearch": ("autoresearch_cmd", "experiment_cmd", "propose_cmd", "report_cmd")}
    named = {plugin: {pair for pair in BURN_DOWN if any(w in pair[1] for w in words)}
             for plugin, words in owners.items()}
    assert all(named.values()), f"a plugin still wired the old way has no burn-down entry: {named}"
    assert set().union(*named.values()) == BURN_DOWN, "a burn-down entry names no plugin"

    # --- 2. the one list: assigned once, read once, and every reader follows it ---------------------
    touches = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if getattr(node, "id", getattr(node, "attr", None)) == "BUILTIN_PLUGINS":
                touches.append((path.relative_to(SRC).as_posix(),
                                "assign" if isinstance(node.ctx, ast.Store) else "read"))
            elif isinstance(node, ast.alias) and node.name == "BUILTIN_PLUGINS":
                touches.append((path.relative_to(SRC).as_posix(), "import"))
    assert sorted(touches) == [("plugins/builtin.py", "assign"), ("plugins/builtin.py", "read")], (
        f"BUILTIN_PLUGINS must be assigned once and read only by bundled_plugins(): {touches}")
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (FakeBundled,))  # the one edit a conversion makes

    # The loader: the plugin's section is routed out of core validation — because of the list alone.
    ConfigLoader(config_dir=global_dir).load_harness()
    with monkeypatch.context() as m:
        m.setattr(builtin, "BUILTIN_PLUGINS", ())
        with pytest.raises(ConfigValidationError, match="fakebundled.*no installed plugin is named"):
            ConfigLoader(config_dir=global_dir).load_harness()

    # The banner and the registry: one real session.
    printed = _capture_start_console(monkeypatch)
    loops = _record_loop(monkeypatch)
    dialed = _record_dials(monkeypatch)
    asyncio.run(_start_async(None, False, False, None))
    i = next(n for n, line in enumerate(printed) if "startup)" in line)
    assert entity("tool", "1 plugin") in printed[i], printed[i]
    assert printed[i + 1] == "  " + entity("tool", "Plugins: fakebundled"), printed[i:i + 2]
    (session,) = loops
    assert session["tool_registry"].schema_of("fakebundled_probe").source_plugin == "fakebundled"
    assert {a[:2] for a in dialed if isinstance(a, tuple)} <= {("127.0.0.1", 9)}, (
        f"the session dialed {dialed}, not only {_DISCARD_URL}")

    # plugins list, doctor, components, --help and the command it mounts.
    listed = _invoke("plugins", "list", "--json")
    assert listed.exit_code == 0, listed.output
    assert [(r["name"], r["from"], r["state_kind"], r["what_it_does"]) for r in json.loads(listed.stdout)] == [
        ("fakebundled", "built in", "on", "Stands in for the first bundled conversion.")]
    doctor = _invoke("doctor").output
    assert "✓ Plugins: fakebundled" in doctor, doctor
    rows = _invoke("components", "list", "--json")
    assert rows.exit_code == 0, rows.output
    assert {r["path"]: (r["layer"], r["current_value"]) for r in json.loads(rows.stdout)
            if r["plugin"] == "fakebundled"} == {
        "fakebundled.enabled": ("default", True), "fakebundled.url": ("global-config", "http://global.example"),
        "fakebundled.label": ("default", "")}
    helped = _invoke("--help")
    assert re.search(r"│ fakecmd\s+Run the fake bundled command\.", helped.output), helped.output
    assert CLI_MODULE not in sys.modules, "--help imported the command's module"
    ran = _invoke("fakecmd")
    assert ran.exit_code == 0 and "fakebundled command ran" in ran.output, ran.output

    # --- 3. the empty slot: two memory plugins are both refused; no occupant, no slot section ------
    one_mem, other_mem = _memory_plugin("recall-a"), _memory_plugin("recall-b")
    with monkeypatch.context() as m:
        m.setattr(builtin, "BUILTIN_PLUGINS", (one_mem, other_mem))
        two = resolve(ConfigLoader(config_dir=global_dir))
        m.setattr(builtin, "BUILTIN_PLUGINS", (one_mem,))
        single = resolve(ConfigLoader(config_dir=global_dir))
    assert two.plan.memory_occupant is None and two.plan.order == ()
    assert [(e.name, e.state) for e in two.plan.entries] == [("recall-a", "refused"), ("recall-b", "refused")]
    assert "each claim the memory slot" in two.plan.entry("recall-a").reason
    empty = _started(two, tmp_path / "state-two").slot
    occupied = _started(single, tmp_path / "state-one").slot  # premise: an occupant IS seated and heard
    assert (empty.occupied, occupied.occupied, occupied.occupant_name) == (False, True, "recall-a")

    async def first_prompt(where: str, slot) -> str:
        loop, prompts = _loop(tmp_path / where, memory_slot=slot)
        await loop.run_turn("hello")
        return prompts[0]

    heard = asyncio.run(first_prompt("loop-occupied", occupied))
    assert "\n\n## Guardrails\nRULES-G\n\n" + MEMORY_SECTION in heard, heard  # the absence below can see it
    for where, slot in (("loop-empty", empty), ("loop-none", None)):
        prompt = asyncio.run(first_prompt(where, slot))
        assert "\n\n## Guardrails\nRULES-G" in prompt and MEMORY_SECTION not in prompt, (where, prompt)
    assert asyncio.run(first_prompt("loop-a", empty)) == asyncio.run(first_prompt("loop-b", None))

    # --- 4. global-only: a workspace value at a machine-level path is dropped, the global stands ----
    ws = tmp_path / "project" / ".localharness"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text("fakebundled:\n  url: http://project.example\n  label: project\n",
                                    encoding="utf-8")
    narrowed = resolve(ConfigLoader(config_dir=global_dir, local_config_dir=ws))
    settings = narrowed.settings["fakebundled"].config
    assert (settings.url, settings.label) == ("http://global.example", "project")
    assert (f"ignoring fakebundled.url in {ws / 'config.yaml'}: only the global config may set it"
            in narrowed.warnings), narrowed.warnings

    # --- 5. the route: served only from the roots the lifecycle accepted ---------------------------
    with monkeypatch.context() as m:
        m.setattr(builtin, "BUILTIN_PLUGINS", (Kept, Strays, Dormant))
        shipped = resolve(ConfigLoader(config_dir=global_dir))
    assert [(e.name, e.state) for e in shipped.plan.entries] == [
        ("kept", "on"), ("strays", "on"), ("dormant", "off")]
    state = tmp_path / "state"
    lifecycle = _started(shipped, state)
    assert lifecycle.loaded_names == ["kept", "strays"], "a refused root must cost only its serving"
    assert lifecycle.artifact_roots == {"kept": state / "artifacts" / "kept"}
    (refused,) = [w for w in lifecycle.warnings if w.startswith("plugin strays:")]
    assert str(state / "elsewhere") in refused and str(state / "artifacts" / "strays") in refused, refused
    # Files at core's own path for the refused and the off plugin: only the refusal can keep them unserved.
    strayed = write_artifact(artifact_root(state, "strays"), "strays", PNG, "image/png")
    off = write_artifact(artifact_root(state, "dormant"), "dormant", PNG, "image/png")
    root = lifecycle.artifact_roots["kept"]
    served = write_artifact(root, "kept", PNG, "image/png")
    gif_id = mint_artifact_id()
    (root / f"{gif_id}.gif").write_bytes(GIF)

    async def route():
        (tmp_path / "web").mkdir()
        _, channel, _, client = await _stack(tmp_path / "web", runtime={"artifact_roots": lifecycle.artifact_roots})
        try:
            got = {"unauthenticated": await client.get(f"/api/artifacts/kept/{served.id}"),
                   "refused root": await client.get(f"/api/artifacts/strays/{strayed.id}", headers=BEARER),
                   "gif": await client.get(f"/api/artifacts/kept/{gif_id}", headers=BEARER),
                   "valid": await client.get(f"/api/artifacts/kept/{served.id}", headers=BEARER)}

            def touched(*_a, **_k):
                raise AssertionError("the route touched a root for a plugin that is off")

            with monkeypatch.context() as spies:
                spies.setattr(server_mod, "_find_artifact", touched)
                spies.setattr(auth, "confine", touched)
                got["off"] = await client.get(f"/api/artifacts/dormant/{off.id}", headers=BEARER)
            return got
        finally:
            await client.aclose()
            await channel.stop()

    got = asyncio.run(route())
    assert {k: r.status_code for k, r in got.items()} == {
        "unauthenticated": 401, "refused root": 404, "gif": 415, "valid": 200, "off": 404}
    valid = got["valid"]
    assert (valid.headers["content-type"], valid.headers["cache-control"]) == ("image/png", IMMUTABLE)
    assert valid.content == PNG
