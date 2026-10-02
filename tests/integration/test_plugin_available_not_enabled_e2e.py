"""Roadmap criterion 2 of the plugin substrate — available is not enabled — in ONE composed test.

Criterion 2, verbatim (ROADMAP.md, Phase 44):

    **Available is not enabled** (§8 "Third-party available, not enabled"): after install,
    `plugins list` shows the example plugin as *available* with its exact enable command and its
    module's import sentinel is NOT written; the startup banner prints the one-line hint;
    `plugins enable <name>` writes the GLOBAL layer's `overrides.yaml` through the atomic overlay
    writer (the user's `config.yaml` is byte-identical afterwards; `--workspace` is refused for a
    third-party `enabled`); on the next `start` the sentinel appears, its undeclared tool classifies
    `host: dangerous`, `ingest: untrusted`, `result_origin: untrusted`, gate "ask", and a declared
    `gate_family: read_only` is still asked about; a variant that raises on load is disabled for the
    session with a startup warning and a `doctor` line while the harness keeps running; an
    out-of-range `requires_localharness` is skipped with the reason shown in `plugins list` and
    `doctor`.

What each step proves, in order:
- ENAB-06: installed is not enabled. Listing reads entry-point metadata only (no sentinel, not in
  sys.modules); a start with it merely installed prints the hint and still imports nothing; it is
  imported exactly when the machine layer enables it and start runs again. And plugin code never
  loads from a project: a folder plugin planted in the workspace's own `plugins/` is never listed or
  imported.
- ENAB-04: the banner's one-line hint carries the exact enable command.
- SAFE-06: enabling a plugin you installed is a machine-level act — `--workspace` is refused, and an
  `enabled: true` in the project's config.yaml is dropped with a warning naming the key and the file.
  Its declared gate family is honoured only when it asks at least as often as an undeclared tool:
  `allow` is clamped (a startup warning names the plugin and the family), `read_only` is not a family
  at all and counts as undeclared.
- SAFE-01: a tool that declares nothing fails closed on all four axes, and the root capability floor
  keeps it from the root agent with a warning that names it.
- PAPI-11: a plugin whose import raises is disabled for the session and named in the startup warnings
  and by doctor, while the session starts, runs its other plugins and closes "complete".
- PAPI-12: an out-of-range `requires_localharness` is skipped, with its reason in the startup warnings,
  `plugins list` and doctor.

"Asked about" is evaluated in `guarded`, with an empty grant store, on the schema the session
registered (read the way the gate reads it). The session itself runs in the default `auto`, where the
owner's 2026-09-11 ruling lets an undeclared tool run unasked — pinned in 44-11's
tests/unit/test_third_party_gate_clamp.py — so "asked about" cannot be observed in an `auto` session
and SAFE-06 is stated relative to an undeclared tool, as its own sentence is.

REAL: every CLI command (the Typer app through CliRunner), `init --workspace` included; discovery
through the unmocked importlib.metadata and the real global `plugins/` folder scan; the resolver,
the plan, the lifecycle, the SAFE-06 clamp, the registry and the root capability floor; three real
sessions (`_start_async` with config_dir None, so the workspace is discovered). STUBBED:
`_stub_start_boundaries(..., real_plugins=True)` — the LLM probe, the tokenizer, the REPL loop (a
no-op: these sessions take no turns) — and the provider's base_url is the loopback discard port.

NOT proven here: a guarded session actually prompting (the verdict is asked directly, see above);
a tool call from a plugin that failed, which never registers anything.
"""
from __future__ import annotations

import asyncio
import json
import sys

import pytest
import yaml
from typer.testing import CliRunner

from localharness import resolved_version
from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import GateSettings, Verdict
from localharness.agent.verdict import GateContext, evaluate
from localharness.cli.app import app
from localharness.cli.slash_commands import set_plugin_rows
from tests.dispatch_support import isolate_discord_env
from tests.integration.test_plugin_fixture_registers_everything_e2e import (
    PKG,
    SENTINEL_ENV,
    example_modules,
    forget_example,
)
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _capture_start_console, _read_sessions, _stub_start_boundaries
from tests.unit.test_start_plugins import _record_loop
from tests.unit.test_workspace_state_landing import _boom, _drive, _hermetic

FOLDER_PKG = "localharness_folder_plugins"
HINT = ("i 1 plugin available, not enabled: example — run `localharness plugins enable example` "
        "to turn it on")
runner = CliRunner()

# Folder plugins in the machine's plugins/ dir, each a small __init__.py binding `plugin`.
_PLUGIN = '''\
{first}from localharness.plugins.api import Plugin, PluginManifest
from localharness.tools.base import Tool, ToolSchema


class _Tool(Tool):
    def info(self):
        return ToolSchema(name="{name}_tool", description="a folder plugin's tool", parameters={{}}{declares})

    async def _execute(self, **_):
        return self.ok("{name} ran")


class _Plugin(Plugin):
    """the {name} folder plugin"""

    manifest = PluginManifest(name="{name}", version="1", kind="tools"{requires})

    async def tools(self, ctx):
        return [_Tool()]


plugin = _Plugin
'''
_HONEST = ', ingest="none", host="safe", result_origin="trusted"'
FOLDER_PLUGINS = {  # enabled in this order: boom last, so every earlier resolve still succeeds
    "bare": {},                                                     # declares nothing
    "readonlyfam": {"declares": _HONEST + ', gate_family="read_only"'},  # not a family: undeclared
    "allowfam": {"declares": _HONEST + ', gate_family="allow"'},    # a family the clamp takes away
    "future": {"requires": ', requires_localharness=">=9"'},        # out of range: skipped
    "boom": {"first": 'raise RuntimeError("boom at import")\n'},    # raises on load
}


@pytest.fixture(autouse=True)
def _process_state():
    """The slash table and sys.modules are process-wide: what this test adds leaves with it."""
    before = {m: sys.modules[m] for m in {*example_modules(), *_folder_modules()}}
    yield
    set_plugin_rows(())
    for name in {*example_modules(), *_folder_modules()}:
        del sys.modules[name]
    sys.modules.update(before)


def _folder_modules() -> set[str]:
    return {m for m in sys.modules if m == FOLDER_PKG or m.startswith(FOLDER_PKG + ".")}


def _invoke(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"`localharness {' '.join(args)}` raised {result.exception!r}\n{result.output}")
    return result


def _listed(name: str) -> dict:
    result = _invoke("plugins", "list", "--json")
    assert result.exit_code == 0, result.output
    rows = {r["name"]: r for r in json.loads(result.stdout)}
    assert name in rows, f"{name} is not listed: {sorted(rows)}"
    return rows[name]


def test_available_is_not_enabled(tmp_path, monkeypatch, fake_home):
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, fake_home, home)
    isolate_discord_env(monkeypatch, tmp_path)  # no Discord env or token file reaches the dispatch plugin
    # Every extra "installed", so the Plugins: lines below hold whether or not discord.py is.
    from localharness.plugins import resolve as _resolve
    monkeypatch.setitem(_resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)
    _stub_start_boundaries(global_dir, monkeypatch, real_plugins=True)  # discovery stays REAL
    _offline_provider(global_dir)
    sentinel = tmp_path / "example-imported"
    monkeypatch.setenv(SENTINEL_ENV, str(sentinel))
    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)  # an in-project workspace never asks
    printed = _capture_start_console(monkeypatch)
    loops = _record_loop(monkeypatch)

    def start() -> tuple[list[str], str]:
        """One real session; (everything it printed, its summary line)."""
        n = len(printed)
        asyncio.run(_drive())  # _start_async(None, False, False, None): the workspace is discovered
        out = printed[n:]
        return out, next(line for line in out if "startup)" in line)

    # --- installed, listed as available, never imported ----------------------------------------
    forget_example(sentinel)
    row = _listed("example")
    assert (row["state_kind"], row["state"], row["enable_command"]) == (
        "available", "available — turn on: localharness plugins enable example",
        "localharness plugins enable example")
    assert not sentinel.exists() and not example_modules(), "listing an available plugin imported it"

    # --- start: the one-line hint, and still nothing imported ------------------------------------
    out, _ = start()
    assert HINT in out, f"no exact hint line in {out}"
    assert [line.split("Plugins: ")[1].split("[/]")[0] for line in out if "Plugins:" in line] == ["web, memory, dispatch, autoresearch"]  # web (46-02) and memory (47) are bundled and on by default; dispatch (49) and autoresearch (50) are bundled and on by default
    assert not sentinel.exists() and not example_modules(), "start imported an available plugin"
    rows = _read_sessions(global_dir)
    assert len(rows) == 1 and rows[0][3] == "complete"

    # --- a project cannot enable it: --workspace is refused, a project's `enabled` is dropped ----
    assert _invoke("init", "--workspace").exit_code == 0
    ws = (proj / ".localharness").resolve()
    refused = _invoke("plugins", "enable", "example", "--workspace")
    assert refused.exit_code == 2 and "machine-level" in refused.output, refused.output
    assert not (ws / "overrides.yaml").exists(), "a refused enable wrote the project's overrides"
    with (ws / "config.yaml").open("a", encoding="utf-8") as f:
        f.write("example:\n  enabled: true\n")
    sneaky_marker = tmp_path / "sneaky-imported"  # plugin code planted in the PROJECT's plugins/
    (ws / "plugins" / "sneaky").mkdir()
    (ws / "plugins" / "sneaky" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sneaky_marker)!r}).write_text('imported')\n")
    forget_example(sentinel)
    out, summary = start()
    assert [line.split("Plugins: ")[1].split("[/]")[0] for line in out if "Plugins:" in line] == ["web, memory, dispatch, autoresearch"] and HINT in out, out  # web (46-02) and memory (47) are bundled and on by default; dispatch (49) and autoresearch (50) are bundled and on by default
    assert f"ignoring example.enabled in {ws / 'config.yaml'}: only the global config may set it" in summary
    assert not sentinel.exists() and not example_modules(), "a project's `enabled` loaded the plugin"
    assert "sneaky" not in {r["name"] for r in json.loads(_invoke("plugins", "list", "--json").stdout)}
    rows = _read_sessions(ws)
    assert len(rows) == 1 and rows[0][3] == "complete"

    # --- the machine enables it: overrides.yaml only, config.yaml byte-identical ------------------
    config_bytes = (global_dir / "config.yaml").read_bytes()
    enabled = _invoke("plugins", "enable", "example")
    assert enabled.exit_code == 0, enabled.output
    assert (global_dir / "config.yaml").read_bytes() == config_bytes, "enable rewrote config.yaml"
    assert yaml.safe_load((global_dir / "overrides.yaml").read_text()) == {"example": {"enabled": True}}

    # --- five folder plugins in the machine's plugins/, each enabled with the shipped command ----
    for name, parts in FOLDER_PLUGINS.items():
        folder = global_dir / "plugins" / name
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text(_PLUGIN.format(
            name=name, first=parts.get("first", ""), declares=parts.get("declares", ""),
            requires=parts.get("requires", "")))
        result = _invoke("plugins", "enable", name)
        assert result.exit_code == 0, result.output
    assert yaml.safe_load((global_dir / "overrides.yaml").read_text()) == {
        n: {"enabled": True} for n in ("example", *FOLDER_PLUGINS)}

    # --- the next start: imported now; failures named while the session runs; still asked about --
    forget_example(sentinel)
    out, summary = start()
    rows = _read_sessions(ws)
    assert len(rows) == 2 and rows[-1][3] == "complete", "the harness did not keep running"
    assert "plugin boom: could not be imported: RuntimeError: boom at import" in summary, summary
    assert f"plugin future: requires localharness >=9, this is {resolved_version()}" in summary, summary
    assert sentinel.exists() and PKG in sys.modules, "enabled, yet the next start did not import it"
    assert not sneaky_marker.exists(), "plugin code was loaded from a project folder"
    loaded = next(line for line in out if "Plugins: " in line)
    assert set(loaded.split("Plugins: ")[1].split("[/]")[0].split(", ")) == {
        "web", "memory", "dispatch", "autoresearch", "example", "bare", "readonlyfam", "allowfam"}, loaded  # web (46-02) and memory (47) are bundled and on by default; dispatch (49) and autoresearch (50) are bundled and on by default
    registry, cfg = loops[-1]["tool_registry"], loops[-1]["config"]
    guarded = GateContext(boundary=ws, workspace=ws, grants=lambda *_: None, mode="guarded")
    for name in ("bare_tool", "readonlyfam_tool", "allowfam_tool"):
        schema = registry.schema_of(name)  # the live schema, as the gate reads it on every call
        assert schema is not None, f"premise: {name} is not registered (a missing tool also asks)"
        verdict = evaluate(name, {}, tool_meta_from_schema(schema), guarded, GateSettings()).verdict
        assert verdict is Verdict.ASK, f"{name} is {verdict} in guarded with no grants — not asked about"
    bare = registry.schema_of("bare_tool")
    assert (bare.host, bare.ingest, bare.result_origin, bare.gate_family, bare.source_plugin) == (
        "dangerous", "untrusted", "untrusted", None, "bare"), "an undeclared tool did not fail closed"
    assert registry.schema_of("readonlyfam_tool").gate_family is None  # not a family: undeclared
    allow = registry.schema_of("allowfam_tool")
    assert (allow.gate_family, allow.ingest, allow.host, allow.source_plugin) == (
        None, "none", "safe", "allowfam"), "the clamp must take the family and honour the rest"
    assert ("plugin allowfam: tool 'allowfam_tool' declares gate family 'allow'; a plugin you "
            "installed may only declare families that ask at least as often as an undeclared tool"
            in summary), summary
    root = registry.get_tools_for_agent(cfg.name, cfg.division or "", cfg.tools)
    assert "bare_tool" not in root and {"readonlyfam_tool", "allowfam_tool", "example_swatch"} <= set(root)
    assert ("capability floor: the root agent does not hold bare_tool (plugin bare) — it declares no "
            "ingest, which counts as ingest: untrusted" in summary), summary
    assert registry.schema_of("boom_tool") is None and registry.schema_of("future_tool") is None

    # --- plugins list and doctor name the broken and the too-new ------------------------------------
    boom, future = _listed("boom"), _listed("future")
    assert (boom["state_kind"], boom["state"]) == (
        "failed", "failed — could not be imported: RuntimeError: boom at import")
    too_new = f"requires localharness >=9, this is {resolved_version()}"
    assert (future["state_kind"], future["state"]) == ("skipped", f"skipped — {too_new}")
    doctor = _invoke("doctor").output
    assert "✗ boom: failed — could not be imported: RuntimeError: boom at import" in doctor, doctor
    assert f"⚠ future: skipped — {too_new}" in doctor, doctor
    on = next(line for line in doctor.splitlines() if line.startswith("✓ Plugins: "))
    assert "example" in on.removeprefix("✓ Plugins: ").split(", "), on
    assert "✓ example: swatches render in #4a90d9" in doctor, doctor
    assert not sneaky_marker.exists(), "doctor loaded plugin code from a project folder"
