"""MEMP-06, composed: both memory enable keys at every layer, on real surfaces (roadmap criterion 3).

`memory.enabled` is the memory plugin's key; the deprecated `org.memory_enabled` still works, folded
layer by layer (47-03), with ONE deprecation line per start naming the file(s) whose legacy key
decides — or would decide — the outcome. Every test here drives a real surface: `_start_async` for a
session (only the external boundaries stubbed), `CliRunner` for `plugins list`, `doctor`,
`components list/set` and `init`. The workspace layer is a real one: a `.localharness/` in a project
with a `.git` marker, discovered by the real walk under a fake `$HOME` (LOCALHARNESS_HOME cleared,
so discovery is not switched off — test_workspace_state_landing._hermetic).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from tests.unit.test_memory_config_surface import AGENT_MEMORY_PATHS
from tests.unit.test_recall_scope_wiring import _record_memory_plugins
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries
from tests.unit.test_start_plugins import _record_loop
from tests.unit.test_workspace_state_landing import _boom, _hermetic

runner = CliRunner()
MEMORY_TOOLS = {"memory_search", "memory_get", "remember"}
DEPRECATION = "org.memory_enabled is deprecated — use memory.enabled (read from "
ORG_OFF = "org:\n  memory_enabled: false\n"
MEM_OFF = "memory:\n  enabled: false\n"
MEM_ON = "memory:\n  enabled: true\n"


def _layers(tmp_path: Path, monkeypatch, fake_home, *, global_extra: str = "",
            workspace: str | None = None) -> tuple[Path, Path]:
    """A global layer (the stub's known-good config + `global_extra`) under a fake $HOME, and CWD
    in a project; `workspace` is that project's `.localharness/config.yaml` (None: no layer)."""
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, fake_home, home)
    _stub_start_boundaries(global_dir, monkeypatch)
    if global_extra:
        with (global_dir / "config.yaml").open("a", encoding="utf-8") as f:
            f.write(global_extra)
    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)  # an in-project workspace never asks
    ws = proj / ".localharness"
    if workspace is not None:
        ws.mkdir()
        (ws / "config.yaml").write_text(workspace, encoding="utf-8")
    return global_dir, ws


def _start(monkeypatch) -> dict[str, Any]:
    """One real `start` (config_dir=None, so the workspace walk runs). Returns what the root loop
    held while the session was live and every line start printed."""
    import asyncio

    from localharness.cli.start_cmd import _start_async

    printed = _capture_start_console(monkeypatch)
    loops = _record_loop(monkeypatch)
    started = _record_memory_plugins(monkeypatch)
    seen: dict[str, Any] = {}

    async def _repl(self):
        root = loops[0]
        slot = root["memory_slot"]
        seen["occupied"] = slot.occupied
        seen["occupant"] = slot.occupant_name if slot.occupied else None
        seen["tools"] = set(root["tool_registry"]._tools["global"])
        from localharness.cli.slash_commands import find_row
        row = find_row("/memory")
        seen["slash"] = row.plugin if row is not None else None
        # The running memory plugin's router — what its /memory row and the slot read through.
        seen["router"] = started[0].router if slot.occupied else None

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", _repl)
    asyncio.run(_start_async(None, False, False, None))
    seen["printed"] = printed
    seen["deprecations"] = sum(line.count(DEPRECATION) for line in printed)
    return seen


def _deprecation_files(printed: list[str]) -> str:
    (line,) = [p for p in printed if DEPRECATION in p]
    return line.split(DEPRECATION, 1)[1]


def _assert_off(seen: dict) -> None:
    assert seen["occupied"] is False, "memory off, but the memory slot is occupied"
    assert seen["slash"] is None, "memory off, but /memory is in the slash table"
    assert not (MEMORY_TOOLS & seen["tools"]), f"memory off, but the root holds {MEMORY_TOOLS & seen['tools']}"


def _assert_on(seen: dict) -> None:
    assert seen["occupied"] is True and seen["occupant"] == "memory"
    assert seen["slash"] == "memory", "memory on, but /memory is not the memory plugin's row"
    assert MEMORY_TOOLS <= seen["tools"]


# ------------------------------------------------------------------ a real start, every layer


def test_workspace_only_org_flag_turns_memory_off(tmp_path, monkeypatch, fake_home):
    """MEMP-06's proof: the legacy key in the WORKSPACE layer alone turns memory off for that
    project's session, and the start says so exactly once, naming that file."""
    global_dir, ws = _layers(tmp_path, monkeypatch, fake_home, workspace=ORG_OFF)
    seen = _start(monkeypatch)
    _assert_off(seen)
    assert seen["deprecations"] == 1, seen["printed"]
    named = _deprecation_files(seen["printed"])
    assert str(ws / "config.yaml") in named and str(global_dir) not in named, named


def test_memory_enabled_false_is_silent(tmp_path, monkeypatch, fake_home):
    _layers(tmp_path, monkeypatch, fake_home, workspace=MEM_OFF)
    seen = _start(monkeypatch)
    _assert_off(seen)
    assert seen["deprecations"] == 0, seen["printed"]


def test_workspace_memory_enabled_true_overrides_global_org_false(tmp_path, monkeypatch, fake_home):
    """The layers stay separate after the fold: a workspace `memory.enabled: true` wins over the
    machine's legacy `false` — and that legacy key would have decided, so it is named."""
    global_dir, ws = _layers(tmp_path, monkeypatch, fake_home, global_extra=ORG_OFF, workspace=MEM_ON)
    seen = _start(monkeypatch)
    _assert_on(seen)
    assert seen["deprecations"] == 1, seen["printed"]
    named = _deprecation_files(seen["printed"])
    assert str(global_dir / "config.yaml") in named and str(ws) not in named, named


def test_both_keys_in_one_layer_memory_enabled_wins(tmp_path, monkeypatch, fake_home):
    """In one layer the new key wins and the legacy one is inert — so nothing to warn about."""
    _layers(tmp_path, monkeypatch, fake_home, workspace=ORG_OFF + MEM_ON)
    seen = _start(monkeypatch)
    _assert_on(seen)
    assert seen["deprecations"] == 0, seen["printed"]


def test_legacy_key_in_two_layers_one_line(tmp_path, monkeypatch, fake_home):
    global_dir, ws = _layers(tmp_path, monkeypatch, fake_home, global_extra=ORG_OFF, workspace=ORG_OFF)
    seen = _start(monkeypatch)
    _assert_off(seen)
    assert seen["deprecations"] == 1, seen["printed"]
    named = _deprecation_files(seen["printed"])
    assert str(global_dir / "config.yaml") in named and str(ws / "config.yaml") in named, named


def test_memory_on_by_default_and_silent(tmp_path, monkeypatch, fake_home):
    """The control: neither key anywhere -> memory on, no line."""
    _layers(tmp_path, monkeypatch, fake_home, workspace="")
    seen = _start(monkeypatch)
    _assert_on(seen)
    assert seen["deprecations"] == 0, seen["printed"]


# ------------------------------------------------------------------ plugins list, doctor


@pytest.mark.parametrize("workspace", [ORG_OFF, MEM_OFF], ids=["org.memory_enabled", "memory.enabled"])
def test_plugins_list_and_doctor_show_off_for_either_key(tmp_path, monkeypatch, fake_home, workspace):
    _layers(tmp_path, monkeypatch, fake_home, workspace=workspace)
    listed = runner.invoke(app, ["plugins", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    (row,) = [e for e in json.loads(listed.stdout) if e["name"] == "memory"]
    assert row["state_kind"] == "off", row
    assert row["enable_command"] == "localharness plugins enable memory", row

    doc = runner.invoke(app, ["doctor"])
    out = doc.output
    off_rows = [line for line in out.splitlines() if "memory: off" in line]
    assert len(off_rows) == 1 and "localharness plugins enable memory" in off_rows[0], out
    assert "✗" not in off_rows[0], f"memory off is shown as a failure: {off_rows[0]}"
    assert "memory-db" not in out and "memory-embedding" not in out, "an off plugin's checks ran"
    # off is not failed: the same project with memory ON and its checks passing counts the same
    # number of issues — the off row adds none.
    (Path.cwd() / ".localharness" / "config.yaml").write_text("", encoding="utf-8")
    on = runner.invoke(app, ["doctor"])
    assert "memory-embedding" in on.output, on.output
    assert doc.exit_code == on.exit_code, (out, on.output)
    assert _issues(out) == _issues(on.output), (out, on.output)


def _issues(out: str) -> str:
    return next((line for line in out.splitlines() if "issue(s) found" in line or "All checks" in line), "")


# ------------------------------------------------------------------ components list / set


def test_agent_memory_rows_are_plugin_rows(tmp_path, monkeypatch, fake_home):
    _layers(tmp_path, monkeypatch, fake_home)
    monkeypatch.setenv("COLUMNS", "400")
    listed = runner.invoke(app, ["components", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    rows = [r for r in json.loads(listed.stdout) if r["path"].startswith("agent.memory.")]
    assert {r["path"] for r in rows} == AGENT_MEMORY_PATHS
    assert all(r["plugin"] == "memory" for r in rows), [r for r in rows if r["plugin"] != "memory"]
    table = runner.invoke(app, ["components", "list"])
    assert table.exit_code == 0, table.output
    # The human table: each row's cells between the box bars; the last cell is the layer.
    cells = [[c.strip() for c in line.strip().strip("│").split("│")] for line in table.output.splitlines()
             if line.count("│") > 2]
    rows = {c[0]: c[-1] for c in cells if c[0].startswith("agent.memory.")}
    assert set(rows) == AGENT_MEMORY_PATHS, table.output
    for path, layer in rows.items():
        assert layer.endswith("(plugin: memory)"), (path, layer)


def test_components_set_recall_scope(tmp_path, monkeypatch, fake_home):
    """`components set` writes the plugin's setting and the next session's router reads it; a value
    the plugin's model refuses is refused and nothing changes."""
    _layers(tmp_path, monkeypatch, fake_home)
    ok = runner.invoke(app, ["components", "set", "agent.memory.recall_scope", "both"])
    assert ok.exit_code == 0, ok.output
    bad = runner.invoke(app, ["components", "set", "agent.memory.recall_scope", "nonsense"])
    assert bad.exit_code != 0, bad.output
    assert "recall_scope" in bad.output, bad.output
    seen = _start(monkeypatch)
    _assert_on(seen)
    assert seen["router"]._scope == "both"


# ------------------------------------------------------------------ init writes the canonical key


@pytest.mark.parametrize("memory_on", [True, False])
def test_a_fresh_init_starts_with_no_deprecation_line(tmp_path, monkeypatch, fake_home, memory_on):
    """gap 3 / RESEARCH P6: older init wrote `org.memory_enabled: true` into every config. A fresh
    init writes the canonical key (or nothing), and the first start on it is silent."""
    import localharness.cli.init_cmd as init_cmd
    from tests.unit.test_init_cmd import _make_capability_result, _make_detector_result

    global_dir, _ = _layers(tmp_path, monkeypatch, fake_home)
    stub_cfg = (global_dir / "config.yaml").read_text()
    client = MagicMock()
    client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    monkeypatch.setattr(init_cmd, "detect_provider", AsyncMock(return_value=_make_detector_result()))
    monkeypatch.setattr(init_cmd, "LLMClient", MagicMock(return_value=client))
    monkeypatch.setattr(init_cmd, "_detect_max_model_len", lambda *_: None)
    monkeypatch.setattr(init_cmd, "_identify_endpoint_provider", lambda *_: "unknown")
    fake_sys = MagicMock()
    fake_sys.stdin.isatty.return_value = True
    monkeypatch.setattr(init_cmd, "sys", fake_sys)
    fake_confirm = MagicMock()
    fake_confirm.ask.side_effect = [True, memory_on]
    monkeypatch.setattr(init_cmd, "Confirm", fake_confirm)
    result = runner.invoke(app, ["init", "--config-dir", str(global_dir), "--force"])
    assert result.exit_code == 0, result.output
    written = (global_dir / "config.yaml").read_text()
    assert written != stub_cfg and "memory_enabled" not in written, written
    assert ("memory:\n  enabled: false\n" in written) is (not memory_on), written

    seen = _start(monkeypatch)
    assert seen["deprecations"] == 0, seen["printed"]
    (_assert_on if memory_on else _assert_off)(seen)
