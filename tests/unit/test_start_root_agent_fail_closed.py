"""QA-16: `start` refuses a root agent whose config will not load — never a bare fallback session.

Before this, `start` wrapped the root agent's load in `except Exception:` and built
`AgentConfig(name, role, model)` without a word. Everything else the agent file said was dropped:
its permission mode (a `read-only` agent ran `auto`), its deny patterns, `workspace_root`, budget,
MCP servers, memory and context settings. The commonest trigger since plugins is a removed plugin's
`agent.<name>` entry left in overrides.yaml, which fails every agent's validation; a typo in the
agent file takes the same path.

Every drive below is the REAL `_start_async`, offline: the provider points at the discard port and
only the external boundaries are stubbed (`_stub_start_boundaries`). The REPL loop is replaced by
what a person does first — answer the trust question if one comes, then type `/mode` — so a session
that should not exist is caught by what it would have shown.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from localharness.cli.app import app
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _stub_start_boundaries
from tests.unit.test_workspace_state_landing import _drive, _hermetic

ROOT = "orchestrator"
READ_ONLY_AGENT = (
    "name: orchestrator\nrole: General-purpose assistant\nmodel: inherit\n"
    "permissions:\n  mode: read-only\n  deny_patterns: ['bash(sentinel-agent-deny *)']\n"
)
LEFTOVER = "agent:\n  example:\n    size: 16\n"
"""A removed plugin's agent-level settings: no installed plugin is named `example`."""


def session_home(tmp_path: Path, monkeypatch, fake_home, *, agent: str | None) -> Path:
    """A fake `$HOME` whose global layer holds a known-good config.yaml aimed at the discard port,
    plus the root agent file when `agent` is given (None: a fresh install, `start` mints it). CWD is
    a checkout with no `.localharness`, so no workspace layer applies. Returns the global dir."""
    global_dir = _hermetic(monkeypatch, fake_home, tmp_path / "home")
    _stub_start_boundaries(global_dir, monkeypatch)
    _offline_provider(global_dir)
    proj = tmp_path / "home" / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    if agent is not None:
        (global_dir / "agents").mkdir()
        (global_dir / "agents" / f"{ROOT}.yaml").write_text(agent, encoding="utf-8")
    return global_dir


def record_session(monkeypatch) -> dict:
    """Replace the REPL loop with a person's first two moves, and record what they would see.

    The trust question is answered yes and recorded (`asked`), then `/mode` runs through the REPL's
    own handler (`mode_line`). `mode` is the gate's mode after trust settled; `deny` is the running
    loop's own deny list."""
    seen: dict = {"ran": False, "asked": [], "mode": None, "mode_line": None, "deny": []}

    async def run(self):
        seen["ran"] = True
        gate = self._session_gate()
        sent: list[str] = []

        async def asker(request):
            seen["asked"].append(request.reason)
            return SimpleNamespace(allowed=True)

        async def send(text, metadata=None):
            sent.append(text)

        gate.asker = asker
        self._channel.send_message = send
        await self._establish_workspace_trust()
        await self._handle_mode_cmd("")
        seen.update(mode=gate.mode, mode_line=sent[-1],
                    deny=list(self._agent._config.permissions.deny_patterns))

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", run)
    return seen


async def refused(seen: dict) -> int:
    """Drive `start`; return its exit code. A session that ran instead fails with what it ran on."""
    try:
        await _drive()
    except typer.Exit as exc:
        return exc.exit_code
    pytest.fail(
        f"start ran a session instead of refusing: {seen['mode_line']!r}, trust asked "
        f"{len(seen['asked'])}x, agent deny sentinel "
        f"{'kept' if 'bash(sentinel-agent-deny *)' in seen['deny'] else 'DROPPED'}"
    )


async def test_a_leftover_plugin_entry_refuses_start_naming_file_and_line(
    tmp_path, monkeypatch, fake_home, capsys
):
    """QA-16's repro: the agent file sets read-only; the ONLY change is `agent.example` in
    overrides.yaml. `start` must print the loader's error and exit 1 — no session at all."""
    global_dir = session_home(tmp_path, monkeypatch, fake_home, agent=READ_ONLY_AGENT)
    (global_dir / "overrides.yaml").write_text(LEFTOVER, encoding="utf-8")
    seen = record_session(monkeypatch)

    assert await refused(seen) == 1
    assert not seen["ran"]
    err = " ".join(capsys.readouterr().err.split())
    assert "agent.example" in err and "(line 2)" in err, err
    assert str(global_dir / "overrides.yaml") in err, err
    assert "no installed plugin is named `example`" in err, err


async def test_a_typo_in_the_root_agent_file_refuses_start(tmp_path, monkeypatch, fake_home, capsys):
    """The pre-plugin trigger of the same fallback: `permisions:` for `permissions:`."""
    typo = READ_ONLY_AGENT.replace("permissions:", "permisions:")
    session_home(tmp_path, monkeypatch, fake_home, agent=typo)
    seen = record_session(monkeypatch)

    assert await refused(seen) == 1
    assert not seen["ran"]
    err = " ".join(capsys.readouterr().err.split())
    assert f"{ROOT}.yaml" in err and "permisions" in err and "(line 4)" in err, err


async def test_an_agent_file_the_loader_cannot_find_by_name_refuses_start(
    tmp_path, monkeypatch, fake_home, capsys
):
    """No fallback is kept for "not found" either. `start` only reaches the load with an agent read
    from a file (discovered, or minted a line earlier), so not-found means a file whose `name:` is
    not its file name — and a bare config would drop that file's mode just the same."""
    renamed = READ_ONLY_AGENT.replace("name: orchestrator", "name: helper")
    session_home(tmp_path, monkeypatch, fake_home, agent=renamed)
    seen = record_session(monkeypatch)

    assert await refused(seen) == 1
    assert not seen["ran"]
    err = " ".join(capsys.readouterr().err.split())
    assert "helper" in err and "helper.yaml" in err, err


async def test_a_valid_read_only_agent_still_starts_read_only(tmp_path, monkeypatch, fake_home):
    """The control for the three refusals: the same agent file with no leftover starts, runs
    `read-only`, keeps its deny pattern, and asks no trust question (not `auto`)."""
    session_home(tmp_path, monkeypatch, fake_home, agent=READ_ONLY_AGENT)
    seen = record_session(monkeypatch)

    await _drive()

    assert seen["ran"] and seen["mode"] == "read-only", seen
    assert seen["mode_line"].startswith("Permission mode: read-only."), seen["mode_line"]
    assert seen["asked"] == []
    assert "bash(sentinel-agent-deny *)" in seen["deny"]


def test_doctor_reports_the_root_agent_start_would_refuse(tmp_path, monkeypatch, fake_home):
    """doctor used to print `✓ Config valid` in exactly the state `start` now refuses: it loaded
    only config.yaml, and its one agent load (the window check) swallowed errors."""
    global_dir = session_home(tmp_path, monkeypatch, fake_home, agent=READ_ONLY_AGENT)
    (global_dir / "overrides.yaml").write_text(LEFTOVER, encoding="utf-8")

    result = CliRunner().invoke(app, ["doctor", "--config-dir", str(global_dir)])

    out = " ".join(result.output.split())
    assert "Config valid" not in out, out
    assert "Config invalid" in out and "agent.example" in out and "(line 2)" in out, out
    assert result.exit_code != 0
