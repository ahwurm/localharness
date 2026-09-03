"""Subagent runtime paths resolve against the SESSION's config dir (LAYR-02, v013 Risk #6).

Before phase 38, a `--config-dir /foo` session rooted its ROOT agent at `/foo` while every one
of its subagents silently fell through to `AgentLoop`'s own fallbacks: compact.md under a
hardcoded `~/.localharness` (loop.py:781) and a bare `"KILL"` resolved against the PROCESS CWD
(loop.py:725). These tests pin the fix at two levels:

- Layer 1: `_child_runtime_paths` itself — including the `config_dir=None` case, which must stay
  BYTE-IDENTICAL to today's `~/.localharness` fallback (the zero-behavior-change invariant).
- Layer 2 (task 2): every one of the 6 `AgentLoop(...)` construction sites in subagent.py.
"""
from __future__ import annotations

from pathlib import Path

from localharness.agent.subagent import _child_runtime_paths, build_explore_config
from localharness.config.models import AgentConfig, BudgetConfig, PermissionConfig


def _cfg(name: str = "explore", kill_file: str | None = None) -> AgentConfig:
    return AgentConfig(
        name=name,
        role="test child",
        permissions=PermissionConfig(budget=BudgetConfig(kill_file=kill_file)),
    )


# ---------------------------------------------------------------------------
# Layer 1 — the helper
# ---------------------------------------------------------------------------

def test_explicit_config_dir_roots_both_paths(tmp_path):
    """`--config-dir <dir>` puts BOTH the kill file and compact.md under that dir."""
    kill, compact = _child_runtime_paths(_cfg("explore"), tmp_path)
    assert kill == tmp_path / "KILL"
    assert compact == tmp_path / "agents" / "explore" / "compact.md"


def test_default_reproduces_todays_home_fallback(monkeypatch):
    """config_dir=None with no env override == loop.py:781's hardcoded fallback, byte-identical.

    The autouse `_isolate_localharness_home` conftest fixture sets LOCALHARNESS_HOME for EVERY
    test, so this one must unset it (and LOCALHARNESS_DIR) to see the real default chain.
    """
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    cfg = build_explore_config("explore")

    kill, compact = _child_runtime_paths(cfg, None)

    assert compact == Path.home() / ".localharness" / "agents" / cfg.name / "compact.md"
    assert kill == Path.home() / ".localharness" / "KILL"


def test_env_override_roots_paths_under_localharness_home(tmp_path, monkeypatch):
    """LOCALHARNESS_HOME (legacy alias) and LOCALHARNESS_DIR (canonical) both move the children."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "envhome"))
    kill, compact = _child_runtime_paths(_cfg("web-researcher"), None)
    assert kill == tmp_path / "envhome" / "KILL"
    assert compact == tmp_path / "envhome" / "agents" / "web-researcher" / "compact.md"

    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "canonical"))
    kill2, compact2 = _child_runtime_paths(_cfg("web-researcher"), None)
    assert kill2 == tmp_path / "canonical" / "KILL"
    assert compact2 == tmp_path / "canonical" / "agents" / "web-researcher" / "compact.md"


def test_absolute_kill_file_is_honored_not_rerooted(tmp_path):
    """resolve_runtime_path's standing contract: an absolute value is never re-rooted."""
    kill, compact = _child_runtime_paths(_cfg("explore", kill_file="/var/run/CUSTOMKILL"), tmp_path)
    assert kill == Path("/var/run/CUSTOMKILL")
    # ...while compact.md still follows the config dir.
    assert compact == tmp_path / "agents" / "explore" / "compact.md"


def test_no_path_resolves_against_the_process_cwd(tmp_path):
    """The CWD-relative bare "KILL" (loop.py:725) is exactly what this plan retires for children."""
    kill, compact = _child_runtime_paths(_cfg("cruncher"), tmp_path)
    cwd = str(Path.cwd())
    assert not str(kill).startswith(cwd)
    assert not str(compact).startswith(cwd)
    assert kill.is_absolute() and compact.is_absolute()
