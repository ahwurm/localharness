"""QA-18: `org.permissions.mode` reaches the session.

`init`'s "No" to "Allow the agent to change this machine?" writes `org.permissions.mode: read-only`
and prints "✓ Read-only sessions" — and every session still ran `auto`, because the gate was built
from the AGENT's mode alone and nothing read the org's. The rule now: the agent's own mode when it
sets one (the agent file, or the `agent:` section of overrides.yaml), else the org's, else the
default. "Sets one" means the key is present — an explicit `mode: auto` is the agent's choice even
though it is also the default.

The org value may come from a workspace's `.localharness/config.yaml` too, but only to TIGHTEN the
global one (PRD §3.3): a cloned repo must not be able to hand its own sessions a looser mode.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader
from tests.unit.test_start_root_agent_fail_closed import (
    READ_ONLY_AGENT,
    ROOT,
    record_session,
    session_home,
)
from tests.unit.test_workspace_state_landing import _drive

PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m"}
SILENT_AGENT = f"name: {ROOT}\nrole: r\nmodel: inherit\n"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def _org(mode: str) -> dict:
    return {"org": {"permissions": {"mode": mode}}}


# ------------------------------------------------------------------ the rule, at the loader


@pytest.mark.parametrize("config_org, overrides, agent_perms, expected", [
    pytest.param(_org("read-only"), None, None, "read-only", id="init-no-answer"),
    pytest.param(_org("unattended"), None, None, "unattended", id="org-unattended"),
    pytest.param(_org("read-only"), None, {"mode": "auto"}, "auto", id="explicit-default-wins"),
    pytest.param({}, None, None, "auto", id="nobody-sets-one"),
    pytest.param(_org("auto"), _org("read-only"), None, "read-only", id="overrides-beat-config"),
    pytest.param(_org("read-only"), {"agent": {"permissions": {"mode": "guarded"}}}, None,
                 "guarded", id="overrides-agent-section-is-the-agents"),
])
def test_the_session_mode_rule(tmp_path, config_org, overrides, agent_perms, expected):
    _write(tmp_path / "config.yaml", {"version": "1", "provider": PROVIDER, **config_org})
    if overrides is not None:
        _write(tmp_path / "overrides.yaml", overrides)
    agent = {"name": ROOT, "role": "r", "model": "inherit"}
    if agent_perms is not None:
        agent["permissions"] = agent_perms
    _write(tmp_path / "agents" / f"{ROOT}.yaml", agent)

    assert ConfigLoader(config_dir=tmp_path).load_agent(ROOT).permissions.mode == expected


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    global_dir, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _write(global_dir / "agents" / f"{ROOT}.yaml", {"name": ROOT, "role": "r", "model": "inherit"})
    ws.mkdir(parents=True)
    return global_dir, ws


def test_a_workspace_org_mode_may_tighten(layers):
    """The workspace template's own example: `org.permissions.mode: read-only` in a project."""
    global_dir, ws = layers
    _write(global_dir / "config.yaml", {"version": "1", "provider": PROVIDER, **_org("auto")})
    _write(ws / "config.yaml", _org("read-only"))

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)
    assert loader.load_agent(ROOT).permissions.mode == "read-only"


@pytest.mark.parametrize("global_org, ws_mode, expected", [
    pytest.param({}, "unattended", "auto", id="silent-global-keeps-the-default"),
    pytest.param(_org("read-only"), "auto", "read-only", id="global-read-only-stands"),
])
def test_a_workspace_org_mode_may_not_loosen(layers, caplog, global_org, ws_mode, expected):
    global_dir, ws = layers
    _write(global_dir / "config.yaml", {"version": "1", "provider": PROVIDER, **global_org})
    _write(ws / "overrides.yaml", _org(ws_mode))

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)
    with caplog.at_level(logging.WARNING, logger="localharness.config.loader"):
        assert loader.load_agent(ROOT).permissions.mode == expected
        loader.load_agent(ROOT, bypass_cache=True)
    ignored = [r.getMessage() for r in caplog.records if "org.permissions.mode" in r.getMessage()]
    assert len(ignored) == 1 and ws_mode in ignored[0], ignored


# ------------------------------------------------------------------ the session, end to end


def _set_org_mode(global_dir: Path, mode: str) -> None:
    """What `init` writes for its posture answer: `org.permissions.mode` in the global config."""
    with (global_dir / "config.yaml").open("a", encoding="utf-8") as f:
        f.write(f"org:\n  permissions:\n    mode: {mode}\n")


@pytest.mark.parametrize("org_mode", ["read-only", "unattended"])
async def test_the_org_mode_is_the_session_mode(tmp_path, monkeypatch, fake_home, org_mode):
    """QA-18's repro: the org sets the mode, the minted root agent sets none. The session runs the
    org's mode, `/mode` says so, and no auto-mode trust question is asked."""
    global_dir = session_home(tmp_path, monkeypatch, fake_home, agent=None)
    _set_org_mode(global_dir, org_mode)
    seen = record_session(monkeypatch)

    await _drive()

    assert seen["mode_line"].startswith(f"Permission mode: {org_mode}."), seen
    assert seen["asked"] == [], seen


async def test_an_org_auto_session_is_unchanged(tmp_path, monkeypatch, fake_home):
    """The control: `init`'s "Yes" writes `auto`, which is what every session ran before."""
    global_dir = session_home(tmp_path, monkeypatch, fake_home, agent=None)
    _set_org_mode(global_dir, "auto")
    seen = record_session(monkeypatch)

    await _drive()

    assert seen["mode_line"].startswith("Permission mode: auto."), seen
    assert len(seen["asked"]) == 1, seen


async def test_an_agent_looser_than_the_org_keeps_its_mode_and_says_so(
    tmp_path, monkeypatch, fake_home, capsys
):
    """No "most restrictive wins": the agent file's explicit `auto` stands against the org's
    `read-only` — and `start` prints one line naming both, so the owner can decide."""
    loose = READ_ONLY_AGENT.replace("mode: read-only", "mode: auto")
    global_dir = session_home(tmp_path, monkeypatch, fake_home, agent=loose)
    _set_org_mode(global_dir, "read-only")
    seen = record_session(monkeypatch)

    await _drive()

    assert seen["mode_line"].startswith("Permission mode: auto."), seen
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "org.permissions.mode" in ln]
    assert len(lines) == 1, lines
    assert "auto" in lines[0] and "read-only" in lines[0] and ROOT in lines[0], lines
