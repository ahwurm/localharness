"""49-07: the Discord token is never printed, and doctor's deprecation rows warn without failing.

Criteria (ROADMAP Phase 49): 1 (configured by settings, the token a secret no surface shows) and 2
(the env fallback announced as a `warn` doctor row naming the new key, never a red exit).

Every surface is the REAL Typer app (CliRunner) against a tmp GLOBAL dir passed by `--config-dir`,
with the REAL bundled plugin list (dispatch is the fourth). Each surface is its own test so a leak
names its surface; each asserts SENTINEL is absent from stdout, stderr, the captured logs and the
audit file (`_run`). Discovery is stubbed to nothing installed; `extra_installed` is pinned True
for "extra present" and left real (discord.py is not installed here) for "extra absent".
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import components_cmd, plugins_cmd
from localharness.cli.app import app
from localharness.plugins import discovery
from tests.dispatch_support import isolate_discord_env
from tests.unit.test_plugins_enable_setup import _CONFIG

SENTINEL = "SENTINEL.dispatch.token.0000"
ENV = {"LOCALHARNESS_DISCORD_TOKEN": SENTINEL, "LOCALHARNESS_DISCORD_ALLOW": "42",
       "LOCALHARNESS_DISCORD_CHANNELS": "7", "LOCALHARNESS_DISCORD_ACK": "👀"}
runner = CliRunner()


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch, caplog):
    isolate_discord_env(monkeypatch, tmp_path)
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(components_cmd, "console", Console(width=400))
    caplog.set_level(logging.DEBUG)


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    cfg = dict(_CONFIG, org={"audit_log_path": str(g / "audit.jsonl")})
    (g / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return g


def _present(monkeypatch) -> None:
    from localharness.plugins import resolve
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)


def _run(g: Path, caplog, *args: str):
    """One CLI run; the token must be in none of stdout, stderr, logs or the audit file."""
    result = runner.invoke(app, [*args, "--config-dir", str(g)])
    for where, text in (("stdout", result.stdout), ("stderr", result.stderr), ("logs", caplog.text),
                        ("exception", repr(result.exception))):
        assert SENTINEL not in text, f"the token reached {where} of `{' '.join(args)}`:\n{text}"
    audit = g / "audit.jsonl"
    assert not audit.exists() or SENTINEL not in audit.read_text(encoding="utf-8")
    return result


def _enable(g: Path, caplog, *extra: str):
    r = _run(g, caplog, "plugins", "enable", "dispatch", "--set", f"discord.token={SENTINEL}",
             "--set", "discord.allow=42", *extra)
    assert r.exit_code == 0, r.output
    on_disk = yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))
    assert on_disk["dispatch"]["discord"]["token"] == SENTINEL  # written raw, to the 0600 overlay
    return r


@pytest.fixture
def configured(g, caplog, monkeypatch) -> Path:
    _present(monkeypatch)
    _enable(g, caplog)
    return g


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- the token on every surface --------------------------------------------------------------------


def test_enable_set_extra_present(g, caplog, monkeypatch) -> None:
    _present(monkeypatch)
    r = _enable(g, caplog)
    assert "**********" in r.output, r.output
    assert "install extra" not in r.output, r.output


def test_enable_set_extra_absent(g, caplog) -> None:
    """Enable writes, then says the extra is missing — "takes effect on the next start" alone is
    not true without discord.py."""
    r = _enable(g, caplog)
    assert "note: dispatch is missing its install extra — install `localharness[dispatch]` to use it" \
        in _flat(r.output), r.output


def test_components_list_table(configured, caplog) -> None:
    r = _run(configured, caplog, "components", "list")
    assert r.exit_code == 0, r.output
    rows = [ln for ln in r.output.splitlines() if "dispatch.discord." in ln]
    assert {k for k in ("token", "allow", "channels", "ack")
            if any(f"dispatch.discord.{k}" in ln for ln in rows)} == {"token", "allow", "channels", "ack"}, rows
    assert all("(plugin: dispatch)" in ln for ln in rows), rows


def test_components_list_json(configured, caplog) -> None:
    r = _run(configured, caplog, "components", "list", "--json")
    assert r.exit_code == 0, r.output
    assert "dispatch.discord.token" in r.stdout
    json.loads(r.stdout)


def test_components_get_token(configured, caplog) -> None:
    for args in (("components", "get", "dispatch.discord.token"),
                 ("components", "get", "dispatch.discord.token", "--json")):
        r = _run(configured, caplog, *args)
        assert r.exit_code == 0, r.output
        assert "**********" in r.output, r.output


def test_plugins_info(configured, caplog) -> None:
    for args in (("plugins", "info", "dispatch"), ("plugins", "info", "dispatch", "--json")):
        r = _run(configured, caplog, *args)
        assert r.exit_code == 0, r.output


def test_doctor_configured_by_settings(configured, caplog) -> None:
    r = _run(configured, caplog, "doctor")
    assert "Discord configured — 1 allowed user(s); listens in any channel the bot can see" \
        in _flat(r.output), r.output


def test_start_summary(configured, caplog, monkeypatch) -> None:
    """A terminal session with the token configured: the startup console never shows it."""
    from localharness.cli.start_cmd import _start_async
    from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
    from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

    overrides = (configured / "overrides.yaml").read_text(encoding="utf-8")
    _stub_start_boundaries(configured, monkeypatch)  # rewrites config.yaml, keeps overrides.yaml
    _offline_provider(configured)
    assert (configured / "overrides.yaml").read_text(encoding="utf-8") == overrides
    printed = _capture_start_console(monkeypatch)
    asyncio.run(asyncio.wait_for(
        _start_async(None, False, False, str(configured), channel_mode="terminal"), 60))
    summary = next(p for p in printed if "startup)" in p)
    assert "dispatch" in summary or any("Plugins:" in p and "dispatch" in p for p in printed), printed
    assert not any(SENTINEL in p for p in printed), printed
    assert SENTINEL not in caplog.text


def test_comma_list_set_validates_to_both_ids(g, caplog, monkeypatch) -> None:
    _present(monkeypatch)
    _run(g, caplog, "plugins", "enable", "dispatch", "--set", f"discord.token={SENTINEL}",
         "--set", "discord.allow=1,2")
    r = _run(g, caplog, "components", "get", "dispatch.discord.allow", "--json")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["value"] == ["1", "2"], r.stdout


# --- doctor rows and its exit code -----------------------------------------------------------------


def _issues(output: str) -> str | None:
    m = re.search(r"(\d+) issue\(s\) found", output)
    return m.group(1) if m else None


def test_doctor_env_only_warns_per_variable_without_changing_the_exit_code(g, caplog, monkeypatch) -> None:
    _present(monkeypatch)
    bare = _run(g, caplog, "doctor")
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    env = _run(g, caplog, "doctor")
    flat = _flat(env.output)
    for var, field in (("TOKEN", "token"), ("ALLOW", "allow"), ("CHANNELS", "channels"), ("ACK", "ack")):
        assert (f"dispatch: LOCALHARNESS_DISCORD_{var} is deprecated and stops working in 0.17.0 — set "
                f"dispatch.discord.{field}") in flat, env.output
    assert flat.count("dispatch: LOCALHARNESS_DISCORD_") == 4, env.output
    assert "⚠ dispatch: LOCALHARNESS_DISCORD_TOKEN is deprecated" in flat, env.output
    assert "dispatch: dispatch:" not in flat, env.output  # the row name is printed once
    assert (env.exit_code, _issues(env.output)) == (bare.exit_code, _issues(bare.output)), (
        bare.output, env.output)


def test_doctor_not_configured_skip_row(g, caplog, monkeypatch) -> None:
    _present(monkeypatch)
    r = _run(g, caplog, "doctor")
    flat = _flat(r.output)
    assert "Discord not configured" in flat, r.output
    assert "localharness plugins enable dispatch --set discord.token=… --set discord.allow=<your user id>" \
        in flat, r.output


def test_doctor_extra_absent_names_the_install_line_and_runs_no_checks(g, caplog, monkeypatch) -> None:
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    r = _run(g, caplog, "doctor")
    flat = _flat(r.output)
    assert "localharness[dispatch]" in flat, r.output
    assert "Discord configured" not in flat and "Discord not configured" not in flat, r.output
    # RESEARCH open question 2: the deprecation rows appear once the extra is installed.
    assert "LOCALHARNESS_DISCORD_" not in flat, r.output
