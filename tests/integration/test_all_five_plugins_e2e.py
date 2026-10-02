"""CORE-03 at Phase 50 — every real reader of the bundled plugin list sees all five.

`BUILTIN_PLUGINS` is the one list of what ships. This module drives each reader on its real code
path, with `BUILTIN_PLUGINS` NOT patched (a test below reads this file's own source to hold that):
`plugins list`, `components list` (`<name>.enabled` rows), the loader's plugin-name set (the set that
partitions `<name>:` keys out of the config), the resolver's plan, `--channel` validation (spied: it
iterated the five bundled manifests), doctor's `Plugins:` line and the start banner's `Plugins:` line.

The doctor/banner checks pin every install extra as installed (49's `extra_installed` kwdefault) so
dispatch reads ON whether or not discord.py is in this venv; image is off by default, so it is
asserted by state, not by presence in the ON line.

STUBBED (banner only): `_stub_start_boundaries(..., real_plugins=True)` — the LLM probe, the
tokenizer, the REPL loop; the provider is the loopback discard port. NOT proven: a real model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from tests.dispatch_support import isolate_discord_env

FIVE = {"image", "web", "memory", "dispatch", "autoresearch"}
ON_BY_DEFAULT = ["web", "memory", "dispatch", "autoresearch"]  # image is off by default
runner = CliRunner()


@pytest.fixture
def all_extras(monkeypatch, tmp_path):
    """Every install extra reads installed; no Discord env or token file reaches dispatch."""
    from localharness.plugins import resolve
    isolate_discord_env(monkeypatch, tmp_path)
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)


def _run(*args: str, ok: bool = True):
    res = runner.invoke(app, list(args), env={"COLUMNS": "400"})
    assert res.exit_code == 0 or not ok, res.output
    return res


def test_this_file_never_patches_the_bundled_list():
    src = Path(__file__).read_text(encoding="utf-8")
    needle = "BUILTIN" + "_PLUGINS"  # split so this line is not itself a hit
    assert not [ln for ln in src.splitlines() if needle in ln and "setattr" in ln]
    assert "builtin." + needle not in src


@pytest.mark.plugin("autoresearch")  # asserts autoresearch's default ON state
def test_plugins_list_shows_all_five(components_home):
    rows = {r["name"]: r for r in json.loads(_run("plugins", "list", "--json").stdout)}
    assert FIVE <= set(rows), sorted(rows)
    assert rows["autoresearch"]["state_kind"] == "on", rows["autoresearch"]


def test_components_list_has_an_enabled_row_for_each(components_home):
    paths = {r["path"] for r in json.loads(_run("components", "list", "--json").stdout)}
    assert {f"{n}.enabled" for n in FIVE} <= paths, sorted(p for p in paths if p.endswith(".enabled"))


def test_the_loader_partitions_all_five(components_home):
    from localharness.cli.components_cmd import _build_loader
    assert FIVE <= _build_loader().plugin_names()


@pytest.mark.plugin("autoresearch")  # asserts autoresearch's default ON state
def test_the_resolve_plan_has_all_five_bundled(components_home):
    from localharness.cli.components_cmd import _build_loader
    from localharness.plugins.resolve import resolve
    entries = resolve(_build_loader()).plan.entries
    assert {e.name for e in entries if e.bundled} == FIVE, [(e.name, e.bundled) for e in entries]
    assert {e.name: e.state for e in entries}["autoresearch"] == "on"


def test_channel_validation_reads_the_five_manifests(monkeypatch):
    from localharness.plugins import builtin
    from localharness.plugins.channels import channel_names
    seen: list[tuple] = []
    real = builtin.bundled_plugins
    monkeypatch.setattr(builtin, "bundled_plugins", lambda: seen.append(out := real()) or out)
    names = channel_names()
    assert len(seen) == 1 and {c.manifest.name for c in seen[0]} == FIVE, seen
    assert {"terminal", "acp", "discord"} <= names, names


@pytest.mark.plugin("autoresearch")  # asserts autoresearch's default ON state
def test_doctor_plugins_line_lists_the_on_four(components_home, all_extras):
    out = _run("doctor", ok=False).output  # exit 1 here = the unreachable test provider, not a plugin
    line = next(ln for ln in out.splitlines() if ln.startswith("✓ Plugins: "))
    assert line.removeprefix("✓ Plugins: ").split(", ") == ON_BY_DEFAULT, line
    assert "image: off — turn on: localharness plugins enable image" in out, out


async def test_the_start_banner_lists_the_on_four(tmp_path, monkeypatch, all_extras):
    from localharness.cli.start_cmd import _start_async
    from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
    from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries
    _stub_start_boundaries(tmp_path, monkeypatch, real_plugins=True)
    _offline_provider(tmp_path)
    monkeypatch.chdir(tmp_path)
    printed = _capture_start_console(monkeypatch)
    await _start_async(None, False, False, str(tmp_path))
    lines = [p.split("Plugins: ")[1].split("[/]")[0] for p in printed if "Plugins:" in p]
    assert len(lines) == 1, printed
    on = lines[0].split(", ")
    assert [n for n in on if n in FIVE] == ON_BY_DEFAULT, lines  # + any installed third-party plugin
