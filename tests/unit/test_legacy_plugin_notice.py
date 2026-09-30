"""QA-03: plugins built on 0.15.0's documented plugin API vanished silently after the upgrade.

0.15.0's `start` loaded two kinds of plugin that nothing reads since 0.16: installed packages with a
`localharness.tools` or `localharness.hooks` entry point, and `plugins/<dir>/manifest.yaml` folders
in the machine's config dir. After the upgrade `plugins list` printed `[]` and `doctor` said "All
checks passed." — the user was never told a plugin they rely on had stopped loading.

This file proves that such a plugin is NAMED (which one, that it was written for the 0.15 plugin API,
that it is no longer loaded, where the migration note is) and that nothing of it is ever loaded —
first at resolve(), then on the three surfaces that print its warnings: `plugins list` (stderr),
`doctor` (a warning line, never a failure) and the `start` banner (a real `_start_async` drive with
discovery real and the model stubbed). The surface tests were written after the wiring: they are
graded by mutating it (drop the notices from resolve() and all four redden), not relabelled RED.

The distributions here are REAL metadata: a `*.dist-info` on sys.path, read by the unmocked
importlib.metadata, exactly as pip leaves an installed package. Every legacy module writes a
sentinel file when it is imported, so "never loaded" is observed, not inferred.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import doctor_cmd
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.plugins import discovery
from localharness.plugins.resolve import resolve
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_doctor_plugins import _doctor, _section
from tests.unit.test_plugin_resolve import write_folder_plugin
from tests.unit.test_start_cmd import _capture_start_console, _read_sessions, _stub_start_boundaries
from tests.unit.test_workspace_state_landing import _boom, _drive, _hermetic

_CONFIG = {  # the provider is the discard port: nothing here can reach a model
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model"},
}
_EVERY_PART = ("0.15 plugin API", "no longer loaded")


@pytest.fixture(autouse=True)
def _forget_folder_plugins():
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def _global(tmp_path: Path, name: str = "g") -> Path:
    """A machine config dir holding a minimal valid config.yaml."""
    g = tmp_path / name
    g.mkdir(parents=True)
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


def _marking_module(path: Path, sentinel: Path, *classes: str) -> None:
    """A module that writes `sentinel` the moment anything imports it."""
    path.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n"
                    + "".join(f"class {c}:\n    pass\n" for c in classes), encoding="utf-8")


def _dist(site_dir: Path, name: str, version: str, groups: dict[str, dict[str, str]]) -> None:
    """`<site_dir>/<name>-<version>.dist-info/` as an installer leaves it: METADATA and
    entry_points.txt."""
    info = site_dir / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                   encoding="utf-8")
    (info / "entry_points.txt").write_text("".join(
        f"[{group}]\n" + "".join(f"{ep} = {target}\n" for ep, target in eps.items()) + "\n"
        for group, eps in groups.items()), encoding="utf-8")


def install_legacy_dist(site_dir: Path, sentinel: Path, monkeypatch) -> None:
    """lh-oldtool 0.1.0, written for 0.15.0: one `localharness.tools` and one `localharness.hooks`
    entry point, and its module, which marks `sentinel` if it is ever imported."""
    _dist(site_dir, "lh-oldtool", "0.1.0", {"localharness.tools": {"oldtool": "lh_oldtool:OldTool"},
                                            "localharness.hooks": {"oldhooks": "lh_oldtool:OldHooks"}})
    _marking_module(site_dir / "lh_oldtool.py", sentinel, "OldTool", "OldHooks")
    monkeypatch.syspath_prepend(str(site_dir))


@pytest.fixture
def legacy_dist(tmp_path: Path, monkeypatch):
    """The installed 0.15 tool pack; yields the sentinel its module writes when imported."""
    sentinel = tmp_path / "lh_oldtool-imported"
    install_legacy_dist(tmp_path / "site", sentinel, monkeypatch)
    yield sentinel
    sys.modules.pop("lh_oldtool", None)


def legacy_folder(config_dir: Path, name: str, sentinel: Path) -> Path:
    """`<config_dir>/plugins/<name>/` as 0.15.0 read it: a manifest.yaml naming `tools:HelloTool`
    and that tools.py (which marks `sentinel` if imported) — and no __init__.py. Returns the
    manifest's path."""
    folder = config_dir / "plugins" / name
    folder.mkdir(parents=True)
    (folder / "manifest.yaml").write_text(yaml.safe_dump({
        "name": name, "version": "0.1.0", "description": "a 0.15 manifest plugin",
        "tools": [{"name": f"{name}_hello", "description": "say hello",
                   "entrypoint": "tools:HelloTool"}]}), encoding="utf-8")
    _marking_module(folder / "tools.py", sentinel, "HelloTool")
    return folder / "manifest.yaml"


def notices(warnings) -> list[str]:
    """The warnings that name a plugin written for the 0.15 API."""
    return [w for w in warnings if "0.15 plugin API" in w]


def assert_names_the_015_api(line: str) -> None:
    for part in (*_EVERY_PART, discovery.LEGACY_API_DOC):
        assert part in line, f"{part!r} is missing from the notice: {line}"


# --------------------------------------------------------------------------- what resolve() says


def test_a_distribution_built_for_the_015_api_is_named_and_never_loaded(tmp_path, legacy_dist) -> None:
    resolution = resolve(ConfigLoader(config_dir=_global(tmp_path)))

    lines = [w for w in resolution.warnings if "lh-oldtool" in w]
    assert len(lines) == 1, f"expected one line naming lh-oldtool: {resolution.warnings}"
    line = lines[0]
    for part in ("lh-oldtool 0.1.0", "`localharness.tools`", "`oldtool`", "`localharness.hooks`",
                 "`oldhooks`"):
        assert part in line, f"{part!r} is missing from the notice: {line}"
    assert_names_the_015_api(line)
    assert resolution.warnings == (line,), resolution.warnings
    # Never loaded: its module never ran, is not imported, and the plan has no row for it.
    assert not legacy_dist.exists(), "a 0.15 plugin's module was imported"
    assert "lh_oldtool" not in sys.modules
    assert not {e.name for e in resolution.plan.entries} & {"oldtool", "oldhooks", "lh-oldtool"}


def test_a_015_manifest_folder_is_named_and_never_loaded(tmp_path) -> None:
    g = _global(tmp_path)
    sentinel = tmp_path / "oldstyle-imported"
    manifest = legacy_folder(g, "oldstyle", sentinel)
    # Not plugins at all, and never named: the README.md `init` writes into every machine's
    # plugins/, and a folder with neither manifest.yaml nor __init__.py.
    (g / "plugins" / "README.md").write_text("# Plugins\n", encoding="utf-8")
    (g / "plugins" / "notes").mkdir()
    (g / "plugins" / "notes" / "todo.txt").write_text("port oldstyle\n", encoding="utf-8")

    resolution = resolve(ConfigLoader(config_dir=g))

    assert len(resolution.warnings) == 1, resolution.warnings
    line = resolution.warnings[0]
    assert str(manifest) in line, line
    assert_names_the_015_api(line)
    assert not sentinel.exists(), "a 0.15 manifest plugin's tools.py was imported"
    assert "oldstyle" not in {e.name for e in resolution.plan.entries}


def test_a_folder_with_an_init_py_is_todays_api_and_gets_no_notice(tmp_path, monkeypatch) -> None:
    g = _global(tmp_path)
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    write_folder_plugin(g, "ported")  # binds `plugin`: a plugin of today's API ...
    (g / "plugins" / "ported" / "manifest.yaml").write_text("name: ported\nversion: 0.1.0\n")
    old = legacy_folder(g, "oldstyle", tmp_path / "oldstyle-imported")  # ... beside a 0.15 one

    resolution = resolve(ConfigLoader(config_dir=g))

    named = notices(resolution.warnings)
    assert len(named) == 1 and str(old) in named[0], named  # the scan ran and found the old one
    assert not any(str(g / "plugins" / "ported") in w for w in resolution.warnings)
    assert resolution.plan.entry("ported").state == "available"
    assert not (marks / "ported").exists()


def test_a_distribution_that_also_declares_todays_api_gets_no_notice(tmp_path, monkeypatch,
                                                                     legacy_dist) -> None:
    site_dir, sentinel = tmp_path / "site2", tmp_path / "bothapis-imported"
    _dist(site_dir, "lh-bothapis", "0.2.0", {"localharness.tools": {"bt": "lh_bothapis:T"},
                                             "localharness.plugins": {"bothapis": "lh_bothapis:P"}})
    _marking_module(site_dir / "lh_bothapis.py", sentinel, "T", "P")
    monkeypatch.syspath_prepend(str(site_dir))
    try:
        resolution = resolve(ConfigLoader(config_dir=_global(tmp_path)))
    finally:
        sys.modules.pop("lh_bothapis", None)

    named = notices(resolution.warnings)
    assert len(named) == 1 and "lh-oldtool 0.1.0" in named[0], named  # the control is named
    assert not any("lh-bothapis" in w for w in resolution.warnings), resolution.warnings
    assert resolution.plan.entry("bothapis").state == "available"  # listed under today's API
    assert not sentinel.exists()


def test_a_workspace_is_never_a_source(tmp_path) -> None:
    g = _global(tmp_path)
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    old = legacy_folder(g, "oldstyle", tmp_path / "oldstyle-imported")
    in_ws = legacy_folder(ws, "wsold", tmp_path / "wsold-imported")

    resolution = resolve(ConfigLoader(config_dir=g, local_config_dir=ws))

    named = notices(resolution.warnings)
    assert len(named) == 1 and str(old) in named[0], named
    assert not any(str(in_ws.parent) in w for w in resolution.warnings), resolution.warnings


def test_the_check_is_one_metadata_scan(tmp_path, monkeypatch, legacy_dist) -> None:
    calls: list[tuple] = []
    real = importlib.metadata.entry_points

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", spy)
    found = discovery.legacy_notices(_global(tmp_path))

    assert calls == [((), {})], f"expected ONE unfiltered entry-point scan, got {calls}"
    assert len(found) == 1 and "lh-oldtool 0.1.0" in found[0], found


# --------------------------------------------------------------------------- the three surfaces

runner = CliRunner()


@pytest.fixture
def legacy_machine(tmp_path: Path, legacy_dist, monkeypatch) -> tuple[Path, list[str], list[Path]]:
    """A machine with both kinds of 0.15 plugin: lh-oldtool installed, and `oldstyle` in its
    config dir — named `[old] g`, so every printed path is a markup guard (unescaped, rich deletes
    `[old]`). Returns (config dir, the two notices as legacy_notices words them, the sentinels)."""
    monkeypatch.setenv("COLUMNS", "400")
    g = _global(tmp_path, "[old] g")
    folder_sentinel = tmp_path / "oldstyle-imported"
    legacy_folder(g, "oldstyle", folder_sentinel)
    expected = discovery.legacy_notices(g)
    assert len(expected) == 2, f"premise: one distribution and one folder, got {expected}"
    return g, expected, [legacy_dist, folder_sentinel]


def _never_loaded(sentinels: list[Path]) -> None:
    assert not [s for s in sentinels if s.exists()], "a 0.15 plugin's module was imported"
    assert "lh_oldtool" not in sys.modules


def test_plugins_list_names_both_on_stderr(legacy_machine) -> None:
    g, expected, sentinels = legacy_machine

    result = runner.invoke(app, ["plugins", "list", "--config-dir", str(g)])

    assert result.exit_code == 0, result.output
    assert notices(result.stderr.splitlines()) == [f"⚠ {line}" for line in expected], result.stderr
    assert "0.15 plugin API" not in result.stdout, "a notice was printed inside the table"
    _never_loaded(sentinels)


def test_plugins_list_json_stays_valid_json_without_them(legacy_machine) -> None:
    g, expected, sentinels = legacy_machine

    result = runner.invoke(app, ["plugins", "list", "--json", "--config-dir", str(g)])

    assert result.exit_code == 0, result.output
    names = {row["name"] for row in json.loads(result.stdout)}
    assert not names & {"oldtool", "oldhooks", "lh-oldtool", "oldstyle"}, names
    assert "0.15 plugin API" not in result.stdout
    assert notices(result.stderr.splitlines()) == [f"⚠ {line}" for line in expected], result.stderr
    _never_loaded(sentinels)


def _issues(out: str) -> str:
    """Doctor's verdict line: `N issue(s) found.` or `All checks passed.`"""
    found = re.search(r"\d+ issue\(s\) found\.|All checks passed\.", out)
    assert found, f"no verdict in doctor's output:\n{out}"
    return found.group()


def test_doctor_warns_about_both_and_counts_no_failure(tmp_path, monkeypatch) -> None:
    recorded: list[str] = []  # the failure tokens doctor hands its summary: they decide its exit code
    summarize = doctor_cmd._summarize_and_exit

    def spy(failures: list[str]) -> None:
        recorded[:] = failures
        summarize(failures)

    monkeypatch.setattr(doctor_cmd, "_summarize_and_exit", spy)
    monkeypatch.setenv("COLUMNS", "400")
    g = _global(tmp_path, "[old] g")
    (g / "agents").mkdir()
    baseline = _doctor(g)  # the same config, before either 0.15 plugin exists
    failures = list(recorded)
    assert failures, "premise: the spy saw doctor's core failure (its provider is the discard port)"
    sentinels = [tmp_path / "lh_oldtool-imported", tmp_path / "oldstyle-imported"]
    legacy_folder(g, "oldstyle", sentinels[1])
    install_legacy_dist(tmp_path / "site", sentinels[0], monkeypatch)
    try:
        expected = discovery.legacy_notices(g)
        out = _doctor(g)
    finally:
        sys.modules.pop("lh_oldtool", None)

    assert len(expected) == 2, expected
    assert notices(_section(out)) == [f"⚠ {line}" for line in expected], _section(out)
    assert recorded == failures, "a 0.15 plugin's notice changed what doctor counts as a failure"
    assert _issues(out) == _issues(baseline)
    _never_loaded(sentinels)


def test_the_start_banner_names_both_and_the_session_completes(tmp_path, monkeypatch, fake_home,
                                                               legacy_dist) -> None:
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, fake_home, home)
    _stub_start_boundaries(global_dir, monkeypatch, real_plugins=True)  # discovery stays REAL
    _offline_provider(global_dir)
    folder_sentinel = tmp_path / "oldstyle-imported"
    legacy_folder(global_dir, "oldstyle", folder_sentinel)
    expected = discovery.legacy_notices(global_dir)
    assert len(expected) == 2, expected
    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)  # an in-project workspace never asks
    printed = _capture_start_console(monkeypatch)

    asyncio.run(_drive())  # a real session: _start_async(None, False, False, None), zero turns

    summary = next(line for line in printed if "startup)" in line)
    for line in expected:
        assert line in summary, f"the start banner does not carry {line!r}: {summary}"
    rows = _read_sessions(global_dir)
    assert len(rows) == 1 and rows[0][3] == "complete", rows
    _never_loaded([legacy_dist, folder_sentinel])
