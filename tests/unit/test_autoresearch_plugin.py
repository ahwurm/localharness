"""The autoresearch plugin, built beside today's eager wiring and NOT yet registered (50-04): its
manifest, the cheap import (plugins/builtin.py imports it on every --help), the package's lazy
re-exports, and the one offline doctor row (never the api key, never a socket).

The mounted-help tests below prove each `--help` through the lazy mount equals the 50-01 golden
before the cut (50-05) touches cli/app.py."""
from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.autoresearch.plugin import NO_PROPOSER, AutoresearchPlugin
from localharness.plugins import builtin, discovery
from localharness.plugins.api import Check, PluginPaths

runner = CliRunner()
_PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "test-model"}
_PROPOSER = {"base_url": "http://127.0.0.1:9/v1", "model": "p-model", "api_key": "sk-SENTINEL"}


# --------------------------------------------------------------------------- manifest


def test_the_manifest():
    m = AutoresearchPlugin.manifest
    assert (m.name, m.version, m.kind, m.enabled_by_default, m.requires_extra) == (
        "autoresearch", "0.1.0", "dev", True, None)
    assert m.sections == ("proposer", "sentinel")
    assert [(d.name, d.target) for d in m.cli] == [
        ("autoresearch", "localharness.cli.report_cmd:autoresearch_app"),
        ("experiment", "localharness.cli.experiment_cmd:experiment_app"),
        ("propose", "localharness.cli.propose_cmd:propose_app")]
    assert AutoresearchPlugin.ConfigModel is None and AutoresearchPlugin.AgentConfigModel is None
    assert AutoresearchPlugin.wants_artifacts is False
    assert AutoresearchPlugin.__doc__.splitlines()[0] == "experiment loop"


def test_registered_as_the_last_bundled_plugin():
    """The cut appends exactly one class: autoresearch is the fifth and last entry."""
    assert builtin.BUILTIN_PLUGINS[-1] is AutoresearchPlugin
    assert builtin.BUILTIN_PLUGINS.count(AutoresearchPlugin) == 1 and len(builtin.BUILTIN_PLUGINS) == 5


def test_configure_is_ready_and_start_stop_do_nothing():
    import asyncio
    p = AutoresearchPlugin()
    assert asyncio.run(p.configure(None)) == "ready"
    assert asyncio.run(p.start(None)) is None and asyncio.run(p.stop(None)) is None


# --------------------------------------------------------------------------- import cost

_COST = (
    "import sys, localharness.autoresearch.plugin\n"
    "bad = [m for m in ('aiosqlite', 'scipy', 'localharness.autoresearch.archive',\n"
    "                   'localharness.cli.autoresearch_cmd', 'localharness.cli.experiment_cmd',\n"
    "                   'localharness.cli.propose_cmd', 'localharness.cli.report_cmd') if m in sys.modules]\n"
    "assert not bad, bad\n"
)


def test_importing_the_plugin_loads_no_archive_no_scipy_and_no_command_module():
    r = subprocess.run([sys.executable, "-c", _COST], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


_ROOT_COST = (
    "import sys, localharness.cli.app\n"
    "bad = [m for m in ('scipy', 'numpy', 'aiosqlite', 'localharness.autoresearch.archive',\n"
    "                   'localharness.bench.aggregator', 'localharness.cli.autoresearch_cmd',\n"
    "                   'localharness.cli.experiment_cmd', 'localharness.cli.propose_cmd',\n"
    "                   'localharness.cli.report_cmd') if m in sys.modules]\n"
    "assert not bad, bad\n"
)


def test_importing_the_cli_no_longer_loads_the_experiment_loop():
    """After the cut the root app mounts autoresearch/experiment/propose lazily from the manifest."""
    r = subprocess.run([sys.executable, "-c", _ROOT_COST], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_the_package_still_re_exports_the_archive():
    from localharness.autoresearch import ArchiveEntry, ArchiveQuery, ArchiveStore
    from localharness.autoresearch import archive
    assert (ArchiveStore, ArchiveEntry, ArchiveQuery) == (
        archive.ArchiveStore, archive.ArchiveEntry, archive.ArchiveQuery)
    from localharness.autoresearch.archive import ArchiveEntry as E, ArchiveStore as S  # SKILL.md:76
    assert (E, S) == (ArchiveEntry, ArchiveStore)
    with pytest.raises(AttributeError):
        import localharness.autoresearch as pkg
        pkg.NoSuchName  # noqa: B018


# --------------------------------------------------------------------------- doctor


def _home(tmp_path: Path, text: str | None = None, proposer: dict | None = None) -> PluginPaths:
    g = tmp_path / "g"
    g.mkdir()
    data = {"version": "1", "provider": _PROVIDER, **({"proposer": proposer} if proposer else {})}
    (g / "config.yaml").write_text(text if text is not None else yaml.safe_dump(data), encoding="utf-8")
    return PluginPaths(global_config_dir=g, workspace=None, state_dir=g)


def _doctor(paths: PluginPaths, monkeypatch) -> list[Check]:
    from types import SimpleNamespace
    calls: list = []
    monkeypatch.setattr(socket.socket, "connect", lambda *a: calls.append(a) or (_ for _ in ()).throw(
        OSError("doctor opened a socket")))
    rows = AutoresearchPlugin().doctor(SimpleNamespace(paths=paths, config=None, llm=None))
    assert calls == []
    return rows


def test_doctor_names_the_proposer_never_its_key(tmp_path, monkeypatch):
    rows = _doctor(_home(tmp_path, proposer=_PROPOSER), monkeypatch)
    assert rows == [Check(name="autoresearch", status="pass", detail="proposer: p-model at http://127.0.0.1:9/v1")]
    assert "sk-SENTINEL" not in repr(rows)


def test_doctor_without_a_proposer_skips_with_the_hint(tmp_path, monkeypatch):
    rows = _doctor(_home(tmp_path), monkeypatch)
    assert rows == [Check(name="autoresearch", status="skip", detail=NO_PROPOSER,
                          hint="set proposer.base_url / proposer.model (a model distinct from "
                               "provider.default_model)")]


def test_doctor_on_an_unreadable_config_skips_and_does_not_raise(tmp_path, monkeypatch):
    rows = _doctor(_home(tmp_path, text="provider: [unclosed\n"), monkeypatch)
    assert len(rows) == 1 and rows[0].status == "skip"
    assert rows[0].detail.startswith("config not readable: ")


def test_doctor_reads_the_workspace_layer(tmp_path, monkeypatch):
    paths = _home(tmp_path)
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"proposer": {**_PROPOSER, "model": "ws-model"}}))
    rows = _doctor(PluginPaths(global_config_dir=paths.global_config_dir, workspace=ws, state_dir=ws),
                   monkeypatch)
    assert rows[0].status == "pass" and "ws-model" in rows[0].detail


def test_the_row_reaches_localharness_doctor_when_bundled(tmp_path, monkeypatch):
    """Reachable end to end: swapped in as the only bundled plugin, the real `doctor` prints the row
    (the provider is the discard port, so core fails fast and doctor exits 1 on that, not on us)."""
    from localharness.cli.app import app
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (AutoresearchPlugin,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    g = _home(tmp_path, proposer=_PROPOSER).global_config_dir
    out = runner.invoke(app, ["doctor", "--config-dir", str(g)]).output
    assert "proposer: p-model at http://127.0.0.1:9/v1" in out, out
    assert "sk-SENTINEL" not in out


# --------------------------------------------------------------------------- mounted help == golden

from tests.unit.test_autoresearch_cli_golden import CASES, ENV, _golden, _norm_paths, _norm_stamps  # noqa: E402

HELP_CASES = [c for c in CASES if c.endswith("_help")]


def _test_root():
    """A root built with the real root's settings and NO core autoresearch commands: everything
    under autoresearch / experiment / propose comes through the plugin's lazy descriptors."""
    import typer

    from localharness.cli.plugin_mount import PluginCommandGroup
    root = typer.Typer(name="localharness", help="Model-agnostic hierarchical agent harness for local LLMs.",
                       add_completion=True, no_args_is_help=True, rich_markup_mode="rich",
                       context_settings={"help_option_names": ["-h", "--help"]}, cls=PluginCommandGroup)

    @root.callback()
    def _root() -> None:
        """Model-agnostic hierarchical agent harness for local LLMs."""
    return root


@pytest.fixture
def bundled(monkeypatch):
    """REPLACE autoresearch into the bundled set (never append: after the cut it is already there)."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (
        *(p for p in builtin.BUILTIN_PLUGINS if p.manifest.name != "autoresearch"), AutoresearchPlugin))
    assert sum(p is AutoresearchPlugin for p in builtin.bundled_plugins()) == 1


@pytest.mark.plugin("autoresearch")
@pytest.mark.parametrize("case", HELP_CASES)
def test_mounted_help_equals_the_golden(case, tmp_path, bundled):
    root = _test_root()
    assert CASES[case][0] not in typer_commands(root)  # no core twin: only the lazy mount serves it
    result = runner.invoke(root, CASES[case], env=ENV)
    got = {"exit": result.exit_code, "output": _norm_stamps(_norm_paths(result.output, tmp_path))}
    assert got == _golden()[case], f"mounted {case!r} differs from the 50-01 golden"
    assert "--install-completion" not in result.output


def typer_commands(root) -> set[str]:
    import typer
    return set(typer.main.get_command(root).commands)


def test_there_are_thirteen_help_cases():
    assert len(HELP_CASES) == 13


def test_propose_app_is_a_single_command():
    import typer

    from localharness.cli.propose_cmd import propose_app
    from localharness.cli.autoresearch_cmd import autoresearch_app
    from localharness.cli.experiment_cmd import experiment_app
    assert not isinstance(typer.main.get_command(propose_app), typer.core.TyperGroup)
    assert propose_app._add_completion is False
    assert autoresearch_app._add_completion is False and experiment_app._add_completion is False
