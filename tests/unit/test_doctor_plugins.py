"""Doctor's plugins section (PAPI-08, PAPI-11, PAPI-12): core checks first, then each plugin — an on
plugin's own checks, an off or available plugin listed as off/available with its enable command and
never as failed, and a failed, refused or skipped plugin named with its reason.

Every run is the real Typer app with an explicit --config-dir whose provider points at port 9
(discard): the endpoint probe fails fast and never reaches a model, so doctor exits 1 on that CORE
failure in every test here. Plugin lines are asserted on OUTPUT; what decides the exit code is
asserted on the failure TOKENS doctor hands its summary (a spy records them, then runs the real
summary), so "not a failure" is a statement about the exit code, not about a glyph.

`p` is a bundled plugin swapped into BUILTIN_PLUGINS. Folder plugins are 44-09's sentinel writers,
so "never imported" is observed. Entry-point discovery is stubbed to none around the REAL folder
scan, so the venv's installed example plugin stays out — except in the test that uses it on purpose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel
from typer.testing import CliRunner

from localharness import resolved_version
from localharness.cli import doctor_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery
from localharness.plugins.api import Check, Plugin, PluginManifest
from tests.unit.test_doctor_layer_report import _layout, _run_doctor, _squash, _write_workspace
from tests.unit.test_plugin_resolve import write_folder_plugin

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
_CONFIG = {
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model", "available_models": ["test-model"]},
}


class PConfig(BaseModel):
    color: str = "#4a90d9"
    ok: bool = True


class P(Plugin):
    """draws p swatches"""

    manifest = PluginManifest(name="p", version="0.1.0", kind="tools")
    ConfigModel = PConfig

    def doctor(self, ctx):
        if ctx.config.ok:
            return [Check(name="p", status="pass", detail=f"swatches in {ctx.config.color}"),
                    Check(name="p-probe", status="skip", detail="nothing to probe offline")]
        return [Check(name="p", status="fail", detail="cannot reach [the] swatch server",
                      hint="set p.url to a server that answers")]


def _dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def sentinels(tmp_path: Path, monkeypatch):
    """No bundled plugin unless a test adds one; discovery = the real folder scan only; folder
    plugins mark this dir on import and leave sys.modules after."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")  # no folding: one doctor line is one output line
    marks = tmp_path / "sentinels"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    yield marks
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    """A configured global dir whose provider is the discard port."""
    g = tmp_path / "cfg"
    _dump(g / "config.yaml", _CONFIG)
    (g / "agents").mkdir()
    return g


@pytest.fixture
def recorded(monkeypatch) -> list[str]:
    """The failure tokens doctor handed its summary — the list that decides its exit code."""
    got: list[str] = []
    real = doctor_cmd._summarize_and_exit

    def spy(failures: list[str]) -> None:
        got[:] = failures
        real(failures)

    monkeypatch.setattr(doctor_cmd, "_summarize_and_exit", spy)
    return got


def _doctor(cfg: Path) -> str:
    result = runner.invoke(app, ["doctor", "--config-dir", str(cfg)])
    # A crash is not a failing check: doctor must survive every plugin in this file.
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"doctor raised {result.exception!r}\n{result.output}")
    return result.output


def _section(out: str) -> list[str]:
    """The plugins section: from its header line to the blank line before the summary rule."""
    lines = out.splitlines()
    start = next((i for i, line in enumerate(lines) if "Plugins:" in line), None)
    assert start is not None, f"no plugins section in doctor's output:\n{out}"
    end = next(i for i in range(start, len(lines)) if not lines[i].strip())
    return lines[start:end]


def _plugin_failures(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.startswith("plugin")]


# --------------------------------------------------------------------------- the section


def test_no_plugins_is_one_line_and_the_core_checks_alone_decide_the_exit_code(cfg, recorded) -> None:
    out = _doctor(cfg)

    assert _section(out) == ["i  Plugins: none on"]
    assert recorded and _plugin_failures(recorded) == [], recorded  # llm-unreachable, nothing else


def test_the_section_comes_after_the_core_checks_and_before_the_summary(cfg, monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (P,))
    lines = _doctor(cfg).splitlines()

    web = next(i for i, line in enumerate(lines) if "Web search" in line)
    header = lines.index("✓ Plugins: p")
    summary = next(i for i, line in enumerate(lines) if "issue(s) found" in line)
    rule = max(i for i in range(summary) if set(lines[i]) == {"─"})
    assert web < header < rule < summary, lines


def test_an_on_plugin_prints_each_of_its_checks(cfg, recorded, monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (P,))

    assert _section(_doctor(cfg)) == [
        "✓ Plugins: p",
        "✓ p: swatches in #4a90d9",
        "i  p-probe: nothing to probe offline",
    ]
    assert _plugin_failures(recorded) == []


def test_a_failing_check_prints_its_hint_and_is_a_failure(cfg, recorded, monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (P,))
    _dump(cfg / "config.yaml", {**_CONFIG, "p": {"ok": False}})

    assert _section(_doctor(cfg)) == [
        "✓ Plugins: p",
        "✗ p: cannot reach [the] swatch server",   # escaped: rich would eat `[the]`
        "       set p.url to a server that answers",
    ]
    assert _plugin_failures(recorded) == ["plugin-p"]


def test_an_off_bundled_plugin_is_off_with_its_enable_command_never_a_failure(cfg, recorded,
                                                                              monkeypatch) -> None:
    class Off(P):
        manifest = PluginManifest(name="p", version="0.1.0", kind="tools", enabled_by_default=False)

    _doctor(cfg)
    baseline = list(recorded)
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Off,))

    assert _section(_doctor(cfg)) == [
        "i  Plugins: none on",
        "i  p: off — turn on: localharness plugins enable p",
    ]
    assert recorded == baseline  # the same failures, so the same exit code, as with no plugin


def test_an_available_plugin_is_listed_with_its_enable_command_and_never_imported(cfg, recorded,
                                                                                sentinels) -> None:
    write_folder_plugin(cfg, "foo")

    assert _section(_doctor(cfg)) == [
        "i  Plugins: none on",
        "i  foo: available — turn on: localharness plugins enable foo",
    ]
    assert not (sentinels / "foo").exists()
    assert _plugin_failures(recorded) == []


def test_a_plugin_that_raises_at_import_is_named_with_its_reason_and_fails(cfg, recorded,
                                                                            sentinels) -> None:
    write_folder_plugin(cfg, "boom", prelude="raise RuntimeError('kaput')")
    _dump(cfg / "overrides.yaml", {"boom": {"enabled": True}})

    assert _section(_doctor(cfg)) == [
        "i  Plugins: none on",
        "✗ boom: failed — could not be imported: RuntimeError: kaput",
    ]
    assert (sentinels / "boom").exists(), "premise: an enabled plugin is imported"
    assert _plugin_failures(recorded) == ["plugin-boom"]


def test_an_out_of_range_plugin_is_skipped_with_its_reason_not_a_failure(cfg, recorded) -> None:
    write_folder_plugin(cfg, "future", manifest=', requires_localharness=">=9"')
    _dump(cfg / "overrides.yaml", {"future": {"enabled": True}})

    assert _section(_doctor(cfg)) == [
        "i  Plugins: none on",
        f"⚠ future: skipped — requires localharness >=9, this is {resolved_version()}",
    ]
    assert _plugin_failures(recorded) == []


def test_a_plugin_that_fails_to_configure_is_named_and_an_unconfigured_one_warns(cfg, recorded,
                                                                                 monkeypatch) -> None:
    class Broken(P):
        manifest = PluginManifest(name="broken", version="0.1.0", kind="tools")

        async def configure(self, ctx):
            raise RuntimeError("nope")

    class Waiting(P):
        manifest = PluginManifest(name="waiting", version="0.1.0", kind="tools")

        async def configure(self, ctx):
            return ("unconfigured", "waiting.url")

    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Broken, Waiting))

    assert _section(_doctor(cfg)) == [
        "i  Plugins: none on",
        "✗ broken: failed — configure() raised RuntimeError: nope",
        "⚠ waiting: unconfigured — set waiting.url",
    ]
    assert _plugin_failures(recorded) == ["plugin-broken"]


def test_a_resolver_warning_prints_as_a_warning_line(cfg, recorded) -> None:
    write_folder_plugin(cfg, "foo")
    _dump(cfg / "config.yaml", {**_CONFIG, "foo": {"enabled": "yes"}})

    assert _section(_doctor(cfg)) == [
        "i  Plugins: none on",
        "⚠ plugin foo: `foo.enabled` must be true or false, not 'yes' — it stays off",
        "i  foo: available — turn on: localharness plugins enable foo",
    ]
    assert _plugin_failures(recorded) == []


def test_doctor_never_crashes_when_plugins_cannot_be_resolved(cfg, recorded, monkeypatch) -> None:
    def broken(loader, **kwargs):
        raise RuntimeError("resolver down")

    monkeypatch.setattr("localharness.plugins.resolve.resolve", broken)

    assert _section(_doctor(cfg)) == ["✗ Plugins: could not be resolved — RuntimeError: resolver down"]
    assert _plugin_failures(recorded) == ["plugins-unresolved"]


# --------------------------------------------------------------------------- the real example plugin


def test_the_installed_example_plugin_available_then_on(cfg, tmp_path, monkeypatch) -> None:
    """The unmocked entry point: available and not imported, then on with its own check."""
    monkeypatch.setattr(discovery, "discover", _REAL_DISCOVER)
    sentinel = tmp_path / "example-imported"
    monkeypatch.setenv("LOCALHARNESS_EXAMPLE_PLUGIN_SENTINEL", str(sentinel))
    purge = lambda: [sys.modules.pop(m) for m in list(sys.modules)  # noqa: E731
                     if m.startswith("localharness_plugin_example")]
    purge()
    try:
        assert "i  example: available — turn on: localharness plugins enable example" in \
            _section(_doctor(cfg))
        assert not sentinel.exists()

        _dump(cfg / "overrides.yaml", {"example": {"enabled": True}})
        section = _section(_doctor(cfg))
    finally:
        purge()

    assert section[0] == "✓ Plugins: example"
    assert (f"✓ example: swatches render in #4a90d9; artifacts go to {cfg / 'artifacts' / 'example'}"
            in section), section
    assert sentinel.exists()


# --------------------------------------------------------------------------- 44-13's suffix


def test_an_overridden_plugin_setting_names_its_plugin(tmp_path, monkeypatch, fake_home) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (P,))
    layout = _layout(tmp_path, monkeypatch, fake_home)
    _write_workspace(layout, {"p": {"color": "#00ff00"}})

    out = _run_doctor()

    assert _squash("p.color = '#00ff00'  [workspace-config (plugin: p)]  (global: '#4a90d9')") \
        in _squash(out), out
