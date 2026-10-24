"""The project override (its one predicate and its one sentence) and the disable producer."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from localharness.cli.plugins_cmd import (
    OVERRIDE_LINE, OVERRIDE_SAVED, PassProgress, PassResult, PluginSession, Refusal, StepOutcome, Touch,
    disable_pass, override_line, override_of,
)
from localharness.config.loader import ConfigLoader
from localharness.core.bus import EventBus
from localharness.plugins import builtin, discovery
from localharness.plugins import resolve as resolve_module
from localharness.plugins.resolve import resolve
from tests.unit.test_plugin_resolve import write_folder_plugin
from tests.unit.test_plugins_bookkeeping import _CONFIG, Stub

_REAL_DISCOVER = discovery.discover


@pytest.fixture
def layers(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    g, ws = tmp_path / "g", tmp_path / "proj" / ".localharness"
    g.mkdir()
    ws.mkdir(parents=True)
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return SimpleNamespace(g=g, ws=ws)


def _write(path: Path, enabled) -> None:
    path.write_text(yaml.safe_dump({"stub": {"enabled": enabled}}), encoding="utf-8")


def _of(layers, on: bool, *, to_workspace: bool = False, workspace: bool = True):
    loader = ConfigLoader(config_dir=layers.g, local_config_dir=layers.ws if workspace else None)
    return override_of(loader, resolve(loader).plan.entry("stub"), on, to_workspace)


def test_override_of_names_the_overrides_file_that_turns_it_off(layers) -> None:
    _write(layers.ws / "overrides.yaml", False)

    assert _of(layers, True) == ("off", layers.ws / "overrides.yaml")
    assert _of(layers, False) is None


def test_override_of_names_the_config_file_when_only_it_holds_the_switch(layers) -> None:
    _write(layers.ws / "config.yaml", True)

    assert _of(layers, False) == ("on", layers.ws / "config.yaml")
    assert _of(layers, True) is None


def test_override_of_the_overrides_file_outranks_the_config_file(layers) -> None:
    _write(layers.ws / "config.yaml", True)
    _write(layers.ws / "overrides.yaml", False)

    assert _of(layers, True) == ("off", layers.ws / "overrides.yaml")
    assert _of(layers, False) is None


def test_override_of_a_later_layer_that_agrees_leaves_nothing_to_say(layers) -> None:
    _write(layers.ws / "config.yaml", False)
    _write(layers.ws / "overrides.yaml", True)

    assert _of(layers, True) is None
    assert _of(layers, False) == ("on", layers.ws / "overrides.yaml")


def test_override_of_ignores_a_value_that_is_not_a_bool(layers) -> None:
    _write(layers.ws / "overrides.yaml", "no")

    assert _of(layers, True) is None


def test_override_of_ignores_the_machine_layers(layers) -> None:
    _write(layers.g / "overrides.yaml", False)

    assert _of(layers, True) is None


def test_override_of_is_none_for_a_workspace_write(layers) -> None:
    _write(layers.ws / "overrides.yaml", False)

    assert _of(layers, True, to_workspace=True) is None


def test_override_of_is_none_with_no_workspace(layers) -> None:
    assert _of(layers, True, workspace=False) is None


def test_override_of_is_none_for_a_plugin_you_installed(layers) -> None:
    _write(layers.ws / "overrides.yaml", False)
    loader = ConfigLoader(config_dir=layers.g, local_config_dir=layers.ws)

    assert override_of(loader, SimpleNamespace(name="stub", bundled=False), True, False) is None


def test_override_line_fills_the_one_template(layers) -> None:
    file = layers.ws / "overrides.yaml"

    assert override_line("stub", ("off", file)) == (
        f"this project turns it off ({file}) and that still wins — "
        "localharness plugins enable stub --workspace changes it")
    assert override_line("stub", ("on", file), OVERRIDE_SAVED) == (
        f"saved on this machine, but this project turns it on ({file}) and that still wins — "
        "localharness plugins disable stub --workspace changes it")


def test_the_template_and_the_saved_prefix() -> None:
    assert OVERRIDE_LINE.startswith("{saved}this project turns it {state} ({file})")
    assert OVERRIDE_SAVED == "saved on this machine, but "


def _session(layers, **kw) -> PluginSession:
    return PluginSession(layers.g, layers.ws, frozenset(), kw.pop("unconfigured", {}), kw.pop("failed", {}), **kw)


async def _disable(layers, name: str = "stub", **kw):
    progress, reports = PassProgress(), []
    result = await disable_pass(_session(layers, **kw), name, progress=progress,
                                report=lambda kind, payload: reports.append((kind, payload)))
    return result, progress, reports


def _machine(layers) -> dict:
    return yaml.safe_load((layers.g / "overrides.yaml").read_text(encoding="utf-8"))


async def test_disable_pass_writes_the_machine_switch_off(layers) -> None:
    _write(layers.g / "overrides.yaml", True)

    result, progress, reports = await _disable(layers)

    assert _machine(layers) == {"stub": {"enabled": False}}
    assert result == PassResult("disabled", StepOutcome("stub", False))
    assert (progress.phase, progress.wrote) == ("disable", True)
    assert reports == [("wrote", (f"stub disabled in {layers.g / 'overrides.yaml'}", ()))]


async def test_disable_pass_under_a_project_that_turns_it_on_is_an_override_that_still_loads(layers) -> None:
    _write(layers.ws / "config.yaml", True)

    result, progress, reports = await _disable(layers)

    assert _machine(layers) == {"stub": {"enabled": False}}
    assert (result.kind, result.override, result.step, result.loads) == (
        "override", ("on", layers.ws / "config.yaml"), StepOutcome("stub", False), True)
    assert progress.wrote and [kind for kind, _ in reports] == ["wrote"]
    assert not (layers.ws / "overrides.yaml").exists()


@pytest.mark.parametrize("kw", [
    {"touched": {"stub": Touch(False, True, ("failed", "x"))}},
    {"failed": {"stub": "import broke"}},
    {"unconfigured": {"stub": "stub.url"}}])
async def test_disable_pass_keeps_a_plugin_that_would_not_load_out_of_loads(layers, kw) -> None:
    _write(layers.ws / "config.yaml", True)

    result, _, _ = await _disable(layers, **kw)

    assert (result.kind, result.loads) == ("override", False)


async def test_disable_pass_publishes_nothing_and_resolves_once(layers, monkeypatch) -> None:
    async def publish(self, event):
        raise AssertionError("published")

    calls = []

    def counting(loader, *a, **kw):
        calls.append(loader)
        return resolve(loader, *a, **kw)

    monkeypatch.setattr(EventBus, "publish", publish)
    monkeypatch.setattr(resolve_module, "resolve", counting)
    _write(layers.ws / "config.yaml", True)

    await _disable(layers)

    assert len(calls) == 1


@pytest.mark.parametrize("name", ["nope", "org"])
async def test_disable_pass_refuses_a_name_the_plan_refused_or_does_not_hold(layers, monkeypatch, name) -> None:
    monkeypatch.setattr(discovery, "discover", _REAL_DISCOVER)
    write_folder_plugin(layers.g, "org")

    with pytest.raises(Refusal):
        await _disable(layers, name)

    assert not (layers.g / "overrides.yaml").exists()


def test_disable_pass_is_a_coroutine_with_no_await_in_it() -> None:
    assert inspect.iscoroutinefunction(disable_pass)
    assert "await" not in inspect.getsource(disable_pass)
