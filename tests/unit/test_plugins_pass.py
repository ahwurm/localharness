"""The project override: its one predicate and its one sentence."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from localharness.cli.plugins_cmd import OVERRIDE_LINE, OVERRIDE_SAVED, override_line, override_of
from localharness.config.loader import ConfigLoader
from localharness.plugins import builtin, discovery
from localharness.plugins.resolve import resolve
from tests.unit.test_plugins_bookkeeping import _CONFIG, Stub


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
