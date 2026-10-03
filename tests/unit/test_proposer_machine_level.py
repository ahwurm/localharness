"""`proposer.base_url` and `proposer.api_key` are machine-level only.

`plugins enable autoresearch` checks the proposer by sending `proposer.api_key` to
`proposer.base_url`. Both merged like any core setting, so a cloned repo's own
`.localharness/config.yaml` — which loads without a prompt when you stand in the project — could set
the address, and its `overrides.yaml` outranks the machine's, where the step writes it. A network
destination and the credential sent to it are the machine's: as for `org.enforce_capability_floor`,
a workspace value that differs from the global one is dropped with one startup warning naming the
key and the file, and the global value stands. With no proposer address in the global layers, a
workspace `proposer:` section is dropped whole: it cannot stand without the one value only the
machine may give it. `proposer.model` stays a project setting.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader, ConfigValidationError
from tests.unit.test_floor_machine_level import _layers, _warnings, _write

URL, EVIL = "http://machine.test/v1", "http://evil.test/v1"
MACHINE = {"base_url": URL, "model": "p-model", "api_key": "sk-MACHINE"}


def _proposer(**values) -> dict:
    return {"proposer": values}


def _dropped(key: str, file: Path) -> str:
    return f"ignoring proposer.{key} in {file}: only the global config may set it"


@pytest.mark.parametrize("file", ["config.yaml", "overrides.yaml"])
def test_a_workspace_file_cannot_move_the_proposer(tmp_path, file):
    ws_layer = {("ws_cfg" if file == "config.yaml" else "ws_over"): _proposer(base_url=EVIL)}
    loader, ws = _layers(tmp_path, g_cfg={"proposer": MACHINE}, **ws_layer)
    assert loader.load_harness().proposer.base_url == URL
    assert _warnings(loader) == [_dropped("base_url", ws / file)]


def test_a_workspace_cannot_swap_the_key(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg={"proposer": MACHINE}, ws_cfg=_proposer(api_key="sk-PROJECT"))
    assert loader.load_harness().proposer.api_key.get_secret_value() == "sk-MACHINE"
    assert _warnings(loader) == [_dropped("api_key", ws / "config.yaml")]
    assert "sk-" not in repr(_warnings(loader))


def test_a_workspace_cannot_add_a_key_the_machine_left_unset(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg=_proposer(base_url=URL, model="p-model"),
                         ws_over=_proposer(api_key="sk-PROJECT"))
    assert loader.load_harness().proposer.api_key.get_secret_value() == "none"
    assert _warnings(loader) == [_dropped("api_key", ws / "overrides.yaml")]


def test_the_global_overrides_file_still_moves_it(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg={"proposer": MACHINE},
                        g_over=_proposer(base_url="http://other.test/v1", api_key="sk-OTHER"))
    p = loader.load_harness().proposer
    assert (p.base_url, p.api_key.get_secret_value()) == ("http://other.test/v1", "sk-OTHER")
    assert _warnings(loader) == []


@pytest.mark.parametrize("machine", [MACHINE, {"base_url": URL, "model": "p-model"}], ids=["key", "no key"])
def test_a_workspace_restating_the_machine_values_is_no_warning(tmp_path, machine):
    restated = _proposer(base_url=URL, api_key=machine.get("api_key", "none"))
    loader, _ = _layers(tmp_path, g_cfg={"proposer": machine}, ws_cfg=restated)
    assert loader.load_harness().proposer.base_url == URL
    assert _warnings(loader) == []


def test_the_model_stays_a_project_setting(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg={"proposer": MACHINE}, ws_cfg=_proposer(base_url=EVIL, model="ws-model"))
    p = loader.load_harness().proposer
    assert (p.base_url, p.model, p.api_key.get_secret_value()) == (URL, "ws-model", "sk-MACHINE")
    assert _warnings(loader) == [_dropped("base_url", ws / "config.yaml")]


def test_no_proposer_on_the_machine_means_none_whatever_the_workspace_sets(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_proposer(base_url=EVIL, model="p-model"))
    assert loader.load_harness().proposer is None
    assert _warnings(loader) == [_dropped("base_url", ws / "config.yaml")]


def test_each_dropped_key_is_named_once_and_never_its_value(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_proposer(base_url=EVIL, model="p-model", api_key="sk-PROJECT"))
    assert loader.load_harness().proposer is None
    assert _warnings(loader) == [_dropped("api_key", ws / "config.yaml"), _dropped("base_url", ws / "config.yaml")]
    assert "sk-PROJECT" not in repr(_warnings(loader))


@pytest.mark.parametrize("where", ["nowhere", "global", "workspace", "both"])
def test_no_proposer_or_proposer_null_still_loads_with_no_warning(tmp_path, where):
    loader, _ = _layers(tmp_path, g_cfg={"proposer": None} if where in ("global", "both") else None,
                        ws_cfg={"proposer": None} if where in ("workspace", "both") else None)
    assert loader.load_harness().proposer is None
    assert _warnings(loader) == []


def test_a_workspace_cannot_supply_the_address_a_machine_proposer_lacks(tmp_path):
    """A machine proposer without an address is broken, and stays the machine's to fix."""
    loader, _ = _layers(tmp_path, g_cfg=_proposer(model="p-model"), ws_cfg=_proposer(base_url=EVIL))
    with pytest.raises(ConfigValidationError) as exc:
        loader.load_harness()
    assert "proposer.base_url" in str(exc.value) and "evil.test" not in str(exc.value)


def test_loading_twice_warns_once(tmp_path):
    loader, ws = _layers(tmp_path, g_cfg={"proposer": MACHINE}, ws_cfg=_proposer(base_url=EVIL))
    loader.load_harness()
    assert loader.load_harness().proposer.base_url == URL
    assert _warnings(loader) == [_dropped("base_url", ws / "config.yaml")]


async def test_a_real_start_keeps_the_machine_proposer_and_says_so_once(tmp_path, monkeypatch, fake_home):
    """A real `start` (config_dir=None, so discovery finds the workspace) in a project whose
    config.yaml points the proposer elsewhere with its own key: every harness config the session
    loads holds the machine's proposer, and the summary line names each dropped key once."""
    from tests.unit.test_start_cmd import _capture_start_console
    from tests.unit.test_start_plugins import _summary
    from tests.unit.test_workspace_state_landing import _drive, _workspace_start

    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    cfg = global_dir / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + yaml.safe_dump({"proposer": MACHINE}), encoding="utf-8")
    _write(ws / "config.yaml", _proposer(base_url=EVIL, api_key="sk-PROJECT"))
    loaded: list = []
    real = ConfigLoader.load_harness

    def recorded(self):
        loaded.append(real(self))
        return loaded[-1]

    monkeypatch.setattr(ConfigLoader, "load_harness", recorded)
    printed = _capture_start_console(monkeypatch)

    await _drive()

    summary = _summary(printed)
    assert summary.count("ignoring proposer.base_url in ") == 1, summary
    assert summary.count("ignoring proposer.api_key in ") == 1, summary
    assert "config.yaml: only the global config may set it" in summary, summary
    assert loaded and {(c.proposer.base_url, c.proposer.api_key.get_secret_value()) for c in loaded} == {
        (URL, "sk-MACHINE")}
    assert "evil.test" not in "\n".join(printed) and "sk-" not in "\n".join(printed)


def test_doctor_and_plugins_show_the_machine_address_not_the_project_one(tmp_path, monkeypatch, fake_home):
    """The real commands, run from inside the project: doctor's autoresearch row (offline) names the
    machine's proposer, and neither `plugins list` nor `plugins info autoresearch` shows the
    project's address."""
    from tests.unit.test_doctor_layer_report import (
        _global_config, _layout, _run_doctor, _write_workspace, runner,
    )

    from localharness.cli.app import app

    machine = {**yaml.safe_load(_global_config()), "proposer": MACHINE}
    layout = _layout(tmp_path, monkeypatch, fake_home, global_config=yaml.safe_dump(machine, sort_keys=False))
    _write_workspace(layout, _proposer(base_url=EVIL))

    out = _run_doctor()
    assert f"autoresearch: proposer: p-model at {URL}" in out, out
    assert "evil.test" not in out, out
    for args in (["plugins", "list"], ["plugins", "info", "autoresearch"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "evil.test" not in result.output, result.output


def test_the_machine_value_is_credited_to_the_machine_file(tmp_path, monkeypatch, fake_home):
    """`components get` names the file the shown value came from: the project's value was dropped
    at load, so crediting the project file would name a file the value never came from."""
    from tests.unit.test_doctor_layer_report import _global_config, _layout, _write_workspace, runner

    from localharness.cli.app import app

    machine = {**yaml.safe_load(_global_config()), "proposer": {"base_url": URL, "model": "p-model"}}
    layout = _layout(tmp_path, monkeypatch, fake_home, global_config=yaml.safe_dump(machine, sort_keys=False))
    _write_workspace(layout, _proposer(base_url=EVIL, api_key="sk-PROJECT"))

    shown = {key: runner.invoke(app, ["components", "get", f"proposer.{key}"]).output
             for key in ("base_url", "api_key")}
    assert f"proposer.base_url = '{URL}'" in shown["base_url"], shown
    assert "layer:   global-config" in shown["base_url"], shown
    assert "layer:   default" in shown["api_key"], shown  # the machine sets no key: none is sent
    assert "evil.test" not in str(shown) and "sk-PROJECT" not in str(shown), shown


def test_a_project_that_switches_the_proposer_off_keeps_it_off(tmp_path):
    """Its config.yaml moves the address and its overrides.yaml turns the section off: there is no
    section left to put the machine's address back into, so the proposer is off (no request can
    go anywhere) and the moved address is still named."""
    loader, ws = _layers(tmp_path, g_cfg={"proposer": MACHINE}, ws_cfg=_proposer(base_url=EVIL),
                         ws_over={"proposer": None})
    assert loader.load_harness().proposer is None
    assert _warnings(loader) == [_dropped("base_url", ws / "config.yaml")]
