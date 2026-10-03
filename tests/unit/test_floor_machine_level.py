"""`org.enforce_capability_floor` is machine-level only.

The capability floor keeps an untrusted-ingest tool and a host-dangerous tool out of one agent.
Its off switch used to merge like any other `org.*` key, so a cloned repo's own
`.localharness/config.yaml` could turn the floor off for every session started inside it. The
switch has no direction a repository should own, so the loader now treats it as the
`permissions.ask` machine-level keys are treated: a workspace value that differs from the global
one is dropped, with one startup warning naming the key and the file; the global value stands.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from localharness.config.loader import ConfigLoader

KEY = "org.enforce_capability_floor"
PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m"}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def _floor(value: bool) -> dict:
    return {"org": {"enforce_capability_floor": value}}


def _layers(tmp_path: Path, *, g_cfg=None, g_over=None, ws_cfg=None, ws_over=None):
    """A global dir (config.yaml always present, with a provider) and a workspace dir."""
    g, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    _write(g / "config.yaml", {"version": "1", "provider": PROVIDER, **(g_cfg or {})})
    for path, data in ((g / "overrides.yaml", g_over), (ws / "config.yaml", ws_cfg),
                       (ws / "overrides.yaml", ws_over)):
        if data is not None:
            _write(path, data)
    return ConfigLoader(config_dir=g, local_config_dir=ws), ws


def _warnings(loader: ConfigLoader) -> list[str]:
    # getattr: the attribute is part of the fix; the escape-hatch cases must also run on a tree
    # that predates it, where no warning exists to collect.
    return list(getattr(loader, "harness_warnings", []))


def _dropped(file: Path) -> str:
    return f"ignoring {KEY} in {file}: only the global config may set it"


def test_a_workspace_config_cannot_turn_the_floor_off(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_floor(False))
    assert loader.load_harness().org.enforce_capability_floor is True
    assert _warnings(loader) == [_dropped(ws / "config.yaml")]


def test_a_workspace_overrides_file_cannot_turn_the_floor_off(tmp_path):
    loader, ws = _layers(tmp_path, ws_over=_floor(False))
    assert loader.load_harness().org.enforce_capability_floor is True
    assert _warnings(loader) == [_dropped(ws / "overrides.yaml")]


def test_the_global_config_still_turns_the_floor_off(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_floor(False))
    assert loader.load_harness().org.enforce_capability_floor is False
    assert _warnings(loader) == []


def test_the_global_overrides_file_still_turns_the_floor_off(tmp_path):
    loader, _ = _layers(tmp_path, g_over=_floor(False))
    assert loader.load_harness().org.enforce_capability_floor is False
    assert _warnings(loader) == []


def test_the_global_overrides_file_turns_it_back_on(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_floor(False), g_over=_floor(True))
    assert loader.load_harness().org.enforce_capability_floor is True
    assert _warnings(loader) == []


def test_a_workspace_restating_the_global_value_is_no_warning(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_floor(False), ws_cfg=_floor(False))
    assert loader.load_harness().org.enforce_capability_floor is False
    assert _warnings(loader) == []


def test_a_workspace_cannot_switch_it_on_either(tmp_path):
    """The machine owner's off stands: the key is the owner's in both directions."""
    loader, ws = _layers(tmp_path, g_over=_floor(False), ws_cfg=_floor(True))
    assert loader.load_harness().org.enforce_capability_floor is False
    assert _warnings(loader) == [_dropped(ws / "config.yaml")]


def test_both_workspace_files_are_each_named_once(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_floor(False), ws_over=_floor(False))
    assert loader.load_harness().org.enforce_capability_floor is True
    assert _warnings(loader) == [_dropped(ws / "config.yaml"), _dropped(ws / "overrides.yaml")]


def test_loading_twice_warns_once(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg=_floor(False))
    loader.load_harness()
    assert loader.load_harness().org.enforce_capability_floor is True
    assert _warnings(loader) == [_dropped(ws / "config.yaml")]


def test_other_org_keys_in_the_workspace_still_merge(tmp_path):
    """Only the machine-level keys are narrowed: the workspace's other org settings (here
    `log_level`) reach the merged view. `org.audit_log_path` is machine-level too now (a file
    written anywhere on disk), so the project's value is dropped beside the floor's."""
    ws_cfg = {"org": {"enforce_capability_floor": False, "log_level": "debug",
                      "audit_log_path": "ws-audit.jsonl"}}
    loader, _ = _layers(tmp_path, ws_cfg=ws_cfg)
    org = loader.load_harness().org
    assert org.enforce_capability_floor is True and org.log_level == "debug"
    assert org.audit_log_path == "audit.jsonl"


async def test_a_real_start_in_such_a_workspace_keeps_the_floor_and_says_so(
        tmp_path, monkeypatch, fake_home):
    """A real `start` (config_dir=None, so discovery finds the workspace) in a project whose
    config.yaml turns the floor off: the floor is set ON, and the summary line names the drop once."""
    import localharness.tools.capabilities as caps
    from tests.unit.test_start_cmd import _capture_start_console
    from tests.unit.test_start_plugins import _summary
    from tests.unit.test_workspace_state_landing import _drive, _workspace_start

    _home, _global_dir, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _write(ws / "config.yaml", _floor(False))
    calls: list[bool] = []
    monkeypatch.setattr(caps, "_FLOOR_ENABLED", True)  # restored after the test either way
    monkeypatch.setattr(caps, "set_floor_enabled", lambda enabled: calls.append(enabled))
    printed = _capture_start_console(monkeypatch)

    await _drive()

    assert calls == [True], calls
    summary = _summary(printed)
    assert summary.count(f"ignoring {KEY} in ") == 1, summary
    assert "config.yaml: only the global config may set it" in summary, summary
