"""The autostart unit's captured environment is one fingerprinted `launch` entry of the machine record.

`machine_snapshot` reads the unit file's `# env` block and emits it as
`{file: "systemd/localharness-llm.service", kind: "launch", name: "autostart.env", shown: <JSON>}`,
only for the config dir the unit's `ExecStart` serves. The pseudo-file is a pairing key only.

The unit file lives under a redirected `XDG_CONFIG_HOME`; the global dir and the trust store are
conftest's hermetic home. Nothing here asks a service manager anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from localharness.config import trust
from localharness.provider import units as u
from tests.unit.trust_support import g, never_asked, write_yaml  # noqa: F401 — g is a fixture

FILE = "systemd/localharness-llm.service"
KEY = (FILE, "launch", "autostart.env")
INTERPRETER = "/opt/lh/bin/python"
PATH_VALUE = "/usr/bin:/bin"
HF = "/h/hf"
LD = "/opt/cuda/lib:/usr/lib"


@pytest.fixture(autouse=True)
def _redirected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _body(config_dir, captured=None, interpreter=INTERPRETER) -> str:
    env = {"PATH": PATH_VALUE, "HF_HOME": HF} if captured is None else captured
    return u.render_harness_unit(interpreter, config_dir, env, home="/home/u", version="0.19.0")


def _install(config_dir, captured=None) -> Path:
    return u.write_harness_unit(_body(config_dir, captured))


def _mine(snapshot: list[dict]) -> list[dict]:
    return [e for e in snapshot if e["file"] == FILE]


def _entry(captured: dict) -> dict:
    return {"file": FILE, "kind": "launch", "name": "autostart.env",
            "shown": json.dumps(captured, separators=(",", ":"))}


# ------------------------------------------------------------------ emission


def test_a_unit_that_serves_this_dir_becomes_exactly_one_launch_entry(g):
    _install(g)

    assert _mine(trust.machine_snapshot(g)) == [{
        "file": "systemd/localharness-llm.service", "kind": "launch", "name": "autostart.env",
        "shown": '{"PATH":"/usr/bin:/bin","HF_HOME":"/h/hf"}'}]


def test_the_constants_name_that_entry(g):
    assert trust.AUTOSTART_FILE == FILE
    assert trust.AUTOSTART_ENTRY_NAME == "autostart.env"
    assert trust.AUTOSTART_KEY == KEY


def test_the_snapshot_stays_sorted_with_the_entry_in_it(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm"}})
    _install(g)

    snap = trust.machine_snapshot(g)

    assert snap == sorted(snap, key=trust._entry_key)
    assert len(_mine(snap)) == 1 and len(snap) > 1


def test_all_three_names_keep_the_fixed_key_order(g):
    _install(g, {"PATH": PATH_VALUE, "HF_HOME": HF, "LD_LIBRARY_PATH": LD})

    shown = _mine(trust.machine_snapshot(g))[0]["shown"]

    assert list(json.loads(shown)) == ["PATH", "HF_HOME", "LD_LIBRARY_PATH"]
    assert shown == json.dumps({"PATH": PATH_VALUE, "HF_HOME": HF, "LD_LIBRARY_PATH": LD},
                               separators=(",", ":"))


def test_the_same_unit_always_fingerprints_the_same(g):
    _install(g, {"PATH": PATH_VALUE, "LD_LIBRARY_PATH": LD, "HF_HOME": HF})
    first = trust.fingerprint(trust.machine_snapshot(g))
    _install(g, {"LD_LIBRARY_PATH": LD, "HF_HOME": HF, "PATH": PATH_VALUE})

    assert trust.fingerprint(trust.machine_snapshot(g)) == first


def test_only_path_captured_gives_a_one_key_object(g):
    _install(g, {"PATH": PATH_VALUE})

    assert _mine(trust.machine_snapshot(g)) == [_entry({"PATH": PATH_VALUE})]


# ------------------------------------------------------------------ the empty and adjacent cases


def test_a_second_config_dir_the_unit_does_not_name_gets_no_entry(g, tmp_path):
    other = tmp_path / "other-global"
    other.mkdir()
    _install(g)

    assert len(_mine(trust.machine_snapshot(g))) == 1
    assert _mine(trust.machine_snapshot(other)) == []


def test_no_unit_file_gives_no_entry(g):
    assert not u.harness_unit_path().exists()

    assert _mine(trust.machine_snapshot(g)) == []


def test_an_unparsable_env_block_gives_no_entry(g):
    body = _body(g).replace("# env PATH=/usr/bin:/bin", "# env PATH=relative/bin")
    u.write_harness_unit(body)
    assert u.captured_env(body) is None

    assert _mine(trust.machine_snapshot(g)) == []


def test_a_unit_whose_execstart_names_another_dir_gives_no_entry(g, tmp_path):
    u.write_harness_unit(_body(tmp_path / "elsewhere"))

    assert _mine(trust.machine_snapshot(g)) == []


def test_removing_the_unit_removes_the_entry(g):
    _install(g)
    assert _mine(trust.machine_snapshot(g))

    assert u.remove_harness_unit() is True

    assert _mine(trust.machine_snapshot(g)) == []


# ------------------------------------------------------------------ no new kind


def test_the_entry_is_a_launch_entry_and_no_kind_was_added(g):
    _install(g)

    assert _mine(trust.machine_snapshot(g))[0]["kind"] == trust.LAUNCH_KIND
    assert trust.MACHINE_KINDS == frozenset({
        "mcp_server", "embedding_model", "permission", "tool_script", "org_permission",
        "launch", "launch_model"})


def test_confirmed_at_upgrade_never_includes_it(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm"}})
    _install(g)
    snap = trust.machine_snapshot(g)

    confirmed = trust.confirmed_at_upgrade(snap)

    assert confirmed, "the runtime entry is confirmed at the upgrade start"
    assert KEY not in {(e["file"], e["kind"], e["name"]) for e in confirmed}
    assert KEY in {(e["file"], e["kind"], e["name"]) for e in snap}


def test_recorded_autostart_entry_is_none_without_a_record_or_an_entry(g):
    assert trust.recorded_autostart_entry(g) is None

    trust.record_machine(g, trust.machine_snapshot(g))

    assert trust.recorded_autostart_entry(g) is None


def test_recorded_autostart_entry_returns_the_recorded_one(g):
    _install(g)
    trust.record_machine(g, trust.machine_snapshot(g))

    assert trust.recorded_autostart_entry(g) == _entry({"PATH": PATH_VALUE, "HF_HOME": HF})
