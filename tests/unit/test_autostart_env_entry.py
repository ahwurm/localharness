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

from localharness.cli import workspace as ws_mod
from localharness.config import loader as loader_mod
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


# ------------------------------------------------------------------ the entry through every decision


def _decide(g, *, ask=False):
    return ws_mod.decide_machine_trust(g, ask=ask)


def _adopted_unit(g, captured=None) -> None:
    """A record made first (no unit), the unit written, its entry adopted by the act that wrote it."""
    assert _decide(g) == ws_mod.MachineTrust()
    _install(g, captured)
    trust.adopt_entries(g, trust.autostart_entries(g))


def _edit_path_line(new: str) -> None:
    body = u.read_harness_unit()
    u.write_harness_unit(body.replace(f"# env PATH={PATH_VALUE}", f"# env PATH={new}"))


@pytest.fixture(autouse=True)
def _fresh_declines():
    ws_mod._DECLINED.clear()
    yield
    ws_mod._DECLINED.clear()


def test_an_adopted_entry_is_recorded_and_survives_every_re_record(g, monkeypatch):
    never_asked(monkeypatch)
    _adopted_unit(g)
    snap_entry = _mine(trust.machine_snapshot(g))[0]
    assert trust.recorded_autostart_entry(g) == snap_entry

    for _ in range(2):
        result = _decide(g)
        assert result.withheld == {} and result.line == ""
        assert trust.recorded_autostart_entry(g) == snap_entry


def test_a_planted_unit_is_withheld_with_the_real_path_in_the_line(g, monkeypatch):
    never_asked(monkeypatch)
    assert _decide(g) == ws_mod.MachineTrust()
    _install(g)

    result = _decide(g)

    assert result.withheld == {FILE: frozenset({("launch", "autostart.env")})}
    assert f"autostart.env ({u.harness_unit_path()})" in result.line
    assert FILE not in result.line.replace(str(u.harness_unit_path()), "")
    assert trust.recorded_autostart_entry(g) is None


def test_the_withheld_line_names_the_path_under_a_redirected_config_home(g, tmp_path, monkeypatch):
    never_asked(monkeypatch)
    _decide(g)
    _install(g)

    line = _decide(g).line

    assert f"autostart.env ({tmp_path / 'cfg' / 'systemd' / 'user' / 'localharness-llm.service'})" in line


def test_a_planted_unit_is_still_withheld_at_the_upgrade_start(g, monkeypatch):
    """A record without `launch_model` in its kinds is the upgrade start: `confirmed_at_upgrade`
    counts what governed 0.17's launches as confirmed, and never this entry."""
    never_asked(monkeypatch)
    trust.record_machine(g, trust.machine_snapshot(g),
                         kinds=frozenset(trust.MACHINE_KINDS - {trust.LAUNCH_MODEL_KIND}))
    _install(g)

    result = _decide(g)

    assert result.withheld == {FILE: frozenset({("launch", "autostart.env")})}
    assert trust.recorded_autostart_entry(g) is None
    assert trust.LAUNCH_MODEL_KIND in trust.recorded_machine(g)["kinds"]


def test_a_changed_env_line_is_withheld_without_a_terminal(g, monkeypatch):
    never_asked(monkeypatch)
    _adopted_unit(g)
    adopted = trust.recorded_autostart_entry(g)
    _edit_path_line("/evil/bin:/usr/bin")

    result = _decide(g)

    assert result.withheld == {FILE: frozenset({("launch", "autostart.env")})}
    assert f"autostart.env ({u.harness_unit_path()})" in result.line
    assert trust.recorded_autostart_entry(g) == adopted


def test_a_changed_env_line_is_asked_with_the_unit_path_and_a_no_withholds_it(g, monkeypatch):
    _adopted_unit(g)
    _edit_path_line("/evil/bin:/usr/bin")
    questions: list[str] = []

    def no(question: str) -> bool:
        questions.append(question)
        return False

    monkeypatch.setattr(ws_mod, "_confirm_or_no", no)

    result = _decide(g, ask=True)

    assert len(questions) == 1
    question = questions[0]
    assert f"(the unit file {u.harness_unit_path()})" in question
    line = next(row for row in question.splitlines() if "autostart.env" in row)
    assert line.lstrip().startswith("~ autostart.env")
    assert "/evil/bin:/usr/bin" in line
    assert result.withheld == {FILE: frozenset({("launch", "autostart.env")})}


def test_a_yes_records_the_changed_environment(g, monkeypatch):
    _adopted_unit(g)
    _edit_path_line("/new/bin:/usr/bin")
    monkeypatch.setattr(ws_mod, "_confirm_or_no", lambda question: True)

    result = _decide(g, ask=True)

    assert result.withheld == {}
    assert json.loads(trust.recorded_autostart_entry(g)["shown"])["PATH"] == "/new/bin:/usr/bin"


def test_the_pseudo_file_is_never_part_of_the_question_without_the_real_path(g, monkeypatch):
    _adopted_unit(g)
    _edit_path_line("/evil/bin:/usr/bin")
    seen: list[str] = []
    monkeypatch.setattr(ws_mod, "_confirm_or_no", lambda q: seen.append(q) or False)

    _decide(g, ask=True)

    assert str(u.harness_unit_path()) in seen[0]


def _harness_view(g, withheld=None):
    harness = loader_mod.ConfigLoader(config_dir=g, machine_withheld=withheld).load_harness()
    return (harness.server.model_dump(), [p.model_dump() for p in harness.extra_endpoints],
            harness.org.model_dump())


def test_a_withheld_autostart_entry_drops_nothing_the_loader_reads(g):
    write_yaml(g / "config.yaml", {
        "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:8000/v1",
                     "default_model": "m"},
        "server": {"runtime": "vllm", "binary": "/usr/bin/vllm", "model": "m"},
        "extra_endpoints": [{"name": "peer", "base_url": "http://127.0.0.1:9/v1", "gpu": True,
                             "lifecycle": {"runtime": "vllm", "binary": "/usr/bin/vllm", "model": "p", "gpu": True}}],
        "org": {"permissions": {"mode": "unattended"}}})
    write_yaml(g / "overrides.yaml", {"server": {"port": 8123}})
    _install(g)

    whole = _harness_view(g)
    withheld = _harness_view(g, {FILE: frozenset({("launch", "autostart.env")})})

    assert withheld == whole
    assert whole[0]["port"] == 8123 and whole[1][0]["name"] == "peer"


def test_the_pseudo_file_matches_no_real_settings_file_name():
    assert FILE not in ("config.yaml", "overrides.yaml", "org.yaml")
    assert not FILE.startswith(("agents/", "divisions/", "tools/"))


def test_removing_the_unit_records_the_removal_without_a_question(g, monkeypatch):
    never_asked(monkeypatch)
    _adopted_unit(g)
    assert u.remove_harness_unit() is True

    result = _decide(g)

    assert result == ws_mod.MachineTrust()
    assert trust.recorded_autostart_entry(g) is None
    assert KEY not in {(e["file"], e["kind"], e["name"]) for e in trust.recorded_machine(g)["entries"]}


def test_forget_entries_removes_exactly_the_named_entries(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm"}})
    _install(g)
    trust.record_machine(g, trust.machine_snapshot(g))
    before = trust.recorded_machine(g)

    trust.forget_entries(g, [KEY])

    after = trust.recorded_machine(g)
    assert [e for e in before["entries"] if e not in after["entries"]] == [
        e for e in before["entries"] if (e["file"], e["kind"], e["name"]) == KEY]
    assert after["entries"] == [e for e in before["entries"] if (e["file"], e["kind"], e["name"]) != KEY]
    assert after["kinds"] == before["kinds"]
    assert after["fingerprint"] == trust.fingerprint(after["entries"])
    assert after["fingerprint"] != before["fingerprint"]


def test_forget_entries_with_no_record_is_a_no_op(g):
    existed = trust.trust_store_path().exists()

    trust.forget_entries(g, [KEY])

    assert trust.recorded_machine(g) is None
    assert trust.trust_store_path().exists() == existed


def test_forget_entries_ignores_keys_that_are_not_recorded(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm"}})
    trust.record_machine(g, trust.machine_snapshot(g))
    before = trust.recorded_machine(g)

    trust.forget_entries(g, [KEY, ("nowhere.yaml", "launch", "server.binary")])

    assert trust.recorded_machine(g) == before
