"""A 0.17 machine record upgrades with at most the one ruled question (Phase 4, plan 03).

The decision half of the machine trust: `decide_machine_trust` confirms what governed 0.17's
launches once (`confirmed_at_upgrade`), records `launch_model` in `kinds` even when the upgrade
start withholds, keeps a llama.cpp `bind_all: true` quiet while nothing is launching and out of
every record, words the question for it, and remembers a No by writing the narrow value.

The global dir is conftest's hermetic LOCALHARNESS_HOME; answers are `rich.prompt.Confirm.ask`.
"""
from __future__ import annotations

import pytest

from localharness.cli import workspace as ws_mod
from localharness.cli.workspace import (
    MACHINE_DECLINED,
    MACHINE_DECLINED_EARLIER,
    MACHINE_UNASKED,
    MachineTrust,
    decide_machine_trust,
)
from localharness.config import trust
from tests.unit.trust_support import answers, by_name, g, never_asked, write_yaml  # noqa: F401

PROVIDER = {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8081/v1", "default_model": "m"}
LOOSENING = {"permissions": {"workspace_root": "/"}}
OLD_LAUNCH_KEYS = ("binary", "docker_image", "extra_args")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    ws_mod._DECLINED.clear()
    yield
    ws_mod._DECLINED.clear()


def _config(g, **sections) -> None:
    write_yaml(g / "config.yaml", {"provider": PROVIDER, **sections})


def _pre_upgrade(g) -> None:
    """The record 0.17 wrote: every entry the old tables knew, and no `launch_model` kind."""
    snap = trust.machine_snapshot(g)
    old = [e for e in snap if e["kind"] != trust.LAUNCH_MODEL_KIND
           and (e["kind"] != trust.LAUNCH_KIND or e["name"].rsplit(".", 1)[-1] in OLD_LAUNCH_KEYS)]
    trust.record_machine(g, old, kinds=trust.MACHINE_KINDS - {trust.LAUNCH_MODEL_KIND})


def _record_bytes() -> bytes:
    return trust.trust_store_path().read_bytes()


def _recorded_names(g) -> set[str]:
    return {e["name"] for e in trust.recorded_machine(g)["entries"]}


LLAMACPP = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf", "bind_all": True}
VLLM = {"runtime": "vllm", "binary": "/v/vllm", "model": "org/m", "bind_all": True,
        "require_api_key": False, "local_models": []}
BIND_ALL = ("config.yaml", "launch", "server.bind_all")


# ------------------------------------------------------------------ the upgrade start


def test_a_vllm_record_upgrades_with_no_question(g, monkeypatch):
    _config(g, server=VLLM)
    _pre_upgrade(g)
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=True)

    assert got == MachineTrust()
    rec = trust.recorded_machine(g)
    assert trust.LAUNCH_MODEL_KIND in rec["kinds"]
    assert {"server.model", "server.runtime", "server.bind_all"} <= _recorded_names(g)
    assert by_name(rec["entries"], "server.bind_all")["shown"] == "vllm:true:open"


def test_a_record_that_knows_launch_model_confirms_nothing_unasked(g, monkeypatch):
    _config(g, server={**VLLM, "bind_all": False})
    trust.record_machine(g, trust.machine_snapshot(g))
    write_yaml(g / "config.yaml", {"provider": PROVIDER, "server": VLLM})  # bind_all false -> true
    asked = answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True)

    assert len(asked) == 1 and "server.bind_all" in asked[0]
    assert got.withheld == {"config.yaml": frozenset({("launch", "server.bind_all")})}


def test_a_changed_peer_binary_asks_at_the_upgrade_start(g, monkeypatch):
    peer = {"name": "p", "base_url": "http://127.0.0.1:8082/v1", "gpu": True,
            "lifecycle": {"runtime": "vllm", "binary": "/v/old", "model": "org/m"}}
    _config(g, extra_endpoints=[peer])
    _pre_upgrade(g)
    peer["lifecycle"]["binary"] = "/v/new"
    _config(g, extra_endpoints=[peer])
    asked = answers(monkeypatch, True)

    got = decide_machine_trust(g, ask=True)

    assert got == MachineTrust()
    assert len(asked) == 1
    assert "extra_endpoints.p.lifecycle.binary" in asked[0] and "/v/new" in asked[0]
    assert "server.model" not in asked[0] and "lifecycle.model" not in asked[0]


def test_a_changed_server_model_asks_once_the_kind_is_known(g, monkeypatch):
    _config(g, server={**VLLM, "model": "org/a"})
    trust.record_machine(g, trust.machine_snapshot(g))
    _config(g, server={**VLLM, "model": "org/b"})
    asked = answers(monkeypatch, True)

    decide_machine_trust(g, ask=True)

    assert len(asked) == 1 and "~ server.model (config.yaml): org/b" in asked[0]


# ------------------------------------------------------------------ the llama.cpp true is quiet


def test_a_llamacpp_bind_all_true_is_quiet_while_nothing_launches(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=True, launching=False)

    assert got.withheld == {"config.yaml": frozenset({("launch", "server.bind_all")})}
    assert got.line == "" and got.quiet == frozenset({BIND_ALL})
    assert trust.LAUNCH_MODEL_KIND in trust.recorded_machine(g)["kinds"]
    assert "server.bind_all" not in _recorded_names(g)
    before = _record_bytes()

    again = decide_machine_trust(g, ask=True, launching=False)

    assert again == got
    assert _record_bytes() == before


def test_a_start_that_cannot_ask_names_the_llamacpp_entry_and_records_the_kind(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=False, launching=True)

    assert got.withheld == {"config.yaml": frozenset({("launch", "server.bind_all")})}
    assert MACHINE_UNASKED in got.line and "server.bind_all (config.yaml)" in got.line
    rec = trust.recorded_machine(g)
    assert set(rec["kinds"]) == trust.MACHINE_KINDS and "server.bind_all" not in _recorded_names(g)
    before = _record_bytes()

    again = decide_machine_trust(g, ask=False, launching=True)

    assert again.withheld == got.withheld and MACHINE_UNASKED in again.line
    assert _record_bytes() == before


def test_a_yes_to_another_change_never_adopts_a_quiet_entry(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    write_yaml(g / "org.yaml", LOOSENING)
    asked = answers(monkeypatch, True)

    got = decide_machine_trust(g, ask=True, launching=False)

    assert len(asked) == 1
    assert "permissions.workspace_root" in asked[0] and "bind_all" not in asked[0]
    assert got.quiet == frozenset({BIND_ALL}) and got.line == ""
    assert "permissions.workspace_root" in _recorded_names(g)
    assert "server.bind_all" not in _recorded_names(g)


def test_a_removal_beside_a_quiet_entry_is_recorded(g, monkeypatch):
    write_yaml(g / "org.yaml", LOOSENING)
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    assert "permissions.workspace_root" in _recorded_names(g)
    (g / "org.yaml").unlink()
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=True, launching=False)

    assert got.quiet == frozenset({BIND_ALL})
    assert "permissions.workspace_root" not in _recorded_names(g)
    assert "server.bind_all" not in _recorded_names(g)


def test_an_uncovered_non_posture_entry_beside_a_quiet_one_is_asked_about_alone(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    _config(g, server={**LLAMACPP, "binary": "/x/other"})
    asked = answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=False)

    assert len(asked) == 1 and "server.binary" in asked[0] and "bind_all" not in asked[0]
    assert "server.bind_all" not in got.line and MACHINE_DECLINED in got.line
    assert got.withheld == {"config.yaml": frozenset({("launch", "server.binary"),
                                                       ("launch", "server.bind_all")})}
    assert got.quiet == frozenset({BIND_ALL})


def test_two_identical_quiet_entries_are_both_quiet(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    snap = trust.machine_snapshot(g)
    twin = next(e for e in snap if e["name"] == "server.bind_all")
    monkeypatch.setattr(trust, "machine_snapshot", lambda _g: sorted([*snap, dict(twin)], key=trust._entry_key))
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=True, launching=False)

    assert got.line == "" and got.quiet == frozenset({BIND_ALL})
    assert "server.bind_all" not in _recorded_names(g)


def test_a_peer_with_a_remembered_no_is_quiet_and_a_different_value_is_asked(g, monkeypatch):
    peer = {"name": "p", "base_url": "http://127.0.0.1:8082/v1", "gpu": True,
            "lifecycle": {"runtime": "llamacpp", "binary": "/x/b", "model": "/m/b.gguf", "bind_all": True}}
    _config(g, extra_endpoints=[peer])
    _pre_upgrade(g)
    name = "extra_endpoints.p.lifecycle.bind_all"
    trust.record_offer_decline(trust.decline_key("<peer>", name, "llamacpp:true:open"))
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert got.line == "" and got.quiet == frozenset({("config.yaml", "launch", name)})

    # the same peer now keyed: a different shown, a new question
    peer["api_key"] = "k"
    peer["lifecycle"]["require_api_key"] = True
    _config(g, extra_endpoints=[peer])
    asked = answers(monkeypatch, False)

    decide_machine_trust(g, ask=True, launching=True)

    assert len(asked) == 1 and name in asked[0]


# ------------------------------------------------------------------ a No is declined per visible snapshot


def test_the_declined_set_is_keyed_on_the_visible_snapshot(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    write_yaml(g / "org.yaml", LOOSENING)
    answers(monkeypatch, False)

    first = decide_machine_trust(g, ask=True, launching=False)
    never_asked(monkeypatch)
    second = decide_machine_trust(g, ask=True, launching=False)

    assert MACHINE_DECLINED in first.line
    assert MACHINE_DECLINED_EARLIER in second.line
    assert second.quiet == frozenset({BIND_ALL})
