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


# ------------------------------------------------------------------ annotations on diff lines


def test_an_annotation_is_appended_to_its_line_only(g):
    new = [{"file": "config.yaml", "kind": "launch", "name": "server.bind_all", "shown": "llamacpp:true:open"},
           {"file": "config.yaml", "kind": "launch", "name": "server.binary", "shown": "/x"}]
    before = [dict(e) for e in new]

    plain = ws_mod.machine_diff([], new)
    dashed = ws_mod.machine_diff([], new, annotations={BIND_ALL: "X"})
    bracket = ws_mod.machine_diff([], new, annotations={BIND_ALL: "(from Y)"})

    assert dashed.splitlines()[0] == "  + server.bind_all (config.yaml): llamacpp:true:open — X"
    assert bracket.splitlines()[0] == "  + server.bind_all (config.yaml): llamacpp:true:open (from Y)"
    assert dashed.splitlines()[1:] == plain.splitlines()[1:]
    assert new == before


# ------------------------------------------------------------------ the question and a remembered No


def _audit(g) -> list[dict]:
    import json

    path = g / "audit.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("event_type") == "ComponentMutated"]


def _overrides(g) -> dict:
    import yaml

    path = g / "overrides.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _peers(*, bind_all=True, keyed=False) -> list[dict]:
    peer = {"name": "p", "base_url": "http://127.0.0.1:8082/v1", "gpu": True, "api_key": "SECRET-KEY",
            "lifecycle": {"runtime": "llamacpp", "binary": "/x/b", "model": "/m/b.gguf",
                          "bind_all": bind_all, "port": 8082}}
    if keyed:
        peer["lifecycle"]["require_api_key"] = True
    other = {"name": "q", "base_url": "http://127.0.0.1:8083/v1", "gpu": True,
             "lifecycle": {"runtime": "llamacpp", "binary": "/x/c", "model": "/m/c.gguf",
                           "bind_all": False, "port": 8083}}
    return [peer, other]


def _upgrade_asking(g, monkeypatch, *replies, **config) -> tuple[MachineTrust, list[str]]:
    _config(g, **config)
    _pre_upgrade(g)
    asked = answers(monkeypatch, *replies)
    return decide_machine_trust(g, ask=True, launching=True), asked


def test_the_question_says_what_yes_does_to_a_llamacpp_true(g, monkeypatch):
    got, asked = _upgrade_asking(g, monkeypatch, True, server=LLAMACPP)

    assert got == MachineTrust()
    assert ("server.bind_all: true was never applied to llama.cpp; Yes binds every interface, no key"
            in asked[0])
    assert "server.bind_all" in _recorded_names(g)


def test_the_no_key_clause_goes_when_a_key_is_required(g, monkeypatch):
    _, asked = _upgrade_asking(g, monkeypatch, True, server={
        **LLAMACPP, "require_api_key": True, "api_key_file": "/k"})

    assert asked[0].count("Yes binds every interface") == 1
    assert "Yes binds every interface, no key" not in asked[0]
    assert "llamacpp:true:keyed" in asked[0]


def test_the_peer_form_names_the_peers_lifecycle(g, monkeypatch):
    _, asked = _upgrade_asking(g, monkeypatch, True, extra_endpoints=_peers())

    assert ("extra_endpoints.p.lifecycle.bind_all: true was never applied to llama.cpp; "
            "Yes binds every interface, no key") in asked[0]


def test_a_no_writes_false_adopts_it_and_is_never_asked_again(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    asked = answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert len(asked) == 1
    assert _overrides(g) == {"server": {"bind_all": False}}
    shown = {(e["file"], e["name"]): e["shown"] for e in trust.recorded_machine(g)["entries"]}
    assert shown[("overrides.yaml", "server.bind_all")] == "llamacpp:false:open"
    assert not any(v.startswith("llamacpp:true") for v in shown.values())
    assert got.withheld == {"config.yaml": frozenset({("launch", "server.bind_all")})}
    assert got.line == "" and MACHINE_DECLINED not in got.line
    never_asked(monkeypatch)
    for ask, launching in ((True, True), (False, False), (True, False)):
        assert decide_machine_trust(g, ask=ask, launching=launching) == MachineTrust()


def test_deleting_the_override_asks_again_and_never_widens(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    answers(monkeypatch, False)
    decide_machine_trust(g, ask=True, launching=True)
    (g / "overrides.yaml").unlink()
    ws_mod._DECLINED.clear()  # the next process: the in-session No is not what is under test
    asked = answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert len(asked) == 1 and "server.bind_all: true was never applied" in asked[0]
    assert _overrides(g) == {"server": {"bind_all": False}}
    assert got.line == ""


def test_a_no_beside_an_mcp_server_declines_it_and_remembers_the_posture(g, monkeypatch):
    server = {"name": "g2", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}
    write_yaml(g / "agents" / "orchestrator.yaml", {"name": "orchestrator", "role": "R",
                                                     "tools": {"mcp_servers": []}})
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    write_yaml(g / "agents" / "orchestrator.yaml", {"name": "orchestrator", "role": "R",
                                                     "tools": {"mcp_servers": [server]}})
    answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert "g2 (agents/orchestrator.yaml)" in got.line and MACHINE_DECLINED in got.line
    assert "bind_all" not in got.line and "1 change(s)" in got.line
    assert _overrides(g)["server"]["bind_all"] is False
    answers(monkeypatch, False)
    again = decide_machine_trust(g, ask=True, launching=True)
    assert "g2" in again.line and "bind_all" not in again.line


def test_a_peer_no_writes_the_peers_false_by_name_and_remembers_it(g, monkeypatch):
    _config(g, extra_endpoints=_peers())
    _pre_upgrade(g)
    answers(monkeypatch, False)
    name = "extra_endpoints.p.lifecycle.bind_all"

    got = decide_machine_trust(g, ask=True, launching=True)

    listed = {p["name"]: p for p in _overrides(g)["extra_endpoints"]}
    assert listed["p"]["lifecycle"]["bind_all"] is False and listed["q"]["lifecycle"]["bind_all"] is False
    assert listed["p"]["lifecycle"]["binary"] == "/x/b" and listed["p"]["api_key"] == "SECRET-KEY"
    shown = {(e["file"], e["name"]): e["shown"] for e in trust.recorded_machine(g)["entries"]}
    assert shown[("overrides.yaml", name)] == "llamacpp:false:open"
    assert trust.offer_was_declined(trust.decline_key("<peer>", name, "llamacpp:true:open"))
    assert got.line == ""
    never_asked(monkeypatch)
    for ask, launching in ((True, True), (False, False), (True, False)):
        assert decide_machine_trust(g, ask=ask, launching=launching) == MachineTrust()


# ------------------------------------------------------------------ the writer: validated, audited


def test_a_refused_validation_leaves_the_overlay_and_the_audit_log_alone(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    write_yaml(g / "overrides.yaml", {"org": {"default_model": "m"}})
    before = (g / "overrides.yaml").read_bytes()

    def refuse(*_a, **_kw):
        raise ValueError("would not load")

    monkeypatch.setattr("localharness.config.models.HarnessConfig.model_validate", refuse)
    answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert (g / "overrides.yaml").read_bytes() == before and _audit(g) == []
    assert "server.bind_all (config.yaml)" in got.line and MACHINE_DECLINED in got.line


def test_a_config_without_a_provider_is_refused_by_design(g, monkeypatch):
    write_yaml(g / "config.yaml", {"server": LLAMACPP})
    _pre_upgrade(g)
    answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert not (g / "overrides.yaml").exists() and _audit(g) == []
    assert "server.bind_all (config.yaml)" in got.line


def test_an_unparseable_overlay_is_not_written_over(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    (g / "overrides.yaml").write_text("server: [unclosed\n", encoding="utf-8")
    answers(monkeypatch, False)

    decide_machine_trust(g, ask=True, launching=True)

    assert (g / "overrides.yaml").read_text(encoding="utf-8") == "server: [unclosed\n"
    assert _audit(g) == []


def test_the_primary_no_is_audited_once(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    answers(monkeypatch, False)

    decide_machine_trust(g, ask=True, launching=True)

    (event,) = _audit(g)
    assert event["path"] == "server.bind_all" and event["before_value"] is None
    assert event["after_value"] is False and event["layer"] == "user" and event["actor"] == "cli"
    assert event["actor_detail"]


def test_the_primary_audit_records_the_value_the_overlay_held(g, monkeypatch):
    _config(g, server={**LLAMACPP, "bind_all": False})
    write_yaml(g / "overrides.yaml", {"server": {"bind_all": True}})
    _pre_upgrade(g)
    answers(monkeypatch, False)

    decide_machine_trust(g, ask=True, launching=True)

    (event,) = _audit(g)
    assert event["before_value"] is True and event["after_value"] is False


def test_the_peer_no_is_audited_by_name_and_never_leaks_the_peers_key(g, monkeypatch):
    _config(g, extra_endpoints=_peers())
    _pre_upgrade(g)
    answers(monkeypatch, False)

    decide_machine_trust(g, ask=True, launching=True)

    (event,) = _audit(g)
    assert event["path"] == "extra_endpoints.p.lifecycle.bind_all"
    assert event["before_value"] is True and event["after_value"] is False and event["actor"] == "cli"
    assert "SECRET-KEY" not in (g / "audit.jsonl").read_text(encoding="utf-8")


def test_a_null_audit_path_still_writes_and_adopts(g, monkeypatch):
    _config(g, server=LLAMACPP, org={"audit_log_path": None})
    _pre_upgrade(g)
    answers(monkeypatch, False)

    decide_machine_trust(g, ask=True, launching=True)

    assert _overrides(g) == {"server": {"bind_all": False}}
    assert not (g / "audit.jsonl").exists()
    never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=True, launching=True) == MachineTrust()


def test_an_emit_that_fails_never_undoes_the_write(g, monkeypatch):
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)

    async def boom(*_a, **_kw):
        raise OSError("audit log unwritable")

    monkeypatch.setattr("localharness.core.bus.EventBus.publish", boom)
    answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert _overrides(g) == {"server": {"bind_all": False}} and got.line == ""
    never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=True, launching=True) == MachineTrust()


async def test_the_audit_line_is_written_from_inside_a_running_loop(g):
    _config(g, server=LLAMACPP)

    ok = ws_mod.write_overlay_validated(
        g, lambda overlay: overlay.setdefault("server", {}).update(bind_all=False),
        [("server.bind_all", None, False)], actor_detail="test")

    assert ok is True
    (event,) = _audit(g)
    assert event["path"] == "server.bind_all" and event["actor_detail"] == "test"


# ------------------------------------------------------------------ the fresh-install exception (D-24)

KEYED = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
         "require_api_key": True, "api_key_file": "/k", "port": 8080}
SUFFIX = "(read at start and sent as the bearer to :{port})"
MCP = {"name": "g1", "transport": "stdio", "command": "/bin/echo", "args": ["hi"]}


def _with_mcp(g) -> None:
    write_yaml(g / "agents" / "orchestrator.yaml", {"name": "orchestrator", "role": "R",
                                                     "tools": {"mcp_servers": [MCP]}})


def _launch_names(snapshot_or_set) -> set[str]:
    return {n for k, n in snapshot_or_set if k in ("launch", "launch_model")}


def test_a_headless_first_start_withholds_the_launch_entries_and_records_the_kinds(g, monkeypatch):
    _config(g, server=KEYED)
    _with_mcp(g)
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=False)

    held = got.withheld["config.yaml"]
    assert {"server.binary", "server.model", "server.api_key_file", "server.runtime"} <= {n for _, n in held}
    assert all(k in ("launch", "launch_model") for k, _ in held) and "agents/orchestrator.yaml" not in got.withheld
    assert MACHINE_UNASKED in got.line
    rec = trust.recorded_machine(g)
    assert rec["kinds"] == sorted(trust.MACHINE_KINDS)
    assert {(e["kind"], e["name"]) for e in rec["entries"]} == {("mcp_server", "g1")}
    before = _record_bytes()

    again = decide_machine_trust(g, ask=False)

    assert again.withheld == got.withheld and _record_bytes() == before


def test_the_next_terminal_start_asks_with_the_key_file_line_explained(g, monkeypatch):
    _config(g, server=KEYED)
    _with_mcp(g)
    decide_machine_trust(g, ask=False)
    asked = answers(monkeypatch, True)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert got == MachineTrust() and len(asked) == 1
    assert f"+ server.api_key_file (config.yaml): /k {SUFFIX.format(port=8080)}" in asked[0]
    assert "g1" not in asked[0]
    assert "server.binary" in asked[0] and "server.model" in asked[0]
    assert {"server.binary", "server.api_key_file", "server.model"} <= _recorded_names(g)
    never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=True, launching=True) == MachineTrust()


def test_a_terminal_first_start_asks_over_the_whole_snapshot(g, monkeypatch):
    _config(g, server=KEYED)
    _with_mcp(g)
    asked = answers(monkeypatch, True)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert got == MachineTrust() and len(asked) == 1
    assert SUFFIX.format(port=8080) in asked[0] and "g1" in asked[0]
    assert "g1" in {e["name"] for e in trust.recorded_machine(g)["entries"]}
    assert trust.recorded_machine(g)["kinds"] == sorted(trust.MACHINE_KINDS)


def test_a_no_at_a_terminal_first_start_records_nothing_and_withholds_all(g, monkeypatch):
    _config(g, server=KEYED)
    _with_mcp(g)
    answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True, launching=True)

    assert trust.recorded_machine(g) is None
    assert "agents/orchestrator.yaml" in got.withheld and "config.yaml" in got.withheld


def test_a_server_without_a_port_names_the_default_port(g, monkeypatch):
    from localharness.config.models import ManagedServerConfig

    _config(g, server={k: v for k, v in KEYED.items() if k != "port"})
    asked = answers(monkeypatch, True)

    decide_machine_trust(g, ask=True, launching=True)

    port = ManagedServerConfig.model_fields["port"].default
    assert SUFFIX.format(port=port) in asked[0]


def test_the_overrides_port_wins_in_the_suffix(g):
    _config(g, server=KEYED)
    write_yaml(g / "overrides.yaml", {"server": {"port": 9191}})

    assert ws_mod.launch_port(g, "server.api_key_file") == 9191


def test_a_peer_key_file_is_the_same_exception_with_its_own_port(g, monkeypatch):
    peer = {"name": "qwen3.6", "base_url": "http://127.0.0.1:8090/v1", "gpu": True,
            "lifecycle": {"runtime": "llamacpp", "binary": "/x/b", "model": "/m/b.gguf",
                          "require_api_key": True, "api_key_file": "/pk", "port": 8090}}
    _config(g, extra_endpoints=[peer])
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=False)

    assert ("launch", "extra_endpoints.qwen3.6.lifecycle.api_key_file") in got.withheld["config.yaml"]
    assert trust.recorded_machine(g)["entries"] == []
    asked = answers(monkeypatch, True)
    decide_machine_trust(g, ask=True, launching=True)
    assert f"/pk {SUFFIX.format(port=8090)}" in asked[0]


def test_a_first_start_with_no_key_file_is_adopted_silently(g, monkeypatch):
    _config(g, server={k: v for k, v in LLAMACPP.items()})
    never_asked(monkeypatch)

    assert decide_machine_trust(g, ask=False) == MachineTrust()
    assert "server.binary" in _recorded_names(g)


# ------------------------------------------------------------------ planted runtime, deleted key, empty list


def test_a_planted_vllm_runtime_does_not_launder_the_llamacpp_true(g, monkeypatch):
    _config(g, server={"binary": "/x/llama-server", "model": "/m/a.gguf", "runtime": "llamacpp",
                       "bind_all": True})
    write_yaml(g / "overrides.yaml", {"server": {"runtime": "vllm"}})
    _pre_upgrade(g)
    never_asked(monkeypatch)

    assert decide_machine_trust(g, ask=True, launching=True) == MachineTrust()
    assert by_name(trust.recorded_machine(g)["entries"], "server.bind_all")["shown"] == "vllm:true:open"

    (g / "overrides.yaml").unlink()
    asked = answers(monkeypatch, False)
    decide_machine_trust(g, ask=True, launching=True)

    assert len(asked) == 1 and "server.bind_all (config.yaml): llamacpp:true:open" in asked[0]


def test_a_deleted_require_api_key_is_a_diff_of_the_keyed_field(g, monkeypatch):
    _config(g, server={**LLAMACPP, "require_api_key": True, "api_key_file": "/k"})
    asked = answers(monkeypatch, True)
    trust.record_machine(g, [e for e in trust.machine_snapshot(g) if e["name"] != "server.bind_all"])
    decide_machine_trust(g, ask=True, launching=True)  # a Yes to the keyed true
    assert by_name(trust.recorded_machine(g)["entries"], "server.bind_all")["shown"] == "llamacpp:true:keyed"
    assert len(asked) == 1

    _config(g, server=LLAMACPP)  # require_api_key and api_key_file deleted
    answers_2 = answers(monkeypatch, False)
    got = decide_machine_trust(g, ask=True, launching=True)
    assert "~ server.bind_all (config.yaml): llamacpp:true:open" in answers_2[0]
    assert got.line == ""

    ws_mod._DECLINED.clear()
    (g / "overrides.yaml").unlink()  # the remembered No gone: the same diff, now with nobody to ask
    headless = decide_machine_trust(g, ask=False, launching=True)
    assert "server.bind_all (config.yaml)" in headless.line


def test_an_empty_local_models_list_is_no_entry_and_the_upgrade_asks_nothing(g, monkeypatch):
    _config(g, server=VLLM)
    _pre_upgrade(g)
    never_asked(monkeypatch)

    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert "server.local_models" not in _recorded_names(g)


# ------------------------------------------------------------------ CR-02: every spelling the model reads as true


TRUTHY = [True, 1, "1", "true", "True", "yes", "on"]


@pytest.mark.parametrize("spelling", TRUTHY)
def test_cr02_every_truthy_spelling_is_recorded_as_true(g, spelling):
    _config(g, server={**LLAMACPP, "bind_all": spelling})
    assert by_name(trust.machine_snapshot(g), "server.bind_all")["shown"] == "llamacpp:true:open"


@pytest.mark.parametrize("spelling", [False, 0, "0", "false", "no", "off"])
def test_cr02_every_falsy_spelling_is_recorded_as_false(g, spelling):
    _config(g, server={**LLAMACPP, "bind_all": spelling})
    assert by_name(trust.machine_snapshot(g), "server.bind_all")["shown"] == "llamacpp:false:open"


@pytest.mark.parametrize("spelling", TRUTHY)
def test_cr02_a_non_true_spelling_of_true_is_asked_at_the_upgrade_start_not_adopted(g, monkeypatch, spelling):
    got, asked = _upgrade_asking(g, monkeypatch, False, server={**LLAMACPP, "bind_all": spelling})

    assert len(asked) == 1 and "server.bind_all: true was never applied to llama.cpp" in asked[0]
    # the No wrote `bind_all: false` and recorded that; the `true` was never recorded
    assert by_name(trust.recorded_machine(g)["entries"], "server.bind_all")["shown"] == "llamacpp:false:open"
    assert got.withheld == {"config.yaml": frozenset({("launch", "server.bind_all")})}  # narrowed this run


@pytest.mark.parametrize("spelling", TRUTHY)
def test_cr02_it_is_quiet_while_nothing_launches_like_the_literal_true(g, monkeypatch, spelling):
    _config(g, server={**LLAMACPP, "bind_all": spelling})
    _pre_upgrade(g)
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=True, launching=False)

    assert got.quiet == frozenset({BIND_ALL}) and got.line == ""
    assert "server.bind_all" not in _recorded_names(g)


@pytest.mark.parametrize("shown", ["llamacpp:true:open", 'llamacpp:"yes":open', "llamacpp:1:open",
                                   "llamacpp:[]:open", "llamacpp::open"])
def test_cr02_confirmed_at_upgrade_fails_closed_for_anything_but_a_llamacpp_false(shown):
    entry = {"file": "config.yaml", "kind": "launch", "name": "server.bind_all", "shown": shown}
    assert trust.confirmed_at_upgrade([entry]) == []
    assert trust.confirmed_at_upgrade([{**entry, "shown": "llamacpp:false:open"}]) != []
    assert trust.confirmed_at_upgrade([{**entry, "shown": shown.replace("llamacpp", "vllm", 1)}]) != []


def test_cr02_a_truthy_require_api_key_spelling_reads_keyed(g):
    _config(g, server={**LLAMACPP, "require_api_key": "yes", "api_key_file": "/k"})
    assert by_name(trust.machine_snapshot(g), "server.bind_all")["shown"] == "llamacpp:true:keyed"


# ------------------------------------------------------------------ WR-06: the question names where the bearer goes


def test_wr06_a_loopback_endpoint_keeps_the_servers_port_in_the_suffix(g, monkeypatch):
    _config(g, server=KEYED)
    asked = answers(monkeypatch, True)
    decide_machine_trust(g, ask=True, launching=True)
    assert SUFFIX.format(port=8080) in asked[0]


def test_wr06_a_primary_endpoint_that_is_not_this_machine_is_named_in_the_suffix(g, monkeypatch):
    write_yaml(g / "config.yaml", {"provider": {**PROVIDER, "base_url": "http://192.168.1.50:9000/v1"},
                                   "server": KEYED})
    asked = answers(monkeypatch, True)
    decide_machine_trust(g, ask=True, launching=True)
    assert "(read at start and sent as the bearer to 192.168.1.50:9000)" in asked[0]
    assert ":8080)" not in asked[0]


def test_wr06_an_override_of_provider_base_url_wins_in_the_suffix(g):
    _config(g, server=KEYED)
    write_yaml(g / "overrides.yaml", {"provider": {"base_url": "http://evil.example:443/v1"}})
    assert ws_mod._key_file_destination(g, "server.api_key_file") == "evil.example:443"


def test_wr06_a_peer_endpoint_that_is_not_this_machine_is_named_in_the_suffix(g):
    peer = {"name": "qwen3.6", "base_url": "http://10.0.0.7:8090/v1", "gpu": True,
            "lifecycle": {"runtime": "llamacpp", "binary": "/x/b", "model": "/m/b.gguf",
                          "require_api_key": True, "api_key_file": "/pk", "port": 8090}}
    _config(g, extra_endpoints=[peer])
    assert ws_mod._key_file_destination(g, "extra_endpoints.qwen3.6.lifecycle.api_key_file") == "10.0.0.7:8090"


def test_wr07_a_peer_no_that_forks_the_peer_list_says_so_once(g, monkeypatch):
    _config(g, extra_endpoints=_peers())
    _pre_upgrade(g)
    answers(monkeypatch, False)
    said: list[str] = []
    monkeypatch.setattr(ws_mod._notice_console, "print", lambda *a, **k: said.append(str(a[0])))

    decide_machine_trust(g, ask=True, launching=True)

    assert said == [ws_mod.PEER_LIST_COPIED]
    assert "overrides.yaml" in said[0] and "extra_endpoints" in said[0]
    from pathlib import Path
    assert "copies `config.yaml`'s whole\n`extra_endpoints` list there" in (
        Path(ws_mod.__file__).parents[3] / "SECURITY.md").read_text(encoding="utf-8")


def test_wr07_a_peer_no_in_an_existing_override_list_says_nothing(g, monkeypatch):
    _config(g, extra_endpoints=_peers())
    write_yaml(g / "overrides.yaml", {"extra_endpoints": _peers()})
    _pre_upgrade(g)
    answers(monkeypatch, False)
    said: list[str] = []
    monkeypatch.setattr(ws_mod._notice_console, "print", lambda *a, **k: said.append(str(a[0])))

    decide_machine_trust(g, ask=True, launching=True)

    assert said == []


def test_in02_the_helpers_start_uses_are_public_names_of_workspace():
    import localharness.cli.start_cmd as start_cmd
    for name in ("raw_settings", "uncovered", "visible_text", "write_overlay_validated",
                 "launch_port", "primary_server_port"):
        assert callable(getattr(ws_mod, name)), name
    source = open(start_cmd.__file__, encoding="utf-8").read()
    imported = source[source.index("from localharness.cli.workspace import ("):].split(")", 1)[0]
    assert not [n for n in imported.replace(",", " ").split() if n.startswith("_")]


def test_in02_the_primary_port_is_the_merged_server_port(g):
    _config(g, server=KEYED)
    assert ws_mod.primary_server_port(g) == 8080
    write_yaml(g / "overrides.yaml", {"server": {"port": 9191}})
    assert ws_mod.primary_server_port(g) == 9191
