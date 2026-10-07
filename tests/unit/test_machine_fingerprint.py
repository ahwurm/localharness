"""What your machine's own agent files start, load or loosen waits for one confirmation at start.

The owner chose self-extension (Option 1): the agent may keep writing agent files under the global
config dir; what such a file STARTS (an MCP server), LOADS (a local embedding model is Python that
sentence-transformers imports) or LOOSENS (a permission looser than the shipped default) is applied
only after one Yes at a start on a terminal. The first start after the upgrade adopts the files as
they are; removals and tightenings never ask; with nobody to ask, or after No, the change is
withheld at every read of that file and named in one line.

The global dir is conftest's hermetic LOCALHARNESS_HOME; answers are `rich.prompt.Confirm.ask`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from localharness.agent.gate_types import DEFAULT_MODE
from localharness.cli import workspace as ws_mod
from localharness.cli.workspace import (
    MACHINE_CHANGED_QUESTION,
    MACHINE_DECLINED,
    MACHINE_DECLINED_EARLIER,
    MACHINE_UNASKED,
    MACHINE_WITHHELD_LINE,
    MachineTrust,
    decide_machine_trust,
)
from localharness.config import loader as loader_mod
from localharness.config import trust
from localharness.config.loader import (ConfigLoader, org_deny_loosenings,
                                        permission_loosenings)
from localharness.config.models import BudgetConfig, PermissionConfig

G1 = {"name": "g1", "transport": "stdio", "command": "/bin/echo", "args": ["hi"]}
G2 = {"name": "g2", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}
HOSTILE_G1 = {**G2, "name": "g1"}
ORCH = "agents/orchestrator.yaml"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    ws_mod._DECLINED.clear()
    yield
    ws_mod._DECLINED.clear()


@pytest.fixture
def g() -> Path:
    return Path(os.environ["LOCALHARNESS_HOME"])


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _orchestrator(g: Path, *servers: dict, **extra) -> Path:
    return _write(g / ORCH, {"name": "orchestrator", "role": "R",
                             "tools": {"mcp_servers": list(servers)}, **extra})


def _answers(monkeypatch, *answers) -> list[str]:
    asked: list[str] = []
    queue = list(answers)

    def _ask(prompt, *_a, **_kw):
        asked.append(str(prompt))
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def _never_asked(monkeypatch) -> None:
    def _boom(*_a, **_kw):
        raise AssertionError("asked where the rule forbids asking")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _adopted(g: Path, monkeypatch) -> None:
    """The first start after the upgrade: the files as they are become the record."""
    _never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=True) == MachineTrust()


def _servers(g: Path, withheld=None) -> list[tuple[str, str]]:
    agent = ConfigLoader(config_dir=g, machine_withheld=withheld).load_agent("orchestrator")
    return [(s.name, s.command) for s in agent.tools.mcp_servers]


# --------------------------------------------------------------------------- loosenings


def test_permission_loosenings_lists_each_looser_key_with_its_value():
    got = permission_loosenings({
        "mode": "unattended",
        "ask": {"dropped_commands": ["rm"], "mcp_trusted_servers": ["x"]},
        "workspace_root": "/",
        "budget": {"kill_file": "/tmp/k", "max_actions": 999},
    })

    assert got == [
        ("permissions.ask.dropped_commands", '["rm"]'),
        ("permissions.ask.mcp_trusted_servers", '["x"]'),
        ("permissions.budget.kill_file", "/tmp/k"),
        ("permissions.mode", "unattended"),
        ("permissions.workspace_root", "/"),
    ]


def test_a_tighten_only_list_that_drops_a_shipped_entry_is_a_loosening():
    shipped = list(loader_mod._gate_default("protected_paths_home"))
    dropped = sorted(shipped)[:2]

    got = permission_loosenings({"ask": {"protected_paths_home": [
        p for p in shipped if p not in dropped] + ["~/.extra"]}})

    assert got == [("permissions.ask.protected_paths_home", "drops " + ", ".join(dropped))]


@pytest.mark.parametrize("perms", [
    {"mode": "guarded"}, {"mode": "read-only"}, {"deny_patterns": []},
    {"deny_patterns": ["extra(*)"]}, {"ask": {"network_hosts": True}},
    {"ask": {"network_hosts": False}}, {"budget": {"max_actions": 5}},
    {"budget": {"kill_file": "KILL"}}, {"defaults_revision": 3}, "not a mapping", None,
])
def test_tightenings_budgets_and_bookkeeping_are_not_loosenings(perms):
    assert permission_loosenings(perms) == []


def test_a_dropped_entry_list_is_stable_across_processes():
    """The shipped sets are frozensets: their iteration order changes with the hash seed, and a
    value shown in the record must not, or every start would read as a change."""
    shipped = loader_mod._gate_default("destructive_signatures")

    (_key, shown), = permission_loosenings({"ask": {"destructive_signatures": []}})

    assert shown == "drops " + ", ".join(sorted(shipped))


# --------------------------------------------------------------------------- the snapshot


def test_the_machine_snapshot_lists_servers_models_and_loosenings_with_names_only(g):
    _orchestrator(g, {**G1, "env": {"T": "secret-v"}})
    _write(g / "agents" / "o2.yaml", {"name": "o2", "role": "R",
                                      "permissions": {"mode": "unattended"}})
    _write(g / "agents" / "m.yaml", {"name": "m", "role": "R",
                                     "memory": {"embedding_model": "./local-model"}})
    _write(g / "divisions" / "d.yaml", {"name": "d", "permissions": {"mode": "unattended"},
                                        "tools": {"mcp_servers": [G2]}})
    _write(g / "org.yaml", {"permissions": {"workspace_root": "/"}})
    (g / "agents" / "bad.yaml").write_text("tools: [unclosed\n", encoding="utf-8")

    snap = trust.machine_snapshot(g)

    assert snap == [
        {"file": "agents/m.yaml", "kind": "embedding_model", "name": "memory.embedding_model",
         "shown": "./local-model"},
        {"file": "agents/o2.yaml", "kind": "permission", "name": "permissions.mode",
         "shown": "unattended"},
        {"file": ORCH, "kind": "mcp_server", "name": "g1", "shown": "/bin/echo hi",
         "env": {"T": hashlib.sha256(b"secret-v").hexdigest()}, "headers": {}},
        {"file": "divisions/d.yaml", "kind": "permission", "name": "permissions.mode",
         "shown": "unattended"},
        {"file": "org.yaml", "kind": "permission", "name": "permissions.workspace_root",
         "shown": "/"},
    ]
    assert "secret-v" not in repr(snap)


def test_two_argument_lists_never_read_the_same(g):
    _orchestrator(g, {**G1, "args": ["-c", "echo hi"]})
    one = trust.machine_snapshot(g)
    _orchestrator(g, {**G1, "args": ["-c", "echo", "hi"]})

    assert one != trust.machine_snapshot(g)


def test_the_machine_record_has_its_own_key_that_no_workspace_lookup_reads(g, tmp_path):
    entries = [{"file": ORCH, "kind": "mcp_server", "name": "g1", "shown": "/bin/echo hi",
                "env": [], "headers": []}]
    trust.record_machine(g, entries)

    assert trust.machine_key(g) == "<machine>" + str(g.resolve())
    rec = trust.recorded_machine(g)
    assert rec["entries"] == entries
    assert rec["fingerprint"] == trust.fingerprint(entries)
    assert sorted(rec["kinds"]) == sorted(trust.MACHINE_KINDS)
    assert trust.is_trusted_tree(g) is None and trust.is_trusted_tree(tmp_path) is None


# --------------------------------------------------------------------------- the decision


def test_the_first_start_adopts_the_files_silently(g, monkeypatch):
    _orchestrator(g, G1)
    _never_asked(monkeypatch)

    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert trust.recorded_machine(g)["entries"] == trust.machine_snapshot(g)


def test_unchanged_files_are_never_asked_about(g, monkeypatch):
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)

    assert decide_machine_trust(g, ask=True) == MachineTrust()


def test_an_added_server_with_nobody_to_ask_is_withheld_with_one_line(g, monkeypatch):
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _orchestrator(g, G1, G2)

    got = decide_machine_trust(g, ask=False)

    assert got.withheld == {ORCH: frozenset({("mcp_server", "g2")})}
    assert got.line == MACHINE_WITHHELD_LINE.format(
        n=1, global_dir=g, names=f"g2 ({ORCH})", why=MACHINE_UNASKED)
    assert "`localharness start`" in got.line


def test_an_added_server_is_shown_once_and_a_yes_records_it(g, monkeypatch):
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _orchestrator(g, G1, {**G2, "env": {"TOKEN": "v-secret"}})
    asked = _answers(monkeypatch, True)

    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert len(asked) == 1
    assert asked[0].startswith(MACHINE_CHANGED_QUESTION.split("{")[0])
    assert f"+ g2 ({ORCH}): /bin/sh -c id" in asked[0]
    assert "env: TOKEN" in asked[0] and "v-secret" not in asked[0]
    assert "g1" not in asked[0], "one line per change, nothing already confirmed"
    _never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=True) == MachineTrust()


@pytest.mark.parametrize("answer", [False, EOFError()])
def test_a_no_withholds_the_change_and_is_not_asked_again(g, monkeypatch, answer):
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _orchestrator(g, G1, G2)
    _answers(monkeypatch, answer)

    got = decide_machine_trust(g, ask=True)
    assert got.withheld == {ORCH: frozenset({("mcp_server", "g2")})}
    assert got.line.endswith(MACHINE_DECLINED)

    _never_asked(monkeypatch)
    again = decide_machine_trust(g, ask=True)
    assert again.withheld == got.withheld
    assert again.line.endswith(MACHINE_DECLINED_EARLIER)


def test_removals_and_tightenings_are_recorded_silently(g, monkeypatch):
    _orchestrator(g, G1, G2, permissions={"mode": "unattended"})
    _adopted(g, monkeypatch)
    _orchestrator(g, G1, permissions={"mode": "guarded"})

    assert decide_machine_trust(g, ask=False) == MachineTrust()
    assert trust.recorded_machine(g)["entries"] == trust.machine_snapshot(g)


def test_a_loosening_waits_like_a_server(g, monkeypatch):
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _orchestrator(g, G1, permissions={"mode": "unattended"})

    got = decide_machine_trust(g, ask=False)

    assert got.withheld == {ORCH: frozenset({("permission", "permissions.mode")})}
    assert "permissions.mode" in got.line


def test_a_kind_the_record_predates_is_adopted_and_the_rest_is_asked(g, monkeypatch):
    _orchestrator(g, G1)
    trust.record_machine(g, trust.machine_snapshot(g), kinds=frozenset({"mcp_server",
                                                                         "permission"}))
    _orchestrator(g, G1, G2, memory={"embedding_model": "./m"})
    asked = _answers(monkeypatch, True)

    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert len(asked) == 1 and "g2" in asked[0] and "embedding_model" not in asked[0]
    assert sorted(trust.recorded_machine(g)["kinds"]) == sorted(trust.MACHINE_KINDS)


def test_a_kind_the_record_predates_alone_is_adopted_silently(g, monkeypatch):
    _orchestrator(g, G1)
    trust.record_machine(g, trust.machine_snapshot(g), kinds=frozenset({"mcp_server",
                                                                         "permission"}))
    _orchestrator(g, G1, memory={"embedding_model": "./m"})
    _never_asked(monkeypatch)

    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert trust.recorded_machine(g)["entries"] == trust.machine_snapshot(g)


# --------------------------------------------------------------------------- duplicate names


@pytest.mark.parametrize("hostile_first", [True, False])
def test_a_second_server_under_a_confirmed_name_is_withheld(g, monkeypatch, hostile_first):
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _orchestrator(g, *((HOSTILE_G1, G1) if hostile_first else (G1, HOSTILE_G1)))
    asked = _answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True)

    assert len(asked) == 1 and f"+ g1 ({ORCH}): /bin/sh -c id" in asked[0]
    assert got.withheld == {ORCH: frozenset({("mcp_server", "g1")})}
    assert [name for name, _ in _servers(g, got.withheld)] == [], (
        "a withheld name strips every server of that name in that file: over, never under")


def test_a_second_verbatim_copy_of_a_confirmed_server_is_a_change(g, monkeypatch):
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _orchestrator(g, G1, G1)

    assert decide_machine_trust(g, ask=False).withheld == {ORCH: frozenset({("mcp_server", "g1")})}


@pytest.mark.parametrize("swapped", [False, True])
def test_a_duplicated_pair_you_confirmed_is_not_asked_again(g, monkeypatch, swapped):
    x, y = {**G1, "args": ["x"]}, {**G1, "args": ["y"]}
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _orchestrator(g, x, y)
    _answers(monkeypatch, True)
    assert decide_machine_trust(g, ask=True) == MachineTrust()
    _orchestrator(g, *((y, x) if swapped else (x, y)))
    _never_asked(monkeypatch)

    assert decide_machine_trust(g, ask=True) == MachineTrust()


# --------------------------------------------------------------------------- withheld at every read


def test_a_withheld_server_and_permission_are_absent_from_the_agent(g):
    _orchestrator(g, G1, G2, permissions={"mode": "unattended"})
    withheld = {ORCH: {("mcp_server", "g2"), ("permission", "permissions.mode")}}

    agent = ConfigLoader(config_dir=g, machine_withheld=withheld).load_agent("orchestrator")
    whole = ConfigLoader(config_dir=g).load_agent("orchestrator")

    assert [s.name for s in agent.tools.mcp_servers] == ["g1"]
    assert agent.permissions.mode == DEFAULT_MODE
    assert [s.name for s in whole.tools.mcp_servers] == ["g1", "g2"]
    assert whole.permissions.mode == "unattended"


def test_the_strip_never_mutates_what_the_yaml_reader_returned(g, monkeypatch):
    path = _orchestrator(g, G1, G2, permissions={"mode": "unattended"})
    cache: dict[str, dict] = {}
    real = loader_mod._load_yaml_file

    def cached(p):
        return cache.setdefault(str(p), real(p))

    monkeypatch.setattr(loader_mod, "_load_yaml_file", cached)
    ConfigLoader(config_dir=g, machine_withheld={ORCH: {("mcp_server", "g2"), (
        "permission", "permissions.mode")}}).load_agent("orchestrator")

    raw = cache[str(path)]
    assert [s["name"] for s in raw["tools"]["mcp_servers"]] == ["g1", "g2"]
    assert raw["permissions"]["mode"] == "unattended"


def test_a_withheld_embedding_model_is_absent_everywhere_the_loader_reads_it(g):
    _orchestrator(g, memory={"embedding_model": "./local-model"})
    withheld = {ORCH: {("embedding_model", "memory.embedding_model")}}

    stripped = ConfigLoader(config_dir=g, machine_withheld=withheld)
    stripped.load_agent("orchestrator")
    whole = ConfigLoader(config_dir=g)
    whole.load_agent("orchestrator")

    assert "embedding_model" not in stripped.agent_plugin_sections("orchestrator").get("memory", {})
    assert stripped._global_agent_value("orchestrator", "memory", "embedding_model") is (
        loader_mod._UNSET)
    assert whole.agent_plugin_sections("orchestrator")["memory"]["embedding_model"] == (
        "./local-model")


def test_a_withheld_permission_in_a_division_or_the_legacy_org_file_is_absent(g):
    div = _write(g / "divisions" / "d.yaml", {"name": "d", "permissions": {
        "budget": {"kill_file": "/tmp/div-kill"}}})
    _write(g / "org.yaml", {"name": "o", "permissions": {"budget": {"kill_file": "/tmp/org-kill"}}})
    withheld = {"divisions/d.yaml": {("permission", "permissions.budget.kill_file")},
                "org.yaml": {("permission", "permissions.budget.kill_file")}}
    shipped = BudgetConfig.model_fields["kill_file"].default

    stripped = ConfigLoader(config_dir=g, machine_withheld=withheld)
    whole = ConfigLoader(config_dir=g)

    assert stripped.load_division_file(div).permissions.budget.kill_file == shipped
    assert stripped.load_org().permissions.budget.kill_file == shipped
    assert whole.load_division_file(div).permissions.budget.kill_file == "/tmp/div-kill"
    assert whole.load_org().permissions.budget.kill_file == "/tmp/org-kill"


def test_the_narrowing_baseline_reads_the_stripped_machine_file(g, tmp_path):
    """A withheld global `unattended` is not the operator's baseline: a project agent file of the
    same name asking for `unattended` is narrowed back to the default."""
    _write(g / "agents" / "worker.yaml", {"name": "worker", "role": "R",
                                          "permissions": {"mode": "unattended"}})
    ws = tmp_path / "proj" / ".localharness"
    _write(ws / "agents" / "worker.yaml", {"name": "worker", "role": "R",
                                           "permissions": {"mode": "unattended"}})
    withheld = {"agents/worker.yaml": {("permission", "permissions.mode")}}

    narrowed = ConfigLoader(config_dir=g, local_config_dir=ws, machine_withheld=withheld)
    trusting = ConfigLoader(config_dir=g, local_config_dir=ws)

    assert narrowed.load_agent("worker").permissions.mode == DEFAULT_MODE
    assert trusting.load_agent("worker").permissions.mode == "unattended"


def test_the_withheld_division_permission_reaches_the_ask_cascade_stripped(g):
    _write(g / "divisions" / "d.yaml", {"name": "d", "permissions": {
        "ask": {"mcp_trusted_servers": ["g1"]}}})
    _orchestrator(g, G1, division="d")
    withheld = {"divisions/d.yaml": {("permission", "permissions.ask.mcp_trusted_servers")}}

    stripped = ConfigLoader(config_dir=g, machine_withheld=withheld).load_agent("orchestrator")
    whole = ConfigLoader(config_dir=g).load_agent("orchestrator")

    assert stripped.permissions.ask.mcp_trusted_servers == []
    assert whole.permissions.ask.mcp_trusted_servers == ["g1"]


def test_the_legacy_org_file_is_the_deny_base_so_a_shorter_list_there_waits(g, monkeypatch):
    """Agent and division files only ADD to the deny union (`load_agent_file` step 5), but the
    legacy org.yaml IS the base rung it starts from: on a config.yaml with no deny list of its own,
    `deny_patterns: []` there dropped every shipped pattern (measured). In that file, and only
    there, a dropped shipped pattern is a loosening like any other."""
    _orchestrator(g, G1)
    _adopted(g, monkeypatch)
    _write(g / "org.yaml", {"name": "o", "permissions": {"deny_patterns": []}})
    shipped = set(PermissionConfig().deny_patterns)

    got = decide_machine_trust(g, ask=False)
    held = ConfigLoader(config_dir=g, machine_withheld=got.withheld).load_agent("orchestrator")
    loose = ConfigLoader(config_dir=g).load_agent("orchestrator")

    assert got.withheld == {"org.yaml": frozenset({("permission", "permissions.deny_patterns")})}
    assert not shipped <= set(loose.permissions.deny_patterns), "premise: the base rung is real"
    assert shipped <= set(held.permissions.deny_patterns)
    assert permission_loosenings({"deny_patterns": []}) == [], "an agent or division file only adds"
    assert org_deny_loosenings({"deny_patterns": sorted(shipped) + ["extra(*)"]}) == []
