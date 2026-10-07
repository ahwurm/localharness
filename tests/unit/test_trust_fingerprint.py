"""Trust is for what you saw: the MCP servers a project's agent files start are recorded at the Yes
that showed them, and a later change is shown and asked about once, at start, on a terminal.

The record lives in the machine's trust store (`config/trust.py`), pointed at a tmp dir by the
autouse home fixture in conftest; the answers are `rich.prompt.Confirm.ask`, patched. Nothing here
spawns anything: the decision is a value (`cli.workspace.ProjectTrust`) the loader acts on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from localharness.cli import workspace as ws_mod
from localharness.cli.workspace import (
    EXECUTABLES_CHANGED_QUESTION,
    NEXT_START_REVIEW_NOTICE,
    NOT_STARTED_DECLINED,
    NOT_STARTED_DECLINED_EARLIER,
    NOT_STARTED_UNASKED,
    OFFER_PROMPT,
    TRUST_ONLY_PROMPT,
    TRUST_QUESTION,
    ProjectTrust,
    decide_project_trust,
    executables_diff,
    untrusted_remedy,
)
from localharness.config import trust
from localharness.config.paths import WORKSPACE_DIR_NAME

EVIL = {"name": "evil", "transport": "stdio", "command": "/bin/echo", "args": ["pwned"],
        "env": {"LD_PRELOAD": "/x.so", "A": "1"}}
WEB = {"name": "web", "transport": "streamable_http", "url": "https://h.example/mcp",
       "headers": {"Authorization": "Bearer t"}}
EVIL2 = {"name": "evil2", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}
BENIGN_A = {"name": "a", "transport": "stdio", "command": "/bin/echo", "args": ["benign"]}
HOSTILE_A = {"name": "a", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}


@pytest.fixture(autouse=True)
def _fresh_process_memory(monkeypatch):
    """The declined set is process-wide by design (a /plugins restart must not re-ask); each test
    is its own process here. LOCALHARNESS_DIR outranks the hermetic LOCALHARNESS_HOME — cleared so
    the store is the tmp one."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    ws_mod._DECLINED.clear()
    yield
    ws_mod._DECLINED.clear()


def _agent(ws: Path, file: str, *servers: dict) -> Path:
    path = ws / "agents" / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": Path(file).stem, "role": "R",
                                    "tools": {"mcp_servers": list(servers)}}), encoding="utf-8")
    return path


def _ws(tmp_path: Path, name: str = "proj") -> Path:
    ws = tmp_path / name / WORKSPACE_DIR_NAME
    (ws / "agents").mkdir(parents=True)
    return ws


def _root(ws: Path) -> Path:
    return ws.resolve().parent


def _answers(monkeypatch, *answers) -> list[str]:
    """Confirm.ask answering in order (an exception instance is raised); records each question."""
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


def _entry(root: Path) -> dict:
    data = yaml.safe_load(trust.trust_store_path().read_text(encoding="utf-8")) or {}
    return data.get(str(root.resolve()), {})


def _approve(ws: Path) -> None:
    """A Yes given with the current list shown: trusted, and exactly this set approved."""
    trust.record_trust(_root(ws), True)
    trust.record_executables(_root(ws), trust.executables_snapshot(ws))


# --------------------------------------------------------------------------- the snapshot


def test_the_snapshot_hashes_env_and_header_values(tmp_path):
    import hashlib
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _agent(ws, "b.yaml", WEB)

    snap = trust.executables_snapshot(ws)

    assert snap == [
        {"file": "a.yaml", "name": "evil", "transport": "stdio", "command": "/bin/echo",
         "args": ["pwned"],
         "env": {"A": hashlib.sha256(b"1").hexdigest(),
                 "LD_PRELOAD": hashlib.sha256(b"/x.so").hexdigest()},
         "url": "", "headers": {}},
        {"file": "b.yaml", "name": "web", "transport": "streamable_http", "command": "",
         "args": [], "env": {}, "url": "https://h.example/mcp",
         "headers": {"Authorization": hashlib.sha256(b"Bearer t").hexdigest()}},
    ]
    blob = json.dumps(snap)
    assert "/x.so" not in blob and "Bearer t" not in blob, "a value is often a secret"


def test_a_file_that_does_not_parse_or_is_not_a_mapping_is_skipped(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    (ws / "agents" / "c.yaml").write_text("tools: [unclosed\n", encoding="utf-8")
    (ws / "agents" / "d.yaml").write_text("- just\n- a list\n", encoding="utf-8")

    assert [e["file"] for e in trust.executables_snapshot(ws)] == ["a.yaml"]


@pytest.mark.parametrize("change", [
    {"command": "/bin/sh"},
    {"args": ["other"]},
    {"env": {"LD_PRELOAD": "/x.so", "B": "1"}},
    {"transport": "streamable_http", "url": "https://elsewhere.example/mcp"},
])
def test_the_fingerprint_moves_with_what_would_run(tmp_path, change):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    before = trust.fingerprint(trust.executables_snapshot(ws))
    _agent(ws, "a.yaml", {**EVIL, **change})

    assert before.startswith("sha256:")
    assert trust.fingerprint(trust.executables_snapshot(ws)) != before


def test_the_fingerprint_moves_with_a_url_or_a_header_name(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "b.yaml", WEB)
    before = trust.fingerprint(trust.executables_snapshot(ws))
    _agent(ws, "b.yaml", {**WEB, "url": "https://h.example/other"})
    by_url = trust.fingerprint(trust.executables_snapshot(ws))
    _agent(ws, "b.yaml", {**WEB, "headers": {"X-Api-Key": "Bearer t"}})

    assert len({before, by_url, trust.fingerprint(trust.executables_snapshot(ws))}) == 3


def test_the_fingerprint_moves_with_env_and_header_values(tmp_path):
    """§7.5: a value-only change (swapping a TOKEN) now trips the fingerprint."""
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _agent(ws, "b.yaml", WEB)
    before = trust.fingerprint(trust.executables_snapshot(ws))
    _agent(ws, "a.yaml", {**EVIL, "env": {"LD_PRELOAD": "/other.so", "A": "2"}})
    _agent(ws, "b.yaml", {**WEB, "headers": {"Authorization": "Bearer rotated"}})

    assert trust.fingerprint(trust.executables_snapshot(ws)) != before


# --------------------------------------------------------------------------- the record


def test_recording_trust_keeps_the_servers_already_approved(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _approve(ws)
    approved = _entry(_root(ws))["executables"]

    trust.record_trust(_root(ws), True)

    assert _entry(_root(ws))["executables"] == approved


def test_a_yes_without_the_list_records_nothing_approved_yet(tmp_path):
    root = _root(_ws(tmp_path))
    trust.record_trust(root, True, unseen_executables=True)

    executables = _entry(root)["executables"]
    assert {k: executables[k] for k in ("fingerprint", "servers")} == trust.NOTHING_APPROVED
    assert executables["recorded"]


def test_a_yes_without_the_list_never_replaces_an_approved_set(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _approve(ws)
    approved = _entry(_root(ws))["executables"]

    trust.record_trust(_root(ws), True, unseen_executables=True)

    assert _entry(_root(ws))["executables"] == approved


def test_recording_servers_keeps_the_trust_answer(tmp_path):
    ws = _ws(tmp_path)
    trust.record_trust(_root(ws), False)
    trust.record_executables(_root(ws), [])

    assert trust.is_trusted(_root(ws)) is False


def test_a_record_written_before_this_release_has_no_servers_recorded(tmp_path):
    root = _root(_ws(tmp_path))
    trust.trust_store_path().write_text(yaml.safe_dump({str(root): {"trusted": True}}),
                                        encoding="utf-8")

    assert trust.recorded_executables(root) is None
    assert trust.is_trusted(root) is True


# --------------------------------------------------------------------------- the decision


def test_no_workspace_and_the_flag_both_mean_nothing_to_withhold(tmp_path, monkeypatch):
    _never_asked(monkeypatch)
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)

    assert decide_project_trust(None, ask=True, trust_flag=False) == ProjectTrust(True)
    assert decide_project_trust(ws, ask=True, trust_flag=True) == ProjectTrust(True)
    assert not trust.trust_store_path().exists(), "the flag is for one run: nothing recorded"


def test_an_unrecorded_project_starts_nothing_and_names_every_remedy(tmp_path, monkeypatch):
    _never_asked(monkeypatch)
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)

    got = decide_project_trust(ws, ask=True, trust_flag=False)

    assert got == ProjectTrust(False, untrusted_remedy("terminal", _root(ws), recorded_no=False))


def test_a_recorded_no_starts_nothing_and_says_you_said_no(tmp_path, monkeypatch):
    _never_asked(monkeypatch)
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    trust.record_trust(_root(ws), False)

    got = decide_project_trust(ws, ask=True, trust_flag=False, channel_mode="mobile")

    assert got == ProjectTrust(False, untrusted_remedy("mobile", _root(ws), recorded_no=True))


def test_a_pre_upgrade_record_for_this_project_adopts_its_servers_silently(tmp_path, monkeypatch):
    _never_asked(monkeypatch)
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    trust.trust_store_path().write_text(yaml.safe_dump({str(_root(ws)): {"trusted": True}}),
                                        encoding="utf-8")

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(True)
    assert trust.recorded_executables(_root(ws))["fingerprint"] == trust.fingerprint(
        trust.executables_snapshot(ws))


def test_an_unchanged_set_is_never_asked_about(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _approve(ws)
    _never_asked(monkeypatch)

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(True)


def test_a_yes_given_without_the_list_starts_nothing_until_asked(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    trust.record_trust(_root(ws), True, unseen_executables=True)
    _never_asked(monkeypatch)

    got = decide_project_trust(ws, ask=False, trust_flag=False)

    assert got == ProjectTrust(False, NOT_STARTED_UNASKED.format(
        one_run="pass --trust-project for one run"))


def test_a_change_that_only_removes_servers_is_recorded_silently(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _agent(ws, "b.yaml", WEB)
    _approve(ws)
    (ws / "agents" / "b.yaml").unlink()
    _never_asked(monkeypatch)

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(True)
    assert trust.recorded_executables(_root(ws))["servers"] == trust.executables_snapshot(ws)


def test_a_change_with_nobody_to_ask_starts_nothing_and_keeps_the_record(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _approve(ws)
    before = _entry(_root(ws))
    _agent(ws, "a.yaml", EVIL, EVIL2)
    _never_asked(monkeypatch)

    got = decide_project_trust(ws, ask=False, trust_flag=False, channel_mode="acp")

    assert got == ProjectTrust(False, NOT_STARTED_UNASKED.format(
        one_run="set LOCALHARNESS_TRUST_PROJECT=1 for one run"))
    assert _entry(_root(ws)) == before


def test_a_change_is_shown_and_a_yes_records_it(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _approve(ws)
    _agent(ws, "a.yaml", EVIL, EVIL2)
    asked = _answers(monkeypatch, True)

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(True)
    assert len(asked) == 1
    assert asked[0].startswith(EXECUTABLES_CHANGED_QUESTION.split("{")[0])
    assert "+ evil2 (a.yaml): /bin/sh -c id" in asked[0]
    assert trust.recorded_executables(_root(ws))["servers"] == trust.executables_snapshot(ws)


@pytest.mark.parametrize("answer", [False, EOFError()])
def test_a_no_or_a_closed_terminal_starts_nothing_and_keeps_the_record(tmp_path, monkeypatch,
                                                                       answer):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _approve(ws)
    before = _entry(_root(ws))
    _agent(ws, "a.yaml", EVIL, EVIL2)
    _answers(monkeypatch, answer)

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(
        False, NOT_STARTED_DECLINED)
    assert _entry(_root(ws)) == before


def test_a_no_is_not_asked_again_in_the_same_process(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _approve(ws)
    _agent(ws, "a.yaml", EVIL, EVIL2)
    _answers(monkeypatch, False)
    decide_project_trust(ws, ask=True, trust_flag=False)
    _never_asked(monkeypatch)

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(
        False, NOT_STARTED_DECLINED_EARLIER)


def test_a_project_trusted_only_through_its_parent_is_asked_about_its_servers(tmp_path,
                                                                              monkeypatch):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    trust.record_trust(tmp_path, True)  # the parent folder's record, nothing for this project
    _never_asked(monkeypatch)

    assert decide_project_trust(ws, ask=False, trust_flag=False) == ProjectTrust(
        False, NOT_STARTED_UNASKED.format(one_run="pass --trust-project for one run"))

    asked = _answers(monkeypatch, True)
    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(True)
    assert "+ evil (a.yaml): /bin/echo pwned" in asked[0]
    assert trust.recorded_executables(_root(ws))["servers"] == trust.executables_snapshot(ws)
    assert trust.is_trusted(_root(ws)) is None, "the parent's record still decides trust"


def test_a_project_under_a_trusted_parent_with_no_servers_asks_nothing(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    trust.record_trust(tmp_path, True)
    _never_asked(monkeypatch)

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(True)


# --------------------------------------------------------------------------- the remedy line


def test_the_remedy_on_start_names_the_flag_the_question_and_the_store(tmp_path):
    root = tmp_path / "proj"
    text = untrusted_remedy("terminal", root, recorded_no=False)

    assert "--trust-project" in text
    assert "trust question" in text and "`localharness start`" in text
    assert f"`{root}: {{trusted: true}}`" in text
    assert str(trust.trust_store_path()) in text


@pytest.mark.parametrize("mode", ["mobile", "acp"])
def test_the_remedy_on_web_and_acp_names_the_environment_variable(tmp_path, mode):
    text = untrusted_remedy(mode, tmp_path, recorded_no=False)

    assert "LOCALHARNESS_TRUST_PROJECT=1" in text
    assert "--trust-project" not in text


def test_the_remedy_after_a_no_says_so_and_names_the_entry_to_change(tmp_path):
    text = untrusted_remedy("terminal", tmp_path, recorded_no=True)

    assert "you said No" in text
    assert "`trusted: true`" in text
    assert str(trust.trust_store_path()) in text


# --------------------------------------------------------------------------- the diff


def test_the_diff_shows_added_removed_and_changed_servers_by_name(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", EVIL)
    _agent(ws, "b.yaml", WEB)
    old = trust.executables_snapshot(ws)
    (ws / "agents" / "b.yaml").unlink()
    _agent(ws, "a.yaml", {**EVIL, "args": ["other"]}, EVIL2)

    diff = executables_diff(old, trust.executables_snapshot(ws))

    assert "+ evil2 (a.yaml): /bin/sh -c id" in diff
    assert "- web (b.yaml)" in diff
    assert "~ evil (a.yaml): /bin/echo other" in diff
    assert "env: A, LD_PRELOAD" in diff
    assert "/x.so" not in diff
    assert all(line.startswith("  ") for line in diff.splitlines())


def test_a_change_pairs_once_and_a_second_entry_under_that_name_is_added(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", BENIGN_A)
    _agent(ws, "b.yaml", WEB)
    old = trust.executables_snapshot(ws)
    (ws / "agents" / "b.yaml").unlink()
    _agent(ws, "a.yaml", HOSTILE_A, {**BENIGN_A, "args": ["changed"]})
    new = trust.executables_snapshot(ws)

    marks = [line.split()[0] for line in executables_diff(old, new).splitlines()]
    quiet = [line.split()[0] for line in executables_diff(old, new, removed=False).splitlines()]

    assert marks == ["~", "+", "-"], marks
    assert quiet == ["~", "+"], "removals are listed only when asked for"


def test_the_diff_cannot_be_redrawn_by_control_characters_in_a_file(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", {"name": "ok\x1b[2K\rbenign", "transport": "stdio",
                          "command": "/bin/sh", "args": ["-c", "id\n  + fine (x.yaml): true"]})

    diff = executables_diff([], trust.executables_snapshot(ws))

    assert "\x1b" not in diff and "\r" not in diff
    assert len(diff.splitlines()) == 1, "an argument's newline must not fake a second line"
    assert "\\x1b" in diff and "\\x0a" in diff, "shown, not hidden"


# --------------------------------------------------------------------------- duplicate names


@pytest.mark.parametrize("order", ["hostile-first", "hostile-last"])
def test_a_second_server_under_an_approved_name_is_asked_about(tmp_path, monkeypatch, order):
    """ToolsConfig has no unique-name rule and the MCP manager connects every entry: a server added
    under a name you approved must never collapse onto the approved one."""
    ws = _ws(tmp_path)
    _agent(ws, "orchestrator.yaml", BENIGN_A)
    _approve(ws)
    before = _entry(_root(ws))
    pair = (HOSTILE_A, BENIGN_A) if order == "hostile-first" else (BENIGN_A, HOSTILE_A)
    _agent(ws, "orchestrator.yaml", *pair)
    asked = _answers(monkeypatch, False)

    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(
        False, NOT_STARTED_DECLINED)
    assert len(asked) == 1
    assert "+ a (orchestrator.yaml): /bin/sh -c id" in asked[0]
    assert _entry(_root(ws)) == before

    ws_mod._DECLINED.clear()
    _never_asked(monkeypatch)
    assert decide_project_trust(ws, ask=False, trust_flag=False) == ProjectTrust(
        False, NOT_STARTED_UNASKED.format(one_run="pass --trust-project for one run"))


@pytest.mark.parametrize("swapped", [False, True])
def test_a_duplicated_pair_you_approved_is_not_asked_about_again(tmp_path, monkeypatch, swapped):
    ws = _ws(tmp_path)
    x = {**BENIGN_A, "args": ["x"]}
    y = {**BENIGN_A, "args": ["y"]}
    _agent(ws, "orchestrator.yaml", x, y)
    _approve(ws)
    approved = trust.fingerprint(trust.executables_snapshot(ws))
    _agent(ws, "orchestrator.yaml", *((y, x) if swapped else (x, y)))
    _never_asked(monkeypatch)

    assert trust.fingerprint(trust.executables_snapshot(ws)) == approved
    assert decide_project_trust(ws, ask=True, trust_flag=False) == ProjectTrust(True)


# --------------------------------------------------------------------------- every Yes records the list


@pytest.fixture
def standing_in(tmp_path, monkeypatch, fake_home):
    """A git project inside a fake home whose workspace already exists; cwd in it, a terminal."""
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    fake_home(home)
    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    (proj / WORKSPACE_DIR_NAME / "agents").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    return proj


def test_the_start_question_lists_the_servers_and_a_yes_records_them(standing_in, monkeypatch):
    from localharness.cli.workspace import settle_startup_trust

    ws = standing_in / WORKSPACE_DIR_NAME
    _agent(ws, "orchestrator.yaml", {"name": "evil", "transport": "stdio",
                                     "command": "/bin/echo", "args": ["pwned"]})
    asked = _answers(monkeypatch, True)

    settle_startup_trust(None)

    assert len(asked) == 1
    assert asked[0].startswith(TRUST_ONLY_PROMPT)
    assert asked[0].rstrip().endswith("+ evil (orchestrator.yaml): /bin/echo pwned")
    assert trust.is_trusted(standing_in) is True
    assert trust.recorded_executables(standing_in)["servers"] == trust.executables_snapshot(ws)


def test_a_project_without_servers_is_asked_the_unchanged_question(standing_in, monkeypatch):
    from localharness.cli.workspace import settle_startup_trust

    asked = _answers(monkeypatch, True)

    settle_startup_trust(None)

    assert asked == [TRUST_ONLY_PROMPT]
    assert trust.recorded_executables(standing_in)["servers"] == []


def test_a_new_project_is_offered_the_unchanged_question(tmp_path, monkeypatch, fake_home):
    from localharness.cli.workspace import settle_startup_trust

    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    fake_home(home)
    (home / "fresh").mkdir()
    monkeypatch.chdir(home / "fresh")
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    asked = _answers(monkeypatch, True)

    settle_startup_trust(None)

    assert asked == [OFFER_PROMPT]
    assert trust.recorded_executables(home / "fresh")["servers"] == []


def test_the_outside_workspace_question_lists_the_servers_and_a_yes_records_them(
        tmp_path, monkeypatch, fake_home):
    from localharness.cli.workspace import resolve_workspace_layer

    fake_home(tmp_path / "home")
    (tmp_path / "home" / WORKSPACE_DIR_NAME).mkdir()
    root = tmp_path / "outside"
    ws = root / WORKSPACE_DIR_NAME
    _agent(ws, "orchestrator.yaml", EVIL)
    deep = root / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    asked: list[str] = []

    assert resolve_workspace_layer(asker=lambda q: asked.append(q) or True) == ws

    assert len(asked) == 1
    assert asked[0].startswith(TRUST_QUESTION.format(parent=root.resolve()))
    assert "+ evil (orchestrator.yaml): /bin/echo pwned" in asked[0]
    assert "env: A, LD_PRELOAD" in asked[0] and "/x.so" not in asked[0]
    assert trust.is_trusted(root) is True
    assert trust.recorded_executables(root)["servers"] == trust.executables_snapshot(ws)


# --------------------------------------------------------------------------- the session question


def _gate(tmp_path: Path, boundary: Path, asker=None, **kw):
    from localharness.agent.gate import PermissionGate
    from localharness.config.grants import GrantStore

    return PermissionGate(boundary=boundary, workspace=boundary, grants=GrantStore(
        tmp_path / "grants.yaml"), mode="auto", asker=asker, channel_name="test", **kw)


@pytest.mark.asyncio
async def test_trusted_for_this_run_keeps_the_mode_and_records_nothing(tmp_path):
    from localharness.cli.session_trust import establish_session_trust

    project = tmp_path / "proj"
    project.mkdir()

    async def _boom(_request):
        raise AssertionError("asked a run that was trusted on the command line")

    gate = _gate(tmp_path, project, asker=_boom, trusted_for_run=True)

    assert await establish_session_trust(gate) == "auto"
    assert trust.is_trusted_tree(project) is None
    assert not trust.trust_store_path().exists()


@pytest.mark.asyncio
async def test_a_yes_in_session_never_approves_servers_it_did_not_show(tmp_path):
    from localharness.agent.gate_types import Decision
    from localharness.cli.session_trust import establish_session_trust

    project = tmp_path / "proj"
    _agent(project / WORKSPACE_DIR_NAME, "orchestrator.yaml", EVIL)

    async def _yes(_request):
        return Decision(kind="allow_once")

    notices: list[str] = []
    assert await establish_session_trust(_gate(tmp_path, project, asker=_yes),
                                         notices.append) == "auto"

    assert trust.is_trusted(project) is True
    executables = trust.recorded_executables(project)
    assert executables["fingerprint"] == "" and executables["servers"] == []
    assert notices == [NEXT_START_REVIEW_NOTICE]


@pytest.mark.asyncio
async def test_a_yes_in_session_for_a_project_without_servers_says_nothing_more(tmp_path):
    from localharness.agent.gate_types import Decision
    from localharness.cli.session_trust import establish_session_trust

    project = tmp_path / "plain"
    (project / WORKSPACE_DIR_NAME / "agents").mkdir(parents=True)

    async def _yes(_request):
        return Decision(kind="allow_once")

    notices: list[str] = []
    await establish_session_trust(_gate(tmp_path, project, asker=_yes), notices.append)

    assert notices == []
    assert trust.recorded_executables(project)["fingerprint"] == ""
