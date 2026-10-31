"""The gate's deny class on an agent's task and session folders: refused in every mode, keyed to the path class."""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate_types import GateSettings, Grant, ToolMeta, Verdict
from localharness.agent.verdict import (
    PRIVATE_DOC_REASON,
    PRIVATE_PLAN_REASON,
    PRIVATE_STATE_REASON,
    GateContext,
    evaluate,
)

SETTINGS = GateSettings()
WRITE_META = ToolMeta(destructive=True, group="fs.write")
MODES = ("guarded", "trusted", "auto", "unattended")


def no_grants(workspace: Path, klass: str, key: str) -> Grant | None:
    return None


def make_ctx(workspace: Path, **kw) -> GateContext:
    kw.setdefault("boundary", workspace)
    kw.setdefault("grants", no_grants)
    kw.setdefault("has_review_surface", True)
    kw.setdefault("mode", "guarded")
    return GateContext(workspace=workspace, **kw)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    workspace = (home / "proj").resolve()
    workspace.mkdir()
    return workspace


def write(ctx: GateContext, path: Path | str, tool: str = "write"):
    params = {"path": str(path), "content": "x"} if tool == "write" else {"path": str(path), "old_string": "a", "new_string": "b"}
    return evaluate(tool, params, WRITE_META, ctx, SETTINGS)


@pytest.mark.parametrize("mode", MODES)
def test_a_write_to_a_tasks_plan_is_denied_in_every_mode_and_names_the_plan_command(ws, mode):
    plan = ws.parent / ".localharness" / "agents" / "a" / "tasks" / "t1" / "plan.md"
    plan.parent.mkdir(parents=True)
    result = write(make_ctx(ws, mode=mode), plan)
    assert result.verdict is Verdict.DENY
    assert "/task plan" in result.reason


def test_a_write_to_the_agents_own_config_is_not_denied(ws):
    config = ws.parent / ".localharness" / "agents" / "a" / "config.yaml"
    assert write(make_ctx(ws), config).verdict is not Verdict.DENY


SHELL_META = ToolMeta(destructive=True, group="shell")
READ_META = ToolMeta(group="fs.read", gate_family="allow")
PRIVATE_PREFIXES = (PRIVATE_PLAN_REASON, PRIVATE_STATE_REASON, PRIVATE_DOC_REASON.split("{doc}")[0])
SHELL_MODES = ("guarded", "auto", "unattended")


def shell(ctx: GateContext, command: str, settings: GateSettings = SETTINGS):
    return evaluate("bash_exec", {"command": command}, SHELL_META, ctx, settings)


def is_private(result) -> bool:
    return result.verdict is Verdict.DENY and result.reason.startswith(PRIVATE_PREFIXES)


@pytest.fixture(params=("global", "project"))
def config(request, ws):
    return ws.parent / ".localharness" if request.param == "global" else ws / ".localharness"


DENIED = (
    "echo x > {c}/agents/{n}/tasks/t1/plan.md",
    "printf x | tee {c}/agents/{n}/sessions/jtbd.md",
    "cp /tmp/x {c}/agents/{n}/tasks/t1/",
    "mv /tmp/x {c}/agents/{n}/tasks/t1/research.md",
    "rm {c}/agents/{n}/sessions/asleep.json",
    "rm -rf {c}/agents/{n}/tasks",
    "ln -s /tmp/evil {c}/agents/{n}/tasks/t1/plan.md",
    "ln {c}/agents/{n}/tasks/t1/plan.md /tmp/y",
    "rm -rf {c}/agents/{n}",
    "mv {c}/agents/{n} {c}/agents/other",
)


@pytest.mark.parametrize("mode", SHELL_MODES)
@pytest.mark.parametrize("name", ("a", "b"))
@pytest.mark.parametrize("command", DENIED)
def test_a_static_shell_write_removal_or_link_is_denied(ws, config, mode, name, command):
    assert is_private(shell(make_ctx(ws, mode=mode), command.format(c=config, n=name)))


@pytest.mark.parametrize("mode", SHELL_MODES)
@pytest.mark.parametrize("tool", ("write", "edit"))
@pytest.mark.parametrize("name", ("a", "b"))
def test_a_task_doc_write_or_edit_is_denied(ws, config, mode, name, tool):
    assert is_private(write(make_ctx(ws, mode=mode), config / "agents" / name / "tasks" / "t1" / "decisions.md", tool))


def test_the_refusal_is_keyed_to_the_path_class(ws):
    ctx, root = make_ctx(ws), ws.parent / ".localharness" / "agents" / "a"
    assert "/task accept decisions" in write(ctx, root / "tasks" / "t1" / "decisions.md").reason
    assert "/task plan" in shell(ctx, f"echo x > {root}/tasks/t1/plan.md").reason
    for command in (f"echo x > {root}/sessions/jtbd.md", f"rm -rf {root}/tasks", f"rm -rf {root}/tasks/t1", f"rm -rf {root}"):
        assert shell(ctx, command).reason == PRIVATE_STATE_REASON
    assert write(ctx, root / "tasks").reason == PRIVATE_STATE_REASON


def test_a_workspace_symlink_to_the_private_folder_is_denied(ws):
    tasks = ws.parent / ".localharness" / "agents" / "a" / "tasks"
    tasks.mkdir(parents=True)
    (ws / "link").symlink_to(tasks)
    assert is_private(write(make_ctx(ws), "link/t1/plan.md"))
    assert is_private(shell(make_ctx(ws), "echo x > link/t1/plan.md"))


def test_state_dirs_guard_a_folder_outside_every_config_folder_also_through_a_symlink(ws, tmp_path):
    state = tmp_path / "state"
    (state / "agents" / "a" / "tasks" / "t1").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(state)
    target = state / "agents" / "a" / "tasks" / "t1" / "plan.md"
    assert not is_private(write(make_ctx(ws), target))
    assert is_private(write(make_ctx(ws, state_dirs=(state,)), target))
    assert is_private(write(make_ctx(ws, state_dirs=(alias,)), target))


@pytest.mark.parametrize("path", (
    "agents/a/config.yaml", "agents/a/memory.db", "agents/a/tasks-old/x", "agents/a", "agents/a/notes.md",
))
def test_nothing_above_the_two_folders_is_denied(ws, config, path):
    assert not is_private(write(make_ctx(ws), config / path))


@pytest.mark.parametrize("command", ("rm -rf {c}/agents", "mv {c}/agents {c}/agents.bak", "rm -rf {ws}", "rm {c}/agents/a/config.yaml", "cp /tmp/x {c}/agents/a",
                                     "rm {c}/agents/a.yaml", "mv {c}/agents/a.yaml {c}/agents/b.yaml"))
def test_a_removal_or_write_above_the_two_folders_is_not_denied(ws, config, command):
    assert not is_private(shell(make_ctx(ws), command.format(c=config, ws=ws)))


@pytest.mark.parametrize("mode", SHELL_MODES)
@pytest.mark.parametrize("command", ("rm -rf {c}/agents/v1.2", "mv {c}/agents/v1.2 {c}/agents/old"))
def test_a_dotted_agent_folder_is_denied_by_type_and_a_dotted_file_is_not(ws, config, mode, command):
    (config / "agents" / "v1.2").mkdir(parents=True)
    (config / "agents" / "a.yaml").write_text("x")
    assert is_private(shell(make_ctx(ws, mode=mode), command.format(c=config)))
    assert not is_private(shell(make_ctx(ws, mode=mode), f"rm {config}/agents/a.yaml"))


@pytest.mark.parametrize("command", (
    "python3 -c \"open('{c}/agents/a/tasks/t1/plan.md','w')\"",
    "sh script.sh",
    "echo x > $D/plan.md",
    "echo x > {c}/agents/a/tasks/*/plan.md",
    "cd $D && echo x > agents/a/tasks/t1/plan.md",
    "rm",
))
def test_interpreter_script_variable_and_glob_targets_are_as_today(ws, config, command):
    assert not is_private(shell(make_ctx(ws), command.format(c=config)))


def test_a_read_of_a_task_doc_is_allowed(ws):
    doc = ws.parent / ".localharness" / "agents" / "a" / "tasks" / "t1" / "plan.md"
    assert evaluate("read", {"path": str(doc)}, READ_META, make_ctx(ws), SETTINGS).verdict is Verdict.ALLOW
    assert shell(make_ctx(ws), f"cat {doc}").verdict is Verdict.ALLOW


def test_owner_set_protected_entries_do_not_change_the_deny_class(ws):
    root = ws.parent / ".localharness"
    settings = GateSettings(protected_config_dir_entries=())
    ctx = make_ctx(ws)
    assert is_private(shell(ctx, f"echo x > {root}/agents/a/tasks/t1/plan.md", settings))
    assert is_private(shell(ctx, f"rm -rf {root}/agents/a", settings))
    assert not is_private(shell(ctx, f"echo x > {root}/config.yaml", settings))


def test_the_case_insensitive_spelling_is_denied(ws):
    assert is_private(write(make_ctx(ws), ws.parent / ".localharness" / "AGENTS" / "a" / "TASKS" / "t1" / "x.md"))


def test_an_empty_state_dirs_still_guards_the_config_folders_and_empty_calls_are_as_today(ws):
    ctx = make_ctx(ws, state_dirs=())
    assert is_private(write(ctx, ws.parent / ".localharness" / "agents" / "a" / "sessions" / "x"))
    assert not is_private(write(ctx, ""))
    assert shell(ctx, "").verdict is Verdict.ALLOW


def test_the_first_private_target_in_segment_order_is_named(ws):
    root = ws.parent / ".localharness" / "agents" / "a"
    command = f"echo x > a.txt; rm {root}/sessions/x; echo y > {root}/tasks/t1/plan.md"
    assert shell(make_ctx(ws), command).reason == PRIVATE_STATE_REASON
