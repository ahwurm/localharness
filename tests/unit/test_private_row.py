"""A private session's agent cannot write what a later session would load; reads and scratch pass."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from localharness.agent.gate_types import GateSettings, Grant, ToolMeta, Verdict
from localharness.agent.verdict import (
    NO_TRACE_COMMAND_REASON,
    NO_TRACE_FILE_REASON,
    NO_TRACE_PATH_REASON,
    GateContext,
    evaluate,
    instructions_name,
)

SETTINGS = GateSettings()
WRITE = ToolMeta(destructive=True, group="fs.write")
SHELL = ToolMeta(destructive=True, group="shell")
MODES = ("guarded", "trusted", "auto", "unattended")
FILE = NO_TRACE_FILE_REASON.format(name="CLAUDE.md")
COMMAND = NO_TRACE_COMMAND_REASON.format(name="CLAUDE.md")


def no_grants(workspace: Path, klass: str, key: str) -> Grant | None:
    return None


class World:
    def __init__(self, home: Path, in_project: bool = False):
        self.cfg = home / ".localharness"
        self.ws = home / "proj"
        self.scratch = (self.ws / ".localharness" if in_project else self.cfg) / "run" / "aaaaaaaaaaaa" / "scratch"
        self.scratch.mkdir(parents=True)
        self.ws.mkdir(exist_ok=True)
        (self.ws / "CLAUDE.md").write_text("rules")
        (self.ws / "a.txt").write_text("a")
        (self.ws / "x").write_text("x")
        os.link(self.ws / "CLAUDE.md", self.ws / "HARD.md")
        (self.ws / "VIA.md").symlink_to("CLAUDE.md")
        (self.ws / "AGENTS.md").symlink_to("real.md")
        (self.ws / "real.md").write_text("r")

    def ctx(self, **kw) -> GateContext:
        kw.setdefault("mode", "guarded")
        kw.setdefault("leaves_no_trace", True)
        return GateContext(
            boundary=self.ws, workspace=self.ws, grants=no_grants, has_review_surface=True,
            config_dir=self.cfg, run=self.scratch.parent, scratch=self.scratch,
            state_dirs=(self.scratch.parent / "state",), **kw)

    def write(self, path: Path | str, tool: str = "write", **kw):
        params = ({"path": str(path), "content": "x"} if tool == "write"
                  else {"path": str(path), "old_string": "a", "new_string": "b"})
        return evaluate(tool, params, WRITE, self.ctx(**kw), SETTINGS)

    def shell(self, command: str, **kw):
        return evaluate("bash_exec", {"command": command, "working_dir": str(self.ws)}, SHELL, self.ctx(**kw), SETTINGS)


def _home(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    return home


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(_home(tmp_path, monkeypatch))


@pytest.fixture
def project_world(tmp_path, monkeypatch):
    return World(_home(tmp_path, monkeypatch), in_project=True)


WRITES = ["CLAUDE.md", "AGENTS.md", "NEWDIR/../CLAUDE.md", "claude.md", "HARD.md", "VIA.md", "real.md"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tool", ["write", "edit"])
@pytest.mark.parametrize("name", WRITES)
def test_a_write_or_edit_of_an_instructions_file_or_what_reaches_it_is_refused(world, mode, tool, name):
    result = world.write(world.ws / name, tool, mode=mode)
    assert result.verdict is Verdict.DENY
    assert result.reason.startswith("A private session does not write ") and "write NOTES.md instead" in result.reason


def test_a_new_instructions_file_is_refused_and_the_name_is_the_candidate_spelling(world):
    (world.ws / "CLAUDE.md").unlink()
    assert world.write(world.ws / "CLAUDE.md").reason == FILE
    assert world.write(world.ws / "agents.md").reason == NO_TRACE_FILE_REASON.format(name="AGENTS.md")
    assert world.write(world.ws / "sub" / "INSTRUCTIONS.md").reason == NO_TRACE_FILE_REASON.format(
        name=".localharness/INSTRUCTIONS.md")


COMMANDS = [
    "ln CLAUDE.md x2", "ln -f x CLAUDE.md", "ln -s CLAUDE.md x2", "cp -l CLAUDE.md x2", "cp --link CLAUDE.md x2",
    "cp -al CLAUDE.md x2", "cp -a CLAUDE.md x2", "mv CLAUDE.md x2", "rm CLAUDE.md", "rm -f ./CLAUDE.md",
    "git mv NOTES.md CLAUDE.md", "git checkout x -- CLAUDE.md", "git restore CLAUDE.md", "patch CLAUDE.md",
    "echo x >> HARD.md", "echo x > HARD.md", "echo x > CLAUDE.md", "sed -i s/a/b/ CLAUDE.md", "tee VIA.md",
    "cp a.txt CLAUDE.md", "cd sub && echo x >> CLAUDE.md", "rsync --link-dest=. CLAUDE.md x2",
    "install a.txt AGENTS.md", "echo x >> CLAUDE.md 2>$ERR", "cat x > CLAUDE.md 2> $LOCALHARNESS_SCRATCH/err",
    "echo x > HARD.md 2>$TMPDIR/err", "cp -s CLAUDE.md x2",
    "echo x >> $PWD/CLAUDE.md", 'cat >> "$PWD/AGENTS.md" <<EOF\nx\nEOF', "echo x > $HOME/.localharness/agents/a.yaml",
    "echo x > $PWD/.agents/skills/s/SKILL.md",
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("command", COMMANDS)
def test_a_shell_command_that_may_change_an_instructions_file_is_refused(world, mode, command):
    result = world.shell(command, mode=mode)
    assert result.verdict is Verdict.DENY
    assert "later session would load" in result.reason


@pytest.mark.parametrize("command", ["git restore CLAUDE.md", "git checkout x -- CLAUDE.md", "patch CLAUDE.md"])
def test_a_command_that_only_names_the_file_gets_the_command_reason(world, command):
    assert world.shell(command).reason == COMMAND


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("path", [".agents/skills/s/SKILL.md", ".localharness/config.yaml", ".Agents/x", "sub/.localharness/a"])
def test_a_skill_or_a_projects_localharness_folder_is_refused(world, mode, path):
    result = world.write(world.ws / path, mode=mode)
    assert result.verdict is Verdict.DENY
    assert result.reason == NO_TRACE_PATH_REASON.format(path=repr(str(world.ws / path)))
    assert result.reason.endswith("; ask for it again in a normal session.")
    assert world.shell(f"mkdir -p {path}", mode=mode).verdict is Verdict.DENY
    assert world.shell(f"touch {path}", mode=mode).verdict is Verdict.DENY


@pytest.mark.parametrize("mode", MODES)
def test_the_config_folder_is_refused_outside_scratch(world, mode):
    for path in (world.cfg / "x", world.cfg / "grants.yaml", world.cfg / "run" / "aaaaaaaaaaaa" / "state" / "y"):
        assert world.write(path, mode=mode).verdict is Verdict.DENY
    assert world.shell(f"echo x > {world.cfg}/x", mode=mode).verdict is Verdict.DENY
    assert world.shell(f"sed -i s/a/b/ {world.cfg}/grants.yaml", mode=mode).verdict is Verdict.DENY


def test_a_path_with_an_escape_or_a_newline_comes_out_one_quoted_line(world):
    result = world.write(world.ws / ".agents" / "a\x1b[31m\nb")
    assert result.verdict is Verdict.DENY
    assert "\x1b" not in result.reason and "\n" not in result.reason and "'" in result.reason


PASSES = [
    "cp CLAUDE.md NOTES.md", "cp -r CLAUDE.md NOTES.md", "cat CLAUDE.md", "grep x CLAUDE.md", "ls .agents",
    "echo x > NOTES.md", "echo x > $LOCALHARNESS_SCRATCH/o", "cp a.txt {s}/a.txt", "echo x > {s}/CLAUDE.md",
    "cp CLAUDE.md {s}/c.md", "rm -rf {s}/t3",
]


@pytest.mark.parametrize("command", PASSES)
def test_a_copy_a_read_and_scratch_pass_the_row(world, command):
    result = world.shell(command.format(s=world.scratch))
    assert result.verdict is not Verdict.DENY, result.reason


@pytest.mark.parametrize("mode", MODES)
def test_a_private_project_run_writes_its_scratch_and_not_its_state(project_world, mode):
    w = project_world
    run = w.scratch.parent
    assert w.write(w.scratch / "o", mode=mode).verdict is not Verdict.DENY
    assert w.shell("echo x > .localharness/run/aaaaaaaaaaaa/scratch/o", mode=mode).verdict is not Verdict.DENY
    assert w.shell(f"echo x > {w.scratch}/o", mode=mode).verdict is not Verdict.DENY
    for path in (run / "state" / "memory.db", run.parent / "bbbbbbbbbbbb" / "scratch" / "o"):
        assert w.write(path, mode=mode).verdict is Verdict.DENY
        assert w.shell(f"echo x > {path}", mode=mode).verdict is Verdict.DENY
    denied = w.write(w.ws / ".localharness" / "agents" / "a.yaml", mode=mode)
    assert denied.verdict is Verdict.DENY and denied.reason.startswith("A private session does not write ")


def test_a_read_and_a_scratch_write_pass_the_row(world):
    assert world.write(world.ws / "NOTES.md").verdict is not Verdict.DENY
    assert world.write(world.scratch / "o").verdict is not Verdict.DENY
    assert world.write(world.scratch / "CLAUDE.md").verdict is not Verdict.DENY


@pytest.mark.parametrize("name", WRITES + [".agents/skills/s/SKILL.md", ".localharness/config.yaml"])
def test_a_normal_run_is_unchanged(world, name):
    result = world.write(world.ws / name, leaves_no_trace=False)
    assert result.reason is None or "later session would load" not in result.reason
    for command in ("echo x >> HARD.md", "git restore CLAUDE.md", "cp -l CLAUDE.md x2", "echo x > .agents/a"):
        assert "later session would load" not in (world.shell(command, leaves_no_trace=False).reason or "")


def test_instructions_name_folds_case_and_follows_a_candidate_link(world):
    root = world.ws
    assert instructions_name("docs/Claude.MD", root) == "claude.md"
    assert instructions_name("NOTES.md", root) is None
    assert instructions_name("real.md", root, (root / "AGENTS.md",)) == "real.md"
    assert instructions_name("AGENTS.md", root, (root / "AGENTS.md",)) == "agents.md"
    assert instructions_name("CLAUDE.md", root, (root / "AGENTS.md",)) is None
