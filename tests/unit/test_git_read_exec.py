"""GitReadTool: declaration, validation before any process, per-call pins, the streaming read (GITREAD-02, -03, -05).

Real git runs in tmp_path only, under git_support's from-scratch environment; the owner's global git
configuration, ~/.localharness and the project's own .git are never touched. Timeouts are enforced by
the tool itself (its own read timeout), not by pytest-timeout.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from localharness.tools.builtin import git_read_tool as grt
from localharness.tools.builtin.git_read_tool import GitReadTool
from tests.unit.git_support import fake_git, git, make_repo, requires_git


@pytest.fixture(autouse=True)
def _scratch_home(tmp_path, monkeypatch):
    home = tmp_path / "_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    b = Path(os.path.realpath(tmp_path)) / "w"
    b.mkdir()
    return b


@pytest.fixture
def root(base: Path) -> Path:
    return make_repo(base / "ROOT", {"a.txt": "one\n", "README": "hello\n"})


# ----------------------------------------------------------------------------------- the declaration

def test_declaration_is_the_fixed_one(tmp_path, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise AssertionError("constructing GitReadTool ran a process")

    monkeypatch.setattr(subprocess, "run", boom)
    tool = GitReadTool(tmp_path)  # no repository at all: construction runs nothing
    schema = tool.info()
    assert schema.name == "git_read"
    assert schema.group == "fs.read"
    assert schema.gate_family == "allow"
    assert schema.ingest == "none" and "ingest" in schema.model_fields_set
    assert schema.host == "safe"
    assert schema.result_origin == "trusted"
    assert schema.destructive is False
    assert GitReadTool.timeout_s == 60.0
    props = schema.parameters["properties"]
    assert set(props) == {"subcommand", "revision", "paths", "max_lines", "stat", "cached"}
    assert schema.parameters["required"] == ["subcommand"]
    assert props["subcommand"]["enum"] == ["diff", "show", "log"]
    assert props["paths"] == {
        "type": "array", "items": {"type": "string"}, "description": props["paths"]["description"],
    }
    assert props["max_lines"]["type"] == "integer"
    assert (props["max_lines"]["default"], props["max_lines"]["minimum"], props["max_lines"]["maximum"]) == (
        400, 1, 4000,
    )
    assert props["stat"]["type"] == "boolean" and props["stat"]["default"] is False
    assert props["cached"]["type"] == "boolean" and props["cached"]["default"] is False
    assert props["revision"]["type"] == "string"


# ------------------------------------------------------------------ validation before any process (D-63)

@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [
        {"subcommand": "status"},
        {"subcommand": "diff", "revision": "--output=x"},
        {"subcommand": "diff", "paths": ["../x"]},
        {"subcommand": "diff", "paths": ["/definitely/outside/the/root"]},
        {"subcommand": "log", "max_lines": 0},
        {"subcommand": "log", "cached": True},
    ],
)
async def test_invalid_call_spawns_nothing(root, monkeypatch, kwargs):
    import asyncio
    import subprocess

    def boom(*a, **k):
        raise AssertionError("a process was spawned for an invalid call")

    async def aboom(*a, **k):
        raise AssertionError("a process was spawned for an invalid call")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", aboom)
    result = await GitReadTool(root).run(**kwargs)
    assert result.success is False
    assert result.error_type == "validation_error"


# ------------------------------------------------------------------------- per-call pin re-check (D-11)

@requires_git
async def test_git_dir_swapped_after_construction_is_refused(base, root):
    outside = make_repo(base / "outside", {"secret": "s\n"})
    tool = GitReadTool(root)
    assert (await tool.run(subcommand="log")).success is True
    (root / ".git").rename(base / "moved-git")
    (root / ".git").symlink_to(outside / ".git")
    result = await tool.run(subcommand="log")
    assert result.success is False
    assert result.error_type == "permission_denied"
    assert result.error.startswith("git_read refused: the git directory is outside the workspace root (")


# --------------------------------------------------------------------------------------- the reads

@requires_git
async def test_diff_show_log_basic_reads(root):
    tool = GitReadTool(root)
    (root / "a.txt").write_text("one\ntwo\n")
    diff = await tool.run(subcommand="diff")
    assert diff.success and "diff --git" in diff.output and "+two" in diff.output
    git(root, "add", "a.txt")
    assert (await tool.run(subcommand="diff")).output == "(no output)"
    staged = await tool.run(subcommand="diff", cached=True)
    assert staged.success and "+two" in staged.output
    git(root, "commit", "-q", "-m", "second")
    shown = await tool.run(subcommand="show", revision="HEAD")
    assert shown.success and "second" in shown.output and "+two" in shown.output
    log = await tool.run(subcommand="log")
    assert log.success and "diff --git" in log.output  # D-35: -p
    stat = await tool.run(subcommand="log", stat=True)
    assert stat.success and " | " in stat.output and "diff --git" not in stat.output


@requires_git
async def test_clean_tree_diff_is_no_output(root):
    result = await GitReadTool(root).run(subcommand="diff")
    assert result.success is True
    assert result.output == "(no output)"
    assert result.truncated is False


@requires_git
async def test_absolute_revision_with_no_paths_is_git_usage_failure(root):
    result = await GitReadTool(root).run(subcommand="diff", revision="/etc/passwd")
    assert result.success is False
    assert result.error_type == "execution_error"
    assert "git exited with 129" in result.error
    assert "usage" in result.error
    assert result.output == ""


@requires_git
async def test_non_utf8_blob_is_replaced_not_fatal(root):
    (root / "bin.txt").write_bytes(b"caf\xe9\n")
    git(root, "add", "bin.txt")
    git(root, "commit", "-q", "-m", "latin1")
    result = await GitReadTool(root).run(subcommand="show", revision="HEAD")
    assert result.success is True
    assert "�" in result.output


# ------------------------------------------------------------------------------------- failure shapes

@requires_git
async def test_signalled_git_is_a_failure_naming_the_signal(root, base, monkeypatch):
    fake = fake_git(base / "fake", 'case " $* " in *" log "*) kill -ABRT $$ ;; esac')
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    result = await GitReadTool(root).run(subcommand="log")
    assert result.success is False
    assert result.error_type == "execution_error"
    assert "-6" in result.error


@requires_git
async def test_stderr_is_capped_at_two_thousand_characters(root, base, monkeypatch):
    fake = fake_git(
        base / "fake",
        "case \" $* \" in *\" log \"*) head -c 5000 /dev/zero | tr '\\0' x >&2; exit 1 ;; esac",
    )
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    result = await GitReadTool(root).run(subcommand="log")
    assert result.success is False
    assert result.error.startswith("git exited with 1: ")
    assert len(result.error) - len("git exited with 1: ") == 2000
    assert set(result.error[len("git exited with 1: "):]) == {"x"}


# ------------------------------------------------------------------------------- truncation (D-36)

@requires_git
async def test_log_stops_at_max_lines_with_the_truncation_line(root):
    for i in range(30):
        (root / "a.txt").write_text(f"v{i}\n")
        git(root, "commit", "-q", "-am", f"c{i}")
    result = await GitReadTool(root).run(subcommand="log", max_lines=5)
    assert result.success is True
    assert result.truncated is True
    lines = result.output.split("\n")
    assert len(lines) == 6
    assert lines[-1] == "[git_read: truncated at 5 lines]"
    assert lines[0].startswith("commit ")


@requires_git
async def test_exactly_max_lines_gets_no_truncation_line(root):
    (root / "a.txt").write_text("one\ntwo\n")
    git(root, "commit", "-q", "-am", "second")
    tool = GitReadTool(root)
    full = await tool.run(subcommand="show", revision="HEAD", max_lines=4000)
    assert full.success and full.truncated is False
    n = len(full.output.rstrip("\n").split("\n"))
    assert 5 < n < 4000
    exact = await tool.run(subcommand="show", revision="HEAD", max_lines=n)
    assert exact.truncated is False
    assert "truncated" not in exact.output
    assert exact.output == full.output
    short = await tool.run(subcommand="show", revision="HEAD", max_lines=n - 1)
    assert short.truncated is True
    assert short.output.endswith("[git_read: truncated at %d lines]" % (n - 1))


@requires_git
async def test_huge_output_is_terminated_not_buffered(root):
    (root / "big.txt").write_text("".join(f"line {i}\n" for i in range(20000)))
    git(root, "add", "big.txt")
    git(root, "commit", "-q", "-m", "big")
    result = await GitReadTool(root).run(subcommand="show", revision="HEAD", max_lines=50)
    assert result.success is True and result.truncated is True
    assert len(result.output.split("\n")) == 51


# --------------------------------------------------------------------------------------- timeout

@requires_git
async def test_read_past_the_timeout_is_killed(root, base, monkeypatch):
    fake = fake_git(base / "fake", 'case " $* " in *" log "*) exec sleep 30 ;; esac')
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(grt, "GIT_READ_TIMEOUT_SECONDS", 1)
    t0 = time.monotonic()
    result = await GitReadTool(root).run(subcommand="log")
    assert time.monotonic() - t0 < 10
    assert result.success is False
    assert result.error_type == "timeout_error"
    assert result.error == "git_read timed out after 1s"
