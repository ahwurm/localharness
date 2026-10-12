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
from tests.unit.git_support import (
    REAL_GIT,
    fake_git,
    git,
    git_env,
    git_with_env,
    make_repo,
    marker_script,
    partial_clone,
    requires_git,
    signed_history,
    submodule_repo,
)


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


@requires_git
async def test_a_line_past_the_stream_limit_is_a_marked_truncation_not_a_silent_drop(root, base, monkeypatch):
    body = 'case " $* " in *" log "*) echo first; head -c 9000000 /dev/zero | tr "\\0" x; echo; exit 0 ;; esac'
    fake = fake_git(base / "fake", body)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    result = await GitReadTool(root).run(subcommand="log", max_lines=100)
    assert result.success is True
    assert result.truncated is True
    assert result.output.startswith("first\n")
    # WR-59: the marker names the real cause and the lines actually kept, not the max_lines cap.
    assert result.output.endswith(
        "[git_read: output cut after 1 lines: a line longer than 8 MiB; narrow with paths]")
    assert "100 lines" not in result.output
    assert len(result.output) < 1024 * 1024


@requires_git
async def test_a_first_line_past_the_stream_limit_is_never_no_output(root, base, monkeypatch):
    """WR-03: the very first line is the over-long one: the answer is the marker, not '(no output)'."""
    body = 'case " $* " in *" log "*) head -c 9000000 /dev/zero | tr "\\0" x; echo; exit 0 ;; esac'
    fake = fake_git(base / "fake", body)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    result = await GitReadTool(root).run(subcommand="log")
    assert result.success is True and result.truncated is True
    assert result.output == "[git_read: output cut after 0 lines: a line longer than 8 MiB; narrow with paths]"
    assert "(no output)" not in result.output


@requires_git
async def test_output_is_cut_at_a_byte_cap_with_its_own_marker(root, base, monkeypatch):
    """WR-58: max_lines bounds the count only; many medium lines stop at MAX_RETURNED_BYTES."""
    monkeypatch.setattr(grt, "MAX_RETURNED_BYTES", 1000)
    (root / "wide.txt").write_text("".join(("x" * 99) + "\n" for _ in range(100)))
    git(root, "add", "wide.txt")
    git(root, "commit", "-q", "-m", "wide")
    result = await GitReadTool(root).run(subcommand="show", revision="HEAD", max_lines=4000)
    assert result.success is True and result.truncated is True
    marker = result.output.rsplit("\n", 1)[-1]
    assert marker.startswith("[git_read: truncated at 1000 bytes after ")
    assert marker.endswith(" lines; narrow with paths]")
    assert len(result.output) < 1000 + 200
    assert "truncated at 4000 lines" not in result.output


@requires_git
async def test_a_single_line_over_the_byte_cap_keeps_its_head(root, base, monkeypatch):
    """WR-58: the first line alone is past the cap: the part that fits is kept, then the marker."""
    monkeypatch.setattr(grt, "MAX_RETURNED_BYTES", 500)
    body = 'case " $* " in *" log "*) head -c 5000 /dev/zero | tr "\\0" y; echo; exit 0 ;; esac'
    fake = fake_git(base / "fake", body)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    result = await GitReadTool(root).run(subcommand="log")
    assert result.truncated is True
    assert result.output.startswith("y" * 500 + "\n")
    assert result.output.endswith("[git_read: truncated at 500 bytes after 1 lines; narrow with paths]")
    assert result.output != "(no output)"


@requires_git
async def test_output_exactly_at_the_byte_cap_gets_no_marker(root, base, monkeypatch):
    body = 'case " $* " in *" log "*) printf "aaaa\\nbbbb\\n"; exit 0 ;; esac'
    fake = fake_git(base / "fake", body)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(grt, "MAX_RETURNED_BYTES", 10)
    result = await GitReadTool(root).run(subcommand="log")
    assert result.truncated is False and result.output == "aaaa\nbbbb\n"


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


# ============================================================================ the no-execution matrix
#
# Each vector: the CONTROL (bare git under git_support's environment with the fixture's config) must
# touch the marker, which proves the fixture really arms the vector; then every listed tool call must
# leave the marker absent. A control that does not fire means the fixture is wrong, not the tool safe.

TOOL_READS = [
    pytest.param({"subcommand": "diff"}, id="diff"),
    pytest.param({"subcommand": "diff", "stat": True}, id="diff-stat"),
    pytest.param({"subcommand": "diff", "revision": "HEAD"}, id="diff-HEAD"),
    pytest.param({"subcommand": "show", "revision": "HEAD"}, id="show-HEAD"),
    pytest.param({"subcommand": "log"}, id="log"),
]


@pytest.fixture
def scripts(base: Path) -> Path:
    d = base / "scripts"
    d.mkdir()
    return d


@pytest.fixture
def armed(base: Path) -> Path:
    """A repository with two commits touching a.txt, a modified worktree file, and no vector configured."""
    repo = make_repo(base / "ROOT", {"a.txt": "one\n", "README": "hello\n"})
    (repo / "a.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "second")
    (repo / "a.txt").write_text("three\n")
    return repo


def _fires(marker: Path) -> bool:
    """True when the marker exists; consumes it so the next call starts clean."""
    fired = marker.exists()
    marker.unlink(missing_ok=True)
    return fired


async def _tool_leaves_absent(repo: Path, marker: Path, kwargs: dict, *, bump: Path | None = None) -> None:
    marker.unlink(missing_ok=True)
    if bump is not None:
        os.utime(bump, (time.time() + 5, time.time() + 5))
    result = await GitReadTool(repo).run(**kwargs)
    assert not marker.exists(), f"the marker ran under {kwargs}: {result}"


# ------------------------------------------------------------------------------------ diff.external

@requires_git
async def test_control_diff_external_runs_under_bare_git(armed, scripts):
    script, marker = marker_script(scripts, "ext")
    git(armed, "config", "diff.external", str(script))
    assert git(armed, "diff").returncode == 0
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize("kwargs", TOOL_READS)
async def test_diff_external_never_runs(armed, scripts, kwargs):
    script, marker = marker_script(scripts, "ext")
    git(armed, "config", "diff.external", str(script))
    await _tool_leaves_absent(armed, marker, kwargs)


# --------------------------------------------------------------------------- aliases shadowing verbs (D-73)

@requires_git
async def test_control_alias_marker_runs_under_bare_git(armed, scripts):
    script, marker = marker_script(scripts, "al")
    git(armed, "config", "alias.mydiff", f"!{script}")
    git(armed, "mydiff")
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"subcommand": "diff"}, id="diff"),
        pytest.param({"subcommand": "show", "revision": "HEAD"}, id="show-HEAD"),
        pytest.param({"subcommand": "log"}, id="log"),
    ],
)
async def test_aliases_shadowing_the_verbs_never_run(armed, scripts, kwargs):
    script, marker = marker_script(scripts, "al")
    for verb in ("diff", "log", "show"):
        git(armed, "config", f"alias.{verb}", f"!{script}")
    # git ignores an alias that shadows a built-in even for bare git; the row guards a regression.
    await _tool_leaves_absent(armed, marker, kwargs)


# ------------------------------------------------------------------------------------ textconv drivers

def _arm_textconv(repo: Path, script: Path) -> None:
    (repo / ".gitattributes").write_text("*.txt diff=tc\n")
    git(repo, "add", ".gitattributes")
    git(repo, "commit", "-q", "-m", "attrs")
    git(repo, "config", "diff.tc.textconv", str(script))
    (repo / "a.txt").write_text("four\n")


@requires_git
async def test_control_textconv_runs_under_bare_git(armed, scripts):
    script, marker = marker_script(scripts, "tc", passthrough=True)
    _arm_textconv(armed, script)
    git(armed, "diff", "--no-ext-diff")
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [
        *TOOL_READS,
        pytest.param({"subcommand": "log", "stat": True}, id="log-stat"),
    ],
)
async def test_textconv_never_runs(armed, scripts, kwargs):
    script, marker = marker_script(scripts, "tc", passthrough=True)
    _arm_textconv(armed, script)
    await _tool_leaves_absent(armed, marker, kwargs)


@requires_git
async def test_control_textconv_on_blob_show_runs_under_bare_git(armed, scripts):
    script, marker = marker_script(scripts, "tc", passthrough=True)
    _arm_textconv(armed, script)
    git(armed, "show", "--textconv", "HEAD:a.txt")
    assert _fires(marker)


@requires_git
async def test_blob_show_returns_the_raw_blob_without_textconv(armed, scripts):
    script, marker = marker_script(scripts, "tc", passthrough=True)
    _arm_textconv(armed, script)
    marker.unlink(missing_ok=True)
    result = await GitReadTool(armed).run(subcommand="show", revision="HEAD:a.txt")
    assert result.success is True
    assert result.output == "two\n"
    assert not marker.exists()


# ------------------------------------------------------------------------------------------- pagers

def _pty_run(cwd: Path, argv: list[str], env: dict[str, str]) -> None:
    """Run `argv` with stdout on a pseudo terminal (the only way git starts a pager) and wait for it."""
    import pty
    import select
    import subprocess

    try:
        master, slave = pty.openpty()
    except OSError:  # IN-54: a sandbox without /dev/ptmx or /dev/pts
        pytest.skip("no pseudo terminal available")
    env = {**env, "TERM": "xterm"}
    proc = None
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave)
        os.close(slave)
        slave = -1
        deadline = time.monotonic() + 20  # IN-02: a git or pager that never closes the slave cannot hang the run
        try:
            while time.monotonic() < deadline:  # drain so a full pty buffer cannot block git
                ready, _, _ = select.select([master], [], [], max(0.0, min(1.0, deadline - time.monotonic())))
                if ready and not os.read(master, 65536):
                    break
        except OSError:
            pass
        proc.wait(timeout=20)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        if slave != -1:
            os.close(slave)
        os.close(master)


def _pty_git(cwd: Path, *args: str) -> None:
    _pty_run(cwd, [REAL_GIT, *args], git_env(cwd.parent))


def _arm_pagers(repo: Path, script: Path) -> None:
    for key in ("core.pager", "pager.log", "pager.diff", "pager.show"):
        git(repo, "config", key, str(script))


@requires_git
@pytest.mark.skipif(os.name == "nt", reason="needs a pseudo terminal")
async def test_control_pager_runs_on_a_terminal(armed, scripts):
    script, marker = marker_script(scripts, "pg")
    _arm_pagers(armed, script)
    _pty_git(armed, "log")
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"subcommand": "diff"}, id="diff"),
        pytest.param({"subcommand": "show", "revision": "HEAD"}, id="show-HEAD"),
        pytest.param({"subcommand": "log"}, id="log"),
    ],
)
async def test_pagers_never_run(armed, scripts, kwargs):
    script, marker = marker_script(scripts, "pg")
    _arm_pagers(armed, script)
    await _tool_leaves_absent(armed, marker, kwargs)


@requires_git
@pytest.mark.skipif(os.name == "nt", reason="needs a pseudo terminal")
@pytest.mark.parametrize("verb", ["diff", "show", "log"])
async def test_tool_argv_neutralises_the_pager_even_on_a_terminal(armed, scripts, verb):
    """The tool's pipes never give git a terminal, so the prefix is proven by running its own argv on a pty."""
    script, marker = marker_script(scripts, "pg")
    _arm_pagers(armed, script)
    call = grt._validate_call(armed, verb, "HEAD" if verb == "show" else None)
    _pty_run(armed, grt._read_argv(call, call.paths), grt._git_env(armed))
    assert not marker.exists()


# --------------------------------------------------------------------------------- hooks and fsmonitor

@requires_git
async def test_control_index_hook_runs_under_bare_git(armed, scripts):
    hooks = scripts / "hooks"
    _script, marker = marker_script(hooks, "post-index-change")
    git(armed, "config", "core.hooksPath", str(hooks))
    os.utime(armed / "README", (time.time() + 5, time.time() + 5))
    git(armed, "diff")
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({"subcommand": "diff"}, id="diff"), pytest.param({"subcommand": "diff", "stat": True}, id="diff-stat")],
)
async def test_index_hook_never_runs(armed, scripts, kwargs):
    hooks = scripts / "hooks"
    _script, marker = marker_script(hooks, "post-index-change")
    git(armed, "config", "core.hooksPath", str(hooks))
    await _tool_leaves_absent(armed, marker, kwargs, bump=armed / "README")


@requires_git
async def test_control_fsmonitor_runs_under_bare_git(armed, scripts):
    script, marker = marker_script(scripts, "fsm")
    git(armed, "config", "core.fsmonitor", str(script))
    git(armed, "diff")
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({"subcommand": "diff"}, id="diff"), pytest.param({"subcommand": "diff", "stat": True}, id="diff-stat")],
)
async def test_fsmonitor_never_runs(armed, scripts, kwargs):
    script, marker = marker_script(scripts, "fsm")
    git(armed, "config", "core.fsmonitor", str(script))
    await _tool_leaves_absent(armed, marker, kwargs)


# --------------------------------------------------------------------------------- signature programs

def _arm_signatures(repo: Path, script: Path) -> str:
    tag = signed_history(repo, "v1")
    git(repo, "config", "log.showSignature", "true")
    git(repo, "config", "gpg.program", str(script))
    return tag


@requires_git
async def test_control_gpg_program_runs_for_commit_and_tag(armed, scripts):
    script, marker = marker_script(scripts, "gpg")
    tag = _arm_signatures(armed, script)
    git(armed, "log", "-1")
    assert _fires(marker)
    git(armed, "show", tag)
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"subcommand": "log"}, id="log"),
        pytest.param({"subcommand": "show", "revision": "HEAD"}, id="show-HEAD"),
        pytest.param({"subcommand": "show", "revision": "v1"}, id="show-tag"),
    ],
)
async def test_gpg_program_never_runs(armed, scripts, kwargs):
    script, marker = marker_script(scripts, "gpg")
    _arm_signatures(armed, script)
    await _tool_leaves_absent(armed, marker, kwargs)


@requires_git
@pytest.mark.parametrize("pretty_key", ["format.pretty", "pretty.mine"])
async def test_signature_placeholder_in_a_configured_pretty_format_never_runs_gpg(armed, scripts, pretty_key):
    """CR-02 / CR-51: `%G?` in format.pretty or a pretty.* alias makes bare git run gpg.program per signed
    commit; the tool pins --pretty=medium and the gpg programs, so the marker stays absent."""
    script, marker = marker_script(scripts, "gpgfmt")
    tag = signed_history(armed, "v1")
    git(armed, "config", "gpg.program", str(script))
    git(armed, "config", "gpg.openpgp.program", str(script))
    git(armed, "config", pretty_key, "format:%H %G?")
    if pretty_key == "pretty.mine":
        git(armed, "config", "format.pretty", "mine")
    git(armed, "log", "-1")
    assert _fires(marker), "control: the fixture must arm the vector under bare git"
    git(armed, "show", "-s")
    assert _fires(marker)
    for kwargs in (
        {"subcommand": "log"},
        {"subcommand": "log", "stat": True},
        {"subcommand": "show", "revision": "HEAD"},
        {"subcommand": "show", "revision": tag},
    ):
        await _tool_leaves_absent(armed, marker, kwargs)


# --------------------------------------------------------------------------------------- lazy fetch

@requires_git
async def test_control_promisor_fetch_runs_under_bare_git(base, scripts):
    script, marker = marker_script(scripts, "up")
    dst = partial_clone(base, script)
    git(dst, "show", "HEAD:a.txt", check=False)
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"subcommand": "show", "revision": "HEAD:a.txt"}, id="show-blob"),
        pytest.param({"subcommand": "log"}, id="log"),
    ],
)
async def test_missing_blobs_are_never_fetched(base, scripts, kwargs):
    script, marker = marker_script(scripts, "up")
    dst = partial_clone(base, script)
    marker.unlink(missing_ok=True)
    result = await GitReadTool(dst).run(**kwargs)
    assert result.success is False
    assert not marker.exists()


# ----------------------------------------------------------------------------- core.sshCommand

def _arm_ssh_command(repo: Path, script: Path) -> None:
    git(repo, "config", "core.sshCommand", str(script))
    git(repo, "remote", "add", "origin", "ssh://example.invalid/repo.git")


@requires_git
async def test_control_ssh_command_runs_under_bare_git(armed, scripts):
    script, marker = marker_script(scripts, "ssh")
    _arm_ssh_command(armed, script)
    git(armed, "fetch", "origin", check=False)
    assert _fires(marker), "control: the fixture must arm the vector under bare git"


@requires_git
@pytest.mark.parametrize("kwargs", TOOL_READS)
async def test_ssh_command_in_the_repo_config_never_runs(armed, scripts, kwargs):
    """A repo-config `core.sshCommand` runs on a network transport; a git_read call never opens one."""
    script, marker = marker_script(scripts, "ssh")
    _arm_ssh_command(armed, script)
    await _tool_leaves_absent(armed, marker, kwargs)


# --------------------------------------------------------------------------------- submodule filters

@requires_git
async def test_control_submodule_filter_runs_under_bare_git(base, scripts):
    script, marker = marker_script(scripts, "sf", passthrough=True)
    sup = submodule_repo(base, script)
    marker.unlink(missing_ok=True)
    out = git(sup, "diff").stdout
    assert "Submodule sub" in out  # diff.submodule=log is what bare git honours
    assert _fires(marker)


@requires_git
@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({"subcommand": "diff"}, id="diff"), pytest.param({"subcommand": "diff", "stat": True}, id="diff-stat")],
)
async def test_submodule_filter_never_runs(base, scripts, kwargs):
    script, marker = marker_script(scripts, "sf", passthrough=True)
    sup = submodule_repo(base, script)
    await _tool_leaves_absent(sup, marker, kwargs)


@requires_git
async def test_submodule_pointer_change_is_short_form(base, scripts):
    script, marker = marker_script(scripts, "sf", passthrough=True)
    sup = submodule_repo(base, script)
    (sup / "sub" / "f.txt").write_text("committed in sub\n")
    git(sup / "sub", "commit", "-q", "-am", "inner")  # runs the clean filter: that is the bare-git control
    assert _fires(marker)
    assert "Submodule sub" in git(sup, "diff").stdout
    marker.unlink(missing_ok=True)
    result = await GitReadTool(sup).run(subcommand="diff")
    assert result.success is True
    assert "Subproject commit" in result.output
    assert "Submodule sub" not in result.output
    assert not marker.exists()


# -------------------------------------------------------------------------------- inherited environment

@requires_git
@pytest.mark.parametrize("variant", ["external", "parameters", "count", "git_dir", "all"])
async def test_inherited_git_environment_is_ignored(armed, base, scripts, monkeypatch, variant):
    script, marker = marker_script(scripts, "env")
    other = make_repo(base / "other", {"other.txt": "other\n"})
    (other / "other.txt").write_text("other changed\n")
    env = {
        "external": {"GIT_EXTERNAL_DIFF": str(script)},
        "parameters": {"GIT_CONFIG_PARAMETERS": f"'diff.external'='{script}'"},
        "count": {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "diff.external", "GIT_CONFIG_VALUE_0": str(script)},
        "git_dir": {"GIT_DIR": str(other / ".git")},
    }
    chosen = {k: v for d in env.values() for k, v in d.items()} if variant == "all" else env[variant]
    if variant != "all":
        # The control: bare git under the same inherited variable obeys it.
        git_with_env(armed, chosen, "diff", check=False)
        if variant == "git_dir":
            out = git_with_env(armed, chosen, "diff").stdout
            assert "other.txt" in out and "a.txt" not in out
        else:
            assert _fires(marker)
    marker.unlink(missing_ok=True)
    for key, value in chosen.items():
        monkeypatch.setenv(key, value)
    result = await GitReadTool(armed).run(subcommand="diff")
    assert result.success is True
    assert "a.txt" in result.output and "-two" in result.output and "+three" in result.output
    assert "other.txt" not in result.output
    assert not marker.exists()
