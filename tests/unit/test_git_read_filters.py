"""git_read clean-filter narrowing (GITREAD-04): the pathspec magic parser, the narrowing table, and the
precheck + trailing line proven on real clean and process filters (D-14, D-15, D-16, D-34, D-45..D-47).

Real git runs in tmp_path only, under git_support's from-scratch environment. A marker filter that never
runs is the proof that a repository-configured command stayed unexecuted; each fixture has a bare-git
control that does run it. Timeouts are enforced by the tool itself, not by pytest-timeout.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from localharness.tools.builtin import git_read_tool as grt
from localharness.tools.builtin.git_read_tool import (
    GitReadTool,
    GitReadValidationError,
    _magic,
    _narrow_for_filters,
    _validate_call,
)
from tests.unit.git_support import fake_git, filter_repo, git, requires_git


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

# ------------------------------------------------------------------------------------ _magic (D-47)

@pytest.mark.parametrize(
    "element, expected",
    [
        ("x", ("plain", [], "x")),
        ("docs/a b", ("plain", [], "docs/a b")),
        (":(top)x", ("long", ["top"], "x")),
        (":(top,attr:text)x", ("long", ["top", "attr:text"], "x")),
        (":(glob, literal)x", ("long", ["glob", "literal"], "x")),
        (":(exclude)x", ("long", ["exclude"], "x")),
        (":/x", ("short", ["top"], "x")),
        (":!x", ("short", ["exclude"], "x")),
        (":^x", ("short", ["exclude"], "x")),
        (":!:x", ("short", ["exclude"], "x")),
        (":/!x", ("short", ["top", "exclude"], "x")),
        (":foo", ("short", [], "foo")),
    ],
)
def test_magic_parses_every_form(element, expected):
    assert _magic(element) == expected


@pytest.mark.parametrize("element", [":(top", ":(glob,literal", ":(", ":(attr:text x"])
def test_unterminated_long_magic_is_rejected(element):
    with pytest.raises(GitReadValidationError):
        _magic(element)


@pytest.mark.parametrize("element", [":(top", ":(glob,literal"])
def test_validation_rejects_unterminated_magic_before_any_process(tmp_path, element, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise AssertionError("a process ran during validation")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(GitReadValidationError) as err:
        _validate_call(tmp_path, "diff", None, [element])
    assert "unterminated" in str(err.value)


# ------------------------------------------------------------------------ _narrow_for_filters table

BARE = [":(attr:!filter)", ":(attr:-filter)"]

NARROW_TABLE = [
    (["x"], [":(attr:!filter)x", ":(attr:-filter)x"]),
    (["docs/"], [":(attr:!filter)docs/", ":(attr:-filter)docs/"]),
    ([":(top)x"], [":(attr:!filter,top)x", ":(attr:-filter,top)x"]),
    ([":/x"], [":(attr:!filter,top)x", ":(attr:-filter,top)x"]),
    ([":(attr:text)x"], [":(attr:!filter text)x", ":(attr:-filter text)x"]),
    (
        [":(top,attr:text,glob)x"],
        [":(attr:!filter text,top,glob)x", ":(attr:-filter text,top,glob)x"],
    ),
    ([":(literal)a*b"], [":(attr:!filter,literal)a*b", ":(attr:-filter,literal)a*b"]),
    ([], BARE),
    ([":!vendor/"], [*BARE, ":!vendor/"]),
    ([":^vendor/"], [*BARE, ":^vendor/"]),
    ([":(exclude)vendor/"], [*BARE, ":(exclude)vendor/"]),
    ([":(exclude,top)vendor/"], [*BARE, ":(exclude,top)vendor/"]),
    ([":!:vendor/"], [*BARE, ":!:vendor/"]),
    (
        ["docs/", ":!docs/sub"],
        [":(attr:!filter)docs/", ":(attr:-filter)docs/", ":!docs/sub"],
    ),
    (
        [":!docs/sub", "docs/"],
        [":!docs/sub", ":(attr:!filter)docs/", ":(attr:-filter)docs/"],  # input order is kept
    ),
    (
        ["a", "b"],
        [":(attr:!filter)a", ":(attr:-filter)a", ":(attr:!filter)b", ":(attr:-filter)b"],
    ),
]


@pytest.mark.parametrize("paths, expected", NARROW_TABLE)
def test_narrowing_table(paths, expected):
    assert _narrow_for_filters(paths) == expected


@pytest.mark.parametrize("paths, _expected", NARROW_TABLE)
def test_no_output_element_combines_exclude_with_attr(paths, _expected):
    for element in _narrow_for_filters(paths):
        kind, words, _rest = _magic(element)
        has_attr = any(w.startswith("attr:") for w in words)
        assert not (has_attr and "exclude" in words), element
        assert element != ":(exclude,attr:filter)"
        # at most one attr: word per element (a second one is fatal in git, row b3)
        assert sum(1 for w in words if w.startswith("attr:")) <= 1, element


def test_narrowing_never_mutates_its_input():
    paths = [":(top)x", ":!y"]
    _narrow_for_filters(paths)
    assert paths == [":(top)x", ":!y"]



def test_constants_are_the_decided_ones():
    assert grt.FILTER_KEY_PATTERN == r"^filter\..*\.(clean|process)$"
    assert grt.FILTER_OMITTED_LINE.format(key="filter.x.clean") == (
        "[git_read: a filter driver is configured ('filter.x.clean'); paths it binds were omitted"
        " \u2014 cached: true or a two-revision diff includes them]"
    )


# ------------------------------------------------------------------------- real clean/process filters

OMITTED_X_CLEAN = (
    "[git_read: a filter driver is configured ('filter.x.clean'); paths it binds were omitted"
    " \u2014 cached: true or a two-revision diff includes them]"
)
OMITTED_X_PROCESS = OMITTED_X_CLEAN.replace("filter.x.clean", "filter.x.process")


class Calls:
    """Every git invocation the tool made: the sync ones (pins, precheck) and the async read."""

    def __init__(self) -> None:
        self.sync: list[list[str]] = []
        self.reads: list[list[str]] = []

    def prechecks(self) -> list[list[str]]:
        return [a for a in self.sync if "--get-regexp" in a]

    def all_argv(self) -> list[list[str]]:
        return [*self.sync, *self.reads]


@pytest.fixture
def calls(monkeypatch) -> Calls:
    rec = Calls()
    real_run = subprocess.run
    real_exec = asyncio.create_subprocess_exec

    def run(args, *a, **k):
        rec.sync.append([str(x) for x in args])
        return real_run(args, *a, **k)

    async def exec_(*args, **k):
        rec.reads.append([str(x) for x in args])
        return await real_exec(*args, **k)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_)
    return rec


def _assert_no_forbidden_shapes(rec: Calls) -> None:
    for argv in rec.all_argv():
        for element in argv:
            assert element != ":(exclude,attr:filter)"
            if element.startswith(":(") and ")" in element:
                words = [w.strip() for w in element[2 : element.index(")")].split(",")]
                assert not ("exclude" in words and any(w.startswith("attr:") for w in words)), element


def _control_fires(repo: Path, marker: Path, *args: str) -> None:
    """A bare git on the same repository runs the driver: the fixture can detect a filter run at all."""
    marker.unlink(missing_ok=True)
    git(repo, *args, check=False)
    assert marker.exists(), "control: bare git did not run the filter"
    marker.unlink()


@requires_git
@pytest.mark.parametrize("kind, line", [("clean", OMITTED_X_CLEAN), ("process", OMITTED_X_PROCESS)])
async def test_worktree_diff_is_narrowed_and_the_filter_never_runs(base, calls, kind, line):
    repo, marker = filter_repo(base, kind)
    _control_fires(repo, marker, "diff")
    result = await GitReadTool(repo).run(subcommand="diff")
    assert result.success is True, result.error
    assert "+worktree" in result.output and "a/README" in result.output
    assert "docs/a.md" not in result.output and "vendor/v.txt" not in result.output
    assert result.output.splitlines()[-1] == line
    assert not marker.exists()
    assert len(calls.prechecks()) == 1
    assert calls.reads[0][-3:] == ["--", ":(attr:!filter)", ":(attr:-filter)"]
    _assert_no_forbidden_shapes(calls)


@requires_git
@pytest.mark.parametrize("kind", ["clean", "process"])
@pytest.mark.parametrize("call", [{"stat": True}, {"revision": "HEAD"}, {"revision": "HEAD", "stat": True}])
async def test_stat_and_one_revision_diffs_are_narrowed_too(base, calls, kind, call):
    repo, marker = filter_repo(base, kind)
    _control_fires(repo, marker, "diff", *(["--stat"] if call.get("stat") else []), *([call["revision"]] if "revision" in call else []))
    result = await GitReadTool(repo).run(subcommand="diff", **call)
    assert result.success is True, result.error
    assert "README" in result.output
    assert "docs/a.md" not in result.output and "vendor/v.txt" not in result.output
    assert result.output.splitlines()[-1].startswith("[git_read: a filter driver is configured (")
    assert not marker.exists()


@requires_git
async def test_top_magic_composes_into_the_long_form(base, calls):
    repo, marker = filter_repo(base, "clean")
    result = await GitReadTool(repo).run(subcommand="diff", paths=[":(top)README"])
    assert result.success is True, result.error
    assert "+worktree" in result.output and "docs/a.md" not in result.output
    assert calls.reads[0][-3:] == ["--", ":(attr:!filter,top)README", ":(attr:-filter,top)README"]
    assert not marker.exists()
    _assert_no_forbidden_shapes(calls)


@requires_git
async def test_exclude_only_paths_diff_only_the_unbound_paths_outside_it(base, calls):
    repo, marker = filter_repo(base, "clean")
    result = await GitReadTool(repo).run(subcommand="diff", paths=[":!vendor/"])
    assert result.success is True, result.error
    assert "a/README" in result.output
    assert "vendor/v.txt" not in result.output and "docs/a.md" not in result.output
    assert calls.reads[0][-4:] == ["--", ":(attr:!filter)", ":(attr:-filter)", ":!vendor/"]
    assert not marker.exists()
    assert result.output.splitlines()[-1] == OMITTED_X_CLEAN
    _assert_no_forbidden_shapes(calls)


@requires_git
@pytest.mark.parametrize("kind, line", [("clean", OMITTED_X_CLEAN), ("process", OMITTED_X_PROCESS)])
async def test_cached_two_revision_show_and_log_include_the_bound_paths(base, calls, kind, line):
    repo, marker = filter_repo(base, kind, stage=True)
    tool = GitReadTool(repo)
    cached = await tool.run(subcommand="diff", cached=True)
    assert cached.success and "docs/a.md" in cached.output and "vendor/v.txt" in cached.output
    assert line not in cached.output
    assert calls.prechecks() == []  # D-14: cached owes no precheck
    two = await tool.run(subcommand="diff", revision="HEAD~1..HEAD")
    assert two.success and "docs/a.md" in two.output and "+second" in two.output
    assert "filter driver is configured" not in two.output
    assert len(calls.prechecks()) == 1  # the precheck ran, git then said two revisions
    before = len(calls.prechecks())
    shown = await tool.run(subcommand="show", revision="HEAD")
    logged = await tool.run(subcommand="log")
    for r in (shown, logged):
        assert r.success and "docs/a.md" in r.output
        assert "filter driver is configured" not in r.output
    assert len(calls.prechecks()) == before
    assert not marker.exists()
    _assert_no_forbidden_shapes(calls)


@requires_git
async def test_no_driver_is_a_plain_diff_with_no_trailing_line(base, calls):
    repo, marker = filter_repo(base, "none")
    result = await GitReadTool(repo).run(subcommand="diff")
    assert result.success is True, result.error
    assert "docs/a.md" in result.output and "vendor/v.txt" in result.output
    assert "filter driver is configured" not in result.output
    assert calls.reads[0][-1] == "--"
    assert not marker.exists()


@requires_git
async def test_a_failing_precheck_fails_closed_and_spawns_no_read(base, calls, monkeypatch):
    repo, _marker = filter_repo(base, "clean")
    fake = fake_git(base / "fake", 'case " $* " in *" --get-regexp "*) echo boom >&2; exit 6 ;; esac')
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    result = await GitReadTool(repo).run(subcommand="diff")
    assert result.success is False
    assert result.error_type == "execution_error"
    assert result.error.startswith("git exited with 6: boom")
    assert calls.reads == []


@requires_git
async def test_a_signalled_precheck_fails_closed(base, calls, monkeypatch):
    repo, _marker = filter_repo(base, "clean")
    fake = fake_git(base / "fake", 'case " $* " in *" --get-regexp "*) kill -ABRT $$ ;; esac')
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    result = await GitReadTool(repo).run(subcommand="diff")
    assert result.success is False and "-6" in result.error
    assert calls.reads == []


@requires_git
async def test_a_hung_precheck_times_out_and_spawns_no_read(base, calls, monkeypatch):
    repo, _marker = filter_repo(base, "clean")
    fake = fake_git(base / "fake", 'case " $* " in *" --get-regexp "*) exec sleep 30 ;; esac')
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(grt, "GIT_PIN_TIMEOUT_SECONDS", 0.5)
    result = await GitReadTool(repo).run(subcommand="diff")
    assert result.success is False and "timed out" in result.error
    assert calls.reads == []


@requires_git
async def test_a_driver_key_with_a_space_is_named_through_repr(base):
    repo, marker = filter_repo(base, "clean", driver_key="filter.a b.clean")
    result = await GitReadTool(repo).run(subcommand="diff")
    assert result.success is True, result.error
    assert result.output.splitlines()[-1] == OMITTED_X_CLEAN.replace("filter.x.clean", "filter.a b.clean")
    assert not marker.exists()


@requires_git
async def test_a_driver_reached_through_include_is_seen_between_two_calls(base, calls):
    repo, marker = filter_repo(base, "none")
    script = base / "filters" / "drv"
    git(repo, "config", "include.path", "inc.cfg")  # relative to .git/config
    (repo / ".git" / "inc.cfg").write_text("")
    tool = GitReadTool(repo)
    first = await tool.run(subcommand="diff")
    assert first.success and "docs/a.md" in first.output
    assert "filter driver is configured" not in first.output
    (repo / ".git" / "inc.cfg").write_text(f'[filter "x"]\n\tclean = {script}\n')
    second = await tool.run(subcommand="diff")
    assert second.success, second.error
    assert "docs/a.md" not in second.output and "a/README" in second.output
    assert second.output.splitlines()[-1] == OMITTED_X_CLEAN
    assert not marker.exists()
    assert len(calls.prechecks()) == 2  # no cache of any kind


@requires_git
async def test_truncation_line_comes_before_the_filter_line(base):
    repo, marker = filter_repo(base, "clean")
    result = await GitReadTool(repo).run(subcommand="diff", max_lines=1)
    assert result.success is True and result.truncated is True
    assert result.output.splitlines()[-2:] == ["[git_read: truncated at 1 lines]", OMITTED_X_CLEAN]
    assert not marker.exists()


@requires_git
async def test_a_failed_narrowed_read_carries_no_trailing_line(base):
    repo, _marker = filter_repo(base, "clean")
    result = await GitReadTool(repo).run(subcommand="diff", revision="nonexistent-rev")
    assert result.success is False
    assert "filter driver is configured" not in (result.error or "") + result.output


@requires_git
async def test_an_odd_revision_answer_narrows_the_safe_direction(base, monkeypatch):
    repo, marker = filter_repo(base, "clean")
    monkeypatch.setattr(grt, "_run_git", _rev_parse_times_out(grt._run_git))
    result = await GitReadTool(repo).run(subcommand="diff", revision="HEAD~1..HEAD")
    assert result.success is True, result.error
    assert "docs/a.md" not in result.output
    assert result.output.splitlines()[-1] == OMITTED_X_CLEAN
    assert not marker.exists()


def _rev_parse_times_out(real):
    def wrapper(root, args, **k):
        if "--revs-only" in args:
            raise subprocess.TimeoutExpired("git", 1)
        return real(root, args, **k)

    return wrapper
