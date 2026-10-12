"""git_read validation, argv snapshots and the from-scratch environment (GITREAD-01, GITREAD-02).

Pure unit tests: nothing here spawns git except through a spy on subprocess.run.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from localharness.tools.builtin import git_read_tool as grt
from localharness.tools.builtin.git_read_tool import (
    DEFAULT_MAX_LINES,
    GIT_PIN_TIMEOUT_SECONDS,
    GIT_READ_PREFIX,
    GitReadValidationError,
    ValidatedCall,
    _git_env,
    _read_argv,
    _run_git,
    _validate_call,
)

GUARDS = ["--no-ext-diff", "--no-textconv", "--ignore-submodules=dirty"]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "ROOT"
    (r / "src").mkdir(parents=True)
    (r / "src" / "a.py").write_text("x\n")
    return r


def test_constants():
    assert grt.GIT_READ_TIMEOUT_SECONDS == 30
    assert grt.GIT_PIN_TIMEOUT_SECONDS == 5
    assert grt.DEFAULT_MAX_LINES == 400
    assert grt.MAX_LINES_LIMIT == 4000
    assert grt.STDERR_CAP_CHARS == 2000
    assert grt.PATH_FORM == "a path relative to the workspace root"
    assert GIT_READ_PREFIX == (
        "git", "--no-pager", "--no-lazy-fetch",
        "-c", "core.fsmonitor=false", "-c", "core.pager=cat",
        "-c", "core.hooksPath=/dev/null", "-c", "log.showSignature=false",
        "-c", "format.pretty=medium",
        "-c", "gpg.program=false", "-c", "gpg.openpgp.program=false",
        "-c", "gpg.ssh.program=false", "-c", "gpg.x509.program=false",
        "-c", "color.ui=never", "-c", "diff.submodule=short",
        "-c", "mailmap.file=", "-c", "mailmap.blob=",
        "-c", "core.attributesFile=/dev/null", "-c", "diff.orderFile=/dev/null",
    )
    assert grt.VERB_GUARD_FLAGS == tuple(GUARDS)


# ------------------------------------------------------------------ top-level argument types

@pytest.mark.parametrize("bad", ["status", "", None, 1, "DIFF", ["diff"]])
def test_subcommand_rejected(root, bad):
    with pytest.raises(GitReadValidationError):
        _validate_call(root, bad)


@pytest.mark.parametrize("sub", ["diff", "show", "log"])
def test_subcommand_accepted(root, sub):
    call = _validate_call(root, sub)
    assert call == ValidatedCall(sub, None, (), DEFAULT_MAX_LINES, False, False)


# ------------------------------------------------------------------ revision (D-02)

@pytest.mark.parametrize(
    "rev",
    ["", "-p", "--output=x", "-c", "--no-index", "-L1,2:f", "-Sfoo", "a\x00b", "a\nb", "a\tb", "a\x7f", 5],
)
def test_revision_rejected(root, rev):
    with pytest.raises(GitReadValidationError):
        _validate_call(root, "show", revision=rev)


@pytest.mark.parametrize("rev", ["HEAD", "main", "HEAD~1..HEAD", "A...B", "HEAD:README.md", "/etc/passwd"])
def test_revision_accepted_unchanged(root, rev):
    assert _validate_call(root, "show", revision=rev).revision == rev


# ------------------------------------------------------------------ paths (D-03)

@pytest.mark.parametrize(
    "p",
    ["", "a\x00", "a\tb", ":(prefix:3)x", ":(top,prefix:1)x", "../x", "a/../../x", ":/../../x",
     "-dash", "--output=x"],
)
def test_path_rejected(root, p):
    with pytest.raises(GitReadValidationError):
        _validate_call(root, "diff", paths=[p])


def test_absolute_outside_rejected_with_form_sentence(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    cases = [
        [str(outside / "f")],
        [str(outside), str(tmp_path / "other")],
        [str(root / "a"), "/etc"],
    ]
    for bad in cases:
        with pytest.raises(GitReadValidationError) as ei:
            _validate_call(root, "diff", paths=bad)
        assert "a path relative to the workspace root" in str(ei.value)
    with pytest.raises(GitReadValidationError) as ei:
        _validate_call(root, "diff", paths=["../a", "../b"])
    assert "a path relative to the workspace root" in str(ei.value)


def test_absolute_inside_relativized(root, tmp_path):
    assert _validate_call(root, "diff", paths=[str(root)]).paths == (".",)
    assert _validate_call(root, "diff", paths=[str(root / "src" / "a.py")]).paths == ("src/a.py",)
    link = tmp_path / "LINK"
    os.symlink(root, link)
    assert _validate_call(root, "diff", paths=[str(link / "src" / "a.py")]).paths == ("src/a.py",)


def test_absolute_dash_after_relativize_rejected(root):
    with pytest.raises(GitReadValidationError):
        _validate_call(root, "diff", paths=[str(root) + "/-dash"])


@pytest.mark.parametrize(
    "p",
    [":(top)x", ":/x", ":(glob)**/*.py", ":(literal)a*b", ":(icase)readme", ":(exclude)vendor",
     ":!vendor", ":^vendor", ":(attr:text)x", ":(top)../x", "a/../b", "./x"],
)
def test_path_accepted_unchanged(root, p):
    assert _validate_call(root, "diff", paths=[p]).paths == (p,)


def test_paths_container_types(root):
    assert _validate_call(root, "diff", paths=None).paths == ()
    assert _validate_call(root, "diff", paths=[]).paths == ()
    for bad in ("a", [1], ("a",), {"a"}):
        with pytest.raises(GitReadValidationError):
            _validate_call(root, "diff", paths=bad)


# ------------------------------------------------------------------ max_lines / stat / cached (D-01, D-63)

@pytest.mark.parametrize("bad", [0, 4001, -1, True, False, "5", 1.5])
def test_max_lines_rejected(root, bad):
    with pytest.raises(GitReadValidationError):
        _validate_call(root, "log", max_lines=bad)


@pytest.mark.parametrize("ok", [1, 400, 4000])
def test_max_lines_accepted(root, ok):
    assert _validate_call(root, "log", max_lines=ok).max_lines == ok


def test_max_lines_none_defaults(root):
    assert _validate_call(root, "log", max_lines=None).max_lines == 400


def test_flags_must_be_real_bools(root):
    for kw in ("stat", "cached"):
        for bad in ("true", 1, 0, None):
            with pytest.raises(GitReadValidationError):
                _validate_call(root, "diff", **{kw: bad})


@pytest.mark.parametrize("sub", ["show", "log"])
def test_cached_only_with_diff(root, sub):
    with pytest.raises(GitReadValidationError) as ei:
        _validate_call(root, sub, cached=True)
    assert "cached" in str(ei.value)
    assert "\n" not in str(ei.value)


def test_cached_with_diff_ok(root):
    assert _validate_call(root, "diff", cached=True).cached is True


# ------------------------------------------------------------------ argv snapshots (D-04, D-35, D-43, D-44)

def _argv(root, *a, pathspecs=None, **kw):
    call = _validate_call(root, *a, **kw)
    return _read_argv(call, list(call.paths) if pathspecs is None else pathspecs)


def test_argv_snapshots(root):
    P = list(GIT_READ_PREFIX)
    assert _argv(root, "diff") == [*P, "diff", *GUARDS, "--"]
    assert _argv(root, "diff", revision="HEAD", paths=["a"], cached=True, stat=True) == [
        *P, "diff", *GUARDS, "--cached", "--stat", "HEAD", "--", "a"]
    assert _argv(root, "diff", cached=True) == [*P, "diff", *GUARDS, "--cached", "--"]
    # CR-02 / CR-51: show and log pin the built-in pretty format, so a configured format.pretty or
    # pretty.* alias (which may hold %G?) never applies; diff has no pretty format.
    assert _argv(root, "show", revision="HEAD") == [*P, "show", *GUARDS, "--pretty=medium", "HEAD", "--"]
    assert _argv(root, "show", stat=True) == [*P, "show", *GUARDS, "--stat", "--pretty=medium", "--"]
    assert _argv(root, "log") == [*P, "log", *GUARDS, "-p", "--pretty=medium", "--"]
    assert _argv(root, "log", stat=True) == [*P, "log", *GUARDS, "--stat", "--pretty=medium", "--"]
    assert not any(t.startswith("--pretty") for t in _argv(root, "diff"))


def test_argv_uses_given_pathspecs(root):
    call = _validate_call(root, "diff", paths=["a"])
    argv = _read_argv(call, [":(exclude)x", "a"])
    assert argv[-3:] == ["--", ":(exclude)x", "a"]


def test_no_safe_directory_anywhere(root):
    assert not any("safe.directory" in t for t in GIT_READ_PREFIX)
    for a in (("diff",), ("show",), ("log",)):
        assert not any("safe.directory" in t for t in _argv(root, *a))


# ------------------------------------------------------------------ environment (D-05, D-74)

def test_env_exact(root, tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith(("GIT_", "LC_", "XDG_")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/x")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "evil")
    monkeypatch.setenv("GIT_DIR", "/elsewhere")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'a=b'")
    monkeypatch.setenv("LANG", "de_DE")
    monkeypatch.setenv("LC_ALL", "de_DE")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/x")
    assert _git_env(root) == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/x",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_CEILING_DIRECTORIES": os.path.dirname(os.path.realpath(root)),
    }


@pytest.mark.unsets_home
def test_env_without_home(root, monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    assert "HOME" not in _git_env(root)


def test_env_relative_path_entries_dropped(root, monkeypatch):
    monkeypatch.setenv("PATH", ".:bin::/usr/bin:/bin")
    assert _git_env(root)["PATH"] == "/usr/bin:/bin"
    monkeypatch.setenv("PATH", "bin:.")
    assert "PATH" not in _git_env(root)


def test_env_path_entries_inside_the_root_dropped(root, tmp_path, monkeypatch):
    """WR-53: an absolute entry under the workspace root (a venv's bin, node_modules/.bin) is dropped,
    also when it is reached through a symlink or is the root itself."""
    inside = root / ".venv" / "bin"
    inside.mkdir(parents=True)
    link = tmp_path / "lnk"
    os.symlink(inside, link)
    monkeypatch.setenv("PATH", os.pathsep.join([str(inside), str(link), str(root), "/usr/bin", "/bin"]))
    assert _git_env(root)["PATH"] == "/usr/bin:/bin"
    monkeypatch.setenv("PATH", str(inside))
    assert "PATH" not in _git_env(root)


def test_env_ceiling_is_realpath_parent(tmp_path):
    real = tmp_path / "DIR" / "sub"
    real.mkdir(parents=True)
    link = tmp_path / "elsewhere" / "L"
    link.parent.mkdir()
    os.symlink(real, link)
    ceiling = _git_env(link)["GIT_CEILING_DIRECTORIES"]
    assert ceiling == os.path.realpath(tmp_path / "DIR")
    assert ceiling and not ceiling.startswith(":")


# ------------------------------------------------------------------ the runner (D-06, D-37)

def test_run_git_spawn_shape(root, monkeypatch):
    seen: dict = {}

    def spy(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", spy)
    _run_git(root, ["version"])
    assert tuple(seen["argv"][: len(GIT_READ_PREFIX)]) == GIT_READ_PREFIX
    assert seen["argv"][-1] == "version"
    kw = seen["kw"]
    assert kw["cwd"] == os.path.realpath(root)
    assert kw["env"] == _git_env(root)
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["timeout"] == GIT_PIN_TIMEOUT_SECONDS
    assert not kw.get("shell", False)
    assert kw["capture_output"] is True


def test_run_git_timeout_override(root, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: seen.update(kw) or subprocess.CompletedProcess(argv, 0, b"", b""))
    _run_git(root, ["version"], timeout=30)
    assert seen["timeout"] == 30
