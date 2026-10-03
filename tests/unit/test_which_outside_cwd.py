"""A program is never taken from the current folder by accident (SEC-11, research Q8).

CPython 3.12's `shutil.which` puts the current directory first on Windows whenever the name has
no directory part — even with an explicit `path=` — so a repository holding `bash.exe` would be run
by every `bash_exec` call. `which_outside_cwd` searches each PATH entry by its own path (a name WITH
a directory part is looked up in that directory alone), skips the entries that mean "the current
directory" ("" and "."), and honours an absolute entry even when it is the folder you stand in.

The Windows-only insert itself cannot run here; the helper's rule is proven with a fake PATH and a
recording `shutil.which`, which shows the bare-name branch (the only one that inserts the current
directory) is never taken."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from localharness.core import which as which_mod
from localharness.core.which import which_outside_cwd

posix_only = pytest.mark.skipif(os.name != "posix", reason="executable bits are POSIX")


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def layout(tmp_path: Path, monkeypatch):
    """`bin/tool` is the real program; the folder you stand in holds a decoy `tool`."""
    real = _exe(tmp_path / "bin" / "tool")
    here = tmp_path / "repo"
    decoy = _exe(here / "tool")
    monkeypatch.chdir(here)
    return real, decoy, here


def _path(monkeypatch, *entries: str) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(entries))


@posix_only
def test_the_program_on_path_wins_over_a_decoy_in_the_current_folder(layout, monkeypatch):
    real, _decoy, _here = layout
    _path(monkeypatch, str(real.parent))
    assert which_outside_cwd("tool") == str(real)


@posix_only
def test_empty_and_dot_entries_are_skipped(layout, monkeypatch):
    real, _decoy, _here = layout
    _path(monkeypatch, "", ".", str(real.parent))
    assert which_outside_cwd("tool") == str(real)


@posix_only
def test_only_the_decoy_reachable_finds_nothing(layout, monkeypatch):
    _path(monkeypatch, "", ".")
    assert which_outside_cwd("tool") is None


@posix_only
def test_an_absolute_entry_is_honoured_even_when_it_is_the_current_folder(layout, monkeypatch):
    real, decoy, here = layout
    _path(monkeypatch, str(here), str(real.parent))
    assert which_outside_cwd("tool") == str(decoy)


def _recorder(monkeypatch) -> list[tuple[tuple, dict]]:
    calls: list[tuple[tuple, dict]] = []

    def which(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(which_mod.shutil, "which", which)
    return calls


def test_every_lookup_names_its_directory_and_never_passes_path(tmp_path, monkeypatch):
    """Each call is one argument whose directory part is its PATH entry: the bare-name branch,
    where CPython inserts the current directory on Windows, is never reached."""
    entries = [str(tmp_path / "a"), "", str(tmp_path / "b"), os.curdir, str(tmp_path / "c")]
    _path(monkeypatch, *entries)
    calls = _recorder(monkeypatch)

    assert which_outside_cwd("tool") is None

    assert [kwargs for _args, kwargs in calls] == [{}, {}, {}]
    assert [args for args, _kw in calls] == [(os.path.join(e, "tool"),) for e in entries
                                             if e not in ("", os.curdir)]
    for (arg,), _kw in calls:
        assert os.path.dirname(arg) in entries


@pytest.mark.parametrize("cmd", [os.path.join(".", "x"), os.path.join(os.sep, "usr", "bin", "env")])
def test_a_command_with_a_directory_part_is_looked_up_as_given(monkeypatch, cmd):
    calls = _recorder(monkeypatch)
    which_outside_cwd(cmd)
    assert calls == [((cmd,), {})]


def test_without_path_the_default_search_path_is_used_minus_the_current_folder(monkeypatch):
    monkeypatch.delenv("PATH", raising=False)
    calls = _recorder(monkeypatch)
    which_outside_cwd("tool")
    expected = [e for e in os.defpath.split(os.pathsep) if e not in ("", os.curdir)]
    assert [args for args, _kw in calls] == [(os.path.join(e, "tool"),) for e in expected]


def test_bash_exec_looks_bash_up_outside_the_current_folder(monkeypatch):
    from localharness.tools.builtin import bash_tool

    asked: list[str] = []
    monkeypatch.delenv("LOCALHARNESS_BASH", raising=False)
    monkeypatch.setattr(bash_tool, "which_outside_cwd",
                        lambda name: asked.append(name) or "/usr/bin/bash")
    found = bash_tool._find_bash()
    assert asked == ["bash"]
    if os.name != "nt":
        assert found == "/usr/bin/bash"


@pytest.mark.parametrize("module, attr, call, name", [
    ("localharness.provider.server", "which_outside_cwd", lambda m: m.find_vllm(Path("/nonexistent")), "vllm"),
    ("localharness.plugins.setup", "which_outside_cwd", lambda m: m.gpu_name(), "nvidia-smi"),
    ("localharness.cli.update_cmd", "which_outside_cwd", None, "uv"),
])
def test_the_other_lookups_go_through_the_helper(monkeypatch, tmp_path, module, attr, call, name):
    import importlib

    mod = importlib.import_module(module)
    asked: list[str] = []
    monkeypatch.setattr(mod, attr, lambda n: asked.append(n) or None)
    if call is None:  # update: only a uv-tool install looks uv up
        uv_prefix = tmp_path / "uv" / "tools" / "localharness"
        uv_prefix.mkdir(parents=True)
        monkeypatch.setattr(mod.sys, "prefix", str(uv_prefix))
        mod._upgrade_command()
    else:
        call(mod)
    assert name in asked
