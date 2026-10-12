"""git_read registration: the boundary keyword, silent pin gating, the floor, start's hoist, children (GITREAD-03, -05).

Real git runs in tmp_path only, under git_support's from-scratch environment and a scratch HOME.
"""
from __future__ import annotations

import ast
import logging
import os
import shutil
from pathlib import Path

import pytest

from localharness.tools.builtin import git_read_tool as grt
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.git_read_tool import GitReadTool
from localharness.tools.registry import ToolRegistry
from tests.unit.git_support import git, make_repo, requires_git

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"


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
def repo(base: Path) -> Path:
    return make_repo(base / "REPO")


@pytest.fixture
def outside(base: Path) -> Path:
    return make_repo(base / "outside")


def _held(reg: ToolRegistry) -> dict:
    return reg._tools["global"]


def _spy(monkeypatch) -> list[Path]:
    calls: list[Path] = []
    real = grt.git_read_pin_failure

    def spy(root):
        calls.append(root)
        return real(root)

    monkeypatch.setattr(grt, "git_read_pin_failure", spy)
    return calls


# ------------------------------------------------------------------ the boundary keyword (D-08, D-09, D-60)

async def test_no_boundary_registers_no_git_read_and_runs_no_git(monkeypatch):
    calls = _spy(monkeypatch)
    for kwargs in ({}, {"boundary": None}):
        reg = ToolRegistry()
        await register_builtin_tools(reg, **kwargs)
        assert "git_read" not in _held(reg)
    assert calls == []


@requires_git
async def test_healthy_repository_registers_git_read_at_the_realpath(repo, base, monkeypatch):
    calls = _spy(monkeypatch)
    link = base / "link"
    os.symlink(repo, link)
    reg = ToolRegistry()
    await register_builtin_tools(reg, boundary=link)
    tool = _held(reg)["git_read"]
    assert isinstance(tool, GitReadTool)
    assert tool._root == Path(os.path.realpath(repo))
    assert len(calls) == 1


@requires_git
async def test_git_read_lands_beside_the_other_builtins(repo):
    reg = ToolRegistry()
    await register_builtin_tools(reg, boundary=repo)
    assert {"read", "glob", "grep", "bash_exec", "git_read"} <= set(_held(reg))


@requires_git
async def test_boundary_below_the_toplevel_registers_nothing_silently(repo, capsys, caplog):
    sub = repo / "sub"
    (sub / ".localharness").mkdir(parents=True)
    from localharness.agent.gate import derive_session_boundary

    assert derive_session_boundary(cwd=sub, local_dir=sub / ".localharness") == Path(os.path.realpath(sub))
    reg = ToolRegistry()
    with caplog.at_level(logging.DEBUG):
        await register_builtin_tools(reg, boundary=sub)
    assert "git_read" not in _held(reg)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@requires_git
async def test_dot_git_symlink_to_outside_registers_nothing_silently(base, outside, capsys, caplog):
    d = base / "DIR"
    d.mkdir()
    os.symlink(outside / ".git", d / ".git")
    reg = ToolRegistry()
    with caplog.at_level(logging.DEBUG):
        await register_builtin_tools(reg, boundary=d)
    assert "git_read" not in _held(reg)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_a_directory_without_a_repository_registers_nothing_and_runs_no_git(base, monkeypatch):
    calls = _spy(monkeypatch)
    plain = base / "plain"
    plain.mkdir()
    reg = ToolRegistry()
    await register_builtin_tools(reg, boundary=plain)
    assert "git_read" not in _held(reg)
    assert calls == []


def _swap(repo: Path, outside: Path, rel: str) -> Path:
    link = repo / ".git" / rel
    target = outside / ".git" / rel
    assert target.exists()
    if link.is_dir() and not link.is_symlink():
        shutil.rmtree(link)
    else:
        link.unlink()
    os.symlink(target, link)
    return link


_RELS = ["objects", "objects/pack", "refs/heads/main", "config"]


@requires_git
@pytest.mark.parametrize("rel", _RELS)
async def test_symlink_in_git_dir_blocks_registration_silently(repo, outside, rel, capsys, caplog):
    _swap(repo, outside, rel)
    reg = ToolRegistry()
    with caplog.at_level(logging.DEBUG):
        await register_builtin_tools(reg, boundary=repo)
    assert "git_read" not in _held(reg)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@requires_git
@pytest.mark.parametrize("rel", _RELS)
async def test_symlink_swapped_in_after_registration_is_refused_per_call(repo, outside, rel):
    reg = ToolRegistry()
    await register_builtin_tools(reg, boundary=repo)
    tool = _held(reg)["git_read"]
    (repo / "README").write_text("hello\nmore\n")
    assert (await tool.run(subcommand="diff")).success is True
    link = _swap(repo, outside, rel)
    result = await tool.run(subcommand="diff")
    assert result.success is False
    assert result.error_type == "permission_denied"
    assert result.error == (
        "git_read refused: the git directory is outside the workspace root (" + repr(str(link)) + ")"
    )
    assert "fatal" not in result.error


@requires_git
def test_home_boundary_is_none_and_registers_nothing(base, monkeypatch):
    import asyncio

    from localharness.agent.gate import derive_session_boundary

    home = Path(os.environ["HOME"])
    git(home, "init", "-q")
    boundary = derive_session_boundary(cwd=home)
    assert boundary is None
    calls = _spy(monkeypatch)
    reg = ToolRegistry()
    asyncio.run(register_builtin_tools(reg, boundary=boundary))
    assert "git_read" not in _held(reg) and calls == []


# ------------------------------------------------------------------ the root floor, the harness names (D-19, D-64)

@requires_git
async def test_root_floor_admits_git_read_and_changes_nothing(repo):
    from localharness.config.models import ToolConfig
    from localharness.tools.capabilities import apply_root_capability_floor
    from tests.unit.test_capabilities import WEB, _production_registry

    reg = await _production_registry()
    before = ToolConfig()
    expected = apply_root_capability_floor(before, reg.global_schemas(), enabled=True)
    await reg.register(GitReadTool(repo), scope="global")
    cfg = ToolConfig()
    assert apply_root_capability_floor(cfg, reg.global_schemas(), enabled=True) == WEB == expected
    assert "git_read" not in cfg.deny


def test_git_read_is_a_harness_tool_name():
    from localharness.tools.builtin.bash_tool import _HARNESS_TOOL_NAMES

    assert "git_read" in _HARNESS_TOOL_NAMES
