"""The four login-time directories are protected paths (spec 20 section 12, owner-confirmed 2026-10-13).

A write below ``~/.config/systemd``, ``~/.local/share/systemd/user``, ``~/.config/environment.d``
or ``~/.config/autostart`` changes what the user's session starts at the next login, so the
agent's ``write`` is a ``protected-path`` ask in every mode. The test is component-wise: a sibling
whose name merely starts with the same letters is outside.

``Path.home()`` is conftest's hermetic fake home; nothing here touches the real one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate_types import (
    PROTECTED_PATHS_HOME_DEFAULT,
    GateSettings,
    ToolMeta,
    Verdict,
)
from localharness.agent.verdict import GateContext, evaluate

SETTINGS = GateSettings()
WRITE = ToolMeta(group="fs.write", destructive=True)

DIRS = [
    ".config/systemd",
    ".local/share/systemd/user",
    ".config/environment.d",
    ".config/autostart",
]


def _ctx(tmp_path: Path, mode: str) -> GateContext:
    root = (tmp_path / "proj").resolve()
    root.mkdir(exist_ok=True)
    return GateContext(boundary=root, workspace=root, grants=lambda *a: None, mode=mode,
                       has_review_surface=True)


def test_the_four_directories_are_the_tail_of_the_home_list_in_order():
    assert PROTECTED_PATHS_HOME_DEFAULT[-4:] == (
        "~/.config/systemd",
        "~/.local/share/systemd/user",
        "~/.config/environment.d",
        "~/.config/autostart",
    )
    assert "~/.ssh" in PROTECTED_PATHS_HOME_DEFAULT  # every earlier entry is kept
    assert "~/Documents/WindowsPowerShell" in PROTECTED_PATHS_HOME_DEFAULT


@pytest.mark.parametrize("mode", ["auto", "guarded"])
@pytest.mark.parametrize("relative", [
    ".config/systemd/user/evil.service",
    ".local/share/systemd/user/x.service",
    ".config/environment.d/50-path.conf",
    ".config/autostart/x.desktop",
])
def test_a_write_below_the_directory_is_a_protected_path_ask(tmp_path, mode, relative):
    target = Path.home() / relative
    result = evaluate("write", {"path": str(target), "content": "x"}, WRITE, _ctx(tmp_path, mode),
                      SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "protected-path"


@pytest.mark.parametrize("mode", ["auto", "guarded"])
@pytest.mark.parametrize("relative", DIRS)
def test_the_directory_itself_is_a_protected_path_ask(tmp_path, mode, relative):
    target = Path.home() / relative
    result = evaluate("write", {"path": str(target), "content": "x"}, WRITE, _ctx(tmp_path, mode),
                      SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "protected-path"


@pytest.mark.parametrize("relative", [
    ".config/systemd-backup/x",
    ".config/autostartx/y",
    ".config/environment.dx/z",
    ".local/share/systemd/usr/w",
])
def test_a_longer_sibling_name_is_not_a_protected_path(tmp_path, relative):
    target = Path.home() / relative
    result = evaluate("write", {"path": str(target), "content": "x"}, WRITE,
                      _ctx(tmp_path, "auto"), SETTINGS)
    klass = result.request.klass if result.request is not None else None
    assert klass != "protected-path"
