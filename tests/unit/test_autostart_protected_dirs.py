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


# ------------------------------------------------------------------ the directories follow the XDG variables
#
# The user manager, the desktop autostart and `environment.d` read `$XDG_CONFIG_HOME` / `$XDG_DATA_HOME` when
# those are set, and so does the harness's own unit path. The protected set is expanded at evaluation time
# from the same list, so a relocated login-time directory is as protected as the default one.

XDG_RELATIVES = [
    ("XDG_CONFIG_HOME", "systemd/user/evil.service"),
    ("XDG_CONFIG_HOME", "environment.d/50-path.conf"),
    ("XDG_CONFIG_HOME", "autostart/x.desktop"),
    ("XDG_DATA_HOME", "systemd/user/x.service"),
]


def _ask(tmp_path: Path, target: Path, mode: str):
    return evaluate("write", {"path": str(target), "content": "x"}, WRITE, _ctx(tmp_path, mode), SETTINGS)


@pytest.mark.parametrize("mode", ["auto", "guarded"])
@pytest.mark.parametrize("variable, relative", XDG_RELATIVES)
def test_a_write_below_a_relocated_login_directory_is_a_protected_path_ask(tmp_path, monkeypatch, mode, variable,
                                                                           relative):
    moved = tmp_path / "moved"
    monkeypatch.setenv(variable, str(moved))

    result = _ask(tmp_path, moved / relative, mode)

    assert result.verdict is Verdict.ASK
    assert result.request.klass == "protected-path"


@pytest.mark.parametrize("variable, directory", [
    ("XDG_CONFIG_HOME", "systemd"), ("XDG_CONFIG_HOME", "environment.d"), ("XDG_CONFIG_HOME", "autostart"),
    ("XDG_DATA_HOME", "systemd/user"),
])
def test_the_relocated_directory_itself_is_a_protected_path_ask(tmp_path, monkeypatch, variable, directory):
    moved = tmp_path / "moved"
    monkeypatch.setenv(variable, str(moved))

    result = _ask(tmp_path, moved / directory, "auto")

    assert result.verdict is Verdict.ASK and result.request.klass == "protected-path"


@pytest.mark.parametrize("variable, sibling", [
    ("XDG_CONFIG_HOME", "systemd-backup/x"),
    ("XDG_CONFIG_HOME", "autostartx/y"),
    ("XDG_CONFIG_HOME", "environment.dx/z"),
    ("XDG_CONFIG_HOME", "other/x"),
    ("XDG_DATA_HOME", "systemd/usr/w"),
    ("XDG_DATA_HOME", "applications/x.desktop"),
])
def test_a_sibling_under_the_relocated_base_is_not_a_protected_path(tmp_path, monkeypatch, variable, sibling):
    moved = tmp_path / "moved"
    monkeypatch.setenv(variable, str(moved))

    result = _ask(tmp_path, moved / sibling, "auto")

    klass = result.request.klass if result.request is not None else None
    assert klass != "protected-path"


@pytest.mark.parametrize("relative", [rel for _var, rel in XDG_RELATIVES[:3]])
def test_the_home_forms_stay_protected_when_a_variable_is_set(tmp_path, monkeypatch, relative):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "moved"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "moved-data"))

    result = _ask(tmp_path, Path.home() / ".config" / relative, "auto")

    assert result.verdict is Verdict.ASK and result.request.klass == "protected-path"


@pytest.mark.parametrize("value", ["", "relative/dir", "~/elsewhere"])
def test_an_empty_or_relative_variable_is_ignored_as_the_xdg_specification_says(tmp_path, monkeypatch, value):
    monkeypatch.setenv("XDG_CONFIG_HOME", value)
    target = tmp_path / "proj" / "relative" / "dir" / "systemd" / "user" / "x.service"

    result = _ask(tmp_path, target, "auto")

    klass = result.request.klass if result.request is not None else None
    assert klass != "protected-path"


def test_the_relocation_follows_the_one_list_a_layer_can_tighten(tmp_path, monkeypatch):
    # no second table: an entry added to `protected_paths_home` under `~/.config/` is followed under the variable
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "moved"))
    settings = GateSettings(protected_paths_home=(*PROTECTED_PATHS_HOME_DEFAULT, "~/.config/extra-login"))

    result = evaluate("write", {"path": str(tmp_path / "moved" / "extra-login" / "x"), "content": "x"}, WRITE,
                      _ctx(tmp_path, "auto"), settings)

    assert result.verdict is Verdict.ASK and result.request.klass == "protected-path"


def test_the_harness_own_unit_path_is_inside_the_protected_set_under_a_redirected_variable(tmp_path, monkeypatch):
    from localharness.provider import units

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "moved"))

    result = _ask(tmp_path, units.harness_unit_path(), "auto")

    assert result.verdict is Verdict.ASK and result.request.klass == "protected-path"
