"""The shared text limits and the one-line notice formatter."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from localharness.channels.base import notice
from localharness.core import caps as caps_module
from localharness.core.caps import Caps

RLO, ZWSP = chr(0x202E), chr(0x200B)


def test_caps_is_one_immutable_tuple_of_the_six_limits():
    assert Caps == (80, 240, 4000, 8000, 8, 6)
    assert Caps._fields == ("LINE", "FINDING", "BLOCK", "DOC", "TOP", "QUESTIONS")
    with pytest.raises(AttributeError):
        Caps.LINE = 1  # type: ignore[misc]


def test_caps_module_imports_nothing_from_localharness():
    tree = ast.parse(Path(caps_module.__file__).read_text())
    named = [
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [n])
    ]
    assert not any(name and name.startswith("localharness") for name in named)


def test_the_template_renders_head_commands_and_tail():
    assert (
        notice("t3", "plan panel running", ["/task stop (after this unit)"], "up to 13 min")
        == "◆ t3 plan panel running · /task stop (after this unit) · up to 13 min"
    )


def test_a_hostile_notice_renders_clean_with_exactly_its_own_mark():
    out = notice(f"◆ t3{RLO}{ZWSP}\x1b[2Jx\nfake", "running", ["go"], "")
    assert out.startswith("◆ ") and out.count("◆") == 1
    assert not any(c in out for c in (RLO, ZWSP, "\x1b", "\n"))
    assert "◇ t3" in out
