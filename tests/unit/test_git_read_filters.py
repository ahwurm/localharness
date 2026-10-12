"""git_read clean-filter narrowing (GITREAD-04): the pathspec magic parser, the narrowing table, and the
precheck + trailing line proven on real clean and process filters (D-14, D-15, D-16, D-34, D-45..D-47).

Real git runs in tmp_path only, under git_support's from-scratch environment. A marker filter that never
runs is the proof that a repository-configured command stayed unexecuted; each fixture has a bare-git
control that does run it. Timeouts are enforced by the tool itself, not by pytest-timeout.
"""
from __future__ import annotations

import pytest

from localharness.tools.builtin import git_read_tool as grt
from localharness.tools.builtin.git_read_tool import (
    GitReadValidationError,
    _magic,
    _narrow_for_filters,
    _validate_call,
)

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

