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


RUN = "/task run (retries)"
STOP_WORDS = ("/task stop (after this unit)", "Ctrl+C (now)", "stop (now)", "cancel (now)")


@pytest.mark.parametrize(
    ("subject", "state", "commands", "tail", "expected"),
    [
        (
            "t3", "step 2/5 blocked", [RUN, "show", "accept"], "2 blockers",
            "◆ t3 step 2/5 blocked · /task run (retries) · show · accept · 2 blockers",
        ),
        (
            "t1000", "step 10/12 blocked", [RUN, "show review", "accept (ends t1000)"],
            "2 blockers · cap 170 min",
            "◆ t1000 step 10/12 blocked · /task run (retries) · 2 blockers · cap 170 min",
        ),
        (
            "t10", "step 10/12 executing", [STOP_WORDS[0]],
            "typing waits · cap 170 min · up to 25 min",
            "◆ t10 step 10/12 executing · /task stop (after this unit) · typing waits",
        ),
        (
            "t1000", "the plan names a step the plan does not have", [STOP_WORDS[0]], "up to 6 min",
            "◆ t1000 the plan names a step the plan does not have · /task stop",
        ),
    ],
    ids=["uncut", "cut_commands_first", "cut_tail_clauses_from_the_right", "cut_parenthetical_last"],
)
def test_cut_order(subject, state, commands, tail, expected):
    assert notice(subject, state, commands, tail) == expected


def test_exact_cap_is_uncut_and_one_over_with_nothing_to_cut_is_whole():
    at = notice("t3", "a" * 75, [])
    assert len(at) == Caps.LINE and at.endswith("a" * 75)
    over = notice("t3", "a" * 76, [])
    assert len(over) == Caps.LINE + 1


def test_exact_cap_with_a_tail_is_uncut_and_one_more_cuts_it():
    head = "◆ t3 " + "a" * 40
    tail = "b" * (Caps.LINE - len(head) - 3)
    assert len(notice("t3", "a" * 40, [], tail)) == Caps.LINE
    assert notice("t3", "a" * 40, [], tail + "b") == head


def test_empty_commands_and_tail_have_no_trailing_separator():
    assert notice("t3", "plan unreviewed", [], "") == "◆ t3 plan unreviewed"
    assert notice("t3", "plan unreviewed", ["", "  "], " \n ") == "◆ t3 plan unreviewed"


def test_a_field_empty_after_one_line_is_omitted():
    assert notice("t3", ZWSP, ["go"], "") == "◆ t3 · go"
    assert notice("", "idle", [], "") == "◆ idle"


def test_length_counts_code_points_and_kept_order_is_the_order_given():
    out = notice("t3", "step 2/5", ["b", "a"], "z · y")
    assert out == "◆ t3 step 2/5 · b · a · z · y" and len(out) == len(out.encode().decode())
    long = notice("t1000", "x" * 30, ["first", "second " + "w" * 40], "p · q · r")
    assert long == "◆ t1000 " + "x" * 30 + " · first · p · q · r"


@pytest.mark.parametrize("stop_word", STOP_WORDS)
@pytest.mark.parametrize(
    ("subject", "state"), [("t10", "step 10/12 executing"), ("t1000", "step 10/12 executing")]
)
def test_stop_word_survives_the_cut(stop_word, subject, state):
    head = f"◆ {subject} {state}"
    out = notice(subject, state, [stop_word, "show review"], "typing waits · cap 170 min · up to 25 min")
    assert len(out) <= Caps.LINE and out.startswith(head)
    assert stop_word.split(" (")[0] in out
