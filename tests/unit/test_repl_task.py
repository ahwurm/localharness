"""The shared text limits and the one-line notice formatter."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from localharness.agent.subagent import DelegationResult
from localharness.channels.base import notice
from localharness.core import caps as caps_module
from localharness.core.caps import Caps
from localharness.orchestrator.task_flow import verbs
from tests.unit.test_task_flow_record import BLOCKER_LINE, PLAN_TWO, make_flow, planned

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


PHONE = ("stop (now)", "stop (now)")
BOX = ("/task stop (after this unit)", "Ctrl+C (now)")


def sinking(flow, levers):
    lines = []
    flow.emit = lambda *row: lines.append(verbs.progress(flow, *row, levers=levers))
    return lines


async def test_tracer_the_turns_first_running_row_says_typing_waits_and_the_cap_once(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.open("fix login", 1)
    lines = sinking(flow, PHONE)
    await flow.run("new", "Objective: fix login", by="model")
    await flow.run("plan", PLAN_TWO, by="model")
    assert lines == ["◆ t1 new panel running · stop (now) · typing waits · cap 180 min · up to 1 min",
                     "◆ t1 plan panel running · stop (now) · up to 13 min"]
    flow.end_turn()
    await flow.run("step", by="model")
    assert lines[2] == "◆ t1 step 1/2 executing · stop (now) · typing waits · cap 180 min · up to 25 min"


async def test_tracer_the_box_line_is_cut_from_the_right_to_the_line_cap(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.open("fix login", 1)
    lines = sinking(flow, BOX)
    await flow.run("plan", PLAN_TWO, by="person")
    assert lines == ["◆ t1 plan panel running · /task stop (after this unit) · typing waits"]


async def test_tracer_each_units_running_state_names_the_unit_as_the_spec_does(tmp_path):
    flow, _, rows, _ = make_flow(tmp_path, {"spec-compliance": [BLOCKER_LINE, "NONE"], "executor": "done", "explorer": "a claim"})
    flow.open("fix login", 1)
    await flow.run("new", "Objective: fix login")
    await flow.run("research", "where is login")
    await flow.run("research")
    await flow.run("plan", PLAN_TWO)
    for _ in range(6):
        await flow.run("step")
    await flow.run("task")
    assert [row[1] for row in rows] == [
        "new panel running", "research q1 running", "research panel running", "plan panel running",
        "step 1/2 executing", "step 1/2 checking", "step 1/2 executing", "step 1/2 checking",
        "step 2/2 executing", "step 2/2 checking", "task panel running"]
    assert {row[2] for row in rows} == {("<stop>",)}


async def test_tracer_busy_runs_from_the_row_to_the_return_and_a_cancel_leaves_it_with_the_unit(tmp_path):
    seen, started = [], asyncio.Event()

    async def hang(config, task):
        seen.append(flow.busy)
        started.set()
        await asyncio.sleep(3600)

    flow, *_ = await planned(tmp_path, {"executor": hang})
    assert flow.busy is False
    run = asyncio.create_task(flow.run("step"))
    await started.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert seen == [True] and flow.busy is True and flow.unit == ("step 1/2 executing", 25)
    flow.end_turn()
    assert flow.busy is False and flow.unit is None
    await flow.run("plan")
    assert flow.busy is False


async def test_tracer_called_holds_the_boundaries_the_model_called_and_not_the_persons(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.open("fix login", 1)
    await flow.run("new", "Objective: fix login", by="person")
    assert flow.called == set()
    await flow.run("plan", PLAN_TWO, by="model")
    await flow.run("step", by="model")
    assert flow.called == {"plan", "step"}
