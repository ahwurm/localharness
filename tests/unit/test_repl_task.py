"""The shared text limits and the one-line notice formatter."""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest

from localharness.agent.subagent import DelegationResult
from localharness.channels.base import notice
from localharness.core import caps as caps_module
from localharness.core.caps import Caps
from localharness.core.types import human_message
from localharness.orchestrator.task_flow import runner, verbs
from localharness.orchestrator.task_flow.filer import digest
from localharness.orchestrator.task_flow.panel import FlowCap
from localharness.orchestrator.task_flow.record import LINES, Row, Unit, file_doc, need, set_verdict
from tests.unit.test_task_flow_record import BLOCKER_LINE, PLAN_TWO, TYPED, make_flow, planned

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


async def test_tracer_the_cap_clause_counts_the_units_own_reservation_as_unspent(tmp_path):
    flow, *_ = await planned(tmp_path, cap=10.0)
    flow.end_turn()
    lines = sinking(flow, PHONE)
    await flow.run("step")
    assert lines == ["◆ t1 step 1/2 executing · stop (now) · typing waits · cap 10 min · up to 10 min"]


LEVERS = {word: (word, "Ctrl+C (now)" if word.startswith("/task") else word) for word in STOP_WORDS}
FIELDS = {"verb": "run", "stage": "plan", "verdict": "pass", "doc": "plan", "n": 2, "bound": 13,
          "next": "/task run (next step)", "why": "revisions used up", "unit": "step 2", "step": (2, 5)}
PLACEHOLDER = re.compile(r"<(stop|cancel|verb|stage|verdict|doc|n|bound|next|why|unit)>")
MODEL_ONLY = ("person decides", "decisions unfiled", "research full", "nothing to review")


def FIRST(command):
    return command.split(" (")[0]


@pytest.mark.parametrize("task", ("t3", "t10"))
@pytest.mark.parametrize("stop_word", STOP_WORDS)
@pytest.mark.parametrize("key", list(LINES))
def test_every_row_renders_through_notice_with_each_stop_word(key, stop_word, task):
    out = verbs.render(key, LEVERS[stop_word], task=task, **FIELDS)
    if key in MODEL_ONLY:
        assert out == ""
        return
    assert out.startswith(f"◆ {task} ") and out.count("◆") == 1
    assert not PLACEHOLDER.search(out)
    assert len(out) <= Caps.LINE or " · " not in out
    assert ("files half-edited" in out) == (key == "step stopped")
    if "<stop>" in LINES[key].commands:
        assert FIRST(stop_word) in out
    if "<cancel>" in LINES[key].commands:
        assert FIRST(LEVERS[stop_word][1]) in out


@pytest.mark.parametrize("stop_word", STOP_WORDS)
@pytest.mark.parametrize("key", [k for k in LINES if k not in MODEL_ONLY and LINES[k].commands])
def test_cut_at_t1000_and_step_10_of_12_keeps_the_first_verb(key, stop_word):
    out = verbs.render(key, LEVERS[stop_word], task="t1000", **{**FIELDS, "step": (10, 12)})
    first = need(key, task="t1000", stop=LEVERS[stop_word][0], cancel=LEVERS[stop_word][1], **FIELDS).commands[0]
    assert FIRST(first) in out


def test_model_only_rows_are_the_four_with_a_rendering_and_no_commands():
    assert {k for k, line in LINES.items() if line.model and not line.commands} == set(MODEL_ONLY)
    assert all(verbs.render(k, LEVERS[STOP_WORDS[0]], task="t3", **FIELDS) == "" for k in MODEL_ONLY)


def test_command_order_is_run_then_show_then_accept_and_the_first_carries_task():
    seen = set()
    for key in LINES:
        parts = verbs.render(key, LEVERS[STOP_WORDS[0]], task="t3", **FIELDS).split(" · ")[1:]
        words = [p.removeprefix("/task ").split()[0] for p in parts]
        if {"run", "show", "accept"} <= set(words):
            seen.add(key)
            order = [words.index(w) for w in ("run", "show", "accept")]
            assert order == sorted(order) and parts[0].startswith("/task")
    assert {"step blocked", "blocked"} <= seen


def test_examples_render_exactly_at_80_columns_through_render():
    box = LEVERS[STOP_WORDS[0]]
    assert verbs.render("running", box, task="t3", unit="plan panel", bound=13) == \
        "◆ t3 plan panel running · /task stop (after this unit) · up to 13 min"
    assert verbs.render("step blocked", box, task="t3", step=(2, 5), n=2) == \
        "◆ t3 step 2/5 blocked · /task run (retries) · show · accept · 2 blockers"
    assert verbs.render("review blocked", box, task="t3", n=1) == \
        "◆ t3 review blocked · /task plan · show review · accept (ends t3) · 1 blocker"
    assert verbs.render("plan unreviewed", box, task="t3", bound=13) == \
        "◆ t3 plan unreviewed · /task plan (re-panels) · up to 13 min"


async def test_example_a_second_running_row_on_the_phone_carries_only_the_bound(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.said |= {"typing", "cap"}
    assert verbs.progress(flow, "t3", "step 2/5 executing", ("<stop>",), "up to 25 min", PHONE) == \
        "◆ t3 step 2/5 executing · stop (now) · up to 25 min"


def test_a_row_without_a_stop_passes_through_progress_unchanged(tmp_path):
    flow, *_ = make_flow(tmp_path)
    assert verbs.progress(flow, "t3", "plan reopened", ("<cancel>",), "", PHONE) == "◆ t3 plan reopened · stop (now)"
    assert flow.said == set()


def test_an_armed_row_says_the_cap_so_the_next_running_row_does_not(tmp_path):
    flow, *_ = make_flow(tmp_path)
    assert verbs.progress(flow, "t3", "plan armed", ("<stop>",), "cap 180 min", PHONE) == "◆ t3 plan armed · stop (now) · cap 180 min"
    assert verbs.progress(flow, "t3", "plan panel running", ("<stop>",), "up to 13 min", PHONE) == \
        "◆ t3 plan panel running · stop (now) · typing waits · up to 13 min"


def test_bound_at_the_defaults_is_the_units_computed_from_the_table(tmp_path):
    flow, *_ = make_flow(tmp_path)
    units = {Unit("new", "panel"): 1, Unit("plan", "panel"): 13, Unit("research", "explore"): 6, Unit("research", "panel"): 7,
             Unit("step", "execute", 1): 25, Unit("step", "panel", 1): 6, Unit("task", "panel"): 7}
    assert {u: verbs.bound(flow, u) for u in units} == units
    assert verbs.bound(flow, Unit("reply", "panel"), ("factual", "recency")) == 28


@pytest.mark.parametrize(("seconds", "printed"), [(1500, 25), (1499, 24), (600, 10), (1501, 25), (59, 0)])
def test_bound_of_an_executor_against_a_remainder_is_floored_to_whole_minutes(tmp_path, seconds, printed):
    flow, *_ = make_flow(tmp_path)
    flow.cap = FlowCap(seconds / 60)
    assert verbs.bound(flow, Unit("step", "execute", 1)) == printed


def test_the_cap_clause_is_the_limits_table_less_the_seconds_spent(tmp_path):
    flow, *_ = make_flow(tmp_path)
    assert verbs.cap_minutes(flow) == 180
    held = flow.cap.reserve(flow.limits.executor, 1)
    assert verbs.cap_minutes(flow) == 180
    flow.cap.settle(held, 61)
    assert verbs.cap_minutes(flow) == 178


def test_a_plan_pass_adds_the_revision_command_the_unreviewed_decisions_and_the_reopened_steps():
    box = LEVERS[STOP_WORDS[0]]
    fields = {"task": "t3", "stage": "plan", "verdict": "pass", "next": "/task run (step 1)", "bound": 6}
    assert verbs.render("pass", box, **fields) == notice(
        "t3", "plan pass", ["/task run (step 1)", "plan <change> (revises)"], "up to 6 min")
    out = verbs.render("pass", box, **fields, reopened=(3, 4, 5), decisions_unreviewed=True)
    assert out == notice("t3", "plan pass, steps 3-5 reopened",
                         ["/task run (step 1)", "plan <change> (revises)", "accept decisions (unreviewed)"], "up to 6 min")


@pytest.mark.parametrize(("reopened", "said"), [((2,), "step 2"), ((2, 3), "steps 2-3"), ((2, 4), "steps 2, 4"), ((1, 2, 3, 5), "steps 1-3, 5")])
def test_the_reopened_steps_read_as_ranges(reopened, said):
    out = verbs.render("pass", LEVERS[STOP_WORDS[0]], task="t3", stage="plan", verdict="accepted", next="/task run", bound=6, reopened=reopened)
    assert f"plan accepted, {said} reopened" in out


def test_a_stage_other_than_plan_gets_no_plan_extras():
    out = verbs.render("pass", LEVERS[STOP_WORDS[0]], task="t3", stage="new", verdict="pass", next="/task plan", bound=1,
                       reopened=(2,), decisions_unreviewed=True)
    assert out == notice("t3", "new pass", ["/task plan"], "up to 1 min")


PLAN_UNFILED = ('Call task(stage="plan") to read the decisions and the plan\'s shape, '
                'then call it again with doc= your plan.')
DECISIONS = ("decisions.md", None)


def command(flow, args, turn=2, levers=BOX):
    return verbs.task_command(flow, args, turn=turn, running=False, levers=levers)


def asked(*turns):
    return [human_message(text, subtype, n) for n, subtype, text in turns]


def test_task_command_new_opens_the_task_and_the_objective_is_the_typed_turn(tmp_path):
    flow, *_ = make_flow(tmp_path)
    reply = command(flow, "new fix the Login bug")
    assert (flow.record.id, flow.record.objective, flow.record.turn) == (1, "fix the Login bug", 2)
    assert reply.lines == ("◆ t1 interview open · /task plan (skips the questions)",)
    assert reply.turn == ("fix the Login bug", "typed")
    assert flow.armed is None and flow.said == set()


def test_task_command_new_with_only_whitespace_is_bare_new_and_with_no_task_says_so(tmp_path):
    flow, *_ = make_flow(tmp_path)
    reply = command(flow, "new   ")
    assert reply == verbs.Reply(("◆ task no task · /task new <objective>",))
    assert flow.record is None


def test_task_command_matches_the_verb_lower_cased_and_keeps_the_objective_as_typed(tmp_path):
    flow, *_ = make_flow(tmp_path)
    reply = command(flow, "NeW   Fix İt ΣΑΣ  now ")
    assert reply.turn == ("Fix İt ΣΑΣ  now", "typed") and flow.record.objective == "Fix İt ΣΑΣ  now"


def test_task_command_plan_files_the_skip_from_the_typed_turns_and_arms_the_plan(tmp_path):
    messages = asked((2, "typed", "fix the Login bug"), (3, "typed", "use the existing session table"),
                     (3, "steering", "and keep the cookie name"))
    flow, *_ = make_flow(tmp_path, messages=messages)
    flow.turn = lambda: 4
    command(flow, "new fix the Login bug")
    reply = command(flow, "plan", turn=4)
    text = "fix the Login bug (turn 2)\nturn 3: use the existing session table\nturn 3: and keep the cookie name"
    assert flow.docs.read(1, "decisions.md") == text
    row = flow.record.filed[DECISIONS]
    assert (row.verdict, row.digest, flow.record.filed_turn["new"]) == ("unreviewed", digest(text), 4)
    assert reply.lines == ("◆ t1 decisions filed · from 3 turns", "◆ t1 plan armed · /task stop (after this unit) · cap 180 min")
    assert reply.turn == (PLAN_UNFILED, "request")
    assert flow.armed == "plan" and "cap" in flow.said and flow.record.paused is False


def test_task_command_skip_counts_only_the_objective_when_nothing_else_was_typed(tmp_path):
    flow, *_ = make_flow(tmp_path, messages=asked((2, "typed", "fix login"), (1, "typed", "before the task")))
    command(flow, "new fix login")
    reply = command(flow, "plan", turn=3)
    assert flow.docs.read(1, "decisions.md") == "fix login (turn 2)"
    assert reply.lines[0] == "◆ t1 decisions filed · from 1 turn"


def test_task_command_armed_plan_with_decisions_filed_prints_no_skip_line_and_clears_the_pause(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.open("fix login", 1)
    flow.docs.write(1, "decisions.md", "d")
    file_doc(flow.record, "new", DECISIONS, digest("d"))
    flow.record.paused = True
    reply = command(flow, "plan", levers=PHONE)
    assert reply.lines == ("◆ t1 plan armed · stop (now) · cap 180 min",)
    assert flow.docs.read(1, "decisions.md") == "d" and flow.record.paused is False


@pytest.mark.parametrize(
    ("keys", "newest"),
    [((), None), (("decisions.md",), "decisions"), (("decisions.md", "research.md"), "research"),
     (("decisions.md", "plan.md"), "plan"), (("plan.md", "step-2.md", "step-10.md"), "step-10"),
     (("plan.md", "step-1.md", "review.md"), "review")])
def test_task_command_new_closes_the_open_task_naming_its_newest_doc_and_opens_one_past_the_highest(tmp_path, keys, newest):
    flow, *_ = make_flow(tmp_path)
    flow.open("a", 1), flow.open("b", 1), flow.open("c", 1)
    for key in keys:
        flow.docs.write(3, key, "x")
        flow.record.filed[(key, None)] = Row(digest("x"))
    reply = command(flow, "new next one")
    shown = f" · /task show t3/{newest}" * bool(newest)
    assert reply.lines == (f"◆ t3 closed{shown} · not resumable, folder kept",
                           "◆ t4 interview open · /task plan (skips the questions)")
    assert flow.record.id == 4 and flow.docs.path(3, "").is_dir()
    assert reply.turn == ("next one", "typed")


def test_task_command_is_synchronous():
    assert not asyncio.iscoroutinefunction(verbs.task_command)


SENTENCE_FOR = {
    "research": ('Call task(stage="research", question=...) once per question the plan needs answered from the '
                 'repository, then task(stage="research") to review the answers.'),
    "new blocked": 'Call task(stage="new") to read the findings, then call it again with doc= your revised decisions.',
    "new unreviewed": 'Call task(stage="new").',
    "new reopened": ('Call task(stage="new") to read the decisions, then call it again with doc= the decisions '
                     "revised as the person's messages since they were filed ask."),
    "plan unfiled": PLAN_UNFILED,
    "plan unreviewed": 'Call task(stage="plan").',
    "plan blocked": 'Call task(stage="plan") to read the findings, then call it again with doc= your revised plan.',
    "plan reopened": ('Call task(stage="plan") to read the plan, then call it again with doc= the plan revised '
                      "as the person's messages since it was filed ask."),
    "run": 'Call task(stage="run") and call it again after each result until it says done, blocked or halted.',
}
ARMED = "◆ t1 {} armed · /task stop (after this unit) · cap 180 min"
PLAN_ROW = ("plan.md", None)


def staged(tmp_path, decisions=None, plan=None, messages=TYPED, stage=None, filed=3, blockers=2):
    flow, *_ = make_flow(tmp_path, messages=messages)
    flow.open("fix login", 1)
    for boundary, key, verdict in (("new", DECISIONS, decisions), ("plan", PLAN_ROW, plan)):
        if verdict:
            file_doc(flow.record, boundary, key, digest(key[0]))
            set_verdict(flow.record, key, verdict)
            flow.record.filed_turn[boundary] = filed
            flow.record.counts[boundary].blockers, flow.record.counts[boundary].used = blockers, 3
            flow.record.counts[boundary].previous = blockers + 1
    flow.record.stage = stage or ("plan" if plan else "new")
    return flow


def test_task_command_sentence_is_chosen_from_the_record_and_is_the_specs_own(tmp_path):
    flow = staged(tmp_path)
    record = flow.record
    assert verbs.sentence("research", record) == SENTENCE_FOR["research"]
    assert verbs.sentence("plan", record) == SENTENCE_FOR["plan unfiled"]
    for verdict, key in (("blocked", "blocked"), ("unreviewed", "unreviewed"), ("pass", "reopened"), ("accepted", "reopened")):
        for boundary, row in (("new", DECISIONS), ("plan", PLAN_ROW)):
            record.filed[row] = Row("d", verdict)
            assert verbs.sentence(boundary, record) == SENTENCE_FOR[f"{boundary} {key}"]
    assert verbs.sentence("step", record) == SENTENCE_FOR["run"]


def test_task_command_reopen_of_a_passed_plan_with_a_typed_turn_since_the_filing_sets_the_flag(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="pass", messages=asked((2, "typed", "fix login"), (5, "typed", "smaller steps")))
    reply = command(flow, "plan", turn=6)
    assert flow.record.flags == {"new": False, "plan": True}
    assert reply.lines == ("◆ t1 plan reopened · Ctrl+C (now)", ARMED.format("plan"))
    assert reply.turn == (SENTENCE_FOR["plan reopened"], "request") and flow.sentence == ""
    assert flow.armed == "plan" and flow.said == {"cap"}


@pytest.mark.parametrize("verdict", ("pass", "accepted"))
def test_task_command_nothing_to_revise_arms_nothing_when_no_typed_turn_came_after_the_filing(tmp_path, verdict):
    flow = staged(tmp_path, decisions="pass", plan=verdict, messages=asked((2, "typed", "fix login"), (3, "typed", "at the filing")))
    reply = command(flow, "plan", turn=6)
    assert reply == verbs.Reply(("◆ t1 nothing to revise · /task plan <change>",))
    assert flow.record.flags["plan"] is False and flow.armed is None and flow.said == set()


@pytest.mark.parametrize("verdict", ("pass", "accepted", "unreviewed", "blocked"))
def test_task_command_plan_change_is_the_typed_turn_and_the_sentence_rides_the_note(tmp_path, verdict):
    flow = staged(tmp_path, decisions="pass", plan=verdict)
    reply = command(flow, "plan Add a Retry step")
    assert reply.turn == ("Add a Retry step", "typed")
    expected = {"pass": "reopened", "accepted": "reopened", "unreviewed": "unreviewed", "blocked": "blocked"}[verdict]
    assert flow.sentence == SENTENCE_FOR[f"plan {expected}"]
    assert flow.record.flags["plan"] is (verdict != "blocked") and reply.lines[-1] == ARMED.format("plan")


@pytest.mark.parametrize("stage", ("plan", "task"))
def test_task_command_reopen_of_a_blocked_plan_or_a_blocked_task_review_resets_the_count_and_keeps_the_blockers(tmp_path, stage):
    flow = staged(tmp_path, decisions="pass", plan="blocked", stage=stage)
    reply = command(flow, "plan")
    counts = flow.record.counts["plan"]
    assert (counts.used, counts.previous, counts.blockers, counts.revised) == (0, None, 2, False)
    assert flow.record.flags["plan"] is False
    assert reply.lines == (ARMED.format("plan"),) and reply.turn == (SENTENCE_FOR["plan blocked"], "request")


def test_task_command_reopen_of_an_unreviewed_plan_sets_the_flag_and_names_the_panel_call(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="unreviewed")
    reply = command(flow, "plan")
    assert flow.record.flags["plan"] is True and reply.turn == ('Call task(stage="plan").', "request")
    assert reply.lines == ("◆ t1 plan reopened · Ctrl+C (now)", ARMED.format("plan"))


def test_task_command_bare_new_reopens_the_decisions_by_their_row(tmp_path):
    flow = staged(tmp_path)
    reply = command(flow, "new")
    assert reply.lines == ("◆ t1 interview open · /task plan (skips the questions)", ARMED.format("new"))
    assert reply.turn == (runner.INTERVIEW_SENTENCE, "request") and flow.record.flags["new"] is False
    for verdict in ("blocked", "unreviewed", "pass", "accepted"):
        flow = staged(tmp_path / verdict, decisions=verdict)
        reply = command(flow, "new")
        flagged = verdict != "blocked"
        assert flow.record.flags["new"] is flagged
        assert reply.lines == (("◆ t1 new reopened · Ctrl+C (now)",) * flagged + (ARMED.format("new"),))
        assert reply.turn[1] == "request" and (flow.record.counts["new"].used == 0) is (not flagged)


def test_task_command_bare_new_once_the_plan_is_filed_says_open_and_not_blocked(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="pass")
    reply = command(flow, "new")
    assert reply == verbs.Reply(("◆ t1 open and not blocked · /task plan (revises) · new <objective> (closes t1)",))
    assert flow.record.flags == {"new": False, "plan": False} and flow.armed is None


def test_task_command_research_arms_with_its_sentence_after_the_skip(tmp_path):
    flow = staged(tmp_path, messages=asked((1, "typed", "fix login")))
    reply = command(flow, "research", turn=2)
    assert flow.docs.read(1, "decisions.md") == "fix login (turn 1)"
    assert reply.lines == ("◆ t1 decisions filed · from 1 turn", ARMED.format("research"))
    assert reply.turn == (SENTENCE_FOR["research"], "request") and flow.armed == "research"


async def test_task_command_run_arms_the_step_boundary_with_the_run_sentence(tmp_path):
    flow, *_ = await planned(tmp_path)
    flow.end_turn()
    reply = command(flow, "run")
    assert reply.lines == (ARMED.format("run"),) and reply.turn == (SENTENCE_FOR["run"], "request")
    assert flow.armed == "step"


async def test_task_command_run_on_a_step_blocked_past_its_limit_prints_the_row_and_arms_nothing(tmp_path):
    flow, *_ = await planned(tmp_path)
    flow.end_turn()
    row = flow.record.filed[("plan.md", "step-1")]
    row.verdict, row.counts.blockers, row.counts.used = "blocked", 2, 1
    reply = command(flow, "run")
    assert reply == verbs.Reply(("◆ t1 step 1/2 blocked · /task run (retries) · show · accept · 2 blockers",))
    assert flow.armed is None and flow.said == set()


async def test_task_command_refile_flag_is_left_by_a_read_and_a_refused_doc_and_consumed_by_the_refile(tmp_path):
    flow, script, *_ = await planned(tmp_path)
    flow.end_turn()
    command(flow, "plan")
    assert flow.record.flags["plan"] is True
    await flow.run("plan", None, by="model")
    await flow.run("plan", "not a plan", by="model")
    assert flow.record.flags["plan"] is True
    panels = len(script.named("big-picture"))
    await flow.run("plan", PLAN_TWO, by="model")
    assert flow.record.flags["plan"] is False and len(script.named("big-picture")) == panels + 1


async def test_task_command_reopen_flag_is_cleared_when_the_turn_ends(tmp_path):
    flow, *_ = await planned(tmp_path)
    flow.end_turn()
    command(flow, "plan")
    flow.end_turn()
    assert flow.record.flags == {"new": False, "plan": False} and flow.armed is None


NO_TASK = verbs.Reply(("◆ task no task · /task new <objective>",))
DONE = verbs.Reply(("◆ t1 done · /task show review · new <objective>",))
UNSET = {"new": False, "plan": False}


@pytest.mark.parametrize("args", ("research", "plan", "run", "new", "plan change it", "run now", "research why"))
def test_task_command_no_task_every_verb_but_new_objective_prints_the_no_task_line(tmp_path, args):
    flow, *_ = make_flow(tmp_path)
    assert command(flow, args) == NO_TASK
    assert flow.record is None and flow.armed is None and flow.said == set()


@pytest.mark.parametrize("args", ("research", "plan", "run", "new", "plan change it", "run now"))
def test_task_command_done_every_verb_but_new_objective_prints_the_done_line(tmp_path, args):
    flow = staged(tmp_path, decisions="pass", plan="accepted", stage="done")
    assert command(flow, args) == DONE
    assert flow.armed is None and flow.said == set() and flow.record.flags == UNSET and flow.sentence == ""


def test_task_command_done_new_objective_opens_the_next_task_without_a_closed_line(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="accepted", stage="done")
    reply = command(flow, "new the next job")
    assert reply.lines == ("◆ t2 interview open · /task plan (skips the questions)",)
    assert reply.turn == ("the next job", "typed")


def test_task_command_need_decisions_blocked_refuses_plan_and_research(tmp_path):
    flow = staged(tmp_path, decisions="blocked")
    line = verbs.Reply(("◆ t1 decisions blocked · /task new (reopens) · accept",))
    assert command(flow, "plan") == line and command(flow, "research") == line
    assert flow.armed is None and flow.record.counts["new"].used == 3


def test_task_command_need_research_closed_points_at_the_research_or_else_the_plan(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="pass")
    flow.record.filed[("step-1.md", None)] = Row("x")
    assert command(flow, "research") == verbs.Reply(("◆ t1 research closed · /task show plan · new <objective> (closes t1)",))
    flow.record.filed[("research.md", None)] = Row("r")
    assert command(flow, "research") == verbs.Reply(("◆ t1 research closed · /task show research · new <objective> (closes t1)",))
    assert flow.armed is None


def test_task_command_need_run_without_a_plan_says_what_it_needs(tmp_path):
    flow = staged(tmp_path, decisions="pass")
    assert command(flow, "run") == verbs.Reply(("◆ t1 run needs a plan · /task plan",))
    assert flow.armed is None and flow.said == set()


@pytest.mark.parametrize(("stage", "blockers", "line"), [
    ("plan", 2, "◆ t1 plan blocked · /task plan (reopens) · show plan · accept · 2 blockers"),
    ("plan", 1, "◆ t1 plan blocked · /task plan (reopens) · show plan · accept · 1 blocker"),
    ("task", 1, "◆ t1 review blocked · /task plan · show review · accept (ends t1) · 1 blocker")])
def test_task_command_blocked_plan_refuses_run_with_its_row_and_the_count(tmp_path, stage, blockers, line):
    flow = staged(tmp_path, decisions="pass", plan="blocked", stage=stage, blockers=blockers)
    assert command(flow, "run") == verbs.Reply((line,))
    assert flow.armed is None and flow.record.flags == UNSET and flow.record.counts["plan"].used == 3


def test_task_command_stale_plan_names_the_file_and_the_accept_name(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="pass")
    assert command(flow, "run") == verbs.Reply(("◆ t1 plan.md edited since filed · /task accept plan (uses your edit)",))
    assert flow.armed is None and flow.said == set() and flow.record.flags == UNSET


def test_task_command_stale_doc_with_a_name_accept_cannot_take_suggests_no_accept(tmp_path):
    flow = staged(tmp_path, decisions="pass", plan="pass")
    line = verbs._refusal(flow.record, need("edited since filed", doc="odd name.md"), BOX)
    assert line == "◆ t1 odd name.md edited since filed"


def test_task_command_a_refused_folder_write_is_a_line_not_a_crash(tmp_path):
    flow = staged(tmp_path)
    flow.docs.adopted.clear()
    reply = command(flow, "plan")
    assert reply.turn is None and reply.lines == ("◆ t1 t1 was not made by this session; it is read-only",)
    assert flow.armed is None


def test_task_command_an_unknown_verb_and_a_bare_task_print_the_usage_line(tmp_path):
    flow = staged(tmp_path)
    for args in ("", "  ", "frobnicate", "show", "accept plan", "stop", "clear"):
        assert command(flow, args) == verbs.Reply((verbs.USAGE,))
    assert all(word in verbs.USAGE for word in ("new", "research", "plan", "run", "show", "accept", "stop", "clear"))
    assert flow.armed is None and flow.record.flags == UNSET
