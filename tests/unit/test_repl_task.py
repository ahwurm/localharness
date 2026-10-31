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
from localharness.orchestrator.task_flow.filer import PLAN_SHAPE, digest, parse_plan, render_section, rerecord
from localharness.orchestrator.task_flow.panel import Finding, FlowCap
from localharness.orchestrator.task_flow.record import LINES, Row, Unit, file_doc, need, set_verdict, stop_step
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
    line = verbs._refusal(flow.record, need("edited since filed", doc="odd name.md"))
    assert line == "◆ t1 odd name.md edited since filed"


def test_task_command_a_refused_folder_write_is_a_line_not_a_crash(tmp_path):
    flow = staged(tmp_path)
    flow.docs.adopted.clear()
    reply = command(flow, "plan")
    assert reply.turn is None and reply.lines == ("◆ t1 t1 was not made by this session; it is read-only",)
    assert flow.armed is None


def test_task_command_an_unknown_verb_prints_the_usage_line(tmp_path):
    flow = staged(tmp_path)
    for args in ("frobnicate", "stop", "show-plan", "accepting"):
        assert command(flow, args) == verbs.Reply((verbs.USAGE,))
    assert all(word in verbs.USAGE for word in ("new", "research", "plan", "run", "show", "accept", "stop", "clear"))
    assert flow.armed is None and flow.record.flags == UNSET


PLAN_FILED = ("plan.md", None)
FINDING = Finding("BLOCKER", "step 2 is vague", "", False, "big-picture")
FOREIGN = "not adopted · not made by this session"


def shelf(tmp_path, verdict="blocked", blockers=2, stage="plan"):
    """A flow with decisions and a two-step plan filed, the plan's review doc holding one round."""
    flow, *_ = make_flow(tmp_path)
    flow.open("fix login", 1)
    record, docs = flow.record, flow.docs
    file_doc(record, "new", DECISIONS, docs.write(1, "decisions.md", "turn 1")[0])
    file_doc(record, "plan", PLAN_FILED, docs.write(1, "plan.md", PLAN_TWO)[0])
    rerecord(record, parse_plan(PLAN_TWO))
    set_verdict(record, DECISIONS, "pass")
    set_verdict(record, PLAN_FILED, verdict)
    record.counts["plan"].blockers, record.stage = blockers, stage
    section = render_section(1, [FINDING], verdict, blockers)
    record.filed[("plan-review.md", None)] = Row(docs.append_section(1, "plan-review.md", section))
    return flow


def executed(flow, n=1, verdict="blocked", blockers=1):
    record, docs = flow.record, flow.docs
    record.stage = "run"
    record.filed[(f"step-{n}.md", None)] = Row(docs.write(1, f"step-{n}.md", f"step {n} log")[0])
    record.filed[("plan.md", f"step-{n}")].verdict = verdict
    record.filed[("plan.md", f"step-{n}")].counts.blockers = blockers
    section = render_section(1, [FINDING], verdict, blockers)
    record.filed[(f"step-{n}-review.md", None)] = Row(docs.append_section(1, f"step-{n}-review.md", section))


def show(flow, args="", phone=False):
    return verbs.task_command(flow, f"show {args}".rstrip(), turn=2, running=False, levers=BOX, phone=phone).lines


def foreign_folder(flow, n=9, **docs):
    folder = flow.docs.state_dir / "agents" / "a" / "tasks" / f"t{n}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (docs or {"plan.md": "their plan"}).items():
        (folder / name.replace("_md", ".md").replace("_", "-")).write_text(text)
    return folder


def test_show_a_blocked_plan_on_the_terminal_is_the_header_the_doc_then_the_last_section_of_its_review(tmp_path):
    flow = shelf(tmp_path)
    flow.docs.append_section(1, "plan-review.md", render_section(2, [FINDING], "blocked", 2))
    flow.record.filed[("plan-review.md", None)] = Row(digest(flow.docs.read(1, "plan-review.md")))
    out = show(flow, "plan")
    assert out[0] == "plan.md · blocked, 2 blockers" and out[1] == PLAN_TWO
    assert out[2].startswith("## round 2 · blockers 2") and "round 1" not in out[2] and len(out) == 3


def test_show_a_blocked_plan_on_the_phone_is_one_notice_header_findings_then_the_doc(tmp_path):
    flow = shelf(tmp_path)
    (out,) = show(flow, "plan", phone=True)
    head, rest = out.split("\n", 1)
    assert head == "plan.md · blocked, 2 blockers" and rest.endswith("\n" + PLAN_TWO)
    assert out.index("## round 1") < out.index("## Step 1") and "verdict: blocked" in out


def test_show_cuts_a_doc_on_the_phone_at_the_doc_cap_with_the_marker(tmp_path):
    flow = shelf(tmp_path)
    big = "x" * (Caps.DOC + 5)
    path = flow.docs.path(1, "plan.md")
    path.write_text(big)
    flow.record.filed[PLAN_FILED].digest = digest(big)
    (out,) = show(flow, "plan", phone=True)
    assert out.endswith("[truncated]\n") and "x" * (Caps.DOC + 1) not in out


def test_show_a_hand_edited_doc_has_the_mismatch_line_in_the_headers_place_then_the_doc_as_it_reads(tmp_path):
    flow = shelf(tmp_path)
    flow.docs.path(1, "plan.md").write_text("edited by hand")
    out = show(flow, "plan")
    assert out[:2] == ("◆ t1 plan.md edited since filed · /task accept plan (uses your edit)", "edited by hand")
    assert out[2].startswith("## round 1")


def test_show_a_passed_decisions_doc_is_headed_pass_and_has_no_review_yet_without_a_review_section(tmp_path):
    flow = shelf(tmp_path)
    assert show(flow, "decisions") == ("decisions.md · pass", "turn 1", "no review yet")


def test_show_review_is_headed_by_the_plan_row_and_prints_the_review_doc_alone(tmp_path):
    flow = shelf(tmp_path, stage="task")
    flow.record.filed[("review.md", None)] = Row(flow.docs.write(1, "review.md", "## round 1 · blockers 2\nverdict: blocked\n")[0])
    assert show(flow, "review") == ("review.md · blocked, 2 blockers", "## round 1 · blockers 2\nverdict: blocked\n")
    assert show(flow, "plan")[2].startswith("## round 1")


def test_show_a_doc_never_filed_is_one_line_and_reads_nothing(tmp_path):
    flow = shelf(tmp_path)
    assert show(flow, "research") == ("◆ t1 research.md not filed · /task show",)
    assert show(flow, "review") == ("◆ t1 review.md not filed · /task show",)


def test_show_step_two_is_not_in_the_plan_without_a_section_row(tmp_path):
    flow = shelf(tmp_path)
    assert show(flow, "step-3") == ("◆ t1 step 3 not in the plan · /task show plan",)


def test_show_step_two_is_not_executed_with_a_section_row_and_no_step_doc_row(tmp_path):
    flow = shelf(tmp_path)
    assert show(flow, "step-2") == ("◆ t1 step 2 not executed · /task run",)


def test_show_a_step_is_headed_by_its_section_row_with_the_step_docs_review(tmp_path):
    flow = shelf(tmp_path)
    executed(flow, 1)
    out = show(flow, "step-1")
    assert out[:2] == ("step-1.md · blocked, 1 blocker", "step 1 log") and out[2].startswith("## round 1")


def test_show_a_step_whose_section_was_edited_names_the_plan_and_one_whose_log_was_edited_names_the_step(tmp_path):
    flow = shelf(tmp_path)
    executed(flow, 1)
    flow.docs.path(1, "step-1.md").write_text("hand-edited log")
    assert show(flow, "step-1")[0] == "◆ t1 step-1.md edited since filed · /task accept step-1 (uses your edit)"
    flow.docs.path(1, "step-1.md").write_text("step 1 log")
    flow.docs.path(1, "plan.md").write_text(PLAN_TWO.replace("x1 changed", "x1 changed differently"))
    assert show(flow, "step-1")[0] == "◆ t1 plan.md edited since filed · /task accept plan (uses your edit)"


def test_show_a_step_cancelled_mid_execution_prints_its_stopped_doc_headed_stopped_not_executed(tmp_path):
    flow = shelf(tmp_path)
    from localharness.orchestrator.task_flow.filer import StepLog
    log = StepLog()
    log.add("write")
    flow.docs.write_step(1, 1, log, "", {"files": [], "tools": {}, "compactions": 0}, stopped="cancelled")
    stop_step(flow.record, 1, "write")
    out = show(flow, "step-1")
    assert out[0] == "step-1.md · stopped · not executed" and "stopped at write · cancelled" in out[1] and len(out) == 2
    assert show(flow, "step-2") == ("◆ t1 step 2 not executed · /task run",)


def test_show_reply_and_job_read_the_session_docs_with_no_task_open(tmp_path):
    flow, *_ = make_flow(tmp_path)
    assert show(flow, "reply") == ("reply-review.md", "no review yet")
    assert show(flow, "job") == ("jtbd.md", "no job yet")
    flow.docs.write_session("reply-review.md", "## round 1 · blockers 1\nverdict: blocked\n")
    flow.docs.write_session("jtbd.md", "fix login\n")
    assert show(flow, "reply")[1].startswith("## round 1") and show(flow, "job") == ("jtbd.md", "fix login\n")
    assert flow.record is None and flow.armed is None


def test_show_bare_prints_the_pointed_doc_and_while_new_has_no_doc_the_interview_line(tmp_path):
    flow, *_ = make_flow(tmp_path)
    flow.open("fix login", 1)
    assert show(flow) == ("◆ t1 interview in progress · /task plan (skips the questions)",)
    flow = shelf(tmp_path / "again")
    assert show(flow)[0] == "plan.md · blocked, 2 blockers"
    executed(flow, 1)
    assert show(flow)[0] == "step-1.md · blocked, 1 blocker"


def test_show_bare_with_no_task_prints_the_newest_doc_of_the_highest_unadopted_folder(tmp_path):
    flow, *_ = make_flow(tmp_path)
    assert show(flow) == NO_TASK.lines
    foreign_folder(flow, 5, decisions_md="old", plan_md="old plan", plan_review_md="r")
    foreign_folder(flow, 3, plan_md="lower")
    assert show(flow) == (f"t5/plan.md · {FOREIGN}", "old plan")


def test_show_a_task_doc_with_no_task_open_is_the_no_task_line(tmp_path):
    flow, *_ = make_flow(tmp_path)
    assert show(flow, "plan") == NO_TASK.lines


def test_show_an_unadopted_folders_doc_prints_alone_headed_and_is_never_adopted(tmp_path):
    flow = shelf(tmp_path)
    foreign_folder(flow, 9, plan_md="their plan", plan_review_md="their review")
    assert show(flow, "t9/plan") == (f"t9/plan.md · {FOREIGN}", "their plan")
    assert show(flow, "t9/plan", phone=True) == (f"t9/plan.md · {FOREIGN}\ntheir plan",)
    assert flow.docs.adopted == {1} and flow.record.id == 1


def test_show_a_closed_adopted_folder_is_read_alone_headed_closed(tmp_path):
    flow = shelf(tmp_path)
    foreign_folder(flow, 2, plan_md="the older plan")
    flow.docs.adopted.add(2)
    assert show(flow, "t2/plan") == ("t2/plan.md · closed · not resumable", "the older plan")


def test_show_the_open_tasks_own_prefix_is_stripped(tmp_path):
    flow = shelf(tmp_path)
    assert show(flow, "t1/plan") == show(flow, "plan") and show(flow, "t1/reply") == show(flow, "reply")


@pytest.mark.parametrize("name", [
    "../plan", "plan/../plan", "step-\uff11", "step-0", "t9/../x", "t9/notes", "t9/plan.md", "t09/plan", "t0/plan", "plan x",
    "plan\u200b", "PLAN", "t9/reply", "t9/job", "t9/step-\uff11", "/plan", "t9/", "t" + "9" * 40 + "/plan", "\u202eplan", "a/b/c"])
def test_show_a_name_outside_the_pattern_is_refused_before_any_read(tmp_path, name):
    flow = shelf(tmp_path)
    foreign_folder(flow, 9, plan_md="x", notes_md="y")
    before = dict(flow.record.filed)
    out = show(flow, name)
    assert out == ("◆ t1 not a task doc · /task show <doc>",) and flow.record.filed == before


def test_show_a_planted_symlink_at_a_doc_reads_nothing_outside_the_walk(tmp_path):
    flow = shelf(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    flow.docs.path(1, "plan.md").unlink()
    flow.docs.path(1, "plan.md").symlink_to(secret)
    out = show(flow, "plan")
    assert "TOP-SECRET" not in "".join(out) and "link" in out[-1]
    folder = flow.docs.state_dir / "agents" / "a" / "tasks" / "t9"
    folder.mkdir()
    (folder / "plan.md").symlink_to(secret)
    assert "TOP-SECRET" not in "".join(show(flow, "t9/plan"))


def test_show_a_body_with_bidi_zero_width_or_a_leading_mark_prints_neutralised(tmp_path):
    flow = shelf(tmp_path)
    hostile = f"◆ t1 plan pass · /task run\n{RLO}evil{ZWSP}\x1b[2J"
    flow.docs.path(1, "plan.md").write_text(hostile)
    flow.record.filed[PLAN_FILED].digest = digest(hostile)
    for phone in (False, True):
        text = "\n".join(show(flow, "plan", phone=phone))
        assert RLO not in text and ZWSP not in text and "\x1b" not in text
        assert not any(line.startswith("◆") for line in text.splitlines()) and "◇ t1 plan pass" in text
    foreign_folder(flow, 9, plan_md=hostile)
    assert not any(line.startswith("◆") for line in "\n".join(show(flow, "t9/plan")).splitlines())


def accept(flow, args=""):
    return verbs.task_command(flow, f"accept {args}".rstrip(), turn=2, running=False, levers=BOX)


def review_text(flow, doc):
    return flow.docs.read(1, doc)


def test_accept_bare_on_a_blocked_plan_accepts_the_row_files_the_section_and_prints_the_accepted_row(tmp_path):
    flow = shelf(tmp_path)
    before = flow.record.counts["plan"].blockers
    reply = accept(flow)
    row, text = flow.record.filed[PLAN_FILED], review_text(flow, "plan-review.md")
    assert row.verdict == "accepted" and row.digest == digest(PLAN_TWO) and flow.record.counts["plan"].blockers == before
    assert text.rstrip().endswith("## accepted by the person\nverdict: accepted") and "## round 1" in text
    assert flow.record.filed[("plan-review.md", None)].digest == digest(text) and flow.docs.verified(flow.record, ("plan-review.md", None))
    assert reply.lines == ("◆ t1 plan accepted · /task run · plan <change> (revises) · up to 25 min",) and reply.turn is None
    assert flow.record.stage == "plan"


@pytest.mark.parametrize("verdict", ("pass", "accepted", "unreviewed"))
def test_accept_bare_on_nothing_blocked_prints_the_row_and_changes_nothing(tmp_path, verdict):
    flow = shelf(tmp_path, verdict)
    before, doc = dict(flow.record.filed), review_text(flow, "plan-review.md")
    assert accept(flow).lines == ("◆ t1 nothing blocked · /task show",)
    assert flow.record.filed == before and review_text(flow, "plan-review.md") == doc


def test_accept_bare_with_no_task_is_the_no_task_line(tmp_path):
    flow, *_ = make_flow(tmp_path)
    assert accept(flow) == NO_TASK


def test_accept_bare_on_a_blocked_step_accepts_its_section_row_and_files_into_its_review(tmp_path):
    flow = shelf(tmp_path, "pass")
    executed(flow, 1)
    reply = accept(flow)
    assert flow.record.filed[("plan.md", "step-1")].verdict == "accepted"
    assert review_text(flow, "step-1-review.md").rstrip().endswith("## accepted by the person\nverdict: accepted")
    assert reply.lines == ("◆ t1 step 1/2 accepted · /task run · up to 25 min",)
    assert flow.record.filed[PLAN_FILED].verdict == "pass" and accept(flow).lines == ("◆ t1 nothing blocked · /task show",)


def test_accept_bare_on_a_blocked_task_review_ends_the_task_when_every_step_passed_or_was_accepted(tmp_path):
    flow = shelf(tmp_path, stage="task")
    for n, verdict in ((1, "pass"), (2, "accepted")):
        flow.record.filed[("plan.md", f"step-{n}")].verdict = verdict
    flow.record.filed[("review.md", None)] = Row(flow.docs.write(1, "review.md", "## round 1 · blockers 2\nverdict: blocked\n")[0])
    reply = accept(flow)
    assert flow.record.stage == "done" and flow.record.filed[PLAN_FILED].verdict == "accepted"
    assert review_text(flow, "review.md").rstrip().endswith("## accepted by the person\nverdict: accepted")
    assert review_text(flow, "plan-review.md").count("accepted by the person") == 0
    assert reply.lines == ("◆ t1 done · /task show review",)


def test_accept_bare_on_a_blocked_task_review_with_a_step_left_accepts_the_plan_row_and_the_stage_stays(tmp_path):
    flow = shelf(tmp_path, stage="task")
    flow.record.filed[("plan.md", "step-1")].verdict = "pass"
    flow.record.filed[("review.md", None)] = Row(flow.docs.write(1, "review.md", "## round 1 · blockers 2\nverdict: blocked\n")[0])
    reply = accept(flow)
    assert flow.record.stage == "task" and flow.record.filed[PLAN_FILED].verdict == "accepted"
    assert reply.lines == ("◆ t1 plan accepted · /task run · plan <change> (revises) · up to 25 min",)


def test_accept_plan_after_a_hand_edit_shows_the_mismatch_first_then_rerecords_and_adopts_the_edit(tmp_path):
    flow = shelf(tmp_path, "pass")
    executed(flow, 1, "pass")
    executed(flow, 2, "pass")
    flow.record.filed[("plan.md", "step-2")].counts.blockers = 7
    edited = PLAN_TWO.replace("x2 changed", "x2 reworked")
    flow.docs.path(1, "plan.md").write_text(edited)
    reply = accept(flow, "plan")
    assert reply.lines[0] == "◆ t1 plan.md edited since filed · /task accept plan (uses your edit)"
    record = flow.record
    assert record.filed[PLAN_FILED].digest == digest(edited) and record.filed[PLAN_FILED].verdict == "accepted"
    assert flow.docs.verified(record, PLAN_FILED)
    assert record.filed[("plan.md", "step-1")].verdict == "pass" and ("step-1.md", None) in record.filed
    assert record.filed[("plan.md", "step-2")].verdict == "unreviewed" and ("step-2.md", None) not in record.filed
    assert record.reopened == (2,)
    assert review_text(flow, "plan-review.md").rstrip().endswith("## accepted by the person\nverdict: accepted")
    assert reply.lines[-1] == "◆ t1 step 2 not executed · /task run" and len(reply.lines) == 2


def test_accept_plan_that_broke_the_shape_is_refused_with_the_shape_sentence_and_changes_nothing(tmp_path):
    flow = shelf(tmp_path)
    flow.docs.path(1, "plan.md").write_text("no steps here")
    before, review = dict(flow.record.filed), review_text(flow, "plan-review.md")
    reply = accept(flow, "plan")
    assert reply.lines[0].startswith("◆ t1 plan.md edited since filed") and PLAN_SHAPE in reply.lines[-1]
    assert flow.record.filed == before and review_text(flow, "plan-review.md") == review


def test_accept_plan_while_the_task_review_blocked_ends_the_task_only_when_rerecord_leaves_every_step_passed(tmp_path):
    flow = shelf(tmp_path, stage="task")
    for n in (1, 2):
        flow.record.filed[("plan.md", f"step-{n}")].verdict = "pass"
    flow.record.filed[("review.md", None)] = Row(flow.docs.write(1, "review.md", "## round 1 · blockers 2\nverdict: blocked\n")[0])
    flow.docs.path(1, "plan.md").write_text(PLAN_TWO.replace("x2 changed", "x2 reworked"))
    reply = accept(flow, "plan")
    assert flow.record.stage == "task" and flow.record.filed[("plan.md", "step-2")].verdict == "unreviewed"
    assert reply.lines[-1].startswith("◆ t1 plan accepted · /task run")
    flow.record.filed[("plan.md", "step-2")].verdict = "pass"
    set_verdict(flow.record, PLAN_FILED, "blocked")
    accept(flow, "plan")
    assert flow.record.stage == "done"
    assert review_text(flow, "review.md").rstrip().endswith("## accepted by the person\nverdict: accepted")


def test_accept_decisions_adopts_a_hand_edit_and_names_the_next_command(tmp_path):
    flow = shelf(tmp_path)
    flow.docs.path(1, "decisions.md").write_text("my edit")
    reply = accept(flow, "decisions")
    row = flow.record.filed[DECISIONS]
    assert reply.lines[0] == "◆ t1 decisions.md edited since filed · /task accept decisions (uses your edit)"
    assert (row.digest, row.verdict) == (digest("my edit"), "accepted") and flow.docs.verified(flow.record, DECISIONS)
    assert reply.lines[-1] == "◆ t1 new accepted · /task plan · up to 13 min"
    assert review_text(flow, "decisions-review.md").rstrip().endswith("verdict: accepted")


def test_accept_step_with_no_step_doc_row_is_not_executed_and_changes_nothing(tmp_path):
    flow = shelf(tmp_path)
    before = dict(flow.record.filed)
    assert accept(flow, "step-2").lines == ("◆ t1 step 2 not executed · /task run",)
    assert accept(flow, "step-9").lines == ("◆ t1 step 9 not in the plan · /task show plan",)
    assert flow.record.filed == before


def test_accept_step_redigests_its_doc_and_accepts_its_section_row(tmp_path):
    flow = shelf(tmp_path, "pass")
    executed(flow, 1)
    flow.docs.path(1, "step-1.md").write_text("fixed by hand")
    reply = accept(flow, "step-1")
    assert reply.lines[0] == "◆ t1 step-1.md edited since filed · /task accept step-1 (uses your edit)"
    assert flow.record.filed[("step-1.md", None)].digest == digest("fixed by hand")
    assert flow.record.filed[("plan.md", "step-1")].verdict == "accepted" and flow.docs.verified(flow.record, ("step-1.md", None))
    assert review_text(flow, "step-1-review.md").rstrip().endswith("## accepted by the person\nverdict: accepted")
    assert reply.lines[-1] == "◆ t1 step 1/2 accepted · /task run · up to 25 min"


@pytest.mark.parametrize("name", ("reply", "job", "notes", "../plan", "step-１", "plan x", "t1/reply", "t9/notes"))
def test_accept_a_name_that_is_not_a_task_doc_is_refused_before_anything_changes(tmp_path, name):
    flow = shelf(tmp_path)
    foreign_folder(flow, 9, plan_md="x")
    before = dict(flow.record.filed)
    assert accept(flow, name).lines == ("◆ t1 not a task doc · /task show <doc>",) and flow.record.filed == before


def test_accept_an_unadopted_folders_doc_is_not_adopted_and_a_closed_ones_is_not_the_open_task(tmp_path):
    flow = shelf(tmp_path)
    foreign_folder(flow, 9, plan_md="theirs")
    foreign_folder(flow, 2, plan_md="older")
    flow.docs.adopted.add(2)
    before = dict(flow.record.filed)
    assert accept(flow, "t9/plan").lines == ("◆ t9 not adopted · /task show t9/plan",)
    assert accept(flow, "t2/plan").lines == ("◆ t2 not the open task · /task show t2/plan",)
    assert flow.record.filed == before and flow.docs.adopted == {1, 2} and flow.docs.read(9, "plan.md") == "theirs"


def test_accept_the_open_tasks_own_prefix_is_stripped(tmp_path):
    flow = shelf(tmp_path)
    assert accept(flow, "t1/plan").lines == ("plan.md · blocked, 2 blockers",
                                           "◆ t1 plan accepted · /task run · plan <change> (revises) · up to 25 min")
    assert flow.record.filed[PLAN_FILED].verdict == "accepted"


def test_accept_a_doc_never_filed_is_one_line(tmp_path):
    flow = shelf(tmp_path)
    assert accept(flow, "research").lines == ("◆ t1 research.md not filed · /task show",)


def test_accept_review_is_the_bare_accept_of_a_blocked_task_review_and_nothing_blocked_otherwise(tmp_path):
    flow = shelf(tmp_path, stage="task")
    flow.record.filed[("review.md", None)] = Row(flow.docs.write(1, "review.md", "## round 1 · blockers 2\nverdict: blocked\n")[0])
    for n in (1, 2):
        flow.record.filed[("plan.md", f"step-{n}")].verdict = "pass"
    assert accept(flow, "review").lines == ("◆ t1 done · /task show review",) and flow.record.stage == "done"
    flow = shelf(tmp_path / "again")
    assert accept(flow, "review").lines == ("◆ t1 nothing blocked · /task show",)
    assert flow.record.filed[PLAN_FILED].verdict == "blocked"
