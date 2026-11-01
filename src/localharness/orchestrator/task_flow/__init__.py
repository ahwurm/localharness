"""The task flow: a table-driven pipeline of fresh children run one unit per call."""
from __future__ import annotations
from contextlib import suppress

from typing import Any, Awaitable, Callable, Iterable

from localharness.orchestrator.task_flow.filer import Docs
from localharness.orchestrator.task_flow.panel import FlowCap, Panel, critic_names
from localharness.orchestrator.task_flow.record import Record, Row, clear_turn, limited, open_task, set_stage
from localharness.orchestrator.task_flow.runner import root_note, run_unit

__all__ = ["TaskFlow"]


class TaskFlow:
    """One task flow: the open record, the one panel, and the host's callables the runner reaches through."""

    def __init__(self, *, run_config_child: Callable[..., Awaitable[Any]], emit: Callable[[str, str, tuple, str], None],
                 exclude: Callable[[float], None], docs: Docs, config: Any, root_tools: Iterable[str], llm: Any = None,
                 token_counter: Any = None, messages: Callable[[], Iterable[dict]] = lambda: (), bus: Any = None,
                 record: Record | None = None, turn: Callable[[], int] = lambda: 0) -> None:
        self.run_config_child, self.emit, self.exclude, self.docs, self.config = run_config_child, emit, exclude, docs, config
        self.root_tools, self.llm, self.token_counter, self.messages, self.bus = list(root_tools), llm, token_counter, messages, bus
        self.limits = limited(config.task_flow, known=critic_names())[0]
        self.cap = FlowCap(self.limits.flow_cap_minutes)
        self.reply_panel = Panel(run_config_child, self.limits, self.cap, docs)
        self.record, self.excluded, self.turn = record, 0.0, turn
        if record is not None:
            docs.adopted.add(record.id)
        self.called, self.said, self.sentence = set[str](), set[str](), ""
        self.armed: str | None = None
        self.unit: tuple[str, int] | None = None
        self.busy = False
        self.reply_panel.emit = emit
        self.reply_panel.on_delivered = self.settle_close

    def open(self, objective: str, turn: int) -> Record:
        self.record = open_task(self.docs.new_task(), objective, turn)
        set_stage(self.record, "opened")
        with suppress(OSError):
            self.docs.scratch(self.record.id).mkdir(parents=True, exist_ok=True, mode=0o700)
        return self.record

    def close(self, doc: str) -> str:
        """The model closes a task it finished before any plan: the result is filed as review.md and the task ends done."""
        record = self.record
        if record is None:
            return "no task is open"
        if record.stage not in ("new", "research", "plan"):
            return "a task past its plan ends through its steps"
        if ("decisions.md", None) not in record.filed:
            return "decisions.md must be filed first"
        record.closing = doc
        return f"t{record.id} closing \u00b7 the critics read your reply as the result; it ends done when they pass it"

    def settle_close(self, verdict: str | None) -> None:
        """After the reply the critics read: a closing task ends done on a pass and stays open otherwise."""
        record = self.record
        if record is None or record.closing is None:
            return
        doc, record.closing = record.closing, None
        if verdict != "pass":
            if self.emit:
                self.emit(f"t{record.id}", "not closed", ("task(stage=\"done\") again once the reply passes",), f"the reply was {verdict or 'unreviewed'}")
            return
        digest, _ = self.docs.write(record.id, "review.md", doc)
        record.filed[("review.md", None)] = Row(digest, "pass")
        set_stage(record, "task pass")

    async def run(self, boundary: str, carried: str | None = None, by: str = "model") -> str:
        return await run_unit(self, boundary, carried, by)

    def note(self) -> str:
        return root_note(self)

    def end_turn(self) -> None:
        if self.record is not None:
            clear_turn(self.record)
        self.cap.reset()
        self.excluded, self.reply_panel.jtbd_stood = 0.0, False
        self.called.clear()
        self.said.clear()
        self.sentence, self.armed, self.unit, self.busy = "", None, None, False
