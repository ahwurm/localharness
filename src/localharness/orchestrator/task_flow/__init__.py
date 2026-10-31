"""The task flow: a table-driven pipeline of fresh children run one unit per call."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable

from localharness.orchestrator.task_flow.filer import Docs
from localharness.orchestrator.task_flow.panel import FlowCap, Panel, critic_names
from localharness.orchestrator.task_flow.record import Record, clear_turn, limited, open_task, set_stage
from localharness.orchestrator.task_flow.runner import root_note, run_unit

__all__ = ["TaskFlow"]


class TaskFlow:
    """One task flow: the open record, the one panel, and the host's callables the runner reaches through."""

    def __init__(self, *, run_config_child: Callable[..., Awaitable[Any]], emit: Callable[[str, str, tuple, str], None],
                 exclude: Callable[[float], None], docs: Docs, config: Any, root_tools: Iterable[str], llm: Any = None,
                 token_counter: Any = None, messages: Callable[[], Iterable[dict]] = lambda: (), bus: Any = None,
                 record: Record | None = None) -> None:
        self.run_config_child, self.emit, self.exclude, self.docs, self.config = run_config_child, emit, exclude, docs, config
        self.root_tools, self.llm, self.token_counter, self.messages, self.bus = list(root_tools), llm, token_counter, messages, bus
        self.limits, self.notes = limited(config.task_flow, known=critic_names())
        self.cap = FlowCap(self.limits.flow_cap_minutes)
        self.reply_panel = Panel(run_config_child, self.limits, self.cap, docs)
        self.record, self.revision_block, self.jtbd_stood, self.excluded = record, "", False, 0.0

    def open(self, objective: str, turn: int) -> Record:
        self.record = open_task(self.docs.new_task(), objective, turn)
        set_stage(self.record, "opened")
        return self.record

    async def run(self, boundary: str, carried: str | None = None, by: str = "model") -> str:
        return await run_unit(self, boundary, carried, by)

    def note(self) -> str:
        return root_note(self)

    def end_turn(self) -> None:
        if self.record is not None:
            clear_turn(self.record)
        self.cap.reset()
        self.excluded, self.jtbd_stood = 0.0, False
