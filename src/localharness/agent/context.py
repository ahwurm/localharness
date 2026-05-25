"""ContextManager: build_messages with repair_tool_pairing boundary guard."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localharness.core.types import Message

log = logging.getLogger("localharness.agent.context")

RESPONSE_RESERVE_TOKENS: int = 4096


class TokenCounter:
    """Token counting with tiktoken (preferred) or char heuristic fallback."""

    def __init__(self) -> None:
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            pass

    def count(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return len(text) // 4  # char heuristic fallback

    def count_messages(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            total += 4  # message overhead tokens
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += self.count(content)
            # tool_calls in assistant messages
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                total += self.count(fn.get("name", ""))
                total += self.count(fn.get("arguments", ""))
        return total


@dataclass
class TokenBudget:
    total_limit: int
    current_usage: int
    tool_schema_tokens: int
    headroom: int = 0

    def __post_init__(self) -> None:
        self.headroom = self.total_limit - self.current_usage - self.tool_schema_tokens - RESPONSE_RESERVE_TOKENS

    @property
    def usage_fraction(self) -> float:
        return (self.current_usage + self.tool_schema_tokens) / self.total_limit if self.total_limit > 0 else 1.0

    @property
    def needs_summary_compact(self) -> bool:
        return self.usage_fraction >= 0.80

    @property
    def needs_full_compact(self) -> bool:
        return self.usage_fraction >= 0.95


def _collect_valid_tool_ids(messages: list[Message]) -> set[str]:
    """Return the set of all tool_call IDs referenced in assistant tool_calls."""
    valid: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    valid.add(tc_id)
    return valid


def _repair_tool_pairing(messages: list[Message]) -> list[Message]:
    """Remove orphaned tool role messages. Returns new list; input unchanged."""
    valid_ids = _collect_valid_tool_ids(messages)
    result = []
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id", "") not in valid_ids:
            continue
        result.append(m)
    return result


class ToolResultCapStage:
    """Stage 1: Cap oversized tool result messages."""

    def __init__(self, max_chars: int = 50_000) -> None:
        self.max_chars = max_chars

    def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        result = []
        modified = False
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content") or ""
                if len(content) > self.max_chars:
                    truncated = content[: self.max_chars] + f"\n... [truncated at {self.max_chars} chars]"
                    result.append({**m, "content": truncated})
                    modified = True
                    continue
            result.append(m)
        return result, modified


class BoundaryGuardStage:
    """Stage 2: Repair orphaned tool_use/tool_result pairs."""

    def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        repaired = _repair_tool_pairing(messages)
        return repaired, repaired != messages


class SummaryCompactionStage:
    """Stage 3: Summarize middle messages at >= 80% context utilization."""

    def __init__(
        self,
        preserve_first_n: int,
        preserve_last_n: int,
        llm_summarize_fn: Any = None,
        compact_md_path: Path | None = None,
    ) -> None:
        self.preserve_first_n = preserve_first_n
        self.preserve_last_n = preserve_last_n
        self.llm_summarize_fn = llm_summarize_fn
        self.compact_md_path = compact_md_path

    async def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        if budget.usage_fraction < 0.80:
            return messages, False

        first_boundary = self._safe_cut_boundary(messages, self.preserve_first_n, "forward")
        last_boundary = self._safe_cut_boundary(messages, len(messages) - self.preserve_last_n, "backward")

        if last_boundary <= first_boundary:
            return messages, False

        middle = messages[first_boundary:last_boundary]
        if len(middle) <= 2:
            return messages, False

        if self.llm_summarize_fn is None:
            return messages, False

        try:
            summary_text = await self.llm_summarize_fn(middle)
        except Exception as exc:
            log.warning("Summarization failed: %s. Skipping summary compaction.", exc)
            return messages, False

        summary_message = {"role": "assistant", "content": f"[Context Summary]\n{summary_text}"}
        new_messages = messages[:first_boundary] + [summary_message] + messages[last_boundary:]

        if self.compact_md_path is not None:
            _write_compact_md(self.compact_md_path, summary_text)

        log.info("Summary compaction: %d messages → 1 summary", len(middle))
        return new_messages, True

    def _safe_cut_boundary(self, messages: list[Message], start_idx: int, direction: str) -> int:
        """Find a safe cut boundary that does not split tool_use/tool_result pairs."""
        n = len(messages)
        start_idx = max(0, min(start_idx, n))

        if direction == "forward":
            i = start_idx
            while i < n:
                if _is_safe_cut_after(messages, i - 1):
                    return i
                i += 1
            return start_idx
        else:  # backward
            i = start_idx
            while i > 0:
                if _is_safe_cut_after(messages, i - 1):
                    return i
                i -= 1
            return start_idx


class FullAutoCompactStage:
    """Stage 4: Emergency full-session compaction at >= 95% utilization."""

    def __init__(
        self,
        llm_summarize_fn: Any = None,
        compact_md_path: Path | None = None,
    ) -> None:
        self.llm_summarize_fn = llm_summarize_fn
        self.compact_md_path = compact_md_path
        # Use aggressive boundaries for emergency compaction
        self._summary_stage = SummaryCompactionStage(
            preserve_first_n=1,
            preserve_last_n=2,
            llm_summarize_fn=llm_summarize_fn,
            compact_md_path=compact_md_path,
        )

    async def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        if budget.usage_fraction < 0.95:
            return messages, False
        # Reuse SummaryCompactionStage with aggressive settings, forcing it to fire
        # Create a fake budget at 80% to trigger it
        forced_budget = TokenBudget(
            total_limit=budget.total_limit,
            current_usage=int(budget.total_limit * 0.81),
            tool_schema_tokens=0,
        )
        return await self._summary_stage.apply(messages, forced_budget, token_counter)


class CompactionPipeline:
    """Runs all 4 compaction stages in order."""

    def __init__(
        self,
        token_counter: TokenCounter,
        tool_result_cap: int = 50_000,
        preserve_first_n: int = 4,
        preserve_last_n: int = 8,
        llm_summarize_fn: Any = None,
        compact_md_path: Path | None = None,
    ) -> None:
        self._token_counter = token_counter
        self._stages: list = [
            ToolResultCapStage(max_chars=tool_result_cap),
            BoundaryGuardStage(),
            SummaryCompactionStage(
                preserve_first_n=preserve_first_n,
                preserve_last_n=preserve_last_n,
                llm_summarize_fn=llm_summarize_fn,
                compact_md_path=compact_md_path,
            ),
            FullAutoCompactStage(
                llm_summarize_fn=llm_summarize_fn,
                compact_md_path=compact_md_path,
            ),
        ]

    async def run(
        self,
        messages: list[Message],
        budget: TokenBudget,
    ) -> tuple[list[Message], bool]:
        import inspect
        any_modified = False
        working = messages
        for stage in self._stages:
            call = stage.apply(working, budget, self._token_counter)
            if inspect.iscoroutine(call):
                result, modified = await call
            else:
                result, modified = call
            if modified:
                working = result
                any_modified = True
                # Recompute budget after modification
                new_usage = self._token_counter.count_messages(working)
                budget = TokenBudget(
                    total_limit=budget.total_limit,
                    current_usage=new_usage,
                    tool_schema_tokens=budget.tool_schema_tokens,
                )
        return working, any_modified


def _is_safe_cut_after(messages: list[Message], idx: int) -> bool:
    """Return True if it's safe to cut the message list after index idx."""
    if idx < 0:
        return True  # beginning of list is safe
    msg = messages[idx]
    role = msg.get("role")
    # Safe after a user message
    if role == "user":
        return True
    # Safe after a tool result (last one in a batch means all results collected)
    if role == "tool":
        # Check if the next message (if any) is also a tool result
        next_idx = idx + 1
        if next_idx < len(messages) and messages[next_idx].get("role") == "tool":
            return False  # more tool results follow — not a safe boundary
        return True
    # Safe after an assistant message with no tool_calls
    if role == "assistant" and not msg.get("tool_calls"):
        return True
    return False


def _write_compact_md(path: Path, content: str) -> None:
    """Atomically write content to compact.md."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    os.replace(str(tmp), str(path))


def load_compact_md(compact_md_path: Path) -> Message | None:
    """Load compact.md from disk and return as a system message, or None if not found."""
    if compact_md_path.exists():
        content = compact_md_path.read_text().strip()
        if content:
            return {"role": "system", "content": f"[Prior Session Context]\n{content}"}
    return None


class ContextManager:
    """Manages message list preparation for LLM requests.

    repair_tool_pairing is called on every build_messages call to ensure
    no orphaned tool_result messages are sent to the LLM (which causes HTTP 400).
    """

    def __init__(
        self,
        max_context_tokens: int = 128_000,
        preserve_first_n: int = 4,
        preserve_last_n: int = 8,
        pipeline: CompactionPipeline | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.preserve_first_n = preserve_first_n
        self.preserve_last_n = preserve_last_n
        self._pipeline = pipeline
        self._token_counter = TokenCounter()

    def repair_tool_pairing(self, messages: list[Message]) -> list[Message]:
        """Remove orphaned tool role messages.

        An orphaned tool message is one whose tool_call_id does not appear in
        any preceding assistant message's tool_calls list.
        Returns a new list — input is never modified.
        """
        return _repair_tool_pairing(messages)

    def build_messages(
        self,
        messages: list[Message],
        tool_schemas: list[dict] | None = None,
    ) -> list[Message]:
        """Return a repaired copy of messages ready for LLM request.

        Does NOT modify the input list. Applies compaction pipeline if set,
        otherwise just repairs tool pairing.
        """
        copied = list(messages)
        return self.repair_tool_pairing(copied)
