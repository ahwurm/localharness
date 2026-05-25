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
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.preserve_first_n = preserve_first_n
        self.preserve_last_n = preserve_last_n

    def repair_tool_pairing(self, messages: list[Message]) -> list[Message]:
        """Remove orphaned tool role messages.

        An orphaned tool message is one whose tool_call_id does not appear in
        any preceding assistant message's tool_calls list.
        Returns a new list — input is never modified.
        """
        # Collect all valid tool_call IDs from assistant messages
        valid_ids: set[str] = set()
        for m in messages:
            if m.get("role") == "assistant":
                for tc in (m.get("tool_calls") or []):
                    tc_id = None
                    if isinstance(tc, dict):
                        tc_id = tc.get("id")
                    else:
                        tc_id = getattr(tc, "id", None)
                    if tc_id:
                        valid_ids.add(tc_id)

        # Keep messages that are not orphaned tool results
        result = []
        for m in messages:
            if m.get("role") == "tool":
                tool_call_id = m.get("tool_call_id", "")
                if tool_call_id not in valid_ids:
                    continue  # drop orphan
            result.append(m)
        return result

    def build_messages(
        self,
        messages: list[Message],
        tool_schemas: list[dict] | None = None,
    ) -> list[Message]:
        """Return a repaired copy of messages ready for LLM request.

        Does NOT modify the input list. Phase 3 does not implement summarize-middle
        (that is CTX-02 in Phase 4). Just copy + repair.
        """
        copied = list(messages)
        return self.repair_tool_pairing(copied)
