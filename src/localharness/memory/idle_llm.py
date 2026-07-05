"""The SINGLE safe idle-window LLM path — the machine-safety spine of Phase 36.

This box HARD-HUNG twice in 24h under vLLM long-context prefill. Every Phase-36
model-look (chapter-writer 36-04, reconciliation 36-05, mining 36-06) MUST route its
idle LLM work through here — never call a client directly — so the freeze-class safety
is unified and no look can fork a weaker copy. Two safety properties live here:

  1. Cancellable race (`run_cancellable`): a user turn must never wait behind an idle
     generation. The FIRST_COMPLETED race is extracted VERBATIM from the shipped
     consolidation.py:317-338 pattern (regression-locked by test_inference_gate_cancel.py).
     On cancel the generation task is `.cancel()`'d so it raises inside the held
     `_inference_gate` and RELEASES the serial slot promptly.
  2. Context bound (`complete_cancellable` char_cap): the prompt is truncated BEFORE the
     call, so an idle look can never launch an unattended long-context prefill.

`LLMTextAdapter` bridges the real `LLMClient` (`complete(messages)->(message,usage)`) to
the seam's `.complete(prompt)->str` contract — the wiring the replay seam never had
(start_cmd.py passes no `llm=`; it was only ever fake-LLM tested).

`grounded`/`ground_numbers` enforce the pre-committed KILL: no schema/chapter token that
isn't derivable from its member lessons. This module has NO store writes and NO locking
of its own — the only serialization is the client's own `_inference_gate`.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

log = logging.getLogger(__name__)


async def run_cancellable(coro: Awaitable[Any], cancel_event: asyncio.Event) -> Any | None:
    """Race a generation coro against `cancel_event` (extracted verbatim from the shipped
    consolidation.py:317-338 pattern). On cancel, `gen_task.cancel()` so it raises inside
    the held `_inference_gate` and RELEASES the serial slot promptly. Returns the result,
    or None on cancel/exception (never raises into the idle pass)."""
    gen_task = asyncio.ensure_future(coro)
    cancel_task = asyncio.ensure_future(cancel_event.wait())
    done, _ = await asyncio.wait(
        {gen_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if gen_task in done:
        cancel_task.cancel()
        try:
            return await gen_task
        except Exception:
            log.exception("idle LLM generation failed (non-fatal)")
            return None
    gen_task.cancel()  # releases the inference gate's slot
    try:
        await gen_task
    except (asyncio.CancelledError, Exception):
        pass
    return None


class LLMTextAdapter:
    """Adapt the real `LLMClient` (`complete(messages)->(message,usage)`) to the seam
    contract `.complete(prompt: str) -> str`. The replay seam was fake-LLM tested but
    NEVER wired to a real client (start_cmd.py passes no `llm=`); this adapter is that
    missing bridge. `Message` is `dict[str, Any]` (core.types), so a plain user-role dict
    is the wire shape; the returned message carries text on `.content` (may be None)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def complete(self, prompt: str) -> str:
        msg, _usage = await self._client.complete([{"role": "user", "content": prompt}])
        return getattr(msg, "content", None) or ""


async def complete_cancellable(llm: Any, prompt: str, cancel_event: asyncio.Event, *, char_cap: int = 6000) -> str | None:
    """Bounded, cancellable text completion. `char_cap` enforces the machine-safety
    context bound (never an unattended long-context prefill) BEFORE the call; races via
    `run_cancellable`. `llm` is any object with async `.complete(prompt)->str`
    (`LLMTextAdapter` in prod, the fake doubles in tests)."""
    prompt = prompt[:char_cap]
    return await run_cancellable(llm.complete(prompt), cancel_event)
