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
import string
from collections.abc import Awaitable
from typing import Any

log = logging.getLogger(__name__)

_PUNCT = string.punctuation  # stripped from a token's edges before matching (FIX 1b)


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


def grounded(claim: str, corpus: str, *, min_token_len: int = 6) -> bool:
    """A generated claim is grounded iff a MAJORITY of its >=min_token_len tokens appear
    verbatim in the corpus (extracted from consolidation.py:295-301; critic M4: an
    any-single-token check was trivially passed by confabulations sharing one common word
    like "contains"). An empty-token claim (all short words) -> True (nothing to verify).

    FIX 1b: matching is case-folded and a token's leading/trailing punctuation is stripped, so
    a chapter's own title casing/markdown ("Listens." / "**Port") can no longer make a
    genuinely-derivable claim unmatchable (run-3: the majority net was case+punct-sensitive and
    KILLed every grounded draft). Strictly more permissive on case/punctuation only — a token
    absent from the corpus in EVERY case is still unmatched (the anti-hallucination intent holds).
    This is the BROAD kill gate; `ground_numbers` layers the narrower numeric net on top."""
    corpus = corpus.lower()
    tokens = [tok for t in claim.split() if len(tok := t.strip(_PUNCT).lower()) >= min_token_len]
    if not tokens:
        return True
    matched = sum(1 for t in tokens if t in corpus)
    return matched * 2 >= len(tokens)


def strip_chapter_title(text: str) -> str:
    """Return a chapter's BODY for grounding — drop a leading markdown title line and strip
    emphasis markers. The chapter-writer prompt asks for a *titled* chapter and the model renders
    a markdown heading (e.g. "**Port Configuration**"), whose words are a HEADING, not an asserted
    claim; counting them against the majority-token grounding bar is structurally unwinnable
    (run-3: all three otherwise-grounded drafts KILLed on their title tokens). Grounding the body
    only aligns the gate with what the chapter actually asserts. A single-line draft (no separate
    title line) is returned intact (only markers stripped) — we never strip the sole content line."""
    lines = text.splitlines()
    if len(lines) > 1:
        first = lines[0].strip()
        core = first.strip("*_#` ").strip()
        is_heading = bool(core) and (
            first.startswith("#")
            or (first.startswith("**") and first.endswith("**"))
            or (first.startswith("__") and first.endswith("__"))
            or (first.startswith("*") and first.endswith("*"))
        )
        if is_heading:
            lines = lines[1:]
    body = "\n".join(lines)
    for mark in ("*", "`", "#"):  # de-emphasize; grounded() strips any residual token-edge punct
        body = body.replace(mark, " ")
    return body


def ground_numbers(text: str, sources: list[str]) -> list[str]:
    """Numeric tokens in `text` absent from every source (hierarchy.flag_unverified_figures,
    which reuses the shipped cruncher number-net). A non-empty return == unverified figures;
    the caller must NOT write those tokens (the SEMA-05 kill is stricter than hierarchy.py's
    flag-don't-reject: for chapters/mined facts, reject)."""
    from localharness.memory.hierarchy import flag_unverified_figures

    return flag_unverified_figures(text, sources)
