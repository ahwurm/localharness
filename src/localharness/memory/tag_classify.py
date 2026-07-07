"""Mint-time two-step filing (Amendment 4 / M1): a freshly-minted atom is filed by TWO
sequential single-choice CLOSED-SET picks over disjoint menus — (1) bucket from {personal,
project}, (2) child from that bucket's edge-eligible children, or "none". Both picks are
discrimination against a tiny shown menu with one-line definitions (the ergonomics-validated
shape, survey §3.4) — the model NAMES nothing here (Amendment 3). Parser-validated; garbage
degrades to bucket-only (if pick-1 was valid) or untagged — it NEVER blocks the mint (a tagging
failure degrades recall, never integrity). All LLM work routes through the cancellable,
char-capped idle path.
"""
from __future__ import annotations

import logging
from typing import Any

from localharness.memory.idle_llm import complete_cancellable

log = logging.getLogger(__name__)

# Distinctive prompt markers so an offline/scripted double can dispatch each pick deterministically.
_BUCKET_MARKER = "Sort this memory into ONE bucket"
_CHILD_MARKER = "Pick ONE tag for this memory"
_CLASSIFY_CHARS = 1400  # bounded prompt (machine-safety context cap; the menus are tiny)


def _norm(text: str) -> str:
    """A structured pick is one word — read the first non-empty token, lowercased and stripped of
    markup/punctuation, and validate it against the shown menu (never trust free-text beyond that)."""
    for raw in (text or "").strip().splitlines():
        w = raw.strip().strip("`*_#-.:>[]() \t").lower()
        if w:
            parts = w.split()
            return parts[0] if parts else ""
    return ""


def _menu(tags: list[Any]) -> str:
    return "\n".join(f"  {t.name} — {t.definition}" for t in tags)


async def file_atom_tags(
    store: Any, llm: Any, cancel_event: Any, *, atom_id: int, topic: str, claim: str,
    provenance: str = "mint",
) -> tuple[str | None, str | None]:
    """Two-step closed-set filing for one atom. Writes atom_tags rows (provenance='mint' at mint
    time; 'backfill' from the idle classify step, F4) for the chosen bucket and, if any, the
    chosen child. Returns (bucket_name|None, child_name|None) for observability. Never raises —
    a failure just leaves the atom less-tagged."""
    try:
        buckets = await store.buckets()
        if not buckets:
            return None, None
        by_bucket = {b.name: b for b in buckets}
        bucket_prompt = (
            f"{_BUCKET_MARKER}. Answer with EXACTLY one bucket name and nothing else.\n"
            + _menu(buckets)
            + f"\n\nmemory: topic={topic}; claim={claim}\nbucket:"
        )
        raw_b = await complete_cancellable(llm, bucket_prompt, cancel_event, char_cap=_CLASSIFY_CHARS)
        bucket = _norm(raw_b or "")
        if bucket not in by_bucket:
            return None, None  # invalid bucket -> untagged (the mint already happened)
        await store.add_atom_tag(atom_id, by_bucket[bucket].id, provenance)

        children = await store.active_children(by_bucket[bucket].id)
        if not children:
            return bucket, None
        by_child = {c.name: c for c in children}
        child_prompt = (
            f"{_CHILD_MARKER}, or answer 'none'. Answer with EXACTLY one name and nothing else.\n"
            + _menu(children)
            + "\n  none — none of the above fit\n"
            + f"\nmemory: topic={topic}; claim={claim}\ntag:"
        )
        raw_c = await complete_cancellable(llm, child_prompt, cancel_event, char_cap=_CLASSIFY_CHARS)
        child = _norm(raw_c or "")
        if child in by_child:
            await store.add_atom_tag(atom_id, by_child[child].id, provenance)
            return bucket, child
        return bucket, None  # 'none' or garbage -> bucket-only
    except Exception:
        log.warning("mint-time tag filing failed (non-fatal)", exc_info=True)
        return None, None
