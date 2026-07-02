"""Persisted gist/schema hierarchy (v2.0 HIER-02..04) — fuzzy-trace made durable.

The cruncher builds a lossy gist-over-verbatim tree every over-window run and throws it
away when the turn ends. This module gives that tree rows in the memory graph:

- one **schema node** per run (the natural group: this document/question),
- one **gist node** per reduce output, `member_of` the schema, `derived_from` the
  previous level's gists (matched by content identity — level 1's inputs are the leaf
  extracts, which deliberately get NO rows: hundreds of leaves per run would bloat the
  store; the episodic session log is their durable home),
- a **final-answer node** derived from the last level,
- the **number-provenance net extended to memory** (HIER-04): a figure in a gist that
  appears in none of its inputs is tagged `unverified-figures` — the DRM-lure class,
  flagged, never rejected.

Gists sit BELOW the injection confidence threshold: they route retrieval (the graph
neighborhood in `memory_search`), they don't occupy the injected block. Verbatim
pointers: provenance carries the parent session id (the episodic log holds the full
trace) + the granted ContentStore handles; honest residual — ContentStore is EPHEMERAL,
so the session id is the load-bearing pointer. Leaf isolation is untouched: persistence
reads only strings the cruncher already holds; no ContentStore access, no grant widening.
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from localharness.core.reduce import ReduceLevel
    from localharness.memory.sqlite import MemoryStore

log = logging.getLogger(__name__)

_GIST_CONFIDENCE = 0.6  # below the 0.7 injection threshold: gists route, they don't inject


def _h8(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:8]


def flag_unverified_figures(text: str, sources: list[str]) -> list[str]:
    """HIER-04: numeric tokens in `text` absent (normalized) from every source string.
    Reuses the shipped v0.5.1 cruncher number-net verbatim — one normalizer, two nets."""
    from localharness.agent.subagent import _cruncher_unverified_numbers

    return _cruncher_unverified_numbers(text, sources)


async def persist_gist_tree(
    store: "MemoryStore",
    *,
    question: str,
    leaf_extracts: list[str],
    trace: list["ReduceLevel"],
    final_answer: str,
    session_id: str,
    source_handles: list[str],
) -> int:
    """Persist one cruncher run's gist tree into the memory graph. Returns nodes written.
    Never raises into the cruncher (caller guards); every write goes through the
    supersede-safe, read-back-verified store_fact path."""
    # Run identity includes leaf CONTENT (whole-milestone critic B3): question + count
    # alone collided two different documents analyzed under the same question in one
    # session — Company B's gist tree silently superseded Company A's (the exact
    # cross-entity misattribution shape the number-net exists for).
    run = _h8(question, session_id or "", *[e[:500] for e in leaf_extracts[:8]])
    prov = f"session:{session_id};handles:{','.join(source_handles[:4])}"

    schema = await store.store_fact(
        key=f"schema/doc/{run}",
        value=f"Document-analysis run ({len(leaf_extracts)} sections): {question[:300]}",
        tags=["schema"],
        confidence=_GIST_CONFIDENCE,
        source="cruncher",
        provenance=prov,
        node_kind="schema",
    )
    written = 1

    # content → node idS of the PREVIOUS level's gists (level 1's inputs are leaf
    # extracts — no rows, so level-1 gists link only member_of the schema). LIST values
    # (critic M1): two same-level batches can reduce to byte-identical text (sparse
    # broad queries → "(nothing relevant)" boilerplate); a scalar dict silently dropped
    # the earlier twin's outgoing edge. Content match is a proxy, so ambiguity attaches
    # to ALL matching parents — a conservative superset, never a lost edge.
    prev_out_ids: dict[str, list[int]] = {}
    for lvl in trace:
        this_ids: dict[str, list[int]] = {}
        for i, out in enumerate(lvl.outputs):
            flags = flag_unverified_figures(out, lvl.batches[i])
            tags = ["gist", f"level:{lvl.level}"] + (["unverified-figures"] if flags else [])
            if flags:
                log.warning(
                    "gist L%d-%d carries %d figure(s) absent from its inputs: %s",
                    lvl.level, i, len(flags), ", ".join(flags[:5]),
                )
            gist = await store.store_fact(
                key=f"gist/{run}/L{lvl.level}-{i}",
                value=out,
                tags=tags,
                confidence=_GIST_CONFIDENCE,
                source="cruncher",
                provenance=prov,
                node_kind="gist",
            )
            written += 1
            await store.add_edge(gist.id, schema.id, "member_of")
            for item in lvl.batches[i]:
                for parent_id in prev_out_ids.get(item, []):
                    await store.add_edge(gist.id, parent_id, "derived_from")
            this_ids.setdefault(out, []).append(gist.id)
        prev_out_ids = this_ids

    if final_answer.strip():
        flags = flag_unverified_figures(final_answer, leaf_extracts)
        tags = ["gist", "final"] + (["unverified-figures"] if flags else [])
        final = await store.store_fact(
            key=f"gist/{run}/final",
            value=final_answer,
            tags=tags,
            confidence=_GIST_CONFIDENCE,
            source="cruncher",
            provenance=prov,
            node_kind="gist",
        )
        written += 1
        await store.add_edge(final.id, schema.id, "member_of")
        for parent_ids in prev_out_ids.values():
            for parent_id in parent_ids:
                await store.add_edge(final.id, parent_id, "derived_from")

    log.info("gist tree persisted: run=%s nodes=%d levels=%d", run, written, len(trace))
    return written
