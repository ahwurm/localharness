"""SEMA-02/03 write half (Phase 36, the chapter-writer) — one grounded chapter per cluster.

The CLS slow loop's WRITE half, and the FIRST time the harness's own model writes memory.
find_stable_clusters (36-01) hands us a recurring lesson cluster; we dereference the
payload-thin stat rows, build a char-bounded corpus, run ONE bounded + cancellable generation
through the serial inference gate (36-03), apply the pre-committed grounding KILL (no schema
token underivable from its members — a hallucinated chapter is worse than no chapter), write
ONE schema node (36-02 contract: node_kind='schema', confidence 0.8, key schema/cluster/*)
with a member_of edge to every member, and fold the members out of the ambient index so the
chapter REPLACES the pile ("100 lessons -> one line").

It is also PGATE-03-rider's ONLY consumer of the tier:surprising_failure queue: the aux_members
36-01 attached are folded under the chapter (member_of + a consumed tag) and unclaimed rows
drain after a bounded number of idle cycles — raw stat facts NEVER promote above the deliberate
<0.7 clamp (the schema is the visibility artifact, per the iteration-1 CONTEXT ruling). _has_work
is left untouched: the drain rides passes triggered by real work, it never re-pins the idle probe.

Never raises into the idle pass (every per-cluster body + the drain is guarded); all LLM work is
cancellable and char-bounded (machine-safety: this box hard-hung twice under long-context prefill).
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from localharness.memory.clustering import find_stable_clusters
from localharness.memory.idle_llm import complete_cancellable, ground_numbers, grounded
from localharness.memory.sqlite import (
    SCHEMA_CONFIDENCE,
    SCHEMA_DEPTH_TAG_PREFIX,
    SCHEMA_KEY_PREFIX,
    SCHEMA_TIER_TAG,
)

if TYPE_CHECKING:
    from localharness.memory.sqlite import Fact

log = logging.getLogger(__name__)

_DEMOTED_RS = 0.15   # mirrors consolidation._DEMOTED_RS: below the 0.2 index gate, still searchable


def _h8(*parts: str) -> str:
    """Stable cluster identity over sorted member keys (mirrors predictive_write_gate._h8 /
    hierarchy._h8): a re-run over the same members SUPERSEDES the chapter, never duplicates."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:8]


async def _dereference(store, rows: list["Fact"]) -> list[str]:
    """CONTEXT ruling 1: payload-thin stat facts (source='predictive_write_gate') carry only
    tool+day boilerplate; their real content lives in the event log, reachable via
    provenance=session_id. Best-effort deref, GRACEFUL DEGRADATION — a miss appends nothing,
    NEVER invents (kill discipline). Members AND aux take the SAME path (aux ARE these rows)."""
    out: list[str] = []
    for r in rows:
        if r.source != "predictive_write_gate" or not r.provenance:
            continue
        try:
            history = await store.get_history(session_id=r.provenance, message_types=None)
            for rec in history:
                content = str(rec.get("content", "")).strip()
                if content:
                    out.append(content)
                    break
        except Exception:
            continue  # an unreadable/trimmed event log is not fatal
    return out


async def _write_one(store, llm, cancel_event, cluster, new_depth, corpus_char_cap):
    """Build a grounded, char-bounded corpus, generate ONE cancellable chapter, apply the
    grounding + numeric KILL, and (on pass) write the schema fact + member_of edges. Returns the
    schema Fact, or None (cancelled / ungrounded / unverified-figure) — the None cases ARE the
    pre-committed kill in action; the caller decides break (cancel) vs continue (kill)."""
    members = cluster.members
    member_bodies = [m.value for m in members] + [a.value for a in cluster.aux_members]
    derefs = await _dereference(store, list(members) + list(cluster.aux_members))
    corpus = "\n".join(member_bodies + derefs)[:corpus_char_cap]

    prompt = (
        "Write ONE 1-2 sentence 'chapter' summarizing how this behaves, titled by the shared "
        "theme. Assert ONLY what the lessons below support — invent no new facts, tools, or "
        "numbers.\n\n" + corpus
    )
    text = await complete_cancellable(llm, prompt, cancel_event, char_cap=corpus_char_cap)
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    # Pre-committed KILL (ROADMAP): a chapter whose tokens aren't derivable from its members is
    # worse than no chapter. grounded() is the broad majority-token net; ground_numbers layers the
    # stricter numeric net (SEMA-05 rejects an unverified figure, unlike hierarchy's flag-only).
    if not grounded(text, corpus):
        log.info("chapter-writer: ungrounded chapter rejected (kill)")
        return None
    if ground_numbers(text, member_bodies):
        log.info("chapter-writer: chapter carries an unverified figure — rejected (kill)")
        return None

    schema = await store.store_fact(
        key=f"{SCHEMA_KEY_PREFIX}{_h8(*sorted(m.key for m in members))}",
        value=text,
        tags=["schema", SCHEMA_TIER_TAG, f"{SCHEMA_DEPTH_TAG_PREFIX}{new_depth}"],
        confidence=SCHEMA_CONFIDENCE,
        source="chapter_writer",
        provenance=f"cluster:{'|'.join(sorted(cluster.sessions))}"[:200],
        node_kind="schema",
    )
    for m in members:
        try:
            await store.add_edge(schema.id, m.id, "member_of")
        except Exception:
            log.exception("chapter-writer: member_of edge failed (non-fatal)")
    await _fold_out_members(store, members)
    return schema


async def _fold_out_members(store, members) -> None:
    """Demote each folded member below the 0.2 index gate (retrieval_strength -> _DEMOTED_RS) so
    the chapter REPLACES the pile in the ambient block — the member stays active + FTS-searchable
    (a raw UPDATE of a non-indexed column only). Mirrors consolidation._untag_candidate /
    consolidation._DEMOTED_RS=0.15. Confidence (trust) is never touched — only accessibility."""
    assert store._db is not None
    for m in members:
        await store._db.execute(
            "UPDATE facts SET retrieval_strength = MIN(retrieval_strength, ?) WHERE id = ?",
            (_DEMOTED_RS, m.id),
        )
    await store._db.commit()


async def write_cluster_schemas(
    store, llm, cancel_event, *,
    min_cluster_size: int = 2, min_sessions: int = 2, write_budget: int = 3,
    depth_cap: int = 2, corpus_char_cap: int = 6000, stale_looks: int = 5,
) -> list["Fact"]:
    """Turn stable lesson clusters into grounded chapter schema nodes — the SEMA-02/03 write half.

    For up to write_budget stable clusters (biggest first): dereference the payload-thin members +
    aux into a char-bounded corpus, generate ONE cancellable chapter, apply the grounding KILL, and
    on pass write ONE schema (0.8, member_of edges). Never raises into the idle pass; all LLM work
    is bounded + cancellable (machine-safety). Returns the schemas written this pass (partial on
    cancel). (Depth cap + member fold-out land in Task 2; aux consume + drain in Task 3.)"""
    clusters = await find_stable_clusters(
        store, min_cluster_size=min_cluster_size, min_sessions=min_sessions
    )

    schemas: list["Fact"] = []
    for cluster in clusters[:write_budget]:
        if cancel_event.is_set():
            break
        # SEMA-03 depth cap: lesson(0) -> chapter(depth:1) -> chapter-of-chapters(depth:2) -> stop.
        new_depth = cluster.depth + 1
        if new_depth > depth_cap:
            log.debug("chapter-writer: depth cap reached (%d > %d) — cluster refused", new_depth, depth_cap)
            continue
        try:
            schema = await _write_one(store, llm, cancel_event, cluster, new_depth, corpus_char_cap)
        except Exception:
            log.exception("chapter-writer: cluster write failed (non-fatal)")
            continue
        if schema is None:
            if cancel_event.is_set():
                break     # cancelled mid-generation — stop further clusters (partial result)
            continue       # ungrounded / unverified-figure kill — skip this cluster, keep going
        schemas.append(schema)
    return schemas
