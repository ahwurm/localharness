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
<0.7 clamp (the schema is the visibility artifact, per the iteration-1 CONTEXT ruling). The idle-work
staleness probe is left untouched: the drain rides passes triggered by real work, never re-pinning it.

Never raises into the idle pass (every per-cluster body + the drain is guarded); all LLM work is
cancellable and char-bounded (machine-safety: this box hard-hung twice under long-context prefill).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING

from localharness.memory.clustering import find_stable_clusters
from localharness.memory.idle_llm import (
    complete_cancellable,
    ground_numbers,
    grounded,
    strip_chapter_title,
)
from localharness.memory.sqlite import (
    SCHEMA_CONFIDENCE,
    SCHEMA_DEPTH_TAG_PREFIX,
    SCHEMA_KEY_PREFIX,
    SCHEMA_TIER_TAG,
    _row_to_fact,
)

if TYPE_CHECKING:
    from localharness.memory.sqlite import Fact

log = logging.getLogger(__name__)

_DEMOTED_RS = 0.15   # mirrors consolidation._DEMOTED_RS: below the 0.2 index gate, still searchable
_PENDING_TAG = "pending_consolidation"
_CONSUMED_TAG = "tier:consumed"          # the queue-drain marker: folded under (or aged out beneath) a chapter
_FAILURE_TIER = "tier:surprising_failure"


def _h8(*parts: str) -> str:
    """Stable cluster identity over sorted member keys (mirrors predictive_write_gate._h8 /
    hierarchy._h8): a re-run over the same members SUPERSEDES the chapter, never duplicates."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:8]


def _consumed_tags(tags: list[str], *, stale: bool = False) -> str:
    """Drop pending_consolidation, add tier:consumed (+ a 'stale' breadcrumb when aged out).
    Mirrors consolidation._untag_candidate's raw tag edit; confidence is NEVER in the SET list —
    raw stat facts never promote above the <0.7 clamp (the schema is the visibility artifact)."""
    kept = [t for t in tags if t != _PENDING_TAG]
    if _CONSUMED_TAG not in kept:
        kept.append(_CONSUMED_TAG)
    if stale and "stale" not in kept:
        kept.append("stale")
    return json.dumps(kept)


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


def _attempt_entry(cluster, new_depth) -> dict:
    """Run-2 ruling 4 (observability): the base record for ONE chapter-writer attempt. Field
    names mirror the graders' per_schema_grounding entries (key/value/grounded/grounded_majority/
    unverified_numbers) so rejected attempts render everywhere written schemas do."""
    return {
        "key": f"(unwritten:{_h8(*sorted(m.key for m in cluster.members))})",
        "value": "",
        "grounded": None,
        "grounded_majority": None,
        "unverified_numbers": [],
        "written": False,
        "reason": None,
        "members": len(cluster.members),
        "sessions": sorted(cluster.sessions),
        "depth": new_depth,
    }


async def _adopt_refresh_key(store, members, fresh_key: str, *,
                             refresh_overlap: float, claimed: set[str]) -> str:
    """CHAPTER REFRESH (run-14 fix): chapter identity must survive membership drift. The fresh
    key is h8(exact member set), so a cluster that gained/lost ANY member (a rescued residue
    atom, a correction row entering the pool, a foldout) would mint a near-identical SIBLING
    beside the still-active old chapter (run 14: three duplicate pairs; the stale sibling failed
    B4). If the new cluster's member overlap with an existing ACTIVE chapter — |∩| / min(|old|,
    |new|) — clears `refresh_overlap`, ADOPT that chapter's key: store_fact then supersedes on
    the old key (one active chapter, history preserved), the supersede-never-duplicate law one
    level up. At most one adoption per key per pass (`claimed`): a facet SPLIT of an old chapter
    keeps both facets — the second claimant falls back to its fresh key."""
    new_ids = {m.id for m in members}
    assert store._db is not None
    best_key, best_ov = None, 0.0
    async with store._db.execute(
        "SELECT id, key FROM facts WHERE agent_id = ? AND status = 'active' "
        "AND node_kind = 'schema'", (store._agent_id,),
    ) as cur:
        rows = await cur.fetchall()
    for sid, skey in rows:
        if skey in claimed:
            continue
        async with store._db.execute(
            "SELECT dst_id FROM edges WHERE kind = 'member_of' AND src_id = ?", (sid,),
        ) as cur:
            mem = {r[0] for r in await cur.fetchall()}
        if not mem:
            continue
        ov = len(mem & new_ids) / min(len(mem), len(new_ids))
        if ov > best_ov:
            best_ov, best_key = ov, skey
    if best_key is not None and best_ov >= refresh_overlap and best_key != fresh_key:
        log.info("chapter-writer refresh: adopting %s (overlap %.2f) — supersede, not sibling",
                 best_key, best_ov)
        claimed.add(best_key)
        return best_key
    claimed.add(fresh_key)
    return fresh_key


async def _active_chapter_primary_members(store) -> dict[int, tuple[str, frozenset[int]]]:
    """Every ACTIVE chapter for this agent -> (key, its PRIMARY member-id set). PRIMARY = member_of
    dsts MINUS aux tier:surprising_failure rows: those failure-telemetry rows are attached
    OPPORTUNISTICALLY by _consume_aux (a failure adjacent to no chapter one pass lands under a
    different chapter the next), so they are non-deterministic noise a set-containment test must
    exclude — the LIVE root cause was exactly this (a chapter's 6 primary members were a strict
    subset of a 12-member one, but its raw member_of also held an aux row absent from the other, so
    the raw sets were NOT subsets). The candidate at guard time carries only its primary
    cluster.members, so both sides must be primary for a like-for-like compare. Edge convention
    mirrors clustering._chapter_member_ids (src=chapter, dst=member)."""
    assert store._db is not None
    async with store._db.execute(
        "SELECT id, key FROM facts WHERE agent_id = ? AND status = 'active' AND node_kind = 'schema'",
        (store._agent_id,),
    ) as cur:
        chapters = await cur.fetchall()
    out: dict[int, tuple[str, frozenset[int]]] = {}
    for sid, skey in chapters:
        async with store._db.execute(
            "SELECT e.dst_id FROM edges e JOIN facts f ON f.id = e.dst_id "
            "WHERE e.kind = 'member_of' AND e.src_id = ? "
            f"AND f.tags NOT LIKE '%\"{_FAILURE_TIER}\"%'",
            (sid,),
        ) as c2:
            out[sid] = (skey, frozenset(r[0] for r in await c2.fetchall()))
    return out


async def _supersede_chapter(store, old_id: int, new_id: int) -> bool:
    """Cross-key supersede for the containment guard: mark an existing chapter superseded BY the new
    one. Mirrors store_fact's internal supersede (status='superseded', superseded_by chain,
    retrieval_strength demoted to <=0.1) but across DIFFERENT keys — store_fact only supersedes on an
    identical key, and the subsumed chapter's key is NOT the one the candidate adopted. APPEND-ONLY:
    the row stays queryable via get_fact_history (never deleted). Returns True iff it was still
    active — a chapter already re-keyed away by the _adopt_refresh_key path this same write is
    skipped (the WHERE status='active' matches nothing, rowcount 0), so it is never double-counted."""
    assert store._db is not None
    cur = await store._db.execute(
        "UPDATE facts SET status = 'superseded', updated_at = ?, superseded_by = ?, "
        "retrieval_strength = MIN(retrieval_strength, 0.1) "
        "WHERE id = ? AND agent_id = ? AND status = 'active'",
        (int(time.time()), new_id, old_id, store._agent_id),
    )
    await store._db.commit()
    return cur.rowcount > 0


async def _corroborate_chapter(store, chapter_id: int) -> None:
    """FOLD touch: the candidate is a nested-duplicate VIEW of a survivor chapter, so instead of a
    twin we record a freshness touch on the survivor — an updated_at bump so the chapter it folds
    into does not decay out of the index for want of the refresh the (skipped) mint would otherwise
    have given it (this replaces the per-pass corroboration _adopt_refresh_key+store_fact applies on
    a normal re-derivation; mirrors store_fact's corroboration effect — freshness only, trust/rs
    untouched). Best-effort — a touch failure never blocks the fold."""
    assert store._db is not None
    try:
        await store._db.execute(
            "UPDATE facts SET updated_at = ? WHERE id = ? AND agent_id = ? AND status = 'active'",
            (int(time.time()), chapter_id, store._agent_id),
        )
        await store._db.commit()
    except Exception:
        log.debug("chapter containment: corroboration touch failed (non-fatal)", exc_info=True)


async def _write_one(store, llm, cancel_event, cluster, new_depth, corpus_char_cap,
                     attempts_log: list | None = None, *,
                     refresh_overlap: float = 0.7, claimed_keys: set[str] | None = None,
                     containment_guard: bool = True, containment_counts: dict | None = None):
    """Build a grounded, char-bounded corpus, generate ONE cancellable chapter, apply the
    grounding + numeric KILL, and (on pass) write the schema fact + member_of edges. Returns the
    schema Fact, or None (cancelled / ungrounded / unverified-figure) — the None cases ARE the
    pre-committed kill in action; the caller decides break (cancel) vs continue (kill). Every
    attempt (written or rejected, with its reason) is appended to attempts_log when provided —
    run-2 ruling 4: 'no chapter written' must leave a forensic trail, never an empty grading."""
    attempt = _attempt_entry(cluster, new_depth)
    if attempts_log is not None:
        attempts_log.append(attempt)
    members = cluster.members
    member_bodies = [m.value for m in members] + [a.value for a in cluster.aux_members]
    derefs = await _dereference(store, list(members) + list(cluster.aux_members))
    corpus = "\n".join(member_bodies + derefs)[:corpus_char_cap]

    prompt = (
        "Write ONE 1-2 sentence 'chapter' summarizing how this behaves, titled by the shared "
        "theme. COMPOSE THE BODY FROM THE LESSONS' OWN WORDING — reuse their exact terms and "
        "phrases; introduce no word, tool, name, or number that is not already in the lessons "
        "below. Assert ONLY what the lessons support.\n\n" + corpus
    )
    text = await complete_cancellable(llm, prompt, cancel_event, char_cap=corpus_char_cap)
    if text is None:
        attempt["reason"] = "cancelled" if cancel_event.is_set() else "generation_failed"
        return None
    text = text.strip()
    attempt["value"] = text[:300]
    if not text:
        attempt["reason"] = "empty_generation"
        return None

    # Pre-committed KILL (ROADMAP): a chapter whose tokens aren't derivable from its members is
    # worse than no chapter. FIX 1a: ground the BODY, not the markdown title — the prompt asks for
    # a titled chapter, so the model's "**Title**" heading tokens (never in the plain member
    # corpus) would otherwise sink the majority net (run-3 KILLed all 3 grounded drafts on titles).
    body = strip_chapter_title(text)
    if not grounded(body, corpus):
        log.info("chapter-writer: ungrounded chapter rejected (kill)")
        attempt.update(grounded=False, grounded_majority=False, reason="ungrounded")
        return None
    unverified = ground_numbers(body, member_bodies)
    if unverified:
        log.info("chapter-writer: chapter carries an unverified figure — rejected (kill)")
        attempt.update(grounded=False, grounded_majority=True,
                       unverified_numbers=unverified, reason="unverified_figures")
        return None
    attempt.update(grounded=True, grounded_majority=True)

    fresh_key = f"{SCHEMA_KEY_PREFIX}{_h8(*sorted(m.key for m in members))}"

    # CHAPTER CONTAINMENT GUARD — the write-time sibling of clustering's (component-time) incest
    # guard. Compare this candidate's PRIMARY member set (aux-excluded, like-for-like) against every
    # OTHER active chapter's. A chapter whose key == fresh_key is THIS candidate's own identity (an
    # idempotent re-derivation / the _adopt_refresh_key target) — store_fact corroborates/supersedes
    # it on the shared key, so it is never a self-fold or self-supersede.
    contained_ids: list[int] = []
    if containment_guard:
        member_ids = frozenset(m.id for m in members)
        chapters = await _active_chapter_primary_members(store)
        for eid, (ekey, existing) in chapters.items():
            if ekey == fresh_key:
                continue
            if member_ids and member_ids <= existing:
                # candidate ⊆ existing (incl. equality): a nested-duplicate VIEW — do NOT mint a
                # twin. Freshen the richer survivor (the mint we skip would have refreshed it).
                await _corroborate_chapter(store, eid)
                if containment_counts is not None:
                    containment_counts["folded"] = containment_counts.get("folded", 0) + 1
                log.info("chapter containment: candidate ⊆ chapter %s (%d ⊆ %d members) — folded, no twin",
                         ekey, len(member_ids), len(existing))
                attempt.update(written=False, reason="folded_containment")
                return None
        # E ⊊ M: existing chapters strictly inside the candidate — supersede each after the mint.
        contained_ids = [eid for eid, (ekey, existing) in chapters.items()
                         if ekey != fresh_key and existing and existing < member_ids]

    attempt.update(written=True, reason="written")
    key = await _adopt_refresh_key(store, members, fresh_key,
                                   refresh_overlap=refresh_overlap,
                                   claimed=claimed_keys if claimed_keys is not None else set())
    schema = await store.store_fact(
        key=key,
        value=text,
        tags=["schema", SCHEMA_TIER_TAG, f"{SCHEMA_DEPTH_TAG_PREFIX}{new_depth}"],
        confidence=SCHEMA_CONFIDENCE,
        source="chapter_writer",
        provenance=f"cluster:{'|'.join(sorted(cluster.sessions))}"[:200],
        node_kind="schema",
    )
    attempt["key"] = schema.key
    for m in members:
        try:
            await store.add_edge(schema.id, m.id, "member_of")
        except Exception:
            log.exception("chapter-writer: member_of edge failed (non-fatal)")
    await _fold_out_members(store, members)

    # Supersede EACH contained chapter the mint did not already re-key. _adopt_refresh_key re-keys at
    # most the single best-overlap chapter (store_fact supersedes THAT one on the adopted key); any
    # OTHER strictly-contained chapter — the live-stranded case — is superseded here. _supersede_
    # chapter skips one already re-keyed away (rowcount 0), so no double count. Append-only.
    if containment_guard and contained_ids:
        for eid in contained_ids:
            if eid == schema.id:
                continue
            if await _supersede_chapter(store, eid, schema.id):
                if containment_counts is not None:
                    containment_counts["superseded"] = containment_counts.get("superseded", 0) + 1
                log.info("chapter containment: superseded contained chapter id=%d under %s",
                         eid, schema.key)
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


async def _consume_aux(store, schema, aux_members) -> set[int]:
    """CLAIMED path (PGATE-03 rider): fold each aux tier:surprising_failure row under this chapter
    — member_of edge + drop pending_consolidation + add tier:consumed + demote rs. Confidence is
    NEVER touched (ruling: raw stat facts never promote above the <0.7 clamp; the schema is the
    visibility artifact). Returns the consumed ids so the drain sweep skips them. Guards each row."""
    assert store._db is not None
    consumed: set[int] = set()
    for a in aux_members:
        try:
            await store.add_edge(schema.id, a.id, "member_of")
            await store._db.execute(
                "UPDATE facts SET tags = ?, retrieval_strength = MIN(retrieval_strength, ?) WHERE id = ?",
                (_consumed_tags(a.tags), _DEMOTED_RS, a.id),
            )
            await store._db.commit()
            consumed.add(a.id)
        except Exception:
            log.exception("chapter-writer: aux consume failed for id=%s (non-fatal)", getattr(a, "id", "?"))
    return consumed


async def _drain_stale_failures(store, claimed_ids, *, stale_looks) -> int:
    """UNCLAIMED path: a tier:surprising_failure row adjacent to no stable cluster gets a per-key
    look counter bumped each pass; at stale_looks (default 5) it drains (drop pending_consolidation,
    add tier:consumed + a 'stale' breadcrumb, demote rs) so the quarantine queue cannot grow
    unboundedly. Thereafter it keeps fading under the existing decay step — never deleted, still
    searchable. Confidence is never touched. Deliberately does NOT re-add surprising_failure rows to
    the idle-work staleness probe (consolidation.py:558-561): that would resurrect the exact
    'pins the probe True forever' anti-pattern the codebase guards — the drain instead piggybacks on
    passes triggered by real work and touches no scheduling state. Returns the drained count."""
    from localharness.memory.consolidation import _get_meta, _set_meta  # LAZY (avoid import cycle)

    assert store._db is not None
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts "
        "WHERE agent_id = ? AND status = 'active' "
        f"AND tags LIKE '%\"{_FAILURE_TIER}\"%' AND tags LIKE '%\"{_PENDING_TAG}\"%'",
        (store._agent_id,),
    ) as cur:
        queue = [_row_to_fact(r) for r in await cur.fetchall()]

    now = int(time.time())
    drained = 0
    for row in queue:
        if row.id in claimed_ids:
            continue  # folded under a chapter this pass — already consumed
        meta_key = f"failure/looks/{row.key}"
        try:
            raw = await _get_meta(store, meta_key)
            looks = int(raw) + 1 if raw else 1
        except (TypeError, ValueError):
            looks = 1
        await _set_meta(store, meta_key, str(looks))
        if looks >= stale_looks:
            await store._db.execute(
                "UPDATE facts SET tags = ?, retrieval_strength = MIN(retrieval_strength, ?), updated_at = ? WHERE id = ?",
                (_consumed_tags(row.tags, stale=True), _DEMOTED_RS, now, row.id),
            )
            await store._db.commit()
            drained += 1
    return drained


async def write_cluster_schemas(
    store, llm, cancel_event, *,
    min_cluster_size: int = 2, min_sessions: int = 2, write_budget: int = 3,
    depth_cap: int = 2, corpus_char_cap: int = 6000, stale_looks: int = 5,
    embedder=None, embed_sim: float = 0.55, refresh_overlap: float = 0.7,
    containment_guard: bool = True, containment_counts: dict | None = None,
    attempts_log: list | None = None,
) -> list["Fact"]:
    """Turn stable lesson clusters into grounded chapter schema nodes — the SEMA-02/03 write half.

    For up to write_budget stable clusters (biggest first): dereference the payload-thin members +
    aux into a char-bounded corpus, generate ONE cancellable chapter, apply the grounding KILL, and
    on pass write ONE schema (0.8, member_of edges) + fold the members out of the ambient index.
    Depth is capped at depth_cap (lesson -> chapter -> chapter-of-chapters, then refused). It is the
    tier:surprising_failure consumer: claimed aux rows are folded (confidence untouched) and unclaimed
    rows drain after stale_looks idle cycles. Never raises into the idle pass; all LLM work is bounded
    + cancellable (machine-safety). Returns the schemas written this pass (partial on cancel)."""
    clusters = await find_stable_clusters(
        store, min_cluster_size=min_cluster_size, min_sessions=min_sessions,
        embedder=embedder, embed_sim=embed_sim,  # tier-1: embedding leg (2-factor, never alone)
    )

    schemas: list["Fact"] = []
    claimed: set[int] = set()
    claimed_refresh_keys: set[str] = set()  # one identity adoption per key per pass (facet split safe)
    for cluster in clusters[:write_budget]:
        if cancel_event.is_set():
            break
        # SEMA-03 depth cap: lesson(0) -> chapter(depth:1) -> chapter-of-chapters(depth:2) -> stop.
        new_depth = cluster.depth + 1
        if new_depth > depth_cap:
            log.debug("chapter-writer: depth cap reached (%d > %d) — cluster refused", new_depth, depth_cap)
            if attempts_log is not None:
                attempts_log.append(dict(_attempt_entry(cluster, new_depth), reason="depth_cap"))
            continue
        try:
            schema = await _write_one(store, llm, cancel_event, cluster, new_depth,
                                      corpus_char_cap, attempts_log=attempts_log,
                                      refresh_overlap=refresh_overlap,
                                      claimed_keys=claimed_refresh_keys,
                                      containment_guard=containment_guard,
                                      containment_counts=containment_counts)
        except Exception:
            log.exception("chapter-writer: cluster write failed (non-fatal)")
            if attempts_log is not None and attempts_log and attempts_log[-1].get("reason") is None:
                attempts_log[-1]["reason"] = "error"
            continue
        if schema is None:
            if cancel_event.is_set():
                break     # cancelled mid-generation — stop further clusters (partial result)
            continue       # ungrounded / unverified-figure kill — skip this cluster, keep going
        schemas.append(schema)
        try:
            claimed |= await _consume_aux(store, schema, cluster.aux_members)
        except Exception:
            log.exception("chapter-writer: aux consume failed (non-fatal)")

    # The tier:surprising_failure drain rides EVERY pass (even one that wrote nothing) so unclaimed
    # rows age out — but it never re-pins the idle-work probe (it piggybacks on real-work passes).
    try:
        await _drain_stale_failures(store, claimed, stale_looks=stale_looks)
    except Exception:
        log.exception("chapter-writer: stale-failure drain failed (non-fatal)")
    return schemas
