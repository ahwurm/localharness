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

from localharness.memory.clustering import Cluster, _depth_from_tags, find_stable_clusters
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
                             refresh_overlap: float, claimed: set[str],
                             chapters: dict[int, tuple[str, frozenset[int]]] | None = None,
                             prefer_key: str | None = None) -> str:
    """CHAPTER REFRESH (run-14 fix): chapter identity must survive membership drift. The fresh
    key is h8(exact member set), so a cluster that gained/lost ANY member (a rescued residue
    atom, a correction row entering the pool, a foldout) would mint a near-identical SIBLING
    beside the still-active old chapter (run 14: three duplicate pairs; the stale sibling failed
    B4). If the new cluster's member overlap with an existing ACTIVE chapter — |∩| / min(|old|,
    |new|) — clears `refresh_overlap`, ADOPT that chapter's key: store_fact then supersedes on
    the old key (one active chapter, history preserved), the supersede-never-duplicate law one
    level up. At most one adoption per key per pass (`claimed`): a facet SPLIT of an old chapter
    keeps both facets — the second claimant falls back to its fresh key.

    Overlap is scored on the SHARED active-primary member map (`chapters`, threaded from the pass;
    built here if absent), the SAME sets the containment guard compares — NOT raw member_of. Before
    #67 the old side counted raw member_of (dead members + aux tier:surprising_failure rows) while the
    candidate side counted primary-only, so aux inflated the min() denominator and a legitimate refresh
    scored below threshold -> a duplicate sibling minted beside the still-active original.

    #64 (identity heal): on the recheck path `prefer_key` is the ORIGINAL chapter being healed. Its own
    surviving-member set self-scores 1.0, and a near-duplicate can tie at 1.0 — so adoption HARD-PREFERS
    the original: a DIFFERENT chapter's key is adopted only when its overlap is STRICTLY GREATER than the
    original's. A tie therefore keeps the heal on the original's key (store_fact same-key supersede,
    history preserved) instead of misattributing the healed content to another chapter's identity."""
    new_ids = {m.id for m in members}
    if chapters is None:
        chapters = await _active_chapter_primary_members(store)
    prefer_ov = 0.0
    best_key, best_ov = None, 0.0
    for _sid, (skey, mem) in chapters.items():
        if not mem:
            continue
        ov = len(mem & new_ids) / min(len(mem), len(new_ids))
        if prefer_key is not None and skey == prefer_key:
            prefer_ov = ov               # the original is the fallback identity, never an "other"
            continue
        if skey in claimed:
            continue
        if ov > best_ov:
            best_ov, best_key = ov, skey
    # #64 hard-prefer: adopt a DIFFERENT chapter only when STRICTLY better than the original's overlap
    # (prefer_ov is 0.0 on the write path, where prefer_key is None -> unchanged best-overlap behavior).
    strictly_better = prefer_key is None or best_ov > prefer_ov
    if best_key is not None and best_ov >= refresh_overlap and strictly_better and best_key != fresh_key:
        log.info("chapter-writer refresh: adopting %s (overlap %.2f) — supersede, not sibling",
                 best_key, best_ov)
        claimed.add(best_key)
        return best_key
    if prefer_key is not None and prefer_key != fresh_key:
        claimed.add(prefer_key)          # heal keeps the original's identity (tie or no better other)
        return prefer_key
    claimed.add(fresh_key)
    return fresh_key


async def _active_chapter_primary_members(store) -> dict[int, tuple[str, frozenset[int]]]:
    """Every ACTIVE chapter for this agent -> (key, its ACTIVE-PRIMARY member-id set). Built ONCE per
    pass and threaded into BOTH the containment guard and refresh adoption (#71) so every comparison
    uses the SAME member sets. PRIMARY = member_of dsts that are still ACTIVE (#66 — a superseded member
    is dead, never counted) MINUS aux tier:surprising_failure rows: those failure-telemetry rows are attached
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
            "AND f.status = 'active' "                        # #66: superseded members are DEAD, never counted —
            f"AND f.tags NOT LIKE '%\"{_FAILURE_TIER}\"%'",   # a chapter that lost members must compare by its
            (sid,),                                           # LIVE set (both fold+supersede AND adoption overlap).
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
                     containment_guard: bool = True, containment_counts: dict | None = None,
                     exclude_chapter_ids: frozenset[int] = frozenset(),
                     member_map: dict[int, tuple[str, frozenset[int]]] | None = None,
                     prefer_key: str | None = None):
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

    # SHARED active-primary member map (#66/#67/#71): built ONCE per pass and threaded in; derived here
    # only for a direct call (member_map=None). The SAME map feeds the containment guard AND adoption so
    # every set-comparison uses identical semantics (active-only, aux-excluded). It is a per-pass SNAPSHOT:
    # on the write path clusters are disjoint components (a stale entry is disjoint from any later candidate
    # -> inert); on the recheck path the healed chapter keeps its key (adoption re-keys it) and #70 re-reads
    # status, so staleness never mis-routes an identity.
    chapters = member_map if member_map is not None else await _active_chapter_primary_members(store)

    # CHAPTER CONTAINMENT GUARD — the write-time sibling of clustering's (component-time) incest
    # guard. Compare this candidate's PRIMARY member set (aux-excluded, like-for-like) against every
    # OTHER active chapter's. A chapter whose key == fresh_key is THIS candidate's own identity (an
    # idempotent re-derivation / the _adopt_refresh_key target) — store_fact corroborates/supersedes
    # it on the shared key, so it is never a self-fold or self-supersede.
    contained_ids: list[int] = []
    if containment_guard:
        member_ids = frozenset(m.id for m in members)
        for eid, (ekey, existing) in chapters.items():
            # STALENESS RE-DRAFT (exclude-self): the chapter being refreshed is EXCLUDED from the
            # containment comparison. Its surviving primary members equal the re-draft's members, so
            # without this the guard would FOLD the re-draft back into the very stale chapter it is
            # meant to replace — leaving the stale body active. Excluding it lets _adopt_refresh_key
            # (below) adopt the stale key and SUPERSEDE it (same-key refresh), the intended outcome.
            if eid in exclude_chapter_ids:
                continue
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
                         if eid not in exclude_chapter_ids
                         and ekey != fresh_key and existing and existing < member_ids]

    attempt.update(written=True, reason="written")
    key = await _adopt_refresh_key(store, members, fresh_key,
                                   refresh_overlap=refresh_overlap,
                                   claimed=claimed_keys if claimed_keys is not None else set(),
                                   chapters=chapters, prefer_key=prefer_key)
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
    # #71: derive the active-primary member map ONCE and thread it into every _write_one (guard +
    # adoption) instead of re-deriving all chapters' member sets per write (~O(chapters × writes)).
    member_map = await _active_chapter_primary_members(store)
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
                                      containment_counts=containment_counts,
                                      member_map=member_map)
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


# ---------------------------------------------------------------------------
# CHAPTER STALENESS RE-CHECK (d1-replication-20260712 §7 / B5) — evidentiary erosion
# ---------------------------------------------------------------------------
# A chapter can be grounded when written yet later carry a claim (esp. a figure) that NO active
# member supports, because a member atom was superseded out from under it (get_facts_by_ids is
# active-only). The eval's grader renders only ACTIVE members and KILLs such a chapter. This step
# runs before the writer each idle pass and closes that gap with the grader's OWN matchers.


def _bump(counts: dict | None, key: str) -> None:
    if counts is not None:
        counts[key] = counts.get(key, 0) + 1


async def _retire_chapter(store, chapter_id: int, *, successor: int | None = None) -> bool:
    """Retire a stale chapter: mark it non-active (status='superseded') and demote its retrieval_strength.
    `successor` sets superseded_by (#64 — when the heal replaced the chapter under a DIFFERENT key, the
    retired original must forward to its replacement, never dead-end at superseded_by=NULL); default None
    is the plain retire (nothing replaces it — the <2-members / refused-redraft arms). APPEND-ONLY: the
    row stays queryable via get_fact_history, never deleted. Returns False (no-op) if the row is already
    non-active — so a chapter the re-draft already superseded on its own key is never double-processed."""
    assert store._db is not None
    cur = await store._db.execute(
        "UPDATE facts SET status = 'superseded', updated_at = ?, superseded_by = ?, "
        "retrieval_strength = MIN(retrieval_strength, 0.1) "
        "WHERE id = ? AND agent_id = ? AND status = 'active'",
        (int(time.time()), successor, chapter_id, store._agent_id),
    )
    await store._db.commit()
    return cur.rowcount > 0


async def _active_members(store, chapter_id: int) -> list["Fact"]:
    """The chapter's CURRENT active members — grader-equivalent (mirrors sema05 `_schema_members`):
    the depth-1 neighborhood resolved through get_facts_by_ids, which is active-only, so a superseded
    member silently drops (exactly the B5 mechanism). Both the writer's fold-out members and any
    consumed aux tier:surprising_failure rows are member_of dsts, so both surface here (the numeric
    net grades against all of them, as the writer/grader corpus does)."""
    nb = await store.neighborhood(chapter_id, depth=1)
    ids = [nid for nid, _d in nb if nid != chapter_id]
    return await store.get_facts_by_ids(ids)


def _stale_attempt(chapter, n_members: int, *, reason: str) -> dict:
    """A forensic attempts_log entry for a retire that never reaches _write_one (ANALYSIS §7: every
    disposition must leave a trail). Field-compatible with _attempt_entry so it renders alongside the
    writer's own attempts; `staleness_recheck_of` labels it as re-check activity."""
    return {
        "key": chapter.key, "value": chapter.value[:300],
        "grounded": False, "grounded_majority": None, "unverified_numbers": [],
        "written": False, "reason": reason, "members": n_members,
        "sessions": [], "depth": _depth_from_tags(chapter.tags),
        "staleness_recheck_of": chapter.key,
    }


async def _recheck_one(store, llm, cancel_event, chapter, *, corpus_char_cap, refresh_overlap,
                       containment_guard, containment_counts, attempts_log, counts, claimed_keys,
                       member_map=None):
    """Re-validate ONE active chapter against its current active members; heal or retire on a fail."""
    members = await _active_members(store, chapter.id)
    bodies = [m.value for m in members]
    derefs = await _dereference(store, members)
    corpus = "\n".join(bodies + derefs)[:corpus_char_cap]
    body = strip_chapter_title(chapter.value)
    # The grader's exact verdict: majority-token grounding AND a clean numeric net, body-only.
    if grounded(body, corpus) and not ground_numbers(body, bodies):
        # #68 (starvation): advance the recheck cursor. The re-check window is ORDER BY updated_at ASC
        # LIMIT cap, so a revalidation that wrote NOTHING left the same <=cap grounded chapters filling
        # every window forever — erosion OUTSIDE them never detected. Bump updated_at (freshness only,
        # mirroring _corroborate_chapter; a re-confirmed chapter earning freshness is consistent with the
        # fold-touch) so the chapter rotates to the BACK. Content/trust/rs/id/history all untouched.
        await _corroborate_chapter(store, chapter.id)
        _bump(counts, "revalidated")
        return  # healthy — content unchanged (idempotency); only the recheck cursor advances

    # STALE. Only substantive (non-failure-telemetry) members can seed a re-draft; aux
    # tier:surprising_failure rows are opportunistic and never primary membership.
    primary = [m for m in members if _FAILURE_TIER not in m.tags]
    if len(primary) < 2:
        # < 2 active members: nothing to re-draft from -> retire (append-only).
        if await _retire_chapter(store, chapter.id):
            _bump(counts, "retired")
            if attempts_log is not None:
                attempts_log.append(_stale_attempt(chapter, len(members),
                                                   reason="stale_retired_insufficient_members"))
        return

    # ONE re-draft through the existing writer path on the surviving members (ALL guards apply:
    # hallucination kill, numeric net, containment). This chapter is EXCLUDED from the containment
    # comparison so the re-draft SUPERSEDES it on its own key rather than folding back into it.
    orig_depth = _depth_from_tags(chapter.tags)
    sessions = (frozenset(chapter.provenance[len("cluster:"):].split("|"))
                if chapter.provenance.startswith("cluster:") else frozenset())
    cluster = Cluster(members=sorted(primary, key=lambda m: m.key),
                      sessions=sessions, depth=max(0, orig_depth - 1), aux_members=[])
    schema = await _write_one(
        store, llm, cancel_event, cluster, orig_depth, corpus_char_cap,
        attempts_log=attempts_log, refresh_overlap=refresh_overlap, claimed_keys=claimed_keys,
        containment_guard=containment_guard, containment_counts=containment_counts,
        exclude_chapter_ids=frozenset({chapter.id}), member_map=member_map,
        prefer_key=chapter.key,   # #64: adoption HARD-PREFERS the chapter being healed on an overlap tie
    )
    if attempts_log:  # label the re-draft attempt _write_one just appended as re-check activity
        attempts_log[-1]["staleness_recheck_of"] = chapter.key
    if cancel_event.is_set():
        return  # cancelled mid-generation — leave the chapter untouched, catch it next pass
    if schema is not None:
        # Grounded re-draft written. If it adopted THIS chapter's key (the hard-prefer default, #64),
        # store_fact already superseded it with a successor and this retire is a no-op. If adoption chose
        # a DIFFERENT key (only when strictly better), the original is still active — retire it WITH the
        # successor so its history forwards to the replacement, never dead-ending at superseded_by=NULL.
        await _retire_chapter(store, chapter.id, successor=schema.id)
        _bump(counts, "redrafted")
    else:
        # Re-draft refused/failed/folded elsewhere (its rejection is already in attempts_log) ->
        # retire the stale chapter so it can never KILL the grade. Append-only.
        if await _retire_chapter(store, chapter.id):
            _bump(counts, "retired")


async def recheck_stale_chapters(
    store, llm, cancel_event, *,
    cap: int = 10, corpus_char_cap: int = 6000, refresh_overlap: float = 0.7,
    containment_guard: bool = True, containment_counts: dict | None = None,
    attempts_log: list | None = None, counts: dict | None = None,
) -> None:
    """Re-validate active chapters against their CURRENT active members and heal the stale ones.

    For up to `cap` active chapters (oldest-touched first — updated_at ASC, id ASC — a re-drafted
    chapter's fresh updated_at rotates it to the back): re-run the grader's grounded()+ground_numbers()
    matchers. Grounded -> untouched. Stale with >= 2 active members -> ONE re-draft on the survivors
    through the writer path (grounded -> supersede on the chapter's key, history preserved). Stale
    with < 2 members, or a re-draft the writer refuses -> retire (mark non-active, append-only —
    NEVER deleted). Idempotent on healthy chapters (writes nothing). Never raises into the idle pass;
    the single re-draft generation is cancellable. `counts` accumulates revalidated/redrafted/retired."""
    assert store._db is not None
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts "
        "WHERE agent_id = ? AND status = 'active' AND node_kind = 'schema' "
        "ORDER BY updated_at ASC, id ASC LIMIT ?",
        (store._agent_id, cap),
    ) as cur:
        chapters = [_row_to_fact(r) for r in await cur.fetchall()]

    claimed_keys: set[str] = set()  # one identity adoption per key per pass (facet-split safe)
    # #71: one active-primary member map for the whole re-check pass, threaded into each heal's guard
    # + adoption (a per-pass snapshot; #70 re-reads each item's status so a mid-pass supersede is skipped).
    member_map = await _active_chapter_primary_members(store)
    for chapter in chapters:
        if cancel_event.is_set():
            return
        try:
            await _recheck_one(
                store, llm, cancel_event, chapter,
                corpus_char_cap=corpus_char_cap, refresh_overlap=refresh_overlap,
                containment_guard=containment_guard, containment_counts=containment_counts,
                attempts_log=attempts_log, counts=counts, claimed_keys=claimed_keys,
                member_map=member_map)
        except Exception:
            log.exception("chapter staleness re-check failed for %s (non-fatal)", chapter.key)
