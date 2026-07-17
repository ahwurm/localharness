"""Idle-cycle tag DISCOVERY (Stage C, Amendment 3/4): find NEW child tags over bucket-only atoms
by DETERMINISTIC multi-factor agreement, accrue Bayesian evidence with recency decay, and — at a
threshold — make ONE model call to NAME an already-bounded group. Responsibility split: SEED =
ours, CLASSIFY = model (mint), DISCOVER = this algorithm, NAME = model, TEND = curation. The model
never invents a boundary; its only creative act is labeling a group the algorithm already found —
the closed-ended operation a 27B does reliably.

Multi-factor grouping (the mega-blob lesson generalized: NO single signal may weld a group): a
pair of atoms is candidate-linked only if >= _MIN_FACTORS of three factors agree —
  (a) TEMPORAL: same sitting, or adjacent sittings (by session-start order);
  (b) EMBEDDING: cosine(embed(a), embed(b)) >= _EMBED_SIM (a pluggable embedder — ONE factor,
      never the mechanism); with no embedder this leg is absent and the other two must BOTH agree
      (the stricter 2-factor degrade);
  (c) TRACE: a and b co-fired in >= 1 activation trace (fire-together-wire-together, the P0 log).
Connected components of >= _FLOOR_MEMBERS atoms are candidate groups.

Evidence ladder (Amendment 4 — the old binary >=2-atom/>=2-sitting quarantine is the FLOOR of this
ladder, not a separate rule): each candidate accrues distinct_sittings + trace-reuse, the reuse
term DECAYED by recency (half-life _DECAY_HALF_LIFE_S). A candidate INCORPORATES when it clears the
floor (>= _FLOOR_MEMBERS members across >= _FLOOR_SITTINGS sittings) AND its evidence score
(distinct_sittings + decayed reuse) >= _INCORPORATE_SCORE: ONE NAME call, then stem-dedup against
existing tags (fold into the existing tag on a stem match), else a NEW active discovered child tag.
A candidate that stops accruing (unmatched, decayed reuse < _PRUNE_REUSE_FLOOR, stale) is PRUNED
(retired + members detached, back to the bucket-only pool) — synaptic pruning. All model work
routes through the cancellable, char-capped idle path; discovery never raises into the idle loop.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from localharness.memory.idle_llm import _stem, complete_cancellable

log = logging.getLogger(__name__)

_NAME_MARKER = "Name this group of related memories"

# --- deterministic constants (documented; tunable on the manifest A/B later) ---
_EMBED_SIM = 0.55            # cosine >= this counts the embedding-proximity factor
_MIN_FACTORS = 2             # a pair links only if >= 2 of {temporal, embedding, trace} agree
_FLOOR_MEMBERS = 2           # ladder floor: >= 2 atoms (the old quarantine floor)
_FLOOR_SITTINGS = 2          # ladder floor: >= 2 distinct sittings (recurrence, not one hot evening)
_INCORPORATE_SCORE = 3.0     # distinct_sittings + decayed_reuse must reach this to incorporate
_DECAY_HALF_LIFE_S = 14 * 86400   # reuse-evidence recency half-life (~2 weeks; ACT-R-flavoured)
_PRUNE_REUSE_FLOOR = 0.5     # decayed reuse below this (+ stale + unmatched) -> prune
_PRUNE_AGE_S = 21 * 86400    # a candidate is "stale" once this long past its last accrual
_MATCH_JACCARD = 0.5         # member-set overlap for a current group to be "the same" candidate
_NAME_MEMBERS_SHOWN = 8      # member values shown to the namer (bounded prompt)
# F5: a trace row that fired more atoms than this is a GENERIC probe (one rich mixed-topic
# retrieval must not clique-link the pool via temporal+trace with zero semantic agreement) —
# it contributes NO pair evidence. The same hub-guard shape as the tag-df cut, on trace rows.
_TRACE_FANOUT_CAP = 10


@dataclass
class DiscoveryReport:
    proposed: list[str] = field(default_factory=list)      # new candidate tags created this pass
    incorporated: list[str] = field(default_factory=list)  # candidates promoted to active (or folded)
    merged: list[str] = field(default_factory=list)        # candidates folded into an existing tag
    pruned: list[str] = field(default_factory=list)        # candidates retired


def _is_sitting(prov: str) -> bool:
    """clustering._is_sitting convention: a real sitting id carries no ':' (bookkeeping is 'x:y')."""
    return bool(prov) and ":" not in prov


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv) if nu and nv else 0.0


def _decay(reuse: float, last_ts: int | None, now: int) -> float:
    if not reuse or not last_ts:
        return float(reuse or 0)
    return reuse * (0.5 ** ((now - last_ts) / _DECAY_HALF_LIFE_S))


def _is_prunable(tag: Any, now: int) -> bool:
    """Synaptic-pruning predicate: a candidate that stopped accruing (stale past _PRUNE_AGE_S) AND
    whose decayed reuse fell below _PRUNE_REUSE_FLOOR is retired. Shared by the idle discovery pass
    and the turn-end micro-pass (#90) so both prune on the SAME rule (a candidate that would still
    match a live cluster is re-accrued within the stale window, resetting last_accrual_ts)."""
    decayed = _decay(tag.reuse_count, tag.last_accrual_ts, now)
    stale = tag.last_accrual_ts is not None and (now - tag.last_accrual_ts) > _PRUNE_AGE_S
    return stale and decayed < _PRUNE_REUSE_FLOOR


def _incorporation_eligible(tag: Any, member_count: int, now: int) -> bool:
    """The evidence-ladder gate discover_tags applies before it NAMEs a candidate, evaluated from a
    candidate's STORED evidence (#90): >= _FLOOR_MEMBERS members across >= _FLOOR_SITTINGS sittings
    AND (distinct_sittings + decayed reuse) >= _INCORPORATE_SCORE. Evidence rules UNCHANGED — the
    micro-pass only runs the NAME step that cancellation starves, never lowers the bar."""
    score = tag.distinct_sittings + _decay(tag.reuse_count, tag.last_accrual_ts, now)
    return (member_count >= _FLOOR_MEMBERS and tag.distinct_sittings >= _FLOOR_SITTINGS
            and score >= _INCORPORATE_SCORE)


def _clean_name(raw: str) -> str:
    """A NAME answer is one short tag — first non-empty line, 1-2 lowercase alnum/hyphen tokens."""
    for line in (raw or "").strip().splitlines():
        w = line.strip().strip("`*_#-.:>[]() \t").lower()
        if not w:
            continue
        toks = [re.sub(r"[^a-z0-9-]", "", t) for t in w.split()][:2]
        name = "-".join(t for t in toks if t)
        return name if 2 <= len(name) <= 40 else ""
    return ""


async def _sitting_rank(store: Any) -> dict[str, int]:
    """sitting id -> rank by session start time (for temporal ADJACENCY)."""
    assert store._db is not None
    async with store._db.execute(
        "SELECT id FROM sessions WHERE agent_id = ? ORDER BY started_at, id", (store._agent_id,)
    ) as cur:
        return {r[0]: i for i, r in enumerate(await cur.fetchall())}


async def _cofire_pairs(
    store: Any, atom_ids: set[int], *, injection_weight: float = 1.0
) -> dict[tuple[int, int], float]:
    """Unordered atom-id pairs that co-fired in >= 1 activation trace (both in fired_ids), mapped
    to a co-fire WEIGHT — a materialized-view discount over the fidelity-preserving log (owner
    reversal 2026-07-17). A retrieval-source co-fire (the model's own memory_search / memory_get)
    weighs 1.0; an ambient-INJECTION co-fire weighs `injection_weight` (our every-turn guess, a
    weaker signal). A pair co-firing under BOTH takes the MAX (retrieval dominates) — so with zero
    injection rows every weight is 1.0 and a weighted sum reduces to today's plain distinct-pair
    count (byte-identical). Rows over _TRACE_FANOUT_CAP are skipped (F5, the hub guard) — for
    injection rows too: a mega shelf is a generic probe, not pair evidence. Presence in the dict
    is still binary co-fire (weight > 0), so the multi-factor TRACE leg is unchanged; the discount
    lands on co-fire STRENGTH downstream. injection_weight=0.0 drops injection-only pairs entirely."""
    pairs: dict[tuple[int, int], float] = {}
    for tr in await store.recent_activation_traces(limit=500):
        if len(tr.fired_ids) > _TRACE_FANOUT_CAP:
            continue  # hub-stimulus guard: a mega-row must not clique-link the pool
        w = injection_weight if tr.source == "injection" else 1.0
        fired = [i for i in tr.fired_ids if i in atom_ids]
        for i in range(len(fired)):
            for j in range(i + 1, len(fired)):
                key = (min(fired[i], fired[j]), max(fired[i], fired[j]))
                if w > pairs.get(key, 0.0):  # max-source: retrieval (1.0) dominates injection
                    pairs[key] = w
    return pairs


def _components(ids: list[int], links: list[tuple[int, int]]) -> list[list[int]]:
    adj: dict[int, set[int]] = {i: set() for i in ids}
    for a, b in links:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        stack, comp = [i], []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps


async def _incorporate(store, llm, cancel_event, cand, group, bucket, report) -> None:
    """The candidate cleared the ladder: ONE NAME call, validate, then either fold into an existing
    tag (stem-dedup) or become a new active discovered child tag. Garbage/empty -> stays a
    candidate (accrues next cycle). The model's only creative act."""
    shown = "\n".join(f"- {m.value}" for m in group[:_NAME_MEMBERS_SHOWN])
    prompt = (f"{_NAME_MARKER} under the '{bucket.name}' category. Answer with ONE short lowercase "
              f"1-2 word tag name and nothing else.\n{shown}\nname:")
    name = _clean_name(await complete_cancellable(llm, prompt, cancel_event, char_cap=1600) or "")
    if not name:
        return
    siblings = [t for t in await store.list_tags()
                if t.parent_id == bucket.id and t.id != cand.id and t.status in ("seeded", "active")]
    target = next((t for t in siblings if _stem(t.name) == _stem(name) or t.name == name), None)
    if target is not None:  # stem-dedup: fold members into the existing tag
        await store.move_atom_tags(cand.id, target.id, provenance="discovery")
        await store.set_tag_status(cand.id, "merged", merged_into=target.id)
        report.merged.append(target.name)
        report.incorporated.append(target.name)
        return
    if await store.get_tag(name) is not None:
        return  # name taken by a NON-sibling (other bucket) -> avoid a unique-name collision; wait
    # Rename the candidate in place AND replace the "discovery candidate (unincorporated)"
    # placeholder definition (#90) — an incorporated tag joins the classify menu, so it needs a
    # real one-liner, not the placeholder that misled mint-time filing.
    await store.set_tag_status(cand.id, "active", name=name,
                               definition=f"discovered {bucket.name} tag: memories about {name}")
    report.incorporated.append(name)


async def discover_tags(store: Any, llm: Any, cancel_event: Any, *, embedder: Any,
                        injection_weight: float = 1.0,
                        now: int | None = None) -> DiscoveryReport:
    """One discovery pass over every bucket's bucket-only atoms. Returns a DiscoveryReport. Never
    raises into the idle loop. `embedder` may be None (stricter 2-factor temporal+trace degrade).
    `injection_weight` discounts ambient-injection co-fire vs model-initiated retrieval (owner
    reversal 2026-07-17) — the production seam passes agent.memory.consolidation.trace_injection_
    weight; the default 1.0 is the no-discount identity for un-threaded callers."""
    report = DiscoveryReport()
    if now is None:
        now = int(time.time())
    # F7 (run-9 forensics): name the embedder actually in play — MiniLM vs the HashingEmbedder
    # fallback must be distinguishable from logs alone.
    log.info("tag discovery: embedder=%s",
             type(embedder).__name__ if embedder is not None else "none")
    try:
        rank = await _sitting_rank(store)
        for bucket in await store.buckets():
            if getattr(cancel_event, "is_set", lambda: False)():
                break
            pool = await store.atoms_without_child_tag(bucket_id=bucket.id)
            by_id = {f.id: f for f in pool}
            ids = list(by_id)
            cofire = await _cofire_pairs(store, set(ids), injection_weight=injection_weight)
            vecs: dict[int, list[float]] = {}
            if embedder is not None and ids:
                vecs = dict(zip(ids, embedder.embed([by_id[i].value for i in ids])))

            # multi-factor pair links -> candidate groups (>= _FLOOR_MEMBERS)
            links: list[tuple[int, int]] = []
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    a, b = ids[x], ids[y]
                    fa, fb = by_id[a], by_id[b]
                    factors = 0
                    if fa.provenance and fb.provenance and (
                        fa.provenance == fb.provenance
                        or (fa.provenance in rank and fb.provenance in rank
                            and abs(rank[fa.provenance] - rank[fb.provenance]) == 1)):
                        factors += 1
                    if embedder is not None and _cosine(vecs[a], vecs[b]) >= _EMBED_SIM:
                        factors += 1
                    if (min(a, b), max(a, b)) in cofire:
                        factors += 1
                    if factors >= _MIN_FACTORS:
                        links.append((a, b))
            groups = [[by_id[i] for i in c] for c in _components(ids, links)
                      if len(c) >= _FLOOR_MEMBERS]

            # existing proposed candidates in this bucket (for matching + pruning)
            existing = [t for t in await store.list_tags(status="proposed")
                        if t.parent_id == bucket.id and t.origin == "discovered"]
            members_of = {t.id: {a.id for a in await store.atoms_for_tag(t.id)} for t in existing}
            matched: set[int] = set()

            for group in groups:
                mids = {m.id for m in group}
                sittings = {m.provenance for m in group if _is_sitting(m.provenance)}
                # Weighted co-fire STRENGTH: injection-source pairs contribute their discount,
                # retrieval-source their full 1.0 (the materialized-view weight; the log stays
                # ground truth). With no injection rows this is the old distinct-pair count.
                this_reuse = sum(w for (p, q), w in cofire.items() if p in mids and q in mids)
                cand = next((t for t in existing
                             if (mids & members_of[t.id])
                             and len(mids & members_of[t.id]) / len(mids | members_of[t.id])
                             >= _MATCH_JACCARD), None)
                if cand is None:
                    sig = hashlib.sha1("|".join(sorted(m.key for m in group)).encode()).hexdigest()[:8]
                    cand = await store.create_tag(f"cand-{sig}", "discovery candidate (unincorporated)",
                                                  parent_id=bucket.id, status="proposed",
                                                  origin="discovered")
                    report.proposed.append(cand.name)
                    existing.append(cand)
                    members_of[cand.id] = set()
                for m in group:
                    if m.id not in members_of[cand.id]:
                        await store.add_atom_tag(m.id, cand.id, "discovery")
                members_of[cand.id] |= mids
                matched.add(cand.id)

                prior = await store.get_tag_by_id(cand.id)
                new_reuse = _decay(prior.reuse_count, prior.last_accrual_ts, now) + this_reuse
                distinct_sittings = len(sittings)
                await store.bump_tag_evidence(cand.id, distinct_sittings=distinct_sittings,
                                              reuse_count=int(round(new_reuse)), last_accrual_ts=now)
                score = distinct_sittings + new_reuse
                if (len(mids) >= _FLOOR_MEMBERS and distinct_sittings >= _FLOOR_SITTINGS
                        and score >= _INCORPORATE_SCORE):
                    await _incorporate(store, llm, cancel_event, cand, group, bucket, report)

            # prune stale, unmatched candidates (synaptic pruning)
            for t in existing:
                if t.id in matched:
                    continue
                fresh = await store.get_tag_by_id(t.id)
                if fresh is None or fresh.status != "proposed":
                    continue
                if _is_prunable(fresh, now):
                    await store.set_tag_status(t.id, "retired")
                    await store.remove_atom_tags_for_tag(t.id)
                    report.pruned.append(t.name)
    except Exception:
        log.exception("tag discovery failed (non-fatal)")
    return report


async def _proposed_discovered(store: Any) -> list[Any]:
    """Proposed, discovery-origin candidates oldest-first (by tag id — monotonic creation order)."""
    cands = [t for t in await store.list_tags(status="proposed") if t.origin == "discovered"]
    cands.sort(key=lambda t: t.id)
    return cands


async def name_eligible_candidates(
    store: Any, llm: Any, cancel_event: Any, *, limit: int = 2,
    stop: Any = None, now: int | None = None,
) -> int:
    """Turn-end micro-pass unit (#90): NAME up to `limit` ALREADY-eligible proposed discovery
    candidates, oldest-first — the model-names-the-cluster step (`_incorporate`) that idle discovery
    accrues candidates toward but rarely reaches live (cancelled first; the audit's 35 candidates
    stuck at 'proposed' with the placeholder definition). ONE small model call per candidate, the
    evidence gate UNCHANGED (read from stored evidence). `stop()` (budget/cancel) is polled between
    candidates — a cancel between units loses nothing (a half-classified candidate stays proposed).
    Returns the count incorporated/folded. Never raises into the caller."""
    if now is None:
        now = int(time.time())
    report = DiscoveryReport()
    named = attempts = 0
    try:
        for cand in await _proposed_discovered(store):
            if attempts >= limit:
                break
            if (stop is not None and stop()) or getattr(cancel_event, "is_set", lambda: False)():
                break
            if cand.parent_id is None:
                continue
            bucket = await store.get_tag_by_id(cand.parent_id)
            if bucket is None:
                continue
            members = await store.atoms_for_tag(cand.id)
            if not _incorporation_eligible(cand, len(members), now):
                continue  # not yet warranted — idle discovery keeps accruing it (bar unchanged)
            attempts += 1  # a model call is spent on this candidate
            before = len(report.incorporated)
            await _incorporate(store, llm, cancel_event, cand, members, bucket, report)
            if len(report.incorporated) > before:
                named += 1
    except Exception:
        log.exception("micro-pass candidate naming failed (non-fatal)")
    return named


async def prune_stale_candidates(
    store: Any, *, now: int | None = None, limit: int | None = None,
) -> int:
    """Turn-end micro-pass unit (#90): retire stale, unincorporated discovery candidates — pure SQL,
    no model calls, the SAME `_is_prunable` rule discovery applies inline. Returns the count pruned.
    Never raises into the caller."""
    if now is None:
        now = int(time.time())
    pruned = 0
    try:
        for t in await _proposed_discovered(store):
            if limit is not None and pruned >= limit:
                break
            if _is_prunable(t, now):
                await store.set_tag_status(t.id, "retired")
                await store.remove_atom_tags_for_tag(t.id)
                pruned += 1
    except Exception:
        log.exception("micro-pass candidate prune failed (non-fatal)")
    return pruned
