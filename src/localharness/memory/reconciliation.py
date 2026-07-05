"""PGATE-03 deferred half — the idle model-look that reconciles the correction_pending
quarantine (tri-outcome); reverts are the COMMON case (0.250 measured precision) and the
quarantine-only shape is the majority. Never raises into the pass; the LLM look is
cancellable + serial-inference-gate-safe (all model work routes through idle_llm).

This module is the ONLY consumer that can clear `tier:correction_pending`: consolidation's
promote-recurring EXCLUDES that tier (commit 3747216), so a disputed row lives forever
unless reconciliation resolves it. The live gate (predictive_write_gate.py:223-249) writes
disputes in TWO shapes, and every disposition here resolves so NO row re-surfaces forever:

  SHAPE (a) supersede-WRAP on an existing key — the wrapped value is the SUSPECT the user
    corrected; get_fact_history holds the pre-dispute antecedent to restore.
  SHAPE (b) quarantine-ONLY fact on a fresh `correction/quarantine/{h8}` key (the MEASURED
    majority — only 7 suspects in the whole 34 trace); the user's own words, NO antecedent.

Tri-outcome per fact: CONFIRM (shape-aware finalize), REVERT (restore shape a / clear shape
b), or UNDECIDED (stays quarantined under a per-key look-count/TTL bound). Reconciliation
NEVER promotes (every write stays < 0.7); the normal consolidation promote-path elevates a
settled fact later.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from localharness.memory.idle_llm import complete_cancellable, grounded

if TYPE_CHECKING:
    from localharness.memory.sqlite import Fact, MemoryStore

log = logging.getLogger(__name__)

# Mirror predictive_write_gate exactly (local copies keep that file an untouched, uncoupled
# dependency — the empty-diff discipline the gate's sibling design depends on).
_DISPUTE_MARKER = "[disputed by user correction — pending reconciliation]"  # shape (a) wrap
_QUARANTINE_PREFIX = "user correction (pending reconciliation): "           # shape (b) value prefix
_RETIRE_RS = 0.15  # demote a retired / cleared / stale row out of the retrieval competition


@dataclass
class ReconcileReport:
    confirmed: int = 0            # shape (b): user's own statement settled (prefix-stripped)
    confirmed_corrected: int = 0  # shape (a): grounded corrected value written over the wrong suspect
    retired: int = 0              # shape (a): CONFIRM with no derivable value — suspect retired (retag + rs 0.15)
    reverted_restored: int = 0    # shape (a): a pre-dispute antecedent was restored active
    reverted_cleared: int = 0     # shape (b) / all-disputed: nothing to restore, retagged out of the queue
    undecided: int = 0
    cancelled: bool = False


def _strip_dispute_prefix(value: str) -> str:
    """Strip a leading dispute marker or quarantine prefix — used for prompt display (both
    shapes) and for shape-(b)'s settled value. Shape-(a)'s settled value is NEVER derived by
    stripping: it is the model-derived corrected text, or a retire (see CONFIRM dispatch)."""
    if value.startswith(_DISPUTE_MARKER):
        return value[len(_DISPUTE_MARKER):].lstrip()
    if value.startswith(_QUARANTINE_PREFIX):
        return value[len(_QUARANTINE_PREFIX):]
    return value


def _settle_tags(fact: "Fact", new_tier: str) -> list[str]:
    """Drop the two queue tags and stamp the disposition tier (for the value-changing
    store_fact settlements — shape-(b) confirm and shape-(a) confirm-corrected)."""
    return [t for t in fact.tags
            if t not in ("tier:correction_pending", "pending_consolidation")] + [new_tier]


async def _retag_out(store: "MemoryStore", fact: "Fact", *, new_tier: str, provenance: str) -> None:
    """Drain a correction_pending row OUT of the queue with a RAW tag UPDATE — never store_fact
    (an identical-value store_fact hits the corroboration no-op branch, sqlite.py:666-679, and
    would leave the row tagged, re-surfacing forever). Also demotes retrieval_strength so the
    drained row leaves the retrieval competition. Mirrors consolidation._untag_candidate."""
    new_tags = _settle_tags(fact, new_tier)
    assert store._db is not None
    await store._db.execute(
        "UPDATE facts SET tags = ?, provenance = ?, retrieval_strength = MIN(retrieval_strength, ?) "
        "WHERE id = ?",
        (json.dumps(new_tags), provenance, _RETIRE_RS, fact.id),
    )
    await store._db.commit()


async def _revert(store: "MemoryStore", disputed: "Fact") -> str:
    """Reverse a false-positive correction. Returns "restored" (shape a: a pre-dispute
    antecedent was restored active — a clean third row in the chain, no delete, no new
    primitive) or "cleared" (shape b / all-disputed: nothing to restore — retagged OUT of the
    queue with a breadcrumb). NEVER a no-op: every REVERT resolves so no row re-surfaces forever.

    Pitfall 5: after stacked disputes the pre-dispute value is NOT history[1]; search history
    newest-first for the first version that is neither `_DISPUTE_MARKER`-prefixed NOR still
    tagged `tier:correction_pending` (the tag screen is what catches shape (b), whose sole
    row is prefix-different but IS correction_pending)."""
    history = await store.get_fact_history(disputed.key)  # newest first
    pre = next((v for v in history
                if not v.value.startswith(_DISPUTE_MARKER)
                and "tier:correction_pending" not in v.tags), None)
    if pre is not None:  # SHAPE (a): restore the pre-dispute value as a clean new row
        await store.store_fact(
            key=disputed.key, value=pre.value,  # value CHANGES -> no corroboration no-op
            tags=[t for t in pre.tags if not t.startswith("tier:")],
            confidence=pre.confidence, source="consolidation_reconciliation",
            provenance=f"revert-of:{disputed.provenance or disputed.key}",
        )
        return "restored"
    # SHAPE (b) quarantine-only key (or an all-disputed chain): nothing clean to restore. Retag
    # OUT so the queue DRAINS (BLOCKER 1) — a raw UPDATE via _retag_out, NOT store_fact (an
    # identical-value store_fact corroborate-no-ops and would leave it quarantined forever).
    await _retag_out(store, disputed, new_tier="tier:reconcile_cleared",
                     provenance=f"revert-cleared:{disputed.provenance or disputed.key}")
    return "cleared"


async def _correction_context(store: "MemoryStore", fact: "Fact", cap: int) -> str:
    """Best-effort dereference of the correction text behind a dispute: the gate stamped
    `provenance = event.id == user_signals.event_id`, and user_signals stores the FULL
    user_message look-ready. Skip on ANY miss — grounding falls back to the disputed value."""
    if not fact.provenance or store._db is None:
        return ""
    try:
        async with store._db.execute(
            "SELECT user_message, corrected_turn_summary FROM user_signals "
            "WHERE agent_id = ? AND event_id = ?",
            (store._agent_id, fact.provenance),
        ) as cur:
            row = await cur.fetchone()
    except Exception:
        return ""
    if not row:
        return ""
    return " ".join(x for x in row if x)[:cap]


def _confirm_prompt(shape_b: bool, disputed: str, ctx: str) -> str:
    """A compact 3-way instruction. Shape (b) asks CONFIRM/REVERT/UNDECIDED over the user's own
    statement; shape (a) asks the model to STATE the corrected value the fact should hold."""
    ctx_line = f"\nUser correction on record: {ctx}" if ctx else ""
    if shape_b:
        return (
            "Idle memory review. A user's own statement was quarantined pending review:\n"
            f'"{disputed}"{ctx_line}\n\n'
            "Answer EXACTLY one line, nothing else:\n"
            "CONFIRM — it is a genuine fact worth settling as remembered\n"
            "REVERT — it is noise / not worth remembering\n"
            "UNDECIDED — cannot tell"
        )
    return (
        "Idle memory review. A remembered fact was disputed by a user correction and is "
        f'pending review.\nDisputed (possibly-wrong) value: "{disputed}"{ctx_line}\n\n'
        "If the correction was RIGHT the disputed value is WRONG — state what the fact SHOULD "
        "say, using ONLY the correction above. Answer EXACTLY one line, nothing else:\n"
        "CONFIRM: <the corrected value the fact should now hold>\n"
        "REVERT — the correction was a false positive; the original fact was fine\n"
        "UNDECIDED — cannot tell"
    )


async def reconcile_corrections(
    store: "MemoryStore",
    llm: Any,
    cancel_event: asyncio.Event,
    *,
    ttl_looks: int = 3,
    corpus_char_cap: int = 4000,
) -> ReconcileReport:
    """Give every active `tier:correction_pending` fact a cancellable tri-outcome model-look
    and resolve BOTH gate shapes so the queue drains and no row re-surfaces forever. Never
    raises into the idle pass (each per-fact body is guarded, mirroring the consolidation step)."""
    from localharness.memory.consolidation import _get_meta, _set_meta
    from localharness.memory.sqlite import _row_to_fact

    report = ReconcileReport()
    if store._db is None:
        return report
    # Load the queue (mirror consolidation.py:154-163): active, tagged tier:correction_pending.
    async with store._db.execute(
        f"SELECT {store._FACT_COLS} FROM facts "
        "WHERE agent_id = ? AND status = 'active' "
        "AND tags LIKE '%\"tier:correction_pending\"%' "
        "ORDER BY created_at ASC, id ASC",
        (store._agent_id,),
    ) as cur:
        queue = [_row_to_fact(r) for r in await cur.fetchall()]

    for fact in queue:
        if cancel_event.is_set():  # a user turn is already waiting — stop the idle pass
            report.cancelled = True
            break
        try:
            shape_b = fact.key.startswith("correction/quarantine/")
            disputed = _strip_dispute_prefix(fact.value)
            ctx = await _correction_context(store, fact, corpus_char_cap)
            prompt = _confirm_prompt(shape_b, disputed, ctx)

            raw = await complete_cancellable(llm, prompt, cancel_event, char_cap=corpus_char_cap)
            if raw is None:  # cancelled mid-look (or the generation failed) — stop cleanly
                report.cancelled = True
                break
            answer = raw.strip()
            head = (answer.split(":", 1)[0].split() or [""])[0].upper()
            corrected = answer.split(":", 1)[1].strip() if ":" in answer else ""

            if head == "CONFIRM":
                if shape_b:
                    # The quarantined row IS the user's own statement — settle it with the prefix
                    # STRIPPED. `disputed != fact.value`, so store_fact writes a NEW clean row (not
                    # the corroboration no-op that would bake the prefix in); the marker-bearing
                    # audit row survives in get_fact_history.
                    await store.store_fact(
                        key=fact.key, value=disputed,
                        tags=_settle_tags(fact, "tier:reconcile_confirmed"),
                        confidence=fact.confidence,  # UNCHANGED, still < 0.7 (never promotes here)
                        source="consolidation_reconciliation",
                        provenance=f"confirm:{fact.provenance or fact.key}",
                    )
                    report.confirmed += 1
                elif corrected and grounded(corrected, f"{ctx} {disputed}"):
                    # Shape (a): the correction was RIGHT -> the suspect was WRONG. Write the
                    # model-derived CORRECTED value (grounded against the correction + disputed
                    # text), NEVER the stripped suspect — re-stating a corrected value is the
                    # bandshell failure class.
                    await store.store_fact(
                        key=fact.key, value=corrected,
                        tags=_settle_tags(fact, "tier:reconcile_confirmed"),
                        confidence=fact.confidence,
                        source="consolidation_reconciliation",
                        provenance=f"confirm-corrected:{fact.provenance or fact.key}",
                    )
                    report.confirmed_corrected += 1
                else:
                    # Shape (a) CONFIRM with no derivable/groundable value -> RETIRE the suspect
                    # (retag out of the queue + demote rs). The wrong value exits circulation, the
                    # queue drains, history keeps everything — and it is NEVER re-asserted settled.
                    await _retag_out(store, fact, new_tier="tier:reconcile_retired",
                                     provenance=f"retire:{fact.provenance or fact.key}")
                    report.retired += 1
            elif head == "REVERT":
                outcome = await _revert(store, fact)  # gated on the ACTUAL outcome — never a no-op
                if outcome == "restored":
                    report.reverted_restored += 1
                else:
                    report.reverted_cleared += 1
            else:  # UNDECIDED (or any unparseable answer) — stays quarantined under a TTL bound
                meta_key = f"reconcile/looks/{fact.key}"
                prev = await _get_meta(store, meta_key)
                looks = (int(prev) if prev and prev.isdigit() else 0) + 1
                await _set_meta(store, meta_key, str(looks))
                if looks >= ttl_looks:  # bounded queue: DRAIN permanently via a raw retag
                    await _retag_out(store, fact, new_tier="tier:reconcile_stale",
                                     provenance=f"stale:{fact.provenance or fact.key}")
                report.undecided += 1
        except Exception:  # one bad fact is non-fatal — the idle pass never raises out
            log.exception("reconcile_corrections per-fact look failed (non-fatal)")
    return report
