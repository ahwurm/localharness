"""The PRIMARY semantic feeder (MOVE 2) — the idle transcript miner.

Domain knowledge about the USER'S WORLD (what they work on, prefer, plan, decide) enters the
semantic hierarchy HERE: an idle model-look walks the un-mined transcript and extracts TYPED
atoms — `(topic, claim, evidence-span)` — writing one `sem/{topic-slug}/{h8(claim)}` fact per
atom. Same-topic atoms share a key namespace, giving clustering natural handles (shared prefix +
FTS token + graph edge to the session). An atom that CORRECTS an existing active atom carries an
optional `replaces=<id>` marker (the miner is shown the active atoms in-prompt, capped) and lands
on the OLD key — supersede chain, history preserved — so a stale value never stays active beside
its correction (claim-hash keys alone can never collide; run-2 ruling 3). This is the feeder that was previously missing: mining
shipped as a budgeted rider and the orphaned replay seam wrote unreachable `replay/*` keys — the
two near-duplicate idle extractors are now ONE (the replay seam is retired into this walk).

Four bounds keep it honest and cheap (CONTEXT ruling 3, carried forward):
  1. WATERMARK-BOUNDED (cost): only history newer than "mining/last_ts" is read; the watermark
     advances per COMPLETED chunk, so a cancelled/over-budget pass resumes where it stopped and
     cost is per-window, never O(lifetime history).
  2. CHUNKED FULL-COVERAGE: the ENTIRE un-mined window is walked in `corpus_char_cap` chunks
     (a loop, not one nibble) so a long backlog is fully mined across a pass — one LLM look per
     chunk, cancellation-checked between chunks.
  3. GROUNDED (kill discipline): each atom is attributed to its SOURCE record (best token
     overlap of its claim+evidence) and the claim must pass `grounded` against that record's
     text — no token that isn't derivable from the cited span. Ungrounded atoms are rejected
     and counted, never written.
  4. BUDGETED: at most `write_budget` writes per pass. A budget-truncated chunk does NOT advance
     the watermark, so the next pass re-mines it (identical claims corroborate, never duplicate).

Load-bearing (research doc §4): provenance is the SOURCE record's `session_id` PER ATOM, not the
mining batch — the cluster stability bar counts distinct provenance sessions, so a batch-level
provenance would make every mined atom share one session and never satisfy the ≥2-session bar.

Write confidence is 0.65 — searchable but sub-injection: a single mention is not yet ambient;
ambient status is EARNED by distinct-day recurrence (the store's ladder) or chapter membership.

Machine-safety: ALL LLM work routes through `idle_llm.complete_cancellable` (cooperatively
cancellable + char-capped). This function NEVER raises into the idle scheduler.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from localharness.memory.consolidation import _get_meta, _set_meta
from localharness.memory.idle_llm import complete_cancellable, grounded
from localharness.memory.sqlite import FactQuery
from localharness.memory.tag_classify import file_atom_tags

log = logging.getLogger(__name__)

_MINING_WATERMARK_KEY = "mining/last_ts"  # DISTINCT from consolidation/last_run
_PER_RECORD_CHARS = 400   # per-record slice fed to the model (and grounded against)
_KNOWN_ATOMS_CAP = 30     # existing active atoms shown to the miner (ruling 3), value-trimmed

_PROMPT = (
    "Extract durable facts about the USER'S WORLD from this transcript — what the user works "
    "on, prefers, plans, or decides. One fact per line, formatted EXACTLY as pipe-separated "
    "fields:\n"
    "topic | claim | evidence\n"
    "where `topic` is a SHORT, GENERIC 1-3 word subject label (e.g. subagents, kyoto trip, gpu "
    "ops) — NOT a per-claim phrase, and NEVER a fact id or key. If a fact continues a topic "
    "already present in the known facts listed below, REUSE that fact's exact topic label "
    "verbatim instead of inventing a new one. `claim` is one self-contained fact, and `evidence` "
    "is a short VERBATIM quote from the transcript that supports it. If a fact CORRECTS or "
    "CONTRADICTS one of the known facts listed below on the same topic, append a fourth field to "
    "that line: | replaces=<the known fact's id>. ONLY facts explicitly stated in the "
    "transcript — no inference, no new numbers:\n\n"
)


def _h8(claim: str) -> str:
    """Stable 8-hex key suffix for a claim so an identical re-mined claim collapses onto the
    same `sem/` key and corroborates (store_fact) rather than duplicating."""
    return hashlib.sha1(claim.strip().encode("utf-8")).hexdigest()[:8]


def _slug(topic: str) -> str:
    """A topic namespace slug: lowercase, non-alphanumeric runs -> single hyphen, trimmed.
    Same-topic atoms therefore share the `sem/{slug}/` prefix — a natural clustering handle."""
    s = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return s or "misc"


def _slug_of_key(key: str) -> str:
    """The `<slug>` component of a `sem/<slug>/<hash>` key — recovers a real topic label from a
    key-shaped topic paste (falls back to _slug for a non-sem string)."""
    parts = key.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "sem" and parts[1] else _slug(key)


def _norm(value: str) -> str:
    """Whitespace-collapsed, lowercased value — the in-pass duplicate-mint identity (FIX 2)."""
    return " ".join(value.split()).lower()


def _coerce_key_topic(topic: str, known_keys: set[str]) -> tuple[str | None, str] | None:
    """FIX 2a (run-3): Qwen sometimes pastes a known atom's KEY (or its `sem/<slug>` prefix) into
    the `topic` field instead of using `replaces=<id>`, which _slug() mangles into a shadow-
    duplicate ('sem/sem-vllm-port-…') that leaves the stale value active beside its correction.
    Detect that paste against the known ACTIVE atoms and recover the real slug so no shadow key is
    minted. For an UNAMBIGUOUS full-key paste, also return the matched key as an implied `replaces`
    so a correction supersedes the atom it names. A prefix-only paste is AMBIGUOUS (usually a NEW
    fact on the topic, not a correction — run-3's summarizer id 57->62), so we recover the slug but
    do NOT supersede (superseding a distinct fact would lose data). Returns
    (implied_replaces_or_None, recovered_slug), or None when `topic` is not a key-shaped paste of a
    known active atom (-> normal mint, current behavior)."""
    if not topic.startswith("sem/"):
        return None
    for k in known_keys:  # exact / leading full-key paste -> implied supersede of that atom
        if topic == k or topic.startswith(k + "/") or topic.startswith(k + ":"):
            return k, _slug_of_key(k)
    for k in known_keys:  # `sem/<slug>` prefix paste -> slug recovery only (no supersede)
        if topic == k.rsplit("/", 1)[0]:
            return None, _slug_of_key(k)
    return None


def _tokens(text: str, *, min_len: int = 5) -> list[str]:
    """Content tokens >= min_len chars (drops stopwords/punctuation) — the attribution probe."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= min_len]


def _parse_atoms(raw: str) -> list[tuple[str, str, str, str | None]]:
    """Parse `topic | claim | evidence [| replaces=<id>]` lines into typed atoms; evidence falls
    back to the claim when omitted; `replaces` (ruling 3) is optional and None when absent.
    Malformed lines (no pipe, empty topic/claim) are skipped."""
    atoms: list[tuple[str, str, str, str | None]] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip(" -•\t")
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        topic, claim = parts[0], parts[1] if len(parts) > 1 else ""
        evidence = parts[2] if len(parts) > 2 and parts[2] else claim
        replaces = next(
            (p[len("replaces="):].strip() for p in parts[3:] if p.startswith("replaces=")), None,
        )
        if topic and claim:
            atoms.append((topic, claim, evidence, replaces))
    return atoms


async def _active_sem_atoms(store: Any, cap: int = _KNOWN_ATOMS_CAP) -> list[tuple[str, str]]:
    """The newest active sem/ atoms as (key, value) — shown to the miner (ruling 3) so it can
    mark `replaces=<id>` when a span corrects one. Capped; value trimmed at render time."""
    assert store._db is not None
    async with store._db.execute(
        "SELECT key, value FROM facts WHERE agent_id = ? AND status = 'active' "
        "AND key LIKE 'sem/%' ORDER BY updated_at DESC, id DESC LIMIT ?",
        (store._agent_id, cap),
    ) as cur:
        return [(r[0], r[1]) for r in await cur.fetchall()]


_STOP_WORDS = frozenset({
    "this", "that", "with", "from", "have", "been", "were", "will", "what", "when",
    "then", "than", "they", "them", "their", "there", "your", "into", "also", "only",
    "over", "some", "such", "very", "just", "like", "does", "each", "other", "which",
    "would", "could", "should", "about", "after", "before", "where", "while",
})


def _salient_words(v: str) -> set[str]:
    """SALIENT word/number tokens for BOTH B4 nets (F1 rescue overlap + F2 resurrection sweep):
    >=4 chars (the embeddings.py floor — a port like 8081 passes; a bare '3' does not) and not a
    common stopword. Keying both defenses on salient tokens means a short or generic shared token
    can never supersede or retract a fact."""
    return {t for t in re.findall(r"[a-z0-9]+", v.lower())
            if len(t) >= 4 and t not in _STOP_WORDS}


async def _active_corrections(store: Any, cap: int = 10) -> list[tuple[str, str]]:
    """B4(ii) known-window: active tier:reconcile_confirmed facts (the CURRENT corrected values)
    shown to the miner so a late chunk can't silently resurrect a value we already fixed."""
    assert store._db is not None
    async with store._db.execute(
        "SELECT key, value FROM facts WHERE agent_id = ? AND status = 'active' "
        "AND tags LIKE '%\"tier:reconcile_confirmed\"%' ORDER BY updated_at DESC, id DESC LIMIT ?",
        (store._agent_id, cap),
    ) as cur:
        return [(r[0], r[1]) for r in await cur.fetchall()]


async def _reconciled_pairs(store: Any) -> list[tuple[str, str]]:
    """B4(ii): for each active tier:reconcile_confirmed fact whose history has a superseded
    predecessor, the (stale_value, active_value) pair — the exact value a late chunk must not
    resurrect. Recovered via supersede history (get_fact_history), newest superseded first."""
    pairs: list[tuple[str, str]] = []
    for f in await store.query_facts(FactQuery(tags=["tier:reconcile_confirmed"], limit=200)):
        stale = next((v.value for v in await store.get_fact_history(f.key)
                      if v.status == "superseded"), None)
        if stale and stale != f.value:
            pairs.append((stale, f.value))
    return pairs


async def _sweep_resurrections(store: Any, minted_ids: list[int],
                               pairs: list[tuple[str, str]]) -> int:
    """Retract any freshly-minted atom that re-asserts a reconciled-away value. A resurrection =
    the atom shares a SALIENT token DISTINCT to the stale value (stale minus active) and NONE
    distinct to the active value (F2: the salient floor keeps a short generic token like '3' from
    retracting unrelated facts; a pair whose only distinction is sub-salient forms no net at all).
    Simplest sound mechanism: mark it superseded-on-arrival (a raw status UPDATE, same
    derived-state precedent as consolidation's raw writes) so it never enters the active pool;
    history is preserved. Returns the count retracted."""
    if not pairs or not minted_ids:
        return 0
    retracted = 0
    for atom in await store.get_facts_by_ids(minted_ids):
        atoks = _salient_words(atom.value)
        for stale, active in pairs:
            stale_distinct = _salient_words(stale) - _salient_words(active)
            active_distinct = _salient_words(active) - _salient_words(stale)
            if stale_distinct and (atoks & stale_distinct) and not (atoks & active_distinct):
                await store._db.execute(
                    "UPDATE facts SET status = 'superseded', "
                    "retrieval_strength = MIN(retrieval_strength, 0.1) "
                    "WHERE agent_id = ? AND id = ? AND status = 'active'",
                    (store._agent_id, atom.id),
                )
                await store._db.commit()
                log.warning("mining B4(ii): retracted resurrected stale value on %s (stale=%r)",
                            atom.key, stale)
                retracted += 1
                break
    return retracted


def _source_record(claim: str, evidence: str, records: list[dict]) -> dict | None:
    """The record whose text best overlaps the atom's (claim+evidence) tokens — the atom's
    episodic source, whose session_id becomes the per-atom provenance. None when nothing
    overlaps (a fabricated atom with no home in the window)."""
    probe = set(_tokens(f"{claim} {evidence}"))
    if not probe:
        return None
    best, best_score = None, 0
    for r in records:
        content = str(r.get("content", ""))[:_PER_RECORD_CHARS].lower()
        score = sum(1 for t in probe if t in content)
        if score > best_score:
            best, best_score = r, score
    return best


@dataclass
class MineReport:
    written: int = 0
    rejected_ungrounded: int = 0
    cancelled: bool = False


async def mine_transcript(
    store: Any,
    llm: Any,
    cancel_event: Any,
    *,
    write_budget: int = 25,
    corpus_char_cap: int = 6000,
    completions_log: list | None = None,
    file_tags: bool = True,
) -> MineReport:
    """Walk the un-mined transcript in `corpus_char_cap` chunks, extract typed `sem/` atoms
    (grounded, per-atom source provenance, 0.65), advancing the watermark per completed chunk.
    Cancellable, budgeted, never raises. Returns a MineReport. FIX 2c: when `completions_log` is
    provided, each chunk's RAW model completion (pre-parse) is appended for forensics (run-3's
    completions were unrecoverable, making the shadow-duplicate root-cause inferential)."""
    report = MineReport()
    minted_ids: list[int] = []
    # FIX 2 (run-10): (slug, normalized value) -> the key it was minted on THIS pass, so a second
    # atom carrying the same value on a DIFFERENT key (the double-replace duplicate) is skipped.
    minted_norm_key: dict[tuple[str, str], str] = {}
    try:
        raw_wm = await _get_meta(store, _MINING_WATERMARK_KEY)
        try:
            watermark = int(raw_wm) if raw_wm else 0
        except (TypeError, ValueError):
            watermark = 0

        records = await store.get_history(limit=1_000_000)
        window = sorted(
            (r for r in records if int(r.get("ts", 0) or 0) > watermark and r.get("content")),
            key=lambda r: int(r.get("ts", 0) or 0),
        )
        if not window:
            return report  # nothing new — watermark unchanged; cost paid only on the read

        i = 0
        budget_hit = False
        while i < len(window) and not budget_hit:
            if getattr(cancel_event, "is_set", lambda: False)():
                report.cancelled = True
                break
            # Assemble one chunk up to corpus_char_cap chars of record content.
            chunk: list[dict] = []
            chars = 0
            while i < len(window) and chars < corpus_char_cap:
                r = window[i]
                chunk.append(r)
                chars += len(str(r.get("content", ""))[:_PER_RECORD_CHARS]) + 1
                i += 1
            corpus = "\n".join(
                str(r.get("content", ""))[:_PER_RECORD_CHARS] for r in chunk
            )[:corpus_char_cap]

            # Ruling 3: show the miner the EXISTING active atoms (capped) so a correcting span
            # can mark `replaces=<id>` instead of minting a colliding-proof new claim-hash key
            # that leaves the stale value active forever. Reloaded PER CHUNK so an atom minted
            # by an earlier chunk of this same pass is already replaceable by a later chunk.
            known = await _active_sem_atoms(store)
            known_keys = {k for k, _v in known}
            # FIX 2: the already-active side of the dedupe — (slug, normalized value) -> its key.
            active_norm_key = {(_slug_of_key(k), _norm(v)): k for k, v in known}
            # FIX 2b: SHORT OPAQUE ids ([a1]) alongside the topic label + full key — a short id is
            # a cleaner `replaces=` target than a key-shaped string the model may mis-paste as a
            # topic; the key + topic label stay visible so raw-key/reuse forms keep working.
            short_ids = {f"a{n + 1}": k for n, (k, _v) in enumerate(known)}
            known_block = (
                "Known facts already on file — REUSE a fact's topic when your new fact continues "
                "it; to correct one, add `| replaces=<its id>` (e.g. replaces=a1):\n"
                + "\n".join(f"[a{n + 1}] {k} (topic {_slug_of_key(k)}): {v[:120]}"
                            for n, (k, v) in enumerate(known)) + "\n\n"
            ) if known else ""
            # B4(ii) known-window: current corrected values, so a late chunk sees the truth
            # instead of resurrecting a value reconciliation already superseded.
            corrections = await _active_corrections(store)
            corrections_block = (
                "Current corrected facts — these values are CURRENT; do NOT re-assert any older "
                "value for these:\n"
                + "\n".join(f"- {k}: {v[:120]}" for k, v in corrections) + "\n\n"
            ) if corrections else ""

            prompt = _PROMPT + known_block + corrections_block + corpus
            raw = await complete_cancellable(
                llm, prompt, cancel_event,
                char_cap=corpus_char_cap + len(_PROMPT) + len(known_block)
                + len(corrections_block) + 64,
            )
            if completions_log is not None and raw is not None:
                completions_log.append({"chunk_start_ts": int(chunk[0].get("ts", 0) or 0),
                                        "chunk_records": len(chunk), "raw": raw})
            if raw is None:
                # Cancelled mid-look — watermark NOT advanced past this chunk (next pass re-mines).
                report.cancelled = True
                break

            for topic, claim, evidence, replaces in _parse_atoms(raw):
                if report.written >= write_budget:
                    budget_hit = True
                    break
                src = _source_record(claim, evidence, chunk)
                # GROUNDING KILL-NET: reject an atom with no source or whose claim is not a
                # majority-token match in the cited record's text (applies to replacements too).
                if src is None or not grounded(claim, str(src.get("content", ""))[:_PER_RECORD_CHARS]):
                    report.rejected_ungrounded += 1
                    continue
                provenance = src.get("session_id") or str(src.get("ts", ""))
                slug = _slug(topic)
                replaces_present = replaces is not None  # B4(i): was a replaces= field present?
                # FIX 2a: the model pasted a known atom's key/prefix into the `topic` field instead
                # of using replaces=<id>. Recover the real slug (kill the shadow-dup mangled key)
                # and, for an unambiguous full-key paste, coerce an implied replaces so a correction
                # supersedes — same validity checks as an explicit marker (active + same-slug below).
                if not replaces:
                    coerced = _coerce_key_topic(topic, known_keys)
                    if coerced is not None:
                        implied, slug = coerced
                        replaces = implied or replaces
                        log.warning("mining: coerced key-shaped topic %r -> slug=%r replaces=%r",
                                    topic, slug, replaces)
                if replaces in short_ids:  # FIX 2b: resolve a short opaque id (replaces=a1) to a key
                    replaces = short_ids[replaces]
                key = f"sem/{slug}/{_h8(claim)}"
                # Ruling 3 supersede path: a VALID replaces (an active atom the miner was shown, same
                # topic slug) writes the corrected value onto the OLD key — store_fact supersedes
                # (history preserved), the stale value leaves the active set. An invalid replaces
                # (hallucinated id / cross-topic) falls back to a normal mint.
                if replaces and replaces in known_keys and replaces.startswith(f"sem/{slug}/"):
                    key = replaces
                elif replaces_present:
                    # B4(i), HARDENED (F1): a slug is a MANY-atom namespace (the prompt tells the
                    # model to REUSE topic labels), so a present-but-invalid/empty replaces= may
                    # ride a NEW same-topic fact, not a correction — force-superseding the newest
                    # sibling would silently destroy an unrelated atom. Rescue ONLY when the
                    # target is unambiguous: exactly ONE known ACTIVE atom on this slug AND the
                    # new claim shares >=1 salient token with that atom's value. Anything else
                    # falls back to a normal mint (degrades to a duplicate, never data loss).
                    same_slug = [(k, v) for k, v in known if k.startswith(f"sem/{slug}/")]
                    if len(same_slug) == 1 and _salient_words(claim) & _salient_words(same_slug[0][1]):
                        key = same_slug[0][0]
                        log.warning("mining B4(i): present-but-invalid replaces=%r rescued -> "
                                    "supersede sole same-slug %s", replaces, key)
                # FIX 2 (run-10, ids 51/52): a completion emitting the SAME corrected fact twice
                # with two different replaces= targets superseded BOTH stale atoms, leaving duplicate
                # active rows (identical value, same slug, different keys). Skip a mint whose
                # (slug, normalized value) already landed on a DIFFERENT key this pass or is already
                # active on one. A SAME-key write is corroboration (store_fact), never a duplicate —
                # the equality guard leaves it untouched.
                norm_val = _norm(claim)
                prior_key = (minted_norm_key.get((slug, norm_val))
                             or active_norm_key.get((slug, norm_val)))
                if prior_key is not None and prior_key != key:
                    log.warning("mining: dedupe — value %r already active on %s; skip duplicate "
                                "mint on %s", claim[:80], prior_key, key)
                    continue
                fact = await store.store_fact(
                    key=key,
                    value=claim,
                    tags=["sem", "pending_consolidation"],
                    confidence=0.65,  # searchable, sub-injection — ambient status is EARNED
                    source="transcript_mining",
                    provenance=provenance,  # the SOURCE record's session, PER ATOM (load-bearing)
                    node_kind="fact",
                )
                report.written += 1
                minted_ids.append(fact.id)
                minted_norm_key[(slug, norm_val)] = key  # FIX 2: record for the rest of this pass
                # Mint-time filing (M1): two-step closed-set classify -> atom_tags(provenance=mint).
                # Never blocks the mint; skipped when tagging is disabled.
                if file_tags:
                    await file_atom_tags(store, llm, cancel_event,
                                         atom_id=fact.id, topic=topic, claim=claim)

            if budget_hit:
                break  # partially-written chunk: don't advance — next pass re-mines (corroborates)
            newest_ts = max(int(r.get("ts", 0) or 0) for r in chunk)
            await _set_meta(store, _MINING_WATERMARK_KEY, str(newest_ts))

        # B4(ii) post-mining sweep: a newly minted atom that re-asserts a reconciled-away value is
        # retracted on arrival (never enters the active pool). Runs once over this pass's mints.
        await _sweep_resurrections(store, minted_ids, await _reconciled_pairs(store))
    except Exception:
        log.exception("mine_transcript failed (non-fatal; the idle look swallows)")
    return report
