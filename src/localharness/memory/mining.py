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
     advances over the longest fully-mined ts-CONTIGUOUS prefix (FIX 3: session walking is out of
     ts-order, so a raw chunk-max could skip un-mined older records — the contiguous rule keeps a
     cancelled/over-budget pass's tail for the next pass, cost per-window, never O(lifetime history)).
  2. CHUNKED FULL-COVERAGE, PER-SESSION: the un-mined window is grouped by session_id and walked
     session-by-session in chronological order, each session in `corpus_char_cap` chunks that never
     straddle a sitting boundary (FIX 3a); one LLM look per chunk, cancellation-checked between chunks.
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
from dataclasses import dataclass, field
from typing import Any

from localharness.memory.consolidation import _get_meta, _set_meta
from localharness.memory.idle_llm import complete_cancellable, grounded
from localharness.memory.sqlite import FactQuery
from localharness.memory.tag_classify import file_atom_tags

log = logging.getLogger(__name__)

_MINING_WATERMARK_KEY = "mining/last_ts"  # DISTINCT from consolidation/last_run
_PER_RECORD_CHARS = 400   # per-record slice fed to the model (and grounded against)
_KNOWN_ATOMS_CAP = 50     # existing active atoms shown to the miner (ruling 3), value-trimmed.
                          # FIX 3: default known-window (config mining_known_atoms_cap) — >= the
                          # write_budget default so every atom a single pass mints stays a valid
                          # same-pass `replaces=` target despite per-session chunking's higher chunk
                          # count (the DB surfaces this-pass mints first, by updated_at DESC).
_RETRY_MIN_RECORDS = 3    # FIX 2: a substantive chunk (>= this many records) that parses ZERO
                          # atoms is re-mined ONCE — below it, a chunk is legitimately empty
# FIX 4 (provenance-collapse guard): mining consumes only the OPERATIVE CONVERSATIONAL SURFACE —
# what the user and assistant actually said. Tool I/O (tool_result records) is structurally OUT of
# scope, so a store read-back (memory_search/memory_get echoing a prior fact VERBATIM into a LATER
# session) — and any FUTURE echo tool — is never even read: it cannot be re-mined and cannot advance
# a fact's provenance via store_fact's distinct-day ladder. Positive allowlist, NOT a denylist of
# named echo tools (which would need per-tool upkeep and silently miss a new one). Proven no-loss on
# the designed month (all 17 atoms ground in conversation; 0 need tool I/O). Config-overridable via
# mining_operative_message_types; None/empty means "unrestricted" (legacy all-types fetch).
_OPERATIVE_MESSAGE_TYPES = ("user_message", "assistant_message")

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
    """Whitespace-collapsed, lowercased, trailing sentence punctuation stripped — the duplicate-
    mint identity (FIX 2). REVIEW FIX: without the strip, 'X' vs 'X.' were distinct identities,
    so the same value re-mined with a trailing period landed on a fresh _h8 key beside the
    original as a duplicate ACTIVE row — the exact class the run-10 dedupe closed for
    byte-identical strings."""
    return " ".join(value.split()).lower().rstrip(" .,;:!?")


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


async def _active_slug_atoms(store: Any, slug: str) -> list[tuple[str, str]]:
    """ALL active fact atoms in one `sem/{slug}/` namespace — the novelty gate's comparison set.
    Unlike the capped known window, this is the WHOLE active slug (correctness must not depend
    on prompt visibility — same principle as the in-pass minted registry)."""
    assert store._db is not None
    async with store._db.execute(
        "SELECT key, value FROM facts WHERE agent_id = ? AND status = 'active' "
        "AND node_kind = 'fact' AND key LIKE ?",
        (store._agent_id, f"sem/{slug}/%"),
    ) as cur:
        rows = await cur.fetchall()
    return [(r[0], r[1]) for r in rows]


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
    # STAGE 1 (extraction science plan): coverage/residue — recall observability, no behavior
    # change. `residue` = committed records that were never the SOURCE of a WRITTEN atom (stable
    # record id + session + ts + chars + preview); its cross-run intersection (R) adjudicates
    # systematic vs stochastic under-extraction BEFORE any repair mechanism is built.
    records_seen: int = 0
    records_cited: int = 0
    residue: list[dict] = field(default_factory=list)
    # RESIDUE LEDGER (repair loop): enqueued = this pass's uncited non-trivial records entering
    # the ledger; drained = pending records given an isolated look this pass; rescued/retired =
    # this pass's DRAIN outcomes (the silent main-walk gap rescue is not counted here).
    residue_enqueued: int = 0
    residue_drained: int = 0
    residue_rescued: int = 0
    residue_retired: int = 0
    # NOVELTY GATE (precision): fresh mints folded into an existing same-slug atom as
    # corroboration because they paraphrase it (salient-token Jaccard >= threshold).
    folded: int = 0


async def mine_transcript(
    store: Any,
    llm: Any,
    cancel_event: Any,
    *,
    write_budget: int = 50,
    corpus_char_cap: int = 6000,
    known_atoms_cap: int = _KNOWN_ATOMS_CAP,
    operative_message_types: Any = _OPERATIVE_MESSAGE_TYPES,
    residue_enabled: bool = True,
    residue_attempt_cap: int = 2,
    residue_record_budget: int = 40,
    residue_min_chars: int = 20,
    novelty_fold_threshold: float = 0.5,
    completions_log: list | None = None,
    file_tags: bool = True,
) -> MineReport:
    """Walk the un-mined transcript session-by-session in `corpus_char_cap` chunks (never crossing a
    session boundary), extract typed `sem/` atoms (grounded, per-atom source provenance, 0.65),
    advancing the watermark over the longest fully-mined ts-contiguous prefix (FIX 3 no-loss).
    Cancellable, budgeted, never raises. Returns a MineReport. FIX 2c: when `completions_log` is
    provided, each chunk's RAW model completion (pre-parse) is appended for forensics (run-3's
    completions were unrecoverable, making the shadow-duplicate root-cause inferential).

    FIX 4: only records whose `type` is in `operative_message_types` are FETCHED — mining reads the
    conversational surface (user + assistant), never tool_result read-backs, so an echoed prior fact
    can never re-mine and collapse its provenance (None/empty = unrestricted, the legacy fetch).

    RESIDUE repair loop (core design, 2026-07-09): after the main walk, PRIOR passes' pending
    residue (committed records that never sourced a written atom) is re-mined in ISOLATION —
    residue-only chunks where a skimmed fact cannot lose the attention contest again — sequentially,
    at most `residue_record_budget` records per pass. Then THIS pass's uncited non-trivial records
    (>= `residue_min_chars`) are enqueued for the NEXT pass: amortized — a pass never drains what it
    just enqueued, and a quiet store drains nothing. A record still barren after
    `residue_attempt_cap` isolated looks is RETIRED: permanently out of the mining window, never
    deleted (history stays append-only — retire selects, never destroys)."""
    report = MineReport()
    minted_ids: list[int] = []
    # FIX 2 (run-10): (slug, normalized value) -> the key it was minted on THIS pass, so a second
    # atom carrying the same value on a DIFFERENT key (the double-replace duplicate) is skipped.
    minted_norm_key: dict[tuple[str, str], str] = {}
    # REVIEW FIX: key -> claim for every atom minted THIS pass. Unioned into the per-chunk
    # replaces-resolution view so a same-pass correction still supersedes an atom that scrolled
    # out of the capped known window (mints-in-pass > known_atoms_cap). The known window is a
    # prompt-VISIBILITY cap; supersede correctness must never depend on it.
    minted_pass: dict[str, str] = {}
    # STAGE 1 coverage: committed records by stable id, and the ids that sourced a written atom.
    seen_records: dict[str, dict] = {}
    cited_ids: set[str] = set()
    try:
        raw_wm = await _get_meta(store, _MINING_WATERMARK_KEY)
        try:
            watermark = int(raw_wm) if raw_wm else 0
        except (TypeError, ValueError):
            watermark = 0

        # FIX 4: fetch ONLY the operative conversational surface — tool_result read-backs never enter
        # the record stream, so an echoed prior fact is never read, never re-mined, never advances a
        # fact's provenance. Exclusion is at INPUT CONSTRUCTION (get_history's type filter), not a
        # post-assembly corpus trim. None/empty operative set => unrestricted (legacy all-types fetch).
        mt = list(operative_message_types) if operative_message_types else None
        records = await store.get_history(limit=1_000_000, message_types=mt)
        window = sorted(
            (r for r in records if int(r.get("ts", 0) or 0) > watermark and r.get("content")),
            key=lambda r: int(r.get("ts", 0) or 0),
        )
        # RESIDUE: fetch the ledger's pending set up front (the per-pass drain budget). Pending is
        # PRIOR passes' residue only — this pass's own residue is enqueued at the end, after the
        # drain, so a pass can never drain what it just enqueued (the amortized invariant).
        pending: list[dict] = []
        if residue_enabled:
            pending = await store.residue_pending(cap=residue_record_budget)
        if not window and not pending:
            return report  # nothing new & nothing owed — cost paid only on the read

        # FIX 3a: group the ts-sorted window by session_id so a chunk never straddles two sittings.
        # A dict preserves first-seen order and `window` is ts-sorted, so sessions come out in
        # chronological (min-ts) order; `walk` re-lays the records session-contiguous in that order.
        # This preserves the sequential `replaces=` visibility the supersede path needs: sessions are
        # mined in order and _active_sem_atoms reloads per chunk, so an atom minted while mining an
        # earlier session is a valid `replaces=` target while mining a later one.
        sessions: dict[str, list[dict]] = {}
        for r in window:
            sessions.setdefault(str(r.get("session_id") or ""), []).append(r)
        walk = [r for recs in sessions.values() for r in recs]

        async def _mine_one_chunk(chunk: list[dict], *, retry_on_zero: bool,
                                  drain: bool = False, idx: int = -1) -> tuple[set[str], bool]:
            """One isolated model look at `chunk` — known/corrections preamble, one cancellable
            completion (+ the FIX-2 zero-yield retry, MAIN walk only — the drain's attempt
            ladder IS its retry), then the grounded/supersede/dedupe atom loop. Shared verbatim
            by the main walk and the residue drain so the two paths can never drift. Returns
            (record ids cited by THIS chunk's written atoms, budget_hit); sets report.cancelled
            on a cancel (caller breaks; no state is committed for a cancelled look)."""
            cited_chunk: set[str] = set()
            corpus = "\n".join(
                str(r.get("content", ""))[:_PER_RECORD_CHARS] for r in chunk
            )[:corpus_char_cap]

            # Ruling 3: show the miner the EXISTING active atoms (capped) so a correcting span
            # can mark `replaces=<id>` instead of minting a colliding-proof new claim-hash key
            # that leaves the stale value active forever. Reloaded PER CHUNK so an atom minted
            # by an earlier chunk of this same pass is already replaceable by a later chunk.
            known = await _active_sem_atoms(store, known_atoms_cap)
            # REVIEW FIX: the replaces-RESOLUTION view = capped known window ∪ this pass's mints
            # (minted last, so a this-pass supersede's corrected value wins a key collision).
            # The cap bounds what the PROMPT shows; a same-pass target must resolve even after
            # scrolling out of the window (write_budget may legally exceed the cap's own bound).
            known_map = {**dict(known), **minted_pass}
            known_keys = set(known_map)
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
            char_cap = (corpus_char_cap + len(_PROMPT) + len(known_block)
                        + len(corrections_block) + 64)
            raw = await complete_cancellable(llm, prompt, cancel_event, char_cap=char_cap)
            if raw is None:
                report.cancelled = True  # cancelled mid-look — caller breaks, nothing committed
                return cited_chunk, False
            atoms = _parse_atoms(raw)
            # FIX 2 (extraction-yield): a SUBSTANTIVE chunk (>= _RETRY_MIN_RECORDS records) that
            # parses ZERO atoms is a whiff — a flat refusal ("I cannot extract any facts…", live in
            # run-5) or a bad roll — and the watermark would otherwise advance past it with nothing
            # re-mined (there was no per-chunk floor). Re-mine the SAME chunk EXACTLY once (bounded —
            # never loops); nothing minted yet, so the known-block is unchanged and the retry prompt
            # is identical. A tiny chunk (< K records) is legitimately empty and is NOT retried.
            retried = False
            if not atoms and retry_on_zero and len(chunk) >= _RETRY_MIN_RECORDS:
                retried = True
                raw_retry = await complete_cancellable(llm, prompt, cancel_event, char_cap=char_cap)
                if raw_retry is None:
                    report.cancelled = True  # cancelled mid-retry — watermark NOT advanced
                else:
                    atoms = _parse_atoms(raw_retry)
                    if completions_log is not None:  # keep the retry's raw for forensics too
                        completions_log.append({"chunk_start_ts": int(chunk[0].get("ts", 0) or 0),
                                                "chunk_records": len(chunk), "raw": raw_retry,
                                                "retry": True})
                log.warning("mining FIX 2: re-mined zero-yield chunk idx=%d records=%d "
                            "first_pass_yield=0 retry_yield=%d", idx, len(chunk), len(atoms))
            # Per-chunk forensics (FIX 2c + attribution): records, final yield, whether re-mined —
            # so run-12 can separate FIX-1's coverage gain from FIX-2's retry-recovered atoms.
            if completions_log is not None:
                completions_log.append({"chunk_start_ts": int(chunk[0].get("ts", 0) or 0),
                                        "chunk_records": len(chunk), "raw": raw,
                                        "atoms_yielded": len(atoms), "retried": retried,
                                        "residue_drain": drain})
            if report.cancelled:
                return cited_chunk, False

            for topic, claim, evidence, replaces in atoms:
                if report.written >= write_budget:
                    return cited_chunk, True
                src = _source_record(claim, evidence, chunk)
                # GROUNDING KILL-NET: reject an atom with no source or whose claim is not a
                # majority-token match in the cited record's text (applies to replacements too).
                if src is None or not grounded(claim, str(src.get("content", ""))[:_PER_RECORD_CHARS]):
                    report.rejected_ungrounded += 1
                    continue
                provenance = src.get("session_id") or str(src.get("ts", ""))
                # SELF-ECHO GUARD (completes FIX 4): only the USER'S OWN WORDS are recurrence
                # evidence. A fact may be BORN from any operative record (mints below keep real
                # provenance), but corroboration/fold from an ASSISTANT-sourced atom passes
                # provenance="" — store_fact's touch path then refreshes accessibility
                # (updated_at) while confidence and provenance stay untouched. Without this, the
                # agent restating a fact would step the ladder and fabricate multi-session
                # chapter evidence from its own mouth. Deterministic; no knob — epistemics.
                user_evidence = str(src.get("type", "")) == "user_message"
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
                    same_slug = [(k, v) for k, v in known_map.items() if k.startswith(f"sem/{slug}/")]
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
                # NOVELTY GATE (2026-07-09 dogfood — mining's PRECISION half): a fresh mint that
                # PARAPHRASES an active same-slug atom must not land as a sibling row (live store:
                # 8 near-identical "GTM plan" atoms from one conversation; the exact-match dedupe
                # above cannot see rewordings). Fold it into the best-matching atom as
                # CORROBORATION — same key + EXISTING value routes store_fact to the recurrence
                # ladder (+0.07 on a distinct provenance day, provenance advances), so paraphrased
                # recurrence EARNS ambient status exactly like verbatim recurrence and the store
                # stays one-atom-per-fact. First phrasing wins; value change remains supersede-only.
                # Supersede-redirected keys (replaces=/B4(i)) never reach here (fresh-mint guard):
                # a correction is similar to its target BY CONSTRUCTION and must replace.
                # Deterministic: salient-token SUBSET + equal numbers + Jaccard floor (config;
                # 1.0 ≈ off) — see the subset-rule comment below for why nothing looser is safe.
                if key == f"sem/{slug}/{_h8(claim)}":
                    probe = _salient_words(claim)
                    # NUMBER GUARD: facts differing only by a number ("room 3" vs "room 7") are
                    # DISTINCT, not paraphrases — short numerals fall below the salient floor, so
                    # word sets alone would call them identical. Paraphrases share their numbers;
                    # a differing number set blocks the fold outright.
                    nums = set(re.findall(r"[0-9]+", claim))
                    cands = await _active_slug_atoms(store, slug)
                    cands += [(k, v) for k, v in minted_pass.items()
                              if k.startswith(f"sem/{slug}/")]
                    if len(probe) >= 2 and not any(k == key for k, _v in cands):
                        best_key, best_val, best_j = None, "", 0.0
                        for k, v in cands:
                            if set(re.findall(r"[0-9]+", v)) != nums:
                                continue  # number mismatch — a different fact, never a fold
                            w = _salient_words(v)
                            # SUBSET RULE: fold only PROVABLE redundancy — every salient token of
                            # the new claim already lives in the existing atom (a restatement /
                            # shorthand adds nothing). Same-frame-different-slot facts ("building a
                            # SUMMARIZER subagent" vs "building a CITATION subagent") each carry a
                            # distinguishing token, which breaks subset in both directions — token
                            # sets cannot tell synonyms from contrasts, so anything less strict
                            # destroys real facts. A missed fold is just a dup that decay handles;
                            # a false fold is data loss. Asymmetric cost -> conservative rule.
                            if not probe <= w:
                                continue
                            j = len(probe & w) / len(probe | w)
                            if j > best_j:
                                best_key, best_val, best_j = k, v, j
                        if best_key is not None and best_j >= novelty_fold_threshold:
                            await store.store_fact(
                                key=best_key,
                                value=best_val,  # EXISTING value — corroboration, never overwrite
                                tags=["sem", "pending_consolidation"],
                                confidence=0.65,
                                source="transcript_mining",
                                # self-echo guard: assistant folds are evidence-inert
                                provenance=provenance if user_evidence else "",
                                node_kind="fact",
                            )
                            report.folded += 1
                            if src.get("id") is not None:  # the record WAS mined — cited, not residue
                                cited_ids.add(str(src["id"]))
                                cited_chunk.add(str(src["id"]))
                            log.info("mining fold: %r ~ %s (J=%.2f) — corroborated, no sibling mint",
                                     claim[:70], best_key, best_j)
                            continue
                # self-echo guard, verbatim path: this write CORROBORATES (identical active value
                # on the key) rather than minting — if the source is not the user's own words,
                # pass provenance="" so the ladder cannot step and provenance cannot advance.
                # A genuinely NEW mint keeps its real provenance whatever the source type.
                prov_arg = provenance
                if not user_evidence:
                    prior = await store.get_fact(key)
                    if prior is not None and prior.value == claim:
                        prov_arg = ""  # assistant restatement: accessibility refresh only
                fact = await store.store_fact(
                    key=key,
                    value=claim,
                    tags=["sem", "pending_consolidation"],
                    confidence=0.65,  # searchable, sub-injection — ambient status is EARNED
                    source="transcript_mining",
                    provenance=prov_arg,  # the SOURCE record's session, PER ATOM (load-bearing)
                    node_kind="fact",
                )
                report.written += 1
                if src.get("id") is not None:  # STAGE 1: this record sourced a WRITTEN atom
                    cited_ids.add(str(src["id"]))
                    cited_chunk.add(str(src["id"]))
                minted_ids.append(fact.id)
                minted_norm_key[(slug, norm_val)] = key  # FIX 2: record for the rest of this pass
                minted_pass[key] = claim  # REVIEW FIX: a replaces-target for the whole pass
                known_keys.add(key)       # ...and already within THIS chunk's remaining atoms
                known_map[key] = claim
                # Mint-time filing (M1): two-step closed-set classify -> atom_tags(provenance=mint).
                # Never blocks the mint; skipped when tagging is disabled.
                if file_tags:
                    await file_atom_tags(store, llm, cancel_event,
                                         atom_id=fact.id, topic=topic, claim=claim)
            return cited_chunk, False

        i = 0
        chunk_idx = 0
        budget_hit = False
        mined_objs: set[int] = set()  # id() of every record in a fully-mined (committed) chunk
        wm_idx = 0  # REVIEW FIX: forward-only cursor over window's mined ts-contiguous prefix
        while i < len(walk) and not budget_hit:
            if getattr(cancel_event, "is_set", lambda: False)():
                report.cancelled = True
                break
            # Assemble one chunk up to corpus_char_cap chars — but NEVER across a session boundary
            # (an oversized session sub-splits here via the same char-cap loop, now session-scoped).
            chunk: list[dict] = []
            chars = 0
            sid0 = str(walk[i].get("session_id") or "")
            while (i < len(walk) and chars < corpus_char_cap
                   and str(walk[i].get("session_id") or "") == sid0):
                r = walk[i]
                chunk.append(r)
                chars += len(str(r.get("content", ""))[:_PER_RECORD_CHARS]) + 1
                i += 1
            _, budget_hit = await _mine_one_chunk(chunk, retry_on_zero=True, idx=chunk_idx)
            if report.cancelled:
                break  # cancelled mid-look — watermark NOT advanced past this chunk (next pass re-mines)
            if budget_hit:
                break  # partially-written chunk: don't advance — next pass re-mines (corroborates)
            # FIX 2/3 no-loss watermark: advance ONLY over the longest fully-mined ts-CONTIGUOUS
            # prefix of the (ts-sorted) window. Session walking is out of ts-order — a later-walked
            # session may pre-date an earlier one — so committing a raw chunk-max could skip un-mined
            # OLDER records and silently drop them; the contiguous-prefix rule keeps FIX 2's no-loss
            # property under reordering (a budget/cancel abort leaves the tail ts > watermark, so the
            # next pass re-mines it, corroborating). A full walk commits the global max, as before.
            mined_objs.update(id(r) for r in chunk)
            for r in chunk:  # STAGE 1: committed — the miner fully processed these records
                if r.get("id") is not None:
                    seen_records[str(r["id"])] = r
            new_wm = watermark
            # REVIEW FIX: forward-only cursor — mined_objs only grows and `window` is fixed, so
            # the prefix never rewinds; the old from-zero rescan per chunk was O(chunks × window)
            # on a large backfill (exactly the mode the raised write-budget ceiling invites).
            while wm_idx < len(window) and id(window[wm_idx]) in mined_objs:
                new_wm = max(new_wm, int(window[wm_idx].get("ts", 0) or 0))
                wm_idx += 1
            if new_wm > watermark:
                await _set_meta(store, _MINING_WATERMARK_KEY, str(new_wm))
                watermark = new_wm
            chunk_idx += 1

        # STAGE 1 coverage build: residue = committed records never cited by a written atom.
        report.records_seen = len(seen_records)
        report.records_cited = sum(1 for rid in seen_records if rid in cited_ids)
        # content_h8: record ids are per-run uuids, so the CROSS-RUN intersection keys on a stable
        # content fingerprint instead (manifest-scripted turns are byte-stable across runs).
        report.residue = [
            {"id": rid, "session_id": r.get("session_id"), "ts": int(r.get("ts", 0) or 0),
             "chars": len(str(r.get("content", ""))), "preview": str(r.get("content", ""))[:80],
             "content_h8": hashlib.sha1(str(r.get("content", "")).encode("utf-8")).hexdigest()[:8]}
            for rid, r in seen_records.items() if rid not in cited_ids
        ]

        # RESIDUE repair loop — drain PRIOR passes' residue, then enqueue this pass's own.
        if residue_enabled:
            # Gap rescue (silent): a pending record the MAIN walk just cited needs no drain —
            # flip it rescued and drop it from this pass's drain (no double look).
            if cited_ids:
                await store.residue_rescue(sorted(cited_ids))
                pending = [p for p in pending if str(p["record_id"]) not in cited_ids]
            if pending and not report.cancelled and not budget_hit:
                by_id = {str(r.get("id")): r for r in records if r.get("id") is not None}
                residue_walk: list[dict] = []
                orphans: list[str] = []
                for p in pending:
                    rec = by_id.get(str(p["record_id"]))
                    residue_walk.append(rec) if rec is not None else orphans.append(str(p["record_id"]))
                if orphans:
                    # No longer fetchable (e.g. the operative surface narrowed since enqueue) — a
                    # barren look by definition; bump toward retirement rather than pend forever.
                    report.residue_drained += len(orphans)
                    report.residue_retired += await store.residue_bump(
                        orphans, attempt_cap=residue_attempt_cap)
                    log.warning("mining residue: %d ledger records no longer fetchable; bumped",
                                len(orphans))
                # Isolated, sequential, per-session drain chunks (same assembly rule as the main
                # walk). One look at a time — never fanned out (single shared GPU).
                j = 0
                while j < len(residue_walk):
                    if getattr(cancel_event, "is_set", lambda: False)():
                        report.cancelled = True
                        break
                    rchunk: list[dict] = []
                    rchars = 0
                    rsid0 = str(residue_walk[j].get("session_id") or "")
                    while (j < len(residue_walk) and rchars < corpus_char_cap
                           and str(residue_walk[j].get("session_id") or "") == rsid0):
                        rr = residue_walk[j]
                        rchunk.append(rr)
                        rchars += len(str(rr.get("content", ""))[:_PER_RECORD_CHARS]) + 1
                        j += 1
                    cited_chunk, r_budget_hit = await _mine_one_chunk(
                        rchunk, retry_on_zero=False, drain=True)
                    if report.cancelled:
                        break  # an aborted look is NOT a barren look — attempts stay unbumped
                    rids = [str(r.get("id")) for r in rchunk if r.get("id") is not None]
                    report.residue_drained += len(rids)
                    report.residue_rescued += await store.residue_rescue(
                        [rid for rid in rids if rid in cited_chunk])
                    report.residue_retired += await store.residue_bump(
                        [rid for rid in rids if rid not in cited_chunk],
                        attempt_cap=residue_attempt_cap)
                    if r_budget_hit:
                        break  # write budget exhausted — the rest of pending waits, unbumped
            # Enqueue THIS pass's residue for the NEXT pass (intake triviality filter: the metric
            # above still reports trivial records; the ledger just never chews them).
            report.residue_enqueued = await store.residue_enqueue(
                [e for e in report.residue if e.get("chars", 0) >= residue_min_chars])

        # B4(ii) post-mining sweep: a newly minted atom that re-asserts a reconciled-away value is
        # retracted on arrival (never enters the active pool). Runs once over this pass's mints.
        await _sweep_resurrections(store, minted_ids, await _reconciled_pairs(store))
    except Exception:
        log.exception("mine_transcript failed (non-fatal; the idle look swallows)")
    return report
