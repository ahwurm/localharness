# Spec 05: Memory System

**Component:** `src/localharness/memory/` (+ `cli/session_accumulator.py`)
**Requirements:** MEM-01..04, WRITE-01..06, RANK-01..05, HIER-02..04, CONS-01..06, COLL-01..04, PGATE-01..04, SESS-02..05, TIME-01..04
**Status:** v2 (predictive) — supersedes the v1 three-tier-scope spec

> **Ground-truth note.** This document is structural: it names modules, tables, and
> concepts. Where a number or a decision is load-bearing (the injection gate, the quadrant
> definitions, the confidence tiers), the **source docstrings are canonical** and win over
> this prose. Specs describe intent; the code is the contract.

---

## 1. Purpose

The memory system gives one agent durable state across sittings, and decides — with **zero
extra model calls** — what is worth keeping and what earns space in the prompt. It is built
on one cognitive frame, **Complementary Learning Systems (CLS)**: a fast path captures
sparse episodes as they happen (the event bus writing sessions; the write gates capturing
lessons), and a slow path integrates them into durable, prompt-visible knowledge during
idle "sleep" (the consolidation pass). The two halves are deliberately split because
integration is expensive and must never block the user's turn.

Four properties hold across every write path, old and new:

- **Nothing is overwritten.** A conflicting fact *supersedes* the old one; the old row
  stays queryable. Every fact carries **provenance** to its source episode (WRITE-02/04).
- **One visibility line governs the prompt.** A fact enters the injected memory block only
  at **confidence ≥ 0.7 AND retrieval_strength ≥ 0.2**. Everything the harness auto-captures
  lands *below* that line until the slow path confirms it (§5).
- **The injected block is byte-stable between consolidations.** Reads never reorder it, so
  the inference server's prefix cache survives (RANK-04, §6).
- **Ranking costs zero decode tokens.** Activation scoring runs as SQL scalar functions
  (RANK-02, §6).

The v1 spec described scoped agent→division→org storage. Those columns still exist
(`division_id` / `org_id` default `''`) and division/org markdown is still read into
context, but the live subject of this system — and of this rewrite — is the **single-agent
prediction-error write path** and the **activation-ranked, byte-stable injection** built on
top of it. The scope machinery is legacy substrate; the predictive gate is the work.

---

## 2. Module map & honest status

Every module below is a bus subscriber wired beside `MemoryStore` at startup
(`cli/start_cmd.py`). Each swallows its own exceptions and is logged, never re-raised — a
memory fault can never break the agent loop or the user's turn.

| Module | Role | Status |
|--------|------|--------|
| `memory/sqlite.py` `MemoryStore` | The store: facts/edges/FTS5/sessions + v4 telemetry; supersede-not-overwrite; the injected index render; activation scalars; the pure-SQL priors + quadrant functions | **live, default-on** |
| `memory/gate.py` `WriteGate` | Motif capture floor: `resolved_error` / `stuck_recovered` / `novelty` candidates, sub-0.7 (`write_gate_enabled`, default on) | **live, default-on** |
| `memory/predictive_gate.py` `PredictiveGate` | Scores every tool outcome against that tool's own history; emits the surprise event contract; **writes no facts** (`predictive_gate.enabled`, default on) | **collect-only** |
| `memory/user_signals.py` `UserSignalDetector` | Zero-NLU correction/confirmation/interruption tripwire; logs look-ready records + staged snapshots; **writes no facts** | **collect-only** |
| `memory/predictive_write_gate.py` `PredictiveWriteGate` | The live write decision: turns surprise scores + scoped corrections into real, reversible, sub-0.7 writes (`predictive_gate.write_live`, default on — the pre-committed KILL lever) | **live, default-on (capture-only)** |
| `memory/consolidation.py` `ConsolidationScheduler` / `ConsolidationPass` | Idle CLS slow-integrate: fold / promote / decay / trim; its LLM session-replay claim-extractor is a built, guarded, cancellable seam | **deterministic core live; LLM replay built, wired off** |
| `memory/hierarchy.py` | Persists the cruncher's gist/schema tree into the memory graph; routes structure-aware search | **live, default-on** |
| `memory/markdown.py` `MarkdownMemory` | `MEMORY.md` regeneration (facts + session shelf), preserving hand-written sections | **live, default-on** |
| `cli/session_accumulator.py` `SessionAccumulator` | Bus-subscribed sitting counters → the payload-first one-line session summary | **live, default-on** |
| L3 model-stated `expect:` slot; L2 logprob surprisal; idle deep correction/surprise reconciliation | The next rungs of the mechanisms ladder | **future phase (36/37)** |

"Collect-only" is a hard contract: those modules write **only** the v4 telemetry tables
and never touch `facts` / `sessions` / `edges`. "Capture-only" means a module *does* write
facts, but every write is below the visibility line (§5), so it changes what is *captured*,
not what the agent *does*.

---

## 3. The store

One SQLite database per agent (`agents/{agent_id}/memory.db`), async via `aiosqlite`, WAL
mode, `busy_timeout` so the agent loop and the idle consolidation pass wait instead of
throwing "database is locked". Schema evolves through a stepwise migration ladder; each
rewrite script is a single `BEGIN IMMEDIATE … PRAGMA user_version = N; COMMIT` transaction,
so a crash rolls back to the prior intact version and the next open retries cleanly.
`CURRENT_SCHEMA_VERSION = 4`.

### 3.1 Facts — supersede-not-overwrite, with a trust/accessibility split

`facts` rows are the graph nodes. The load-bearing columns beyond the v1 key/value/tags/
confidence/source/timestamps:

```sql
status        TEXT NOT NULL DEFAULT 'active',   -- 'active' | 'superseded'
superseded_by INTEGER,                          -- successor row id
provenance    TEXT NOT NULL DEFAULT '',         -- source session id / origin (WRITE-04)
retrieval_strength REAL NOT NULL DEFAULT 0.5,   -- accessibility; decays with disuse (RANK-03)
importance    REAL NOT NULL DEFAULT 0.0,        -- write-time tag-heuristic prior
access_count  INTEGER NOT NULL DEFAULT 0,       -- folded ACT-R use counter
last_accessed_at INTEGER,                        -- folded ACT-R recency
access_count_staged  INTEGER NOT NULL DEFAULT 0, -- staged read counters (RANK-04)
last_accessed_staged INTEGER,
node_kind     TEXT NOT NULL DEFAULT 'fact'      -- fact | gist | schema (RANK-01)

-- The active-tier invariant: one truth per key, superseded rows keep the key.
CREATE UNIQUE INDEX ux_facts_active_key ON facts(agent_id, key) WHERE status = 'active';
CREATE INDEX idx_facts_active_recency  ON facts(agent_id, updated_at DESC) WHERE status = 'active';
```

`store_fact` is the single write seam and enforces WRITE-01/02/04:

- **No active row for the key** → insert a new active row.
- **Active row, identical value** → *corroboration touch*: bump `updated_at`,
  `confidence = MAX(old, new)`; no duplicate row (so restart re-fires and repeat captures
  are no-ops, not churn).
- **Active row, different value** → *supersede*: mark the old row `status='superseded'`,
  drop its `retrieval_strength` immediately (it loses the retrieval competition — RANK-03
  interference, not erasure), insert a fresh active row, link `superseded_by`. History stays
  reachable via `get_fact_history` / `FactQuery(include_superseded=True)`.

Every write is **read-back-verified**: the active row is re-read and compared before the
write is claimed; a mismatch raises `MemoryVerifyError`. This closes the
"claims-to-write-but-didn't" failure class at the store boundary. A `delete_fact` exists for
explicit user-initiated removal only; no harness path calls it (supersede, never delete).

`importance` is set at write time by a **closed tag heuristic** (`_IMPORTANCE_PRIORS`),
never an LLM rater. A tier tag with no entry there ranks at the 0.0 floor — so every
candidate-producing tier (`tier:resolved_error`, `tier:stuck_recovered`,
`tier:surprising_failure`, `tier:correction_pending`, `remember`) must have an explicit
prior or its graded salience silently degrades to a no-op.

### 3.2 Edges — the typed graph

`edges(src_id, dst_id, kind)` carries `derived_from` / `member_of` / `supports` /
`contradicts`. `supersedes` is deliberately **not** an edge kind — it stays a `facts` column
because it is the hot-path exclusion mechanism, not a relationship to walk. `neighborhood()`
is a Python frontier BFS with a real visited set (depth hard-capped at 4), chosen over a
recursive CTE that enumerated all simple paths and degraded the every-turn retrieval path.

### 3.3 Sessions

One `sessions` row per sitting: `started_at` / `ended_at`, turn/action/token counts,
`exit_reason`, and a `summary` that stays `NULL` until the sitting ends. That single
`summary IS NOT NULL` predicate does two structural jobs for the injected shelf (§9): it
excludes both the still-open current sitting and vacuous sittings that derived no summary.

### 3.4 The v4 telemetry tables (COLL substrate)

Additive-only — four tables, zero touches to `facts`/`sessions`/`edges`, so the injected
block is byte-stable **by construction**. They are the substrate the predictive layer
measures against and the record Phase 36 will re-derive thresholds from.

- **`tool_observations`** — one row per scored tool result. `is_error` derives from
  `Observation.error IS NOT NULL` (`exit_code` is a dead field). `duration_ms` is the
  Action→Observation timestamp delta (zero loop instrumentation). `event_id` is UNIQUE for
  idempotent re-ingestion (`INSERT OR IGNORE`). `source ∈ {live, backfill}`.
- **`surprise_scores`** — the persisted `SurpriseScored`. `expectation_json` snapshots the
  exact prior that produced the score (so thresholds can be re-derived offline under any
  windowing). `quadrant ∈ {routine, surprising_failure, unsurprising_failure, quiet_surprise,
  cold_start}`.
- **`user_signals`** — the zero-NLU labels. `signal_type ∈ {correction, confirmation,
  interruption}`; `trigger_family ∈ {negation, correction_phrase, frustration, reask,
  confirmation, interruption}`; the **full** user message is stored (look-ready records for
  the future model look).
- **`staged_snapshots`** — credit-assignment candidates. `candidate_type ∈ {bump, suspect}`.

### 3.5 FTS5 & JSONL history

`facts_fts` is an FTS5 index over `key`/`value`/`tags`, kept in sync by triggers; the update
trigger is narrowed to `key`/`value`/`tags` so activation bumps never churn the index.
`memory_search` queries route FTS5 MATCH through a sanitizer that quotes every whitespace
token, so operator characters in real corpora (`000660.KS`, `built-in`, `-1.5σ`) are literal
terms, never syntax (WRITE-05). `history.jsonl` remains the append-only, event-sourced
session log — the source of truth for replay and reconstruction; it is never rewritten, only
appended, and compaction is logical (`replaces_ids`), not physical.

---

## 4. The write path

Two harness-initiated write paths run side by side. Both emit candidates **below** the
visibility line; neither ever calls a model inline.

### 4.1 Motif WriteGate — the capture floor

`WriteGate` subscribes to `Observation` / `StuckRecovered` and turns discrete,
already-on-the-bus signals that the agent's world-model was *wrong then corrected* into
fact candidates, at zero added latency (dict ops + one SQLite write on fire). Three tiers,
each a distinct confidence below 0.7:

- **`resolved_error` (0.65)** — a tool that errored and later succeeded. The highest-warrant
  learning signal; cross-turn by design. Keyed `gate/resolved_error/<tool>/<lesson>/<session>`
  so identical lessons accumulate one row per episode and true recurrence stays visible to
  consolidation.
- **`stuck_recovered` (0.60)** — a repeated-action loop that broke free. Tagged `salient`: one
  occurrence is warrant enough for promotion (a rare, high-salience event).
- **`novelty` (0.50)** — first successful use of a tool. Telemetry + candidate only; by design
  it never promotes ("used a tool once" is not a durable lesson).

Every fire publishes a `MemoryGateFired` event (the live observability surface — fire counts
per tier, watched rather than pre-measured). Suppressed/pending moments deliberately emit
nothing. This gate is the **floor**: it works from turn one, before any tool has enough
history for a statistical prior to exist.

### 4.2 The predictive layer

The floor is motif-shaped and binary. The predictive layer adds *graded* surprise measured
against each tool's own history — the north star's "prediction-error-gated writes" with
actual predictions.

**Per-tool statistical priors, in pure SQL (COLL-01).** `get_tool_prior(tool)` computes a
`ToolPrior` in one indexed aggregate over `tool_observations`: observation count, error
rate, latency mean/variance, output-size mean/variance. It is **walk-forward** — only rows
strictly earlier than the scored observation count, so an outcome never contaminates its own
prior. Empty history maps to `None`, never a fabricated zero: cold start is carried honestly.

**The three-event contract (COLL-04).** For each tool call the predictive layer publishes:

1. `ExpectationAttached` at dispatch — the L1 prior the harness held (`source='l1_priors'`;
   the model-stated `l3_expect_slot` is a future phase).
2. `OutcomeObserved` on completion — is_error, output_len, duration_ms.
3. `SurpriseScored` — the graded composite plus its quadrant.

The composite score is `error_surprisal + w_lat·|z_latency| + w_size·|z_size|`: the
information-theoretic surprise of the boolean outcome against the tool's own error rate,
plus weighted absolute deviations of latency and output size. Every term degrades to `0.0`
below the cold-start floor (default `min_prior_n = 5`), so an empty prior scores exactly
zero.

**The quadrant taxonomy (canonical).** `compute_quadrant` maps an outcome onto the
expectation × outcome grid the binary motif gate structurally cannot express. `predicted_fail`
means the tool's own history says error is the base case (`error_rate ≥ 0.5`). Below the
floor or with no prior → `cold_start`.

| | **outcome succeeded** | **outcome errored** |
|---|---|---|
| **predicted success** (error_rate < 0.5) | `routine` — no write; the junk the motif gate can't suppress | `surprising_failure` — a normally-reliable tool errored; **the write quadrant** |
| **predicted failure** (error_rate ≥ 0.5) | `quiet_surprise` — predicted to fail, **succeeded** | `unsurprising_failure` — no write; the quadrant the motif gate can't even name |

**`quiet_surprise` is canonically defined as predicted-fail-but-SUCCEEDED.** (Distinct from a
"succeeded-but-differently" latency/size anomaly — that magnitude rides in the graded *score*,
is currently unnamed, and is future work; the quadrant name belongs to the error-flip case.)

**Graded confidence anchored to measured percentiles (PGATE-01).**
`graded_confidence(score) = clamp(0.5 + 0.07·score, 0.5, 0.69)`. The anchors come from the
collect-only distribution (routine's P90 score sits at the floor; the surprising median
grades to ~0.65). Two facts matter: the number is **strictly < 0.7** so a stat write can
never enter the injected index, and it only *grades importance within* the tier — the
**categorical quadrant gate**, not the score, makes the write/no-write call.

**Quadrant-gated writes (PGATE-01/02).** `PredictiveWriteGate` reads the already-published
`SurpriseScored` (zero recompute) and writes **only** on `surprising_failure`, keyed by
`(tool, day)` so a same-day retry burst corroborates into one row. `unsurprising_failure`,
`routine`, `cold_start`, and `quiet_surprise` write nothing — PGATE-02's suppression is the
*absence of a branch*, not a branch. This is the exact junk the motif floor could not
suppress.

**Correction handling (PGATE-03).** User messages are themselves an expectation signal.
`UserSignalDetector` (collect-only) classifies each turn from a zero-NLU trigger lexicon —
correction-class families (`negation` / `correction_phrase` / `frustration`) checked before
`confirmation` before `interruption`, first match wins — and, for corrections, snapshots the
explicitly-staged facts as `suspect` (confirmations snapshot as `bump`). `PredictiveWriteGate`
then writes, reversibly and coarsely:

- An **explicit `correction_phrase`** ("i meant / actually / instead / you misunderstood")
  supersedes the **single most-recently-staged suspect** fact — a marker prefixes the *full*
  original value (never truncated; a plain `get_fact` still returns the real content), at
  confidence 0.6. Scoped to one suspect, never the whole staged sitting; reversible via
  history.
- **Every other in-scope correction, and the no-suspect case**, writes a standalone
  **quarantine** fact keyed to the correction, at confidence 0.65 — the user's own words,
  additive, never touching the staged rows. A bare `negation` ("no") is deliberately
  quarantine-only: it is far too broad to rewrite an already-retrieved fact on.

**The `write_live` lever.** `PredictiveWriteGate` fires only when
`predictive_gate.write_live` is set (default on, **fail-closed** if unreadable). It is the
pre-committed KILL: flip it off and the harness reverts to motif-only capture while the
collect-only scorer keeps persisting scores as pure telemetry — the strongest form of "the
motif floor stays provably unchanged" (the gate is a separate sibling subscriber; the diff
on `gate.py` is empty by construction).

---

## 5. The visibility line

There is exactly one gate between the store and the prompt, in `_render_memory_index` (and
mirrored in `flush_memory_md`):

```sql
WHERE status = 'active' AND confidence >= 0.7 AND retrieval_strength >= 0.2
```

Confidence ≥ 0.7 is *trust*; retrieval_strength ≥ 0.2 is *accessibility*. A fact must clear
**both** to occupy prompt space. **Every fact any write gate in §4 produces is below 0.7** —
motif tiers (0.50–0.65), stat writes (`graded_confidence` ≤ 0.69), corrections/quarantine
(0.60/0.65), and gists (0.60). They live in the store, are searchable via the tool path, and
route retrieval through the graph — but they **do not enter the injected block** until the
slow path promotes them (§7).

State this plainly, because it is the whole honest posture of the tranche: **the new write
paths change what the harness captures, not how the agent behaves.** Turning the predictive
gate on adds rows below the line; it does not, on its own, alter a single injected byte.

---

## 6. Activation & ranking

**ACT-R base-level activation, in SQL (RANK-02).** Ordering is a registered scalar function,
so it costs zero decode tokens and does not depend on SQLite's optional math build. The
single-trace simplification is `ln(1 + n) − d·ln(age)` with canonical decay `d = 0.5`, where
`n` is the folded access count and age is measured from the most recent read-or-write. Two
score variants:

- `lh_slow_score` — the **injected-block** score: `importance + base_activation` over the
  **folded** columns only, with age quantized to **days**. Every input moves only at a
  consolidation fold, a genuine write, or a day boundary.
- `lh_fused_score` — the **tool-path** score: adds fresh staged counters, hour-granular age,
  `ln(confidence)`, and the BM25 relevance term. Only the tool result — appended *after* the
  prefix cache — may re-rank freely on every call.

**Staging discipline + byte-stable injection (RANK-04).** This is prefix-cache economics, not
cosmetics: on reference architecture A, one changed byte near the top of a 32k-token prompt
costs ~16 s of time-to-first-token. So a plain read never touches anything the injected block
orders. Reads accumulate in the `*_staged` twin columns; the block reads only the base
columns. The single moment a read can reorder the block is `fold_staged_access`, called at the
consolidation boundary — which also restores accessibility (a heavily-used, decayed fact can
climb organically back above the line instead of needing a fresh supersede-write). The render
also forces the partial recency index and quantizes age to days so the block's bytes flip only
with genuine change or the daily date boundary.

---

## 7. Idle consolidation — the CLS slow-integrate pass

`ConsolidationScheduler` lives inside the harness process (no daemon, no cron): a
session-start staleness check plus an in-session idle timer fire a pass, and **any user turn
cancels an in-flight pass instantly** (the serial inference gate is non-preemptive, so a pass
must yield rather than make the user wait behind its generation). Default-on, config-off
(`memory.consolidation.*`). A `ConsolidationPass` runs six steps, the deterministic core with
no model at all:

1. **Fold** staged read-counters into the base columns (§6).
2. **Promote recurring candidates.** The promotion warrant is **cross-episode recurrence** —
   the same lesson captured from ≥2 distinct sessions (grouped by the gate's content hash, not
   by tool alone) — **or** an existing promoted record (schema-consistent fast track) **or** a
   `salient` flag (one stuck-recovery is enough). A promoted record crosses to confidence 0.8
   (above the line), composed only of verbatim candidate bodies, linked `derived_from` its
   sources. Novelty carries none of these warrants and never promotes.
   **`tier:correction_pending` rows are excluded from promotion** — a disputed supersede or a
   quarantine fact must not graduate into the injected block until the Phase-36 model look
   reconciles it. This exclusion is a live predicate, not a convention.
3. **Replay (LLM seam).** The rationalization engine: extract durable claims from recent
   history via a cancellable, guarded LLM call (iteration cap, dedup-before-generate,
   verify-against-leaf: a majority of a claim's long tokens must appear verbatim in the
   source). **Built, guarded, and wired OFF by default** (`llm=None`) until its output quality
   is iterated live.
4. **Decay** — retrieval_strength halves per idle half-life (default 30 days), floored at 0.05;
   trust (confidence) never decays. Facts fade from the *index*, never from the *store*.
5. **Cap-trim** — a **soft** capacity cap (default 256 active facts), enforced *here*, never at
   admission. Trim = demote the lowest-activation actives below the line; nothing is deleted,
   and a record promoted in this same pass is never demoted in it.
6. **Proxies** — promote-then-superseded churn rate + a promotion-sample hook (the dispatch
   layer can pipe samples out for passive owner spot-check); fire counters alone can't see
   silent corruption.

---

## 8. Gist / schema hierarchy

When the cruncher reads an over-window document, it builds a lossy gist-over-verbatim tree and
normally discards it. `hierarchy.persist_gist_tree` gives that tree rows in the memory graph:
one **schema node** per run, one **gist node** per reduce output (`member_of` the schema,
`derived_from` the previous level's gists), and a final-answer node. Gists sit at confidence
0.60 — **below the line** by construction, so writing one mid-session can never reorder the
injected block. They *route* retrieval (the graph neighborhood in `memory_search`) rather than
occupy the prompt: gists route the search, leaf records anchor the answer. The 0.5.1
number-provenance net extends here (HIER-04): a figure in a gist absent from all of its inputs
is tagged `unverified-figures` — flagged, never rejected — the same DRM-lure guard the
cruncher ships. A rolling compaction summary persists the same way (one gist per sitting,
superseding itself each re-fire).

---

## 9. Session shelf

`SessionAccumulator` subscribes to the bus and keeps sitting-scoped counters (turns, tool
calls, delegations, gate captures, the opening ask) with zero model calls. On close,
`derive_session_summary` composes one **payload-first** line — leading with the resolved-error
or unstuck lesson, else the trimmed opening ask (`asked: "…"`), never with novelty — or
returns `None` for a sitting with nothing discriminating (suppressed, never padded with
filler). `end_session` writes it to the `sessions` row.

The injected "Recent Session History" shelf renders **from the sessions table**, not from
`MEMORY.md`: each entry gets a relative-day + clock label (`- today 11:47am: …`), newest
first, hard-capped at 8 lines (`_SESSION_SHELF_HARD_CAP`, a system invariant — config can go
lower, never higher). Labels are computed once per render against the local day, so the block
stays byte-stable within a day and flips only at the local day boundary, phasing with the
system prompt's own date line (TIME-04, no new cache-bust class). Dropped sittings stay in the
table: absence from the prompt is not forgetting.

---

## 10. Honest limits (named, not hidden)

- **Correction detection is lexical.** The trigger lexicon is a recall-first tripwire, not a
  classifier — measured recall is **~23%** on a real hand-labeled census, so most corrections
  that don't use a trigger word are missed. Precision comes from a later model look, which is
  future work; today a false trigger costs one logged record and a miss costs a missed
  correction.
- **Expected-failure suppression and stuck-recovery are synthetic-tested only.** The real
  bus-events trace to date carries **zero** `unsurprising_failure` events and **zero**
  `StuckRecovered` events. PGATE-02's live suppression and the `stuck_recovered` tier are
  proven by unit tests, not by a real occurrence — disclosed, never silently declared victory.
- **Statistical priors are per-tool and silent until a tool has ≥5 observations.** Below the
  cold-start floor a tool's surprise is neutral `0.0` and its quadrant is `cold_start`; the
  predictive layer contributes nothing for a fresh tool. The **motif floor (§4.1) covers from
  turn one**, which is why it stays as the floor.
- **The injected block does not yet change the agent's FIRST move.** Injected memory improves
  *recovery* — the agent consults what it knows once context makes it relevant. Steering the
  opening move is explicitly future work. Every new write in this tranche lands below the
  visibility line, so it changes capture, not the next action.

---

## 11. Configuration

All under an agent's `memory:` key (see `config/models.py`; every field auto-enumerates as a
`agent.memory.*` component-registry axis, tunable with no code edit):

```yaml
memory:
  inject_into_context: true          # inject the memory block into the system prompt
  index_mode: true                   # inline the INDEX (names + one-liners), bodies on demand
  max_session_history_entries: 8     # session-shelf lines (hard-capped at 8)
  write_gate_enabled: true           # the motif capture floor (§4.1)
  consolidation:                     # the CLS slow pass (§7)
    enabled: true
    idle_minutes: 10.0
    staleness_hours: 6.0
    max_active_facts: 256            # SOFT cap, trimmed by demotion
    decay_half_life_days: 30.0
  predictive_gate:                   # the predictive layer (§4.2)
    enabled: true                    # collect-only scorer + user-signal detector
    write_live: true                 # PredictiveWriteGate live writes — the KILL lever
    min_prior_n: 5                   # cold-start floor
    latency_weight: 0.5
    size_weight: 0.25
    lexicon: { … }                   # zero-NLU trigger families (TriggerLexiconConfig)
```

---

## 12. Cross-references

| Topic | Spec |
|-------|------|
| Event types (`Observation`, `Action`, `MemoryGateFired`, `ExpectationAttached`, `OutcomeObserved`, `SurpriseScored`, `StuckRecovered`), EventBus | `01-event-bus.md` |
| Config models, component registry | `06-config.md` |
| Context window, compaction, the cruncher | `08-context-management.md` |
| Threat model, architecture principles | `SECURITY.md`, `00-architecture-overview.md` |

> Specs 05 (this doc) and 08 predate parts of the ContentStore/eviction subsystem; where
> they and a source docstring disagree, **the docstring is ground truth**.
