# Changelog

All notable changes to LocalHarness are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: interfaces may change).

## [0.7.0] — 2026-07-04

### Added
- **Sessions are real now, and the next sitting remembers the last one.** Each run of
  `localharness start` is a "sitting" with its own session record. When it ends, the
  harness derives a one-line, payload-first summary of what actually happened — the error
  you resolved, the stuck loop it recovered from — and saves it to the agent's memory
  (a sitting with nothing of substance is suppressed, never padded with filler). A fresh
  sitting can then answer "what did we do last time?" straight from its injected memory
  block, with zero tool calls. Summaries the harness writes when it compacts context now
  persist too, so what got compacted away isn't lost to the next sitting. This closes the
  0.6.0 known-limit "session-history recording is not wired yet."
- **Session summaries carry the topic, and the shelf carries the timeline.** The
  end-of-sitting summary now leads with a trimmed slice of what you asked
  (`asked: "any fun events this weekend…" — 3 turns, 1 delegation`) whenever there is no
  error-lesson to lead with — a pure conversation is no longer invisible to memory. The
  injected "Recent Session History" renders each sitting with a relative day and clock
  time (`- today 11:47am: …`), newest first, hard-capped at 8 lines; older sittings stay
  in the store and remain searchable. Rendering reads the sessions table directly — the
  on-disk format is unchanged, and the block stays byte-stable within a day (the labels
  flip with the existing daily date change, preserving the prefix cache). Still zero
  model calls end to end; empty sittings still write nothing.
- **A live "thinking…" indicator in the REPL.** While the model is generating, the
  terminal shows an animated spinner that clears the instant real output arrives — a
  tool-call line, the answer panel, or the input prompt. Replaces the occasional static
  "Working…" line, so long generations no longer look like a hang.

### Changed
- **The root agent is now the orchestrator, by name.** `localharness start` creates and
  selects the root agent as `orchestrator` (was `default`), matching the architecture's
  own vocabulary. Existing installations migrate automatically on first start and keep
  every memory: `agents/default.yaml` becomes `agents/orchestrator.yaml`, and the memory
  store (facts, sessions, MEMORY.md, history) is adopted under the new name — one-time,
  idempotent, crash-safe. If you already have your own agent named `orchestrator`, the
  migration refuses (nothing is merged or overwritten) and the root keeps its old name;
  a console warning explains how to resolve. `--agent default` redirects to
  `orchestrator` with a note. Direct subagent addressing is unchanged, except the
  name `orchestrator` is now reserved for the root (mirroring the old `default` guard).

### Fixed
- **Conversational turns no longer surface meta-narration or duplicated answers** (#6).
  The act-guard — the nudge that turns "I'll go look that up" into an actual tool call —
  used to ask the model to restate its final answer when no tools were needed, which
  could produce a narrated duplicate ("You're right, my previous reply was just a
  conversational response…") in the output panel. The nudge now requests a sentinel the
  harness swallows, and the original reply is delivered unchanged. The leak had been
  live since v0.5.0.

## [0.6.0] — 2026-07-03

### Added
- **Persistent memory that learns from experience.** Agents now write, rank, consolidate,
  and retrieve long-term memory automatically — no memory prompt-engineering required:
  - **Two write paths.** A `remember` tool for deliberate saves, plus an automatic write
    gate that captures lessons from prediction-error signals the harness already emits —
    a tool call that failed then later succeeded, a stuck loop that recovered — at zero
    additional model calls.
  - **Nothing is overwritten.** A conflicting fact *supersedes* the old one (still
    retrievable on explicit request); every fact carries provenance to its source episode.
  - **Ranking at zero token cost.** Activation scoring (used-often + used-recently,
    ACT-R-inspired) runs in-query inside SQLite to decide which facts earn prompt space.
    Trust and accessibility are tracked as separate numbers.
  - **A byte-stable injected index.** The in-prompt memory block is an index (fact names +
    one-line payload-first descriptions) whose bytes are untouched by reads and retrieval —
    reads accumulate in a staging ledger instead of reordering anything. The bytes change
    only at consolidation folds, genuine new writes (a `remember`, a supersede), and day
    boundaries — preserving the inference server's prefix cache between those points.
    Measured on reference architecture A: one changed byte near the top of a 32k-token
    prompt costs ~16 s of time-to-first-token, so this stability is load-bearing, not
    cosmetic.
  - **Idle consolidation ("sleep").** When the user goes idle (or a session starts with a
    stale watermark), a background pass promotes lessons seen in ≥2 episodes (or one
    salient recovery) into the prompt-visible tier, decays unused facts out of the index
    (never out of the store), and trims a soft capacity cap by demotion — never silent
    deletion. A user message cancels an in-flight pass immediately.
  - **Hierarchy + structure-aware search.** Over-window document analyses persist their
    gist/schema trees, and `memory_search` routes through the graph neighborhood — gists
    route the search, leaf records anchor the answer. The 0.5.1 number-provenance net
    extends to memory: a figure in a promoted gist that appears in no source leaf is
    flagged.

### Changed
- **Local inference is serialized by default.** Completions against a local endpoint queue
  through a two-layer inference gate — an in-process semaphore
  (`LOCALHARNESS_MAX_CONCURRENT_INFERENCE`, default 1) and a cross-process per-endpoint
  lock (`LOCALHARNESS_INFERENCE_LOCK=0` disables) — held for the full request including
  stream consumption. Motivation: a 2026-07-02 hard-freeze postmortem on unified-memory
  hardware — concurrent prefills can push the GPU driver into allocation failure that no
  OOM killer sees. Decode on one GPU is engine-serialized anyway, so serial costs almost
  no wall-clock. The cruncher's map stage follows the same default
  (`LOCALHARNESS_CRUNCHER_MAP_CONCURRENCY`, was hard-coded 8).
- **`web_fetch` downloads are streamed and capped** at `LOCALHARNESS_FETCH_MAX_BODY_BYTES`
  (4 MB default) — unbounded response bodies were a whole-box OOM vector via one
  pathological URL. The cap is surfaced in the retained page, never silent.
- The cruncher's chunk-size cap doubled (16k → 32k tokens) after a chunk-size/quality
  knee measurement.
- The injected memory block's index render is now activation-ordered,
  retrieval-strength-gated, and payload-first. (Index-mode injection itself — names +
  descriptions in the prompt, full bodies on demand via `memory_get`/`memory_search` —
  was already the default.)
- The 0.5.3 FTS5 MATCH quoting is generalized into a full query sanitizer: operator
  characters anywhere in a `memory_search` query are literal terms, never syntax.

### Fixed
- Promoted lesson payloads survive every render layer (payload-first line rendering; the
  index line budget no longer truncates the discriminating content out of a lesson).
- The live-vLLM test suite resolves endpoint/model from `LOCALHARNESS_LIVE_MODEL` /
  `LOCALHARNESS_LIVE_BASE_URL` env vars instead of a hermetic-fixture placeholder that
  broke server-side token counting.

### Known limits (named, not hidden)
- Session-history recording is not wired yet; the injected block omits that section until
  it is (planned next milestone).
- The consolidation LLM-replay step ships wired OFF (built, gated, unit-tested — including
  cancellation of its in-flight generation the instant a user message arrives); 0.6
  consolidation decisions are heuristic (recurrence + salience), not model-judged.
- Auto-capture is motif-gated (resolved-error / stuck-recovery); a statistical surprise
  gate is future work.
- Injected memory currently improves *recovery* (the agent consults what it knows after
  context makes it relevant); steering the *first* move is explicitly future work.
- There is no user-facing forget tool yet: auto-captured facts can be superseded and decay
  out of the index, but not user-deleted; a curation surface is future work.
- A persisted gist's leaf pointers are session-scoped (content handles do not survive the
  session); the gist text itself persists and stays searchable.
- Live-verified end-to-end on reference architecture A (DGX Spark GB10, Qwen3.6-27B on
  vLLM); other setups are covered by the hermetic suite only.

## [0.5.3] — 2026-07-03

(0.5.2 is intentionally skipped — that number is already publicly attached to the
in-progress hierarchical-memory milestone on the devnotes page.)

### Changed
- **The default subagent roster is now quarantined-or-read-only.** `data-analyst` and
  `frontend-designer` no longer ship in the default roster: both hold `bash_exec`
  (host-dangerous), which sat uneasily next to the harness's fenced-by-construction
  security story, and neither had bench coverage. A live quality battery (2026-07-03,
  receipts in the repo history of this entry) found `frontend-designer`'s first-run
  build task hangs against an undeclared Playwright dependency; `data-analyst` passed
  its battery cleanly and was demoted on security posture alone. Both are preserved as
  fully documented opt-in configs under `examples/agents/` — drop one into
  `~/.localharness/agents/` to restore it. The remaining defaults: `explore`
  (read-only), `web-researcher` (web-quarantined), `search-verifier` (blind verifier),
  `cruncher` (grant-fed reducer), plus your own YAML agents.
- The grant-target safety gate and its tests now exercise host-dangerous CONFIG
  children (yaml allowlists) rather than host-dangerous builtins — there are none left.

### Fixed
- `memory_search` no longer fails on hyphenated queries ("no such column: in") —
  FTS5 MATCH input is tokenized and quoted, so operator characters in real-world
  queries (`built-in`, `000660.KS`, `P/GP`) are literal terms, never syntax.
- No-tool instant answers no longer terminate with a meta "I already provided the
  answer" summary. The act-guard and self-check prompts now state that only the
  latest reply is user-visible, and the self-check confirm path is a deterministic
  sentinel (`CONFIRMED`) whose summary selection walks back to the answer it
  confirmed — cheaper and loss-free versus asking the model to repeat itself.

## [0.5.1] — 2026-06-26

### Added
- **Number provenance for over-window document reading.** LocalHarness reads documents
  larger than the model's context window by fully reading every section and combining the
  notes — losslessly, on local hardware. As of 0.5.1, when that combine runs in multiple
  stages, every figure in the final answer is checked back against the verbatim per-section
  notes the model actually read, and any figure that doesn't trace back is surfaced. This
  keeps the numbers in an over-window answer anchored to their source — built for financial
  filings, contracts, and long reports, where a drifted figure is unacceptable.

  Verified by deterministic tests in both directions (through the real reduction path), an
  independent adversarial review, and a live run on a real 27B local model over a
  ~600k-character filing (0 of 24 figures flagged on a faithful answer; all 24 confirmed
  present in the notes). Surfaced as a warning, never a silent rewrite; engages on large
  multi-stage reductions; figure-matching is heuristic.

## [0.5.0] — 2026-06-25

### Added
- **Lossless, secured over-window context.** Cross-agent content grants (hand a child a
  handle, not re-inlined bytes) plus a "cruncher" subagent that reads an over-window
  document by handle and faithfully map-reduces it — every chunk fully read, nothing
  truncated — with structure-aware splitting and per-section context headers.
- A **capability floor**: untrusted-ingested web/tool content cannot co-reside with
  host-dangerous tools in the same agent (defense-in-depth above model refusal).
- Bench instrumentation for over-window eviction, plus a scored faithfulness scenario.
