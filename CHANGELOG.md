# Changelog

All notable changes to LocalHarness are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: interfaces may change).

## [0.9.1] — 2026-07-12

### Security
- **Default deny list grows destructive service/process commands** (#15). New default-denied
  `bash_exec` forms: `docker stop/kill/rm`, `systemctl
  stop/disable/kill/mask`, `pkill`, `killall`, `kill`, `shutdown`, `reboot`, `poweroff`,
  `docker compose down` / `docker-compose down`, plus embedded forms of `rm -rf` and `sudo`
  (e.g. inside `x && …` chains). Read-only ops (`docker ps`, `systemctl status`, …) stay
  allowed. Found in live use: an agent stopped its own vLLM inference containers mid-run.
  **Behavior change:** agents that legitimately need these commands must now re-allow them
  via an explicit `org.permissions.deny_patterns` override in the root config (division-
  and agent-level config can only narrow, never re-allow). **Existing installs do not pick
  the new list up automatically:** `localharness init` writes the fully-resolved deny list
  into `~/.localharness/config.yaml`, so an already-generated config keeps its old
  7-pattern list until you re-run `init` or add the new patterns to
  `org.permissions.deny_patterns` by hand.
- **The shipped `sudo` deny pattern had never matched.** `bash_exec(sudo:*)` required a
  literal colon after `sudo`, so this defense-in-depth layer never fired on a real command
  in any release. Corrected to `bash_exec(*sudo *)`, with tests that replay real commands.
- **Opt-in workspace fence: `permissions.workspace_root`.** When set, `write`/`edit`
  targets and `bash_exec`'s `working_dir` must resolve inside it (symlink-safe via
  `resolve()`), and a confined bash call's default working directory becomes the workspace
  root. Off by default (`None`) — existing file-write/run behavior is unchanged.
  Motivating incident, also from live use: model-authored files landed inside the
  harness's own repo checkout. The repo's own eval scripts now run their subject agent
  confined to a per-run scratch directory.
- Honest limits: deny patterns match raw argument strings (no shell parsing — a `cd`
  inside a single command string is not caught); the no-OS-level-sandbox gap is unchanged
  and documented in SECURITY.md. The deny list and the fence are policy layers, not a
  sandbox.

### Fixed
- **`doctor` probed `/v1/v1/models` and reported the model check green anyway** (#16). The
  model-availability probe appended `/v1/models` to a base URL that already ends in `/v1`
  (and `/api/tags` onto the `/v1` base for Ollama, whose tags endpoint lives at the server
  root), and a non-2xx response still passed. Doctor now builds probe URLs from the
  stripped server root and fails the model check on HTTP errors. Present since the first
  release.
- Removed the dead `DEFAULT_DENY_PATTERNS` constant (`config/defaults.py`) — it referenced
  a nonexistent tool name and was consumed nowhere; the live defaults are the
  `PermissionConfig` field defaults.

### Changed
- **Memory: `mining_novelty_fold_threshold` default 0.5 → 0.70** — the parameter-sweep
  winner. Folding is more conservative: near-duplicate facts still merge, distinct facts
  are safer from erroneous merges.
- Eval harness: the subject agent now runs with service-ops commands denied and a confined
  per-run workspace; the eval loop's context bound was raised 32,768 → 131,072 tokens (the
  served window) — the old 32k cap dated to a hardware fault since resolved by RMA.

## [0.9.0] — 2026-07-11

### Added
- **Amortized re-mining (the "residue ledger").** Transcript records the extractor saw but
  never used are queued and re-mined during later idle passes in small isolated chunks,
  with a bounded attempt cap that retires exhausted entries — nothing is deleted, coverage
  becomes recoverable instead of lossy (schema v7).
- **Mining coverage telemetry.** Every consolidation pass reports records seen, cited,
  queued, re-mined, rescued, and retired.
- **Novelty gate.** A re-extracted fact folds into its existing atom (strict token-subset +
  matching numbers + similarity threshold) instead of minting a duplicate — deliberately
  conservative so distinct facts never merge.
- **Self-echo guard.** Only user-authored evidence advances a fact's confidence — the model
  restating its own claims no longer reinforces them.
- **Embedding-similarity clustering edge, two-factor.** Topics can relate via embedding
  cosine similarity, but never on similarity alone — an edge still requires a shared
  salient token.
- **Chapter refresh.** A topic chapter keeps its identity as membership drifts — a grown
  cluster supersedes its old chapter instead of spawning a duplicate.
- **Cluster hygiene.** A chapter can no longer be absorbed into a cluster containing its
  own member atoms.
- Eval protocol: configurable idle passes; facet-aware chapter grading.

### Changed
- Terminal view: tool calls render as compact one-line summaries instead of full bodies;
  all dynamic text is markup-escaped so bracketed content displays verbatim.
- Production mining budget default raised to 50 records/pass; write budgets configurable
  to 10k.

### Fixed
- A correction now retires the stale fact even when the corrected value already exists in
  the store (the duplicate-skip previously dropped the supersede silently).
- The supersede directive is recognized in any field position of an extraction line, not
  only the canonical last field.
- Terminal display corrupted bracketed text by interpreting it as style markup; misleading
  result line counts removed.

### Known issues
- Under a stricter grounding re-grade (supermajority instead of majority) the passing
  validation verdict flips; the standard-rule pass is disclosed as such by the eval.
- Long-output models can pressure the context-compaction budget (emergency truncations
  observed); compaction rework is queued.
- Occasional recoverable HTTP 400s on parallel tool calls with larger models.
- Tool-grounded and ungrounded numeric claims currently receive equal memory trust; a
  source-reliability gate is planned.

## [0.8.1] — 2026-07-09

### Fixed
- **The live bench suite can no longer pass vacuously.** Three open live-suite bugs shared
  one disease: with a misconfigured vLLM endpoint the suite could run to GREEN while every
  run errored (#3, #4, #5). The synthetic gate corpus scored its success rubric with an
  empty `contains:` needle — `"" in text` is always true, so a run that died before
  emitting a token still scored 1.0; the corpus now demands a literal the prompt tells the
  model to emit, and `SuccessCriteria` rejects any empty-needle `contains:`/`regex:`/bare
  rubric at load time so the footgun cannot silently recur. The budget-cap test asserted
  only `success is False`, which any unrelated failure also satisfied — it now requires
  exactly the capped behavior: one action dispatched (an early-erroring run dispatches
  zero → RED, an uncapped run dispatches two → RED), the second step's effect absent. And
  the live provider baked its own model/endpoint defaults while the preflight resolved
  `base_url` from a different source, so an opted-in run could probe one server and test
  another — both now share a single `live_target()` resolver (`LOCALHARNESS_LIVE_*` env)
  that hard-fails loud when unpinned, and the preflight verifies the pinned model is
  actually served. Verified against a live endpoint: the full live suite is green when
  correctly pinned, and RED/loud on wrong-model, unpinned, and errored runs.

## [0.8.0] — 2026-07-05

### Added
- **The write gate now predicts.** Alongside the motif capture floor (a tool error that
  later resolved, a stuck loop that recovered), the harness now scores *every* tool outcome
  against that tool's own history — per-tool statistical priors computed in pure SQL, zero
  model calls, zero decode tokens. Each outcome falls into one of four quadrants — *routine*,
  *surprising failure*, *unsurprising failure*, *quiet surprise* — plus an honest *cold start*
  when a tool has too little history to predict from. A **surprising failure** (a
  normally-reliable tool erroring) becomes a captured memory; an **expected** failure stops
  being news — the junk the old motif gate had no way to suppress. The whole
  expectation→outcome→surprise reconciliation rides the event bus the harness already carries,
  as a sibling subscriber that changes no existing gate.
- **User corrections write quarantined, reversible facts.** A zero-NLU tripwire flags
  correction-class turns ("no", "i meant…", "actually…"). An explicit correction *phrase*
  scoped-supersedes the single most-recently-staged fact it was most likely about — the full
  original text is preserved behind a dispute marker, never truncated or erased — while every
  other in-scope correction writes a standalone quarantine fact from the user's own words.
  All reversible through fact history; all keyed so a repeat corroborates instead of
  duplicating.
- **Everything new is capture, not behavior.** Every write in this tranche lands *below* the
  injected-memory visibility line (confidence ≥ 0.7 **and** retrieval-strength ≥ 0.2): the new
  facts live in the store and are searchable, but they do not enter the prompt — and so do not
  change what the agent *does* — until an idle consolidation pass confirms them. Correction and
  disputed rows are held back from promotion pending a deeper reconciliation pass. One config
  lever (`predictive_gate.write_live`) reverts the entire live-write path to motif-only capture
  while the scorer keeps logging telemetry — a pre-committed KILL.
- **Proven by offline replay.** Replaying one real bus-events trace through both gates, the
  predictive gate's capture recall on the surprising-failure population is **73/73** versus the
  motif gate's **68/73** — measured directly against the population, independent of any
  hand-label proxy. The junk-write-rate comparison is methodology-sensitive (it depends on the
  grading applied — under the strictest re-grade the precision comparison inverts, while the
  recall result is grading-independent), so both gradings ship in the repo's report tooling
  (`scripts/gate_replay_comparison.py`) rather than as one headline number.

### Known limits (named, not hidden)
- **Correction detection is lexical.** The trigger lexicon is a recall-first tripwire, not a
  classifier — measured recall ~23% on a real hand-labeled census, so corrections that don't
  use a trigger word are missed. A model look is future work; a false trigger costs one logged
  record, a miss costs a missed correction.
- **Expected-failure suppression and stuck-recovery are synthetic-proven.** The real trace
  carries zero unsurprising-failure and zero stuck-recovered events; both are covered by unit
  tests, not a real occurrence — disclosed, not declared won.
- **Statistical priors are per-tool and silent below 5 observations** of a tool (neutral
  score, cold-start quadrant); the motif capture floor covers from turn one.
- **The injected block still does not steer the first move.** New writes sit below the
  visibility line, so they capture — they do not yet change the agent's opening action.

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
