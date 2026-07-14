# Changelog

All notable changes to LocalHarness are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: interfaces may change).

## [0.9.7] — 2026-07-14

Five bugs (#79–#83) found by forensic analysis of a failed live replay session — two of
four user turns died in stuck-loop escalations, and the run's own event ledger showed
why: the harness misled the model (a bash tool that wasn't bash, "success" lines for
writes that changed nothing), then its stuck failsafe warned off stale evidence, forgot
its own warning, and killed turns without a summary. All five fixes are deterministic
mechanisms, filed first and fixed test-first. They are designed to break repeat loops
and end unrecoverable turns honestly; live validation is a single replay (n=1) still
pending at release time — no reliability improvement is claimed beyond the mechanisms
described.

### Fixed
- **`bash_exec` now actually runs bash** (#79): it executed via the platform `/bin/sh`
  (dash on Debian/Ubuntu), so bashisms silently misbehaved — observed live as
  `mkdir -p …/{a,b}` creating a literal junk `{a`-named tree with exit 0, corrupting the
  model's picture of its own workspace from its first action. Note: commands that relied
  on strict POSIX-sh behavior now get bash semantics.
- **Write tool reports create/overwrite/no-change honestly** (#80): overwrite results are
  now `Created … (N bytes)`, `Overwrote … (was M bytes, now N bytes)`, or — for
  byte-identical content — an explicit `No change: … do not rewrite it; take the next
  step.` with `unchanged=True`. Previously every write returned the same
  `Written N bytes` success line, so a model rewriting an identical file saw progress
  each time (observed live: the same bytes written 3× in each of two failed turns).
  Anything parsing the old overwrite message must adapt; `mode=append` is unchanged.
- **Stuck detector gives a clean slate after warning** (#81): the repeat-detection window
  is now cleared when a recovery warning fires, and escalation requires fresh evidence —
  the same signature repeated again post-warning, or more than `max_nudges_per_turn`
  (new config, default 3) warnings in one turn. Previously the window was never cleared:
  a model that complied with the warning was re-warned on stale counts, and one further
  repeat killed the turn. Behavior change: uniformly-identical repeats now escalate one
  iteration later (warn at 2, clean slate, escalate on the post-warning repeat).
- **Recovery warnings persist in the conversation** (#82): the nudge was appended only to
  the immediate LLM request and vanished from context the next iteration — session
  ledgers showed the model replying to a warning that wasn't in its history. It is now a
  durable user-role message; history consumers will see it.
- **Stuck escalation ends with the model's partial work, not a dead notice** (#83): on
  escalation the agent now gets one forced tool-less wrap-up ("what's done, what
  remains, next step" — mirroring the existing budget-exhaustion summary) before the
  turn fails; the plain notice remains as fallback on provider error. Costs one extra
  LLM call at escalation time. Turn telemetry (`TurnFailed`, reason `stuck_detected`)
  is unchanged.

## [0.9.6] — 2026-07-14

Shaped by a maintainer live-dogfooding session: four bugs found in real use (#75–#78,
including two mined from the session's own event ledger) plus the first slice of the
during-turn UX work. All receipted and test-first as before.

### Added
- **In-turn progress narration** (terminal): during a long multi-stage task, the model's
  short stage statements now render as dim lines opening each chunk of tool activity —
  "· Pulling the data… ◆ read … · Now building the model…" — at most one per chunk, first
  line only, 160-char cap, with the final answer still rendered exactly once in its panel.
  A one-line prompt nudge encourages the model to announce stage transitions; the render
  is the mechanism, the nudge only shapes wording.

### Fixed
- **Truncated tool calls are never executed** (#77): when a completion hits the
  output-token ceiling mid-tool-call, the loop previously executed the mangled call —
  yielding either a confusing validation error retried blind, or a "successful" silently
  truncated file, at ~a minute of dead air and a full context resend per retry (a real
  session burned ~2M tokens in one such turn). The finish reason is now captured (it was
  read nowhere) in both tool-call modes; a length-truncated response's calls are
  suppressed and the model is told exactly what happened and to write in smaller pieces.
  The write tool's description now steers large files toward `mode=append` chunks.
- **Memory consolidation defers while a turn is in flight** (#78): "activity" previously
  meant new user messages only, so a long agent turn could have the background pass (and
  its embedder) running concurrently with live inference on the same GPU — observed for a
  full 194 seconds inside an active turn. Turn start now defers/cancels the pass exactly
  like typing does, and idleness is measured from turn END (it was measured from the
  user's message, i.e. turn start).
- **The agent knows its working directory** (#75): the system prompt stated the date but
  never the location, so "make a folder for yourself" landed in `$HOME` instead of the
  project directory the session was launched from. The prompt now states the launch
  directory with guidance to create files under it unless told otherwise.
- **The embedding model loads once and silently** (#76): the tag-discovery embedder was
  reconstructed — weights reloaded — on every idle pass, and its loader chatter
  (HuggingFace/torch/tqdm write directly to the terminal, bypassing logging) printed over
  the interactive input box. It now loads once per process inside an output-suppression
  guard; the quiet "· dreaming…" status is the only visible sign of background memory work.
- **Cleaner tool counter lines**: the burst counters dropped their decorative description
  suffix (`◆ web_search · web_fetch · 23/23`); the separate untrusted-web-results security
  note is unchanged.

## [0.9.5] — 2026-07-14

The third audit wave: 13 fixes (#62–#74) closing out every verified finding from the
post-ship audit — including the memory (chapter-writer) batch that needed maintainer
design rulings. Same discipline: each bug filed before its fix, fixed test-first, closed
by its fixing commit.

### Fixed
- **Healing a memory chapter can no longer steal another chapter's identity** (#64): when
  the staleness recheck rewrites a chapter, the rewrite previously re-picked which chapter
  to replace by best overlap with no preference for the original — on a tie (common, since
  a chapter overlaps itself perfectly) the healed content could take over a near-duplicate's
  identity while the original's history dead-ended. The heal now hard-prefers the original's
  key and only moves to another chapter when strictly better, always recording a successor.
- **The chapter-writing kill switch now kills** (#65, maintainer-ruled): setting
  `schema_writer_enabled: false` stops BOTH chapter-writing paths — the writer AND the
  staleness recheck, which previously kept minting/rewriting chapters with the "kill lever"
  off. The recheck keeps its own sub-switch for fine control when the master is on.
- **Chapter bookkeeping integrity** (#66, #67, #69, #70, #71): containment and
  refresh-adoption now score against the same ACTIVE-primary member sets — dead members and
  auxiliary rows no longer skew the comparisons that decide folding vs minting (the
  duplicate-sibling-chapter class); one claimed-key set is shared per consolidation pass
  (the writer can no longer undo the recheck's heal in the same pass); chapters superseded
  mid-pass are skipped instead of redrafted; and member maps are built once per pass
  instead of per write (thousands of redundant queries removed).
- **The chapter recheck no longer starves** (#68): a healthy revalidation now advances the
  chapter's recheck cursor, so the oldest-first window rotates across the whole population
  instead of re-checking the same few forever. Disclosed behavior note: decay and ranking
  read the same timestamp, so a chapter that keeps passing revalidation now also stays
  fresher — revalidation counts as corroboration, consistent with how folds already work.
- **Chapter redrafts read history once, asynchronously** (#72): previously each redraft
  performed two synchronous full-file history reads on the event loop.
- **Doomed inference requests fail fast; queue waits are visible and bounded** (#62,
  maintainer-ruled semantics): a dead endpoint is now detected by a sub-millisecond TCP
  probe BEFORE entering the inference queue — previously such a request could wait 90+
  seconds behind unrelated work with only a generic spinner. Waiting for a busy-but-healthy
  server logs its state after ~2s and is bounded by a new, generous
  `provider.inference_queue_wait_seconds` (default 600, 0 disables) — the ceiling applies
  to the WAIT only; a generation in flight is never timed out.
- **Delegation outcomes are receipted** (#73): every `agent` tool completion now renders a
  truthful terminal line derived from the tool result — `◆ agent <name> — completed` or
  `— FAILED: <reason>` — so a failed delegation is visible regardless of how the model
  narrates it (observed live: a failure presented as a sub-agent success). The delegation
  tool's description now teaches passing a self-contained task, with worked examples.
- **The glob tool finds files under a trailing `**`** (#74): Python's `pathlib` returns
  directories only for patterns ending in a bare `**`; the tool now normalizes these, and
  its description discloses that `~` is expanded and relative patterns root at the process
  working directory. This combination had let the assistant confidently tell a user their
  own agent was "a built-in" that couldn't be edited.
- **Internal eval-sweep token pollution** (#63 — dev-only script, not shipped in the
  package): dispute-bookkeeping prefixes are stripped before the stale-token comparison,
  preventing false retractions of freshly mined facts containing ordinary words like
  "user".

## [0.9.4] — 2026-07-14

The second wave of the same audit (see 0.9.3): seven agent-lifecycle bugs found by driving
the full create → see → use → persist → collide → escape journey against a live server
(#55–#61), each filed before its fix and fixed test-first.

### Fixed
- **`localharness agent create` refuses to overwrite an existing agent** (#55): it
  previously reported "✓ created" while silently replacing the file — erasing, in the live
  test, the user's explicit read-only tool restriction. Now: an explicit error naming the
  path, exit 1; `--force` opts into replacement. This brings the CLI path in line with the
  invariant the conversational flow already enforced.
- **Answering "change" at the creation confirm prompt now actually changes something**
  (#56): the follow-up is appended to the stored description so both the original intent
  and the correction reach the generator — previously the correction was silently discarded
  and an identical config regenerated. A too-short first description no longer wedges the
  wizard permanently.
- **Generated YAML is validated before you are asked to approve it** (#57): validation
  previously ran only at deploy, so you could approve the product's own output and then get
  a raw Pydantic error wall (complete with a pydantic.dev link). Now an invalid generation
  triggers one regeneration with the error fed back, then a truthful abort; errors are
  compacted to `field: message`; and the generation prompt states the legal values for the
  enum fields it names, derived from the models so they cannot drift.
- **A newly created agent exists immediately** (#58): deploy now registers it into the live
  session — `/agents` lists it and the delegation tool advertises it — where previously it
  was invisible and unreachable until restart. Covered by unit/integration tests (not a
  live-model run); editing an existing agent's YAML by hand still needs a restart.
- **You can escape the creation wizard like a human** (#59): leading cancellation phrases
  ("never mind", "forget it", "actually, cancel", …) deterministically cancel — previously
  only four undocumented exact-match words worked, and "actually, never mind, forget it"
  became the agent's description. Descriptions that merely contain such words ("an agent
  that helps me cancel subscriptions") are unaffected, and the escape is now advertised in
  the wizard's own prompts.
- **`/quit` mid-wizard cancels the wizard, not your session** (#60): it previously
  hard-exited the entire session silently while bare "quit" cancelled safely. The first
  `/quit` now cancels creation and says so; a second one exits.
- **The CLI spec no longer documents commands that don't exist** (#61): `agent run` and
  `agent delete` are marked as planned (design retained), and a note documents how to edit
  or remove an agent today.

## [0.9.3] — 2026-07-14

A hardening release: 28 bugs found by a systematic post-ship audit — adversarial code
verification plus scripted "live user journey" testing against a real server — each filed
as a GitHub issue before its fix (#27–#54), fixed test-first, and closed by the fixing
commit. No new features beyond a `--version` flag and honest new warnings.

### Added
- `localharness --version`: prints the installed version.
- `localharness model` now **warns when your configured default model is not among the
  models the server is actually serving (or you have downloaded)** (#50) — previously the
  list simply showed no `[active]` marker and said nothing, so a default that had drifted
  from reality was invisible until `start` failed.
- `localharness model --config-dir` for parity with every other command (#35).

### Fixed
- **Conversational agent creation is now honest, and the schema gap behind most of its
  failures is fixed** (#33, #27, #29, #28): the YAML-generation prompt now states the real
  config contract — required fields, allowed keys, and the nested `tools`/`permissions`
  shapes, all derived from the Pydantic models so prompt and schema cannot drift. Failures
  are truthful ("Agent was NOT created …") instead of the previous unconditional "Agent
  created." after a failed deploy; a provider error during generation no longer kills the
  whole session; a nameless config fails explicitly instead of deploying as a silent
  "new-agent" placeholder that overwrote its predecessor. Live-tested against a real model:
  the truthful-failure fixes held on every attempt; the schema fix turned a reproducible
  0-of-4 deploy failure into a working generation — but generation is still sampling-based,
  so a model can occasionally emit invalid YAML on a given attempt (you now get the truth
  when it does).
- **`localharness start` no longer hangs forever on a startup failure** (#43): any hard
  failure after the memory store opened (e.g. the configured model isn't served) used to
  leave a non-daemon database worker thread alive, hanging process exit indefinitely. The
  whole startup window is now covered by ordered teardown — the same scenario exits
  non-zero in seconds.
- **Startup actually checks whether your model is reachable** (#44): the capability probe's
  failure result was computed and then ignored, so the "Cannot reach model" guard could
  never fire and the eventual error blamed the tokenizer. Startup now fails fast, names the
  real cause (model not served vs endpoint unreachable), and points at `localharness doctor`
  / `localharness model`.
- **A `/model` switch is durable and honest** (#30, #31, #32, #34): a failed tokenizer
  rebind restores the previous binding and tells you in-channel (previously it reported
  success and every later turn errored until restart); the context-window budget is re-read
  from the server per switch (a 128K→32K hot-swap no longer keeps the old ceiling and 400s
  mid-session); the probe runs off the event loop (no more multi-second UI freezes); and a
  managed-server switch persists the server's launch model, so the next cold start boots
  what you switched to.
- **`--config-dir` now truly isolates an instance** (#35): the user overlay, kill-file,
  audit log, and REPL history all previously resolved to hardcoded `~/.localharness` paths
  regardless of `--config-dir`, so two instances silently shared (and clobbered) state. One
  resolution rule now applies everywhere (explicit flag → `LOCALHARNESS_DIR` →
  `LOCALHARNESS_HOME` (legacy) → `~/.localharness`); defaults are unchanged for
  single-instance setups. Also: the model-switch pin warning now catches division-level
  pins (#36), an audit-log write failure is no longer misreported as a failed persist
  (#37), a reachable-but-malformed server response is no longer diagnosed as "Is it
  running?" (#38), and `localharness model ""` is rejected instead of silently persisting
  an empty default (#39).
- **Correcting one memory can no longer corrupt a different one** (#45): a user correction
  used to dispute whichever fact was most recently *retrieved*, with no check that it was
  related — silently downgrading an unrelated fact below the injection threshold. A
  correction now only disputes a content-related staged fact, and quarantines otherwise.
  **Reconciliation — the only repair path for disputed facts — now runs before the heavy
  cancellable steps** (#46), so ordinary typing no longer starves it forever.
- **Ctrl+C during generation cancels the turn, not the session** (#47) — previously the
  safe case (idle) was absorbed while the case you actually reach for (mid-generation)
  killed the whole session. A second Ctrl+C while cancelling still hard-exits.
- **Unknown slash commands are rejected instantly and deterministically** (#48) — `/typo`
  no longer gets sent to the model as chat to improvise an answer.
- **The first-run hint actually reaches interactive terminals** (#49): "Describe a task,
  or /help for commands." now renders inside the managed input box (it was fragile
  scrollback the box repainted over — visible in piped mode, missing in a real TTY, i.e.
  for every actual human). Returning sessions get a short `/help` reminder.
- **Internal eval-harness honesty** (#40, #41, #42 — dev-only scripts, not shipped in the
  package): the memory-quality grader could certify a run as passing even when one of its
  required checks had failed (its pass/fail gate and its failure reporting had drifted
  apart); its regression comparator could flag a false regression on a run the grader had
  already excused for a known, disclosed reason; and its dead-server detector could misfire
  on ordinary reply text containing words like "refused" — it now matches connection-level
  failure signatures only.
- **Docs and messages that lied** (#51, #52, #53, #54): the first-run pointer no longer
  links a 404 page; `init --help` documents the real probe order (`:8000` — vLLM's default
  — was omitted); `doctor`'s missing-agents-directory remedy now works (`init` creates the
  directory); `validate --strict` discloses that it is reserved instead of silently doing
  nothing. Plus: `model` fat-finger help (case-insensitive "did you mean", numeric-range
  hints) and a README row for `model`.

## [0.9.2] — 2026-07-12

### Added
- **Security defaults now update themselves on first start after an upgrade** (#15).
  `init` bakes the fully-resolved `org.permissions.deny_patterns` into your config.yaml,
  so a later growth of the shipped default deny list (v0.9.1 grew it 7→24) used to require
  a manual re-run or hand-edit — the gap the v0.9.1 notes disclosed. Now the first
  `localharness start` after a package upgrade folds the new defaults in automatically:
  additive only (your own entries are never removed or reordered, no other key is touched),
  a timestamped backup is written first, and it is **revision-stamped** — so a default you
  deliberately deleted is respected and never re-added, and an up-to-date config is a silent
  zero-cost no-op. Run `localharness config migrate --dry-run` to preview it, or
  `localharness config migrate` to apply it explicitly (same engine).

### Fixed
- **Background memory work no longer prints into the REPL — it shows as a quiet
  "· dreaming…" status instead** (#20). When a session starts (or goes idle) with pending
  work, the memory consolidation/mining pass used to spill its internal progress and
  warning lines into the interactive prompt — landing over the input box before you had
  typed anything, so it read as "something is broken." Those details now go to a log file
  (`<agent-dir>/memory.log`) instead, and while a pass runs the terminal shows a single
  unobtrusive `· dreaming…` spinner that clears the instant the pass ends, you start
  typing, or a new turn begins — it never draws over the input box (it can only appear when
  the prompt isn't up). Terminal-only: Discord, bench, and eval channels are unchanged. The
  leak was Python's default last-resort stderr handler surfacing the memory subsystem's
  `WARNING`/`EXCEPTION` log records, since the interactive start path configured no logging
  handler of its own.
- **Every LLM call path now streams at the transport level** (#18). Timeouts and
  cancellations no longer leave the local server generating into the void: with a
  whole-response request the client read-timeout races the *entire* generation, and a
  cancel leaves the engine decoding up to `max_tokens` of wasted GPU with no client
  listening. Two defects are closed. (1) XML tool-call mode was silently non-streaming —
  `_complete_xml` accepted a `stream` flag and ignored it, so any model whose capability
  probe fell back to XML mode lost streaming across the whole agent loop, with no log
  signal. (2) Four call sites still sent whole-response requests and now stream: the idle
  memory-consolidation adapter (cancelled by design on every idle→active transition, so it
  fired constantly), the compaction summarizer, the autoresearch proposer, and REPL
  agent-creation. Streaming makes the read-timeout apply between chunks and a client
  disconnect observable mid-generation. Present since the first release (the idle-
  consolidation path since v0.9.0).
- **Directory `grep` is bounded and returns partial results instead of hanging** (#21).
  A directory search used to materialize the entire recursive tree (`sorted(rglob(...))`) with
  no exclusions and read every file whole — including `.git`, virtualenvs, caches, and multi-GB
  binaries — so a `grep` at a repo root hung the full tool timeout and returned nothing (observed
  live: four `path: "."` greps each burned the 35 s timeout for zero output). The walk is now an
  iterative `os.scandir` that prunes hidden and vendor/VCS dirs (`.git`, `.venv`, `node_modules`,
  `__pycache__`, caches, `dist`/`build`) at traversal time, skips files over 1 MB and binaries
  (NUL sniff), and stops at a 20 000-file / 20-second budget — returning the matches found so far
  with an explicit `... (scan capped: …)` note and `truncated=True` rather than dying silent at
  the harness timeout. Opt hidden/vendor dirs back in with `include_hidden=True`. Present since
  the tool's first release.
- **`localharness components set agent.<axis> <value>` now works instead of always failing**
  (#22). Every agent-scoped set — including the documented safety lever
  `components set agent.memory.consolidation.tag_grouping_enabled <true|false>` — died with a
  Pydantic `Extra inputs are not permitted` error before writing anything. The set validated the
  whole merged config against `HarnessConfig`, whose top level has no `agent` key (agent axes live
  in the separate `AgentConfig` model), so the advertised mechanism was structurally dead for every
  agent axis even though `components get`/`list` displayed those axes fine. The set now routes
  `agent.*` validation through `AgentConfig` and writes an `agent:` section into the user overlay
  (`~/.localharness/overrides.yaml`) — the same layer `components get` reads back, and one that
  `ConfigLoader.load_agent()` now deep-merges as a LOW-priority default beneath each agent's own
  yaml (per-agent config still wins). `load_harness` and the harness-path validation exclude that
  `agent:` section (it is not a `HarnessConfig` field), so `org.*`/`provider.*` sets are byte-
  unchanged even after an agent axis has been set. Scope note: `bench`/eval and programmatic
  subagents build `AgentConfig` directly and do NOT read the overlay — only the live
  `localharness start` load path honors it. Present since `components set` first shipped.

### Changed
- **Tag-keyed memory grouping ships built-but-OFF, honestly.** This release contains the
  "tags become grouping truth" re-key (memory folding and correction validity keyed to the
  validated tag axis instead of the free-text topic word, so a wrong topic word cannot merge
  unrelated facts or authorize a correction) behind
  `agent.memory.consolidation.tag_grouping_enabled` — **default `false`**. Its pre-committed
  regression gate fired on the first live proof: with tag-keying on, a correction whose tag
  classification differs from its target's stopped superseding the stale value (a real
  reconciliation gap the old topic-word keying caught). Per the pre-committed kill, the
  mechanism ships dormant: `false` is byte-identical to the previous behavior (test-pinned on
  both re-keyed paths), and the flag is the re-attempt surface. Also included, inert until
  then: a backup-guarded, idempotent, bounded-revert tag backfill script and a reproducible
  grouping-regression comparator.
- Clustering cleanup: the dead slug-based grouping helper was removed; chapter membership has
  derived from validated tags (not topic words) since v0.9.0 — now regression-locked both ways.

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
