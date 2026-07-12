"""All 12 Pydantic config models for LocalHarness."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS


class ToolConfig(BaseModel):
    """
    Tool access specification for an agent or division.

    Tool scope resolution order (most specific wins for conflicts):
      global built-ins → division additions → agent additions
      deny lists are evaluated last and always win.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    inherit: list[Literal["global", "division", "org"]] = Field(
        default=["global"],
        description=(
            "Which tool scopes to inherit. 'global' always provides the built-in tools "
            "(glob, grep, read, write, bash). 'division' inherits the division's tool list. "
            "'org' inherits the org-level tool list."
        ),
    )

    add: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names to add to this agent's accessible set. "
            "Must be registered tool names or MCP server tool names."
        ),
    )

    deny: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names to explicitly deny. Denial takes precedence over any 'add' or 'inherit'. "
            "Can also be a glob pattern, e.g. 'bash' or 'mcp_*'."
        ),
    )

    mcp_servers: list["MCPServerConfig"] = Field(
        default_factory=list,
        description=(
            "MCP servers to discover tools from. Tools from these servers are added "
            "to this agent's accessible set (subject to 'deny' list)."
        ),
    )

    @field_validator("inherit", mode="before")
    @classmethod
    def normalize_inherit(cls, v: Any) -> list[str]:
        """Accept string shorthand: inherit: division → inherit: [division]"""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("deny")
    @classmethod
    def validate_deny_not_global(cls, v: list[str]) -> list[str]:
        """Warn if user tries to deny a global built-in — they usually mean something else."""
        # Do not raise — just allow (user may intentionally deny bash)
        return v


class MCPServerConfig(BaseModel):
    """Configuration for an MCP tool server."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(
        description="Human-readable name for this server. Used in tool prefixing: {name}_{tool_name}.",
    )

    transport: Literal["stdio", "streamable_http"] = Field(
        description="MCP transport type.",
    )

    # stdio transport fields
    command: Optional[str] = Field(
        default=None,
        description="Command to launch the MCP server process (stdio transport).",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments to pass to the command (stdio transport).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set for the MCP server process.",
    )

    # streamable_http transport fields
    url: Optional[str] = Field(
        default=None,
        description="Base URL for the MCP server (streamable_http transport).",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers to include in all requests (streamable_http transport).",
    )

    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Timeout for tool calls to this server.",
    )

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError("MCPServerConfig: 'command' is required for stdio transport")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("MCPServerConfig: 'url' is required for streamable_http transport")
        return self


class BudgetConfig(BaseModel):
    """Execution budget constraints for one agent session."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    max_actions: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Maximum number of tool calls in one session.",
    )

    max_duration_minutes: float = Field(
        default=30.0,
        ge=0.1,
        le=1440.0,  # 24 hours
        description="Maximum wall-clock duration for one session.",
    )

    kill_file: Optional[str] = Field(
        default="~/.localharness/KILL",
        description=(
            "Path to the kill switch file. Agent checks for this file before each iteration. "
            "If it exists, the agent stops immediately and the file is removed. "
            "Set to null to disable the kill switch."
        ),
    )


class PermissionConfig(BaseModel):
    """
    Permission policy for an agent.

    Default mode is 'auto': allow everything except the deny_patterns list.
    An agent can only narrow its inherited permission policy, never broaden it.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    mode: Literal["auto", "manual"] = Field(
        default="auto",
        description=(
            "'auto': allow all tool calls except those matching deny_patterns. "
            "'manual': deny all tool calls that are not in explicit allow_patterns "
            "(v2 feature — not implemented in v1)."
        ),
    )

    deny_patterns: list[str] = Field(
        default_factory=lambda: [
            # --- credential / config file writes ---
            "write(*/.env)",
            "write(*/secrets*)",
            "write(*/config.yaml)",
            "write(*/agents/*.yaml)",
            # --- privilege escalation + recursive delete + world-writable ---
            # `sudo *` (bare) + `*sudo *` (embedded); the earlier `sudo:*` glob required a
            # literal colon after 'sudo' (fnmatch) and so matched NO real sudo command.
            "bash_exec(*sudo *)",
            "bash_exec(rm -rf *)",
            "bash_exec(*rm -rf *)",  # embedded: `cd /tmp && rm -rf x`
            "bash_exec(chmod 777 *)",
            # --- destructive service / process ops (issue #15: the 2026-07-11 run where the
            # subject bash_exec'd `docker stop` against its OWN vLLM server). Destructive VERBS
            # only — read-only ops (docker ps, docker logs, systemctl status, journalctl) stay
            # allowed. Leading+trailing * catches bare AND embedded (`cd x && docker stop y`).
            "bash_exec(*docker stop*)",
            "bash_exec(*docker kill*)",
            "bash_exec(*docker rm*)",  # covers `docker rm` and `docker rmi`
            "bash_exec(*docker compose down*)",
            "bash_exec(*docker-compose down*)",
            "bash_exec(*systemctl stop*)",
            "bash_exec(*systemctl disable*)",
            "bash_exec(*systemctl kill*)",
            "bash_exec(*systemctl mask*)",
            "bash_exec(*pkill*)",
            "bash_exec(*killall*)",
            "bash_exec(kill *)",     # bare `kill -9 <pid>`
            "bash_exec(* kill *)",   # embedded; space-delimited so `grep skill` is NOT matched
            "bash_exec(*shutdown*)",
            "bash_exec(*reboot*)",
            "bash_exec(*poweroff*)",
        ],
        description=(
            "List of deny patterns. Each pattern is in the form: "
            "'{tool_name}({argument_glob})'. "
            "A tool call is denied if its name and any argument matches a pattern. "
            "Pattern matching uses fnmatch semantics over the RAW pre-expansion argument "
            "strings (no shell parsing). The shipped defaults block credential/config writes, "
            "privilege escalation (sudo), recursive delete, world-writable chmod, and "
            "destructive service/process ops (docker stop/kill/rm, systemctl "
            "stop/disable/kill/mask, pkill/killall/kill, shutdown/reboot/poweroff) — while "
            "leaving read-only ops (docker ps, systemctl status, journalctl) allowed. "
            "The deny list is evaluated after inheritance resolution — "
            "an agent's deny list is the UNION of its own list and all inherited lists. "
            "Agents can add to the deny list but never remove inherited entries."
        ),
    )

    allow_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit allow list for 'manual' mode (v2). "
            "In 'auto' mode, this field is ignored."
        ),
    )

    workspace_root: Optional[str] = Field(
        default=None,
        description=(
            "Opt-in filesystem confinement. When set, write/edit target paths and bash_exec "
            "working_dir must resolve (symlink-safe, expanduser'd) INSIDE this directory or the "
            "tool returns a permission_denied error. Default None = UNCONFINED: file-write "
            "capability is a core product feature and stays fully enabled by default. Harness-run "
            "evals set this to an isolated per-run scratch dir."
        ),
    )

    budget: "BudgetConfig" = Field(
        default_factory=lambda: BudgetConfig(),
        description="Execution budget for this agent.",
    )

    @field_validator("deny_patterns")
    @classmethod
    def validate_pattern_format(cls, patterns: list[str]) -> list[str]:
        """
        Validate deny pattern format: must be 'tool_name(arg_glob)' or 'tool_name'.
        """
        pattern_re = re.compile(r"^[a-z_][a-z0-9_]*(\(.+\))?$")
        for p in patterns:
            if not pattern_re.match(p):
                raise ValueError(
                    f"Invalid deny pattern {p!r}. "
                    f"Format: 'tool_name' or 'tool_name(argument_glob)'. "
                    f"Example: 'bash(sudo:*)'"
                )
        return patterns


class MemoryConsolidationConfig(BaseModel):
    """Idle-time memory consolidation (v2.0 CONS-01..06) — the CLS slow-integrate pass.

    An in-harness feature (default-on, config-off; NO cron job, no daemon assumption):
    triggered by a session-start staleness check + an in-session idle timer, and
    cooperatively cancelled the instant a user turn arrives. All fields auto-enumerate
    as `agent.memory.consolidation.*` registry axes."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "If True (default), the harness consolidates memory during idle: folds staged "
            "read-counters, promotes cross-episode recurring candidates, decays retrieval "
            "strength, and trims the active tier back under the soft cap. Set False to "
            "disable all background memory work (like disabling the cruncher)."
        ),
    )
    idle_minutes: float = Field(
        default=10.0, ge=0.5, le=1440.0,
        description="In-session idle trigger: minutes with no user activity before a pass may fire.",
    )
    staleness_hours: float = Field(
        default=6.0, ge=0.1, le=720.0,
        description="Session-start trigger: run a pass at startup when the last one is older than this.",
    )
    max_active_facts: int = Field(
        default=256, ge=8, le=100_000,
        description=(
            "SOFT cap on the ACTIVE tier (index-eligible facts). Admission never blocks — "
            "the consolidation pass trims back under the bound by demoting the lowest-"
            "activation facts below the index gate (never deleting). The forcing function "
            "that makes the store structure knowledge instead of hoarding it."
        ),
    )
    decay_half_life_days: float = Field(
        default=30.0, ge=0.1, le=3650.0,
        description=(
            "Retrieval-strength half-life: accessibility halves after this many idle days "
            "(trust/confidence never decays — the storage-vs-retrieval strength split)."
        ),
    )
    iteration_cap: int = Field(
        default=200, ge=1, le=10_000,
        description="Hard per-pass work cap (letta #957 infinite-loop class guardrail).",
    )

    # --- Phase 36 (the chapter-writer) axes: the idle LLM passes, each independently
    # gated. All require an LLM wired into the scheduler (start_cmd); with no LLM the
    # deterministic core above is byte-unchanged (each step early-returns). ---
    schema_writer_enabled: bool = Field(
        default=True,
        description=(
            "Phase 36 SEMA-02/03: the idle chapter-writer clusters promoted lessons and "
            "writes one grounded schema per stable cluster (requires an LLM wired)."
        ),
    )
    reconcile_enabled: bool = Field(
        default=True,
        description=(
            "Phase 36 PGATE-03: an idle model-look reconciles the correction_pending "
            "quarantine (confirm / revert / undecidable). Also broadens the idle-work probe "
            "to fire on a pending correction queue."
        ),
    )
    mining_enabled: bool = Field(
        default=True,
        description=(
            "Phase 36 PGATE-03: an idle model-look mines transcripts for missed corrections "
            "and plain personal facts (grounded, budgeted, injectable)."
        ),
    )
    cluster_min_sessions: int = Field(
        default=2, ge=1, le=100,
        description="A cluster is chapter-worthy only if its members span at least this many distinct sittings.",
    )
    schema_write_budget: int = Field(
        default=3, ge=1, le=10_000,
        description=(
            "Max schema chapters written per idle cycle. Ceiling matches mining_write_budget's: "
            "the designed-month eval derives this from its manifest (len(topics)+1) and passes it "
            "to the CTOR, so any manifest-scale budget must validate instead of crashing "
            "construction at le=50."
        ),
    )
    schema_depth_cap: int = Field(
        default=2, ge=1, le=5,
        description="Max schema depth above lessons: lesson->chapter(1)->chapter-of-chapters(2)->stop.",
    )
    reconcile_ttl_looks: int = Field(
        default=3, ge=1, le=20,
        description="An undecidable correction fact leaves the reconciliation queue after this many looks.",
    )
    mining_write_budget: int = Field(
        default=50, ge=1, le=10_000,
        description=(
            "Max semantic atoms mined per idle cycle (MOVE 2: mining is the primary feeder; the "
            "walk is idle-window local-GPU and cancellable, so the cost is sleep-time). Default 50 "
            "= ~2x the densest single-pass yield observed on the designed month (~25 atoms), so a "
            "normal idle window drains without deferral; if a pass still exceeds it the un-mined "
            "tail is DEFERRED (watermark commits only per fully-mined chunk), never lost, so this "
            "is a throughput knob, not a correctness one. The old le=50 ceiling couldn't express a "
            "production-scale budget (the single-pass eval had to bypass the ctor to set 500) — "
            "raised to 10_000 (iteration_cap's scale) so any single-pass/backfill bound is settable."
        ),
    )
    mining_corpus_char_cap: int = Field(
        default=6000, ge=500, le=100_000,
        description=(
            "FIX 3b: mining chunk size — the per-chunk corpus char budget the transcript walk "
            "fills before one LLM look. Was a hardcoded 6000; surfaced as a knob so an empirical "
            "sweep can tune it later (default preserves today's behaviour). Chunks never span a "
            "session_id boundary; an oversized session sub-splits by this cap."
        ),
    )
    mining_known_atoms_cap: int = Field(
        default=50, ge=5, le=200,
        description=(
            "FIX 3: how many newest active sem/ atoms are shown to the miner as `replaces=` targets. "
            "Was a fixed 30; per-session chunking multiplies chunk count (~+40%), scrolling this "
            "window faster. VISIBILITY only: supersede correctness never depends on this cap — "
            "every this-pass mint stays a valid replaces= target via mining's in-pass minted "
            "registry even after scrolling out — so write_budget may exceed it freely. The cap "
            "bounds the prompt preamble (what the miner is SHOWN as reuse/correction targets)."
        ),
    )
    mining_operative_message_types: list[str] = Field(
        default_factory=lambda: ["user_message", "assistant_message"],
        description=(
            "FIX 4 (provenance-collapse guard) — mining consumes only this OPERATIVE CONVERSATIONAL "
            "SURFACE (what the user and assistant actually said). Tool I/O (tool_result records) is "
            "structurally OUT of scope at INPUT CONSTRUCTION, so a store read-back — "
            "memory_search/memory_get echoing a prior fact VERBATIM into a LATER session — is never "
            "read by the miner and can never be re-mined. Without this, per-session chunking would "
            "re-mine an echoed fact and store_fact's distinct-day ladder would ADVANCE its provenance "
            "to the later session, collapsing the >=2-distinct-session evidence a chapter needs (and "
            "starving A1, which keys recall on provenance-day). A positive ALLOWLIST (not a denylist "
            "of named echo tools) is robust: a NEW echo tool needs no upkeep to stay out, and mining "
            "the user's actual words — not incidental file/command output — is also a quality gain. "
            "Proven no-loss on the designed month (all 17 atoms ground in conversation; 0 need tool "
            "I/O). Mirrors mining._OPERATIVE_MESSAGE_TYPES; empty => unrestricted (legacy all-types)."
        ),
    )
    mining_residue_enabled: bool = Field(
        default=True,
        description=(
            "RESIDUE LEDGER (core repair loop): the miner's first pass is attention-limited and "
            "lossy by design; committed records that never sourced a written atom are enqueued and "
            "re-mined in ISOLATION on the NEXT idle pass (amortized — a pass never drains its own "
            "residue; a quiet store drains nothing). Off = enqueue and drain both inert; the "
            "coverage metric still reports residue."
        ),
    )
    mining_residue_attempt_cap: int = Field(
        default=2, ge=1, le=10,
        description=(
            "K: isolated looks a residue record gets before RETIREMENT (out of the mining window "
            "forever; the history record is never deleted — retire selects, never destroys). "
            "Bounds the ledger: every record exits in <= K drains. Hyperparameter — sweep on the "
            "eval, do not taste-pick."
        ),
    )
    mining_residue_record_budget: int = Field(
        default=40, ge=1, le=10_000,
        description=(
            "Max pending residue records drained per pass (sequential, isolated chunks — never "
            "fanned out; single shared GPU). A cold backlog drains over several quiet cycles "
            "instead of one long burst. Hyperparameter — sweep on the eval."
        ),
    )
    mining_residue_min_chars: int = Field(
        default=20, ge=0, le=4000,
        description=(
            "Intake triviality filter: an uncited record shorter than this never enters the "
            "ledger ('ok'/'thanks' are not facts to rescue). The coverage METRIC still reports "
            "it — the ledger just never chews it. Hyperparameter — sweep on the eval."
        ),
    )
    mining_novelty_fold_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description=(
            "NOVELTY GATE (mining precision): a fresh mint that is PROVABLY redundant vs an "
            "active same-slug atom FOLDS into it as corroboration (recurrence ladder) instead of "
            "minting a paraphrase sibling (live dogfood: 8 near-identical 'GTM plan' atoms from "
            "one conversation). Fold requires ALL of: the new claim's salient tokens are a "
            "SUBSET of the atom's (a restatement adds nothing; a distinguishing token — "
            "'summarizer' vs 'citation' — blocks both directions), equal number sets ('room 3' "
            "vs 'room 7' are different facts), and Jaccard >= this threshold (floor against a "
            "tiny probe folding into a rich atom). Supersedes exempt. 1.0 ≈ off. A missed fold "
            "is a dup that decay handles; a false fold destroys a fact — the rule is "
            "deliberately conservative. Hyperparameter — sweep on the eval, do not taste-pick."
        ),
    )
    clustering_embed_sim_threshold: float = Field(
        default=0.55, ge=0.0, le=1.0,
        description=(
            "Tier-1 embedding leg of chapter clustering (owner 2026-07-10): two pool atoms link "
            "when cosine(embed) >= this AND they share >= 1 salient token — 2-FACTOR by doctrine, "
            "an embedding edge never welds a group alone (mega-blob lesson; also keeps the lexical "
            "HashingEmbedder fallback safe). Matches discovery's _EMBED_SIM default (0.55). Only "
            "active when consolidation has an embedder. Hyperparameter — sweep on the eval."
        ),
    )
    chapter_refresh_overlap: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description=(
            "CHAPTER REFRESH (run-14 fix): a new cluster whose member overlap with an existing "
            "ACTIVE chapter (|intersection| / min(|old|, |new|)) clears this threshold ADOPTS "
            "that chapter's key — store_fact supersedes on the old key (one active chapter, "
            "history preserved) instead of minting a near-identical sibling. Membership drift "
            "(a rescued residue atom joining, a correction row entering the pool) is an UPDATE, "
            "not a new chapter. At most one adoption per key per pass, so a facet SPLIT of an "
            "old chapter keeps both facets. Hyperparameter — sweep on the eval."
        ),
    )
    mint_tagging_enabled: bool = Field(
        default=True,
        description=(
            "Tag-graph M1: file each freshly-mined atom via a two-step closed-set classifier into "
            "a bucket (+optional child tag). Requires the mining LLM; a tagging failure never "
            "blocks the mint (degrades recall, never integrity)."
        ),
    )
    tag_discovery_enabled: bool = Field(
        default=True,
        description=(
            "Tag-graph discovery (v1): an idle multi-factor pass proposes NEW child tags over "
            "bucket-only atoms, accrues Bayesian evidence, and incorporates one (model NAMES it) "
            "at threshold. Requires the LLM; degrades to 2-factor when no embedder is available."
        ),
    )


class TriggerLexiconConfig(BaseModel):
    """COLL-02 zero-NLU trigger word lists (owner steer 2026-07-04: TRIGGERS, NOT
    CLASSIFIERS — a tripwire for a later model look, recall-first by design; a false
    trigger costs one logged record, a miss costs another missed correction). Matching rules
    live in memory/user_signals.py: single-word triggers match on token boundaries,
    multi-word triggers as substrings, all lowercased. Tunable per-family via the
    component registry with zero code edits."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    negation: list[str] = Field(
        default_factory=lambda: [
            "no", "nope", "nah", "not that", "not what i", "that's not", "thats not",
            "that's wrong", "thats wrong", "wrong",
        ],
        description="Correction-class triggers: negations ('no' is deliberately broad — owner-endorsed).",
    )
    correction_phrase: list[str] = Field(
        default_factory=lambda: [
            "i meant", "i said", "actually", "instead", "rather", "incorrect",
            "i didn't ask", "i didnt ask", "you misunderstood", "that isn't", "that isnt",
        ],
        description="Correction-class triggers: explicit correction phrasing.",
    )
    frustration: list[str] = Field(
        default_factory=lambda: [
            "ugh", "ffs", "wtf", "damn", "dammit", "fuck", "fucking", "shit",
            "come on", "seriously", "annoying", "frustrated", "frustrating",
            "still wrong", "still broken", "broken again",
        ],
        description="Correction-class triggers: frustration/profanity markers (owner-endorsed).",
    )
    confirmation: list[str] = Field(
        default_factory=lambda: [
            "exactly", "correct", "perfect", "right", "yes", "yep", "yeah",
            "that's right", "thats right", "that's it", "thats it", "spot on",
            "nice", "great", "awesome", "thanks", "thank you",
        ],
        description="Positive-label triggers ('exactly / right' are the owner's own examples).",
    )
    interruption: list[str] = Field(
        default_factory=lambda: [
            "stop", "wait", "hold on", "hang on", "never mind", "nevermind",
            "forget it", "cancel", "cancel that", "one sec", "pause",
        ],
        description=(
            "Interruption triggers — a WEAKER, separate label class, never conflated "
            "with corrections (owner ruling 2026-07-04 22:44). LEXICAL by design: the "
            "REPL has no mid-turn cancel seam (34-RESEARCH Pitfall 5)."
        ),
    )


class PredictiveGateConfig(BaseModel):
    """Collect-only predictive gate (Phase 34, COLL-01..04): per-tool statistical priors
    score every outcome; user-signal triggers log labeled prediction errors. Score
    everything, gate nothing — no store write and no behavior change keys off any of it
    until Phase 35 sets thresholds from the observed distribution."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "If True (default), the collect-only scorer + user-signal detector subscribe "
            "to the bus and persist surprise scores / signal labels (schema v4 tables). "
            "Zero model calls, zero gating — pure measurement feeding Phase 35."
        ),
    )
    min_prior_n: int = Field(
        default=5, ge=1, le=1000,
        description="Cold-start floor: below this many prior observations of a tool, its surprise score is neutral 0.0.",
    )
    latency_weight: float = Field(
        default=0.5, ge=0.0, le=10.0,
        description="Weight of |latency z-score| in the composite surprise score.",
    )
    size_weight: float = Field(
        default=0.25, ge=0.0, le=10.0,
        description="Weight of |output-size z-score| in the composite surprise score.",
    )
    pending_cap: int = Field(
        default=256, ge=8, le=10000,
        description="Max in-flight Action→Observation correlations held; overflow drops oldest (skip-under-load, collect-only can afford drops).",
    )
    reask_threshold: float = Field(
        default=0.8, ge=0.5, le=1.0,
        description="difflib.SequenceMatcher ratio above which a user message counts as a re-ask of an earlier message in the same sitting.",
    )
    reask_window: int = Field(
        default=50, ge=1, le=500,
        description="How many prior user messages per sitting the re-ask check compares against.",
    )
    write_live: bool = Field(
        default=True,
        description=(
            "Phase 35 (PGATE): if True (default), PredictiveWriteGate turns surprise scores "
            "and scoped corrections into real sub-0.7 fact writes. The pre-committed KILL "
            "lever — set False to revert to motif-only capture while the collect-only scorer "
            "keeps persisting scores as telemetry. "
            "Mutable via `localharness components set agent.memory.predictive_gate.write_live <true|false>`."
        ),
    )
    lexicon: TriggerLexiconConfig = Field(
        default_factory=TriggerLexiconConfig,
        description="COLL-02 trigger word lists — see TriggerLexiconConfig.",
    )


class MemoryConfig(BaseModel):
    """Memory backend configuration for an agent."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    sqlite_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the SQLite facts store for this agent. "
            "Defaults to ~/.localharness/agents/{agent_name}/memory.db if not set."
        ),
    )

    history_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the JSONL chat history file. "
            "Defaults to ~/.localharness/agents/{agent_name}/events.jsonl."
        ),
    )

    notes_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the MEMORY.md persistent notes file. "
            "Defaults to ~/.localharness/agents/{agent_name}/MEMORY.md."
        ),
    )

    max_notes_chars: int = Field(
        default=16_000,
        ge=0,
        le=200_000,
        description=(
            "Maximum characters of MEMORY.md to inject into context on each turn. "
            "Notes are injected from the top (most recent entries are at the bottom — "
            "the whole file is included until this limit is reached)."
        ),
    )

    shared_read: list[Literal["division", "org"]] = Field(
        default_factory=list,
        description=(
            "Which memory scopes this agent can read from in addition to its own. "
            "'division': can read the division's shared.db and DIVISION.md. "
            "'org': can read the org-level GUARDRAILS.md (v2)."
        ),
    )

    inject_into_context: bool = Field(
        default=True,
        description=(
            "If True, inject MEMORY.md contents and recent SQLite facts into the system prompt "
            "at the start of each turn. If False, memory is available via tools only."
        ),
    )

    index_mode: bool = Field(
        default=True,
        description=(
            "If True (default), inline only a MEMORY INDEX — one line per persistent fact "
            "(name + one-line description, not the full body) plus the most recent "
            "`max_session_history_entries` session entries. Full fact bodies are served on "
            "demand via the memory_get / memory_search tools, so the per-turn memory tax "
            "stays small instead of growing with the whole MEMORY.md. If False, the entire "
            "MEMORY.md is inlined every turn (legacy behaviour)."
        ),
    )

    max_session_history_entries: int = Field(
        default=8,
        ge=0,
        le=200,
        description=(
            "When index_mode is True, how many of the most recent Session History entries to "
            "inline. Older entries stay in MEMORY.md / history.jsonl and are not injected. "
            "The injected index hard-caps this at 8 lines (TIME-03) — values above 8 render "
            "8; lower values render fewer."
        ),
    )

    write_gate_enabled: bool = Field(
        default=True,
        description=(
            "If True (default), the prediction-error write gate auto-captures memory fact "
            "candidates from bus signals the loop already emits (a tool error that later "
            "resolved, stuck-then-recovered, first-use novelty) — zero added LLM calls, "
            "written BELOW the injection confidence threshold until consolidation promotes "
            "them. Set False to disable all harness-initiated memory writes (the `remember` "
            "tool is unaffected)."
        ),
    )

    consolidation: MemoryConsolidationConfig = Field(
        default_factory=MemoryConsolidationConfig,
        description="Idle-time consolidation pass (v2.0 CONS) — see MemoryConsolidationConfig.",
    )

    predictive_gate: PredictiveGateConfig = Field(
        default_factory=PredictiveGateConfig,
        description="Collect-only predictive gate (Phase 34 COLL) — see PredictiveGateConfig.",
    )


class ContextConfig(BaseModel):
    """Context window management configuration."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    max_context_tokens: int = Field(
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        ge=1_000,
        le=2_000_000,
        description=(
            "Context budget in tokens. At runtime `start` derives the effective window "
            "from the SERVED max_model_len minus the output reservation; this config value "
            "acts only as an explicit cap/override (used when set and <= served-reserve). "
            "The default tracks the served reference window (single source of truth). "
            "If this exceeds the real window, compaction never triggers and long "
            "turns die at the provider's input cap instead of compacting."
        ),
    )

    compaction_threshold_pct: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description=(
            "Trigger summarize-middle compaction when context utilization exceeds this percentage. "
            "Default: 80% — compact when 80% of max_context_tokens is used."
        ),
    )

    preserve_first_n_messages: int = Field(
        default=4,
        ge=1,
        description=(
            "When compacting, always preserve the first N messages in the conversation "
            "(typically: system prompt + initial task). "
            "These are never summarized."
        ),
    )

    preserve_last_n_messages: int = Field(
        default=8,
        ge=2,
        description=(
            "When compacting, always preserve the last N messages "
            "(current working context). These are never summarized."
        ),
    )

    max_tool_output_chars: int = Field(
        default=32_000,
        ge=100,
        le=500_000,
        description=(
            "Maximum characters of a single tool result kept in context. Consumed as the "
            "compaction pipeline's tool_result_cap: on every over-threshold build the "
            "deterministic ToolResultCapStage head+tail truncates any tool result longer than this "
            "(keeping both ends, with an elision marker), before the LLM summary/compaction stages "
            "run — so oversized results are capped even after the per-turn LLM fire cap is spent."
        ),
    )

    tool_result_eviction: bool = Field(
        default=True,
        description=(
            "If True (default), once context usage passes 50% any bulky tool result "
            "(over `tool_result_evict_threshold_chars`, beyond the most recent few) has its "
            "body replaced with a restorable stub keyed by a deterministic content hash; the "
            "model re-pulls the full body on demand with tool_result_get('<id>'). This frees "
            "the largest, fastest-growing context consumer long before LLM summary-compaction "
            "(0.80) would otherwise fire. Deterministic ids keep the vLLM prefix cache stable."
        ),
    )

    tool_result_evict_threshold_chars: int = Field(
        default=8_000,
        ge=500,
        le=500_000,
        description=(
            "Char size above which a tool result becomes eligible for eviction to a restorable "
            "stub (see `tool_result_eviction`). Smaller results aren't worth stubbing."
        ),
    )

    system_prompt_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a markdown file containing the agent's system prompt. "
            "If not set, a default system prompt is generated from the agent's 'role' field."
        ),
    )

    microagents: list[str] = Field(
        default_factory=list,
        description=(
            "Names of microagent prompt files to inject into context (keyword-triggered, v2). "
            "Each microagent is a .md file in ~/.localharness/microagents/. "
            "v1: all listed microagents are always injected."
        ),
    )


class ScheduleConfig(BaseModel):
    """Scheduled execution configuration (v2 — parsed but not executed in v1)."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    cron: Optional[str] = Field(
        default=None,
        description=(
            "Cron expression for scheduled execution. "
            "Standard 5-field cron: 'minute hour day month weekday'. "
            "Example: '30 5 * * 1-5' = 5:30 AM Monday through Friday."
        ),
    )

    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone name for cron evaluation. "
            "Example: 'America/New_York', 'Europe/London', 'Asia/Seoul'."
        ),
    )

    task: Optional[str] = Field(
        default=None,
        description=(
            "Task description or path to task file to run on schedule. "
            "If a path, must be an absolute path or relative to ~/.localharness/."
        ),
    )

    @field_validator("cron")
    @classmethod
    def validate_cron_expression(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        parts = v.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression {v!r}: expected 5 fields "
                f"(minute hour day month weekday), got {len(parts)}"
            )
        return v

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        import zoneinfo
        try:
            zoneinfo.ZoneInfo(v)
        except zoneinfo.ZoneInfoNotFoundError:
            raise ValueError(f"Unknown timezone {v!r}. Use an IANA timezone name.")
        return v


class StuckDetectorConfig(BaseModel):
    """Stuck-loop detection knobs. Extracted from StuckDetector class defaults
    in agent/loop.py so they become addressable via component registry (REG-04).
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    window_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "Number of recent tool-call signatures to keep in the sliding window. "
            "Larger window = more lenient stuck detection."
        ),
    )
    recovery_threshold: int = Field(
        default=2,
        ge=1,
        le=100,
        description=(
            "Number of identical signatures in the window that triggers RECOVERING state "
            "(injects recovery_injection.message into the conversation)."
        ),
    )
    escalation_threshold: int = Field(
        default=3,
        ge=1,
        le=100,
        description=(
            "Number of identical signatures in the window that triggers ESCALATE state "
            "(terminates the turn with stuck reason)."
        ),
    )


class RecoveryInjectionConfig(BaseModel):
    """Recovery prompt injected into the conversation when StuckState.RECOVERING.
    Extracted from StuckDetector.recovery_message hardcoded string so the wording
    becomes a mutable component (REG-04).
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    message: str = Field(
        default=(
            "You have attempted the same tool call multiple times with identical arguments "
            "and received the same result. That approach is not working. "
            "Consider a fundamentally different strategy: try different arguments, "
            "use a different tool, or conclude that the information is not available this way."
        ),
        min_length=1,
        description=(
            "Message injected into the conversation when stuck recovery fires. "
            "Mutable via `localharness components set agent.recovery_injection.message <value>`."
        ),
    )


class SelfCheckConfig(BaseModel):
    """Optional bounded self-review pass before finalizing (MECH-01).

    A loop-structure mechanism: when enabled, the agent reviews its own
    about-to-be-final answer once before returning it (an extra conditional
    LLM round-trip + a re-entry into the agent loop). This is NOT a prompt
    edit (the `role` string is untouched) and NOT a numeric hyperparameter
    (it adds a control-flow branch). Addressable via the component registry
    as `agent.self_check.{enabled,max_passes}`.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Run a bounded self-review pass before finalizing. "
            "Mutable via `localharness components set agent.self_check.enabled <true|false>`."
        ),
    )
    max_passes: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Max self-review passes before a forced finalize. "
            "Bounded (ge=1, le=3) so the review step provably terminates."
        ),
    )


class RoleSectionsConfig(BaseModel):
    """Orthogonal, individually-addressable sections of the agent system prompt (MODP-01/02).

    Each section is an OPTIONAL append to `agent.role`. All default to "" so that with
    NO section set the assembled system prompt is BYTE-IDENTICAL to the bare `role`
    monolith (ROADMAP success criterion 4 — the load-bearing safety rail). `role` itself
    stays the canonical full text (it is required and has no default; there is no fixed
    monolith to carve, so the decomposition is ADDITIVE, never subtractive). A prompt
    experiment mutates exactly ONE section via its dot-path, e.g.
    `agent.role_sections.tool_use`, without rewriting the whole role (MODP-02).
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    identity: str = Field(
        default="",
        description=(
            "Identity / mandate section (who the agent is). "
            "Mutable via `localharness components set agent.role_sections.identity \"<text>\"`."
        ),
    )
    tool_use: str = Field(
        default="",
        description=(
            "Tool-use policy section (how/when to call tools). "
            "Mutable via `localharness components set agent.role_sections.tool_use \"<text>\"`."
        ),
    )
    stopping: str = Field(
        default="",
        description=(
            "Stopping & persistence section (when to stop vs. keep going / ask). "
            "Mutable via `localharness components set agent.role_sections.stopping \"<text>\"`."
        ),
    )
    output: str = Field(
        default="",
        description=(
            "Output discipline section (answer format / terseness). "
            "Mutable via `localharness components set agent.role_sections.output \"<text>\"`."
        ),
    )


class CruncherConfig(BaseModel):
    """Cruncher capability — a trusted (clean-origin) sub-agent's bounded code-exec for the
    composition the verbs can't express (two-body joins, numeric/structured aggregation,
    build-an-index-then-query). Restricted + sandboxed (restricted builtins, no __import__/open,
    RLIMIT_AS, cancellable subprocess) and seeded ONLY with clean-origin handle bodies — untrusted
    (web/memory) content is never bound into exec. Default off. Addressable as `agent.cruncher.*`.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    exec_enabled: bool = Field(
        default=False,
        description=(
            "Grant a trusted (clean-origin) cruncher the bounded restricted cruncher_exec for "
            "joins/aggregation/index over its granted handles. Web/memory crunchers are always "
            "verbs-only regardless. Default off (opt-in). "
            "Mutable via `localharness components set agent.cruncher.exec_enabled <true|false>`."
        ),
    )
    cell_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        le=600.0,
        description=(
            "Per-exec-cell wall-clock cap (seconds), enforced by a cancellable subprocess — on "
            "breach the cell is killed and returns a flagged partial, never spins. "
            "Mutable via `localharness components set agent.cruncher.cell_timeout_s <float>`."
        ),
    )
    mem_limit_mb: int = Field(
        default=512,
        ge=64,
        le=8192,
        description=(
            "Address-space cap (RLIMIT_AS, in MB) for an exec cell's subprocess, so a runaway "
            "allocation can't exhaust host memory. "
            "Mutable via `localharness components set agent.cruncher.mem_limit_mb <int>`."
        ),
    )


class AgentConfig(BaseModel):
    """
    Complete configuration for one agent.

    This is the model that the agent loop receives after inheritance resolution.
    All fields have their final resolved values — no inheritance placeholders remain.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    # --- Identity ---
    name: str = Field(
        description=(
            "Agent identifier. Must be unique across all agents. "
            "Format: lowercase alphanumeric with hyphens. Example: 'morning-briefing'. "
            "This becomes the agent_id used in events and file paths."
        ),
    )

    division: Optional[str] = Field(
        default=None,
        description=(
            "Division this agent belongs to. "
            "Must match the 'name' of an existing DivisionConfig. "
            "If set, the agent inherits division defaults before its own config is applied."
        ),
    )

    role: str = Field(
        description=(
            "Human-readable description of what this agent does. "
            "Used as the basis for the default system prompt if system_prompt_file is not set. "
            "Also used to generate the Agent Card for orchestrator routing."
        ),
    )

    # --- Model ---
    model: str = Field(
        default="inherit",
        description=(
            "LLM model identifier. "
            "Use 'inherit' to use the division's model, or the org default. "
            "Examples: 'qwen2.5:72b', 'Qwen/Qwen3.5-122B-A10B', 'llama3.3:70b'. "
            "The model name must match what the detected LLM backend serves."
        ),
    )

    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature for this agent.",
    )

    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128_000,
        description="Maximum tokens to generate in a single LLM response.",
    )

    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=30.0,
        le=3600.0,
        description=(
            "Per-agent HTTP timeout override for LLM API calls. "
            "None means use the global provider timeout. "
            "Use higher values for large models (e.g. 600s for 122B)."
        ),
    )

    # --- Subsystem configs ---
    tools: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Tool access specification.",
    )

    permissions: PermissionConfig = Field(
        default_factory=PermissionConfig,
        description="Permission policy.",
    )

    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory backend configuration.",
    )

    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context window management.",
    )

    schedule: Optional[ScheduleConfig] = Field(
        default=None,
        description="Scheduled execution (v2). Parsed but not executed in v1.",
    )

    # --- Channel ---
    channel: Optional[str] = Field(
        default="terminal",
        description=(
            "Channel for delivering results. "
            "v1: 'terminal' only. "
            "v2: 'discord://channel-name', 'slack://channel-name'."
        ),
    )

    # --- Stuck detection / recovery (REG-04 surfaces 4 + 5) ---
    stuck_detector: StuckDetectorConfig = Field(
        default_factory=StuckDetectorConfig,
        description=(
            "Stuck-loop detection knobs. Addressable via "
            "`agent.stuck_detector.{window_size,recovery_threshold,escalation_threshold}`."
        ),
    )

    recovery_injection: RecoveryInjectionConfig = Field(
        default_factory=RecoveryInjectionConfig,
        description=(
            "Recovery message injected when StuckState.RECOVERING. "
            "Addressable via `agent.recovery_injection.message`."
        ),
    )

    self_check: SelfCheckConfig = Field(
        default_factory=SelfCheckConfig,
        description=(
            "Optional self-review step (loop-structure mechanism, MECH-01). "
            "Addressable via `agent.self_check.{enabled,max_passes}`."
        ),
    )

    role_sections: RoleSectionsConfig = Field(
        default_factory=RoleSectionsConfig,
        description=(
            "Orthogonal system-prompt sections (MODP-01/02). All default to '' so the "
            "unmutated assembly is byte-identical to `role`. Addressable via "
            "`agent.role_sections.{identity,tool_use,stopping,output}`."
        ),
    )

    max_subagent_depth: int = Field(
        default=2,
        ge=1,
        le=4,
        description=(
            "How deep delegation may nest. Depth 0 = this agent; a subagent it spawns runs at "
            "depth 1, a sub-subagent (e.g. a web-researcher's search-verifier) at depth 2. A "
            "subagent at depth d may delegate iff d < max_subagent_depth; =1 disables nesting "
            "(kill-switch). Addressable via `agent.max_subagent_depth`."
        ),
    )

    cruncher: CruncherConfig = Field(
        default_factory=CruncherConfig,
        description=(
            "Cruncher capability config — a trusted clean-origin sub-agent's bounded restricted "
            "code-exec for composition the verbs can't express. Default off. Addressable via "
            "`agent.cruncher.{exec_enabled,cell_timeout_s,mem_limit_mb}`."
        ),
    )

    # --- Agent Card (for orchestrator routing) ---
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "List of capability keywords for Agent Card routing. "
            "The orchestrator uses these to match tasks to agents. "
            "Examples: ['web-search', 'hacker-news', 'summarization']. "
            "If empty, capabilities are inferred from the role description."
        ),
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for organizing and filtering agents.",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9-]{0,62}[a-z0-9]$|^[a-z]$", v):
            raise ValueError(
                f"Agent name {v!r} is invalid. "
                f"Must be lowercase alphanumeric with hyphens, 1-64 chars, "
                f"start and end with a letter or digit. Example: 'hn-monitor'."
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model_not_empty(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("model cannot be empty. Use 'inherit' to inherit from division/org.")
        return v

    @model_validator(mode="after")
    def resolve_memory_defaults(self) -> "AgentConfig":
        """Fill in default memory paths based on agent name if not set."""
        base = Path(f"~/.localharness/agents/{self.name}").expanduser()
        if self.memory.sqlite_path is None:
            object.__setattr__(self.memory, "sqlite_path", str(base / "memory.db"))
        if self.memory.history_path is None:
            object.__setattr__(self.memory, "history_path", str(base / "events.jsonl"))
        if self.memory.notes_path is None:
            object.__setattr__(self.memory, "notes_path", str(base / "MEMORY.md"))
        return self


class DivisionConfig(BaseModel):
    """
    Configuration for a division (group of related agents).

    Division config sets defaults for all agents in the division.
    Agent configs override individual fields. Division inherits from OrgConfig.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(
        description="Division identifier. Lowercase alphanumeric with hyphens.",
    )

    description: str = Field(
        default="",
        description="Human-readable description of this division's purpose.",
    )

    model: str = Field(
        default="inherit",
        description="Default model for all agents in this division. 'inherit' uses org default.",
    )

    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)

    tools: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Shared tool configuration for all agents in this division.",
    )

    permissions: PermissionConfig = Field(
        default_factory=PermissionConfig,
        description=(
            "Baseline permission policy for this division. "
            "Individual agents may add to the deny list but cannot remove inherited entries."
        ),
    )

    context: ContextConfig = Field(
        default_factory=ContextConfig,
    )

    shared_memory: Optional[str] = Field(
        default=None,
        description=(
            "Path to the division's shared SQLite database. "
            "Agents with shared_read=['division'] can read from this. "
            "Defaults to ~/.localharness/divisions/{name}/shared.db."
        ),
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9-]{0,62}[a-z0-9]$|^[a-z]$", v):
            raise ValueError(f"Division name {v!r} is invalid. Same rules as agent name.")
        return v


class OrgConfig(BaseModel):
    """
    Organization-level defaults.

    The org config is the lowest-priority ancestor in the inheritance chain.
    All agents inherit org defaults unless overridden at division or agent level.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(
        default="default",
        description="Organization identifier.",
    )

    default_model: str = Field(
        default="",
        description=(
            "Default LLM model for all agents. "
            "Must be set either here or in each agent/division config. "
            "Empty string means 'use whatever the LLM backend reports as default'."
        ),
    )

    default_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=4096, ge=1, le=128_000)

    permissions: PermissionConfig = Field(
        default_factory=PermissionConfig,
        description=(
            "Org-level baseline permission policy. "
            "Applied to all agents before division or agent-level overrides."
        ),
    )

    context: ContextConfig = Field(
        default_factory=ContextConfig,
    )

    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Structlog log level for the harness process.",
    )

    enforce_capability_floor: bool = Field(
        default=True,
        description=(
            "Capability floor (P-A security spine). When True (default): no single agent's "
            "resolved toolset may combine an untrusted-ingest tool (web_*) with a host-dangerous "
            "one (bash_exec/write/edit/python_exec), enforced at both toolset-resolution "
            "chokepoints; and the root agent has web ingestion stripped (it delegates ingestion "
            "to the web-researcher subagent). When False: floor checks + the root web-strip are "
            "skipped (loud warning) — migration escape hatch only."
        ),
    )

    audit_log_path: Optional[str] = Field(
        default="~/.localharness/audit.jsonl",
        description=(
            "Path to the org-level audit log. "
            "All events from all agents are also written here. "
            "Set to null to disable org-level audit log."
        ),
    )

    hooks: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Free-form per-hook config dict keyed by hook plugin name. "
            "Addressable via `org.hooks.<plugin_name>.<key>`. "
            "Phase 14 ships the schema slot for REG-04 surface 6; hook plugins read their own subtree."
        ),
    )


class ProviderConfig(BaseModel):
    """Detected LLM provider configuration. Written by localharness init."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    provider_type: Literal["ollama", "vllm", "llamacpp", "lmstudio", "unknown"] = Field(
        description="Detected provider type.",
    )
    base_url: str = Field(description="OpenAI-compatible base URL with /v1 suffix.")
    api_key: str = Field(default="none", description="API key (usually 'none' for local servers).")
    default_model: str = Field(description="First model from the detected backend's model list.")
    available_models: list[str] = Field(default_factory=list)
    supports_function_calling: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the default model supports native function calling. "
            "None means not yet probed. Set by PROV-03 startup probe."
        ),
    )
    timeout_seconds: float = Field(
        default=600.0,
        description=(
            "HTTP timeout for LLM API calls. Default 600s suits slow local "
            "single-stream decode — a 4096-token completion at ~10 tok/s is ~410s, "
            "which the previous 300s default killed mid-generation."
        ),
    )


class ProposerConfig(BaseModel):
    """Separate stronger-model config for the autoresearch proposer (PROP-02).
    NO default model/base_url — explicit choice forced (mirrors ScenarioSpec.slice).
    Frontier example: base_url=https://api.anthropic.com/v1, model=claude-..., is_local=False.
    Local 120B+ example: base_url=http://127.0.0.1:11434/v1, model=gpt-oss:120b,
    is_local=True, timeout_seconds=600 (LLMClient requires >=300s when is_local)."""
    model_config = ConfigDict(frozen=False, extra="forbid")
    base_url: str = Field(description="OpenAI-compatible base URL for the proposer model.")
    model: str = Field(description="Proposer model id — MUST differ from provider.default_model.")
    api_key: str = Field(default="none", description="API key ('none' for local).")
    is_local: bool = Field(default=False, description="True for local 120B+; requires timeout>=300s.")
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=3600.0)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


class SentinelConfig(BaseModel):
    """Eval-sentinel thresholds (REP-03/04). Auto-surfaced as `sentinel.*` registry knobs
    (REG-04). SEALED from the proposer — see _OFFREGISTRY_PREFIXES in experiment.py/adoption.py."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    overfit_gap_threshold: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="train_score - holdout_score above this flags overfitting (10pp default).")
    duplicate_similarity: float = Field(
        default=0.90, ge=0.0, le=1.0,
        description="difflib.SequenceMatcher ratio at/above which two consecutive same-component proposals are near-duplicate.")
    duplicate_consecutive_k: int = Field(
        default=3, ge=2, le=50,
        description="N consecutive near-duplicate proposals that trip the search-diversity-collapse alert.")
    saturation_k: int = Field(
        default=5, ge=2, le=100,
        description="A train fixture passed by the last K mutations is 'saturated' (rotation candidate).")
    saturation_ceiling: float = Field(
        default=0.99, ge=0.5, le=1.0,
        description="Per-fixture train score counted as 'passing' for saturation (tolerates Wilson noise).")


class ManagedServerConfig(BaseModel):
    """A model server the harness itself launched (init guided setup) and may
    restart — on `start` after a reboot, or on a REPL /model swap. Absent for
    user-managed servers."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    runtime: Literal["vllm"] = "vllm"
    launch: Literal["binary", "docker"] = Field(
        default="binary",
        description="binary = a vllm executable (system or harness venv); docker = foreground `docker run` (DGX Spark route).",
    )
    binary: Optional[str] = Field(default=None, description="Path to the vllm executable (launch=binary).")
    docker_image: Optional[str] = Field(default=None, description="Image to run (launch=docker).")
    model: str = Field(description="HF repo id or local checkpoint path passed to `vllm serve`.")
    port: int = Field(default=8081, description="Host port the OpenAI API is served on.")
    extra_args: list[str] = Field(default_factory=list, description="Extra `vllm serve` args (from the reference architecture).")
    refarch: Optional[str] = Field(default=None, description="Reference-architecture key this setup came from.")

    @model_validator(mode="after")
    def _launch_target_present(self) -> "ManagedServerConfig":
        if self.launch == "binary" and not self.binary:
            raise ValueError("launch=binary requires `binary` (path to the vllm executable)")
        if self.launch == "docker" and not self.docker_image:
            raise ValueError("launch=docker requires `docker_image`")
        return self


class HarnessConfig(BaseModel):
    """Root harness configuration. Stored at ~/.localharness/config.yaml."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    version: str = Field(default="1", description="Config schema version.")
    provider: ProviderConfig
    org: OrgConfig = Field(default_factory=OrgConfig)
    server: Optional[ManagedServerConfig] = Field(
        default=None,
        description="Harness-managed model server (written by init guided setup). None = user-managed.",
    )
    proposer: Optional[ProposerConfig] = None
    sentinel: SentinelConfig = Field(
        default_factory=SentinelConfig,
        description="Eval-sentinel thresholds (REP-03/04). Addressable via `sentinel.{overfit_gap_threshold,duplicate_similarity,duplicate_consecutive_k,saturation_k,saturation_ceiling}`.")

    @model_validator(mode="after")
    def _proposer_model_distinct(self) -> "HarnessConfig":
        if self.proposer is not None and self.proposer.model == self.provider.default_model:
            raise ValueError(
                "proposer.model must differ from provider.default_model "
                f"(both are {self.provider.default_model!r}); the proposer requires a "
                "distinct, stronger model (PROP-02)."
            )
        return self
