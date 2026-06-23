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
            "write(*/.env)",
            "write(*/secrets*)",
            "write(*/config.yaml)",
            "write(*/agents/*.yaml)",
            "bash_exec(sudo:*)",
            "bash_exec(rm -rf *)",
            "bash_exec(chmod 777 *)",
        ],
        description=(
            "List of deny patterns. Each pattern is in the form: "
            "'{tool_name}({argument_glob})'. "
            "A tool call is denied if its name and any argument matches a pattern. "
            "Pattern matching uses fnmatch semantics. "
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
        default=10,
        ge=0,
        le=200,
        description=(
            "When index_mode is True, how many of the most recent Session History entries to "
            "inline. Older entries stay in MEMORY.md / history.jsonl and are not injected."
        ),
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
            "Maximum characters of tool output to include in a single Observation. "
            "Output exceeding this limit is truncated with a '... [truncated]' suffix."
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
            "Grant a trusted (clean-origin) cruncher the bounded restricted python_exec for "
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


class HarnessConfig(BaseModel):
    """Root harness configuration. Stored at ~/.localharness/config.yaml."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    version: str = Field(default="1", description="Config schema version.")
    provider: ProviderConfig
    org: OrgConfig = Field(default_factory=OrgConfig)
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
