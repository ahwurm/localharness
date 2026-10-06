"""Default value constants for LocalHarness configuration."""

# (Removed dead DEFAULT_DENY_PATTERNS — issue #15: it named the tool "bash(...)" but the
# registered tool is "bash_exec", was imported nowhere, and would have matched nothing if
# wired. The live default deny list is PermissionConfig.deny_patterns in config/models.py.)

# #10: 600s suits slow local single-stream decode — a 4096-token completion at ~10 tok/s
# is ~410s, which the previous 300s default killed mid-generation. Synced with
# LLMConfig.timeout_seconds and ProviderConfig.timeout_seconds.
DEFAULT_TIMEOUT_SECONDS: float = 600.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 5.0
DEFAULT_TEMPERATURE: float = 0.6
# An EXPLICIT per-reply cap for callers that need a bounded, comparable reply — the bench
# harness (bench/orchestrator.py), where every scenario must run on the same output budget or
# the numbers are not comparable. It is NOT a session default and no longer a floor under one:
# a session whose config sets no `max_tokens` sends no cap at all (see MAX_CONFIGURABLE_MAX_TOKENS
# below and provider/client.py's request construction).
DEFAULT_MAX_TOKENS: int = 4096
# The largest per-reply cap the harness accepts — the `le=` bound on every max_tokens field in
# config/models.py, named once so the three fields cannot drift apart. It bounds only values a
# user TYPES. Nothing derives a cap any more: unset means unset, the request omits max_tokens,
# and the served window is the only ceiling.
MAX_CONFIGURABLE_MAX_TOKENS: int = 128_000
# The FULL served window (Qwen/vLLM max_model_len, single source of truth). The harness reserves
# response room internally (agent.context.response_reserve) — never pre-subtract it here.
DEFAULT_MAX_CONTEXT_TOKENS: int = 131_072
DEFAULT_COMPACTION_THRESHOLD_PCT: float = 80.0
DEFAULT_MAX_TOOL_OUTPUT_CHARS: int = 32_000
DEFAULT_MAX_NOTES_CHARS: int = 16_000
DEFAULT_MAX_ACTIONS: int = 100
DEFAULT_MAX_DURATION_MINUTES: float | None = None  # no turn time limit by default (see BudgetConfig)

# Co-author consent (spec 14): the identity credited on commits the harness helps make, when
# the user opts in per-project. Named ONCE here so the name and email cannot drift between the
# prompt text, the trailer helper, and any docs. It is a *co*-author trailer (attribution
# only) — it never becomes the primary author and transfers no IP.
COAUTHOR_NAME: str = "localharness"
COAUTHOR_EMAIL: str = "localharness.agent@gmail.com"

# Revision of the SHIPPED default deny list (PermissionConfig.deny_patterns). Bump by 1
# whenever that list grows/changes in a release. A user config stamps the revision it was
# last synced to in `org.permissions.defaults_revision`; `localharness config migrate` and
# the first `localharness start` after an upgrade additively fold in any newer shipped
# defaults, then stamp the config to this value. A config with the key absent = revision 0.
#   0 -> pre-sync (<= v0.9.0's 7-pattern list, or never stamped)
#   1 -> v0.9.1's 24-pattern list (issue #15: destructive service/process + embedded sudo/rm)
#   2 -> v0.14's embedded `chmod 777` form (issue #159: the prefix-anchored pattern missed
#        `find . -exec chmod 777 {} \;` and `cd x && chmod 777 y`)
#   3 -> memory-store artifacts out of reach of bash_exec AND python_exec (owner order
#        2026-09-18): memory.db + its -wal/-shm/backup siblings, the facts_archive table,
#        memory-archive, and the owner-only `localharness memory` CLI. An existing config
#        picks these up on the next `config migrate` / first `start` — the union is additive,
#        so a user's own entries are kept and nothing they deleted is smuggled back.
# A REMOVAL from the shipped list needs no bump: plan() only adds, and a config that already holds
# the pattern keeps it (write(*/agents/*.yaml) left the defaults this way).
CURRENT_DEFAULTS_REVISION: int = 3
