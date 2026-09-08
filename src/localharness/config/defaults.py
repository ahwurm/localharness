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
# The FLOOR of the per-reply output cap, not the cap itself. An agent that configures no
# `max_tokens` derives its starting cap from the window the server actually serves
# (agent/context.py: derive_output_cap) and never drops below this, so a small window keeps the
# reply length 0.13.0 gave it. A configured number is used exactly and ignores both of these.
DEFAULT_MAX_TOKENS: int = 4096
# ...and the fraction of the served window that derivation takes. A quarter is not a taste
# number, it is the largest fraction that cannot collide with the reply reserve: the reserve
# grows to hold the cap but is bounded at HALF the window (agent/context.py: response_reserve),
# so a quarter-window cap always fits inside that bound with history keeping three quarters.
# It reads 32,768 on the reference 131,072-token window — 8x the flat floor, which is what a
# thinking model spending its whole budget on hidden reasoning needs.
OUTPUT_CAP_WINDOW_FRACTION: float = 1 / 4
# The FULL served window (Qwen/vLLM max_model_len, single source of truth). The harness reserves
# response room internally (agent.context.response_reserve) — never pre-subtract it here.
DEFAULT_MAX_CONTEXT_TOKENS: int = 131_072
DEFAULT_COMPACTION_THRESHOLD_PCT: float = 80.0
DEFAULT_MAX_TOOL_OUTPUT_CHARS: int = 32_000
DEFAULT_MAX_NOTES_CHARS: int = 16_000
DEFAULT_MAX_ACTIONS: int = 100
DEFAULT_MAX_DURATION_MINUTES: float = 30.0

# Revision of the SHIPPED default deny list (PermissionConfig.deny_patterns). Bump by 1
# whenever that list grows/changes in a release. A user config stamps the revision it was
# last synced to in `org.permissions.defaults_revision`; `localharness config migrate` and
# the first `localharness start` after an upgrade additively fold in any newer shipped
# defaults, then stamp the config to this value. A config with the key absent = revision 0.
#   0 -> pre-sync (<= v0.9.0's 7-pattern list, or never stamped)
#   1 -> v0.9.1's 24-pattern list (issue #15: destructive service/process + embedded sudo/rm)
CURRENT_DEFAULTS_REVISION: int = 1
