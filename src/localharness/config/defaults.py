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
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_MAX_CONTEXT_TOKENS: int = 131_072  # served Qwen/vLLM max_model_len (single source of truth)
DEFAULT_COMPACTION_THRESHOLD_PCT: float = 80.0
DEFAULT_MAX_TOOL_OUTPUT_CHARS: int = 32_000
DEFAULT_MAX_NOTES_CHARS: int = 16_000
DEFAULT_MAX_ACTIONS: int = 100
DEFAULT_MAX_DURATION_MINUTES: float = 30.0
