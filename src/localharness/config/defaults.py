"""Default value constants for LocalHarness configuration."""

DEFAULT_DENY_PATTERNS: list[str] = [
    "write(*/.env)",
    "write(*/secrets*)",
    "write(*/config.yaml)",
    "write(*/agents/*.yaml)",
    "bash(sudo:*)",
    "bash(rm -rf *)",
    "bash(chmod 777 *)",
]

DEFAULT_TIMEOUT_SECONDS: float = 300.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 5.0
DEFAULT_TEMPERATURE: float = 0.6
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_MAX_CONTEXT_TOKENS: int = 131_072  # served Qwen/vLLM max_model_len (single source of truth)
DEFAULT_COMPACTION_THRESHOLD_PCT: float = 80.0
DEFAULT_MAX_TOOL_OUTPUT_CHARS: int = 32_000
DEFAULT_MAX_NOTES_CHARS: int = 16_000
DEFAULT_MAX_ACTIONS: int = 100
DEFAULT_MAX_DURATION_MINUTES: float = 30.0
