# Spec 08: Context Management

**Component:** `src/localharness/agent/context.py`
**Requirements:** CTX-01, CTX-02, CTX-03, LOOP-02
**Dependencies:** `core/types.py`, `config/models.py`, `provider/client.py`

---

## Purpose

The context manager is the gatekeeper between `session.messages` (the canonical append-only history) and the message list actually sent to the LLM. Its responsibilities:

1. **Token counting** — track how full the model's context window is.
2. **Tool result budget** — cap individual tool results before they consume the entire window.
3. **Boundary guard** — ensure every `tool_use`/`tool_result` pair is complete; repair or remove orphans before sending any request.
4. **Summary compaction** — when the window reaches 80%, summarize the middle portion to reclaim space while preserving task context.
5. **Full auto-compact** — emergency full-session LLM summary and reset when window reaches 95%.

The context manager is called once per loop iteration, before every LLM request. It returns a message list ready for the API — the caller (agent loop) does not need to think about any of these concerns.

---

## File Layout

```
src/localharness/agent/
    context.py   # ContextManager, TokenCounter, CompactionPipeline, RepairResult
```

---

## Data Structures

### `ContextConfig`

Drawn from `AgentConfig`. The context manager reads these fields.

```python
@dataclass
class ContextConfig:
    max_context_tokens: int
    """The model's context window size in tokens. Set by detect_capabilities()
    at startup. This is the total window — system prompt + history + tools + response.
    The context manager targets % of this value, not an absolute token count."""

    tool_result_max_tokens: int = 2000
    """Maximum tokens allowed for a single tool result before truncation.
    2000 tokens is ~1500 words — sufficient for most tool outputs.
    Increase for agents that regularly read large files."""

    summary_compaction_threshold: float = 0.80
    """Trigger summarize-middle compaction when context usage exceeds this fraction.
    Default: 0.80 (80%). Must be < full_compact_threshold."""

    full_compact_threshold: float = 0.95
    """Trigger full auto-compact when context usage exceeds this fraction.
    Default: 0.95 (95%). Must be > summary_compaction_threshold."""

    preserve_first_n: int = 2
    """Messages to preserve at the start of history during summary compaction.
    Preserves system prompt (always at index 0) + the initial user task message.
    Default 2 is correct for the standard message layout. Do not set below 2."""

    preserve_last_n: int = 6
    """Messages to preserve at the end of history during summary compaction.
    Preserves recent context (last ~3 iterations of back-and-forth).
    Default 6: last assistant msg + up to 2 iterations of tool calls/results."""

    summarization_model: str | None = None
    """Model to use for summarization LLM calls. If None, uses the agent's own model.
    Set to a smaller/faster model (e.g. 'qwen2.5:7b') to reduce summarization cost
    without consuming the agent's main generation budget."""

    summarization_max_tokens: int = 1024
    """Max output tokens for the summarization request. The summary replaces N messages,
    so it must be substantially shorter than the messages it replaces. 1024 is generous."""

    summarization_timeout_seconds: float = 60.0
    """Timeout for the summarization LLM call. Separate from main agent timeout
    because summarization is a short, focused completion, not a long generation."""
```

### `TokenBudget`

```python
@dataclass
class TokenBudget:
    total_limit: int
    """config.max_context_tokens"""

    current_usage: int
    """Estimated token count of messages currently in the session."""

    tool_schema_tokens: int
    """Estimated token count of the tool schemas being sent with this request."""

    headroom: int
    """= total_limit - current_usage - tool_schema_tokens - RESPONSE_RESERVE_TOKENS
    RESPONSE_RESERVE_TOKENS = config.max_tokens (from LLMConfig) — space to leave
    for the model's response. Do not consume this in history."""

    @property
    def usage_fraction(self) -> float:
        return (self.current_usage + self.tool_schema_tokens) / self.total_limit

    @property
    def is_critical(self) -> bool:
        return self.usage_fraction >= 0.95

    @property
    def needs_summary_compact(self) -> bool:
        return self.usage_fraction >= 0.80
```

### `RepairResult`

```python
from dataclasses import dataclass

@dataclass
class RepairResult:
    messages: list[Message]
    """The repaired message list. May be shorter than input."""

    removed_count: int
    """Number of messages removed during repair."""

    repairs_made: list[str]
    """Human-readable descriptions of each repair performed.
    Empty list if no repairs were needed."""

    was_clean: bool
    """True if the input was already clean (no orphans detected)."""
```

### `CompactionResult`

```python
@dataclass
class CompactionResult:
    messages: list[Message]
    compaction_type: Literal["none", "tool_result_cap", "boundary_repair", "summary", "full"]
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    summary_text: str | None
    """Populated only when compaction_type in ('summary', 'full')."""
```

---

## `ContextManager`

```python
from localharness.core.types import Message, ToolSchema
from localharness.provider.client import LLMClient, LLMConfig

class ContextManager:
    """Prepares message lists for LLM requests.

    Created once per agent session. Holds the token counter and compaction
    pipeline for that agent. Not shared between agents.
    """

    def __init__(
        self,
        config: ContextConfig,
        llm_config: LLMConfig,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Args:
            config: ContextConfig derived from the agent's resolved AgentConfig.
            llm_config: The provider config for this agent. Used by TokenCounter
                        to select the appropriate counting strategy.
            llm_client: Optional LLMClient for summarization calls.
                        If None, summarization falls back to compaction without LLM.
                        Pass None only in tests or when summarization is disabled.
        """
        self._config = config
        self._token_counter = TokenCounter(llm_config)
        self._pipeline = CompactionPipeline(config, self._token_counter, llm_client)

    def build_messages(
        self,
        messages: list[Message],
        tool_schemas: list[ToolSchema] | None = None,
    ) -> list[Message]:
        """Prepare messages for an LLM request. Called before every LLM call.

        Applies the full compaction pipeline in stage order. Returns a message
        list that is safe to send to the API — no orphaned tool_results,
        within budget, tool results capped.

        This method does NOT modify the input list. It operates on a copy.
        The caller's session.messages remains unchanged.

        Args:
            messages: The canonical session message list (session.messages).
            tool_schemas: Tool schemas to be sent with the request. Used to
                          account for their token cost in budget calculations.
                          Pass None if sending no tools.

        Returns:
            A new list, ready for the API. May be shorter than the input.

        Raises:
            RepairImpossibleError: The message list contains orphaned tool_result
                                   blocks that cannot be repaired (e.g. the entire
                                   history is malformed). The agent loop must stop.
            ContextOverflowError: Even after full compaction, the message list
                                   exceeds the context window. The agent loop must stop.
        """
        working = list(messages)  # copy; never mutate input

        tool_tokens = self._token_counter.count_schemas(tool_schemas or [])
        budget = self._compute_budget(working, tool_tokens)

        result = self._pipeline.run(working, budget)
        return result.messages

    def check_budget(
        self,
        messages: list[Message],
        tool_schemas: list[ToolSchema] | None = None,
    ) -> TokenBudget:
        """Compute the current token budget without applying any compaction.

        Used by the agent loop for logging (log context % after each iteration).
        Safe to call frequently — no LLM calls, no side effects.
        """
        tool_tokens = self._token_counter.count_schemas(tool_schemas or [])
        return self._compute_budget(messages, tool_tokens)

    def compact(
        self,
        messages: list[Message],
        force: bool = False,
    ) -> CompactionResult:
        """Manually trigger compaction. Exposed for testing and for agent loop
        error recovery paths.

        Args:
            force: If True, run full auto-compact regardless of current usage.
                   Default False (respects normal thresholds).

        Returns:
            CompactionResult describing what was done.

        Raises:
            ContextOverflowError: Even after full compaction, still over budget.
        """

    def _compute_budget(
        self,
        messages: list[Message],
        tool_tokens: int,
    ) -> TokenBudget:
        current_usage = self._token_counter.count_messages(messages)
        return TokenBudget(
            total_limit=self._config.max_context_tokens,
            current_usage=current_usage,
            tool_schema_tokens=tool_tokens,
            headroom=(
                self._config.max_context_tokens
                - current_usage
                - tool_tokens
                - RESPONSE_RESERVE_TOKENS
            ),
        )

# Reserve for model output — do not consume in history
RESPONSE_RESERVE_TOKENS: int = 4096
```

### Error Types

```python
class ContextError(Exception):
    """Base class for context manager errors."""

class RepairImpossibleError(ContextError):
    """The message list cannot be repaired to a valid tool_use/tool_result sequence.
    Indicates a harness bug in message construction. The agent loop should stop
    and log the full message list for debugging."""
    def __init__(self, message: str, messages: list[Message]) -> None:
        super().__init__(message)
        self.messages = messages

class ContextOverflowError(ContextError):
    """Even after maximum compaction, the context exceeds the model window.
    Agent loop must stop — sending this to the LLM would produce garbage."""
    def __init__(self, message: str, usage_fraction: float) -> None:
        super().__init__(message)
        self.usage_fraction = usage_fraction

class SummarizationError(ContextError):
    """LLM summarization call failed. Compaction pipeline falls back to
    truncation-without-summary. This is recoverable — the agent continues."""
    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause
```

---

## Token Counting Strategy

### `TokenCounter`

```python
import tiktoken
from localharness.provider.client import LLMConfig

class TokenCounter:
    """Model-aware token counting with configurable fallback strategies.

    Priority:
    1. tiktoken with detected model encoding (exact for GPT-family, approximate for others)
    2. tiktoken cl100k_base (good approximation for most instruction-tuned models)
    3. Character-based heuristic: chars / 4.0 (last resort, ±30% accuracy)

    The counter is constructed once per agent session and caches the encoding
    object. Token counting in the compaction loop is called many times per second
    — no I/O, no LLM calls.
    """

    def __init__(self, llm_config: LLMConfig) -> None:
        self._strategy = self._select_strategy(llm_config.model)

    def _select_strategy(self, model: str) -> "_CountingStrategy":
        """Select counting strategy based on model name.

        Rules (checked in order):
        1. If model name starts with 'gpt-' or 'text-embedding-': use tiktoken with exact encoding
        2. If tiktoken is installed (it's an optional dep): use cl100k_base encoding
        3. Fallback: CharHeuristic

        Log the selected strategy at INFO on first construction:
        "Token counting: using {strategy} for model {model}"
        """

    def count_messages(self, messages: list[Message]) -> int:
        """Count tokens in a message list.

        Counts content tokens per message + per-message overhead (~4 tokens/message
        for role, delimiters, etc — the standard OpenAI overhead estimate).
        Tool call arguments are counted as their JSON string representation.
        """

    def count_string(self, text: str) -> int:
        """Count tokens in an arbitrary string."""

    def count_schemas(self, schemas: list[ToolSchema]) -> int:
        """Count tokens that tool schemas consume in the context window.
        Schemas are serialized to their JSON representation and counted.
        This is an estimate — actual tokenization of tool definitions varies
        by server and model."""

class _CharHeuristic:
    """Last-resort token estimator: len(text) / 4.0.

    Accuracy: ±30% for English text. Systematically overestimates for code
    (which tokenizes more efficiently) and underestimates for non-Latin scripts.
    Safe to use for compaction threshold decisions because we apply margins
    (80%/95% thresholds) that absorb ±30% estimation error without triggering
    premature or missed compaction.
    """
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)
```

### Why Not the Inference Server's Tokenizer API?

Local inference servers (vLLM, Ollama) expose `/tokenize` endpoints that could produce exact counts. This is not used because:

- It requires a network call for every `count_messages()` invocation.
- `count_messages()` is called in a tight loop (before every LLM request, possibly multiple times during compaction).
- At 51 tok/s generation speed, even a 10ms tokenizer call per count adds measurable latency.
- `cl100k_base` is within 5-10% of exact counts for instruction-tuned models. The compaction thresholds (80%/95%) provide sufficient margin.

If exact counting is critical for a specific model (e.g., a model with unusual tokenization), a `CountingStrategy = "api"` option may be added in v2 with appropriate caching.

---

## Compaction Pipeline: 4 Stages

The pipeline runs stages in order. Each stage receives the output of the previous stage. A stage may be a no-op (if its condition is not met). Stages are not retried — if a stage fails, the error propagates up.

```python
class CompactionPipeline:
    """Applies compaction stages in sequence to a message list."""

    def __init__(
        self,
        config: ContextConfig,
        token_counter: TokenCounter,
        llm_client: LLMClient | None,
    ) -> None:
        self._config = config
        self._counter = token_counter
        self._llm = llm_client
        self._stages = [
            ToolResultCapStage(config, token_counter),
            BoundaryGuardStage(config, token_counter),
            SummaryCompactionStage(config, token_counter, llm_client),
            FullAutoCompactStage(config, token_counter, llm_client),
        ]

    def run(self, messages: list[Message], budget: TokenBudget) -> CompactionResult:
        """Run all stages in order. Returns the final result after all stages."""
        working = list(messages)
        compaction_type = "none"
        summary_text = None
        tokens_before = budget.current_usage

        for stage in self._stages:
            stage_result = stage.apply(working, budget)
            if stage_result.modified:
                working = stage_result.messages
                budget = self._recompute_budget(working, budget.tool_schema_tokens)
                compaction_type = stage_result.stage_name
                if stage_result.summary_text:
                    summary_text = stage_result.summary_text

        tokens_after = self._counter.count_messages(working)

        if budget.usage_fraction > 1.0:
            raise ContextOverflowError(
                f"Context still at {budget.usage_fraction:.0%} after full compaction.",
                usage_fraction=budget.usage_fraction,
            )

        return CompactionResult(
            messages=working,
            compaction_type=compaction_type,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_before=len(messages),
            messages_after=len(working),
            summary_text=summary_text,
        )
```

---

## Stage 1: Tool Result Cap

**Trigger:** Always. Runs on every call, regardless of context usage.

**Purpose:** Prevent any single tool result from consuming an outsized fraction of the context window. A `grep` over a large codebase can return megabytes of text. Without this cap, one tool call fills the entire window.

```python
class ToolResultCapStage:
    """Cap individual tool result messages at config.tool_result_max_tokens.

    Runs unconditionally — before checking any compaction thresholds.
    Truncation is applied to the working copy, never the canonical session.messages.
    """

    def apply(self, messages: list[Message], budget: TokenBudget) -> StageResult:
        modified = False
        result = []
        for msg in messages:
            if msg["role"] != "tool":
                result.append(msg)
                continue
            content = msg.get("content", "")
            token_count = self._counter.count_string(content)
            if token_count <= self._config.tool_result_max_tokens:
                result.append(msg)
                continue
            # Truncate
            truncated = self._truncate_to_budget(content, self._config.tool_result_max_tokens)
            truncation_notice = (
                f"\n\n[Output truncated from {token_count} to "
                f"~{self._config.tool_result_max_tokens} tokens. "
                f"Use more specific queries to retrieve targeted results.]"
            )
            result.append({**msg, "content": truncated + truncation_notice})
            modified = True
        return StageResult(messages=result, modified=modified, stage_name="tool_result_cap")

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens tokens.

        Strategy: binary search on character count using count_string().
        This is called only when truncation is needed (not in the hot path),
        so the binary search cost is acceptable.
        """
```

**Truncation strategy details:**
- Truncation is from the END of the tool output, not the beginning.
- Reasoning: Tool output headers (file paths, line numbers) appear at the start and are more important for the model's orientation than tail content.
- Exception: For `bash` tool results that contain both stdout and stderr, preserve stderr at the tail (it usually contains the error the model needs to see).
- A clear truncation notice is appended so the model knows the output is incomplete and can ask for a more targeted query.

---

## Stage 2: Boundary Guard — `repair_tool_pairing()`

**Trigger:** Always. Runs after tool result cap, before any compaction that might create orphans.

**Purpose:** Ensure every `tool_result` message has a preceding `assistant` message containing a matching `tool_call`. Orphaned pairs cause HTTP 400 from OpenAI-compat APIs and are the most common source of permanent session failure (PITFALLS.md Pitfall 1).

### `repair_tool_pairing()` Full Algorithm

```python
def repair_tool_pairing(messages: list[Message]) -> RepairResult:
    """Scan the message list and remove or repair orphaned tool_use/tool_result pairs.

    DEFINITIONS:
    - tool_use: An assistant message with a non-empty tool_calls list.
      The tool_calls list contains one or more ToolCall objects, each with an id.
    - tool_result: A message with role="tool" and a tool_call_id field.
    - paired: A tool_result whose tool_call_id appears in a preceding tool_use's
      tool_calls list with no intervening assistant message.
    - orphaned: A tool_result with no matching tool_use, or whose matching tool_use
      was removed by compaction.

    ALGORITHM:
    Pass 1 — Build ID map:
        known_tool_call_ids = set()
        For each message in order:
            if message.role == "assistant" and message.tool_calls:
                for tc in message.tool_calls:
                    known_tool_call_ids.add(tc.id)

    Pass 2 — Find orphans:
        orphaned_result_ids = set()
        For each message in order:
            if message.role == "tool":
                if message.tool_call_id not in known_tool_call_ids:
                    orphaned_result_ids.add(message.tool_call_id)

    Pass 3 — Find tool_uses with no results:
        tool_use_ids_with_results = set()
        For each message in order:
            if message.role == "tool":
                tool_use_ids_with_results.add(message.tool_call_id)

        # Find assistant messages where some tool_calls have results but others don't
        # This happens if the model called 2 tools and the session was cut mid-execution
        partially_orphaned_tool_uses = {}  # call_id -> assistant_message_index
        for i, message in enumerate(messages):
            if message.role == "assistant" and message.tool_calls:
                for tc in message.tool_calls:
                    if tc.id not in tool_use_ids_with_results:
                        partially_orphaned_tool_uses[tc.id] = i

    Pass 4 — Repair:
        result = []
        repairs = []

        For each message in messages:
            CASE: role == "tool" and tool_call_id in orphaned_result_ids:
                SKIP (remove orphaned result)
                repairs.append(f"Removed orphaned tool_result {tool_call_id}")

            CASE: role == "assistant" and all tool_calls have no results:
                SKIP (remove tool_use with no results — avoids the inverse orphan)
                repairs.append(f"Removed tool_use with unmatched IDs: {[tc.id for tc in tool_calls]}")

            CASE: role == "assistant" and some tool_calls have no results:
                # Keep only the tool_calls that have results
                keep_calls = [tc for tc in tool_calls if tc.id in tool_use_ids_with_results]
                if keep_calls:
                    append modified message with only keep_calls
                    repairs.append(f"Pruned {len(tool_calls) - len(keep_calls)} unmatched tool_calls")
                else:
                    SKIP entire message
                    repairs.append(f"Removed assistant message with all unmatched tool_calls")

            DEFAULT:
                append message unchanged

        return RepairResult(
            messages=result,
            removed_count=len(messages) - len(result),
            repairs_made=repairs,
            was_clean=len(repairs) == 0,
        )
```

### Post-Repair Validation

After `repair_tool_pairing()` runs, a structural validator asserts the invariant:

```python
def validate_tool_pairing(messages: list[Message]) -> None:
    """Assert that the message list has no orphaned tool_results.

    Called after every repair_tool_pairing() invocation.
    Raises RepairImpossibleError if orphans remain (should never happen).

    This is an assertion, not a recovery path. If it fires, there is a bug
    in repair_tool_pairing() itself.
    """
    known_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                known_ids.add(tc["id"] if isinstance(tc, dict) else tc.id)
    for msg in messages:
        if msg.get("role") == "tool":
            tcid = msg.get("tool_call_id")
            if tcid not in known_ids:
                raise RepairImpossibleError(
                    f"repair_tool_pairing() produced a message list with orphaned tool_result: {tcid}",
                    messages=messages,
                )
```

### Scanning Both Directions

The algorithm above scans forward only, which is correct for most cases. The "both directions" requirement from PITFALLS.md refers to the need to check both forward (for orphaned results) and backward (for tool_uses with no corresponding result). Pass 3 handles the backward check by collecting which tool_use IDs have results anywhere in the list, regardless of position.

---

## Stage 3: Summary Compaction

**Trigger:** `budget.usage_fraction >= config.summary_compaction_threshold` (default 0.80)

**Purpose:** Replace the middle section of the message history with an LLM-generated summary, preserving the first N and last N messages. The first messages contain the system prompt and original task; the last messages contain the most recent context.

### Algorithm

```
SUMMARY COMPACTION ALGORITHM:
──────────────────────────────────────────────────────────────────

Precondition: budget.usage_fraction >= 0.80

1. Identify the preservation zones:
   head = messages[:preserve_first_n]       # default: first 2 (system + task)
   tail = messages[-preserve_last_n:]       # default: last 6

   Note: If len(messages) <= preserve_first_n + preserve_last_n:
   → The history is too short to compact. Return unchanged.
   → Log DEBUG: "Summary compaction skipped: history too short ({N} messages)"

2. Identify the middle (to be summarized):
   middle = messages[preserve_first_n : len(messages) - preserve_last_n]

   If middle is empty: return unchanged.

3. Apply boundary guard to the cut points:
   The cut between head and middle must not orphan tool pairs.
   The cut between middle and tail must not orphan tool pairs.

   → Call _safe_cut_boundary(messages, preserve_first_n, preserve_last_n)
   → This returns adjusted start/end indices that respect tool_use/tool_result pairs.
   → See _safe_cut_boundary() algorithm below.

4. If adjusted middle is empty after boundary adjustment: return unchanged.

5. Invoke LLM summarization:
   summary_text = await _summarize_middle(middle, context_config)

   If summarization fails (SummarizationError):
   → Log WARNING: "Summarization failed: {error}. Skipping summary compaction."
   → Return unchanged (stage result: modified=False)
   → Do NOT raise — let stage 4 (full auto-compact) handle the situation.

6. Build summary placeholder message:
   summary_message = {
       "role": "system",
       "content": (
           "## Conversation Summary\n"
           "The following is a summary of earlier conversation steps that have been "
           "compacted to save context space:\n\n"
           + summary_text
       ),
   }

7. Build compacted message list:
   compacted = head + [summary_message] + tail

8. Apply repair_tool_pairing() to compacted list (safety check):
   If orphans detected after compaction → RepairImpossibleError

9. Return StageResult(messages=compacted, modified=True, summary_text=summary_text)
```

### `_safe_cut_boundary()`

```python
def _safe_cut_boundary(
    messages: list[Message],
    desired_head_end: int,
    desired_tail_start_from_end: int,
) -> tuple[int, int]:
    """Find the nearest cut points that do not split a tool_use/tool_result pair.

    Args:
        messages: Full message list.
        desired_head_end: Index where the head ends (exclusive).
                          Scan forward from here to find a safe cut.
        desired_tail_start_from_end: Number of messages to preserve from the end.
                                     Scan backward from here to find a safe cut.

    Returns:
        (safe_head_end, safe_tail_start): Indices delimiting the safe middle range.
        safe_middle = messages[safe_head_end:safe_tail_start]

    Algorithm for head boundary (scan FORWARD from desired_head_end):
        i = desired_head_end
        while i < len(messages) - desired_tail_start_from_end:
            msg = messages[i]
            # A safe cut point is after a complete exchange:
            # After a tool_result message (the last result in a batch), or
            # After a user message, or
            # At the start of a new assistant message with no pending tool_results.
            if is_safe_cut_point(messages, i):
                return i
            i += 1
        # No safe point found in middle → return original desired indices
        return desired_head_end

    Algorithm for tail boundary (scan BACKWARD from tail_start):
        Similar: find the nearest index from the tail end that is a safe cut point.

    def is_safe_cut_point(messages, i):
        # Safe to cut AFTER messages[i] if:
        # 1. messages[i].role == "tool" AND it is the last tool_result
        #    for its associated tool_use (i.e. all tool_calls in the preceding
        #    assistant message have a result in messages[:i+1])
        # 2. messages[i].role == "user" (user messages are always safe cut points)
        # 3. i == 0 (beginning of list is always safe)
        ...
    """
```

---

## Stage 4: Full Auto-Compact

**Trigger:** `budget.usage_fraction >= config.full_compact_threshold` (default 0.95)

**Purpose:** Emergency full-session compaction. Summarizes the entire session history (excluding system prompt) into a single summary message and resets the working message list. More aggressive than stage 3 — used when the window is nearly full.

```
FULL AUTO-COMPACT ALGORITHM:
──────────────────────────────────────────────────────────────────

Precondition: budget.usage_fraction >= 0.95

1. Separate system prompt from history:
   system_messages = [m for m in messages if m["role"] == "system"]
   history = [m for m in messages if m["role"] != "system"]

   Note: There is always exactly one system message (at index 0).
   If multiple system messages exist (from recovery injection), take the first.

2. Summarize entire history:
   summary_text = await _summarize_middle(history, config)

   If summarization fails:
   → Log ERROR: "Full auto-compact summarization failed. Context overflow imminent."
   → Try truncation fallback: keep only last preserve_last_n messages
   → If even that is over 95%: raise ContextOverflowError

3. Build compacted list:
   compact_history_message = {
       "role": "system",
       "content": (
           "## Full Session Summary\n"
           "All prior conversation has been summarized below to reclaim context space. "
           "Continue working from this summary:\n\n"
           + summary_text
       ),
   }
   compacted = system_messages + [compact_history_message]

4. Apply repair_tool_pairing() (should be clean since we removed all tool messages)

5. Return StageResult(messages=compacted, modified=True, summary_text=summary_text,
                       stage_name="full")
```

---

## LLM Summarization: How to Invoke Without Consuming Agent Budget

Summarization must not use the agent's main LLM context or token budget. It is a side-channel call.

### `_summarize_middle()`

```python
async def _summarize_middle(
    messages_to_summarize: list[Message],
    config: ContextConfig,
    llm_client: LLMClient,
) -> str:
    """Invoke LLM to summarize a message slice.

    Uses a SEPARATE LLMClient instance configured specifically for summarization:
    - Model: config.summarization_model (or agent's model if None)
    - max_tokens: config.summarization_max_tokens (default 1024)
    - temperature: 0.3 (lower = more deterministic summaries)
    - timeout: config.summarization_timeout_seconds (default 60.0)

    The summarization call does NOT go through the agent's main LLMClient instance.
    It uses a lightweight client constructed here with conservative settings.
    This ensures:
    - The summarization call does not appear in the agent's conversation history.
    - The summarization LLM's context is not polluted by the agent's context.
    - The summarization timeout is separate from the agent's generation timeout.

    Args:
        messages_to_summarize: The slice of messages to condense.
        config: ContextConfig with summarization parameters.
        llm_client: The agent's LLMClient, used to derive base_url and api_key
                    for the summarization sub-client.

    Returns:
        Summary string. Never empty — falls back to a structured list of
        "what was attempted / what was found" if the LLM returns an empty response.

    Raises:
        SummarizationError: LLM call failed or timed out.
                            DOES NOT raise on empty response — falls back instead.
    """

    # Build serialized history for the summarization prompt
    history_text = _serialize_for_summary(messages_to_summarize)

    summarization_prompt = [
        {
            "role": "system",
            "content": (
                "You are a summarization assistant. Your task is to condense a conversation "
                "history into a concise summary that preserves all important information: "
                "decisions made, findings discovered, actions taken, and their results. "
                "Be specific — include file paths, error messages, and concrete outcomes. "
                "Do NOT include meta-commentary about the summarization itself. "
                "Format as flowing prose with specific details, not bullet points."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the following conversation history:\n\n"
                f"{history_text}"
            ),
        },
    ]

    # Construct a fresh client for summarization
    summarization_config = LLMConfig(
        base_url=llm_client.config.base_url,
        model=config.summarization_model or llm_client.config.model,
        api_key=llm_client.config.api_key,
        temperature=0.3,
        max_tokens=config.summarization_max_tokens,
        timeout_seconds=config.summarization_timeout_seconds,
        tool_call_mode="text",  # No tools needed for summarization
        is_local=llm_client.config.is_local,
    )
    summarization_client = LLMClient(summarization_config)

    try:
        response = await summarization_client.complete(
            messages=summarization_prompt,
            tools=None,
        )
    except ProviderError as exc:
        raise SummarizationError(
            f"Summarization LLM call failed: {exc}",
            cause=exc,
        ) from exc

    summary = response.content or ""
    if not summary.strip():
        # Fallback: structured extraction from message list
        summary = _extract_fallback_summary(messages_to_summarize)
        log.warning("Summarization returned empty response; using structured fallback")

    return summary

def _serialize_for_summary(messages: list[Message]) -> str:
    """Convert messages to a compact text format for the summarization prompt.

    Format:
    [ASSISTANT]: Content of assistant message (tool_calls shown as "<called tool_name(args)>")
    [TOOL:tool_name]: Result content (truncated to 500 chars if longer)
    [USER]: User message content

    Tool call arguments are shown concisely — the model doesn't need full JSON for summarization.
    """

def _extract_fallback_summary(messages: list[Message]) -> str:
    """Build a structured summary without LLM when summarization fails.

    Extracts:
    - Tool names called (unique, in order)
    - Last assistant message content (the model's last state)
    - Error messages from tool results (if any)

    Returns a 3-5 sentence structured description of what was attempted.
    """
```

---

## Trigger Thresholds

| Threshold | Default | When Fires | What Happens |
|-----------|---------|-----------|-------------|
| `summary_compaction_threshold` | 0.80 | usage >= 80% | Stage 3: summarize-middle |
| `full_compact_threshold` | 0.95 | usage >= 95% | Stage 4: full auto-compact |

**Threshold rationale:**
- 80%: Fires early enough that the summary compaction + the next LLM response will still fit. At 80% on a 128K window, there are 25.6K tokens left. The summary replaces N messages but consumes only ~1K tokens. The next iteration starts well under 80%.
- 95%: Emergency threshold. By this point, the model may already be degrading (context saturation effects appear above 80% for most models). Full compact is the last resort before `ContextOverflowError`.
- The 15% gap between thresholds prevents thrashing: after a successful summary compact at 80%, usage drops significantly. The next summary compact will not trigger again for several more iterations.

**Estimation error margin:** `_CharHeuristic` can be ±30% inaccurate. At 80% threshold with 30% overestimation, the actual usage when the stage fires could be as low as 56%. This is conservative (fires earlier than needed) rather than catastrophic (fires too late). Summary compaction is cheap to run unnecessarily — running it early has no negative consequences beyond a slightly shorter history.

---

## Error Handling

### Summarization Failure

```
SUMMARIZATION FAILURE HANDLING:
────────────────────────────────
Stage 3 (summary compaction):
    SummarizationError → log WARNING, return stage result with modified=False
    Effect: Stage 3 is skipped. Stage 4 (full auto-compact) runs next iteration.
    Agent continues normally.

Stage 4 (full auto-compact):
    SummarizationError → log ERROR, attempt truncation fallback
    Truncation fallback: keep system_messages + last preserve_last_n messages
    If truncation brings usage below 95%: continue
    If truncation still above 95%: raise ContextOverflowError
```

### Token Count Estimation Errors

Token counting uses `try/except` around every `tiktoken` call. If tiktoken raises (encoding not found, internal error), fall back to `_CharHeuristic` transparently. Log at WARNING on first fallback in a session.

```python
def count_string(self, text: str) -> int:
    try:
        return len(self._encoding.encode(text))
    except Exception as exc:
        log.warning("tiktoken error (%s), falling back to char heuristic", exc)
        self._strategy = _CharHeuristic()
        return self._strategy.count(text)
```

### Repair Impossible

`RepairImpossibleError` is raised when `validate_tool_pairing()` detects orphans after `repair_tool_pairing()` runs. This indicates a bug in the repair algorithm, not in user code or model behavior. The agent loop catches this and stops:

```python
# In agent loop._execute_loop():
try:
    request_messages = context_manager.build_messages(session.messages, tool_schemas)
except RepairImpossibleError as exc:
    log.error(
        "repair_tool_pairing() produced invalid result for %s. "
        "Full message list: %s",
        config.name,
        json.dumps(exc.messages, default=str),
    )
    session.terminated_reason = "error"
    return _format_error_summary(session, exc)
```

### Context Overflow

`ContextOverflowError` is raised when all compaction stages have run and context is still above 100%. This means the session has accumulated so much history that even a full summary + system prompt exceeds the model window. This should be extremely rare (would require thousands of tool calls on a 128K model).

```python
# In agent loop._execute_loop():
except ContextOverflowError as exc:
    log.error(
        "Context overflow for %s at %.0f%% after full compaction. "
        "Consider increasing max_context_tokens or reducing max_actions.",
        config.name, exc.usage_fraction * 100,
    )
    session.terminated_reason = "error"
    return _format_error_summary(session, exc)
```

---

## Configuration Reference

### Agent YAML

```yaml
context:
  max_context_tokens: 131072          # set by detect_capabilities(), override here
  tool_result_max_tokens: 2000        # cap per tool result
  summary_compaction_threshold: 0.80  # fire summary compaction at 80%
  full_compact_threshold: 0.95        # fire full compact at 95%
  preserve_first_n: 2                 # keep first N messages in summary compact
  preserve_last_n: 6                  # keep last N messages in summary compact
  summarization_model: null           # null = use agent's own model
  summarization_max_tokens: 1024
  summarization_timeout_seconds: 60.0
```

---

## Dependencies

| Package | Version | Use |
|---------|---------|-----|
| `tiktoken` | optional | Exact token counting for GPT-family models; cl100k_base fallback |
| `openai` | 1.x | LLM client for summarization calls |

No additional dependencies beyond what the provider layer already requires.

---

## Implementation Notes

1. **`build_messages()` always operates on a copy.** The input `messages` list is never modified. All stages receive and return new lists. The canonical session history (`session.messages`) is mutated only by the agent loop's `session.push()` method.

2. **Stages are not stateful across calls.** Each call to `build_messages()` runs the full pipeline from scratch on the current snapshot of the message list. There is no "we already compacted this section" tracking. This is intentional — it makes the pipeline deterministic and easy to test.

3. **Compaction runs on the working copy, not the session.** The compacted message list returned by `build_messages()` is used for the LLM request only. It is not written back to `session.messages`. The session history grows unboundedly — only the LLM request is compacted. If the agent wants a full reset (e.g., after full auto-compact), the agent loop may update `session.messages` to match the compacted list. This is optional — the compacted list for the next request will be compacted again automatically.

4. **Summary placeholder messages use `role: "system"`.** Not `role: "assistant"`. System role prevents the model from treating the summary as its own previous reasoning and potentially contradicting it. The model sees the summary as authoritative external context, not as something it said.

5. **`_safe_cut_boundary()` may expand the middle rather than shrink it.** If the desired cut points fall inside a tool_use/tool_result pair, the algorithm scans outward — it may need to include more messages in the preserved head or tail to find a safe cut. This means the middle to be summarized can be smaller than desired. This is always correct — it's better to summarize less than to orphan a pair.

6. **Logging at every compaction event.** Each compaction stage logs at INFO when it fires:
   - Stage 1: `"Tool result capped: {tool_name} {before} → {after} tokens"`
   - Stage 2: `"Boundary repair: {N} messages removed, {repairs}"`
   - Stage 3: `"Summary compaction: {N} messages → 1 summary ({before} → {after} tokens)"`
   - Stage 4: `"Full auto-compact: {N} messages → 1 summary + system prompt"`
   This is the primary diagnostic for understanding why an agent's history changed.

7. **The `SummaryMessageID` pattern from OpenCode.** OpenCode uses a `SummaryMessageID` field to mark which message in the history is the compaction boundary, enabling efficient incremental compaction. LocalHarness v1 does not implement this — it re-scans the full history on every call. This is adequate for the expected iteration counts in v1. If profiling shows `build_messages()` is a bottleneck, implement `SummaryMessageID` tracking in v2.

8. **Token counting for tool schemas.** The overhead of tool schema tokens is significant — a set of 10 tools with rich parameter descriptions can consume 2-5K tokens. This is subtracted from the effective budget before computing `usage_fraction`. Never compute `usage_fraction` without accounting for tool schema tokens: `(message_tokens + schema_tokens) / max_context_tokens`.
