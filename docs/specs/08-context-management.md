# Spec 08: Context Management

**Component:** `src/localharness/agent/context.py`
**Requirements:** CTX-01, CTX-02, CTX-03, LOOP-02
**Dependencies:** `core/types.py`, `config/models.py`, `provider/client.py`

> **Doc status — reconciled to the implementation on 2026-08-16.** This spec was written
> before the ContentStore/eviction subsystem existed, and parts of it drifted. The three
> specifics that a reader could previously take away as fact and be wrong about — a
> `ContextOverflowError` exception, `preserve_first_n`/`preserve_last_n` defaults of 2/6, and a
> token-denominated `tool_result_max_tokens` field — have been corrected against the source, and
> the eviction layer the spec never documented now has a section of its own. The rest of the
> document remains a design-level sketch: **the docstrings in `agent/context.py` are ground truth
> wherever this spec disagrees with them.** Code blocks below illustrate intent and do not
> reproduce the shipped signatures line for line.

---

## Purpose

The context manager is the gatekeeper between `session.messages` (the canonical append-only history) and the message list actually sent to the LLM. Its responsibilities:

1. **Token counting** — track how full the model's context window is.
2. **Tool result budget** — cap individual tool results before they consume the entire window.
3. **Boundary guard** — ensure every `tool_use`/`tool_result` pair is complete; repair or remove orphans before sending any request.
4. **Eviction** — at 50% usage, page bulky tool-result and web bodies out to the ContentStore, leaving a restorable stub. Cheap, deterministic, and non-lossy; see [Eviction Layer](#eviction-layer-the-contentstore).
5. **Summary compaction** — when the window reaches 80%, summarize the middle portion to reclaim space while preserving task context.
6. **Full auto-compact** — emergency full-session LLM summary and reset when window reaches 95%.
7. **Emergency floor** — if a request still would not fit, drop the oldest whole exchanges (and, last of all, shrink the surviving bodies) so overflow is impossible rather than fatal.

The context manager is called once per loop iteration, before every LLM request. It returns a message list ready for the API — the caller (agent loop) does not need to think about any of these concerns.

---

## File Layout

```
src/localharness/agent/
    context.py   # ContextManager, TokenCounter, CompactionPipeline, ContentStore
src/localharness/tools/builtin/
    tool_result_get_tool.py   # ToolResultGetTool — redeems an eviction stub
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

    max_tool_output_chars: int = 32_000
    """Maximum CHARACTERS — not tokens — of a single tool result kept in context.
    Passed to CompactionPipeline as `tool_result_cap` and consumed by
    ToolResultCapStage, which head+tail truncates anything longer. Range 100–500,000.
    Distinct from the ToolRegistry's own 50,000-char dispatch cap (see Stage 1)."""

    summary_compaction_threshold: float = 0.80
    """Trigger summarize-middle compaction when context usage exceeds this fraction.
    Default: 0.80 (80%). Must be < full_compact_threshold."""

    full_compact_threshold: float = 0.95
    """Trigger full auto-compact when context usage exceeds this fraction.
    Default: 0.95 (95%). Must be > summary_compaction_threshold."""

    preserve_first_n: int = 4
    """Messages to preserve at the start of history during summary compaction.
    Preserves the system prompt (always at index 0) and the opening task exchange.
    Config key: `context.preserve_first_n_messages` (minimum 1). Stage 4 overrides
    this to 1 internally when it forces an emergency compaction — that override is
    not this default and does not change it."""

    preserve_last_n: int = 8
    """Messages to preserve at the end of history during summary compaction.
    Preserves the recent working context — the last few iterations of tool
    calls and results. Config key: `context.preserve_last_n_messages` (minimum 2).
    Stage 4's internal emergency override is 2; again, not this default."""

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
            Nothing for a too-large context. build_messages() has no overflow
            exception: pairing is repaired rather than rejected, and a request that
            still would not fit is cut down by the emergency floor (see below).
            It can propagate RuntimeError from the TokenCounter when no exact
            token source is available and the caller demanded one.
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
            CompactionResult describing what was done. Being over budget after
            compaction is not an error condition — it is what the emergency floor
            exists to handle.
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

**There is no context-specific exception hierarchy.** Earlier drafts of this spec described a
`ContextError` base class with `ContextOverflowError`, `RepairImpossibleError`, and
`SummarizationError` subclasses. None of them were ever implemented, and no module in `src/`
defines or catches them. Do not write `except ContextOverflowError` — it will not import.

The design settled somewhere more boring, and deliberately so: **each failure mode degrades in
place rather than raising past the caller.**

| Condition | What actually happens |
|-----------|----------------------|
| Orphaned `tool_use`/`tool_result` pairs | `_repair_tool_pairing()` drops the orphans and returns a valid list. There is no unrepairable case — repair is a filter, not a validation. |
| Over budget after every compaction stage | The emergency floor drops the oldest whole user-turn exchanges; if the un-droppable remnant still does not fit, `_shrink_content_to_budget()` head+tail truncates the surviving bodies. Both log at ERROR, and a `CompactionTriggered` event is published so the cut is on the ledger. Overflow is designed to be impossible, not fatal. |
| Summarization call fails | The stage returns `modified=False` and the pipeline moves on. The turn continues. |
| No exact token source, and the runtime is one that should have one | `TokenCounter` raises **`RuntimeError`** (a fail-loud refusal to guess), which `start` catches at session setup. This is the only exception this module raises by design. |
| Provider unreachable while probing the tokenizer | `ProviderConnectionError`, from the provider layer. |

The trade-off this encodes: a silently shorter history is recoverable and a dead session is not,
so the module prefers a loudly-logged lossy cut over an exception. The cost is that a caller
cannot distinguish "compacted normally" from "the floor amputated four exchanges" by control
flow — that distinction lives in the log stream and the `CompactionTriggered` event, which is
why both are emitted unconditionally.

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

## Eviction Layer: the ContentStore

Everything above this line describes compaction — the expensive, lossy, LLM-driven path that fires
at 80%. In practice most sessions never get there, because a cheaper layer runs first.

**Eviction pages bulky tool-result bodies out of the prompt and into a content-addressable store,
leaving a stub the model can redeem on demand.** It costs no LLM call, loses nothing, and fires at
half a window rather than four-fifths of one. In the sessions observed so far (one live pass —
a tendency of the design, not a guarantee) it is why doc-heavy sessions ran for hours without a
single summarization.

The prior art is OpenHands' `BrowserOutputCondenser` (stale web pages are the bulkiest, least
re-read observations in an agent trace) and the Manus caveat that history must not be rewritten
every turn — rewriting invalidates the KV cache and costs more than it saves. Both constraints show
up directly in the design below: a usage-fraction gate so eviction is not a per-turn rewrite, and
deterministic content-hash handles so an evicted prompt is byte-identical across turns.

### The ContentStore

`ContentStore` (`agent/context.py`) is one per agent: `handle → (body, origin)`.

- **Handles are a deterministic content hash** — the first 12 hex characters of the body's SHA-1.
  Not a counter, not a timestamp, not a UUID. Two consequences follow, and both are load-bearing:
  identical bodies dedupe to one entry, and a stub rendered from the same body is byte-identical on
  every later turn, so the vLLM prefix cache stays warm. A random id would silently invalidate the
  cache from the first evicted result onward.
- **Origin is a sticky taint.** A body is `trusted` unless it came from the web (`put_web`) or was
  derived from an untrusted handle (`derived_from=`). Taint is monotonic — an untrusted handle
  never relaunders, and only a clean-origin handle may be bound into an exec namespace. That is
  the injection floor; see `SECURITY.md`. **Known gap:** memory-recall output
  (`memory_search`/`memory_get`) is NOT currently tainted — if a large recall result is evicted it
  enters the store with the default trusted origin, although memory can hold content that
  originally arrived from untrusted channels. The cruncher's own design comment treats memory as
  untrusted; the wiring does not yet. Tracked as a hardening gap.
- **Web bodies are LRU-bounded** (32 entries) because they are re-fetchable; trusted bodies are
  durable, because a restore must not fail.
- **A child agent gets a grant view** — built with `(parent, granted)`, it may read only the parent
  handles explicitly granted to it. That is the per-delegation capability; there is no global
  registry and no ambient cross-agent read.

### The 50% gate

On every `build_messages()` pass the manager computes usage once, then:

```
if usage_fraction >= 0.50:      # WEB_EVICT_USAGE_FRACTION
    evict stale web results     # keep newest 2, min 500 chars, URL/query preserved
if usage_fraction >= 0.50:      # TOOL_EVICT_USAGE_FRACTION
    evict large tool results    # keep newest 3, bodies > 8,000 chars
```

The **8,000-character threshold** (`TOOL_EVICT_THRESHOLD_CHARS`, config key
`context.tool_result_evict_threshold_chars`) exists because stubbing a small body saves nothing and
still costs a cache invalidation. The **keep-last window** (3 for tool results, 2 for web) leaves
the most recent observations verbatim — those are the ones the model is actively reasoning over.

Eviction requires a wired eviction store, which the root agent has and a leaf child does not. This
is deliberate: the root registers `tool_result_get` and can redeem a stub, so a child must never
stub a body it would have no way to re-pull. The whole layer is switchable via
`context.tool_result_eviction` (default `true`).

### Stub and restore

An evicted body is replaced in-place with:

```
[tool result evicted — ~N tokens — call tool_result_get('<id>') to restore the full body]
```

`N` is a `chars // 4` approximation, present so the model can judge whether the body is worth
redeeming. `tool_result_get(id)` returns the exact original body from the store. Nothing is lost —
the content moved out of the prompt, not out of the session.

### The restore pin

Restoring a body re-inflates usage, which re-arms the very gate that evicted it, which evicts the
just-restored body again under the same handle. That loop was measured live as a 24-minute turn of
restore/evict/restore.

The fix: **a body pulled back by `tool_result_get` is pinned against the usage-fraction eviction
pass for the rest of the turn.** Precisely:

- Pins are keyed by the **restore call's `tool_call_id`**, and scoped to the turn. The turn's first
  `build_messages()` baselines whichever `tool_result_get` calls history already carried, so a
  restore made in an *earlier* turn is ordinary evictable content again.
- Pins are recomputed on **every** pass, not only when the gate is armed — a restore made while
  usage sat below 50% must still be pinned once usage re-crosses it.
- A pinned result still counts toward the keep-last window. A pin costs exactly its own eviction
  and never pushes another body into protection; non-pinned bodies evict first.
- Pins **expire at turn rollover** (`reset_compaction_guard()`), so they can never accumulate into
  permanent context bloat.

**What a pin does not do, stated plainly:** it shields a body from the *eviction* pass only. The
hard-overflow stages keep priority. Once usage crosses 0.80 for any reason — not necessarily the
pinned body's fault — `ToolResultCapStage` may head+tail truncate a pinned body, and the model then
holds a *partial* view of content it just restored in full, with only the elision marker to say so.
That outcome is bounded and idempotent (the stage is deterministic over the canonical messages, so
there is no spiral), but it is a silent partial view, not merely graceful degradation. If a session
routinely restores bodies while sitting above 0.80, the budget is too small for the workload.

### How eviction (50%) relates to compaction (80%)

They are not alternatives; they are a ladder, and the rungs are ordered by cost:

| | Eviction | Compaction |
|---|---|---|
| Fires at | 50% | 80% (95% for full auto-compact) |
| Cost | No LLM call | An LLM summarization call per fire, capped at 3 per turn |
| Lossy? | No — the body is redeemable via `tool_result_get` | Yes — the summarized middle is gone |
| Effect on prefix cache | Stable (deterministic stubs) | Invalidated from the summary point |

The intended equilibrium is that eviction absorbs the growth so compaction rarely has to run. That
held under the August 2026 live-use test pass: on doc-heavy sessions usage **peaked at 79.10%
without compaction firing at all** — eviction repeatedly pulled the session back from the trigger.
One clean compaction firing was observed in the same pass, on a restore-heavy turn, taking usage
from **106% down to 40%**; the floor and the stages behaved as specified.

Read those two numbers for what they are — observations from one live pass on one hardware and
model configuration, not a guarantee. The 79.10% peak in particular is a near-miss: a workload with
slightly larger tool results would have crossed 0.80 and paid for a summarization. The equilibrium
is a tendency of the design, not an invariant of it.

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

        # NOTE: being over 100% here raises nothing. The caller (build_messages) applies the
        # emergency floor afterwards — dropping oldest exchanges, then shrinking bodies — so the
        # request always fits. See "Error Types" above.

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

**Trigger:** Every *over-threshold* build. Stage 1 and Stage 2 are the pipeline's **deterministic**
stages (no LLM call, no per-turn fire budget), so `build_messages()` runs them on every pass where
usage has already reached the 0.80 compaction threshold — including passes where the LLM stages are
skipped because the per-turn fire cap is spent. Below 0.80 the pipeline is not entered at all;
oversized results below that line are handled by eviction (50%) and by the registry cap at dispatch.

**Purpose:** Prevent any single tool result from consuming an outsized fraction of the context
window. A `grep` over a large codebase can return megabytes of text. Without this cap, one tool call
fills the entire window.

**The cap is denominated in CHARACTERS, not tokens.** There is no `tool_result_max_tokens` field
anywhere in the harness. The knob is `context.max_tool_output_chars` (default **32,000** characters),
which the pipeline receives as its `tool_result_cap` constructor argument.

### Two different caps, two different places

These are easy to conflate and are not the same mechanism:

| Cap | Value | Where it lives | When it applies |
|-----|-------|----------------|-----------------|
| `ToolRegistry.result_size_cap_chars` | 50,000 chars | `tools/registry.py` | At **dispatch** — the moment a tool returns, before its output ever becomes a message. A hard ceiling on what enters history at all. |
| `CompactionPipeline` `tool_result_cap`, from `context.max_tool_output_chars` | 32,000 chars | `agent/context.py`, Stage 1 | At **prompt build** — applied to tool messages already in history, on over-threshold passes. |

The registry cap is the wider one and fires first; the Stage 1 cap is the tighter one and fires
later, on the history. A tool result can therefore be truncated twice by two independent
mechanisms, and the constructor default on `CompactionPipeline` itself is 50,000 — the 32,000 the
harness actually runs on comes from config, not from that default.

```python
class ToolResultCapStage:
    """Stage 1: Pre-clean every tool result (ANSI/whitespace) and cap oversized ones with a
    head+tail keep (lossy first defense; the restorable page-out is the eviction stage)."""

    def __init__(self, max_chars: int = 50_000) -> None:
        self.max_chars = max_chars

    def apply(self, messages, budget, token_counter):
        for m in messages:
            if m.get("role") == "tool":
                cleaned = _clean_tool_output(m.get("content") or "")   # ANSI, whitespace, blank runs
                if len(cleaned) > self.max_chars:
                    cleaned = _head_tail(cleaned, self.max_chars)      # 60% head / 40% tail
        ...
```

**Truncation strategy details:**
- Every tool result is **pre-cleaned** first, whether or not it is oversized: ANSI escape sequences
  stripped, trailing per-line whitespace dropped, runs of 3+ blank lines collapsed. This is cheap,
  deterministic, and runs before length is measured, so the cap reflects real content rather than
  terminal formatting.
- Truncation keeps **both ends** — 60% of the budget from the head, 40% from the tail, with an
  `... [N chars elided — head+tail kept] ...` marker between them. Head-only truncation was
  rejected because it discards exactly where exit codes, stack traces, and test summaries live.
- The elision marker is the model's signal that the output is incomplete, so it can re-run a more
  targeted query or restore the full body via `tool_result_get` if the result was also evicted.

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

> **Design sketch — not implemented.** No `validate_tool_pairing()` and no
> `RepairImpossibleError` exist in `src/` (see the Error Types table above). The
> implemented `repair_tool_pairing()` enforces the invariant by construction — it drops
> orphaned tool results rather than asserting afterward. Kept for design intent only.

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
   head = messages[:preserve_first_n]       # default: first 4 (system + opening exchange)
   tail = messages[-preserve_last_n:]       # default: last 8 (recent working context)

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

   If summarization fails (any exception — no typed `SummarizationError` exists; the
   implementation catches broadly in `SummaryCompactionStage.apply`):
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
   If orphans detected after compaction → repaired in place by `repair_tool_pairing()`
   (the sketch's `RepairImpossibleError` was never implemented)

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

**How "more aggressive" is implemented:** Stage 4 does not have a separate algorithm. It
constructs a `SummaryCompactionStage` with hard-coded `preserve_first_n=1, preserve_last_n=2`
and hands it a synthetic budget pinned just over the 0.80 trigger to force it to fire. Those
**1/2 values are Stage 4's internal override only** — they are not the configured defaults (4/8),
they are not readable or settable from config, and they do not describe Stage 3's behavior.

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
   → Return unchanged (modified=False) and let the emergency floor take the turn:
     it drops the oldest whole exchanges, then shrinks bodies. No exception is raised.

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

    Raises (design sketch — the implemented path raises nothing typed; see the
    Error Types table: `SummaryCompactionStage.apply` catches broadly and skips):
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
| `TOOL_EVICT_USAGE_FRACTION` / `WEB_EVICT_USAGE_FRACTION` | 0.50 | usage >= 50% | Eviction: bulky tool/web bodies → restorable stubs (no LLM call) |
| `summary_compaction_threshold` | 0.80 | usage >= 80% | Stages 1–2 (deterministic) always; Stage 3 summarize-middle if fire budget remains |
| `full_compact_threshold` | 0.95 | usage >= 95% | Stage 4: full auto-compact |
| Emergency floor | > budget − 4,096 reserve | request still would not fit | Drop oldest whole exchanges, then shrink bodies. Never raises. |

**Threshold rationale:**
- 80%: Fires early enough that the summary compaction + the next LLM response will still fit. At 80% on a 128K window, there are 25.6K tokens left. The summary replaces N messages but consumes only ~1K tokens. The next iteration starts well under 80%.
- 95%: Emergency threshold. By this point, the model may already be degrading (context saturation effects appear above 80% for most models). Full compact is the last *summarizing* resort; past it, the emergency floor cuts the history mechanically rather than failing the turn.
- The 15% gap between thresholds prevents thrashing: after a successful summary compact at 80%, usage drops significantly. The next summary compact will not trigger again for several more iterations.

**Estimation error margin:** `_CharHeuristic` can be ±30% inaccurate. At 80% threshold with 30% overestimation, the actual usage when the stage fires could be as low as 56%. This is conservative (fires earlier than needed) rather than catastrophic (fires too late). Summary compaction is cheap to run unnecessarily — running it early has no negative consequences beyond a slightly shorter history.

---

## Error Handling

### Summarization Failure

```
SUMMARIZATION FAILURE HANDLING:
────────────────────────────────
Stage 3 (summary compaction):
    any summarization failure (no typed error exists) → log WARNING, modified=False
    Effect: Stage 3 is skipped. Stage 4 (full auto-compact) runs next iteration.
    Agent continues normally.

Stage 4 (full auto-compact):
    Summarization failure → the stage returns modified=False.
    The emergency floor in build_messages() then guarantees the request fits:
      1. drop the oldest whole user-turn exchanges (never a lone user message —
         a survivor list must still begin with a user turn or the server 400s)
      2. if the un-droppable remnant still overflows, head+tail shrink its bodies
    Both steps log at ERROR and publish CompactionTriggered. Nothing is raised.
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

### Orphaned Tool Pairs

`repair_tool_pairing()` is a filter, not a validator. Any tool message whose `tool_call_id` does
not appear in a preceding assistant message's `tool_calls` is dropped, and a new list is returned.
There is no "unrepairable" outcome and no exception — the agent loop calls `build_messages()`
without an exception handler for this case.

### Context Overflow

**Overflow does not produce an error.** When every compaction stage has run and the message tokens
plus tool-schema tokens still exceed `max_context_tokens − 4,096` (the reply reserve), the
emergency floor in `build_messages()` cuts the history until the request fits:

1. `_hard_truncate_to_budget()` drops the oldest whole user-turn exchanges — a user message
   through everything before the next user message. The unit is deliberately the whole exchange:
   dropping a lone user message would leave a history starting `[system, assistant, tool, …]`,
   which an OpenAI-compatible server rejects with HTTP 400. The leading system message is never
   dropped and the last remaining exchange is never emptied.
2. If the un-droppable remnant (system + the final message) still overflows, and only then,
   `_shrink_content_to_budget()` head+tail truncates the message bodies themselves.

Both log at ERROR with agent, session, and iteration, and publish `CompactionTriggered` so the cut
appears in the event ledger rather than only in the log stream.

This design replaced an earlier "stop the turn" posture after a live run stalled at 101.1%
utilization with compaction latched off. The rule now is that **overflow is impossible, not fatal**
— a loudly-logged lossy history beats a dead session.

---

## Configuration Reference

### Agent YAML

Real key names and shipped defaults. `docs/specs/06-config.md` is the full config reference;
this is the context-relevant subset.

```yaml
context:
  max_context_tokens: 131072              # budget; start refits it to the SERVED window
  model_context_overrides: {}             # per-model pins, keyed by EXACT model name
  compaction_threshold_pct: 80.0          # fire summary compaction at 80%
  preserve_first_n_messages: 4            # keep first N messages in summary compact (min 1)
  preserve_last_n_messages: 8             # keep last N messages in summary compact (min 2)
  max_tool_output_chars: 32000            # CHARACTER cap per tool result (Stage 1)
  tool_result_eviction: true              # page bulky results out to the ContentStore at 50%
  tool_result_evict_threshold_chars: 8000 # only bodies larger than this are evicted
```

Fields the earlier draft of this spec listed that **do not exist**: `tool_result_max_tokens`,
`summary_compaction_threshold`, `full_compact_threshold`, `preserve_first_n`, `preserve_last_n`,
`summarization_model`, `summarization_max_tokens`, `summarization_timeout_seconds`. The 0.95
full-compact threshold and the summarizer's parameters are module constants and call-site
arguments, not config keys.

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

2. **Stages are not stateful across calls; the manager holds two counters that are.** Each call to `build_messages()` runs the full pipeline from scratch on the current snapshot of the message list — there is no "we already compacted this section" tracking, which keeps the stages deterministic and easy to test. The `ContextManager` itself does carry two pieces of **per-turn** state, both reset by `reset_compaction_guard()` at the turn boundary: the summary-compaction fire count (capped at 3 per turn) and the restore-pin baseline (see [Eviction Layer](#eviction-layer-the-contentstore)). Neither survives a turn, and neither makes a stage's output depend on anything but its inputs.

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
