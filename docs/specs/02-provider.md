# Spec 02: LLM Provider Layer

**Component:** `src/localharness/provider/`
**Requirements:** PROV-01, PROV-02, PROV-03, PROV-04, SETUP-01, SETUP-02, SETUP-03
**Dependencies:** `core/types.py`, `config/models.py`

---

## Purpose

The provider layer is LocalHarness's sole interface to inference backends. It isolates all LLM communication behind a single `LLMClient` class, handles the full diversity of local model capabilities (native function calling, XML fallback, text-only), detects endpoints automatically at init time, and applies the correct timeout policy for local inference. No other component in the harness makes HTTP calls to a model backend.

The design axiom: callers ask for completions and tool calls. They never think about whether the model speaks OpenAI-compat JSON or needs XML-formatted prompts, and they never tune timeouts for model size. The provider layer absorbs that complexity entirely.

---

## File Layout

```
src/localharness/provider/
    __init__.py          # exports: LLMClient, LLMConfig, detect_provider
    client.py            # LLMClient, LLMConfig, streaming logic
    detector.py          # DetectorResult, probe algorithm, port order
    fn_call.py           # FnCallConverter, XML format, extraction regex
```

---

## Data Structures

### `LLMConfig`

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class LLMConfig:
    # Required
    base_url: str
    """OpenAI-compat endpoint root, e.g. 'http://localhost:8000/v1'.
    Must NOT have a trailing slash. The client appends paths directly."""

    model: str
    """Model identifier as understood by the inference server.
    For Ollama: 'qwen3.5:122b'. For vLLM: 'Qwen/Qwen3.5-122B-A10B'."""

    # Connection
    api_key: str = "none"
    """API key. Local servers typically accept any non-empty string."""

    timeout_seconds: float = 300.0
    """Total request timeout. Applied to ALL clients in the provider layer,
    including auxiliary calls (model list, capability probe).
    300s minimum for large local models (122B at 51 tok/s → 40s for 2K tokens).
    Set higher for quantized models on slower hardware."""

    connect_timeout_seconds: float = 5.0
    """TCP connection establishment timeout. Separate from total timeout.
    Short (5s) because local servers either answer immediately or are down."""

    # Generation parameters
    temperature: float = 0.6
    max_tokens: int = 4096
    """Max output tokens per completion. Does not affect context window.
    4096 is sufficient for tool-calling turns; increase for long-form generation."""

    # Tool call mode (set by startup probe, not user config)
    tool_call_mode: Literal["native", "xml", "text"] = "native"
    """Detected tool call capability:
    - native: model returns tool_calls in API response JSON
    - xml: model produces XML-tagged tool calls in text response
    - text: model produces prose; harness extracts best-effort
    Never set manually in production — set by detect_capabilities()."""

    # Context window
    context_window: int = 128_000
    """Model's advertised context window in tokens. Used by ContextManager
    to determine compaction thresholds. Populated by detect_capabilities()
    from the /v1/models response if available; fallback to this default."""

    # Endpoint type (set by detector, affects timeout application)
    is_local: bool = True
    """True when base_url resolves to a local address (localhost, 127.0.0.1,
    0.0.0.0, ::1, or a .local hostname). Used to assert extended timeouts
    and skip API key validation."""

    # Optional
    extra_headers: dict[str, str] = field(default_factory=dict)
    """Additional HTTP headers for all requests. Useful for auth middleware
    or custom inference server requirements."""

    stop_sequences: list[str] = field(default_factory=list)
    """Stop sequences appended to every completion request."""
```

### `DetectorResult`

```python
from dataclasses import dataclass
from typing import Literal

ProviderType = Literal["ollama", "vllm", "llamacpp", "lmstudio", "unknown"]

@dataclass
class DetectorResult:
    found: bool
    """False if no server was detected on any probed port."""

    provider_type: ProviderType
    """Identified server software. 'unknown' if the server responded but
    could not be identified (still usable via OpenAI-compat)."""

    base_url: str
    """Resolved base URL including /v1 suffix, ready for LLMConfig."""

    models: list[str]
    """Available model identifiers as returned by the server's model list API.
    Empty list if the server responded but returned no models."""

    suggested_model: str
    """First model from the list, or '' if models is empty."""

    probe_duration_ms: float
    """Wall time for the successful probe, in milliseconds."""

    raw_response: dict
    """Full parsed JSON from the model list endpoint, for debugging."""
```

### `CapabilityResult`

```python
from dataclasses import dataclass

@dataclass
class CapabilityResult:
    tool_call_mode: Literal["native", "xml", "text"]
    context_window: int
    supports_streaming: bool
    probe_duration_ms: float
    probe_error: str | None
    """Non-None if the probe failed; tool_call_mode defaults to 'xml' on failure."""
```

---

## `detector.py`

### Purpose

Probe known local inference server ports in a fixed priority order. Return on first successful response. Cache result to `~/.localharness/config.yaml` on success. Never ask the user questions that can be answered by probing.

### Probe Order and Endpoints

| Priority | Port | Server | Probe URL | ID Signal |
|----------|------|--------|-----------|-----------|
| 1 | 8000 | vLLM | `GET /v1/models` | Response has `data` array with `object: "model"` |
| 2 | 11434 | Ollama | `GET /api/tags` | Response has `models` array |
| 3 | 1234 | LM Studio | `GET /v1/models` | Same structure as vLLM; LM Studio sets custom header `x-lm-studio` |
| 4 | 8080 | llama.cpp | `GET /v1/models` | Same structure as vLLM |

Rationale for order: vLLM is the recommended primary server for LocalHarness (multi-agent serving, MTP support). Ollama is second because it is the most common for casual users. LM Studio before llama.cpp because LM Studio has a GUI that users are more likely to have running.

### Public Interface

```python
import httpx
from localharness.provider.detector import DetectorResult, ProviderType

async def detect_provider(
    timeout_seconds: float = 1.0,
    ports: list[int] | None = None,
) -> DetectorResult:
    """Probe known inference server ports. Returns the first server found.

    Args:
        timeout_seconds: HTTP connect+read timeout per probe attempt.
            Keep short (1s) — a server is either up or it isn't.
        ports: Override probe order. Default: [8000, 11434, 1234, 8080].
            Used by tests to probe non-standard ports.

    Returns:
        DetectorResult with found=False if no server answered on any port.

    Raises:
        Never raises. All probe errors are caught and logged at DEBUG level.
        A failed probe is not an error — it just means that port is empty.
    """

def _build_base_url(port: int) -> str:
    """Return 'http://localhost:{port}/v1' for standard ports,
    except port 11434 (Ollama) which returns 'http://localhost:11434'
    because Ollama's native API root is not at /v1."""

async def _probe_port(
    client: httpx.AsyncClient,
    port: int,
    timeout: float,
) -> DetectorResult | None:
    """Attempt a single port probe. Returns None on any failure."""

def _identify_provider(
    port: int,
    response_json: dict,
    response_headers: httpx.Headers,
) -> ProviderType:
    """Heuristic identification from port, response shape, and headers."""

def _normalize_model_list(
    provider_type: ProviderType,
    response_json: dict,
) -> list[str]:
    """Extract model ID list regardless of API response shape.

    Ollama /api/tags returns: {"models": [{"name": "qwen3.5:122b", ...}]}
    OpenAI-compat /v1/models returns: {"data": [{"id": "model-name", ...}]}
    """
```

### Algorithm

```
detect_provider():
    DEFAULT_PORTS = [8000, 11434, 1234, 8080]
    probe_ports = ports or DEFAULT_PORTS
    start = time.monotonic()

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for port in probe_ports:
            result = await _probe_port(client, port, timeout_seconds)
            if result is not None:
                result.probe_duration_ms = (time.monotonic() - start) * 1000
                return result

    return DetectorResult(
        found=False,
        provider_type="unknown",
        base_url="",
        models=[],
        suggested_model="",
        probe_duration_ms=(time.monotonic() - start) * 1000,
        raw_response={},
    )
```

### Local Address Detection

Used to set `LLMConfig.is_local` and to apply `LOCAL_INFERENCE_TIMEOUT`.

```python
import re
from urllib.parse import urlparse

_LOCAL_PATTERNS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1|.*\.local)$",
    re.IGNORECASE,
)

def is_local_endpoint(base_url: str) -> bool:
    """Return True if base_url resolves to a local address.

    Called at LLMClient construction time. Used to:
    1. Set LLMConfig.is_local for logging
    2. Assert that timeout_seconds >= LOCAL_INFERENCE_TIMEOUT_MIN (300s)
    3. Log 'local endpoint detected, using extended timeout {N}s' at INFO
    """
    host = urlparse(base_url).hostname or ""
    return bool(_LOCAL_PATTERNS.match(host))

LOCAL_INFERENCE_TIMEOUT_MIN: float = 300.0
"""Minimum allowed timeout for local endpoints. Enforced at client construction.
Rationale: 122B MoE at 51 tok/s needs ~40s for a 2K-token tool response.
With retries, network stack delays, and quantization overhead on slower
hardware, 300s is the safe floor. Users can increase via agent YAML."""
```

---

## `client.py`

### `LLMClient`

```python
import openai
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from localharness.core.types import Message, ToolCall, ToolSchema
from localharness.provider.client import LLMConfig
from localharness.provider.fn_call import FnCallConverter

class LLMClient:
    """OpenAI-compatible async LLM client with XML fallback and local timeout handling.

    Constructed once per agent session. Not shared between agents.
    Thread-safe for asyncio concurrent use within a single event loop.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Args:
            config: Fully resolved LLMConfig, including tool_call_mode and
                    is_local set by detect_capabilities() before construction.

        Raises:
            ValueError: If config.is_local is True and config.timeout_seconds
                        < LOCAL_INFERENCE_TIMEOUT_MIN. Fail fast — a too-short
                        timeout on a local endpoint causes retry storms under load.
        """
        if config.is_local and config.timeout_seconds < LOCAL_INFERENCE_TIMEOUT_MIN:
            raise ValueError(
                f"Local endpoint requires timeout >= {LOCAL_INFERENCE_TIMEOUT_MIN}s, "
                f"got {config.timeout_seconds}s. "
                f"Set timeout_seconds in agent YAML or LLMConfig."
            )

        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=openai.Timeout(
                total=config.timeout_seconds,
                connect=config.connect_timeout_seconds,
                read=config.timeout_seconds,
                write=config.timeout_seconds,
            ),
            default_headers=config.extra_headers,
        )
        self._fn_converter = FnCallConverter() if config.tool_call_mode != "native" else None

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
    ) -> ChatCompletionMessage:
        """Single-turn completion. Handles native and XML tool call modes.

        Args:
            messages: Full conversation history in OpenAI message format.
                      Must already have been validated by repair_tool_pairing()
                      before this call. The client does NOT validate.
            tools: Tool schemas to advertise. Ignored when tool_call_mode is 'text'.
                   Converted to XML prompt injection when mode is 'xml'.
            stream: If True, streams internally but returns the assembled message.
                    Use stream_complete() for real-time token delivery.

        Returns:
            Assembled ChatCompletionMessage with content and/or tool_calls populated.

        Raises:
            ProviderConnectionError: TCP connection failed or DNS resolution failed.
            ProviderTimeoutError: Response not received within timeout_seconds.
            ProviderRateLimitError: HTTP 429 from server (rare for local, possible
                                    if inference server has queue limits).
            ProviderAPIError: HTTP 4xx/5xx other than timeout/429. Includes HTTP 400
                              from malformed tool_use/tool_result (should never reach
                              here if repair_tool_pairing() ran).
            MalformedResponseError: Model response could not be parsed into
                                    a valid message structure.
        """

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> ChatCompletionMessage:
        """Streaming completion with per-token callback.

        Streams internally. Calls on_token for each content token as it arrives.
        Tool call chunks are buffered and not delivered via on_token — they are
        assembled and returned in the final ChatCompletionMessage.

        Args:
            on_token: Async callback invoked with each content text token.
                      Used by the terminal channel adapter to print in real time.
                      If None, tokens are buffered silently (same result as complete()).

        Returns:
            Fully assembled ChatCompletionMessage identical to complete()'s return.

        Raises: Same as complete().
        """

    async def detect_capabilities(self) -> CapabilityResult:
        """Probe the model to determine tool call mode and context window.

        Sends a minimal tool-calling test request. Inspects the response to
        determine whether the model returns structured tool_calls or prose.
        Updates self.config.tool_call_mode and self.config.context_window in place.

        The probe request:
          - System: "You are a helpful assistant."
          - User: "What files are in the current directory?"
          - Tools: [{"name": "list_files", "description": "List directory contents",
                     "parameters": {"type": "object", "properties": {}, "required": []}}]
          - max_tokens: 64 (minimize cost)
          - temperature: 0.0 (deterministic)

        Detection logic:
          1. If response.choices[0].message.tool_calls is non-empty → "native"
          2. Else if response text contains <tool_call> tag → "xml"
          3. Else → "text"

        Context window:
          Attempts GET /v1/models and reads context_length from the model entry.
          Falls back to self.config.context_window if not present in response.

        Side effects:
          - Logs detection result at INFO: "Capability probe: mode={mode}, ctx={n}"
          - Sets self._fn_converter = FnCallConverter() if mode != "native"

        Returns:
            CapabilityResult with probe_error=None on success.
            On HTTP error or parse failure: probe_error set, tool_call_mode='xml'
            (XML fallback is safer than assuming native on unknown models).

        Raises:
            Never raises. All errors captured in CapabilityResult.probe_error.
        """

    async def _complete_native(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
    ) -> ChatCompletionMessage:
        """Internal: call OpenAI-compat API with tool_calls parameter."""

    async def _complete_xml(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
    ) -> ChatCompletionMessage:
        """Internal: inject XML tool schema into system prompt, call without tools
        parameter, extract tool calls from response via FnCallConverter."""

    def _build_xml_system_injection(self, tools: list[ToolSchema]) -> str:
        """Serialize tools as XML schema block for injection into system prompt.
        See FnCallConverter XML Format Spec section."""

    def _wrap_error(self, exc: Exception) -> "ProviderError":
        """Map openai SDK exceptions to LocalHarness provider error types."""
```

### Error Types

```python
class ProviderError(Exception):
    """Base class for all provider errors."""
    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause

class ProviderConnectionError(ProviderError):
    """TCP connection could not be established.
    Typically: server not running, wrong port, firewall."""

class ProviderTimeoutError(ProviderError):
    """Request exceeded timeout_seconds.
    On local endpoints, raise budget_consumed_tokens if available from stream."""
    def __init__(self, message: str, tokens_generated: int = 0, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.tokens_generated = tokens_generated

class ProviderRateLimitError(ProviderError):
    """HTTP 429. Inference server queue is full.
    Caller should wait and retry with exponential backoff."""
    def __init__(self, message: str, retry_after_seconds: float | None = None, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.retry_after_seconds = retry_after_seconds

class ProviderAPIError(ProviderError):
    """HTTP 4xx/5xx other than 429.
    Includes HTTP 400 from malformed tool sequences — should not occur
    if ContextManager.repair_tool_pairing() is called before every request."""
    def __init__(self, message: str, status_code: int, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.status_code = status_code

class MalformedResponseError(ProviderError):
    """Model returned a response that could not be parsed.
    Includes: empty choices, null message, unparseable JSON in tool arguments."""
    def __init__(self, message: str, raw: str = "", cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.raw = raw
```

---

## `fn_call.py` — Function Call Converter

### Purpose

Enable agents to use tools with models that do not support native OpenAI-compat function calling. The converter injects tool schemas as XML into the system prompt and extracts tool calls from the model's text response.

### XML Format Specification

The XML format used for prompt injection and extraction:

**System prompt injection block** (appended to system prompt when mode=xml):

```xml
You have access to the following tools. To call a tool, output a tool_call XML block
exactly as shown. Only output tool_call blocks — do not describe tool calls in prose.

<tools>
  <tool name="tool_name">
    <description>Human-readable tool description.</description>
    <parameters>
      <parameter name="param_name" type="string" required="true">
        Description of this parameter.
      </parameter>
      <parameter name="count" type="integer" required="false" default="10">
        Number of results to return.
      </parameter>
    </parameters>
  </tool>
</tools>

To call a tool:
<tool_call>
<name>tool_name</name>
<parameters>{"param_name": "value", "count": 5}</parameters>
</tool_call>

You may call multiple tools in sequence. Wait for tool results before calling the next tool.
After you have all necessary information, respond in plain text without a tool_call block.
```

**Model response format** (what the model should output):

```xml
<tool_call>
<name>glob</name>
<parameters>{"pattern": "**/*.py"}</parameters>
</tool_call>
```

**Multiple tool calls** (sequential, not parallel — local models are unreliable with parallel XML):

```xml
<tool_call>
<name>read_file</name>
<parameters>{"path": "/etc/hosts"}</parameters>
</tool_call>
<tool_call>
<name>glob</name>
<parameters>{"pattern": "src/**/*.py"}</parameters>
</tool_call>
```

### `FnCallConverter`

```python
import re
import json
from localharness.core.types import ToolSchema, ToolCall

# Primary extraction regex. Matches one complete tool_call block.
# Uses DOTALL so newlines in parameters are captured.
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<name>([\w\-]+)</name>\s*<parameters>(.*?)</parameters>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

# Fallback: match tool calls with missing closing tags (truncated output)
_TOOL_CALL_PARTIAL = re.compile(
    r"<tool_call>\s*<name>([\w\-]+)</name>\s*<parameters>([^<]*)",
    re.DOTALL | re.IGNORECASE,
)

class FnCallConverter:
    """Convert between OpenAI native tool_calls format and XML text format.

    Instantiated by LLMClient when tool_call_mode != 'native'.
    Stateless — all methods are pure functions of their inputs.
    """

    def build_system_injection(self, tools: list[ToolSchema]) -> str:
        """Serialize tools as XML schema block for system prompt injection.

        Args:
            tools: List of tool schemas. Empty list → empty string (no injection).

        Returns:
            Multiline string to append to the system prompt, including the
            instruction block and <tools> XML. Never returns None.
        """

    def extract_tool_calls(self, response_text: str) -> list[ToolCall]:
        """Parse tool calls from model text response.

        Extraction pipeline (applied in order, first match wins per block):
        1. Primary regex: full <tool_call>...</tool_call> blocks
        2. JSON repair: attempt jsonrepair on each parameters string
           before failing. Handles truncated JSON from mid-generation stops.
        3. Partial regex: match truncated tool_call blocks (model stopped mid-output)
        4. If no blocks found: return []

        Args:
            response_text: Full text content of the model's response message.

        Returns:
            List of ToolCall objects. Empty list if no tool calls found.
            Malformed blocks (unparseable JSON after repair, unknown tool name)
            are logged at WARNING and skipped — not raised.

        Raises:
            Never raises. Errors are logged and the block is skipped.
        """

    def tool_calls_to_messages(
        self, tool_calls: list[ToolCall]
    ) -> list[dict]:
        """Convert extracted ToolCall list to OpenAI message format for session history.
        Used to normalize XML-extracted calls into the same session format as native calls."""

    def _repair_json(self, raw: str) -> dict | None:
        """Attempt to parse JSON, with lightweight repair for common model errors:
        - Trailing commas before }
        - Single quotes instead of double quotes
        - Unquoted string values (limited repair)
        - Truncated JSON (attempt to close open braces)

        Returns None if repair fails after all attempts.
        Implementation note: Use jsonrepair library if available, else apply
        the manual repairs above. jsonrepair is an optional dependency.
        """

    def schema_to_xml_tool(self, schema: ToolSchema) -> str:
        """Convert a single ToolSchema to the <tool> XML block for prompt injection."""
```

### Fallback Chain

The complete fallback chain for tool call extraction, applied in priority order:

```
1. NATIVE PATH (tool_call_mode = "native")
   └── API response.choices[0].message.tool_calls is non-empty
       → parse directly: tool_calls[i].function.name + .arguments (JSON string)
       → no extraction needed

2. XML PATH (tool_call_mode = "xml")
   └── FnCallConverter.extract_tool_calls(response.content)
       Step 2a: Apply _TOOL_CALL_PATTERN (primary regex)
       Step 2b: For each match, attempt json.loads(parameters)
       Step 2c: If json.loads fails, attempt _repair_json(parameters)
       Step 2d: If repair fails, log WARNING, skip block
       Step 2e: If no complete blocks, apply _TOOL_CALL_PARTIAL (truncated match)
       Step 2f: Return list (may be empty)

3. TEXT PATH (tool_call_mode = "text")
   └── Same as XML but with no system prompt injection
       Model response may contain informal tool invocations
       Extract best-effort — this mode has no reliability guarantee
       Used only as last resort for fully unstructured models

EMPTY RESULT HANDLING:
   If extraction returns [] and the response content is non-empty:
   → No tool calls: treat as final assistant message, break loop
   If extraction returns [] and the response content is empty:
   → MalformedResponseError: model produced no usable output
```

---

## Startup Capability Probe Protocol

Performed once per agent session startup, before the first agent loop iteration. Results cached in the session's resolved `LLMConfig`.

```
CAPABILITY PROBE SEQUENCE:
──────────────────────────
1. Call LLMClient.detect_capabilities()

2. Probe request construction:
   - model: config.model
   - messages: [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user",   "content": "What files are in the current directory?"}
     ]
   - tools: [{"name": "list_files", "description": "List directory contents",
               "parameters": {"type": "object", "properties": {}, "required": []}}]
   - max_tokens: 64
   - temperature: 0.0

3. Send to /v1/chat/completions

4. Parse response:
   CASE A: choices[0].message.tool_calls is non-empty
     → tool_call_mode = "native"
     → Instantiate FnCallConverter: None (not needed)

   CASE B: choices[0].message.content contains "<tool_call>"
     → tool_call_mode = "xml"
     → Instantiate FnCallConverter()

   CASE C: HTTP 400 (server rejected tools parameter)
     → tool_call_mode = "xml"
     → Log WARNING: "Server rejected tools parameter, forcing XML mode"
     → Instantiate FnCallConverter()

   CASE D: All other responses (prose, empty, error)
     → tool_call_mode = "xml"
     → Log WARNING: "Could not confirm native function calling, defaulting to xml"

5. Context window detection (separate request):
   GET /v1/models
   Find entry matching config.model
   Read context_length or max_model_len field
   If found: config.context_window = value
   If not found: keep config.context_window default
   Log: "Model context window: {N} tokens"

6. Log final config at INFO:
   "Capability probe complete: mode={mode}, context_window={N}, timeout={T}s"

7. Persist resolved config to agent session state (not to disk — session-scoped only)
```

---

## Local Endpoint Timeout Detection and Override Logic

This is applied at **every point** where an HTTP client is constructed in the provider layer. It is not optional and is not configurable per-call — only per-agent via `LLMConfig.timeout_seconds`.

```python
def apply_local_timeout_policy(config: LLMConfig) -> LLMConfig:
    """Called at LLMClient.__init__. Enforces minimum timeout for local endpoints.

    Rules:
    1. If is_local_endpoint(config.base_url) is True:
       a. config.is_local = True
       b. If config.timeout_seconds < LOCAL_INFERENCE_TIMEOUT_MIN (300.0):
          RAISE ValueError (do not silently apply; the caller must know)
       c. Log at INFO: "Local endpoint detected ({base_url}), timeout={N}s"

    2. If is_local_endpoint() is False:
       a. config.is_local = False
       b. timeout_seconds left unchanged (cloud API has its own defaults)

    This function is idempotent — calling it multiple times on the same config
    is safe and has no additional effect after the first application.
    """
```

**Auxiliary client rule:** Any HTTP client created anywhere in the provider layer (detector probes, model list fetch, capability probe) must apply the same timeout as the main client when the target is a local endpoint. There is no "short timeout for metadata calls" exemption — metadata calls to Ollama under load can take several seconds.

```python
# Correct: reuse config.timeout_seconds for ALL requests to local servers
httpx.AsyncClient(timeout=config.timeout_seconds)

# Wrong: short timeout for "quick" metadata calls
httpx.AsyncClient(timeout=5.0)  # DO NOT DO THIS for local endpoints
```

---

## Provider Cascade (Future Multi-Model Support)

v1 has a single active provider. The interface is designed so a cascade can be layered on top without modifying `LLMClient`.

```python
class ProviderCascade:
    """v2: Try providers in order. Fall back on connection error or timeout.

    Not implemented in v1. Interface defined here for forward compatibility.
    """

    def __init__(self, providers: list[LLMClient], fallback_on: tuple[type, ...]) -> None:
        """
        Args:
            providers: Ordered list of clients to try.
            fallback_on: Exception types that trigger fallback.
                         Default: (ProviderConnectionError, ProviderTimeoutError)
                         Do NOT fall back on ProviderAPIError or MalformedResponseError —
                         those are semantic errors, not infrastructure failures.
        """

    async def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> ChatCompletionMessage:
        """Try providers[0], fall back to providers[1], etc."""

    async def stream_complete(self, messages: list[Message], tools: list[ToolSchema] | None = None, on_token=None) -> ChatCompletionMessage:
        """Streaming cascade. Falls back to non-streaming on providers that don't support it."""
```

---

## Error Handling Reference

| Condition | Error Type | Agent Loop Response |
|-----------|-----------|---------------------|
| Server not reachable | `ProviderConnectionError` | Emit `Observation(error=...)`, retry once after 2s, escalate |
| Timeout mid-generation | `ProviderTimeoutError` | Emit `Observation(error=...)`, log tokens generated, retry once |
| Queue full (429) | `ProviderRateLimitError` | Wait `retry_after_seconds` or 5s, retry up to 3 times |
| HTTP 400 (malformed) | `ProviderAPIError(status=400)` | This is always a bug — run `repair_tool_pairing()`, do not retry blindly |
| HTTP 5xx | `ProviderAPIError(status=5xx)` | Log at ERROR, emit escalation event |
| Empty response | `MalformedResponseError` | Inject "Please provide a response" and retry once |
| Unparseable tool args | `MalformedResponseError` | Attempt `_repair_json()`, if fails inject schema error and retry |
| XML extraction empty | (no error, return `[]`) | Agent loop treats as final message, breaks |

---

## Configuration

All provider configuration lives in two places:

**`~/.localharness/config.yaml`** (org-level, written by `localharness init`):
```yaml
provider:
  base_url: "http://localhost:8000/v1"
  model: "Qwen/Qwen3.5-122B-A10B"
  api_key: "none"
  timeout_seconds: 300
  tool_call_mode: "native"  # set by detect_capabilities at init
  context_window: 131072    # set by detect_capabilities at init
```

**Agent YAML** (agent-level, overrides org config):
```yaml
model: "inherit"  # use org config, OR specify a model string to override
provider_overrides:
  timeout_seconds: 600  # for agents that run long generations
  temperature: 0.3      # for agents requiring deterministic output
  max_tokens: 8192
```

---

## Dependencies

| Package | Version | Use |
|---------|---------|-----|
| `openai` | 1.x | Primary HTTP client, streaming, tool_calls parsing |
| `httpx` | (transitive via openai) | Detector probes, auxiliary HTTP calls |
| `jsonrepair` | optional | JSON repair in FnCallConverter; falls back to manual repair if absent |

No other external dependencies. The detector, converter, and client are all pure Python + stdlib + openai SDK.

---

## Implementation Notes

1. **Detector is not a long-running service.** It probes once at `localharness init` and writes results to disk. It does NOT probe on every agent start. The cached config is used. Re-run `localharness init` to re-detect.

2. **`LLMClient` is not a singleton.** Each agent session creates its own instance with its own resolved `LLMConfig`. This allows per-agent temperature/max_tokens/timeout without any shared state.

3. **Streaming and non-streaming return identical types.** `stream_complete()` assembles the full message before returning. The only difference is the `on_token` callback for real-time display. Agent loop code does not branch on streaming vs non-streaming.

4. **The `tools` parameter is never passed to the API when `tool_call_mode = "xml"`.** Some servers return errors on unknown parameters. XML mode omits `tools` entirely from the API request and handles schema injection via the system prompt.

5. **`detect_capabilities()` must be called before the first `complete()` call.** If `LLMClient` is constructed with `tool_call_mode="native"` (the default) but the model does not support it, the first `complete()` call with tools will either fail or produce empty tool_calls. Call `detect_capabilities()` in agent loop startup, before entering the while loop.

6. **JSON repair strategy.** The `_repair_json` method in `FnCallConverter` tries these transformations in order: (1) `json.loads` directly, (2) trailing comma removal, (3) single-quote normalization, (4) append `}` to close truncated objects, (5) if `jsonrepair` is installed, delegate to it. Each step tries `json.loads` after transformation. Stop at first success.

7. **Do not log API keys.** The `api_key` field is excluded from all log output. Structured logging must explicitly exclude it: `log.bind(base_url=config.base_url, model=config.model)` — never `log.bind(config=config)`.
