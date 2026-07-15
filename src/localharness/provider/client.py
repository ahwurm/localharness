"""OpenAI-compatible async LLM client with XML fallback and local timeout handling."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

import openai
from openai import AsyncOpenAI

try:
    import fcntl
except ImportError:  # non-POSIX: no cross-process lock, in-process semaphore still applies
    fcntl = None  # type: ignore[assignment]

from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS
from localharness.core.types import Message, ToolCall, ToolSchema
from localharness.provider.detector import LOCAL_INFERENCE_TIMEOUT_MIN
from localharness.provider.fn_call import _TOOL_INJECTION_MARKER, FnCallConverter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

# #62: default ceiling (seconds) on time WAITING for the local inference gate. GENEROUS by
# design — multi-session single-GPU contention is legitimate (a long generation in another
# session is a healthy wait, not a stall), so the ceiling is a backstop against a wedged slot,
# not a scheduler. Shared by LLMConfig's field default and the gate's config read so they cannot
# drift; kept in sync with ProviderConfig.inference_queue_wait_seconds.
_DEFAULT_QUEUE_WAIT_SECONDS = 600.0


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str = "none"
    # #10: 600s suits slow local single-stream decode — a 4096-token completion at ~10 tok/s
    # is ~410s, which the previous 300s default killed mid-generation. Kept in sync with
    # ProviderConfig.timeout_seconds and defaults.DEFAULT_TIMEOUT_SECONDS.
    timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 5.0
    # #62: ceiling on time spent WAITING for the inference gate (semaphore/flock), NEVER the
    # generation itself. None or 0 disables the bound. Threaded from
    # ProviderConfig.inference_queue_wait_seconds by `start`.
    queue_wait_seconds: float | None = _DEFAULT_QUEUE_WAIT_SECONDS
    temperature: float = 0.6
    max_tokens: int = 4096
    tool_call_mode: Literal["native", "xml", "text"] = "native"
    context_window: int = DEFAULT_MAX_CONTEXT_TOKENS
    is_local: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    stop_sequences: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Capability probe result
# ---------------------------------------------------------------------------


@dataclass
class CapabilityResult:
    tool_call_mode: Literal["native", "xml", "text"]
    context_window: int
    supports_streaming: bool
    probe_duration_ms: float
    probe_error: str | None


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class for all provider errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class ProviderConnectionError(ProviderError):
    """TCP connection could not be established."""


class ProviderTimeoutError(ProviderError):
    """Request exceeded timeout_seconds."""

    def __init__(
        self, message: str, tokens_generated: int = 0, cause: Exception | None = None
    ) -> None:
        super().__init__(message, cause)
        self.tokens_generated = tokens_generated


class ProviderRateLimitError(ProviderError):
    """HTTP 429 — inference server queue is full."""

    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.retry_after_seconds = retry_after_seconds


class ProviderAPIError(ProviderError):
    """HTTP 4xx/5xx other than 429."""

    def __init__(
        self, message: str, status_code: int, cause: Exception | None = None
    ) -> None:
        super().__init__(message, cause)
        self.status_code = status_code


class MalformedResponseError(ProviderError):
    """Model returned a response that could not be parsed."""

    def __init__(
        self, message: str, raw: str = "", cause: Exception | None = None
    ) -> None:
        super().__init__(message, cause)
        self.raw = raw


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inference gate — serialize requests to the shared local GPU
# ---------------------------------------------------------------------------
# One local GPU serves every harness process on the box, and concurrency toward it
# multiplies concurrent prefills. On unified-memory hosts (DGX Spark class) those
# allocations compete with ALL host RAM, and the observed failure mode is not a slow
# queue but a hard system freeze: NVRM cannot allocate → SoC wedge (2026-07-02, two
# overlapping harness processes on a 119 GiB box). Decode is engine-serialized anyway,
# so concurrency buys ~no wall-clock on a single GPU. Serial is therefore the default,
# at two independent layers; remote endpoints are ungated (provider limits apply there):
#   in-process  — asyncio.Semaphore; LOCALHARNESS_MAX_CONCURRENT_INFERENCE (default 1)
#   cross-proc  — flock on a per-endpoint lockfile; LOCALHARNESS_INFERENCE_LOCK=0 disables
_MAX_CONCURRENT_INFERENCE = max(1, int(os.environ.get("LOCALHARNESS_MAX_CONCURRENT_INFERENCE", "1")))
_inference_sem = asyncio.Semaphore(_MAX_CONCURRENT_INFERENCE)
_INFERENCE_LOCK_ENABLED = os.environ.get("LOCALHARNESS_INFERENCE_LOCK", "1") != "0"

# #62 (a) FAIL-FAST reachability probe. A cheap TCP connect+close (NO HTTP route → zero server
# load) run BEFORE the queue: a dead endpoint raises immediately instead of consuming a gate slot
# and then a doomed wait. Default on; LOCALHARNESS_INFERENCE_PROBE=0 disables (escape hatch).
_INFERENCE_PROBE_ENABLED = os.environ.get("LOCALHARNESS_INFERENCE_PROBE", "1") != "0"
_PROBE_TIMEOUT_SECONDS = 0.5   # connect budget — a healthy local connect is sub-ms. 0.5s (not
                                # 0.2s) so a Windows dual-stack "localhost" (getaddrinfo returns
                                # ::1 before 127.0.0.1) has room for happy-eyeballs to fall through
                                # to v4 instead of the whole budget being burned on a slow/blocked ::1
_PROBE_CACHE_TTL_SECONDS = 3.0  # trust a recent SUCCESS this long so the healthy hot path pays once
_probe_cache: dict[tuple[str, int], float] = {}  # (host, port) -> monotonic ts of last OK connect
# #62 (b) surface a queue wait once it passes this threshold (one honest INFO, not per poll).
_QUEUE_VISIBILITY_SECONDS = 2.0


def _inference_lock_path(base_url: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9.]+", "-", base_url.split("://", 1)[-1]).strip("-")
    return os.path.join(tempfile.gettempdir(), f"localharness-inference-{safe}.lock")


def _endpoint_host_port(base_url: str) -> tuple[str, int]:
    """(host, port) for a base_url, defaulting the port by scheme (http 80 / https 443)."""
    parts = urlsplit(base_url)
    host = parts.hostname or "localhost"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return host, port


async def _probe_reachable(host: str, port: int) -> bool:
    """Cheap TCP connect+close reachability probe — NO HTTP route is hit (connect only, zero
    server-side load). A successful result is cached for _PROBE_CACHE_TTL_SECONDS so a burst of
    requests to a healthy endpoint probes ~once, not per call. Failures are never cached (the
    server may be coming up)."""
    now = time.monotonic()
    last_ok = _probe_cache.get((host, port))
    if last_ok is not None and (now - last_ok) < _PROBE_CACHE_TTL_SECONDS:
        return True
    try:
        # happy_eyeballs_delay races v6/v4 per RFC 6555 instead of trying them serially — without
        # it, a multi-address host (Windows' getaddrinfo('localhost') = ['::1', '127.0.0.1']) can
        # spend the entire _PROBE_TIMEOUT_SECONDS stuck on a slow/blocked ::1 before ever trying
        # 127.0.0.1. Never forces AF_INET — remote IPv6-only endpoints still connect over v6.
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, happy_eyeballs_delay=0.1), _PROBE_TIMEOUT_SECONDS
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:  # closing a just-opened probe socket must never surface as a failure
        pass
    _probe_cache[(host, port)] = now
    return True


def _queue_wait_ceiling_error(ceiling: float) -> "ProviderTimeoutError":
    return ProviderTimeoutError(
        f"gave up waiting for a model slot after {ceiling:g}s (inference_queue_wait_seconds) — "
        "another request may be stuck; retry, or restart the harness"
    )


class _QueueWaitState:
    """Shared bookkeeping for one gate acquisition so the semaphore wait and the flock wait honor
    ONE ceiling and emit ONE visibility signal between them."""

    def __init__(self, ceiling: float | None):
        self._t0 = time.monotonic()
        self._ceiling = ceiling if ceiling and ceiling > 0 else None  # None/0/neg => disabled
        self._notified = False

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def remaining(self) -> float | None:
        return None if self._ceiling is None else self._ceiling - self.elapsed()

    def check_ceiling(self) -> None:
        """(c) Raise once the TOTAL gate wait has passed the ceiling (no-op when disabled)."""
        if self._ceiling is not None and self.elapsed() >= self._ceiling:
            raise _queue_wait_ceiling_error(self._ceiling)

    def maybe_notify(self) -> None:
        """(b) Emit ONE honest INFO once the wait passes the visibility threshold."""
        if not self._notified and self.elapsed() >= _QUEUE_VISIBILITY_SECONDS:
            self._notified = True
            log.info("waiting for a model slot (another request is in flight)… %.0fs elapsed",
                     self.elapsed())

    def summarize(self) -> None:
        if self.elapsed() > 5:
            log.info("inference gate: waited %.1fs for another slot", self.elapsed())


async def _acquire_sem_bounded(state: _QueueWaitState) -> None:
    """Acquire the in-process inference semaphore, bounded by the shared gate-wait ceiling and
    surfacing the wait past the visibility threshold. Holds one permit on return; raises the
    ceiling error (holding nothing) if the total wait exceeds the ceiling. Uncontended acquire is
    instant. On Python 3.12 a cancelled `wait_for(sem.acquire())` re-releases any granted permit,
    so the slice loop never leaks a permit."""
    while True:
        state.check_ceiling()          # raise if we already blew the ceiling (also guards step>0)
        remaining = state.remaining()  # None => unbounded
        step = _QUEUE_VISIBILITY_SECONDS if remaining is None else min(_QUEUE_VISIBILITY_SECONDS, remaining)
        try:
            await asyncio.wait_for(_inference_sem.acquire(), step)
            return
        except asyncio.TimeoutError:
            state.maybe_notify()       # one-time INFO; loop re-checks the ceiling at the top


@asynccontextmanager
async def _inference_gate(config: LLMConfig):
    """Hold for the FULL request including stream consumption — the GPU is occupied
    until the last token, not until the HTTP call returns.

    #62: before entering the queue a cheap TCP probe fails fast on a dead endpoint (never
    consuming a slot); the wait for the semaphore AND the flock is bounded by
    config.queue_wait_seconds (the gate wait only, never the generation) and surfaced once past a
    short threshold."""
    if not config.is_local:
        yield
        return
    # (a) FAIL-FAST: a dead endpoint raises BEFORE we take a slot or wait — a doomed request must
    # never queue behind healthy in-flight work. TCP connect only; no HTTP route, no server load.
    if _INFERENCE_PROBE_ENABLED:
        host, port = _endpoint_host_port(config.base_url)
        if not await _probe_reachable(host, port):
            raise ProviderConnectionError(
                f"inference endpoint {host}:{port} unreachable (TCP connect failed) — not queueing "
                f"a request that cannot succeed (is the model server up at {config.base_url}?)"
            )
    state = _QueueWaitState(getattr(config, "queue_wait_seconds", _DEFAULT_QUEUE_WAIT_SECONDS))
    await _acquire_sem_bounded(state)  # (b)+(c) on the in-process semaphore; holds one permit
    try:
        if not _INFERENCE_LOCK_ENABLED or fcntl is None:
            yield
            return
        fd = None
        try:
            try:
                fd = os.open(_inference_lock_path(config.base_url), os.O_CREAT | os.O_RDWR, 0o666)
            except OSError as exc:
                # Unwritable tmp / foreign-owned lockfile: degrade to in-process gating —
                # never let the safety layer itself block inference.
                log.warning("inference lock unavailable (%s) — cross-process gating disabled", exc)
                yield
                return
            # Cancellation-safe acquire (v2.0 Phase-31 critic, BLOCKER 1). The old
            # `await asyncio.to_thread(fcntl.flock, fd, LOCK_EX)` parked a REAL OS
            # thread on the fd; cancelling the awaiting task (e.g. a consolidation
            # pass yielding to a user turn) ran the finally-close while that thread's
            # flock was still in-flight in the kernel — the lock was then granted to a
            # struct-file no fd names anymore, so LOCK_UN could never be called and the
            # shared lockfile wedged for every process on the box: the exact freeze
            # this gate exists to prevent, caused by its own cancellation path.
            # A LOCK_NB poll never blocks a thread: each attempt returns instantly on
            # the event loop, cancellation can only land at the sleep, and the
            # finally-close is always safe (either we hold the lock — close releases
            # it — or we don't). 50ms polling is noise against minutes-long holds.
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    state.check_ceiling()  # (c) bound the flock wait too (raises past ceiling)
                    state.maybe_notify()    # (b) surface it once past the threshold
                    await asyncio.sleep(0.05)
            state.summarize()
            yield
        finally:
            if fd is not None:
                os.close(fd)  # releases the flock; kernel also releases on process death
    finally:
        _inference_sem.release()


def _tools_to_api_format(tools: list[ToolSchema]) -> list[dict]:
    """Serialize ToolSchema list to OpenAI tools API format."""
    result = []
    for t in tools:
        fn = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        result.append({"type": "function", "function": fn})
    return result


class LLMClient:
    """OpenAI-compatible async LLM client with XML fallback and local timeout handling."""

    def __init__(self, config: LLMConfig) -> None:
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
                config.timeout_seconds,
                connect=config.connect_timeout_seconds,
                read=config.timeout_seconds,
                write=config.timeout_seconds,
            ),
            default_headers=config.extra_headers,
            # Local single-tenant GPU: a timed-out generation will time out again on
            # retry — the SDK's silent default (2 retries) turned one 600s failure
            # into 30 min of dead air. Fail fast and let the agent loop react.
            max_retries=0 if config.is_local else 2,
        )
        self._fn_converter: FnCallConverter | None = (
            FnCallConverter() if config.tool_call_mode != "native" else None
        )

    async def detect_capabilities(self) -> CapabilityResult:
        """Probe the model to determine tool call mode and context window. Never raises."""
        start = time.monotonic()
        probe_error: str | None = None
        tool_call_mode: Literal["native", "xml", "text"] = "xml"
        context_window = self.config.context_window

        probe_tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List directory contents",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        try:
            async with _inference_gate(self.config):
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What files are in the current directory?"},
                    ],
                    tools=probe_tools,
                    # Generous cap: preamble-prone models spend 30+ tokens narrating before
                    # the call; at 64 the call got truncated and the probe misread a
                    # native-capable server as xml-only (observed on Qwen3.6 NVFP4).
                    max_tokens=256,
                    temperature=0.0,
                )
            msg = response.choices[0].message
            if msg.tool_calls:
                tool_call_mode = "native"
                log.info("Capability probe: native tool calling confirmed")
            elif msg.content and "<tool_call>" in msg.content:
                tool_call_mode = "xml"
                log.warning("Server returned XML tool calls instead of native — using xml mode")
            else:
                tool_call_mode = "xml"
                log.warning("Could not confirm native function calling, defaulting to xml")
        except openai.BadRequestError as exc:
            probe_error = f"HTTP 400: {exc}"
            tool_call_mode = "xml"
            log.warning("Server rejected tools parameter, forcing XML mode: %s", exc)
        except Exception as exc:
            probe_error = str(exc)
            tool_call_mode = "xml"
            log.warning("Capability probe failed, defaulting to xml: %s", exc)

        # Context window detection
        try:
            models_response = await self._client.models.list()
            for m in models_response.data:
                if m.id == self.config.model:
                    ctx = getattr(m, "context_length", None) or getattr(m, "max_model_len", None)
                    if ctx:
                        context_window = int(ctx)
                    break
        except Exception:
            pass  # Keep default context window

        self.config.tool_call_mode = tool_call_mode
        self.config.context_window = context_window
        self._fn_converter = FnCallConverter() if tool_call_mode != "native" else None

        duration_ms = (time.monotonic() - start) * 1000
        log.info(
            "Capability probe complete: mode=%s, context_window=%d, timeout=%.0fs",
            tool_call_mode,
            context_window,
            self.config.timeout_seconds,
        )

        return CapabilityResult(
            tool_call_mode=tool_call_mode,
            context_window=context_window,
            supports_streaming=True,
            probe_duration_ms=duration_ms,
            probe_error=probe_error,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        disable_thinking: bool = False,
    ) -> tuple[Any, Any]:
        """Single-turn completion. Routes to native or XML based on tool_call_mode.

        Returns (message, usage) — usage is openai.types.CompletionUsage or None.

        disable_thinking: per-call opt-in for INTERNAL harness calls (idle mining/
        consolidation via LLMTextAdapter, compaction summarizer): sends
        extra_body={"chat_template_kwargs": {"enable_thinking": false}} so their
        bounded completion budgets aren't spent on hidden chain-of-thought under a
        reasoning parser. Subject/user-facing turns must NOT set it (#11 — thinking
        stays on; this is deliberately per-call, never an is_local blanket). A
        documented no-op for chat templates without the flag — model-agnostic.
        """
        if self.config.tool_call_mode == "native":
            return await self._complete_native(messages, tools, stream, disable_thinking=disable_thinking)
        return await self._complete_xml(messages, tools, stream, disable_thinking=disable_thinking)

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        disable_thinking: bool = False,
    ) -> tuple[Any, Any]:
        """Streaming completion with per-token callback. Returns (message, usage).

        disable_thinking threads through for INTERNAL harness calls (idle mining/
        consolidation via LLMTextAdapter, compaction summarizer) exactly as complete()
        documents it — never set on subject/user-facing turns (#11)."""
        if self.config.tool_call_mode == "native":
            return await self._complete_native(
                messages, tools, stream=True, on_token=on_token, disable_thinking=disable_thinking
            )
        return await self._complete_xml(messages, tools, stream=True, disable_thinking=disable_thinking)

    async def _complete_native(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        disable_thinking: bool = False,
    ) -> tuple[Any, Any]:
        """Call OpenAI-compat API with tool_calls parameter. Returns (message, usage).

        stream=True uses TRUE HTTP streaming. This is load-bearing for slow local
        models, not a UX nicety: with a non-streaming request the client read-timeout
        races the WHOLE generation (a long completion at single-digit tok/s times out,
        and vLLM never notices the hangup — the orphan keeps eating GPU, slowing the
        retry into the same timeout: the observed zombie cascade). With streaming, the
        read-timeout applies BETWEEN chunks, so a healthy generation can run as long
        as the budget allows, and a client disconnect aborts engine-side generation.
        """
        try:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if tools:
                kwargs["tools"] = _tools_to_api_format(tools)
            if self.config.stop_sequences:
                kwargs["stop"] = self.config.stop_sequences
            # Thinking is deliberately left ON for local subjects (#11): reasoning
            # quality wins over latency. Do not re-add enable_thinking:False — the
            # loop strips <think> blocks before history/parse, and the kwarg was
            # silently dropped by Ollama and type-checked by llama.cpp anyway.
            # Sole scoped exception: INTERNAL calls opt in per-request via
            # disable_thinking (C0 sweep: mining/summarizer budgets starved by
            # hidden CoT under --reasoning-parser) — never an is_local blanket.
            if disable_thinking:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            async with _inference_gate(self.config):
                return await self._create_and_consume(kwargs, stream, on_token)
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def _create_and_consume(
        self,
        kwargs: dict[str, Any],
        stream: bool,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[Any, Any]:
        """Issue the completion request and normalize to (message, usage). stream=True uses
        TRUE HTTP streaming — the read-timeout applies BETWEEN chunks and a client disconnect
        aborts engine-side generation — then buffers the full text client-side (parsing still
        sees the whole response). stream=False is a whole-response request. MUST be called
        inside _inference_gate: the GPU is held until the last chunk, not until create() returns.
        Shared by the native and both XML paths so streaming can never silently diverge (#18)."""
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            response = await self._client.chat.completions.create(**kwargs)
            return await self._consume_native_stream(response, on_token)
        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        # Surface finish_reason on the message so the loop's truncation guard sees it on the
        # non-streaming path too (#77). Best-effort: the SDK message is a pydantic model that
        # may reject an unknown attribute — the loop reads it via getattr(..., None), so a
        # rejection simply leaves the guard dormant here (the live loop uses streaming).
        try:
            message.finish_reason = response.choices[0].finish_reason
        except Exception:
            pass
        return message, response.usage

    @staticmethod
    async def _consume_native_stream(
        response: Any,
        on_token: Callable[[str], Awaitable[None]] | None,
    ) -> tuple[Any, Any]:
        """Assemble a chat-completions chunk stream into (message, usage).

        Tool calls are accumulated as plain dicts (index-keyed deltas: id/name arrive
        on the first fragment, arguments accrete across fragments) — dicts re-serialize
        cleanly when the assistant message is replayed in later request history. Usage
        arrives on the final chunk when stream_options.include_usage is set; None if
        the provider omits it (loop falls back to tiktoken estimation).
        """
        from types import SimpleNamespace

        content_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage = None
        finish_reason = None
        async for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            # finish_reason rides the final content/tool chunk ("stop"|"length"|"tool_calls";
            # None on earlier chunks). Capturing the last non-None one lets the loop refuse to
            # execute a tool call assembled from a completion cut at the output ceiling (#77).
            fr = getattr(choices[0], "finish_reason", None)
            if fr is not None:
                finish_reason = fr
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            piece = getattr(delta, "content", None)
            if piece:
                content_parts.append(piece)
                if on_token is not None:
                    await on_token(piece)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = calls.setdefault(
                    idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["function"]["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["function"]["arguments"] += fn.arguments
        tool_calls = [calls[i] for i in sorted(calls)] or None
        message = SimpleNamespace(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        return message, usage

    async def _complete_xml(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        disable_thinking: bool = False,
    ) -> tuple[Any, Any]:
        """Send tools via API for chat-template injection AND fold the XML tool syntax into the
        system prompt, then parse tool calls from text.

        vLLM injects tools via the model's chat template (e.g. Qwen's native format) when the
        template supports it, so kwargs["tools"] is kept — harmless when unsupported. But
        llama.cpp+Gemma-class servers return HTTP 200 while silently dropping an unsupported
        `tools` param, so relying on kwargs["tools"] alone never told those models a tool exists
        (the old code only injected the system-prompt syntax on the BadRequestError fallback
        below, which a silent-drop never triggers). The injection therefore always runs here.
        Falls back to a `tools`-less request if the API rejects the `tools` param outright.
        Returns (message, usage).
        """
        injected_messages = self._fold_tool_injection(
            self._downgrade_history_for_xml(messages), tools
        )
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": injected_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = _tools_to_api_format(tools)
        if self.config.stop_sequences:
            kwargs["stop"] = self.config.stop_sequences
        if disable_thinking:  # internal-call opt-in — see complete()
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            # #18: honor `stream` — a dead param here silently made XML mode non-streaming
            # for the whole loop. Gate held across create + stream consumption; on
            # BadRequestError the except runs AFTER __aexit__ releases this gate, so
            # _complete_xml_fallback safely takes its own turn (no re-entrant deadlock).
            async with _inference_gate(self.config):
                return await self._create_and_consume(kwargs, stream)
        except openai.BadRequestError:
            # Server rejected the request (e.g. tools param) — retry without it. injected_messages
            # already carries the system-prompt injection folded in above, so the fallback's own
            # injection is a no-op (marker-guarded in _fold_tool_injection — never double-injects).
            log.warning("Server rejected request, falling back to system prompt injection")
            return await self._complete_xml_fallback(
                injected_messages, tools, stream, disable_thinking=disable_thinking
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def _complete_xml_fallback(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        disable_thinking: bool = False,
    ) -> tuple[Any, Any]:
        """Legacy fallback: retry without the `tools` param, tool schemas carried purely via the
        system-prompt XML injection (a no-op if `messages` already carries it — see
        _fold_tool_injection).

        Returns (message, usage).
        """
        msgs = self._fold_tool_injection(self._downgrade_history_for_xml(messages), tools)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": msgs,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.stop_sequences:
            kwargs["stop"] = self.config.stop_sequences
        if disable_thinking:  # internal-call opt-in — see complete()
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            async with _inference_gate(self.config):
                return await self._create_and_consume(kwargs, stream)  # #18: honor stream here too
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    def _fold_tool_injection(
        self, messages: list[Message], tools: list[ToolSchema] | None
    ) -> list[Message]:
        """Fold the XML tool-call syntax into the system message: append to its content with a
        blank line, or insert a new system message if none exists. Always returns a shallow copy
        (matching kwargs["messages"]'s prior defensive-copy contract), even as a no-op — which
        happens when there are no tools, no converter, or the marker shows the injection is
        already present (the guard that keeps _complete_xml -> _complete_xml_fallback from
        injecting the block twice).
        """
        msgs = list(messages)
        if not tools or not self._fn_converter:
            return msgs
        injection = self._fn_converter.build_system_injection(tools)
        if not injection:
            return msgs
        if msgs and msgs[0].get("role") == "system":
            if _TOOL_INJECTION_MARKER in (msgs[0].get("content") or ""):
                return msgs
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + injection}
        else:
            msgs = [{"role": "system", "content": injection}] + msgs
        return msgs

    def _downgrade_history_for_xml(self, messages: list[Message]) -> list[Message]:
        """Re-serialize native tool-call history as template-safe text for xml mode.

        The loop records history in native OpenAI form (assistant `tool_calls` fields,
        `role:"tool"` result messages). Chat templates without tool support (Gemma 3 et al.)
        hard-reject that shape — llama.cpp 400s with "Conversation roles must alternate" the
        moment iteration 2 replays the history, and retrying without the `tools` param can't
        fix a role sequence the template itself refuses to render. So: strip `tool_calls`
        fields (re-rendering them as the taught XML when the assistant content would otherwise
        be empty), rewrite tool results as `<tool_response>` text in user role, and merge
        consecutive same-role turns that a strict-alternation template would reject.
        Idempotent — a second pass finds nothing to rewrite.
        """
        out: list[Message] = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                m = {
                    "role": "user",
                    "content": f"<tool_response>\n{m.get('content') or ''}\n</tool_response>",
                }
            elif role == "assistant" and m.get("tool_calls"):
                stripped = {k: v for k, v in m.items() if k != "tool_calls"}
                if not (stripped.get("content") or "").strip():
                    rendered = "\n".join(
                        "<tool_call>\n<name>{}</name>\n<parameters>{}</parameters>\n</tool_call>".format(
                            (c.get("function") or {}).get("name", ""),
                            (c.get("function") or {}).get("arguments", "{}"),
                        )
                        for c in (m.get("tool_calls") or [])
                        if isinstance(c, dict)
                    )
                    stripped["content"] = rendered
                m = stripped
            if out and m.get("role") in ("user", "assistant") and out[-1].get("role") == m.get("role"):
                merged = ((out[-1].get("content") or "") + "\n\n" + (m.get("content") or "")).strip()
                out[-1] = {**out[-1], "content": merged}
            else:
                out.append(dict(m))
        return out

    def _build_xml_system_injection(self, tools: list[ToolSchema]) -> str:
        """Serialize tools as XML schema block."""
        if self._fn_converter:
            return self._fn_converter.build_system_injection(tools)
        return ""

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Map openai SDK exceptions to LocalHarness provider error types."""
        if isinstance(exc, ProviderError):
            # Already one of ours — e.g. the inference gate's #62 fail-fast (ProviderConnectionError)
            # or queue-wait ceiling (ProviderTimeoutError). Pass it through so the loop's specific
            # handlers still fire; re-wrapping would downgrade it to a base ProviderError.
            return exc
        if isinstance(exc, openai.APIConnectionError):
            return ProviderConnectionError(str(exc), cause=exc)
        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeoutError(str(exc), cause=exc)
        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimitError(str(exc), cause=exc)
        if isinstance(exc, openai.APIStatusError):
            return ProviderAPIError(str(exc), status_code=exc.status_code, cause=exc)
        return ProviderError(str(exc), cause=exc)
