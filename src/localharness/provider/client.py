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

import openai
from openai import AsyncOpenAI

try:
    import fcntl
except ImportError:  # non-POSIX: no cross-process lock, in-process semaphore still applies
    fcntl = None  # type: ignore[assignment]

from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS
from localharness.core.types import Message, ToolCall, ToolSchema
from localharness.provider.detector import LOCAL_INFERENCE_TIMEOUT_MIN
from localharness.provider.fn_call import FnCallConverter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


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


def _inference_lock_path(base_url: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9.]+", "-", base_url.split("://", 1)[-1]).strip("-")
    return os.path.join(tempfile.gettempdir(), f"localharness-inference-{safe}.lock")


@asynccontextmanager
async def _inference_gate(config: LLMConfig):
    """Hold for the FULL request including stream consumption — the GPU is occupied
    until the last token, not until the HTTP call returns."""
    if not config.is_local:
        yield
        return
    async with _inference_sem:
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
            t0 = time.monotonic()
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
                    await asyncio.sleep(0.05)
            waited = time.monotonic() - t0
            if waited > 5:
                log.info("inference gate: waited %.1fs for another process's generation", waited)
            yield
        finally:
            if fd is not None:
                os.close(fd)  # releases the flock; kernel also releases on process death


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
    ) -> tuple[Any, Any]:
        """Streaming completion with per-token callback. Returns (message, usage)."""
        if self.config.tool_call_mode == "native":
            return await self._complete_native(messages, tools, stream=True, on_token=on_token)
        return await self._complete_xml(messages, tools, stream=True)

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
                if stream:
                    kwargs["stream"] = True
                    kwargs["stream_options"] = {"include_usage": True}
                    response = await self._client.chat.completions.create(**kwargs)
                    return await self._consume_native_stream(response, on_token)
                response = await self._client.chat.completions.create(**kwargs)
                return response.choices[0].message, response.usage
        except Exception as exc:
            raise self._wrap_error(exc) from exc

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
        async for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
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
        )
        return message, usage

    async def _complete_xml(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        disable_thinking: bool = False,
    ) -> tuple[Any, Any]:
        """Send tools via API for chat-template injection, parse tool calls from text.

        vLLM injects tools via the model's chat template (e.g. Qwen's native format),
        producing much better compliance than custom system-prompt injection.
        Falls back to system-prompt injection if the API rejects the tools param.
        Returns (message, usage).
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
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
            # Gate scoped to the create alone: the except-path re-enters via
            # _complete_xml_fallback, which takes its own turn at the gate.
            async with _inference_gate(self.config):
                response = await self._client.chat.completions.create(**kwargs)
            return response.choices[0].message, response.usage
        except openai.BadRequestError:
            # Server rejected the request (e.g. tools param) — fall back to system prompt injection
            log.warning("Server rejected request, falling back to system prompt injection")
            return await self._complete_xml_fallback(messages, tools, stream, disable_thinking=disable_thinking)
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def _complete_xml_fallback(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        disable_thinking: bool = False,
    ) -> tuple[Any, Any]:
        """Legacy fallback: inject tool schemas into system prompt as XML text.

        Returns (message, usage).
        """
        msgs = list(messages)
        if tools and self._fn_converter:
            injection = self._fn_converter.build_system_injection(tools)
            if injection and msgs and msgs[0].get("role") == "system":
                msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + injection}
            elif injection:
                msgs = [{"role": "system", "content": injection}] + msgs
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
                response = await self._client.chat.completions.create(**kwargs)
            return response.choices[0].message, response.usage
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    def _build_xml_system_injection(self, tools: list[ToolSchema]) -> str:
        """Serialize tools as XML schema block."""
        if self._fn_converter:
            return self._fn_converter.build_system_injection(tools)
        return ""

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Map openai SDK exceptions to LocalHarness provider error types."""
        if isinstance(exc, openai.APIConnectionError):
            return ProviderConnectionError(str(exc), cause=exc)
        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeoutError(str(exc), cause=exc)
        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimitError(str(exc), cause=exc)
        if isinstance(exc, openai.APIStatusError):
            return ProviderAPIError(str(exc), status_code=exc.status_code, cause=exc)
        return ProviderError(str(exc), cause=exc)
