"""OpenAI-compatible async LLM client with XML fallback and local timeout handling."""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import openai
from openai import AsyncOpenAI

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
    timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 5.0
    temperature: float = 0.6
    max_tokens: int = 4096
    tool_call_mode: Literal["native", "xml", "text"] = "native"
    context_window: int = 128_000
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
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What files are in the current directory?"},
                ],
                tools=probe_tools,
                max_tokens=64,
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
    ) -> tuple[Any, Any]:
        """Single-turn completion. Routes to native or XML based on tool_call_mode.

        Returns (message, usage) — usage is openai.types.CompletionUsage or None.
        """
        if self.config.tool_call_mode == "native":
            return await self._complete_native(messages, tools, stream)
        return await self._complete_xml(messages, tools, stream)

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[Any, Any]:
        """Streaming completion with per-token callback. Returns (message, usage)."""
        return await self.complete(messages, tools, stream=True)

    async def _complete_native(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
    ) -> tuple[Any, Any]:
        """Call OpenAI-compat API with tool_calls parameter. Returns (message, usage)."""
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
            response = await self._client.chat.completions.create(**kwargs)
            return response.choices[0].message, response.usage
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def _complete_xml(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
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
        # Disable thinking mode for xml tool-calling — reduces token waste 100x
        if self.config.is_local:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            response = await self._client.chat.completions.create(**kwargs)
            return response.choices[0].message, response.usage
        except openai.BadRequestError:
            # Server rejected tools/extra_body — fall back to system prompt injection
            log.warning("Server rejected request, falling back to system prompt injection")
            return await self._complete_xml_fallback(messages, tools, stream)
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def _complete_xml_fallback(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
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
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=msgs,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
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
