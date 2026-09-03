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
from pathlib import Path
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
    # Runtime family serving this endpoint ("vllm"/"llamacpp"/"ollama"/"lmstudio"/None-unknown).
    # Keys the measured-speed ledger (speed_stats) — None disables recording, never guesses.
    # rebind_endpoint() must carry the TARGET endpoint's type (a stale type would file samples
    # under the wrong runtime).
    provider_type: str | None = None
    # The SESSION's config dir (start's `--config-dir`), for config-dir-relative state — today
    # the speed ledger. None keeps the #35 ambient precedence ($LOCALHARNESS_DIR else ~), which
    # is only ever right when no explicit dir was given: `--config-dir` never exports the env
    # var, so a client without this writes a ledger the REPL (which reads the session's dir)
    # never sees.
    config_dir: str | Path | None = None


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
    # Reachability is a SEPARATE axis from capability: True whenever the server ANSWERED the
    # probe for this model (a 200, or a 400 rejecting `tools`) even if no tool call was ever
    # observed. start's abort gate reads THIS — never probe_error's text, which conflated
    # "inconclusive" with "dead endpoint" in 0.12.0.
    server_reached: bool = False


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

# CAPABILITY probe retry (distinct from the TCP reachability probe above). detect_capabilities
# decides tool_call_mode for the WHOLE session off one generation, so a transient failure used to
# condemn every later turn to xml. Retried on an inconclusive result; a definitive one breaks out
# first, so a genuinely xml-only server pays these attempts once at startup, not per turn.
_PROBE_ATTEMPTS = 3
_PROBE_RETRY_DELAY_S = 1.0
# One prompt per attempt, all obvious list_files triggers. Retrying the IDENTICAL request is not
# an independent sample: llama.cpp routes it to the same slot by prefix match and replays the same
# KV-cache state, so the same completion comes back deterministically — observed live with
# DeepSeek-V4-Flash Q2_K: 9/9 retries no parsable tool call, then the identical request after a
# fresh prompt eval called the tool. Varying the prompt makes each retry a new evaluation.
_PROBE_PROMPTS = (
    "What files are in the current directory?",
    "Show me the files in this directory.",
    "Use the available tool to list this directory's contents.",
)


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


_TOOL_NAME_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]")


def _tools_to_api_format(tools: list[ToolSchema]) -> tuple[list[dict], dict[str, str]]:
    """Serialize ToolSchema list to OpenAI tools API format, sanitizing names for the wire.

    Registry names like `mcp:fetch` / `plugin:research_tools.exa_search` violate the OpenAI
    function-name grammar (^[a-zA-Z0-9_-]{1,64}$); llama.cpp rejects the WHOLE request with
    HTTP 400, silently knocking every MCP/plugin scenario off the native tool path (observed
    live: 44x 400 in one bench run, each retried into the XML fallback plus 3 parse-fail
    nudges per sample). Names are sanitized for the wire; the returned unmap
    (sanitized -> original) restores registry names on parsed responses so dispatch still
    finds the real tool. Models sometimes echo the ORIGINAL name from replayed history —
    unmap.get(name, name) passes those through untouched, so both forms dispatch.
    """
    result: list[dict] = []
    unmap: dict[str, str] = {}
    for t in tools:
        fn = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        original = fn.get("name", "")
        safe = base = _TOOL_NAME_UNSAFE.sub("_", original)[:64] or "tool"
        n = 2
        while safe in unmap and unmap[safe] != original:
            safe = f"{base[:60]}_{n}"
            n += 1
        if safe != original:
            fn["name"] = safe
        unmap[safe] = original
        result.append({"type": "function", "function": fn})
    return result, unmap


def _reasoning_text(obj: Any) -> str | None:
    """Thinking text off a streaming delta or a response message, whichever field the runtime
    spells it with (#142).

    vLLM and llama.cpp send `reasoning_content`; Ollama's OpenAI-compatible endpoint sends
    `reasoning`. Returns "" for a thinking delta carrying no text yet — a DIFFERENT answer
    from None ("no reasoning field on this object at all"), which is what lets the decode
    window open at the first thinking delta instead of at the first answer token. Ollama
    additionally streams an empty `content` while thinking, so a falsy test on either field
    made the whole reasoning phase look like dead air.
    """
    for name in ("reasoning_content", "reasoning"):
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


async def _aclose_quietly(client: AsyncOpenAI) -> None:
    """Close an AsyncOpenAI and its httpx pool, swallowing teardown errors: a failed close must
    never crash a /model swap or the session's ordered shutdown — but is never silent (#154)."""
    try:
        await client.close()
    except Exception as exc:  # noqa: BLE001 — teardown: log it, never propagate it
        log.debug("llm client close failed: %s", exc)


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
        # Sticky per-client memory that the server rejected the `tools` param outright
        # (BadRequestError in xml OR native mode) — without it every later iteration re-sends
        # `tools=`, eats another 400 round-trip, and falls back again. Per-SERVER state:
        # rebind_endpoint() resets it because a different server may accept `tools`.
        self._tools_param_rejected = False
        # Native-mode twin (audit Fix E): a server that 400s the disable_thinking
        # extra_body={"chat_template_kwargs": ...} param. Same sticky/per-server contract.
        self._extra_body_rejected = False
        self._client = self._build_client()
        # #154: the AsyncOpenAI owns an httpx pool (sockets + fds). `_closed` makes aclose()
        # idempotent; `_closing` holds the background closes of clients replaced by
        # rebind_endpoint (a strong ref, so a fire-and-forget task is never GC'd mid-close).
        self._closed = False
        self._closing: set[asyncio.Task] = set()
        self._fn_converter: FnCallConverter | None = (
            FnCallConverter() if config.tool_call_mode != "native" else None
        )
        # Live decode-speed state (speed_stats): while a stream is active this points at that
        # stream's progress dict ({"first_at","chunks","server_tps"}) for the UI's tick-poll;
        # last_gen_tps is the previous stream's VERIFIED rate (exact usage over the measured
        # window, or the engine's own reported rate). Concurrent streams on one client would
        # race the pointer — display-only state, the per-call ledger recording stays correct.
        self._stream_progress: dict | None = None
        self.last_gen_tps: float | None = None
        # Tokens carried per streamed delta. 1.0 without speculative decoding; WITH it (vLLM
        # MTP / EAGLE / Medusa) a delta carries every accepted draft token — 2.8-3.4 measured on
        # qwen3.8-27b + qwen3_5_mtp — so an unscaled chunk count understates the live rate by
        # the acceptance length (23.6 tok/s displayed as 6.9). None = never measured for this
        # model: the live readout is SUPPRESSED rather than guessing 1.0, because that guess is
        # wrong by 3x on exactly the runtimes people most want a speed number from. Seeded from
        # the ledger here (one small read at construction) so a NEW session's FIRST turn is
        # already honest; gen_speed_snapshot stays I/O-free for the UI tick.
        self._tokens_per_chunk: float | None = None
        try:
            from localharness.provider.speed_stats import (
                default_speed_stats_path, tokens_per_chunk as _tpc,
            )
            if self.config.provider_type:
                self._tokens_per_chunk = _tpc(
                    default_speed_stats_path(self.config.config_dir),
                    self.config.provider_type, self.config.model,
                )
        except Exception:  # a ledger miss must never block constructing a client
            self._tokens_per_chunk = None

    def _build_client(self) -> AsyncOpenAI:
        """Construct the AsyncOpenAI bound to the CURRENT self.config (base_url/api_key/headers/
        timeout). Factored out of __init__ so rebind_endpoint() re-points at a different server with
        the SAME timeout + retry policy — the timeout math lives in exactly one place."""
        c = self.config
        return AsyncOpenAI(
            base_url=c.base_url,
            api_key=c.api_key,
            timeout=openai.Timeout(
                c.timeout_seconds,
                connect=c.connect_timeout_seconds,
                read=c.timeout_seconds,
                write=c.timeout_seconds,
            ),
            default_headers=c.extra_headers,
            # Local single-tenant GPU: a timed-out generation will time out again on
            # retry — the SDK's silent default (2 retries) turned one 600s failure
            # into 30 min of dead air. Fail fast and let the agent loop react.
            max_retries=0 if c.is_local else 2,
        )

    async def aclose(self) -> None:
        """Release the underlying AsyncOpenAI — its httpx connection pool, sockets and fds (#154).

        Idempotent (a second call is a no-op) and never raises: callers close in a `finally`,
        where a teardown error would mask the real exit reason. Also drains the background closes
        of any endpoints this client was rebound away from, so shutdown leaves nothing in flight."""
        if not self._closed:
            self._closed = True
            await _aclose_quietly(self._client)
        if self._closing:
            await asyncio.gather(*list(self._closing), return_exceptions=True)

    def _schedule_close(self, client: AsyncOpenAI) -> None:
        """Close a REPLACED AsyncOpenAI from a SYNC method (rebind_endpoint). With a loop running —
        every /model swap path — the close runs as a background task kept in `_closing`; with no
        loop there is nothing to await on, so the transport is left to GC. Logged, never raised."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.debug("rebind_endpoint: no running loop — replaced client left to GC")
            return
        task = loop.create_task(_aclose_quietly(client))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    _REBIND_UNSET: Any = object()  # "param not passed" — None is a real value (unknown runtime)

    def rebind_endpoint(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        provider_type: str | None | Any = _REBIND_UNSET,
    ) -> None:
        """Re-point this client at a DIFFERENT server (a cross-endpoint /model swap). The
        AsyncOpenAI bakes base_url in at construction, so re-pointing REBUILDS it, and resets the
        per-server sticky `_tools_param_rejected` (the new server may accept the `tools` param).

        The caller then sets `config.model` and MUST call `detect_capabilities()` — which
        re-derives `tool_call_mode` AND refreshes `_fn_converter` (created for xml/text, cleared
        for native). So this method deliberately does NOT touch `_fn_converter`: the mode isn't
        known until the post-rebind probe runs, and detect_capabilities() is the single place that
        keeps mode and converter in lockstep (the same path the same-endpoint hot-swap uses).

        Exception-safe (mirrors TokenCounter.rebind, #30): if the rebuild raises, the prior
        client + config are restored and the error re-raised, so a failed re-point never strands a
        half-configured client mid-session."""
        prev = (
            self.config.base_url,
            self.config.api_key,
            self.config.extra_headers,
            self._client,
            self._tools_param_rejected,
            self._extra_body_rejected,
            self.config.provider_type,
        )
        self.config.base_url = base_url
        if api_key is not None:
            self.config.api_key = api_key
        if extra_headers is not None:
            self.config.extra_headers = dict(extra_headers)
        if provider_type is not self._REBIND_UNSET:
            # SET, not leave-unchanged (the extra_headers #3 lesson): callers pass the target's
            # actual type — None included — so a peer of unknown runtime never inherits the old
            # endpoint's type and files speed samples under the wrong ledger key.
            self.config.provider_type = provider_type
        try:
            self._client = self._build_client()
        except Exception:
            (
                self.config.base_url,
                self.config.api_key,
                self.config.extra_headers,
                self._client,
                self._tools_param_rejected,
                self._extra_body_rejected,
                self.config.provider_type,
            ) = prev
            raise
        # #154: the rebuild took, so the endpoint we just left is unreachable through this client —
        # close its pool. AFTER the restore path, which puts prev[3] back as the LIVE client.
        self._schedule_close(prev[3])
        self._closed = False  # a fresh pool: a later aclose() must still close something
        self._tools_param_rejected = False
        self._extra_body_rejected = False
        # Success only (a failed rebind still serves the OLD endpoint, whose rate is still true):
        # the previous server's verified decode rate says nothing about the new one.
        self.last_gen_tps = None

    async def detect_capabilities(self) -> CapabilityResult:
        """Probe the model to determine tool call mode and context window. Never raises."""
        start = time.monotonic()
        # A measured rate belongs to the model that produced it. Every /model swap path assigns
        # config.model then calls this, so clearing here is the one choke point that keeps the
        # OLD model's tok/s from being shown — in green, as VERIFIED — for the new one.
        self.last_gen_tps = None
        probe_error: str | None = None
        tool_call_mode: Literal["native", "xml", "text"] = "xml"
        context_window = self.config.context_window
        # Tracked separately from the mode: reached = the server answered the probe for this
        # model at least once; rejected = it answered 400 to the `tools` param (the one case
        # where taught-xml is the only option left).
        server_reached = False
        server_rejected_tools = False

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

        # ONE request used to decide the whole session: any hiccup — a busy slot, a slow first
        # token, a connection blip — silently downgraded every later turn to xml, and a model
        # whose native syntax is not <tool_call> (DeepSeek emits DSML) then had EVERY call
        # rendered as chat text. Nothing executed and the model looked like it was lying about
        # its work. So an INCONCLUSIVE probe is retried; only a DEFINITIVE answer ends the loop.
        # Definitive = native confirmed, taught-XML seen, or the server rejecting `tools` at all.
        for attempt in range(_PROBE_ATTEMPTS):
            try:
                async with _inference_gate(self.config):
                    response = await self._client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                # Varied per attempt — see _PROBE_PROMPTS: an identical retry
                                # replays the runtime's cached state and its exact completion.
                                "content": _PROBE_PROMPTS[attempt % len(_PROBE_PROMPTS)],
                            },
                        ],
                        tools=probe_tools,
                        # Generous cap: preamble-prone models spend 30+ tokens narrating before
                        # the call; at 64 the call got truncated and the probe misread a
                        # native-capable server as xml-only (observed on Qwen3.6 NVFP4).
                        max_tokens=256,
                        temperature=0.0,
                    )
                server_reached = True  # an HTTP answer for this model — the endpoint is alive
                msg = response.choices[0].message
                if msg.tool_calls:
                    tool_call_mode = "native"
                    probe_error = None
                    log.info("Capability probe: native tool calling confirmed")
                    break
                if msg.content and "<tool_call>" in msg.content:
                    tool_call_mode = "xml"
                    probe_error = None
                    log.warning(
                        "Server returned XML tool calls instead of native — using xml mode"
                    )
                    break
                probe_error = "no tool call in probe response"
            except openai.BadRequestError as exc:
                # A 400 is the server ANSWERING: it does not accept `tools`. Retrying cannot
                # change that, so this is definitive and ends the loop immediately.
                server_reached = True
                server_rejected_tools = True
                probe_error = f"HTTP 400: {exc}"
                tool_call_mode = "xml"
                log.warning("Server rejected tools parameter, forcing XML mode: %s", exc)
                break
            except Exception as exc:
                probe_error = str(exc)
            if attempt < _PROBE_ATTEMPTS - 1:
                log.warning(
                    "Capability probe inconclusive (attempt %d/%d): %s — retrying",
                    attempt + 1,
                    _PROBE_ATTEMPTS,
                    probe_error,
                )
                await asyncio.sleep(_PROBE_RETRY_DELAY_S * (attempt + 1))
        if probe_error and tool_call_mode != "native":
            # ACTUAL attempts, not the ceiling: a definitive answer (HTTP 400) breaks on
            # the first pass, and reporting "after 3 attempts" there sends whoever is
            # debugging a probe failure looking for two retries that never happened.
            if server_reached and not server_rejected_tools:
                # Exhausted-INCONCLUSIVE on a server that ACCEPTED `tools` but never demonstrated
                # one. BOTH static defaults fail silently here, in opposite directions: xml
                # condemns a merely-shy native server whose dialect is not <tool_call> to having
                # every call rendered as chat text, while native leaves the other class in this
                # branch — the server that answers 200 while SILENTLY DROPPING the tools param
                # (llama.cpp without --jinja, Gemma-class templates, see _complete_xml) — with
                # nothing ever telling the model a tool exists: it emits prose, no ParseFailed can
                # fire (loop.py gates that on a recognizable call attempt) and the session is
                # tool-blind for its whole life. So stop guessing and ASK: one deciding attempt
                # with the taught syntax folded in, and decide on what comes back.
                log.warning(
                    "Capability probe never confirmed native tool calling after %d attempts "
                    "(%s) — server accepts the tools param but never used it; running one "
                    "deciding probe with the taught XML syntax folded into the system prompt.",
                    attempt + 1,
                    probe_error,
                )
                tool_call_mode = await self._decide_mode_by_taught_xml(probe_tools)
            else:
                log.warning(
                    "Capability probe never confirmed native tool calling after %d attempts "
                    "(%s) — falling back to xml. A model whose native syntax is not "
                    "<tool_call> will have its calls rendered as chat text in this mode; "
                    "check the runtime's tool-call parser flag (llama.cpp --jinja, "
                    "vLLM --tool-call-parser).",
                    attempt + 1,
                    probe_error,
                )

        # Context window detection — vLLM/OpenAI-compat /v1/models shape ONLY (context_length|
        # max_model_len). llama.cpp/Ollama/LM Studio don't report a window here, so this stays the
        # config default for them. The PROVIDER-AWARE source of truth is context.probe_served_window
        # (llama.cpp /props, Ollama /api/show, LM Studio /api/v0/models); start/repl use IT for the
        # window guard + budget refit. Not unified here to keep CapabilityResult.context_window's
        # contract (and its callers/tests) stable — reconcile if this probe ever needs the peers.
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
            server_reached=server_reached,
        )

    async def _decide_mode_by_taught_xml(
        self, probe_tools: list[dict]
    ) -> Literal["native", "xml"]:
        """ONE deciding attempt for the exhausted-inconclusive branch, decided on EVIDENCE.

        Re-asks with the taught-XML instruction folded into the system prompt exactly as xml mode
        does at runtime (FnCallConverter.build_system_injection — the same text the session would
        actually run on, not a probe-only variant). Decision rule:
          * a <tool_call block comes back -> taught xml is PROVEN workable here -> "xml"
          * no taught XML in the response -> no evidence xml works -> "native", the owner's
            2026-08-10 tiebreak for absence of evidence (and the right answer for the server that
            silently drops `tools` yet calls in its own dialect).
        Single attempt, no loop — the 3-attempt probe above already widened the transient window;
        any transport failure keeps the current xml default (fail-safe) and says so.
        """
        converter = self._fn_converter or FnCallConverter()
        # build_system_injection reads name/description/parameters: hand it the inner function
        # dicts, not the OpenAI {"type": "function", ...} envelope.
        injection = converter.build_system_injection([t["function"] for t in probe_tools])
        try:
            async with _inference_gate(self.config):
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        # Same fold as _fold_tool_injection: system content, blank line, block.
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.\n\n" + injection,
                        },
                        # The most explicit trigger of the three. Reusing a prompt the loop already
                        # sent is safe here: the new system prefix makes this a fresh evaluation,
                        # not a prefix-cache replay (see _PROBE_PROMPTS).
                        {"role": "user", "content": _PROBE_PROMPTS[-1]},
                    ],
                    tools=probe_tools,
                    max_tokens=256,
                    temperature=0.0,
                )
            msg = response.choices[0].message
            if "<tool_call" in (getattr(msg, "content", None) or ""):
                log.warning(
                    "Deciding probe: the model emitted the taught <tool_call> XML — xml is proven "
                    "workable on this server, using xml."
                )
                return "xml"
            log.warning(
                "Deciding probe: no taught <tool_call> XML in the response — no evidence xml "
                "works here, using native. If tool calls now arrive in the model's own dialect "
                "and go unparsed, check the runtime's tool-call parser flag (llama.cpp --jinja, "
                "vLLM --tool-call-parser)."
            )
            return "native"
        except Exception as exc:
            log.warning("Deciding probe failed (%s) — keeping xml (fail-safe).", exc)
            return "xml"

    @staticmethod
    async def _consume_bounded(coro: Awaitable[Any], gen_timeout: float | None) -> tuple[Any, Any]:
        """Await the (create+consume) coroutine, bounding ONLY generation (#92). This runs INSIDE
        the inference gate — i.e. after the permit is held — so gen_timeout caps generation time,
        never the wait for a slot. None = unbounded (today's behavior for every non-tier-2 call)."""
        if gen_timeout is None:
            return await coro
        return await asyncio.wait_for(coro, gen_timeout)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        disable_thinking: bool = False,
        gen_timeout: float | None = None,
    ) -> tuple[Any, Any]:
        """Single-turn completion. Routes to native or XML based on tool_call_mode.

        Returns (message, usage) — usage is openai.types.CompletionUsage or None.

        gen_timeout: per-call bound on GENERATION only (applied after the inference permit is
        acquired). Used by the tier-2 input classifier so its 5s clock is a generation clock, not
        one that a permit-wait can consume (#92). None = unbounded.

        disable_thinking: per-call opt-in for INTERNAL harness calls (idle mining/
        consolidation via LLMTextAdapter, compaction summarizer): sends
        extra_body={"chat_template_kwargs": {"enable_thinking": false}} so their
        bounded completion budgets aren't spent on hidden chain-of-thought under a
        reasoning parser. Subject/user-facing turns must NOT set it (#11 — thinking
        stays on; this is deliberately per-call, never an is_local blanket). A
        documented no-op for chat templates without the flag — model-agnostic.
        """
        if self.config.tool_call_mode == "native":
            return await self._complete_native(messages, tools, stream,
                                               disable_thinking=disable_thinking, gen_timeout=gen_timeout)
        return await self._complete_xml(messages, tools, stream,
                                        disable_thinking=disable_thinking, gen_timeout=gen_timeout)

    async def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        disable_thinking: bool = False,
        gen_timeout: float | None = None,
    ) -> tuple[Any, Any]:
        """Streaming completion with per-token callback. Returns (message, usage).

        disable_thinking / gen_timeout thread through for INTERNAL harness calls exactly as
        complete() documents them — never set on subject/user-facing turns (#11)."""
        if self.config.tool_call_mode == "native":
            return await self._complete_native(
                messages, tools, stream=True, on_token=on_token,
                disable_thinking=disable_thinking, gen_timeout=gen_timeout,
            )
        return await self._complete_xml(messages, tools, stream=True,
                                        disable_thinking=disable_thinking, gen_timeout=gen_timeout)

    async def _complete_native(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        disable_thinking: bool = False,
        gen_timeout: float | None = None,
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
        kwargs, name_unmap = self._native_kwargs(messages, tools, disable_thinking)
        try:
            async with _inference_gate(self.config):
                return await self._consume_bounded(
                    self._create_and_consume(kwargs, stream, on_token, name_unmap=name_unmap),
                    gen_timeout,
                )
        except openai.BadRequestError as exc:
            # Symmetry with _complete_xml (audit Fix E): the XML path self-heals a rejected `tools`
            # param; native used to hard-error on the same 400. Scope the retry to the KNOWN optional
            # params (tools, extra_body/chat_template_kwargs) — a genuinely malformed 400 must still
            # surface, never a blanket retry. Remember the rejection (sticky, per-server) so we don't
            # re-eat it every turn, then retry ONCE with the offending param dropped.
            dropped = self._remember_native_rejection(exc, kwargs)
            if dropped is None:
                raise self._wrap_error(exc) from exc
            log.warning("Server rejected native `%s` param (400) — retrying once without it", dropped)
            kwargs, name_unmap = self._native_kwargs(messages, tools, disable_thinking)  # sticky → omitted
            try:
                async with _inference_gate(self.config):
                    return await self._consume_bounded(
                        self._create_and_consume(kwargs, stream, on_token, name_unmap=name_unmap),
                        gen_timeout,
                    )
            except Exception as retry_exc:
                raise self._wrap_error(retry_exc) from retry_exc
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    def _native_kwargs(
        self, messages: list[Message], tools: list[ToolSchema] | None, disable_thinking: bool
    ) -> tuple[dict[str, Any], dict[str, str] | None]:
        """Build native-mode request kwargs, HONORING the per-server sticky rejections so a param a
        prior turn 400'd on is never re-sent (mirrors _complete_xml's `not self._tools_param_rejected`
        guard). Returns (kwargs, name_unmap). name_unmap is None whenever `tools` is omitted.

        Thinking is deliberately left ON for local subjects (#11): reasoning quality wins over
        latency, the loop strips <think> before history/parse, and the kwarg is silently dropped by
        Ollama / type-checked by llama.cpp anyway. Sole scoped exception: INTERNAL calls opt in via
        disable_thinking (mining/summarizer budgets starved by hidden CoT under --reasoning-parser) —
        never an is_local blanket; a server that 400s it is caught by the _extra_body_rejected flag."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        name_unmap: dict[str, str] | None = None
        if tools and not self._tools_param_rejected:
            kwargs["tools"], name_unmap = _tools_to_api_format(tools)
        if self.config.stop_sequences:
            kwargs["stop"] = self.config.stop_sequences
        if disable_thinking and not self._extra_body_rejected:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return kwargs, name_unmap

    def _remember_native_rejection(
        self, exc: openai.BadRequestError, sent_kwargs: dict[str, Any]
    ) -> str | None:
        """Map a native 400 to a KNOWN rejected optional param, set its per-server sticky flag, and
        return the param name to drop — or None when the error isn't a recognized param rejection (a
        genuinely malformed request → surface it, never blanket-retry). Matches only a param we
        ACTUALLY sent whose name the server's own error text implicates."""
        text = str(exc).lower()
        if "extra_body" in sent_kwargs and any(
            s in text for s in ("chat_template_kwargs", "enable_thinking", "extra_body")
        ):
            self._extra_body_rejected = True
            return "extra_body"
        if "tools" in sent_kwargs and "tool" in text:
            self._tools_param_rejected = True
            return "tools"
        return None

    async def _create_and_consume(
        self,
        kwargs: dict[str, Any],
        stream: bool,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        name_unmap: dict[str, str] | None = None,
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
            # Local dict per call (concurrent streams each keep their own); the instance
            # pointer only serves the UI's tick-poll and is cleared before measurement.
            progress: dict[str, Any] = {"first_at": None, "chunks": 0, "server_tps": None}
            self._stream_progress = progress
            try:
                response = await self._client.chat.completions.create(**kwargs)
                message, usage = await self._consume_native_stream(response, on_token, progress)
            finally:
                self._stream_progress = None
            # Only a cleanly finished stream reaches here — errors/cancels above skip recording.
            self._note_gen_speed(progress, usage)
        else:
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
            # #142: every downstream reader asks for `reasoning_content`, which is what vLLM and
            # llama.cpp send — Ollama names the same thing `reasoning`, so its thinking text
            # reached nobody. Fill the canonical field from whichever one arrived; never
            # overwrite a value the runtime already put there. Best-effort assignment for the
            # same reason finish_reason is above (pydantic may refuse an unknown attribute).
            if getattr(message, "reasoning_content", None) is None:
                alt = _reasoning_text(message)
                if alt is not None:
                    try:
                        message.reasoning_content = alt
                    except Exception:
                        pass
            usage = response.usage
        self._unmap_tool_call_names(message, name_unmap)
        return message, usage

    def _note_gen_speed(self, progress: dict, usage: Any) -> None:
        """Verified decode rate of a cleanly finished stream → last_gen_tps + the speed ledger.

        The engine's own reported rate (llama.cpp `timings.predicted_per_second`) wins when
        present; otherwise exact usage completion_tokens over the measured first-delta→done
        wall window, which must clear speed_stats' substance floors to count as a measurement
        at all (#136). No usage and no engine rate → nothing recorded (never estimate). Ledger
        recording additionally needs config.provider_type (the key) — last_gen_tps still
        updates without it — and lands in config.config_dir, the SESSION's dir, which is what
        the REPL reads back. Never raises: a stats miss must not fail a completion."""
        from localharness.provider.speed_stats import (
            MAX_PLAUSIBLE_TPS, decode_tps, default_speed_stats_path, is_substantive_sample,
            record_tokens_per_chunk, record_tps,
        )
        try:
            tps = progress.get("server_tps")
            if not tps:
                ctokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                first_at = progress.get("first_at")
                if ctokens and first_at is not None:
                    done_at = time.monotonic()
                    # #136: this window times chunk ARRIVAL. A burst the runtime buffered and
                    # flushed in one or two deltas measures transport, not decoding — 60 tokens
                    # in 12ms reads as 4,916 tok/s, which the ceiling below happily admits. Only
                    # a sample with real substance may become a rate at all.
                    if not is_substantive_sample(ctokens, done_at - first_at):
                        log.debug("skipping insubstantial decode sample (%s tokens over %.3fs)",
                                  ctokens, done_at - first_at)
                        return
                    tps = decode_tps(ctokens, first_at, done_at)
                    # Learn tokens-per-chunk for the NEXT stream's live readout. Only from an
                    # exact usage count over a substantive sample — the same evidence the
                    # verified rate is built from. Clamped at >=1.0: a chunk cannot carry less
                    # than one token, and a ratio below 1 would only ever come from counting
                    # deltas the token accounting didn't (e.g. a trailing empty delta).

            if not tps or tps <= 0:
                return
            # #130: a sub-millisecond measured window turns an exact token count into tens of
            # thousands of tok/s. That is a timing artifact, not a decode rate — drop it before it
            # reaches last_gen_tps (the status line) or the ledger's median. Debug, not warning:
            # nothing is wrong with the session, the sample is just unusable.
            if float(tps) > MAX_PLAUSIBLE_TPS:
                log.debug("discarding implausible decode rate %.1f tok/s (degenerate window)", tps)
                return
            self.last_gen_tps = float(tps)
            # Learn tokens-per-delta for the live readout, but only from a sample that survived
            # every rejection above — a run dropped as degenerate must leave the ledger (and the
            # file itself) untouched. Clamped >=1.0: a delta cannot carry less than one token.
            _chunks = progress.get("chunks") or 0
            _ctokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            if _chunks > 0 and _ctokens:
                _ratio = max(1.0, _ctokens / _chunks)
                self._tokens_per_chunk = _ratio   # in-memory even without provider_type
            else:
                _ratio = None
            if self.config.provider_type:
                record_tps(default_speed_stats_path(self.config.config_dir),
                           self.config.provider_type, self.config.model, float(tps))
                if _ratio is not None:
                    record_tokens_per_chunk(
                        default_speed_stats_path(self.config.config_dir),
                        self.config.provider_type, self.config.model, _ratio,
                    )
        except Exception as exc:
            log.warning("speed ledger update failed (non-fatal): %s", exc)

    def gen_speed_snapshot(self) -> tuple[float, bool] | None:
        """(tok/s, verified) for a status line, or None when nothing is known yet.

        While a stream is active: its live chunk rate scaled by the tokens-per-chunk ratio
        learned from the last finished stream, so approximate (verified=False); the number
        self-corrects to the exact rate at stream end. The scale matters under speculative
        decoding, where one delta carries every accepted draft token — counting chunks as
        tokens there understates the rate by the acceptance length (measured 3.40x on
        qwen3.8-27b + MTP). Without spec decode the ratio is 1.0 and this is a no-op.
        Otherwise the last stream's verified rate. Poll-cheap (UI tick): no locks, no I/O."""
        from localharness.provider.speed_stats import decode_tps

        p = self._stream_progress
        if p is not None and p.get("first_at") is not None and self._tokens_per_chunk:
            est_tokens = (p.get("chunks", 0) or 0) * self._tokens_per_chunk
            live = decode_tps(est_tokens, p["first_at"], time.monotonic())
            if live is not None:
                return (live, False)
        if self.last_gen_tps:
            return (self.last_gen_tps, True)
        return None

    @staticmethod
    def _unmap_tool_call_names(message: Any, unmap: dict[str, str] | None) -> None:
        """Restore registry tool names (sanitized -> original) on a parsed response, in place.

        Handles both response shapes: streaming (plain dicts from _consume_native_stream)
        and non-streaming (SDK pydantic objects). Names not in the map — including a model
        echoing the original registry name from history — pass through unchanged.
        """
        if not unmap:
            return
        for tc in getattr(message, "tool_calls", None) or []:
            try:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    fn["name"] = unmap.get(name, name)
                else:
                    name = tc.function.name
                    tc.function.name = unmap.get(name, name)
            except Exception:
                continue

    @staticmethod
    async def _consume_native_stream(
        response: Any,
        on_token: Callable[[str], Awaitable[None]] | None,
        progress: dict | None = None,
    ) -> tuple[Any, Any]:
        """Assemble a chat-completions chunk stream into (message, usage).

        Tool calls are accumulated as plain dicts (index-keyed deltas: id/name arrive
        on the first fragment, arguments accrete across fragments) — dicts re-serialize
        cleanly when the assistant message is replayed in later request history. Usage
        arrives on the final chunk when stream_options.include_usage is set; None if
        the provider omits it (loop falls back to tiktoken estimation).

        `progress` (speed_stats): mutated in place with first payload-delta time and a
        payload-chunk count — content, tool-call AND reasoning deltas all count, so a
        native tool-calling or thinking stream measures like a prose one — plus the
        engine-reported decode rate when the runtime states one (llama.cpp `timings`).
        A reasoning delta counts on its PRESENCE, not its truthiness (#142): thinking that
        has produced no text in this delta is still generation under way, and the window
        must open there or the reported rate divides a full token count by an answer-only
        window (observed shape: Ollama streams `reasoning` plus an empty `content`).
        """
        from types import SimpleNamespace

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage = None
        finish_reason = None
        async for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if progress is not None:
                timings = getattr(chunk, "timings", None)  # llama.cpp extension field
                pps = timings.get("predicted_per_second") if isinstance(timings, dict) else None
                if pps:
                    progress["server_tps"] = pps
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
            reasoning = _reasoning_text(delta)
            if progress is not None and (
                getattr(delta, "content", None)
                or getattr(delta, "tool_calls", None)
                or reasoning is not None
            ):
                progress["chunks"] += 1
                if progress["first_at"] is None:
                    progress["first_at"] = time.monotonic()
            if reasoning:
                reasoning_parts.append(reasoning)
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
            reasoning_content="".join(reasoning_parts) or None,
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
        gen_timeout: float | None = None,
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
        name_unmap: dict[str, str] | None = None
        if tools and not self._tools_param_rejected:
            kwargs["tools"], name_unmap = _tools_to_api_format(tools)
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
                return await self._consume_bounded(
                    self._create_and_consume(kwargs, stream, name_unmap=name_unmap), gen_timeout,
                )
        except openai.BadRequestError:
            # Server rejected the request (e.g. tools param) — retry without it. injected_messages
            # already carries the system-prompt injection folded in above, so the fallback's own
            # injection is a no-op (marker-guarded in _fold_tool_injection — never double-injects).
            # Remember the rejection: re-sending `tools=` next iteration just buys another 400.
            self._tools_param_rejected = True
            log.warning("Server rejected request, falling back to system prompt injection")
            return await self._complete_xml_fallback(
                injected_messages, tools, stream,
                disable_thinking=disable_thinking, gen_timeout=gen_timeout,
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def _complete_xml_fallback(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        stream: bool,
        disable_thinking: bool = False,
        gen_timeout: float | None = None,
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
            async with _inference_gate(self.config):  # #18: honor stream here too
                return await self._consume_bounded(self._create_and_consume(kwargs, stream), gen_timeout)
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
