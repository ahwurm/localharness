"""ContextManager: build_messages with repair_tool_pairing boundary guard."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS
from localharness.core.types import Message

log = logging.getLogger("localharness.agent.context")

RESPONSE_RESERVE_TOKENS: int = 4096

# Stale-web-result eviction (OpenHands BrowserOutputCondenser pattern): cheap first
# line of defense, applied well before LLM summary-compaction (0.80) kicks in.
WEB_EVICT_USAGE_FRACTION: float = 0.50
WEB_EVICT_KEEP_LAST: int = 2
_WEB_EVICT_MIN_CHARS: int = 500          # stubbing tiny results saves nothing
_WEB_TOOLS = frozenset({"web_fetch", "web_search"})
_WEB_STUB_PREFIX = "[web output omitted"

# Large-tool-result eviction (generalizes web eviction to ALL bulky tool outputs).
# A tool result whose char size exceeds the threshold has its body replaced with a
# restorable stub; the full body is kept in the ContentStore keyed by a DETERMINISTIC
# content hash (NOT random/timestamp) so the rendered prompt stays prefix-cache stable.
# The model re-pulls the body with tool_result_get('<id>').
TOOL_EVICT_USAGE_FRACTION: float = 0.50
TOOL_EVICT_KEEP_LAST: int = 3            # leave the most recent K results un-evicted
TOOL_EVICT_THRESHOLD_CHARS: int = 8_000  # bodies under this aren't worth stubbing
_TOOL_STUB_PREFIX = "[tool result evicted"
# MOVE 0c (coordinator ruling 2026-07-06, REPLACING the earlier must-shrink latch): bound the
# summary-compaction storm with hysteresis + a hard floor — never by switching compaction off.
# The latch inherited the root problem: shrink-per-fire near the trigger is often tiny (SEMA-05
# sawtooth 82→70→80→81→73), so one "non-shrinking" fire disabled compaction mid-long-turn and a
# live run overflowed to 101.1% utilization (designed-20260706T143144Z, killed). Now:
# - COMPACTION_TARGET_USAGE_FRACTION: a fire must land utilization AT TARGET, well below the
#   0.80 trigger — the stage WIDENS the summarized span until the target is met or no safe cut
#   remains (hysteresis: trigger high, land low).
# - MAX_COMPACTION_FIRES_PER_TURN: backstop cap on expensive summarizer runs per turn.
# - Emergency floor (build_messages): overflow is IMPOSSIBLE — at the cap / with no safe cut,
#   utilization above 100% hard-truncates the oldest non-system messages, loudly logged.
# Deep re-trigger diagnosis (why utilization re-approaches the trigger) stays a separate open item.
MAX_COMPACTION_FIRES_PER_TURN: int = 3
COMPACTION_TARGET_USAGE_FRACTION: float = 0.60


Origin = Literal["trusted", "untrusted"]
_STORE_WEB_MAX: int = 32  # web (re-fetchable) bodies are LRU-bounded in the unified store


def _content_handle(content: str) -> str:
    """Deterministic content-addressable handle for a body: content hash, no randomness/time.
    Same body -> same handle across turns AND across agents, so a stub stays prefix-cache stable
    and identical bodies dedupe to one entry."""
    return hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()[:12]


class ContentStore:
    """One per-agent content-addressable store: handle -> (body, origin).

    Generalizes the prior evicted-body store (durable, non-web restorable tool bodies) and absorbs
    web page store (LRU-bounded, re-fetchable web bodies) onto a single substrate. Handles are a
    deterministic content hash, so identical bodies dedupe and a given body always restores under
    the same handle.

    - origin: a body is 'trusted' unless it came from the web, was read back from memory, or was
      derived from an untrusted handle. Taint is STICKY and monotonic — once untrusted, a handle
      (and anything derived from it) never relaunders. Only a clean-origin handle may ever be bound
      into an exec namespace (the injection floor).
    - pg-N aliases: per-agent, so each agent's OWN first web fetch is deterministically pg-1 (the
      blind-verifier gate depends on this). No module-global counter.
    - web LRU: web (re-fetchable) bodies are bounded; if one ages out, web_page_query just asks to
      re-fetch (the 're-fetch-stub lever'). Trusted bodies are durable — a restore must not fail.
    - grant view: a child built with (parent, granted) may read ONLY the granted parent handles —
      the per-delegation capability, no global registry. Leaves get parent=None / granted=∅.
    """

    def __init__(
        self,
        max_web: int = _STORE_WEB_MAX,
        parent: "ContentStore | None" = None,
        granted: "frozenset[str] | None" = None,
    ) -> None:
        self._bodies: dict[str, tuple[str, Origin]] = {}
        self._web: "OrderedDict[str, None]" = OrderedDict()
        self._aliases: dict[str, str] = {}
        self._fetch_seq = 0
        self._max_web = max_web
        self._parent = parent
        self._granted = frozenset(granted or ())

    def put(self, body: str, origin: Origin = "trusted", derived_from: str | None = None) -> str:
        """Store a body, return its handle. Origin is sticky: untrusted if explicitly untrusted,
        derived from an untrusted handle, or this exact body was already stored untrusted."""
        h = _content_handle(body)
        prev = self._bodies.get(h)
        tainted = (
            origin == "untrusted"
            or (derived_from is not None and self.origin(derived_from) == "untrusted")
            or (prev is not None and prev[1] == "untrusted")
        )
        self._bodies[h] = (body, "untrusted" if tainted else "trusted")
        return h

    def put_web(self, body: str) -> str:
        """Retain a web body (ALWAYS untrusted), mint a per-agent pg-N alias, LRU-bound the web set.
        Returns the pg-N alias — callers keep using pg-N exactly as before."""
        self._fetch_seq += 1
        alias = f"pg-{self._fetch_seq}"
        h = self.put(body, origin="untrusted")
        self._aliases[alias] = h
        self._web[h] = None
        self._web.move_to_end(h)
        while len(self._web) > self._max_web:
            old, _ = self._web.popitem(last=False)
            self._bodies.pop(old, None)
            for a in [a for a, hh in self._aliases.items() if hh == old]:
                self._aliases.pop(a, None)
        return alias

    def get(self, ref: str) -> str | None:
        """Resolve a handle OR a pg-N alias to its body. Reads through to a GRANTED parent handle
        only (the capability); never an ambient cross-agent read."""
        h = self._aliases.get(ref, ref)
        v = self._bodies.get(h)
        if v is not None:
            if h in self._web:
                self._web.move_to_end(h)  # LRU touch
            return v[0]
        if self._parent is not None and h in self._granted:
            return self._parent.get(h)
        return None

    def origin(self, ref: str) -> Origin | None:
        h = self._aliases.get(ref, ref)
        v = self._bodies.get(h)
        if v is not None:
            return v[1]
        if self._parent is not None and h in self._granted:
            return self._parent.origin(h)
        return None

    def stub_meta(self, ref: str) -> tuple[int, Origin] | None:
        """(byte size, origin) for a handle/alias — used to size the inline stub for the trigger."""
        h = self._aliases.get(ref, ref)
        v = self._bodies.get(h)
        if v is None and self._parent is not None and h in self._granted:
            v = self._parent._bodies.get(h)
        return (len(v[0]), v[1]) if v else None

    def reset(self) -> None:
        """Test/seam hook: clear all bodies, the web LRU, and the pg-N alias sequence."""
        self._bodies.clear()
        self._web.clear()
        self._aliases.clear()
        self._fetch_seq = 0


def _evict_large_tool_results(
    messages: list[Message],
    store: "ContentStore",
    threshold_chars: int = TOOL_EVICT_THRESHOLD_CHARS,
    keep_last: int = TOOL_EVICT_KEEP_LAST,
) -> tuple[list[Message], int]:
    """Replace the bodies of bulky NON-web tool results with a restorable stub keyed by a
    deterministic content hash; the full body is stashed in `store` for tool_result_get.
    Web results are handled by _evict_stale_web_results (URL-restorable, no store needed).
    The newest `keep_last` evictable results are left verbatim for immediate reasoning.
    Returns (new list, evicted count); input messages are never mutated. Deterministic:
    same input -> same stubs (same id), so the prompt stays prefix-cache stable."""
    # tool_call_ids that resolve to web tools — those go through the web path, skip here.
    web_ids: set[str] = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            name = (fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")) or ""
            if name in _WEB_TOOLS:
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                web_ids.add(tc_id)
    evictable = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
        and m.get("tool_call_id") not in web_ids
        and len(m.get("content") or "") > threshold_chars
        and not (m.get("content") or "").startswith(_TOOL_STUB_PREFIX)
    ]
    stale = evictable[:-keep_last] if keep_last > 0 else evictable
    if not stale:
        return messages, 0
    out = list(messages)
    for i in stale:
        m = out[i]
        body = m.get("content") or ""
        rid = store.put(body)
        approx_tokens = len(body) // 4
        out[i] = {**m, "content": (
            f"{_TOOL_STUB_PREFIX} — ~{approx_tokens} tokens — "
            f"call tool_result_get('{rid}') to restore the full body]"
        )}
    return out, len(stale)


def _evict_stale_web_results(
    messages: list[Message], keep_last: int = WEB_EVICT_KEEP_LAST,
) -> tuple[list[Message], int]:
    """Replace the bodies of all but the newest `keep_last` web tool results with a
    restorable stub (URL/query preserved — the agent can re-fetch). Web pages are the
    bulkiest, least re-read observations; dropping their bodies is lossless in practice
    (Manus rule). Returns (new list, evicted count); input messages are never mutated.
    Deterministic: same input -> same stubs, so the rendered prompt stays prefix-cache
    stable between turns that add no new web results."""
    id_meta: dict[str, tuple[str, str]] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            name = (fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")) or ""
            if name not in _WEB_TOOLS:
                continue
            raw = (fn.get("arguments") if isinstance(fn, dict)
                   else getattr(fn, "arguments", None)) or "{}"
            try:
                args = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                args = {}
            hint = args.get("url") or args.get("query") or ""
            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            id_meta[tc_id] = (name, hint)
    web_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool" and m.get("tool_call_id") in id_meta
        and len(m.get("content") or "") >= _WEB_EVICT_MIN_CHARS
        and not (m.get("content") or "").startswith(_WEB_STUB_PREFIX)
    ]
    stale = web_idxs[:-keep_last] if keep_last > 0 else web_idxs
    if not stale:
        return messages, 0
    out = list(messages)
    for i in stale:
        m = out[i]
        name, hint = id_meta[m["tool_call_id"]]
        target = f" {hint}" if hint else ""
        out[i] = {**m, "content": (
            f"{_WEB_STUB_PREFIX} — {name}{target}; {len(m.get('content') or '')} chars "
            f"dropped to free context; call {name} again to re-read]"
        )}
    return out, len(stale)


class TokenCounter:
    """Token counting. Prefers the SERVED model's exact tokenizer via vLLM's
    POST {server_root}/tokenize (model-truth) when base_url+model are given; falls
    back to tiktoken cl100k, then a char heuristic. Counts are content-hash cached.

    cl100k undercounts Qwen by ~1.85x on digit/code text, so the remote path is the
    only accurate source for non-cl100k models. The remote call is sync (urllib) — the
    agent loop is serial and model-bottlenecked, so a ~9ms round-trip is acceptable —
    and is probed once at construction; on any failure it self-disables and never retries.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            pass

        # /tokenize lives at the SERVER ROOT, not under /v1.
        self._tokenize_url: str | None = None
        self._model = model
        self._cache: dict[str, int] = {}
        if base_url and model:
            root = base_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[: -len("/v1")]
            self._tokenize_url = f"{root}/tokenize"
            # Server-or-fail: NO silent fallback. An exact count is mandatory on the live
            # path, so a probe failure is a HARD error — running on an approximate meter is
            # exactly what hid the context-overflow bug. The start path surfaces this.
            if self._remote_count("token") is None:
                raise RuntimeError(
                    f"TokenCounter: exact token counting unavailable — /tokenize unreachable "
                    f"at {self._tokenize_url}. Refusing to fall back to an approximate tokenizer."
                )

    def _remote_count(self, text: str) -> int | None:
        """Exact server-side count via vLLM /tokenize, or None on ANY failure."""
        if not self._tokenize_url:
            return None
        import json as _json
        import urllib.request
        try:
            body = _json.dumps({"model": self._model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(
                self._tokenize_url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            count = data.get("count")
            return int(count) if count is not None else None
        except Exception:
            return None

    def count(self, text: str) -> int:
        if not text:
            return 0
        key = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._tokenize_url is not None:
            # Live mode: the server tokenizer is the only source of truth. Fail loud on a miss.
            n = self._remote_count(text)
            if n is None:
                raise RuntimeError(
                    f"TokenCounter: /tokenize call failed mid-session at {self._tokenize_url}; "
                    f"refusing to substitute an approximate count."
                )
        elif self._encoder is not None:
            # Non-live estimator (unit tests / bench, no server) — explicit, not a fallback.
            # disallowed_special=() so literal special-token text is counted as ordinary text.
            n = len(self._encoder.encode(text, disallowed_special=()))
        else:
            raise RuntimeError("TokenCounter: no tokenizer available (tiktoken missing, no server).")
        if len(self._cache) < 50_000:
            self._cache[key] = n
        return n

    def count_messages(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            total += 4  # message overhead tokens
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += self.count(content)
            # tool_calls in assistant messages
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                total += self.count(fn.get("name", ""))
                total += self.count(fn.get("arguments", ""))
        return total


@dataclass
class TokenBudget:
    total_limit: int
    current_usage: int
    tool_schema_tokens: int
    headroom: int = 0

    def __post_init__(self) -> None:
        self.headroom = self.total_limit - self.current_usage - self.tool_schema_tokens - RESPONSE_RESERVE_TOKENS

    @property
    def usage_fraction(self) -> float:
        return (self.current_usage + self.tool_schema_tokens) / self.total_limit if self.total_limit > 0 else 1.0

    @property
    def needs_summary_compact(self) -> bool:
        return self.usage_fraction >= 0.80

    @property
    def needs_full_compact(self) -> bool:
        return self.usage_fraction >= 0.95


def _collect_valid_tool_ids(messages: list[Message]) -> set[str]:
    """Return the set of all tool_call IDs referenced in assistant tool_calls."""
    valid: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    valid.add(tc_id)
    return valid


def _repair_tool_pairing(messages: list[Message]) -> list[Message]:
    """Remove orphaned tool role messages. Returns new list; input unchanged."""
    valid_ids = _collect_valid_tool_ids(messages)
    result = []
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id", "") not in valid_ids:
            continue
        result.append(m)
    return result


# ANSI escape sequences (CSI + OSC) — terminal colour/cursor noise with no signal for the
# model; stripping them is free bytes under the same cap.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _clean_tool_output(text: str) -> str:
    """Deterministic, signal-preserving pre-clean for tool output: strip ANSI escapes,
    drop trailing per-line whitespace, collapse 3+ blank lines. Cheap (no model call), run
    before length measurement so the cap reflects real content."""
    text = _ANSI_RE.sub("", text)
    text = re.sub(r"[ \t]+(?=\n)", "", text)   # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)     # blank-line runs
    return text


def _head_tail(text: str, max_chars: int) -> str:
    """Keep the head AND tail of oversized output. Head-only truncation throws away the tail
    — exactly where exit codes, errors and test summaries live. 60% head / 40% tail."""
    head_chars = (max_chars * 6) // 10
    tail_chars = max_chars - head_chars
    elided = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f"\n... [{elided} chars elided — head+tail kept] ...\n"
        + text[-tail_chars:]
    )


class ToolResultCapStage:
    """Stage 1: Pre-clean every tool result (ANSI/whitespace) and cap oversized ones with a
    head+tail keep (lossy first defense; the restorable page-out is the eviction stage)."""

    def __init__(self, max_chars: int = 50_000) -> None:
        self.max_chars = max_chars

    def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        result = []
        modified = False
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content") or ""
                cleaned = _clean_tool_output(content)
                if len(cleaned) > self.max_chars:
                    cleaned = _head_tail(cleaned, self.max_chars)
                if cleaned != content:
                    result.append({**m, "content": cleaned})
                    modified = True
                    continue
            result.append(m)
        return result, modified


class BoundaryGuardStage:
    """Stage 2: Repair orphaned tool_use/tool_result pairs."""

    def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        repaired = _repair_tool_pairing(messages)
        return repaired, repaired != messages


class SummaryCompactionStage:
    """Stage 3: Summarize middle messages at >= 80% context utilization, COMPACTING TO TARGET.

    MOVE 0c (coordinator ruling): triggering at 0.80 and landing "slightly lower" produced the
    SEMA-05 sawtooth (82→70→80→81→73 — re-trigger nearly every iteration). A fire must instead
    land utilization at `target_usage_fraction` (default 0.60), WIDENING the summarized span
    (halving the preserved head/tail, floors 1/2) and re-summarizing until the target is met or
    no safe cut remains. Hysteresis: trigger high, land low — real headroom per fire."""

    def __init__(
        self,
        preserve_first_n: int,
        preserve_last_n: int,
        llm_summarize_fn: Any = None,
        compact_md_path: Path | None = None,
        target_usage_fraction: float = COMPACTION_TARGET_USAGE_FRACTION,
    ) -> None:
        self.preserve_first_n = preserve_first_n
        self.preserve_last_n = preserve_last_n
        self.llm_summarize_fn = llm_summarize_fn
        self.compact_md_path = compact_md_path
        self.target_usage_fraction = target_usage_fraction

    async def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        if budget.usage_fraction < 0.80:
            return messages, False
        if self.llm_summarize_fn is None:
            return messages, False

        # Mirror usage_fraction's math: message tokens must fit target*limit minus tool schemas.
        target_tokens = self.target_usage_fraction * budget.total_limit - budget.tool_schema_tokens
        working = messages
        modified = False
        first_n, last_n = self.preserve_first_n, self.preserve_last_n
        while True:
            first_boundary = self._safe_cut_boundary(working, first_n, "forward")
            last_boundary = self._safe_cut_boundary(working, len(working) - last_n, "backward")
            middle = working[first_boundary:last_boundary] if last_boundary > first_boundary else []
            if len(middle) > 2:
                try:
                    summary_text = await self.llm_summarize_fn(middle)
                except Exception as exc:
                    log.warning("Summarization failed: %s. Stopping compaction at current state.", exc)
                    return working, modified
                summary_message = {"role": "assistant", "content": f"[Context Summary]\n{summary_text}"}
                working = working[:first_boundary] + [summary_message] + working[last_boundary:]
                modified = True
                if self.compact_md_path is not None:
                    _write_compact_md(self.compact_md_path, summary_text)
                log.info(
                    "Summary compaction: %d messages → 1 summary (preserve %d/%d)",
                    len(middle), first_n, last_n,
                )
                if token_counter.count_messages(working) <= target_tokens:
                    return working, modified  # landed at target — the point of the fire
            # Above target (or nothing summarizable at this width): WIDEN the span and retry.
            next_f, next_l = max(1, first_n // 2), max(2, last_n // 2)
            if next_f >= first_n and next_l >= last_n:
                return working, modified  # can't widen further / no safe cut remains
            first_n, last_n = min(first_n, next_f), min(last_n, next_l)

    def _safe_cut_boundary(self, messages: list[Message], start_idx: int, direction: str) -> int:
        """Find a safe cut boundary that does not split tool_use/tool_result pairs."""
        n = len(messages)
        start_idx = max(0, min(start_idx, n))

        if direction == "forward":
            i = start_idx
            while i < n:
                if _is_safe_cut_after(messages, i - 1):
                    return i
                i += 1
            return start_idx
        else:  # backward
            i = start_idx
            while i > 0:
                if _is_safe_cut_after(messages, i - 1):
                    return i
                i -= 1
            return start_idx


class FullAutoCompactStage:
    """Stage 4: Emergency full-session compaction at >= 95% utilization."""

    def __init__(
        self,
        llm_summarize_fn: Any = None,
        compact_md_path: Path | None = None,
        target_usage_fraction: float = COMPACTION_TARGET_USAGE_FRACTION,
    ) -> None:
        self.llm_summarize_fn = llm_summarize_fn
        self.compact_md_path = compact_md_path
        # Use aggressive boundaries for emergency compaction
        self._summary_stage = SummaryCompactionStage(
            preserve_first_n=1,
            preserve_last_n=2,
            llm_summarize_fn=llm_summarize_fn,
            compact_md_path=compact_md_path,
            target_usage_fraction=target_usage_fraction,
        )

    async def apply(
        self,
        messages: list[Message],
        budget: TokenBudget,
        token_counter: TokenCounter,
    ) -> tuple[list[Message], bool]:
        if budget.usage_fraction < 0.95:
            return messages, False
        # Reuse SummaryCompactionStage with aggressive settings, forcing it to fire
        # Create a fake budget at 80% to trigger it
        forced_budget = TokenBudget(
            total_limit=budget.total_limit,
            current_usage=int(budget.total_limit * 0.81),
            tool_schema_tokens=0,
        )
        return await self._summary_stage.apply(messages, forced_budget, token_counter)


class CompactionPipeline:
    """Runs all 4 compaction stages in order."""

    def __init__(
        self,
        token_counter: TokenCounter,
        tool_result_cap: int = 50_000,
        preserve_first_n: int = 4,
        preserve_last_n: int = 8,
        llm_summarize_fn: Any = None,
        compact_md_path: Path | None = None,
        target_usage_fraction: float = COMPACTION_TARGET_USAGE_FRACTION,
    ) -> None:
        self._token_counter = token_counter
        self._stages: list = [
            ToolResultCapStage(max_chars=tool_result_cap),
            BoundaryGuardStage(),
            SummaryCompactionStage(
                preserve_first_n=preserve_first_n,
                preserve_last_n=preserve_last_n,
                llm_summarize_fn=llm_summarize_fn,
                compact_md_path=compact_md_path,
                target_usage_fraction=target_usage_fraction,
            ),
            FullAutoCompactStage(
                llm_summarize_fn=llm_summarize_fn,
                compact_md_path=compact_md_path,
                target_usage_fraction=target_usage_fraction,
            ),
        ]

    async def run(
        self,
        messages: list[Message],
        budget: TokenBudget,
    ) -> tuple[list[Message], bool]:
        import inspect
        any_modified = False
        working = messages
        for stage in self._stages:
            call = stage.apply(working, budget, self._token_counter)
            if inspect.iscoroutine(call):
                result, modified = await call
            else:
                result, modified = call
            if modified:
                working = result
                any_modified = True
                # Recompute budget after modification
                new_usage = self._token_counter.count_messages(working)
                budget = TokenBudget(
                    total_limit=budget.total_limit,
                    current_usage=new_usage,
                    tool_schema_tokens=budget.tool_schema_tokens,
                )
        return working, any_modified


def _is_safe_cut_after(messages: list[Message], idx: int) -> bool:
    """Return True if it's safe to cut the message list after index idx."""
    if idx < 0:
        return True  # beginning of list is safe
    msg = messages[idx]
    role = msg.get("role")
    # Safe after a user message
    if role == "user":
        return True
    # Safe after a tool result (last one in a batch means all results collected)
    if role == "tool":
        # Check if the next message (if any) is also a tool result
        next_idx = idx + 1
        if next_idx < len(messages) and messages[next_idx].get("role") == "tool":
            return False  # more tool results follow — not a safe boundary
        return True
    # Safe after an assistant message with no tool_calls
    if role == "assistant" and not msg.get("tool_calls"):
        return True
    return False


def _hard_truncate_to_budget(
    messages: list[Message], max_msg_tokens: int, token_counter: TokenCounter,
) -> tuple[list[Message], int]:
    """MOVE 0c emergency floor (coordinator ruling: overflow must be IMPOSSIBLE). Drop the
    OLDEST non-system messages — whole safe-cut chunks at a time so tool pairs never split —
    until the message tokens fit `max_msg_tokens`. The leading system message and the final
    message are never dropped; pairing is repaired at the end as a belt. Returns
    (truncated_messages, n_dropped). Lossy by design — this is the fallback that runs only when
    compaction (capped or cut-less) could not get back under budget."""
    working = list(messages)
    start = 1 if working and working[0].get("role") == "system" else 0
    dropped = 0
    while token_counter.count_messages(working) > max_msg_tokens and len(working) - start > 1:
        cut = next(
            (i for i in range(start, len(working) - 1) if _is_safe_cut_after(working, i)),
            start,  # no safe boundary at all: drop the single oldest non-system message
        )
        dropped += cut + 1 - start
        del working[start:cut + 1]
    return _repair_tool_pairing(working), dropped


def _write_compact_md(path: Path, content: str) -> None:
    """Atomically write content to compact.md."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    os.replace(str(tmp), str(path))


def load_compact_md(compact_md_path: Path) -> Message | None:
    """Load compact.md from disk and return as a system message, or None if not found."""
    if compact_md_path.exists():
        content = compact_md_path.read_text().strip()
        if content:
            return {"role": "system", "content": f"[Prior Session Context]\n{content}"}
    return None


def make_compaction_summarize_fn(llm: Any) -> Any:
    """Build the `llm_summarize_fn` for SummaryCompactionStage: render the middle messages and ask
    the model for a dense summary.

    The (message, usage) unpack is LOAD-BEARING and shared by start_cmd + the bench runner so it
    lives in ONE tested place: `complete()` returns a (message, usage) TUPLE, and a bare
    `result.content` on that tuple silently raised → SummaryCompactionStage.apply swallowed it →
    summary-compaction was DEAD in prod AND bench (caught only by a live dogfood, no regression
    test). test_compaction_summarize_unpacks_tuple pins it OUTSIDE the swallow."""
    async def summarize(messages: list) -> str:
        prompt = [
            {"role": "system", "content": (
                "Summarize the following conversation history concisely. Preserve key facts, "
                "decisions, and tool results. Output a dense summary paragraph."
            )},
            {"role": "user", "content": "\n".join(
                f"[{m.get('role', '?')}]: {(m.get('content') or '')[:500]}" for m in messages
            )},
        ]
        result = await llm.complete(prompt, tools=None)
        msg = result[0] if isinstance(result, tuple) else result  # (message, usage) tuple — DO NOT regress
        return (getattr(msg, "content", "") or "")
    return summarize


class ContextManager:
    """Manages message list preparation for LLM requests.

    repair_tool_pairing is called on every build_messages call to ensure
    no orphaned tool_result messages are sent to the LLM (which causes HTTP 400).
    """

    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        preserve_first_n: int = 4,
        preserve_last_n: int = 8,
        pipeline: CompactionPipeline | None = None,
        bus: Any = None,
        agent_id: str = "",
        session_id: str = "",
        eviction_store: "ContentStore | None" = None,
        content_store: "ContentStore | None" = None,
        tool_evict_threshold_chars: int = TOOL_EVICT_THRESHOLD_CHARS,
        tool_evict_enabled: bool = True,
        token_counter: "TokenCounter | None" = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.preserve_first_n = preserve_first_n
        self.preserve_last_n = preserve_last_n
        self._pipeline = pipeline
        self._eviction_store = eviction_store
        # The unified per-agent content store: web pages, evicted bodies, granted handles. ALWAYS
        # present (the web/verb tools bind to it). `_eviction_store` stays the explicit-only signal
        # that gates large-tool-result eviction — the root has a tool_result_get to restore; a child
        # does not, so a child must not stub bodies it could never re-pull.
        self._content_store = content_store if content_store is not None else (
            eviction_store if eviction_store is not None else ContentStore()
        )
        self._tool_evict_threshold_chars = tool_evict_threshold_chars
        self._tool_evict_enabled = tool_evict_enabled
        self._token_counter = token_counter or TokenCounter()
        self._bus = bus
        self._agent_id = agent_id
        self._session_id = session_id
        self._iteration = 0
        # MOVE 0c: `_compaction_fires` counts summary compactions in the current turn (backstop
        # cap; reset per turn by reset_compaction_guard). No must-shrink latch exists (coordinator
        # ruling): compaction is never switched off — compact-to-target + the emergency floor in
        # build_messages own the not-shrinking case.
        self._compaction_fires = 0

    def set_iteration(self, iteration: int) -> None:
        """Allow the agent loop to bump iteration so CompactionTriggered events carry it."""
        self._iteration = int(iteration)

    def reset_compaction_guard(self) -> None:
        """MOVE 0c: the agent loop calls this at the start of each turn so the per-turn
        summary-compaction fire cap re-arms (a new turn earns fresh attempts)."""
        self._compaction_fires = 0

    def repair_tool_pairing(self, messages: list[Message]) -> list[Message]:
        """Remove orphaned tool role messages.

        An orphaned tool message is one whose tool_call_id does not appear in
        any preceding assistant message's tool_calls list.
        Returns a new list — input is never modified.
        """
        return _repair_tool_pairing(messages)

    async def build_messages(
        self,
        messages: list[Message],
        tool_schemas: list[dict] | None = None,
    ) -> tuple[list[Message], TokenBudget]:
        """Return repaired messages + budget snapshot for the request that will be sent.

        Budget reflects post-compaction state so callers (heartbeat emitter) see the
        actual size of the next outgoing request. Budget is always returned (never None).
        """
        import json as _json
        copied = list(messages)
        repaired = self.repair_tool_pairing(copied)
        tool_tokens = self._token_counter.count_messages(
            [{"role": "system", "content": _json.dumps([t.model_dump() for t in tool_schemas])}]
        ) if tool_schemas else 0

        # Stale-web eviction: cheap, deterministic, runs BEFORE LLM compaction so bulky
        # page bodies never trigger the expensive path. Threshold-gated (Manus caveat:
        # don't rewrite history — and invalidate the KV cache — every turn).
        evict_check = TokenBudget(
            total_limit=self.max_context_tokens,
            current_usage=self._token_counter.count_messages(repaired),
            tool_schema_tokens=tool_tokens,
        )
        if evict_check.usage_fraction >= WEB_EVICT_USAGE_FRACTION:
            repaired, evicted = _evict_stale_web_results(repaired)
            if evicted:
                log.info(
                    "evicted %d stale web result(s) at %.0f%% context usage",
                    evicted, evict_check.usage_fraction * 100,
                )

        # Generalized large-tool-result eviction: any bulky non-web result body is moved to
        # the ContentStore (deterministic content-hash id) and replaced with a restorable
        # stub the model can re-pull via tool_result_get. Same threshold gate keeps the KV
        # cache stable. Skipped if no store wired or the toggle is off.
        if (
            self._tool_evict_enabled
            and self._eviction_store is not None
            and evict_check.usage_fraction >= TOOL_EVICT_USAGE_FRACTION
        ):
            repaired, t_evicted = _evict_large_tool_results(
                repaired, self._eviction_store,
                threshold_chars=self._tool_evict_threshold_chars,
            )
            if t_evicted:
                log.info(
                    "evicted %d large tool result(s) to restorable stubs at %.0f%% usage",
                    t_evicted, evict_check.usage_fraction * 100,
                )

        if self._pipeline is not None:
            pre_usage = self._token_counter.count_messages(repaired)
            pre_budget = TokenBudget(
                total_limit=self.max_context_tokens,
                current_usage=pre_usage,
                tool_schema_tokens=tool_tokens,
            )
            if pre_budget.needs_summary_compact:
                # MOVE 0c (coordinator ruling): the per-turn fire cap is the ONLY compaction
                # limiter. Compaction is NEVER latched off — a fire that fails to shrink is the
                # stage's compact-to-target problem (it widens the span), and the emergency
                # floor below makes overflow impossible regardless.
                if self._compaction_fires >= MAX_COMPACTION_FIRES_PER_TURN:
                    log.debug(
                        "compaction guard: fire cap reached this turn (%d) — relying on the "
                        "emergency floor for overflow",
                        self._compaction_fires,
                    )
                else:
                    pre_frac = pre_budget.usage_fraction
                    repaired, any_modified = await self._pipeline.run(repaired, pre_budget)
                    if any_modified:
                        self._compaction_fires += 1
                        if self._bus is not None:
                            post_budget = TokenBudget(
                                total_limit=self.max_context_tokens,
                                current_usage=self._token_counter.count_messages(repaired),
                                tool_schema_tokens=tool_tokens,
                            )
                            from localharness.core.events import CompactionTriggered
                            await self._bus.publish(CompactionTriggered(
                                agent_id=self._agent_id,
                                session_id=self._session_id,
                                iteration=self._iteration,
                                pre_usage_fraction=pre_frac,
                                post_usage_fraction=post_budget.usage_fraction,
                                stages_modified=[],
                            ))

        # MOVE 0c EMERGENCY FLOOR (coordinator ruling: overflow must be IMPOSSIBLE). If — fire
        # cap reached, no safe cut, no pipeline at all — utilization still exceeds 100% of the
        # budget, hard-truncate the oldest non-system messages at safe-cut boundaries until the
        # request fits. Loud by design: this only runs when compaction could not save the turn
        # (the live stall: 101.1% with compaction latched off — that latch no longer exists).
        floor_usage = self._token_counter.count_messages(repaired)
        if self.max_context_tokens > 0 and floor_usage + tool_tokens > self.max_context_tokens:
            over_frac = (floor_usage + tool_tokens) / self.max_context_tokens
            repaired, n_dropped = _hard_truncate_to_budget(
                repaired, self.max_context_tokens - tool_tokens, self._token_counter,
            )
            log.error(
                "EMERGENCY context floor: utilization %.1f%% exceeded 100%% of budget after "
                "compaction — hard-truncated %d oldest message(s) to fit "
                "(agent=%s session=%s iter=%d)",
                over_frac * 100, n_dropped, self._agent_id, self._session_id, self._iteration,
            )

        # Recompute budget AFTER any compaction so the return reflects what will ship
        post_usage = self._token_counter.count_messages(repaired)
        budget = TokenBudget(
            total_limit=self.max_context_tokens,
            current_usage=post_usage,
            tool_schema_tokens=tool_tokens,
        )
        return repaired, budget
