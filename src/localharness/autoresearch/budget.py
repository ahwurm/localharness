"""Autoresearch budget controller (AUTO-03): self-metering token window + wallclock pre-flight gate.

There is NO usable Anthropic quota endpoint — the OAuth usage endpoint 429-storms and
returns only rolling-window percentages (no absolute tokens, no clean reset), so a
per-iteration poll would throttle itself and never read headroom (18-RESEARCH §#1). The
deterministic, no-placeholder design is to meter the proposer client's OWN reported usage:
LocalHarness's proposer is its own OpenAI-compatible LLMClient, and `complete()` already
returns `(message, CompletionUsage)` — we capture `total_tokens` (proposer.py) and feed it
to `WindowMeter.record_tokens`.

Two pieces:
  - WindowMeter — self-meters proposer tokens against a user-set 5h rolling window,
    persisted to window.json so a crash-restart in the same window never double-spends.
  - BudgetController — a PRE-FLIGHT gate run BEFORE each iteration. Halts (returns False)
    on ANY of: max_iterations reached, wallclock budget elapsed, or the token window
    exhausted. It NEVER cancels a running experiment (the per-proposal timeout in 18-05 is
    the only mid-experiment kill). Three independent backstops (--max-iterations,
    --budget wallclock, the circuit breaker) mean a metering bug cannot loop forever.

Local harness inference (the bench arms run by run_experiment) is FREE and explicitly NOT
metered here — only the proposer's CompletionUsage.total_tokens flows through record_tokens.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class WindowMeter:
    """Self-meter proposer tokens against a user-set 5h rolling-window budget (AUTO-03).

    The 5h window is a FIXED reset keyed to the first spend (mirrors Anthropic's
    'counter starts on your first prompt' semantics). Persists window.json so a
    crash-restart in the same window does NOT double-spend. Local harness inference
    is NEVER metered here — only the proposer's CompletionUsage.total_tokens.
    """

    WINDOW_SECONDS = 5 * 3600

    def __init__(self, *, window_budget_tokens: int | None, state_path: Path, clock=time.time):
        self._budget = window_budget_tokens  # --claude-window-tokens; None = no token cap
        self._path = Path(state_path)
        self._clock = clock  # wall-epoch clock (FAKE in tests)
        self._load_or_reset()

    def _load_or_reset(self) -> None:
        now = self._clock()
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._start = float(data["window_start"])
                self._spent = int(data["tokens_spent"])
            except Exception:
                self._reset(now)
                return
            if now - self._start >= self.WINDOW_SECONDS:  # stale window -> reset
                self._reset(now)
        else:
            self._reset(now)

    def _reset(self, now: float | None = None) -> None:
        self._start = self._clock() if now is None else now
        self._spent = 0
        self._persist()

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"window_start": self._start, "tokens_spent": self._spent}),
            encoding="utf-8",
        )

    def record_tokens(self, n: int) -> None:
        """Add a proposer call's total_tokens to the window (the SOLE mutator of tokens_spent)."""
        if self._clock() - self._start >= self.WINDOW_SECONDS:
            self._reset()
        self._spent += int(n)
        self._persist()

    def window_exhausted(self) -> bool:
        """Pure read (resets a stale window first); True once spend meets the configured cap."""
        if self._clock() - self._start >= self.WINDOW_SECONDS:
            self._reset()
        return self._budget is not None and self._spent >= self._budget

    def snapshot(self) -> dict:
        return {
            "tokens_spent": self._spent,
            "window_budget": self._budget,
            "window_start": self._start,
            "window_remaining": (
                None if self._budget is None else max(0, self._budget - self._spent)
            ),
        }


class BudgetController:
    """Pre-flight gate BEFORE each iteration. NEVER interrupts a running experiment.

    Halts (can_start_iteration -> False) when ANY of: max_iterations reached,
    wallclock budget elapsed, or the 5h token window exhausted. The default run is
    'until the 5h window is ~spent' + a HIGH max_iterations backstop so a metering
    bug can't loop forever (CONTEXT). No hard error on missing caps.
    """

    def __init__(
        self,
        *,
        budget_seconds: float | None,
        max_iterations: int,
        max_cost: float | None,
        meter,
        clock=time.monotonic,
    ):
        self._budget = budget_seconds  # None = no wallclock cap
        self._max_iter = max_iterations  # ALWAYS a high backstop default at the CLI
        self._max_cost = max_cost  # archive per-row $ sum; ~0 for local proposer (recorded, not enforced here)
        self._meter = meter  # WindowMeter (or FakeWindowMeter in tests)
        self._clock = clock  # monotonic clock (FAKE in tests)
        self._start = clock()
        self._iters = 0

    def can_start_iteration(self) -> bool:
        if self._iters >= self._max_iter:
            return False
        if self._budget is not None and (self._clock() - self._start) >= self._budget:
            return False
        if self._meter.window_exhausted():
            return False
        self._iters += 1
        return True

    def snapshot(self) -> dict:
        return {
            "iterations": self._iters,
            "max_iterations": self._max_iter,
            "wallclock_elapsed": self._clock() - self._start,
            "budget_seconds": self._budget,
            **self._meter.snapshot(),
        }
