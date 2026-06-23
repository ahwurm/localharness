"""CruncherExecTool: a TRUSTED, restricted, cancellable Python exec for the cruncher capability.

Distinct from PythonExecTool (a general STATEFUL REPL): this is a STATELESS, sandboxed cell run in
a cancellable subprocess (spawn — fresh interpreter, no inherited globals) with:
  - restricted builtins: eval/exec/compile/input/__import__/open removed (the reference keeps
    __import__/open — insufficient); no `import` in-cell, so the safe stdlib is PRE-SEEDED instead;
  - RLIMIT_AS so a runaway allocation can't exhaust host memory;
  - a per-cell wall-clock cap enforced by killing the subprocess (the old asyncio.to_thread exec was
    uncancellable).

The STRUCTURAL injection floor is `bind_clean_origin_bodies`, which REFUSES to expose an untrusted
(web/memory) handle: only CLEAN-ORIGIN bodies are ever bound into the namespace, so injected content
is data the verbs read, never code-bound. (The sandbox is defense-in-depth + a runaway bound, not an
escape-proof jail — CPython can't promise that; the floor is origin-gating, asserted deterministically.)
"""
from __future__ import annotations

import builtins as _builtins
import contextlib
import io
import multiprocessing
import traceback
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema

# Reference _SAFE_BUILTINS nulls eval/exec/compile/input; we ADDITIONALLY drop __import__ + open
# (+ breakpoint). No imports in-cell — the safe stdlib is pre-seeded as names (see _SAFE_MODULES).
_BLOCKED_BUILTINS = frozenset({"eval", "exec", "compile", "input", "__import__", "open", "breakpoint"})

# Pure, non-IO, non-host stdlib pre-bound so joins/aggregation/index work WITHOUT `import`.
_SAFE_MODULES = ("re", "json", "math", "statistics", "itertools", "functools", "collections", "datetime")


class UntrustedHandleError(Exception):
    """Raised when a caller tries to bind an untrusted-origin handle into the trusted exec."""


def bind_clean_origin_bodies(store: Any, handles: list[str]) -> dict[str, str]:
    """Resolve granted handles to a namespace seed {h0,h1,…: body, "handles": {handle: body}},
    REFUSING any untrusted-origin handle (the structural floor — untrusted bytes never reach exec).
    Raises UntrustedHandleError on the first untrusted/unknown-origin handle."""
    seed: dict[str, Any] = {}
    mapping: dict[str, str] = {}
    for i, h in enumerate(handles):
        origin = store.origin(h)
        if origin != "trusted":
            raise UntrustedHandleError(
                f"refusing to bind handle {h!r} of origin {origin!r} into trusted exec — "
                "untrusted (web/memory) content is data the verbs read, never code-bound"
            )
        body = store.get(h)
        if body is None:
            continue
        seed[f"h{i}"] = body
        mapping[h] = body
    seed["handles"] = mapping
    return seed


def _safe_builtins() -> dict[str, Any]:
    return {n: getattr(_builtins, n) for n in dir(_builtins) if n not in _BLOCKED_BUILTINS}


def _child(code: str, seed: dict, mem_bytes: int, conn: Any) -> None:
    """Subprocess entrypoint: cap address space, build a restricted namespace, exec, ship stdout."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass  # RLIMIT unavailable on some platforms; the timeout still bounds the cell
    import importlib
    ns: dict[str, Any] = dict(seed)
    ns["__builtins__"] = _safe_builtins()
    for m in _SAFE_MODULES:
        try:
            ns[m] = importlib.import_module(m)
        except Exception:
            pass
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<cruncher_exec>", "exec"), ns)  # noqa: S102 — restricted ns, clean-origin data
        conn.send((True, buf.getvalue() or "(no output)"))
    except BaseException:  # noqa: BLE001 — a cell error (incl. MemoryError) is returned for self-correction
        conn.send((False, (buf.getvalue() + "\n" + traceback.format_exc()).strip()))
    finally:
        conn.close()


def _recv(conn: Any, proc: Any) -> tuple[bool, str]:
    """Block for the child's (ok, output); EOF means it died without sending (RLIMIT_AS/OOM kill)."""
    try:
        return conn.recv()
    except EOFError:
        return (False, f"[cruncher_exec subprocess exited (code {proc.exitcode}) without output "
                       "— likely hit the RLIMIT_AS memory cap]")


class CruncherExecTool(Tool):
    """Trusted, restricted, cancellable exec over a clean-origin cruncher's granted handle bodies.

    Stateless per call (each cell is a fresh sandboxed subprocess). Seeded ONLY with clean-origin
    bodies by the caller (via bind_clean_origin_bodies) — this tool never sees untrusted content."""

    timeout_s = 300.0  # outer floor; the authoritative cap is cell_timeout_s (enforced internally)

    def __init__(self, seed: dict[str, Any], cell_timeout_s: float = 30.0, mem_limit_mb: int = 512) -> None:
        self._seed = seed
        self._cell_timeout_s = float(cell_timeout_s)
        self._mem_bytes = int(mem_limit_mb) * 1024 * 1024

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="cruncher_exec",
            description=(
                "Run restricted Python over your GRANTED handle bodies (clean-origin only) for "
                "joins/aggregation/index the verbs can't express. Bodies are pre-bound as h0, h1, … "
                "and a `handles` dict; re/json/math/statistics/itertools/functools/collections/datetime "
                "are pre-imported. No `import`, no file/network/eval. print() your result. Each cell is "
                "a fresh sandboxed subprocess, time- and memory-capped."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python to run over the bound handle bodies."},
                },
                "required": ["code"],
            },
            destructive=False,
            estimated_tokens=600,
        )

    async def _execute(self, code: str) -> ToolResult:
        import asyncio

        ctx = multiprocessing.get_context("spawn")  # fresh interpreter; no inherited parent globals
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child, args=(code, self._seed, self._mem_bytes, child_conn))
        proc.start()
        child_conn.close()  # parent drops its write-end copy so recv() EOFs if the child dies
        try:
            ok, out = await asyncio.wait_for(
                asyncio.to_thread(_recv, parent_conn, proc), timeout=self._cell_timeout_s
            )
        except asyncio.TimeoutError:
            # Kill the cell; the abandoned recv-thread unblocks (EOF) once the child is gone. Do not
            # close parent_conn here — the thread may still be reading it; GC closes it after.
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return ToolResult(
                output=f"[cruncher_exec timed out after {self._cell_timeout_s:g}s — cell killed, partial discarded]",
                success=True,
                truncated=True,
            )
        # Normal completion: the child sent its message and exited; reap it and free the pipe.
        if proc.is_alive():
            proc.terminate()
        proc.join(1)
        parent_conn.close()
        # A cell error (traceback) is success=True so the model can read it and self-correct.
        return self.ok(out) if ok else ToolResult(output=out, success=True)
