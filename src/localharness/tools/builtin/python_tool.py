"""PythonExecTool: a STATEFUL Python REPL — the namespace persists across calls.

Unlike bash_exec (a one-shot subprocess), this holds a long-lived namespace dict, so
variables, imports, and function definitions survive between calls within a session.
It is a general stateful REPL: a long input or a handle body can be seeded as a variable the
model inspects with code (regex/slice/decompose/join) instead of receiving it in its prompt.
(Re-pointed as the trusted cruncher's bounded exec in P-CRUNCH.)

No sandbox/isolation — same trust posture as the existing bash_exec (the model already
runs arbitrary shell on the host). Run an isolated VM/container if that matters.
"""
import asyncio
import contextlib
import io
import os
import threading
import traceback
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema

AGENT_COMMAND_MARKER = "LOCALHARNESS_COMMIT"
"""The environment variable every command the harness runs for the agent carries (`bash_exec` sets it
on each child). `localharness model` reads it: with it set, a switch is never a local operator's."""

_marker_lock = threading.Lock()
_marker_depth = 0
_marker_set_by_us = False


@contextlib.contextmanager
def _agent_command_marker():
    """While agent code runs in this process, `os.environ` carries the agent-command marker, so a
    child it spawns (`subprocess.run(["localharness", "model", ...])`) inherits it exactly as a
    `bash_exec` child does (WR-05). Counted, so overlapping calls neither drop it early nor leave it
    behind, and a marker that was already set (the harness itself ran under one) is never removed.
    Not a sandbox: code that clears the variable or passes its own `env=` defeats it, which is no
    wider than what `python_exec` can already do to the machine's files (SECURITY.md)."""
    global _marker_depth, _marker_set_by_us
    with _marker_lock:
        if _marker_depth == 0:
            _marker_set_by_us = AGENT_COMMAND_MARKER not in os.environ
            os.environ[AGENT_COMMAND_MARKER] = os.environ.get(AGENT_COMMAND_MARKER, "1")
        _marker_depth += 1
    try:
        yield
    finally:
        with _marker_lock:
            _marker_depth -= 1
            if _marker_depth == 0 and _marker_set_by_us:
                os.environ.pop(AGENT_COMMAND_MARKER, None)


class PythonExecTool(Tool):
    timeout_s: float = 120.0

    def __init__(self, namespace: dict[str, Any] | None = None) -> None:
        # Per-instance, persistent namespace; a caller seeds e.g. {"ctx": <body>} for code over it.
        # One instance per session/agent-loop keeps state from leaking across sessions.
        self._ns: dict[str, Any] = namespace if namespace is not None else {}

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="python_exec",
            group="code",
            gate_family="code", ingest="none", host="dangerous", result_origin="trusted",
            description=(
                "Execute Python in a STATEFUL REPL. The namespace PERSISTS across calls "
                "this session — variables, imports, and function definitions you create "
                "stay available in later calls. Use print(...) to see results (stdout is "
                "captured and returned; the last expression is NOT auto-printed). "
                "Exceptions are returned as a traceback so you can correct and retry."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute in the persistent namespace.",
                    },
                },
                "required": ["code"],
            },
            destructive=True,
            estimated_tokens=300,
        )

    async def _execute(self, code: str) -> ToolResult:
        buf = io.StringIO()

        def _run() -> None:
            with _agent_command_marker(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    exec(compile(code, "<python_exec>", "exec"), self._ns)  # noqa: S102
                except BaseException:  # noqa: BLE001 - surface ANY error to the model, like a REPL
                    buf.write("\n" + traceback.format_exc())

        # Run the (blocking) exec off the event loop so the loop stays responsive.
        await asyncio.to_thread(_run)

        out = buf.getvalue()
        return self.ok(out if out.strip() else "(no output)", chars=len(out))
