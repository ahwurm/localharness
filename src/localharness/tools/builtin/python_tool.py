"""PythonExecTool: a STATEFUL Python REPL — the namespace persists across calls.

Unlike bash_exec (a one-shot subprocess), this holds a long-lived namespace dict, so
variables, imports, and function definitions survive between calls within a session.
It is the substrate for RLM mode: the long input is seeded as a `ctx` variable that the
model inspects with code (regex/slice/decompose) instead of receiving it in its prompt.

No sandbox/isolation — same trust posture as the existing bash_exec (the model already
runs arbitrary shell on the host). Run an isolated VM/container if that matters.
"""
import asyncio
import contextlib
import io
import traceback
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema


class PythonExecTool(Tool):
    timeout_s: float = 120.0

    def __init__(self, namespace: dict[str, Any] | None = None) -> None:
        # Per-instance, persistent namespace. RLM mode seeds e.g. {"ctx": <input>}.
        # One instance per session/agent-loop keeps state from leaking across sessions.
        self._ns: dict[str, Any] = namespace if namespace is not None else {}

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="python_exec",
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
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    exec(compile(code, "<python_exec>", "exec"), self._ns)  # noqa: S102
                except BaseException:  # noqa: BLE001 - surface ANY error to the model, like a REPL
                    buf.write("\n" + traceback.format_exc())

        # Run the (blocking) exec off the event loop so the loop stays responsive.
        await asyncio.to_thread(_run)

        out = buf.getvalue()
        return self.ok(out if out.strip() else "(no output)", chars=len(out))
