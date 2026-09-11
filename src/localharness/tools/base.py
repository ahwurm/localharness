"""Tool base types: ToolProtocol, Tool ABC, ToolSchema, ToolParameter, ToolResult, ToolVetoed."""
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FileReadHook = Callable[[Path], Awaitable[str]]
"""Read a file's text through something other than the disk (PRD §4: Zed's `fs/read_text_file`,
so the agent sees the buffer the user is actually looking at, unsaved edits included)."""

FileWriteHook = Callable[[Path, str], Awaitable[None]]
"""Replace a file's whole text through something other than the disk (PRD §4:
`fs/write_text_file`, which is what puts the change in Zed's review pane)."""


class ToolParameter(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["string", "integer", "number", "boolean", "array", "object"]
    description: str
    enum: list[str] | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, "ToolParameter"] | None = None
    required: list[str] | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    default: Any | None = None


class ToolSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]
    scope: Literal["global", "division", "agent", "mcp"] = "global"
    estimated_tokens: int | None = None
    version: str = "1.0.0"
    destructive: bool = False
    # What KIND of thing this tool does, as one dotted name: fs.read, fs.write, shell, code,
    # delegate, web, memory, or `mcp/<server>` for a discovered MCP tool. The permission gate
    # keys its ask classes on this rather than on tool names (PRD §3.1), and it is the seed of
    # the v0.14 exposure taxonomy, where a GROUP — not a tool — is the unit an agent is granted
    # (PRD §6, .planning/scope-hierarchical-tools-v0.12.md). "other" means unclassified: every
    # registered builtin names its group, and a test asserts none of them is left at the default.
    group: str = "other"


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    output: str
    success: bool = True
    error: str | None = None
    error_type: Literal[
        "validation_error",
        "execution_error",
        "timeout_error",
        "permission_denied",
        "not_found",
    ] | None = None
    duration_ms: int | None = None
    truncated: bool = False
    original_length: int | None = None
    metadata: dict[str, Any] = {}


class ToolVetoed(Exception):
    """Raised by a pre_tool hook to veto execution."""


from typing import Protocol, runtime_checkable


@runtime_checkable
class ToolProtocol(Protocol):
    def info(self) -> ToolSchema: ...
    async def run(self, **kwargs: Any) -> ToolResult: ...


class Tool(ABC):
    """Base class for all LocalHarness tools. Subclass and implement info() and _execute()."""

    timeout_s: float | None = None
    workspace_root: str | None = None  # opt-in write/exec confinement; None = unconfined (default)

    file_read_hook: "FileReadHook | None" = None
    file_write_hook: "FileWriteHook | None" = None
    """Optional editor-backed file I/O for `read`/`write`/`edit` (PRD §4, "Edits through the
    editor"). None — the default and the only value in a terminal session — means the tool
    touches the disk exactly as it always has.

    Set to a coroutine by a channel that owns a real editor: the ACP adapter points them at
    Zed's `fs/read_text_file` / `fs/write_text_file` so the agent sees UNSAVED buffers and every
    write lands in Zed's review pane instead of on disk behind the user's back. They are plain
    instance attributes rather than constructor arguments deliberately — the hooks are known
    only after the ACP client has advertised its capabilities, which is long after
    `register_builtin_tools()` ran, and threading an optional argument through that factory and
    its three callers would have made every call site carry a parameter only one of them can
    ever use. The channel sets them on the registered instances instead
    (`channels/acp.py:_wire_editor_file_io`).

    Contract: `read(path) -> str` raises `OSError` when the client cannot produce the file;
    `write(path, text) -> None` replaces the whole file (ACP has no append mode, so `write`'s
    append branch reads-then-concatenates through the same pair)."""

    def __init__(self, workspace_root: str | None = None) -> None:
        # Filesystem-touching tools (write/edit/bash_exec) accept an optional confinement root.
        # Store-backed tools override __init__ and simply inherit the class-attr default (None),
        # so reading self.workspace_root is always safe.
        self.workspace_root = workspace_root

    def _outside_workspace(self, target: Path) -> "ToolResult | None":
        """Opt-in confinement gate. If workspace_root is set and `target` (already .resolve()'d,
        symlink-safe) is not inside it, return a permission_denied ToolResult; else None."""
        root = self.workspace_root
        if root is None:
            return None
        root_p = Path(root).expanduser().resolve()
        if not target.is_relative_to(root_p):
            return self.err(
                f"Path outside workspace_root blocked: {target} (workspace_root: {root_p})",
                error_type="permission_denied",
            )
        return None

    @abstractmethod
    def info(self) -> ToolSchema: ...

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolResult: ...

    async def run(self, **kwargs: Any) -> ToolResult:
        # The outer bound must EXCEED any per-call inner timeout (e.g. bash_exec's
        # `timeout_s` kwarg, whose _execute has its own wait_for + proc.kill path):
        # if the outer wait_for fires first it cancels _execute mid-await, the inner
        # kill/cleanup path never runs, and the subprocess is orphaned (timeout
        # inversion). Size the outer bound off the call's own timeout_s plus slack so
        # the inner cleanup always wins the race; without a call-level timeout, add
        # bounded slack over the instance default (covers _execute-signature defaults
        # equal to the instance value, e.g. bash's 60/60 tie).
        base = self.timeout_s or 30.0
        try:
            call_timeout = float(kwargs.get("timeout_s") or 0.0)
        except (TypeError, ValueError):
            call_timeout = 0.0
        timeout = max(base, call_timeout + 5.0) if call_timeout else base + min(5.0, base)
        try:
            return await asyncio.wait_for(self._execute(**kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult(
                output="",
                success=False,
                error=f"Tool '{self.info().name}' timed out after {timeout}s",
                error_type="timeout_error",
            )
        except Exception as exc:
            return ToolResult(
                output="",
                success=False,
                error=str(exc),
                error_type="execution_error",
            )

    def ok(self, output: str, **metadata: Any) -> ToolResult:
        # truncated/original_length are real ToolResult fields, not metadata: a producer
        # passing them means the audit trail must see them. Burying truncated in metadata
        # made grep's limit-capped results claim completeness (#133 critic finding).
        truncated = bool(metadata.pop("truncated", False))
        original_length = metadata.pop("original_length", None)
        return ToolResult(output=output, success=True, truncated=truncated,
                          original_length=original_length, metadata=metadata)

    def err(self, message: str, error_type: str = "execution_error", **metadata: Any) -> ToolResult:
        return ToolResult(
            output="",
            success=False,
            error=message,
            error_type=error_type,  # type: ignore[arg-type]
            metadata=metadata,
        )
