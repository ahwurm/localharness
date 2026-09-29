"""Tool base types: ToolProtocol, Tool ABC, ToolSchema, ToolParameter, ToolResult, ToolVetoed."""
import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

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


GateFamily = Literal["write", "shell", "code", "delegate", "network", "allow"]
"""The permission gate's rule-set branches a tool may DECLARE (agent/verdict.py `_kind` / `evaluate`)
— exactly the six branch names, nothing invented. The gate's two other outcomes are not declarable:
`mcp` comes from the MCP wrapper's group, and `tool-unfamiliar` is what an undeclared (None) family
means — asked about once per workspace in `guarded`. Defined here, beside ToolSchema, because
tools/base.py must never depend on the agent package (agent/__init__ imports agent.loop, which
imports the tool registry — a cycle); agent/gate_types.py and agent/verdict.py import it from here."""

GATE_FAMILIES: frozenset[str] = frozenset(get_args(GateFamily))

_log = logging.getLogger(__name__)
_WARNED_FAMILIES: set[tuple[str, str]] = set()


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
    # delegate, web, memory, or `mcp/<server>` for a discovered MCP tool. It is the exposure
    # taxonomy, where a GROUP — not a tool — is the unit an agent is granted (PRD §6,
    # tools/registry.py), the ACP tool kind (channels/acp.py) and the `mcp/` marker. It is NOT a
    # permission-gate input: the gate reads `gate_family` below. "other" means unclassified:
    # every registered builtin names its group, and a test asserts none is left at the default.
    group: str = "other"
    # What the safety model reads instead of a tool's NAME (SAFE-01). Every default FAILS CLOSED:
    # a tool that declares nothing is treated as ingesting attacker-controllable content, able to
    # change the host, returning untrusted results, and in no gate family (asked about in
    # `guarded`). Never provenance: which plugin contributed a tool is not a safety input (CORE-04).
    # Readers: the permission gate classifies by `gate_family` (agent/verdict.py `_kind`); the
    # capability floor and the context store still read today's name sets, which every builtin's
    # declaration matches (the oracle in tests/unit/test_tool_declarations.py).
    # `exclude=True` keeps all five out of `model_dump()`, which is what
    # `provider/client._tools_to_api_format` puts on the wire and the chat template renders into
    # the prompt (measured on the served model: +32 tokens per tool) — harness metadata, not
    # something the model should read or pay for.
    ingest: Literal["untrusted", "none"] = Field("untrusted", exclude=True)
    host: Literal["dangerous", "safe"] = Field("dangerous", exclude=True)
    result_origin: Literal["untrusted", "trusted"] = Field("untrusted", exclude=True)
    gate_family: GateFamily | None = Field(None, exclude=True)
    # Which plugin contributed this tool (PAPI-05), stamped at registration; None for core's own.
    # Provenance only — the safety model NEVER reads it.
    source_plugin: str | None = Field(None, exclude=True)

    @field_validator("gate_family", mode="before")
    @classmethod
    def _unrecognised_family_is_unset(cls, value: Any, info: ValidationInfo) -> Any:
        """SAFE-01: an unrecognised gate_family counts as unset — asked about, never a crash.

        A plugin's typo must fail closed rather than disable the plugin, so this coerces and warns
        (once per tool name and value — info() is called every turn) instead of raising."""
        if value is None or (isinstance(value, str) and value in GATE_FAMILIES):
            return value
        key = (str(info.data.get("name", "?")), repr(value))
        if key not in _WARNED_FAMILIES:
            _WARNED_FAMILIES.add(key)
            _log.warning("tool %r declares unknown gate_family %s — treated as undeclared "
                         "(it is asked about); known families: %s", key[0], key[1],
                         ", ".join(sorted(GATE_FAMILIES)))
        return None


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
    append branch reads-then-concatenates through the same pair). The two are a PAIR on any tool
    that writes — see `_editor_hooks_unpaired`."""

    def _editor_hooks_unpaired(self) -> "ToolResult | None":
        """Refuse to run a writing tool with half the editor seam wired (review finding R5).

        Every editor-backed write is a read-modify-write: ACP has no append, so `write`'s append
        branch and `edit`'s match both have to read the current buffer before rewriting the whole
        file. A write hook without its read hook would read the stale DISK copy (or nothing at
        all) and hand the editor a file with the user's unsaved work deleted. `channels/acp.py`
        wires the pair all-or-nothing; this is the assertion that says so at the only place the
        damage would happen, and it returns an error instead of touching the disk behind the
        buffer. Read-only tools never call it — `read` is wired with the read hook alone."""
        if (self.file_read_hook is None) == (self.file_write_hook is None):
            return None
        missing = "file_read_hook" if self.file_read_hook is None else "file_write_hook"
        return self.err(
            f"Editor-backed file I/O is misconfigured: {missing} is not set while its partner "
            "is. A write through an editor must read the buffer first, so the two hooks are "
            "wired together or not at all — refusing to touch the file on disk.",
            error_type="execution_error",
        )

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
