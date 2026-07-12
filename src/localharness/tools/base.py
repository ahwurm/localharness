"""Tool base types: ToolProtocol, Tool ABC, ToolSchema, ToolParameter, ToolResult, ToolVetoed."""
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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
        return ToolResult(output=output, success=True, metadata=metadata)

    def err(self, message: str, error_type: str = "execution_error", **metadata: Any) -> ToolResult:
        return ToolResult(
            output="",
            success=False,
            error=message,
            error_type=error_type,  # type: ignore[arg-type]
            metadata=metadata,
        )
