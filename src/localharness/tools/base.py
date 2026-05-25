"""Tool base types: ToolProtocol, Tool ABC, ToolSchema, ToolParameter, ToolResult, ToolVetoed."""
import asyncio
from abc import ABC, abstractmethod
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

    @abstractmethod
    def info(self) -> ToolSchema: ...

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolResult: ...

    async def run(self, **kwargs: Any) -> ToolResult:
        timeout = self.timeout_s or 30.0
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
