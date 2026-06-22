"""ToolResultGetTool: restore the full body of an evicted tool result by its id."""
from localharness.agent.context import EvictionStore
from localharness.tools.base import Tool, ToolResult, ToolSchema


class ToolResultGetTool(Tool):
    """Re-pull the full body of a tool result that was evicted to a restorable stub.

    Bulky tool results are replaced in-context with a stub like
    `[tool result evicted — ~N tokens — call tool_result_get('<id>') to restore]`.
    This tool returns the exact original body for that id from the EvictionStore."""

    def __init__(self, store: EvictionStore) -> None:
        self._store = store

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="tool_result_get",
            description=(
                "Restore the full body of a previously evicted tool result. When you see a "
                "stub like \"[tool result evicted — ~N tokens — call tool_result_get('<id>') "
                "to restore]\", pass that exact <id> here to get the original content back."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The id from the eviction stub.",
                    },
                },
                "required": ["id"],
            },
            destructive=False,
            estimated_tokens=400,
        )

    async def _execute(self, id: str) -> ToolResult:
        body = self._store.get(id)
        if body is None:
            return self.err(
                f"No evicted tool result found for id '{id}'.",
                error_type="not_found",
            )
        return self.ok(body)
