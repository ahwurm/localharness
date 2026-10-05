"""ToolResultGetTool: restore the full body of an evicted tool result by its id."""
from localharness.agent.context import ActiveReferenceError, ContentStore
from localharness.tools.base import Tool, ToolResult, ToolSchema


class ToolResultGetTool(Tool):
    """Re-pull the full body of a tool result that was evicted to a restorable stub.

    Bulky tool results are replaced in-context with a stub like
    `[tool result evicted — read_file /notes/y120.md — ~N tokens — call tool_result_get('<id>')
    to restore the full body]`, and the request's last message ends with an `[out of view: …]`
    line listing every such stub. This tool returns the exact original body for that id from
    the ContentStore."""

    def __init__(self, store: ContentStore) -> None:
        self._store = store

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="tool_result_get",
            group="fs.read",
            gate_family="allow", ingest="none", host="safe", result_origin="trusted",
            description=(
                "Restore the full body of a previously evicted tool result. When you see a "
                "stub like \"[tool result evicted — <tool> <argument> — ~N tokens — call "
                "tool_result_get('<id>') to restore the full body]\", or the tail note "
                "\"[out of view: ...]\" listing such stubs, pass that exact <id> here to get "
                "the original content back. Do it BEFORE a step that depends on that content. "
                "For a substantive dependent step, optionally set active_step and source to keep "
                "up to four required snapshots available. A new step replaces the set; the same "
                "source replaces its revision. Send an empty active_step to release the set. "
                "Protection expires at the next human turn. Source is a label, not a file watcher."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The id from the eviction stub.",
                    },
                    "active_step": {
                        "type": "string", "maxLength": 160,
                        "description": "Optional bounded step name; empty releases its references.",
                    },
                    "source": {
                        "type": "string", "maxLength": 240,
                        "description": "Source identity for revision replacement; defaults to id.",
                    },
                },
                "required": ["id"],
            },
            destructive=False,
            estimated_tokens=400,
        )

    async def _execute(
        self, id: str, active_step: str | None = None, source: str | None = None,
    ) -> ToolResult:
        body = self._store.get(id)
        if body is None:
            return self.err(
                f"No evicted tool result found for id '{id}'.",
                error_type="not_found",
            )
        if active_step is not None:
            try:
                self._store.declare_active_reference(active_step, id, source)
            except ActiveReferenceError as exc:
                return self.err(str(exc), error_type="active_reference_blocked")
        return self.ok(body)
