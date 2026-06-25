"""ChunkTool: split a (possibly GRANTED) over-window body into smaller addressable handles.

The R7/R8 navigation verb for J3: a body too large for one window is split into contiguous,
LOSSLESS pieces ("".join(pieces) == body), each retained in the agent's ContentStore under its own
handle. Origin taint is STICKY — a chunk of an untrusted page stays untrusted (derived_from) — so a
chunk can never relaunder attacker bytes into a trusted exec. The cruncher reads each piece by handle
(tool_result_get) and processes it independently, then combines — over-window reduce without ever
holding the whole body in one window."""
from localharness.agent.context import ContentStore
from localharness.tools.base import Tool, ToolResult, ToolSchema

_DEFAULT_MAX_CHARS = 12_000


def split_lossless(body: str, max_chars: int) -> list[str]:
    """Split `body` into contiguous pieces each ≤ max_chars, preferring a newline boundary. LOSSLESS:
    "".join(split_lossless(b, n)) == b for any b, n≥1 (pieces are adjacent slices, nothing dropped)."""
    if max_chars < 1:
        max_chars = 1
    pieces: list[str] = []
    i, n = 0, len(body)
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            nl = body.rfind("\n", i + 1, end)  # back up to a newline for a cleaner cut (keeps it)
            if nl != -1:
                end = nl + 1
        pieces.append(body[i:end])
        i = end
    return pieces


class ChunkTool(Tool):
    """Split a large retained/granted body into smaller handles for over-window processing."""

    def __init__(self, store: ContentStore | None = None) -> None:
        self._store = store if store is not None else ContentStore()

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="chunk",
            description=(
                "Split a large body you hold as a handle (an eviction-stub id, a 'pg-N' page, or a "
                "granted handle) into smaller numbered pieces, each retained under its own handle. "
                "Use when a body is too big to read in one go: chunk it, then read each piece with "
                "tool_result_get('<handle>'), summarize each independently, and combine. Pieces "
                "reconstruct the original exactly; origin taint is preserved."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Handle/alias of the body to split."},
                    "max_chars": {
                        "type": "integer",
                        "description": f"Max characters per piece (default {_DEFAULT_MAX_CHARS}).",
                    },
                },
                "required": ["id"],
            },
            destructive=False,
            estimated_tokens=500,
        )

    async def _execute(self, id: str, max_chars: int | None = None) -> ToolResult:
        body = self._store.get(id)
        if body is None:
            return self.err(
                f"No content found for handle '{id}' (not retained, not granted, or aged out).",
                error_type="not_found",
            )
        size = int(max_chars) if max_chars else _DEFAULT_MAX_CHARS
        pieces = split_lossless(body, size)
        origin = self._store.origin(id) or "trusted"
        # Retain each piece under its own handle; derived_from=id makes taint sticky (untrusted
        # source -> untrusted chunk), so a chunk can never relaunder into a trusted exec.
        handles = [self._store.put(p, derived_from=id) for p in pieces]
        lines = "\n".join(f"  piece {i}: {h}  (~{len(pieces[i])} chars)" for i, h in enumerate(handles))
        return self.ok(
            f"[chunked '{id}' into {len(handles)} piece(s) of ≤{size} chars — origin {origin}]\n"
            f"{lines}\n"
            f"Read each full piece with tool_result_get('<handle>'). Process pieces independently, "
            f"then combine your per-piece findings into the final answer.",
            chunk_handles=handles,
            origin=origin,
        )
