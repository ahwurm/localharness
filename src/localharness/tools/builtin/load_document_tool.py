"""LoadDocumentTool: deliver a large local document as a HANDLE, not bytes (Decision B).

Closes the lossless-retain seam for tool-delivered over-window content: a single big tool result is
NOT reliably evicted (TOOL_EVICT_KEEP_LAST protects the newest results) and would be lossily capped
before any handle exists — so "read a big file → grant its handle" is structurally unreliable. This
tool RETAINS the full body in the agent's ContentStore on produce and returns only a short stub +
handle, so the orchestrator holds a grantable handle without the bytes ever entering its window. It
then delegates over-window analysis to the cruncher with grant_handles=[<handle>].

Origin is TRUSTED (a local file the agent was pointed at, like `read`). Untrusted external ingest
stays the web path (no-host-dangerous agents only); a host-dangerous orchestrator must not pull
untrusted bytes into its store — so this tool does not mint untrusted handles."""
import os

from localharness.agent.context import ContentStore
from localharness.tools.base import Tool, ToolResult, ToolSchema

_MAX_DOC_BYTES = 50 * 1024 * 1024  # 50MB sanity cap


class LoadDocumentTool(Tool):
    """Retain a local document's full body in the store and return a grantable handle + stub."""

    def __init__(self, store: ContentStore | None = None) -> None:
        self._store = store if store is not None else ContentStore()

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="load_document",
            description=(
                "Load a large local document and retain its FULL text under a handle WITHOUT pulling "
                "it into context. Returns the handle + a short stub. Use for a document too big to "
                "read inline: load it, then delegate analysis to the cruncher — "
                "agent('cruncher', task='<your question>', grant_handles=['<handle>']) — which reads "
                "the body by handle, splits it, and returns a faithful answer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the local document."},
                },
                "required": ["path"],
            },
            destructive=False,
            estimated_tokens=400,
        )

    async def _execute(self, path: str) -> ToolResult:
        p = os.path.expanduser(path)
        if not os.path.isfile(p):
            return self.err(f"No file at '{path}'.", error_type="not_found")
        try:
            size = os.path.getsize(p)
            if size > _MAX_DOC_BYTES:
                return self.err(
                    f"Document is {size} bytes (> {_MAX_DOC_BYTES} cap).", error_type="execution_error"
                )
            with open(p, encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError as exc:
            return self.err(f"Could not read '{path}': {exc}", error_type="execution_error")

        handle = self._store.put(body, origin="trusted")
        return self.ok(
            f"[document loaded — {len(body)} chars retained as handle '{handle}' (not shown inline; "
            f"too large for the window)]\n"
            f"To analyze it, delegate to the cruncher: "
            f"agent('cruncher', task='<your question about the document>', grant_handles=['{handle}']).",
            doc_handle=handle,
            chars=len(body),
        )
