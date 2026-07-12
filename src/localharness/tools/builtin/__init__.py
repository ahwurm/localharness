"""Built-in tools registration."""
from typing import Any

from localharness.tools.registry import ToolRegistry


async def register_builtin_tools(
    registry: ToolRegistry,
    memory_store=None,
    eviction_store=None,
    workspace_root: str | None = None,
) -> None:
    """Register all built-in tools at global scope. Call once at harness startup.

    `memory_store`: if provided, registers memory_search / memory_get (queryable-memory
    handle — the system prompt inlines only a fact index, full bodies served on demand).
    `eviction_store`: if provided, registers tool_result_get (restores tool-result bodies
    evicted to stubs by the ContextManager). Both are wired only when their backing store
    exists so the bench/test paths that pass neither keep the original builtin set.
    `workspace_root`: opt-in confinement (issue #15). When set, write/edit targets and
    bash_exec working_dir must resolve inside it; None (default) = unconfined."""
    from localharness.tools.builtin.bash_tool import BashExecTool
    from localharness.tools.builtin.chunk_tool import ChunkTool
    from localharness.tools.builtin.edit_tool import EditTool
    from localharness.tools.builtin.glob_tool import GlobTool
    from localharness.tools.builtin.grep_tool import GrepTool
    from localharness.tools.builtin.load_document_tool import LoadDocumentTool
    from localharness.tools.builtin.read_tool import ReadTool
    from localharness.tools.builtin.web_tool import WebFetchTool, WebPageQueryTool, WebSearchTool
    from localharness.tools.builtin.write_tool import WriteTool

    for tool in [GlobTool(), GrepTool(), ReadTool(),
                 WriteTool(workspace_root=workspace_root), EditTool(workspace_root=workspace_root),
                 BashExecTool(workspace_root=workspace_root),
                 WebSearchTool(), WebFetchTool(), WebPageQueryTool(), ChunkTool(), LoadDocumentTool()]:
        await registry.register(tool, scope="global")

    if memory_store is not None:
        from localharness.tools.builtin.memory_tools import (
            MemoryGetTool,
            MemoryRememberTool,
            MemorySearchTool,
        )
        await registry.register(MemorySearchTool(memory_store), scope="global")
        await registry.register(MemoryGetTool(memory_store), scope="global")
        await registry.register(MemoryRememberTool(memory_store), scope="global")

    if eviction_store is not None:
        from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
        await registry.register(ToolResultGetTool(eviction_store), scope="global")


def bind_agent_store_tools(registry: ToolRegistry, store: Any) -> None:
    """Re-bind this agent's store-backed verb tools onto its registry so each agent's verbs hit ITS
    OWN ContentStore (per-agent isolation; closes the latent tool_result_get root-store leak). Only
    rebinds tools the agent ALREADY has — never adds a withheld capability. Call after from_allowed
    (a child) or after registering builtins (the root)."""
    from localharness.tools.builtin.chunk_tool import ChunkTool
    from localharness.tools.builtin.load_document_tool import LoadDocumentTool
    from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
    from localharness.tools.builtin.web_tool import WebFetchTool, WebPageQueryTool
    for tool in (WebFetchTool(store), WebPageQueryTool(store), ToolResultGetTool(store),
                 ChunkTool(store), LoadDocumentTool(store)):
        registry.rebind_global(tool)
