"""Built-in tools registration."""
from localharness.tools.registry import ToolRegistry


async def register_builtin_tools(
    registry: ToolRegistry,
    memory_store=None,
    eviction_store=None,
) -> None:
    """Register all built-in tools at global scope. Call once at harness startup.

    `memory_store`: if provided, registers memory_search / memory_get (queryable-memory
    handle — the system prompt inlines only a fact index, full bodies served on demand).
    `eviction_store`: if provided, registers tool_result_get (restores tool-result bodies
    evicted to stubs by the ContextManager). Both are wired only when their backing store
    exists so the bench/test paths that pass neither keep the original builtin set."""
    from localharness.tools.builtin.bash_tool import BashExecTool
    from localharness.tools.builtin.edit_tool import EditTool
    from localharness.tools.builtin.glob_tool import GlobTool
    from localharness.tools.builtin.grep_tool import GrepTool
    from localharness.tools.builtin.read_tool import ReadTool
    from localharness.tools.builtin.web_tool import WebFetchTool, WebPageQueryTool, WebSearchTool
    from localharness.tools.builtin.write_tool import WriteTool

    for tool in [GlobTool(), GrepTool(), ReadTool(), WriteTool(), EditTool(), BashExecTool(),
                 WebSearchTool(), WebFetchTool(), WebPageQueryTool()]:
        await registry.register(tool, scope="global")

    if memory_store is not None:
        from localharness.tools.builtin.memory_tools import MemoryGetTool, MemorySearchTool
        await registry.register(MemorySearchTool(memory_store), scope="global")
        await registry.register(MemoryGetTool(memory_store), scope="global")

    if eviction_store is not None:
        from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
        await registry.register(ToolResultGetTool(eviction_store), scope="global")
