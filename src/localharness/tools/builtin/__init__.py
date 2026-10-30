"""Built-in tools registration."""
import asyncio
import logging
from pathlib import Path
from typing import Any

from localharness.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


async def register_builtin_tools(
    registry: ToolRegistry,
    eviction_store=None,
    workspace_root: str | None = None,
    boundary: Path | None = None,
) -> None:
    """Register all built-in tools at global scope. Call once at harness startup.

    `eviction_store`: if provided, registers tool_result_get (restores tool-result bodies
    evicted to stubs by the ContextManager); wired only when the store exists so the bench/test
    paths that pass none keep the original builtin set. The memory tools are the memory plugin's.
    `workspace_root`: opt-in confinement (issue #15). When set, write/edit targets and
    bash_exec working_dir must resolve inside it; None (default) = unconfined.
    `boundary`: the gate's effective session boundary (start passes it). When given and git_read's
    pins pass, `git_read` is registered rooted there; None (no boundary, the bench, components, the
    mobile replay) registers none; a pin failure registers none and says nothing - doctor explains."""
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
                 WebSearchTool(), WebFetchTool(),
                 WebPageQueryTool(result_cap=lambda: registry.result_size_cap_chars),
                 ChunkTool(), LoadDocumentTool()]:
        await registry.register(tool, scope="global")

    if eviction_store is not None:
        from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
        await registry.register(ToolResultGetTool(eviction_store), scope="global")

    if boundary is not None:
        from localharness.tools.builtin import git_read_tool

        # No `.git` at the boundary or above: nothing to read, and no git is run to find out. An
        # unexpected error while checking (WR-04 / IN-51) registers nothing instead of aborting start.
        offer = False
        try:
            if git_read_tool.git_read_repository_present(boundary):
                offer = await asyncio.to_thread(git_read_tool.git_read_pin_failure, boundary) is None
        except Exception:  # noqa: BLE001
            offer = False
        if offer:
            await registry.register(git_read_tool.GitReadTool(boundary), scope="global")


def bind_agent_store_tools(registry: ToolRegistry, store: Any) -> None:
    """Re-bind this agent's store-backed verb tools onto its registry so each agent's verbs hit ITS
    OWN ContentStore (per-agent isolation; closes the latent tool_result_get root-store leak). Only
    rebinds tools the agent ALREADY has — never adds a withheld capability. Call after from_allowed
    (a child) or after registering builtins (the root)."""
    from localharness.tools.base import StoreBound
    from localharness.tools.builtin.chunk_tool import ChunkTool
    from localharness.tools.builtin.load_document_tool import LoadDocumentTool
    from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
    from localharness.tools.builtin.web_tool import WebFetchTool, WebPageQueryTool
    for tool in (WebFetchTool(store),
                 WebPageQueryTool(store, result_cap=lambda: registry.result_size_cap_chars),
                 ToolResultGetTool(store),
                 ChunkTool(store), LoadDocumentTool(store)):
        registry.rebind_global(tool)
    for tool in registry.global_tools():
        if isinstance(tool, StoreBound) and (bound := tool.bound_to(store)) is not tool:
            registry.rebind_global(bound)


async def release_agent_store_tools(registry: ToolRegistry, store: Any) -> None:
    """End this agent's per-store tool state (a child's dispatch `finally` calls it). One tool's
    failure never stops the others or reaches the caller; cancellation still propagates."""
    from localharness.tools.base import StoreReleasing
    for tool in registry.global_tools():
        if isinstance(tool, StoreReleasing):
            try:
                await tool.release_store(store)
            except Exception:  # noqa: BLE001
                log.warning("release_store failed for tool %s", tool.info().name, exc_info=True)
