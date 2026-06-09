"""Built-in tools registration."""
from localharness.tools.registry import ToolRegistry


async def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools at global scope. Call once at harness startup."""
    from localharness.tools.builtin.bash_tool import BashExecTool
    from localharness.tools.builtin.glob_tool import GlobTool
    from localharness.tools.builtin.grep_tool import GrepTool
    from localharness.tools.builtin.read_tool import ReadTool
    from localharness.tools.builtin.web_tool import WebFetchTool, WebSearchTool
    from localharness.tools.builtin.write_tool import WriteTool

    for tool in [GlobTool(), GrepTool(), ReadTool(), WriteTool(), BashExecTool(),
                 WebSearchTool(), WebFetchTool()]:
        await registry.register(tool, scope="global")
