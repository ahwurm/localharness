"""Every parameter a tool advertises to the model is one its _execute accepts."""
from __future__ import annotations

import inspect

from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.browser_session import (
    BrowserClickTool, BrowserOpenTool, BrowserScrollTool, BrowserTypeTool)
from localharness.tools.builtin.browser_tool import BrowserFetchTool
from localharness.tools.registry import ToolRegistry


async def test_every_advertised_parameter_reaches_execute(tmp_path):
    registry = ToolRegistry()
    await register_builtin_tools(registry, workspace_root=str(tmp_path))
    browser = [cls(object()) for cls in (
        BrowserFetchTool, BrowserOpenTool, BrowserClickTool, BrowserTypeTool, BrowserScrollTool)]
    for tool in [*registry.global_tools(), *browser]:
        params = inspect.signature(tool._execute).parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        advertised = set(tool.info().parameters.get("properties", {}))
        assert advertised <= set(params), (tool.info().name, advertised - set(params))
