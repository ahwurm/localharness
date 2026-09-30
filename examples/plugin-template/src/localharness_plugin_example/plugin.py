"""The plugin class: what the example is (its manifest), its settings, and what it contributes.

LocalHarness imports this module only once the plugin is enabled; listing the plugin reads the
package's entry-point metadata alone.
"""
from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

from localharness.plugins.api import (
    Check, CliDescriptor, Plugin, PluginContext, PluginManifest, SlashDescriptor,
)
from localharness.tools.base import ToolProtocol

__version__ = "0.1.0"  # keep equal to pyproject.toml's version — the tests check it


class ExampleConfig(BaseModel):
    """Harness-level settings: the `example:` section of config.yaml or overrides.yaml, at the
    machine or the project layer. Core validates the merged section with this model and hands the
    result to the plugin as ctx.config."""

    model_config = ConfigDict(extra="forbid")  # a misspelled key is an error, as in core's settings
    color: str = Field("#4a90d9", pattern=r"^#[0-9a-fA-F]{6}$",
                       description="The swatch color, as #rrggbb.")
    # A field that names a network endpoint, a credential or an access list is machine-only, so a
    # project's settings can never change it (GLOBAL_ONLY comes from localharness.plugins.api):
    # url: str = Field("", json_schema_extra=GLOBAL_ONLY)  # endpoints, credentials, access lists
    # This plugin has no such setting, so the line stays a comment.


class ExampleAgentConfig(BaseModel):
    """Agent-level settings: the `agent.example` section, which an agent's own file can set for that
    agent. Core validates it with this model and hands the result to the plugin as
    ctx.agent_config."""

    model_config = ConfigDict(extra="forbid")
    size: int = Field(8, ge=1, le=256, description="The swatch's width and height, in pixels.")


class ExamplePlugin(Plugin):
    """renders a solid-color swatch PNG — the copyable example plugin

    The first line of this docstring is what `localharness plugins list` shows for the plugin once it
    is on. Every Plugin method is optional and does nothing by default, so a plugin overrides only
    what it contributes. This one leaves three alone:
    - configure() returns "ready": nothing here needs a setting before it can work. A plugin that
      does (an endpoint, say) returns ("unconfigured", "<name>.<field>") until it is set.
    - start() and stop(): nothing runs between turns, so there is nothing to start or release.
    - channels(): it adds no channel.
    """

    manifest = PluginManifest(
        name="example",
        version=__version__,
        kind="tools",
        requires_localharness=">=0.15,<1",  # the LocalHarness releases this plugin is tested against
        cli=(CliDescriptor(name="example", help="Show what the example plugin does.",
                           target="localharness_plugin_example.cli:app"),),
        slash=(SlashDescriptor(name="/example", help="Show the example plugin's swatch settings",
                               target="localharness_plugin_example.slash:run"),),
    )
    ConfigModel = ExampleConfig
    AgentConfigModel = ExampleAgentConfig
    wants_artifacts = True  # core computes <state dir>/artifacts/example/ → ctx.paths.artifact_dir

    async def tools(self, ctx: PluginContext) -> list[ToolProtocol]:
        """The swatch tool. Its module is imported here, not at the top of this file: an enabled
        plugin's class is also loaded for `localharness --help` and `doctor`, and neither needs the
        tool's code."""
        from localharness_plugin_example.tool import SwatchTool

        return [SwatchTool(ctx)]

    def doctor(self, ctx: PluginContext) -> list[Check]:
        """One check: can the swatch tool write where core told it to? It only looks — doctor runs
        outside a session, often before any swatch exists, and must create nothing."""
        name, root = self.manifest.name, ctx.paths.artifact_dir
        if root is None:
            return [Check(name=name, status="fail", detail="no artifact directory was assigned",
                          hint="the plugin sets wants_artifacts = True, so core should have "
                               "assigned one — report this as a LocalHarness bug")]
        nearest = next(p for p in (root, *root.parents) if p.exists())  # root may not exist yet
        if nearest.is_dir() and os.access(nearest, os.W_OK):
            return [Check(name=name, status="pass",
                          detail=f"swatches render in {ctx.config.color}; artifacts go to {root}")]
        return [Check(name=name, status="fail", detail=f"cannot write under {nearest}",
                      hint=f"check the permissions of {nearest}")]
