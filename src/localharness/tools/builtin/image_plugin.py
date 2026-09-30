"""The image plugin: generate_image, `localharness generate-image` and a ComfyUI doctor check.

Off by default. plugins/builtin.py imports this module for every `--help`, `doctor` and `plugins list`,
enabled or not, so it imports only pydantic and the plugin API; the tool, httpx and the workflow
loader are imported when a session or a check asks for them."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from localharness.plugins.api import (
    GLOBAL_ONLY, Availability, CliDescriptor, Plugin, PluginContext, PluginManifest,
)

if TYPE_CHECKING:
    from localharness.tools.base import ToolProtocol


class ImageConfig(BaseModel):
    """The `image:` section. comfyui_url and workflow are machine-level only: ComfyUI runs whatever
    graph it is sent, so a project folder must never choose the server or the graph."""

    model_config = ConfigDict(extra="forbid")
    comfyui_url: str = Field("", json_schema_extra=GLOBAL_ONLY,
                             description="Address of your ComfyUI server, e.g. http://127.0.0.1:8188.")
    workflow: str = Field("", json_schema_extra=GLOBAL_ONLY,
                          description="Path to your own ComfyUI workflow template; empty uses the "
                                      "shipped qwen-image-2.1.")
    timeout_s: float = Field(570.0, gt=0,
                             description="Seconds to wait for one picture (the first one also loads "
                                         "the model).")

    @field_validator("comfyui_url")
    @classmethod
    def _clean_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("image.comfyui_url must start with http:// or https://")
        return v


class ImagePlugin(Plugin):
    """makes pictures with ComfyUI

    Contributes the generate_image tool, the `localharness generate-image` command and a doctor
    check that asks ComfyUI whether it answers and has the template's model files."""

    manifest = PluginManifest(
        name="image", version="0.1.0", kind="tools", enabled_by_default=False,
        cli=(CliDescriptor(name="generate-image", help="Make a picture with your local ComfyUI server.",
                           target="localharness.cli.generate_image_cmd:app"),),
    )
    ConfigModel = ImageConfig
    wants_artifacts = True  # core computes <state dir>/artifacts/image/ -> ctx.paths.artifact_dir

    async def configure(self, ctx: PluginContext) -> Availability:
        """Cheap and offline (no network at start): ready once the machine-level URL is set."""
        return "ready" if ctx.config.comfyui_url else ("unconfigured", "image.comfyui_url")

    async def tools(self, ctx: PluginContext) -> list[ToolProtocol]:
        from localharness.tools.builtin.generate_image_tool import GenerateImageTool

        return [GenerateImageTool(ctx)]
