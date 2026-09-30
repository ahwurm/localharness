"""The image plugin: generate_image, `localharness generate-image` and a ComfyUI doctor check.

Off by default. plugins/builtin.py imports this module for every `--help`, `doctor` and `plugins list`,
enabled or not, so it imports only pydantic and the plugin API; the tool, httpx and the workflow
loader are imported when a session or a check asks for them."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from localharness.plugins.api import (
    GLOBAL_ONLY, Availability, Check, CliDescriptor, Plugin, PluginContext, PluginManifest,
    SetupField,
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


IMAGE_SETUP_HELP = """\
Image needs ComfyUI running on this machine, with three model files:
  models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors   (about 7.3 GB)
  models/text_encoders/qwen3vl_8b_int8_convrot.safetensors          (about 9.4 GB)
  models/vae/qwen_image_2.1_vae_bf16.safetensors                    (about 0.7 GB)
Start it from your ComfyUI folder: venv/bin/python main.py --listen 127.0.0.1 --port 8188
Then run `localharness doctor` to check.

Or paste this into your coding agent to set it up for your hardware:

  Set up ComfyUI on this machine for LocalHarness image generation. Install ComfyUI and run it on
  127.0.0.1, port 8188 (this machine only, not the network). Put these Qwen-Image-2.1 INT8 files in
  its models folder: diffusion_models/qwen_image_2.1_int8_convrot.safetensors (about 7.3 GB),
  text_encoders/qwen3vl_8b_int8_convrot.safetensors (about 9.4 GB) and
  vae/qwen_image_2.1_vae_bf16.safetensors (about 0.7 GB). The weights are under the Qwen Research
  License: personal, non-commercial use. Keep the UNETLoader weight_dtype at "default" (the fp8 fast
  mode spoils the pictures). On an NVIDIA GB10 (DGX Spark) the INT8 kernels compile on first use and
  need the Python headers: start ComfyUI with C_INCLUDE_PATH pointing at them. You are done when
  http://127.0.0.1:8188/system_stats answers and `localharness doctor` shows image reachable."""


class ImagePlugin(Plugin):
    """makes pictures with ComfyUI

    Contributes the generate_image tool, the `localharness generate-image` command and a doctor
    check that asks ComfyUI whether it answers and has the template's model files."""

    manifest = PluginManifest(
        name="image", version="0.1.0", kind="tools", enabled_by_default=False,
        cli=(CliDescriptor(name="generate-image", help="Make a picture with your local ComfyUI server.",
                           target="localharness.cli.generate_image_cmd:app"),),
        setup=(SetupField(key="comfyui_url", prompt="ComfyUI address", default="http://127.0.0.1:8188"),),
        setup_help=IMAGE_SETUP_HELP,
    )
    ConfigModel = ImageConfig
    wants_artifacts = True  # core computes <state dir>/artifacts/image/ -> ctx.paths.artifact_dir

    async def configure(self, ctx: PluginContext) -> Availability:
        """Cheap and offline (no network at start): ready once the machine-level URL is set."""
        return "ready" if ctx.config.comfyui_url else ("unconfigured", "image.comfyui_url")

    async def tools(self, ctx: PluginContext) -> list[ToolProtocol]:
        from localharness.tools.builtin.generate_image_tool import GenerateImageTool

        return [GenerateImageTool(ctx)]

    def doctor(self, ctx: PluginContext) -> list[Check]:
        """Runs only when image is on and configured (the lifecycle checks configure() first)."""
        from localharness.tools.builtin.generate_image_tool import probe

        return probe(ctx.config.comfyui_url, ctx.config.workflow)
