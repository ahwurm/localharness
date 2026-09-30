"""generate_image on the image plugin's context: settings from ctx.config (never the environment),
the picture written as a core-minted artifact into ctx.paths.artifact_dir, the mime chosen by the
bytes, the declarations, the template contract and the fp8_e4m3fn_fast corruption pin.

ComfyUI is faked at the transport: `generate_image_tool._TRANSPORT = httpx.MockTransport(handler)`."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from localharness.core.events import ARTIFACT_ID_RE, ArtifactRef
from localharness.plugins.api import PluginContext, PluginPaths
from localharness.tools.builtin import generate_image_tool as gi
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.generate_image_tool import GenerateImageTool, build_graph, load_template
from localharness.tools.builtin.image_plugin import ImageConfig
from localharness.tools.registry import ToolRegistry

PNG = b"\x89PNG\r\n\x1a\n-fake-image-bytes"
JPEG = b"\xff\xd8\xff\xe0-fake-jpeg"
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 -fake"
URL = "http://comfy.test"


def _root(tmp_path: Path) -> Path:
    return tmp_path / "state" / "artifacts" / "image"


def _tool(tmp_path: Path, artifact_dir: Path | None | str = "default", **cfg) -> GenerateImageTool:
    cfg.setdefault("comfyui_url", URL)
    return GenerateImageTool(PluginContext(
        bus=None, tools=None, hooks=None, config=ImageConfig(**cfg), agent_config=None,
        paths=PluginPaths(global_config_dir=tmp_path / "cfg", workspace=None,
                          state_dir=tmp_path / "state",
                          artifact_dir=_root(tmp_path) if artifact_dir == "default" else artifact_dir),
        llm=None))


def _done(pid="p1", filename="lh_x_00001_.png"):
    return {pid: {
        "status": {"status_str": "success", "completed": True},
        "outputs": {"459": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}},
    }}


def _fake_comfy(monkeypatch, *, history=None, post_status=200, post_json=None,
                connect_error=False, view_content=PNG):
    """Install a fake ComfyUI on the transport seam. `history` is one /history payload per poll
    (exhausted -> {}, still running). Returns a recorder of what was sent."""
    rec = {"posted": None, "view_params": None, "polls": 0}
    seq = list(history or [])

    def handler(req: httpx.Request) -> httpx.Response:
        if connect_error:
            raise httpx.ConnectError("connection refused", request=req)
        p = req.url.path
        if p == "/prompt":
            rec["posted"] = json.loads(req.content)
            return httpx.Response(post_status, json=post_json if post_json is not None else {"prompt_id": "p1"})
        if p == "/history/p1":
            rec["polls"] += 1
            return httpx.Response(200, json=seq.pop(0) if seq else {})
        if p == "/view":
            rec["view_params"] = dict(req.url.params)
            return httpx.Response(200, content=view_content)
        return httpx.Response(404)

    monkeypatch.setattr(gi, "_TRANSPORT", httpx.MockTransport(handler))
    return rec


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(gi, "_POLL_S", 0.01)


# --- the tool is the plugin's, never core's -------------------------------------

async def test_core_registers_no_generate_image():
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    assert registry.has("generate_image") is False


def test_the_schema_is_named_grouped_and_declared_safe(tmp_path):
    tool = _tool(tmp_path, timeout_s=100.0)
    schema = tool.info()
    assert (schema.name, schema.group) == ("generate_image", "image")
    assert {"ingest", "host", "result_origin"} <= schema.model_fields_set
    assert (schema.ingest, schema.host, schema.result_origin, schema.gate_family) == (
        "none", "safe", "trusted", None)
    assert tool.timeout_s == 130.0  # the registry's outer bound never cuts the tool's own deadline


# --- happy path: a core artifact -------------------------------------------------

async def test_happy_path_writes_a_core_artifact_and_pins_the_graph(monkeypatch, tmp_path):
    rec = _fake_comfy(monkeypatch, history=[{}, _done()])
    r = await _tool(tmp_path).run(prompt="a red fox on snow", width=1000, height=512, steps=8, seed=42)
    assert r.success is True, r.error
    ref = ArtifactRef.model_validate(r.metadata["artifact"])
    assert (ref.plugin, ref.kind, ref.mime) == ("image", "image", "image/png")
    assert ARTIFACT_ID_RE.fullmatch(ref.id)
    dest = _root(tmp_path) / f"{ref.id}.png"
    assert dest.read_bytes() == PNG
    assert r.metadata["path"] == str(dest.absolute())
    for needle in (f"artifact {ref.id}", str(dest.absolute()), "992x512", "seed 42", "8 steps"):
        assert needle in r.output
    graph = rec["posted"]["prompt"]
    assert "_comment" not in graph
    assert graph["452"]["inputs"]["prompt"] == "a red fox on snow"
    assert (graph["456"]["inputs"]["width"], graph["456"]["inputs"]["height"]) == (992, 512)
    assert (graph["458"]["inputs"]["seed"], graph["458"]["inputs"]["steps"]) == (42, 8)
    assert graph["451"]["inputs"]["weight_dtype"] == "default"  # the Qwen-Image corruption pin
    assert rec["view_params"]["filename"] == "lh_x_00001_.png"


@pytest.mark.parametrize(("data", "mime", "suffix"), [(JPEG, "image/jpeg", ".jpg"),
                                                      (WEBP, "image/webp", ".webp")])
async def test_the_mime_is_chosen_by_the_bytes(monkeypatch, tmp_path, data, mime, suffix):
    _fake_comfy(monkeypatch, history=[_done()], view_content=data)
    r = await _tool(tmp_path).run(prompt="x")
    assert r.success is True, r.error
    ref = ArtifactRef.model_validate(r.metadata["artifact"])
    assert ref.mime == mime
    assert (_root(tmp_path) / f"{ref.id}{suffix}").read_bytes() == data


async def test_bytes_that_are_no_known_image_are_refused_and_nothing_written(monkeypatch, tmp_path):
    _fake_comfy(monkeypatch, history=[_done()], view_content=b"GIF89a-not-allowed")
    r = await _tool(tmp_path).run(prompt="x")
    assert r.success is False
    assert "not a PNG, JPEG or WEBP image" in r.error
    assert not _root(tmp_path).exists() or not any(_root(tmp_path).iterdir())


# --- failure modes ---------------------------------------------------------------

async def test_unreachable_server_names_the_fix(monkeypatch, tmp_path):
    _fake_comfy(monkeypatch, connect_error=True)
    r = await _tool(tmp_path).run(prompt="x")
    assert r.success is False
    assert f"ComfyUI unreachable at {URL}" in r.error


async def test_generation_error_status_surfaces_server_messages(monkeypatch, tmp_path):
    _fake_comfy(monkeypatch, history=[
        {"p1": {"status": {"status_str": "error", "completed": False,
                           "messages": [["execution_error", {"node_type": "KSampler"}]]}}},
    ])
    r = await _tool(tmp_path).run(prompt="x")
    assert r.success is False
    assert "generation failed" in r.error and "KSampler" in r.error


async def test_rejected_workflow_surfaces_http_body(monkeypatch, tmp_path):
    _fake_comfy(monkeypatch, post_json={"error": "invalid prompt"}, post_status=400)
    r = await _tool(tmp_path).run(prompt="x")
    assert r.success is False
    assert "rejected" in r.error and "invalid prompt" in r.error


async def test_deadline_exceeded_names_the_setting(monkeypatch, tmp_path):
    _fake_comfy(monkeypatch, history=[])  # never completes
    r = await _tool(tmp_path, timeout_s=0.05).run(prompt="x")
    assert r.success is False
    assert "not finished after" in r.error and "image.timeout_s" in r.error


async def test_empty_prompt_refused(tmp_path):
    r = await _tool(tmp_path).run(prompt="   ")
    assert r.success is False and r.error_type == "validation_error"
    assert "prompt is empty" in r.error


async def test_empty_url_names_the_setting(tmp_path):
    r = await _tool(tmp_path, comfyui_url="").run(prompt="x")
    assert r.success is False
    assert "image.comfyui_url" in r.error


async def test_no_artifact_directory_is_an_error(tmp_path):
    r = await _tool(tmp_path, artifact_dir=None).run(prompt="x")
    assert r.success is False
    assert "no artifact directory was assigned to the image plugin" in r.error


# --- template contract (the model-agnostic seam) ----------------------------------

def test_shipped_template_loads_fills_and_leaves_no_placeholders():
    template, name = load_template("")
    assert name == "qwen-image-2.1"
    graph = build_graph(template, prompt="p", width=1024, height=1024, steps=30, seed=1, prefix="lh_test")
    assert "__LH_" not in json.dumps(graph)


def test_template_without_prompt_slot_is_refused(tmp_path):
    bad = tmp_path / "no-prompt.json"
    bad.write_text(json.dumps({"1": {"class_type": "X", "inputs": {"text": "static"}}}))
    template, _ = load_template(str(bad))
    with pytest.raises(ValueError, match="__LH_PROMPT__"):
        build_graph(template, prompt="p", width=8, height=8, steps=1, seed=1, prefix="x")


async def test_the_workflow_setting_swaps_the_model(monkeypatch, tmp_path):
    other = tmp_path / "flux-schnell.json"
    other.write_text(json.dumps({
        "_comment": "another model entirely",
        "1": {"class_type": "FluxLoader", "inputs": {"ckpt": "flux.safetensors"}},
        "2": {"class_type": "FluxSample", "inputs": {"model": ["1", 0], "prompt": "__LH_PROMPT__",
                                                     "seed": "__LH_SEED__"}},
    }))
    rec = _fake_comfy(monkeypatch, history=[_done()])
    r = await _tool(tmp_path, workflow=str(other)).run(prompt="hi", seed=7)
    assert r.success is True, r.error
    assert r.metadata["template"] == "flux-schnell"
    graph = rec["posted"]["prompt"]
    assert graph["1"]["class_type"] == "FluxLoader" and "_comment" not in graph
    assert graph["2"]["inputs"]["prompt"] == "hi" and graph["2"]["inputs"]["seed"] == 7
