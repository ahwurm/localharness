"""generate_image — opt-in ComfyUI-backed text-to-image: module gating, template contract,
client behavior (submit/poll/download), and the fp8_e4m3fn_fast corruption pin."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from localharness.tools.builtin import generate_image_tool as gi
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.generate_image_tool import GenerateImageTool, build_graph, load_template
from localharness.tools.registry import ToolRegistry

PNG = b"\x89PNG\r\n\x1a\n-fake-image-bytes"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LOCALHARNESS_COMFYUI_URL", "LOCALHARNESS_COMFYUI_WORKFLOW",
                "LOCALHARNESS_COMFYUI_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gi, "_POLL_S", 0.01)


class _Resp:
    def __init__(self, json_data=None, content=b"", status_code=200, text=""):
        self._json = json_data
        self.content = content
        self.status_code = status_code
        self.text = text or (json.dumps(json_data) if json_data is not None else "")

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(self.text, request=None, response=None)  # type: ignore[arg-type]


def _done(pid="p1", filename="lh_x_00001_.png"):
    return {pid: {
        "status": {"status_str": "success", "completed": True},
        "outputs": {"459": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}},
    }}


def _fake_comfy(monkeypatch, *, history=None, post_json=None, post_status=200,
                connect_error=False, view_content=PNG):
    """Patch gi.httpx.AsyncClient. `history` is a sequence of /history payloads, one per poll
    (empty/exhausted -> {}, i.e. still running). Returns a recorder of what was sent."""
    rec = {"posted": None, "view_params": None, "polls": 0}
    seq = list(history or [])

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        async def post(self, url, json=None):
            if connect_error:
                raise httpx.ConnectError("connection refused")
            rec["posted"] = json
            return _Resp(json_data=post_json if post_json is not None else {"prompt_id": "p1"},
                         status_code=post_status)

        async def get(self, url, params=None):
            if "/history/" in url:
                rec["polls"] += 1
                return _Resp(json_data=seq.pop(0) if seq else {})
            rec["view_params"] = params
            return _Resp(content=view_content)

    monkeypatch.setattr(gi.httpx, "AsyncClient", _Client)
    return rec


# --- module gating: registered only when the operator configured an endpoint ---

@pytest.mark.asyncio
async def test_not_registered_without_env():
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    assert registry.has("generate_image") is False


@pytest.mark.asyncio
async def test_registered_when_env_set(monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://127.0.0.1:8188")
    registry = ToolRegistry()
    await register_builtin_tools(registry)
    assert registry.has("generate_image") is True
    assert GenerateImageTool().info().group == "image"  # named group, never "other"


# --- happy path ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_saves_png_and_pins_the_graph(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    rec = _fake_comfy(monkeypatch, history=[{}, _done()])
    r = await GenerateImageTool(workspace_root=str(tmp_path)).run(
        prompt="a red fox on snow", width=1000, height=512, steps=8, seed=42)
    assert r.success is True, r.error
    # file landed in the harness-owned artifacts dir with the reported bytes
    dest = Path(r.metadata["path"])
    assert dest.parent == tmp_path / ".localharness" / "artifacts" / "images"
    assert dest.read_bytes() == PNG
    assert r.metadata["image_id"].startswith("img-") and r.metadata["image_id"] in dest.name
    # honest reporting: snapped size (1000 -> 992), seed, steps, path all in the text
    for needle in ("992x512", "seed 42", "8 steps", str(dest)):
        assert needle in r.output
    # the POSTed graph: substituted values are real ints/strings, comment keys stripped
    graph = rec["posted"]["prompt"]
    assert "_comment" not in graph
    assert graph["452"]["inputs"]["prompt"] == "a red fox on snow"
    assert graph["456"]["inputs"]["width"] == 992
    assert graph["456"]["inputs"]["height"] == 512
    assert graph["458"]["inputs"]["seed"] == 42
    assert graph["458"]["inputs"]["steps"] == 8
    # the known Qwen-Image corruption gotcha stays pinned in the shipped template
    assert graph["451"]["inputs"]["weight_dtype"] == "default"
    # the produced file was fetched via /view with the server-reported name
    assert rec["view_params"]["filename"] == "lh_x_00001_.png"


# --- failure modes ------------------------------------------------------------

@pytest.mark.asyncio
async def test_unreachable_server_names_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://127.0.0.1:8188")
    _fake_comfy(monkeypatch, connect_error=True)
    r = await GenerateImageTool(workspace_root=str(tmp_path)).run(prompt="x")
    assert r.success is False
    assert "unreachable" in r.error and "http://127.0.0.1:8188" in r.error


@pytest.mark.asyncio
async def test_generation_error_status_surfaces_server_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    _fake_comfy(monkeypatch, history=[
        {"p1": {"status": {"status_str": "error", "completed": False,
                           "messages": [["execution_error", {"node_type": "KSampler"}]]}}},
    ])
    r = await GenerateImageTool(workspace_root=str(tmp_path)).run(prompt="x")
    assert r.success is False
    assert "generation failed" in r.error and "KSampler" in r.error


@pytest.mark.asyncio
async def test_deadline_exceeded_is_a_clear_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_TIMEOUT_S", "0.05")
    _fake_comfy(monkeypatch, history=[])  # never completes
    r = await GenerateImageTool(workspace_root=str(tmp_path)).run(prompt="x")
    assert r.success is False
    assert "not finished after" in r.error


@pytest.mark.asyncio
async def test_rejected_workflow_surfaces_http_body(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    _fake_comfy(monkeypatch, post_json={"error": "invalid prompt"}, post_status=400)
    r = await GenerateImageTool(workspace_root=str(tmp_path)).run(prompt="x")
    assert r.success is False
    assert "rejected" in r.error and "invalid prompt" in r.error


@pytest.mark.asyncio
async def test_empty_prompt_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    r = await GenerateImageTool(workspace_root=str(tmp_path)).run(prompt="   ")
    assert r.success is False and r.error_type == "validation_error"


@pytest.mark.asyncio
async def test_unconfigured_bare_tool_errors_cleanly():
    r = await GenerateImageTool().run(prompt="x")
    assert r.success is False
    assert "LOCALHARNESS_COMFYUI_URL" in r.error


# --- template contract (the model-agnostic seam) -------------------------------

def test_shipped_template_loads_fills_and_leaves_no_placeholders():
    template, name = load_template()
    assert name == "qwen-image-2.1"
    graph = build_graph(template, prompt="p", width=1024, height=1024,
                        steps=30, seed=1, prefix="lh_test")
    assert "__LH_" not in json.dumps(graph)


def test_template_without_prompt_slot_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / "no-prompt.json"
    bad.write_text(json.dumps({"1": {"class_type": "X", "inputs": {"text": "static"}}}))
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_WORKFLOW", str(bad))
    template, _ = load_template()
    with pytest.raises(ValueError, match="__LH_PROMPT__"):
        build_graph(template, prompt="p", width=8, height=8, steps=1, seed=1, prefix="x")


@pytest.mark.asyncio
async def test_env_template_override_swaps_the_model(monkeypatch, tmp_path):
    other = tmp_path / "flux-schnell.json"
    other.write_text(json.dumps({
        "_comment": "another model entirely",
        "1": {"class_type": "FluxLoader", "inputs": {"ckpt": "flux.safetensors"}},
        "2": {"class_type": "FluxSample", "inputs": {"model": ["1", 0], "prompt": "__LH_PROMPT__",
                                                     "seed": "__LH_SEED__"}},
    }))
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_WORKFLOW", str(other))
    rec = _fake_comfy(monkeypatch, history=[_done()])
    r = await GenerateImageTool(workspace_root=str(tmp_path)).run(prompt="hi", seed=7)
    assert r.success is True, r.error
    assert r.metadata["template"] == "flux-schnell"
    graph = rec["posted"]["prompt"]
    assert graph["1"]["class_type"] == "FluxLoader" and "_comment" not in graph
    assert graph["2"]["inputs"]["prompt"] == "hi" and graph["2"]["inputs"]["seed"] == 7
