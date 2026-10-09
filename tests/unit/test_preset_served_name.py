"""PRESET-03, the writer half: `localharness model <alias>` and `/model` persistence never write a
llama.cpp alias into `server.model` (D-05).

For vLLM, Ollama and LM Studio `server.model` is the served name the strategies warm-load, so a cold
start relaunches the chosen model (#34). For llama.cpp `server.model` is the GGUF path and the alias is
`served_name`, so the alias must never reach `-m`. Both write sites are held: the overlay on disk and the
live in-memory record (a miss on the second leaves the running session building `-m <alias>`).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.cli.model_ops import persist_default_model
from localharness.config.loader import ConfigLoader
from localharness.provider import server as provider_server

ALIAS = "qwen-27b-swift-1.5"

_BASE = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8080/v1",
        "default_model": "global-model",
        "available_models": ["global-model"],
    },
}

LLAMACPP = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/x/m.gguf",
            "served_name": ALIAS, "port": 8080}
OTHER_RUNTIMES = {
    "vllm": {"runtime": "vllm", "launch": "docker", "docker_image": "img", "model": "old-model", "port": 8081},
    "ollama": {"runtime": "ollama", "model": "old-model", "port": 11434},
    "lmstudio": {"runtime": "lmstudio", "model": "old-model", "port": 1234},
}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _global_dir(tmp_path: Path, **sections) -> Path:
    g = tmp_path / "global"
    _write(g / "config.yaml", {**_BASE, **sections})
    return g


def _overlay(g: Path) -> dict:
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_llama_cpp_alias_never_lands_in_server_model(tmp_path):
    g = _global_dir(tmp_path, server=LLAMACPP)
    harness = ConfigLoader(config_dir=g).load_harness()

    assert await persist_default_model(harness, ALIAS, config_dir=g) is None

    overlay = _overlay(g)
    assert overlay["provider"]["default_model"] == ALIAS
    assert overlay["org"]["default_model"] == ALIAS
    assert "server" not in overlay, "the overlay must not carry a server.model for llama.cpp"
    assert harness.server.model == "/x/m.gguf", "the in-memory record keeps the GGUF path"


@pytest.mark.asyncio
async def test_the_next_launch_still_passes_the_gguf_path_to_dash_m(tmp_path):
    g = _global_dir(tmp_path, server=LLAMACPP)
    harness = ConfigLoader(config_dir=g).load_harness()
    await persist_default_model(harness, ALIAS, config_dir=g)

    live = provider_server.serve_command(harness.server)
    assert live[live.index("-m") + 1] == "/x/m.gguf"
    assert ALIAS not in live[live.index("-m"):live.index("-m") + 2]

    reloaded = ConfigLoader(config_dir=g).load_harness()
    assert reloaded.server.model == "/x/m.gguf"
    cold = provider_server.serve_command(reloaded.server)
    assert cold[cold.index("-m") + 1] == "/x/m.gguf"


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", sorted(OTHER_RUNTIMES))
async def test_vllm_ollama_and_lm_studio_keep_writing_server_model(tmp_path, runtime):
    g = _global_dir(tmp_path, server=OTHER_RUNTIMES[runtime])
    harness = ConfigLoader(config_dir=g).load_harness()

    assert await persist_default_model(harness, "next-model", config_dir=g) is None

    assert _overlay(g)["server"]["model"] == "next-model"
    assert harness.server.model == "next-model"
    assert ConfigLoader(config_dir=g).load_harness().server.model == "next-model"


@pytest.mark.asyncio
async def test_with_no_server_block_no_server_key_is_written(tmp_path):
    g = _global_dir(tmp_path)
    harness = ConfigLoader(config_dir=g).load_harness()
    assert harness.server is None

    await persist_default_model(harness, "next-model", config_dir=g)

    overlay = _overlay(g)
    assert "server" not in overlay
    assert overlay["provider"]["default_model"] == "next-model"


@pytest.mark.asyncio
async def test_a_non_ascii_alias_lands_in_the_provider_and_org_defaults_only(tmp_path):
    g = _global_dir(tmp_path, server=LLAMACPP)
    harness = ConfigLoader(config_dir=g).load_harness()

    await persist_default_model(harness, "qwen-äö", config_dir=g)

    overlay = _overlay(g)
    assert overlay["provider"]["default_model"] == "qwen-äö"
    assert overlay["org"]["default_model"] == "qwen-äö"
    assert "server" not in overlay
    assert harness.server.model == "/x/m.gguf"
    assert "qwen-äö" in overlay["provider"]["available_models"]
