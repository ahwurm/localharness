"""memory's setup step: no questions, one action — put the embedding model in the local Hugging Face
cache so the first memory search does not stop to download it. The action asks first (core asks
manifest.setup_action), does nothing when the model is a local path or already cached, and without
the sentence_transformers package downloads nothing and names the install line.

The download is always the fake: provider.server.download_model is patched in every test that can
reach it, so no test touches the network. This module opts out of conftest's suite-wide
"model cached" patch: each test says what the cache holds."""
from __future__ import annotations

import importlib.util
import subprocess
import sys

import huggingface_hub
import pytest
import yaml

from localharness.config.loader import ConfigLoader
from localharness.core.bus import EventBus
from localharness.memory import plugin as memory_plugin
from localharness.memory.config import MemoryConfig
from localharness.memory.plugin import MemoryPlugin
from localharness.plugins.api import AGENT_PROMPT_PLACEHOLDER, Check, PluginContext, PluginPaths
from localharness.plugins.lifecycle import setup_action_rows
from localharness.plugins.resolve import resolve
from localharness.tools.registry import ToolRegistry

MODEL = "Qwen/Qwen3-Embedding-0.6B"
NOT_CACHED = Check(name="memory-embedding", status="fail",
                   detail=f"embedding model {MODEL} is not in the local Hugging Face cache")
CACHED = Check(name="memory-embedding", status="pass", detail=f"embedding model {MODEL} is in the local cache")


def _ctx(tmp_path, agent_config=None) -> PluginContext:
    return PluginContext(bus=EventBus(), tools=ToolRegistry(), hooks=None, config=None,
                         agent_config=agent_config if agent_config is not None else MemoryConfig(),
                         paths=PluginPaths(tmp_path, None, tmp_path), llm=None)


@pytest.fixture
def downloads(monkeypatch) -> list[str]:
    """Every repo the (fake) core downloader was asked for."""
    from localharness.provider import server
    got: list[str] = []
    monkeypatch.setattr(server, "download_model", lambda repo_id: got.append(repo_id) or f"/cache/{repo_id}")
    return got


def _given(monkeypatch, *, package: bool, check: Check | None = None) -> None:
    monkeypatch.setattr(memory_plugin, "_embedding_package_installed", lambda: package)
    if check is not None:
        monkeypatch.setattr(memory_plugin, "_embedding_check", lambda model: check)


def test_the_manifest_declares_the_question_and_the_next_step() -> None:
    m = MemoryPlugin.manifest
    assert m.setup_action == "Download the embedding model now (about 1.2 GB)?"
    assert m.next_steps == "In a session, /memory shows what it keeps."
    assert m.agent_prompt and not AGENT_PROMPT_PLACEHOLDER.search(m.agent_prompt)
    assert m.setup == ()  # no questions (R6)
    assert MemoryPlugin.ConfigModel is None


def test_without_the_package_it_downloads_nothing_and_names_the_install_line(tmp_path, monkeypatch,
                                                                             downloads) -> None:
    _given(monkeypatch, package=False, check=NOT_CACHED)
    [row] = MemoryPlugin().setup_action(_ctx(tmp_path))

    assert (row.name, row.status) == ("memory-embedding", "warn")
    assert row.detail == "nothing downloaded: the sentence_transformers package is not installed"
    assert "uv tool install 'localharness[embeddings]'" in row.hint
    assert downloads == []


def test_a_cached_model_is_not_downloaded_again(tmp_path, monkeypatch, downloads) -> None:
    _given(monkeypatch, package=True, check=CACHED)
    assert MemoryPlugin().setup_action(_ctx(tmp_path)) == []
    assert downloads == []


def test_a_missing_model_is_downloaded_once_through_the_core_downloader(tmp_path, monkeypatch,
                                                                       downloads) -> None:
    _given(monkeypatch, package=True, check=NOT_CACHED)
    rows = MemoryPlugin().setup_action(_ctx(tmp_path))

    assert downloads == [MODEL]
    assert rows == [Check(name="memory-embedding", status="pass", detail=f"downloaded {MODEL}")]


def test_the_configured_model_is_the_one_downloaded(tmp_path, monkeypatch, downloads) -> None:
    asked: list[str] = []
    monkeypatch.setattr(memory_plugin, "_embedding_package_installed", lambda: True)
    monkeypatch.setattr(memory_plugin, "_embedding_check",
                        lambda model: asked.append(model) or NOT_CACHED)
    rows = MemoryPlugin().setup_action(_ctx(tmp_path, MemoryConfig(embedding_model="org/other-embed")))

    assert asked == ["org/other-embed"] and downloads == ["org/other-embed"]
    assert rows == [Check(name="memory-embedding", status="pass", detail="downloaded org/other-embed")]


def test_a_model_that_is_a_local_path_downloads_nothing(tmp_path, monkeypatch, downloads) -> None:
    local = tmp_path / "my-embedder"
    local.mkdir()
    _given(monkeypatch, package=True, check=NOT_CACHED)  # the cache lookup would say "missing"
    assert MemoryPlugin().setup_action(_ctx(tmp_path, MemoryConfig(embedding_model=str(local)))) == []
    assert downloads == []


async def test_a_failed_download_is_one_failing_row(tmp_path, monkeypatch) -> None:
    """Through the harness's own contained call, as `plugins enable memory` runs it."""
    from localharness.provider import server

    def boom(repo_id):
        raise OSError("disk full")

    _given(monkeypatch, package=True, check=NOT_CACHED)
    monkeypatch.setattr(server, "download_model", boom)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "version": "1",
        "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                     "default_model": "test-model", "available_models": ["test-model"]}}),
        encoding="utf-8")
    resolution = resolve(ConfigLoader(config_dir=tmp_path))
    rows = await setup_action_rows(resolution, "memory", PluginPaths(tmp_path, None, tmp_path))

    assert [(r.status, f"{r.name}: {r.detail}") for r in rows] == [
        ("fail", "memory: its setup action raised OSError: disk full")]


@pytest.mark.parametrize(("package", "cached"), [(False, False), (True, False), (True, True)])
def test_the_check_fails_exactly_while_the_action_has_work(tmp_path, monkeypatch, downloads,
                                                           package, cached) -> None:
    """The step runs the action only while doctor's row fails (52-03's gate), so the REAL check and
    the REAL action are read here against one faked cache: the row fails exactly when the action
    has work to do or to report."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (object() if package else None)
                        if name == "sentence_transformers" else real(name, *a))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename, **kw: f"/cache/{repo_id}/{filename}" if cached else None)
    [row] = [c for c in MemoryPlugin().doctor(_ctx(tmp_path)) if c.name == "memory-embedding"]
    rows = MemoryPlugin().setup_action(_ctx(tmp_path))

    assert (row.status == "fail") is (rows != [])
    assert downloads == ([MODEL] if package and not cached else [])


def test_importing_the_plugin_loads_no_downloader() -> None:
    code = ("import sys, localharness.memory.plugin; print([m for m in "
            "('localharness.provider.server', 'huggingface_hub') if m in sys.modules])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
