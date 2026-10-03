"""A start reaches nothing but the model server (orchestrator ruling R8).

Three things a start used to do on its own: build tiktoken's cl100k vocabulary (fetched from an
OpenAI-hosted URL on a cold cache) for an estimator most setups never use; import the whole MCP
stack (mcp, starlette, uvicorn) with no MCP server configured; and let memory's first embed fetch
a 1.2 GB model without a word. Now the vocabulary is built on the first count that needs it and a
labelled byte estimate stands in when it cannot be had; the MCP stack is imported only for a server
that is configured; a missing embedding model costs one summary line and no wait."""
from __future__ import annotations

import logging
import math
import sys
import types

import pytest

from localharness.agent.context import APPROX_TOKENIZE_SAFETY_FACTOR, TokenCounter
from localharness.memory.plugin import _embedding_check as _REAL_EMBEDDING_CHECK  # before conftest's stub

NOTICE = ("memory: the embedding model Qwen/Qwen3-Embedding-0.6B is not on this machine yet — it "
          "downloads once from huggingface.co (about 1.2 GB) the first time memory needs it")


def _fake_tiktoken(monkeypatch, get_encoding) -> list[str]:
    """A tiktoken whose get_encoding is `get_encoding`; returns the names it was asked for."""
    asked: list[str] = []

    def recorded(name):
        asked.append(name)
        return get_encoding(name)

    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=recorded))
    return asked


def _no_gguf(monkeypatch):
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.load_gguf_tokenizer", lambda model, ptype: None)


# --------------------------------------------------------------------------- the token counter


def test_without_tiktoken_a_count_is_a_labelled_byte_estimate(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # `import tiktoken` raises ImportError
    counter = TokenCounter()

    with caplog.at_level(logging.WARNING, logger="localharness.agent.context"):
        assert counter.count("hello world") == math.ceil(len(b"hello world") / 3)
        assert counter.count("another text") > 0

    notices = [r for r in caplog.records if "byte estimate" in r.getMessage()]
    assert len(notices) == 1, "the estimate is labelled once"


def test_a_vllm_counter_builds_no_vocabulary_at_start(monkeypatch):
    asked = _fake_tiktoken(monkeypatch, lambda name: pytest.fail("built a vocabulary"))
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: 7)
    monkeypatch.setattr(TokenCounter, "_remote_count_messages", lambda self, messages, tools: 9)

    counter = TokenCounter(base_url="http://127.0.0.1:9/v1", model="m", provider_type="vllm")

    assert counter.mode == "vllm"
    assert counter.count("anything") == 7
    assert asked == []


def test_nothing_builds_a_vocabulary_at_construction(monkeypatch):
    asked = _fake_tiktoken(monkeypatch, lambda name: pytest.fail("built at construction"))
    TokenCounter()
    assert asked == []


def test_an_offline_vocabulary_falls_back_and_is_not_retried(monkeypatch):
    def offline(name):
        raise ConnectionError("openaipublic.blob.core.windows.net: no route to host")

    asked = _fake_tiktoken(monkeypatch, offline)
    counter = TokenCounter()

    assert counter.count("abcdef") == 2
    assert counter.count("ghijkl") == 2
    assert asked == ["cl100k_base"], "one attempt, never one per count"


def test_with_tiktoken_the_first_count_builds_cl100k_once_and_counts_as_before(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    real = tiktoken.get_encoding("cl100k_base")  # the old eager implementation's encoder
    sample = "The quick brown fox — 12345 tokens? <|endoftext|> fn(x) => x * 2"
    expected = len(real.encode(sample, disallowed_special=()))
    asked = _fake_tiktoken(monkeypatch, tiktoken.get_encoding)

    counter = TokenCounter()
    assert asked == []
    assert counter.count(sample) == expected
    assert counter.count("a second text") > 0
    assert asked == ["cl100k_base"]

    _no_gguf(monkeypatch)
    approximate = TokenCounter(base_url="http://127.0.0.1:9/v1", model="m", provider_type="ollama")
    assert approximate.approximate
    assert approximate.count(sample) == math.ceil(expected * APPROX_TOKENIZE_SAFETY_FACTOR)


def test_an_ollama_counter_with_no_gguf_and_no_tiktoken_still_counts(monkeypatch):
    """It used to refuse to start ("no counting source exists"); offline it now counts with the
    labelled, inflated byte estimate."""
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    _no_gguf(monkeypatch)

    counter = TokenCounter(base_url="http://127.0.0.1:9/v1", model="m", provider_type="ollama")

    assert counter.approximate
    assert counter.count("hello") == math.ceil(math.ceil(5 / 3) * APPROX_TOKENIZE_SAFETY_FACTOR)


def test_an_unknown_runtime_with_no_tokenize_and_no_tiktoken_still_counts(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: None)

    counter = TokenCounter(base_url="http://127.0.0.1:9/v1", model="m", provider_type="openai")

    assert counter.approximate
    assert counter.count("hello") > 0


def test_a_known_exact_runtime_still_refuses_an_approximate_fallback(monkeypatch):
    monkeypatch.setattr(TokenCounter, "_remote_count", lambda self, text: None)
    with pytest.raises(RuntimeError, match="Refusing an approximate fallback"):
        TokenCounter(base_url="http://127.0.0.1:9/v1", model="m", provider_type="vllm")


# --------------------------------------------------------------------------- the MCP stack


def _forget_mcp(monkeypatch) -> None:
    for name in [m for m in sys.modules if m == "mcp" or m.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "localharness.tools.mcp", raising=False)


async def test_a_stock_start_imports_no_mcp(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    _stub_start_boundaries(tmp_path, monkeypatch)
    _forget_mcp(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    assert "mcp" not in sys.modules
    assert "localharness.tools.mcp" not in sys.modules


async def test_a_start_with_an_mcp_server_still_starts_it(tmp_path, monkeypatch):
    import yaml

    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.start_cmd import _start_async
    from localharness.tools.mcp import MCPClientManager
    from tests.unit.test_start_cmd import _stub_start_boundaries

    agent = _build_agent_yaml("orchestrator", "General-purpose assistant", None)
    agent.setdefault("tools", {})["mcp_servers"] = [{"name": "files", "transport": "stdio",
                                                     "command": "true"}]
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "orchestrator.yaml").write_text(yaml.safe_dump(agent), encoding="utf-8")
    _stub_start_boundaries(tmp_path, monkeypatch)
    started: list = []

    async def startup(self, configs):
        started.extend(c.name for c in configs)
        return {c.name: 1 for c in configs}

    monkeypatch.setattr(MCPClientManager, "startup", startup)
    monkeypatch.setattr(MCPClientManager, "shutdown", lambda self: _done(), raising=False)

    await _start_async(None, False, False, str(tmp_path))

    assert started == ["files"]


async def _done():
    return None


# --------------------------------------------------------------------------- the embedding model


def _record_summary(monkeypatch, loads: list[str]) -> list[tuple[str, int]]:
    """Every line start prints, with how many embedding-model loads had happened when it did."""
    import localharness.cli.start_cmd as start_cmd

    printed: list[tuple[str, int]] = []
    monkeypatch.setattr(start_cmd.console, "print",
                        lambda *a, **k: printed.append((" ".join(str(x) for x in a), len(loads))))
    return printed


def _never_load(monkeypatch) -> list[str]:
    loads: list[str] = []

    def load(self):
        loads.append(self.model_name)
        raise RuntimeError("the embedding model was loaded")

    monkeypatch.setattr("localharness.memory.resonance.ResonanceEngine._ensure_loaded", load)
    return loads


async def test_a_start_never_waits_for_the_embedding_download(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from localharness.memory import plugin as memory_plugin
    from tests.unit.test_start_cmd import _stub_start_boundaries

    monkeypatch.setattr(memory_plugin, "_embedding_check", _REAL_EMBEDDING_CHECK)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(memory_plugin, "_embedding_package_installed", lambda: True)
    monkeypatch.setattr("importlib.util.find_spec",
                        _spec_found_for("sentence_transformers"))
    _stub_start_boundaries(tmp_path, monkeypatch)
    loads = _never_load(monkeypatch)
    printed = _record_summary(monkeypatch, loads)

    await _start_async(None, False, False, str(tmp_path))

    (summary, loads_before) = next(p for p in printed if "startup)" in p[0])
    assert NOTICE in summary
    assert loads_before == 0, "the banner waited for the embedding model"


async def test_a_cached_embedding_model_says_nothing(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    _stub_start_boundaries(tmp_path, monkeypatch)  # conftest: the model reads as cached
    printed = _record_summary(monkeypatch, _never_load(monkeypatch))

    await _start_async(None, False, False, str(tmp_path))

    summary = next(p[0] for p in printed if "startup)" in p[0])
    assert "embedding model" not in summary


def _spec_found_for(name: str):
    import importlib.util

    real = importlib.util.find_spec

    def find_spec(module, *a, **k):
        if module == name:
            return real("json")  # any real spec: "the package is importable"
        return real(module, *a, **k)

    return find_spec


@pytest.mark.parametrize("model", ["./models/not-here", "not a model id"])
def test_a_model_id_the_cache_cannot_read_is_no_notice_and_no_error(monkeypatch, model):
    """The cache lookup raises for a missing local path or a malformed id; the notice runs inside
    memory's start, so it must answer None rather than fail memory for the session."""
    from localharness.memory import plugin as memory_plugin

    monkeypatch.setattr(memory_plugin, "_embedding_check", _REAL_EMBEDDING_CHECK)
    monkeypatch.setattr(memory_plugin, "_embedding_package_installed", lambda: True)
    monkeypatch.setattr("importlib.util.find_spec", _spec_found_for("sentence_transformers"))

    assert memory_plugin._embedding_download_notice(model) is None
