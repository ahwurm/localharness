"""The model's own sampling applies unless a person set a temperature (spec 20 §7; 12-CONTEXT D-24..D-26, D-33).

The path from the config files to the wire, end to end: the loader resolves the root agent's
temperature (None when no rung set one), `start` hands it to both `LLMConfig` calls, and the three
request builders send `temperature` only when it is set. Nothing here opens a socket: the
AsyncOpenAI client is replaced by a scripted `create`, as in test_client_effort_keys.py.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import localharness
from localharness.config.loader import ConfigLoader
from localharness.provider.client import LLMClient, LLMConfig

SRC = Path(localharness.__file__).parent
MSGS = [{"role": "user", "content": "hi"}]
MODES = ("native", "xml", "text")

_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _seed(base: Path, subdir: str, name: str, **fields) -> None:
    _write_yaml(base / subdir / f"{name}.yaml", {"name": name, **fields})


def _global(tmp_path: Path, *, org: dict | None = None) -> Path:
    g = tmp_path / "global"
    cfg = dict(_MINIMAL)
    if org is not None:
        cfg["org"] = org
    _write_yaml(g / "config.yaml", cfg)
    return g


# ---------------------------------------------------------------------------
# The fake server: a request is recorded as the kwargs the SDK would receive
# ---------------------------------------------------------------------------


class _Server:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("stream"):
            return _Stream()
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


class _Stream:
    async def aclose(self) -> None:
        pass

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None),
                finish_reason="stop",
            )],
        )


def _client(mode: str, **cfg) -> tuple[LLMClient, _Server]:
    config = LLMConfig(
        base_url="http://127.0.0.1:0/v1", model="m", is_local=False, tool_call_mode=mode, **cfg,
    )
    with patch("localharness.provider.client.AsyncOpenAI"):
        client = LLMClient(config)
    server = _Server()
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=server.create))
    )
    return client, server


# ---------------------------------------------------------------------------
# Task 1: nothing set anywhere -> None all the way to the wire
# ---------------------------------------------------------------------------


def test_no_rung_set_the_root_loads_with_no_temperature(tmp_path):
    g = _global(tmp_path)
    _seed(g, "agents", "root", role="r")
    cfg = ConfigLoader(config_dir=g).load_agent("root")
    assert cfg.temperature is None


def test_an_init_written_null_is_no_rung(tmp_path):
    g = _global(tmp_path, org={"default_temperature": None})
    _seed(g, "agents", "root", role="r")
    assert ConfigLoader(config_dir=g).load_agent("root").temperature is None


def test_llm_config_default_temperature_is_none():
    assert LLMConfig(base_url="x", model="m").temperature is None


@pytest.mark.parametrize("mode", MODES)
async def test_an_unset_temperature_is_absent_from_every_request(mode):
    client, server = _client(mode)
    await client.complete(MSGS)
    await client.stream_complete(MSGS)
    assert len(server.calls) == 2
    assert all("temperature" not in call for call in server.calls)


@pytest.mark.parametrize("mode", MODES)
async def test_a_set_temperature_rides_every_request(mode):
    client, server = _client(mode, temperature=0.2)
    await client.complete(MSGS)
    await client.stream_complete(MSGS)
    assert [call["temperature"] for call in server.calls] == [0.2, 0.2]


async def test_the_xml_fallback_builder_carries_the_same_rule():
    """`_complete_xml_fallback` is the third builder: reached when a native request is rejected."""
    for temperature, present in ((None, False), (0.7, True)):
        client, server = _client("native", temperature=temperature)
        tool = {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}
        await client._complete_xml_fallback(MSGS, [tool], False)
        assert ("temperature" in server.calls[0]) is present
        if present:
            assert server.calls[0]["temperature"] == 0.7


def _llm_config_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Name) and n.func.id == "LLMConfig")
             or (isinstance(n.func, ast.Attribute) and n.func.attr == "LLMConfig"))
    ]


def test_every_llm_config_in_start_hands_over_the_resolved_root_temperature():
    calls = _llm_config_calls(SRC / "cli" / "start_cmd.py")
    assert len(calls) == 2
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "temperature" in kw
        assert ast.unparse(kw["temperature"]) == "agent_config.temperature"
