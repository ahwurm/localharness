"""autoresearch's setup action: once `plugins enable autoresearch` has written the proposer's address
and model, one GET <base_url>/models (3 s) checks that the proposer answers and serves that model.
It spends nothing on an OpenAI-compatible server and runs only in the step: doctor stays offline (it
may be a paid API). The api key is sent, never shown.

The proposer answers through autoresearch.plugin._TRANSPORT (an httpx.MockTransport): no socket."""
from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest
import yaml

from localharness.autoresearch import plugin as autoresearch_plugin
from localharness.autoresearch.plugin import AutoresearchPlugin
from localharness.core.bus import EventBus
from localharness.plugins.api import Check, PluginContext, PluginPaths
from localharness.tools.registry import ToolRegistry

URL = "http://p.test/v1"
_PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "test-model"}
_PROPOSER = {"base_url": URL, "model": "p-model", "api_key": "sk-SENTINEL"}
SERVES = {"object": "list", "data": [{"id": "p-model", "object": "model"}]}


def _ctx(tmp_path: Path, proposer: dict | None = None, text: str | None = None) -> PluginContext:
    g = tmp_path / "g"
    g.mkdir()
    data = {"version": "1", "provider": _PROVIDER, **({"proposer": proposer} if proposer else {})}
    (g / "config.yaml").write_text(text if text is not None else yaml.safe_dump(data, sort_keys=False),
                                   encoding="utf-8")
    return PluginContext(bus=EventBus(), tools=ToolRegistry(), hooks=None, config=None, agent_config=None,
                         paths=PluginPaths(global_config_dir=g, workspace=None, state_dir=g), llm=None)


@pytest.fixture
def server(monkeypatch):
    """The proposer behind _TRANSPORT: every request recorded; `answer[0]` builds the response."""
    seen: list[httpx.Request] = []
    answer = [lambda request: httpx.Response(200, json=SERVES)]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer[0](request)

    monkeypatch.setattr(autoresearch_plugin, "_TRANSPORT", httpx.MockTransport(handler))
    return seen, answer


def _act(ctx: PluginContext) -> list[Check]:
    rows = AutoresearchPlugin().setup_action(ctx)
    assert "sk-SENTINEL" not in repr(rows)  # the key is sent, never shown
    return rows


@pytest.mark.parametrize("config", ["no proposer", "unreadable"])
def test_nothing_to_check_without_a_readable_proposer(tmp_path, server, config) -> None:
    ctx = _ctx(tmp_path, text="provider: [unclosed\n" if config == "unreadable" else None)
    assert _act(ctx) == []
    assert server[0] == []


def test_a_proposer_that_serves_the_model_passes(tmp_path, server) -> None:
    seen, _ = server
    rows = _act(_ctx(tmp_path, _PROPOSER))

    assert rows == [Check(name="autoresearch-proposer", status="pass",
                          detail=f"the proposer answers at {URL} and serves p-model")]
    [request] = seen
    assert (request.method, str(request.url)) == ("GET", f"{URL}/models")
    assert request.headers["Authorization"] == "Bearer sk-SENTINEL"


@pytest.mark.parametrize("body", [{"object": "list", "data": [{"id": "other-model"}]}, b"not json", []],
                         ids=["other model", "not json", "not an object"])
def test_an_answer_that_does_not_name_the_model_warns(tmp_path, server, body) -> None:
    server[1][0] = lambda request: (httpx.Response(200, content=body) if isinstance(body, bytes)
                                    else httpx.Response(200, json=body))
    [row] = _act(_ctx(tmp_path, _PROPOSER))

    assert (row.name, row.status) == ("autoresearch-proposer", "warn")
    assert "p-model" in row.detail and URL in row.detail and row.hint


def test_a_refusal_is_one_failing_row_naming_the_key_setting(tmp_path, server) -> None:
    server[1][0] = lambda request: httpx.Response(401, json={"error": "bad key sk-SENTINEL"})
    [row] = _act(_ctx(tmp_path, _PROPOSER))

    assert (row.name, row.status, row.detail) == (
        "autoresearch-proposer", "fail", f"the proposer at {URL} answered 401")
    assert "proposer.api_key" in row.hint


def test_no_answer_is_one_failing_row(tmp_path, server) -> None:
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    server[1][0] = refuse
    [row] = _act(_ctx(tmp_path, _PROPOSER))

    assert (row.name, row.status, row.detail) == (
        "autoresearch-proposer", "fail", f"no answer from {URL}/models (ConnectError)")
    assert "proposer.base_url" in row.hint


def test_no_key_sends_no_authorization_header(tmp_path, server) -> None:
    seen, _ = server
    _act(_ctx(tmp_path, {"base_url": URL + "/", "model": "p-model"}))  # api_key defaults to "none"
    [request] = seen
    assert "Authorization" not in request.headers
    assert str(request.url) == f"{URL}/models"


def test_doctor_stays_offline(tmp_path, monkeypatch, server) -> None:
    """Never a request from doctor: the same transport and a socket guard both stay unused."""
    calls: list = []
    monkeypatch.setattr(socket.socket, "connect", lambda *a: calls.append(a) or (_ for _ in ()).throw(
        OSError("doctor opened a socket")))
    rows = AutoresearchPlugin().doctor(_ctx(tmp_path, _PROPOSER))

    assert rows == [Check(name="autoresearch", status="pass", detail=f"proposer: p-model at {URL}")]
    assert server[0] == [] and calls == []
