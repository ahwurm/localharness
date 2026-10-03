"""autoresearch's setup action: once `plugins enable autoresearch` has written the proposer's address
and model, one GET <base_url>/models (3 s) checks that the proposer answers and serves that model.
It spends nothing on an OpenAI-compatible server and runs only in the step: doctor stays offline (it
may be a paid API). The api key is sent, never shown.

The proposer answers through autoresearch.plugin._TRANSPORT (an httpx.MockTransport): no socket."""
from __future__ import annotations

import dataclasses
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


# --- a project file never chooses where the key is sent ---------------------------------------------
# A cloned repo's .localharness/ loads without a prompt when you stand in it, and its overrides.yaml
# outranks the machine's, where the step writes the address. The check reads what `propose` reads:
# the machine-level config only.

EVIL = "http://evil.test/v1"
IGNORED = f"a project file sets proposer.base_url to {EVIL!r}; it is ignored here and by `propose`"
PASSES = Check(name="autoresearch-proposer", status="pass",
               detail=f"the proposer answers at {URL} and serves p-model")


def _in_project(tmp_path: Path, proposer: dict | None, project: dict,
                file: str = "config.yaml") -> PluginContext:
    """`_ctx`, standing in a project whose .localharness/<file> holds `project`."""
    ctx = _ctx(tmp_path, proposer)
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    (ws / file).write_text(yaml.safe_dump(project), encoding="utf-8")
    return dataclasses.replace(ctx, paths=PluginPaths(global_config_dir=ctx.paths.global_config_dir,
                                                      workspace=ws, state_dir=ws))


@pytest.mark.parametrize("file", ["config.yaml", "overrides.yaml"])
def test_a_project_file_cannot_choose_where_the_key_is_sent(tmp_path, server, file) -> None:
    seen, _ = server
    rows = _act(_in_project(tmp_path, _PROPOSER, {"proposer": {"base_url": EVIL}}, file))

    assert [(r.method, str(r.url), r.headers.get("Authorization")) for r in seen] == [
        ("GET", f"{URL}/models", "Bearer sk-SENTINEL")]
    assert rows == [PASSES, Check(name="autoresearch-proposer", status="warn", detail=IGNORED)]


def test_the_check_uses_the_machine_key_and_model_as_propose_does(tmp_path, server) -> None:
    seen, _ = server
    rows = _act(_in_project(tmp_path, _PROPOSER, {"proposer": {"api_key": "sk-PROJECT", "model": "ws-model"}}))

    [request] = seen
    assert request.headers["Authorization"] == "Bearer sk-SENTINEL"
    assert rows == [PASSES] and "sk-PROJECT" not in repr(rows)


def test_a_project_restating_the_address_adds_no_note(tmp_path, server) -> None:
    assert _act(_in_project(tmp_path, _PROPOSER, {"proposer": {"base_url": URL}})) == [PASSES]


def test_a_proposer_only_a_project_file_sets_is_never_contacted(tmp_path, server) -> None:
    rows = _act(_in_project(tmp_path, None, {"proposer": {"base_url": EVIL, "model": "p-model"}}))

    assert server[0] == []
    assert rows == [Check(name="autoresearch-proposer", status="warn", detail=IGNORED)]


@pytest.mark.parametrize(("proposer", "host", "line"), [
    (_PROPOSER, "p.test", "Contacting the proposer at http://p.test (sending proposer.api_key) …"),
    ({"base_url": "http://good.test@evil.test:8001/v1", "model": "p-model"}, "evil.test",
     "Contacting the proposer at http://evil.test:8001 …"),
], ids=["with its key", "the host it really contacts"])
def test_the_host_is_named_before_the_request_goes(tmp_path, server, capsys, proposer, host, line) -> None:
    seen, answer = server
    printed_by_then: list[str] = []
    answer[0] = lambda request: printed_by_then.append(capsys.readouterr().out) or httpx.Response(200, json=SERVES)

    _act(_ctx(tmp_path, proposer))

    assert printed_by_then == [line + "\n"]
    assert [r.url.host for r in seen] == [host]
