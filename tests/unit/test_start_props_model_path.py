"""A llama.cpp attach records the weights path `/props` reports (Phase 5, D-08, D-42).

`start` reads `GET /props` once at every llama.cpp attach, with the bearer `probe_window` used, off the
event loop, and stores `model_path` with `speed_stats.record_model_path` under (llamacpp, resolved model),
so `init`'s ranking can find the file that served before. Every failure is silent. Nothing here dials a
port: `httpx.get` is faked for the helper's own tests, and the start tests fake the helper itself.
"""
from __future__ import annotations

import httpx
import pytest

from localharness.cli import start_cmd
from localharness.cli.start_cmd import _props_model_path as props_model_path  # the real one, held before conftest's stub
from localharness.provider.speed_stats import model_path
from tests.unit.test_window_lines_start import (  # noqa: F401 — `world` is a fixture
    URL,
    Launching,
    World,
    _launch_report,
    world,
)

BASE = "http://127.0.0.1:8080/v1"


class _Response:
    def __init__(self, body=None, *, raises: Exception | None = None):
        self._body, self._raises = body, raises

    def json(self):
        if self._raises is not None:
            raise self._raises
        return self._body


def _fake_get(monkeypatch, response=None, *, raises: Exception | None = None) -> list[tuple]:
    calls: list[tuple] = []

    def get(url, **kw):
        calls.append((url, kw))
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(httpx, "get", get)
    return calls


# ------------------------------------------------------------------ the helper


def test_the_path_is_read_from_props_with_the_bearer_probe_window_uses(monkeypatch):
    calls = _fake_get(monkeypatch, _Response({"model_path": "/m/a.gguf", "n_ctx": 8192}))

    assert props_model_path(BASE, {"api_key": "K"}) == "/m/a.gguf"

    ((url, kw),) = calls
    assert url == "http://127.0.0.1:8080/props"
    assert kw["headers"] == {"Authorization": "Bearer K"} and kw["timeout"] == 3.0


def test_no_key_sends_no_header(monkeypatch):
    calls = _fake_get(monkeypatch, _Response({"model_path": "/m/a.gguf"}))

    assert props_model_path(BASE, {}) == "/m/a.gguf"

    assert not calls[0][1].get("headers")


def test_a_base_url_without_v1_and_a_trailing_slash_resolves_the_same_props(monkeypatch):
    calls = _fake_get(monkeypatch, _Response({"model_path": "/m/a.gguf"}))

    props_model_path("http://127.0.0.1:8080/", {})
    props_model_path("http://127.0.0.1:8080/v1/", {})

    assert [c[0] for c in calls] == ["http://127.0.0.1:8080/props"] * 2


@pytest.mark.parametrize("body", [
    {}, {"model_path": None}, {"model_path": 7}, {"model_path": ["/m/a.gguf"]}, {"model_path": ""},
    ["/m/a.gguf"], "/m/a.gguf", None,
])
def test_a_body_without_a_string_model_path_reads_none(monkeypatch, body):
    _fake_get(monkeypatch, _Response(body))

    assert props_model_path(BASE, {}) is None


@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("slow"), httpx.ConnectError("refused"), OSError("down"), RuntimeError("anything"),
])
def test_a_request_failure_reads_none(monkeypatch, failure):
    _fake_get(monkeypatch, raises=failure)

    assert props_model_path(BASE, {}) is None


def test_a_non_json_body_reads_none(monkeypatch):
    _fake_get(monkeypatch, _Response(raises=ValueError("not json")))

    assert props_model_path(BASE, {}) is None


# ------------------------------------------------------------------ through the real start


def _fake_helper(monkeypatch, answer: str | None) -> list[tuple]:
    calls: list[tuple] = []

    def helper(base_url, launch_key, **kw):
        calls.append((base_url, dict(launch_key)))
        return answer

    monkeypatch.setattr(start_cmd, "_props_model_path", helper)
    return calls


async def test_a_llamacpp_attach_records_the_path_beside_the_alias(world, monkeypatch):
    calls = _fake_helper(monkeypatch, "/m/a.gguf")

    await world.start()

    assert model_path(world.ledger, "llamacpp", "qwen") == "/m/a.gguf"
    assert calls == [(URL, {})]


async def test_a_keyed_launch_record_hands_the_helper_the_same_key_the_probe_used(world, monkeypatch):
    calls = _fake_helper(monkeypatch, "/m/a.gguf")
    world.config(require_api_key=True)
    cfg = world.tmp_path / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("api_key: none", "api_key: SECRET"), encoding="utf-8")

    await world.start()

    assert calls and calls[0][1] == {"api_key": "SECRET"}
    assert world.probe_calls[0][3] == "SECRET"


async def test_a_report_with_no_path_records_nothing_and_the_start_succeeds(world, monkeypatch):
    _fake_helper(monkeypatch, None)

    await world.start()

    assert model_path(world.ledger, "llamacpp", "qwen") is None
    assert world.last().served == 65536  # the window record is made as ever


async def test_the_path_is_recorded_on_every_attach_so_a_changed_file_replaces_it(world, monkeypatch):
    _fake_helper(monkeypatch, "/m/a.gguf")
    await world.start()
    _fake_helper(monkeypatch, "/m/b.gguf")

    await world.start()

    assert model_path(world.ledger, "llamacpp", "qwen") == "/m/b.gguf"


@pytest.mark.parametrize("provider_type", ["vllm", "ollama"])
async def test_another_runtime_never_calls_it(tmp_path, monkeypatch, provider_type):
    w = World(tmp_path, monkeypatch)
    w.config(provider_type=provider_type, record=False)
    calls = _fake_helper(monkeypatch, "/m/a.gguf")

    await w.start()

    assert calls == [] and model_path(w.ledger, provider_type, "qwen") is None


async def test_a_start_that_launched_the_server_does_not_read_props_from_the_attach_branch(world, monkeypatch):
    Launching(world, _launch_report())
    calls = _fake_helper(monkeypatch, "/m/a.gguf")

    await world.start()

    assert calls == [] and model_path(world.ledger, "llamacpp", "qwen") is None


async def test_a_ledger_that_cannot_be_written_never_fails_the_start(world, monkeypatch):
    _fake_helper(monkeypatch, "/m/a.gguf")

    def boom(*a, **kw):
        raise OSError("read-only")

    monkeypatch.setattr("localharness.provider.speed_stats.record_model_path", boom)

    await world.start()  # no exception


async def test_the_read_runs_off_the_event_loop(world, monkeypatch):
    import threading

    main = threading.get_ident()
    seen: list[int] = []

    def helper(base_url, launch_key, **kw):
        seen.append(threading.get_ident())
        return "/m/a.gguf"

    monkeypatch.setattr(start_cmd, "_props_model_path", helper)

    await world.start()

    assert seen and seen[0] != main


# ------------------------------------------------------------------ D-42: no pointer to init


async def test_a_llamacpp_attach_with_no_record_records_the_path_and_points_nowhere(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.config(record=False)
    _fake_helper(monkeypatch, "/m/a.gguf")

    lines = await w.start()

    assert model_path(w.ledger, "llamacpp", "qwen") == "/m/a.gguf"
    assert not [x for x in lines if "localharness init" in x or "to let the harness size the window" in x]
    assert w.window(lines) == []


def test_the_no_record_undo_variant_stays_unused_by_start():
    import inspect

    assert "no-record" not in inspect.getsource(start_cmd._start_async)
