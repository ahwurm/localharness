"""probe_window: served AND trained window per runtime, the bearer on every llama.cpp route,
doctor's meta fallback for any provider type, and probe_served_window as its served half.

`httpx.get` is faked with a callable that records (url, kwargs) and answers from a url -> body map.
"""
from __future__ import annotations

import httpx
import pytest

from localharness.agent import context as ctxmod
from localharness.agent.context import WindowProbe, probe_served_window, probe_window
from tests.unit.gguf_support import write_gguf


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _serve(monkeypatch, bodies: dict[str, object]) -> list[tuple[str, dict]]:
    """Fake httpx.get: answer each url from `bodies` (an exception value is raised), log every call."""
    calls: list[tuple[str, dict]] = []

    def _get(url, timeout=0, **kw):
        calls.append((url, kw))
        body = bodies.get(url, {})
        if isinstance(body, Exception):
            raise body
        return _Resp(body)

    monkeypatch.setattr(httpx, "get", _get)
    return calls


LLAMA = "http://127.0.0.1:8080/v1"
PROPS = "http://127.0.0.1:8080/props"
META = {"data": [{"id": "qwen", "meta": {"n_ctx": 65536, "n_ctx_train": 262144}}]}


def test_llamacpp_models_then_one_props_call_both_carry_the_bearer(monkeypatch):
    calls = _serve(monkeypatch, {f"{LLAMA}/models": META})
    assert probe_window(LLAMA, "qwen", "llamacpp", api_key="K") == WindowProbe(65536, 262144)
    assert calls == [(f"{LLAMA}/models", {"headers": {"Authorization": "Bearer K"}}),
                     (PROPS, {"headers": {"Authorization": "Bearer K"}})]


def test_llamacpp_without_a_key_sends_no_headers(monkeypatch):
    calls = _serve(monkeypatch, {f"{LLAMA}/models": META})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(65536, 262144)
    assert calls == [(f"{LLAMA}/models", {}), (PROPS, {})]


def test_llamacpp_props_fallback_for_served_also_carries_the_bearer(monkeypatch):
    calls = _serve(monkeypatch, {
        f"{LLAMA}/models": {"data": [{"id": "qwen", "meta": {"n_ctx_train": 262144}}]},
        "http://127.0.0.1:8080/props": {"default_generation_settings": {"n_ctx": 32768}},
    })
    assert probe_window(LLAMA, "qwen", "llamacpp", api_key="K") == WindowProbe(32768, 262144)
    assert [u for u, _ in calls] == [f"{LLAMA}/models", "http://127.0.0.1:8080/props"]
    assert all(kw == {"headers": {"Authorization": "Bearer K"}} for _, kw in calls)


def test_llamacpp_props_alone_when_models_has_no_entry(monkeypatch):
    _serve(monkeypatch, {"http://127.0.0.1:8080/props": {"n_ctx": 16384}})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(16384, None)


def test_adjacency_the_id_matched_entry_wins_not_the_first(monkeypatch):
    _serve(monkeypatch, {f"{LLAMA}/models": {"data": [
        {"id": "other", "meta": {"n_ctx": 1024, "n_ctx_train": 2048}},
        {"id": "qwen", "meta": {"n_ctx": 65536, "n_ctx_train": 262144}},
    ]}})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(65536, 262144)


def test_llamacpp_takes_the_only_entry_when_no_id_matches(monkeypatch):
    _serve(monkeypatch, {f"{LLAMA}/models": {"data": [
        {"id": "alias", "meta": {"n_ctx": 4096, "n_ctx_train": 8192}}]}})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(4096, 8192)


def test_llamacpp_two_entries_none_matching_is_unknown(monkeypatch):
    _serve(monkeypatch, {f"{LLAMA}/models": {"data": [
        {"id": "a", "meta": {"n_ctx": 1, "n_ctx_train": 2}},
        {"id": "b", "meta": {"n_ctx": 3, "n_ctx_train": 4}}]}})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(None, None)


def test_vllm_served_is_trained(monkeypatch):
    _serve(monkeypatch, {"http://h:8000/v1/models": {"data": [{"id": "m", "max_model_len": 32768}]}})
    assert probe_window("http://h:8000/v1", "m", "vllm") == WindowProbe(32768, 32768)


def test_vllm_context_length_alias_and_the_bearer(monkeypatch):
    calls = _serve(monkeypatch, {"http://h:8000/v1/models": {"data": [{"id": "m", "context_length": 8192}]}})
    assert probe_window("http://h:8000/v1", "m", "vllm", api_key="K") == WindowProbe(8192, 8192)
    assert calls[0][1] == {"headers": {"Authorization": "Bearer K"}}


@pytest.mark.parametrize("ptype", ["vllm", "unknown", None])
def test_d39_meta_fallback_for_any_non_llamacpp_provider_type(monkeypatch, ptype):
    """A llama-server behind a non-llamacpp provider_type keeps its window (doctor's old reading)."""
    _serve(monkeypatch, {"http://h:8080/v1/models": {"data": [
        {"id": "m", "meta": {"n_ctx": 65536, "n_ctx_train": 262144}}]}})
    assert probe_window("http://h:8080/v1", "m", ptype) == WindowProbe(65536, 262144)


@pytest.mark.parametrize("ptype", ["vllm", "unknown", None])
def test_an_entry_with_none_of_the_fields_is_unknown(monkeypatch, ptype):
    _serve(monkeypatch, {"http://h:8080/v1/models": {"data": [{"id": "m"}]}})
    assert probe_window("http://h:8080/v1", "m", ptype) == WindowProbe(None, None)


def test_a_model_the_endpoint_does_not_serve_is_unknown(monkeypatch):
    _serve(monkeypatch, {"http://h:8000/v1/models": {"data": [
        {"id": "x", "max_model_len": 8192}, {"id": "y", "max_model_len": 4096}]}})
    assert probe_window("http://h:8000/v1", "absent", "vllm") == WindowProbe(None, None)


@pytest.mark.parametrize("ptype", ["vllm", "unknown", None])
def test_a_single_entry_that_does_not_match_the_id_is_still_the_window(monkeypatch, ptype):
    """WR-07: doctor's old derivation matched `id == model or len(entries) == 1`; a single-model server
    whose id is an alias or a path keeps reporting its window, as the llama.cpp branch does."""
    _serve(monkeypatch, {"http://h:8000/v1/models": {"data": [{"id": "/models/x.gguf", "max_model_len": 8192}]}})
    assert probe_window("http://h:8000/v1", "alias", ptype) == WindowProbe(8192, 8192)


def test_a_single_non_matching_entry_keeps_the_meta_fallback(monkeypatch):
    _serve(monkeypatch, {"http://h:8080/v1/models": {"data": [
        {"id": "path-style-id", "meta": {"n_ctx": 65536, "n_ctx_train": 262144}}]}})
    assert probe_window("http://h:8080/v1", "alias", "unknown") == WindowProbe(65536, 262144)


def test_two_entries_none_matching_stay_unknown_for_the_generic_branch(monkeypatch):
    _serve(monkeypatch, {"http://h:8000/v1/models": {"data": [
        {"id": "a", "max_model_len": 1000}, {"id": "b", "max_model_len": 2000}]}})
    assert probe_window("http://h:8000/v1", "c", None) == WindowProbe(None, None)


def test_hostile_numbers_become_none(monkeypatch):
    """T-2-07: strings, zero, negatives and lists steer nothing."""
    _serve(monkeypatch, {f"{LLAMA}/models": {"data": [
        {"id": "qwen", "meta": {"n_ctx": "lots", "n_ctx_train": -5}}]},
        "http://127.0.0.1:8080/props": {"default_generation_settings": {"n_ctx": [1]}}})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(None, None)


def _qwen_gguf(tmp_path, ctx: int = 32768):
    return write_gguf(tmp_path / "q.gguf", {"general.architecture": "qwen3", "qwen3.context_length": ctx})


@pytest.mark.parametrize("resolves", [True, False])
def test_ollama_served_from_api_ps_trained_from_the_gguf(monkeypatch, tmp_path, resolves):
    _serve(monkeypatch, {"http://h:11434/api/ps": {"models": [
        {"model": "qwen", "name": "qwen", "context_length": 4096}]}})
    path = _qwen_gguf(tmp_path)
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.resolve_gguf_path",
                        lambda model, ptype: path if resolves else None)
    expect = WindowProbe(4096, 32768 if resolves else None)
    assert probe_window("http://h:11434/v1", "qwen", "ollama") == expect


def test_lmstudio_served_from_the_loaded_window_trained_from_the_gguf(monkeypatch, tmp_path):
    _serve(monkeypatch, {"http://h:1234/api/v0/models": {"data": [
        {"id": "qwen", "state": "loaded", "loaded_context_length": 8192, "max_context_length": 131072}]}})
    path = _qwen_gguf(tmp_path)
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.resolve_gguf_path", lambda m, p: path)
    assert probe_window("http://h:1234/v1", "qwen", "lmstudio") == WindowProbe(8192, 32768)


def test_an_unreadable_gguf_leaves_trained_none_and_never_raises(monkeypatch, tmp_path):
    _serve(monkeypatch, {"http://h:11434/api/ps": {"models": [
        {"model": "qwen", "name": "qwen", "context_length": 4096}]}})
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf at all")
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.resolve_gguf_path", lambda m, p: bad)
    assert probe_window("http://h:11434/v1", "qwen", "ollama") == WindowProbe(4096, None)
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.resolve_gguf_path",
                        lambda m, p: tmp_path / "missing.gguf")
    assert probe_window("http://h:11434/v1", "qwen", "ollama") == WindowProbe(4096, None)


@pytest.mark.parametrize("ptype", ["llamacpp", "vllm", "ollama", "lmstudio", None])
def test_a_connection_error_is_unknown_not_a_raise(monkeypatch, ptype):
    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.resolve_gguf_path", lambda m, p: None)
    assert probe_window("http://h:1/v1", "m", ptype) == WindowProbe(None, None)


def test_empty_inputs_make_no_http_call(monkeypatch):
    calls = _serve(monkeypatch, {})
    assert probe_window(None, "m") == WindowProbe(None, None)
    assert probe_window("http://x", None) == WindowProbe(None, None)
    assert probe_window("", "m", "llamacpp") == WindowProbe(None, None)
    assert calls == []


_BRANCHES = [
    ("llamacpp", {f"{LLAMA}/models": META}, LLAMA, "qwen"),
    ("vllm", {"http://h:8000/v1/models": {"data": [{"id": "m", "max_model_len": 32768}]}},
     "http://h:8000/v1", "m"),
    (None, {"http://h:8080/v1/models": {"data": [{"id": "m", "meta": {"n_ctx": 7, "n_ctx_train": 9}}]}},
     "http://h:8080/v1", "m"),
    ("ollama", {"http://h:11434/api/ps": {"models": [{"model": "m", "context_length": 4096}]}},
     "http://h:11434/v1", "m"),
    ("lmstudio", {"http://h:1234/api/v0/models": {"data": [{"id": "m", "loaded_context_length": 8192}]}},
     "http://h:1234/v1", "m"),
    ("vllm", {"http://h:8000/v1/models": {"data": []}}, "http://h:8000/v1", "m"),
]


@pytest.mark.parametrize("ptype,bodies,base,model", _BRANCHES)
def test_probe_served_window_is_the_served_half(monkeypatch, ptype, bodies, base, model):
    _serve(monkeypatch, bodies)
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.resolve_gguf_path", lambda m, p: None)
    assert probe_served_window(base, model, ptype, api_key="K") == \
        probe_window(base, model, ptype, api_key="K").served


def test_probe_served_window_delegates(monkeypatch):
    seen = {}

    def _fake(*a, **k):
        seen["args"], seen["kw"] = a, k
        return WindowProbe(123, 456)

    monkeypatch.setattr(ctxmod, "probe_window", _fake)
    assert ctxmod.probe_served_window("http://b/v1", "m", "vllm", api_key="K") == 123
    assert seen == {"args": ("http://b/v1", "m", "vllm"), "kw": {"api_key": "K"}}


# --- the server's default temperature rides the one /props GET (SAMPLE-04, D-31) -----------------

def _with_props(monkeypatch, props, **kw):
    _serve(monkeypatch, {f"{LLAMA}/models": META, PROPS: props})
    return probe_window(LLAMA, "qwen", "llamacpp", **kw)


def test_sampling_reads_params_temperature(monkeypatch):
    got = _with_props(monkeypatch, {"default_generation_settings": {"params": {"temperature": 0.8}}})
    assert got == WindowProbe(65536, 262144, 0.8)


def test_sampling_falls_back_to_temperature_beside_params(monkeypatch):
    got = _with_props(monkeypatch, {"default_generation_settings": {"temperature": 0.7, "params": {}}})
    assert got.sampling == 0.7


def test_sampling_an_int_reads_as_a_float(monkeypatch):
    got = _with_props(monkeypatch, {"default_generation_settings": {"params": {"temperature": 1}}})
    assert got.sampling == 1.0 and isinstance(got.sampling, float)


@pytest.mark.parametrize("bad", [True, "1.0", [1], float("nan"), float("inf"), 1e999, -0.5, None])
def test_sampling_hostile_values_are_none(monkeypatch, bad):
    got = _with_props(monkeypatch, {"default_generation_settings": {"params": {"temperature": bad}}})
    assert got.sampling is None and got.served == 65536


@pytest.mark.parametrize("props", [
    {}, {"default_generation_settings": {}}, {"default_generation_settings": "oops"},
    {"default_generation_settings": {"params": "oops"}}, {"default_generation_settings": [1]}, [],
])
def test_sampling_missing_or_malformed_props_is_none(monkeypatch, props):
    assert _with_props(monkeypatch, props).sampling is None


def test_sampling_props_raising_is_none_and_served_survives(monkeypatch):
    _serve(monkeypatch, {f"{LLAMA}/models": META, PROPS: httpx.ConnectError("refused")})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(65536, 262144, None)


def test_sampling_props_down_and_models_silent_is_all_none(monkeypatch):
    _serve(monkeypatch, {f"{LLAMA}/models": httpx.ConnectError("x"), PROPS: httpx.ConnectError("y")})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(None, None, None)


def test_sampling_exactly_one_props_call_even_when_models_gave_the_window(monkeypatch):
    calls = _serve(monkeypatch, {f"{LLAMA}/models": META,
                                 PROPS: {"default_generation_settings": {"params": {"temperature": 1.0}}}})
    probe_window(LLAMA, "qwen", "llamacpp", api_key="K")
    assert [u for u, _ in calls] == [f"{LLAMA}/models", PROPS]
    assert calls[1][1] == {"headers": {"Authorization": "Bearer K"}}


def test_sampling_props_also_supplies_the_served_fallback_in_the_same_call(monkeypatch):
    calls = _serve(monkeypatch, {
        f"{LLAMA}/models": {"data": [{"id": "qwen", "meta": {"n_ctx_train": 262144}}]},
        PROPS: {"default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 0.5}}}})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(32768, 262144, 0.5)
    assert [u for u, _ in calls].count(PROPS) == 1


def test_sampling_other_runtimes_leave_it_none(monkeypatch, tmp_path):
    monkeypatch.setattr("localharness.agent.gguf_tokenizer.resolve_gguf_path", lambda m, p: None)
    body = {"default_generation_settings": {"params": {"temperature": 0.5}}}
    for ptype, bodies, base in [
        ("vllm", {"http://h:8000/v1/models": {"data": [{"id": "m", "max_model_len": 32768}]}}, "http://h:8000/v1"),
        ("ollama", {"http://h:11434/api/ps": {"models": [{"model": "m", "context_length": 4096}]}},
         "http://h:11434/v1"),
        ("lmstudio", {"http://h:1234/api/v0/models": {"data": [{"id": "m", "loaded_context_length": 8192}]}},
         "http://h:1234/v1"),
    ]:
        calls = _serve(monkeypatch, {**bodies, "http://h:8000/props": body})
        assert probe_window(base, "m", ptype).sampling is None
        assert not any(u.endswith("/props") for u, _ in calls)
