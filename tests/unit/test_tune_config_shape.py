"""The config shape that lets a person pin what the harness derives at launch, and the machine-only
knobs that steer the derivation.

Record fields (`speculative`, `reasoning_parser`, `tool_call_parser`) are runtime-gated, validated
and fingerprinted as `launch_model` entries only when set; the `autosense` knobs and
`pending_preset` are machine-only and never fingerprinted.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from localharness.config import models as M
from localharness.config import trust
from localharness.config.models import ManagedServerConfig
from tests.unit.trust_support import by_name, g, names, write_yaml  # noqa: F401 — g is a fixture

SEVEN = ("model", "mmproj", "served_name", "context", "cache_type_k", "cache_type_v", "cache_ram")
THREE = ("speculative", "reasoning_parser", "tool_call_parser")


def _llamacpp(**kw) -> ManagedServerConfig:
    return ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server",
                               model="/m/a.gguf", **kw)


def _vllm(**kw) -> ManagedServerConfig:
    return ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm",
                               model="org/model", **kw)


# ------------------------------------------------------------------ the record fields


@pytest.mark.parametrize("value", ["off", "draft-mtp"])
def test_speculative_accepts_off_and_draft_mtp_on_llamacpp(value):
    assert _llamacpp(speculative=value).speculative == value


def test_speculative_refuses_other_values():
    with pytest.raises(ValidationError):
        _llamacpp(speculative="on")


def test_unset_record_fields_are_none():
    s = _llamacpp()
    assert (s.speculative, s.reasoning_parser, s.tool_call_parser) == (None, None, None)


def test_a_vllm_record_takes_the_parsers():
    s = _vllm(reasoning_parser="qwen3", tool_call_parser="off")
    assert (s.reasoning_parser, s.tool_call_parser) == ("qwen3", "off")


def test_speculative_on_vllm_is_refused_naming_field_and_runtime():
    with pytest.raises(ValidationError, match=r"runtime=vllm: speculative is llama\.cpp-only"):
        _vllm(speculative="off")


def test_a_parser_on_llamacpp_is_refused_naming_field_and_runtime():
    with pytest.raises(ValidationError, match=r"reasoning_parser is vLLM-only \(runtime: vllm\)"):
        _llamacpp(reasoning_parser="qwen3")


def test_two_parsers_on_llamacpp_use_the_plural_verb():
    with pytest.raises(ValidationError, match=r"reasoning_parser, tool_call_parser are vLLM-only"):
        _llamacpp(reasoning_parser="a", tool_call_parser="b")


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio"])
def test_a_parser_on_ollama_or_lmstudio_is_refused_before_their_early_return(runtime):
    with pytest.raises(ValidationError, match=rf"runtime={runtime}: tool_call_parser is vLLM-only"):
        ManagedServerConfig(runtime=runtime, model="m", tool_call_parser="hermes")


@pytest.mark.parametrize("field", ["reasoning_parser", "tool_call_parser"])
@pytest.mark.parametrize("bad", ["--foo", "a b", "", "x" * 65, "-x", ".x", "a\nb", "a\x00"])
def test_a_bad_parser_name_is_refused(field, bad):
    with pytest.raises(ValidationError, match=r"must be a parser name \(letters, digits, _ \. -\) or off"):
        _vllm(**{field: bad})


@pytest.mark.parametrize("good", ["qwen3_coder", "glm45", "llama3_json", "off", "a" * 64, "x.y-z"])
def test_a_plain_parser_name_is_accepted(good):
    assert _vllm(reasoning_parser=good).reasoning_parser == good


def test_the_gate_tuples():
    assert "speculative" in M.LLAMACPP_ONLY_FIELDS
    assert M.VLLM_ONLY_FIELDS == ("reasoning_parser", "tool_call_parser")


# ------------------------------------------------------------------ the fingerprint


def test_the_launch_model_table_ends_with_the_three_record_fields():
    assert trust.LAUNCH_MODEL_KEYS[-3:] == THREE
    assert trust.LAUNCH_MODEL_KEYS[:7] == SEVEN


def test_a_set_speculative_is_a_launch_model_entry(g):
    write_yaml(g / "config.yaml", {"server": {
        "runtime": "llamacpp", "binary": "/x/b", "model": "/m/a.gguf", "speculative": "off"}})
    entry = by_name(trust.machine_snapshot(g), "server.speculative")
    assert entry == {"file": "config.yaml", "kind": "launch_model",
                     "name": "server.speculative", "shown": "off"}


def test_a_vllm_peer_lifecycle_parser_is_named_by_the_peer(g):
    write_yaml(g / "config.yaml", {"extra_endpoints": [{
        "name": "qv", "base_url": "http://127.0.0.1:8081/v1",
        "lifecycle": {"runtime": "vllm", "launch": "binary", "binary": "/x/vllm",
                      "model": "org/m", "reasoning_parser": "qwen3",
                      "tool_call_parser": "hermes"}}]})
    snap = trust.machine_snapshot(g)
    assert by_name(snap, "extra_endpoints.qv.lifecycle.reasoning_parser")["shown"] == "qwen3"
    assert by_name(snap, "extra_endpoints.qv.lifecycle.tool_call_parser")["kind"] == "launch_model"


def test_a_config_without_the_three_fingerprints_as_the_earlier_seven(g, monkeypatch):
    write_yaml(g / "config.yaml", {"server": {
        "runtime": "llamacpp", "binary": "/x/b", "model": "/m/a.gguf", "context": 8192,
        "cache_type_k": "q8_0"}})
    now = trust.machine_snapshot(g)
    monkeypatch.setattr(trust, "LAUNCH_MODEL_KEYS", SEVEN)
    assert trust.machine_snapshot(g) == now


def test_an_explicit_null_yields_no_entry(g):
    write_yaml(g / "config.yaml", {"server": {
        "runtime": "llamacpp", "binary": "/x/b", "model": "/m/a.gguf",
        "speculative": None, "reasoning_parser": None, "tool_call_parser": None}})
    assert not [n for n in names(trust.machine_snapshot(g)) if n[2].split(".")[-1] in THREE]
