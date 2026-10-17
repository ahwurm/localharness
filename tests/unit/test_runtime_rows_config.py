"""The runtime rows decide the config layer's gates: record validation, the peer binary borrow, trust entries and migration."""
from __future__ import annotations

import pytest

from localharness.config.models import EndpointRef, HarnessConfig, ManagedServerConfig, ProviderConfig

LLAMA_SERVER = "/opt/llama-server"
_PROVIDER = ProviderConfig(provider_type="vllm", base_url="http://127.0.0.1:9/v1", default_model="m")


def _refusal(**kw) -> str:
    with pytest.raises(ValueError) as exc:
        ManagedServerConfig(**kw)
    return str(exc.value)


def _peer(runtime="llamacpp") -> EndpointRef:
    return EndpointRef(
        name="p", base_url="http://127.0.0.1:8082/v1", provider_type="llamacpp", gpu=True,
        lifecycle=ManagedServerConfig(runtime=runtime, model="/x/p.gguf" if runtime == "llamacpp" else "tag"))


def _borrowed(primary: ManagedServerConfig, runtime="llamacpp"):
    cfg = HarnessConfig(provider=_PROVIDER, server=primary, extra_endpoints=[_peer(runtime)])
    return cfg.extra_endpoints[0].lifecycle.binary


@pytest.mark.parametrize("runtime,base", [
    ("vllm", {"binary": "/opt/vllm", "model": "org/repo"}),
    ("ollama", {"model": "tag"}),
    ("lmstudio", {"model": "tag"}),
])
def test_llamacpp_only_refusal_message_is_exact(runtime, base):
    assert f"runtime={runtime}: mmproj is llama.cpp-only (runtime: llamacpp)" in _refusal(runtime=runtime, mmproj="/x", **base)
    assert f"runtime={runtime}: mmproj, served_name are llama.cpp-only (runtime: llamacpp)" in _refusal(
        runtime=runtime, mmproj="/x", served_name="a", **base)


@pytest.mark.parametrize("runtime,base", [
    ("llamacpp", {"model": "/x/m.gguf"}),
    ("ollama", {"model": "tag"}),
    ("lmstudio", {"model": "tag"}),
])
def test_vllm_only_refusal_message_is_exact(runtime, base):
    assert f"runtime={runtime}: reasoning_parser, tool_call_parser are vLLM-only (runtime: vllm)" in _refusal(
        runtime=runtime, reasoning_parser="glm45", tool_call_parser="glm45", **base)
    assert f"runtime={runtime}: tool_call_parser is vLLM-only (runtime: vllm)" in _refusal(
        runtime=runtime, tool_call_parser="glm45", **base)


def test_llamacpp_only_fields_are_reported_before_vllm_only_ones():
    msg = _refusal(runtime="vllm", binary="/opt/vllm", model="org/repo", mmproj="/x", reasoning_parser="glm45")
    assert "llama.cpp-only" in msg and "vLLM-only" not in msg


def test_vllm_binary_launch_without_binary_is_refused():
    assert "launch=binary requires `binary` (path to the vllm executable)" in _refusal(
        runtime="vllm", launch="binary", model="org/repo")


def test_vllm_docker_launch_without_image_is_refused():
    assert "launch=docker requires `docker_image`" in _refusal(runtime="vllm", launch="docker", model="org/repo")


@pytest.mark.parametrize("extra", [["--gpu", "max"], ["--gpu=off"]])
def test_lmstudio_gpu_in_extra_args_is_refused(extra):
    assert (
        "runtime=lmstudio: `--gpu` must not appear in extra_args — it is derived from "
        "`gpu` (--gpu max when gpu=True, --gpu off when False)."
    ) in _refusal(runtime="lmstudio", model="tag", extra_args=extra)


def test_gpu_in_extra_args_is_not_refused_elsewhere():
    ManagedServerConfig(runtime="ollama", model="tag", extra_args=["--gpu", "max"])
    ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", extra_args=["--gpu-layers", "9"])


@pytest.mark.parametrize("kw", [
    {"runtime": "ollama", "model": "tag"},
    {"runtime": "lmstudio", "model": "tag"},
    {"runtime": "llamacpp", "model": "/x/m.gguf"},
])
def test_records_with_no_binary_still_validate(kw):
    assert ManagedServerConfig(**kw).binary is None


def test_a_llamacpp_primary_lends_its_binary_to_a_llamacpp_peer():
    assert _borrowed(ManagedServerConfig(runtime="llamacpp", binary=LLAMA_SERVER, model="/x/m.gguf")) == LLAMA_SERVER


@pytest.mark.parametrize("primary", [
    ManagedServerConfig(binary="/opt/vllm", model="org/repo"),
    ManagedServerConfig(runtime="ollama", binary="/usr/bin/ollama", model="tag"),
    ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf"),
])
def test_no_other_primary_lends_a_binary(primary):
    assert _borrowed(primary) is None


def test_a_peer_of_another_runtime_borrows_nothing():
    primary = ManagedServerConfig(runtime="llamacpp", binary=LLAMA_SERVER, model="/x/m.gguf")
    assert _borrowed(primary, runtime="ollama") is None


def _posture(runtime_key: dict) -> list[str]:
    from localharness.config.trust import _posture_entries
    raws = {"config.yaml": {"server": {"bind_all": True, "require_api_key": True, **runtime_key}}}
    return [e["shown"] for e in _posture_entries(raws)]


@pytest.mark.parametrize("runtime_key,runtimes", [
    ({"runtime": "vllm"}, {"vllm"}),
    ({"runtime": "llamacpp"}, {"llamacpp"}),
    ({}, {"vllm"}),
    ({"runtime": "ollama"}, set()),
    ({"runtime": "lmstudio"}, set()),
    ({"runtime": "bogus"}, set()),
    ({"runtime": ["vllm"]}, set()),
])
def test_trust_posture_entries_exist_for_exactly_the_keyed_runtimes(runtime_key, runtimes):
    assert {s.split(":")[0] for s in _posture(runtime_key)} == runtimes


def test_trust_posture_entries_carry_both_keys():
    assert _posture({"runtime": "llamacpp"}) == ["llamacpp:true:open", "llamacpp:true:open"]


@pytest.mark.parametrize("server,selected", [
    ({"runtime": "vllm"}, True),
    ({}, True),
    ({"runtime": "vllm", "bind_all": False}, False),
    ({"runtime": "llamacpp"}, False),
    ({"runtime": "ollama"}, False),
    ({"runtime": "lmstudio"}, False),
    ({"runtime": "bogus"}, False),
])
def test_bind_all_migration_selects_vllm_sections_without_the_key(server, selected):
    from localharness.config.migrate import _needs_bind_all
    assert (_needs_bind_all({"server": server}) == ("server",)) is selected


def test_bind_all_migration_selects_peer_lifecycles_the_same_way():
    from localharness.config.migrate import _needs_bind_all
    data = {"extra_endpoints": [{"lifecycle": {"runtime": "llamacpp"}}, {"lifecycle": {}}, {"lifecycle": {"runtime": "ollama"}}]}
    assert _needs_bind_all(data) == ("extra_endpoints[1].lifecycle",)


def test_library_names_mtp_only_for_a_gguf_model_and_shows_checkpoints_as_visible_text():
    from localharness.cli import library
    vllm = ManagedServerConfig(binary="/opt/vllm", model="org/repo")
    assert library.preset_clause(vllm) == "(vLLM)"
    assert library._model_text(vllm) == "org/repo"
    ollama = ManagedServerConfig(runtime="ollama", model="tag")
    assert library.preset_clause(ollama) == "(Ollama)"
    assert library._model_text(ollama) == library.shown_path("tag")


def test_library_binary_lookup_uses_the_rows_path_binary(monkeypatch):
    from localharness.cli import library
    seen = []
    monkeypatch.setattr(library, "which_outside_cwd", lambda b: seen.append(b) or "/x")
    monkeypatch.setattr(library.managed_server, "vllm_binary", lambda d: seen.append("vllm") or None)
    assert library._on_path("llamacpp", global_dir=None) is True
    assert library._on_path("vllm", global_dir=None) is False
    assert seen == ["llama-server", "vllm"]
