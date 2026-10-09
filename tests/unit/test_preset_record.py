"""The preset record: ten new `ManagedServerConfig` fields, their runtime gate, the optional
llama.cpp `binary`, and a peer's inherited binary.

The record's own validator describes what a valid record is; whether it can be launched is the
launch decision's job, not the schema's, so a declared-only llama.cpp record validates here."""
from __future__ import annotations

import pytest

from localharness.config.models import (
    LLAMACPP_ONLY_FIELDS,
    EffortLevel,
    EndpointRef,
    HarnessConfig,
    ManagedServerConfig,
    ProviderConfig,
)
from tests.unit.test_floor_machine_level import _layers

GGUF = "~/models/gguf/Swift-1.5-Qwen3.8-27B-NVFP4-Q8mix.gguf"
LLAMA_SERVER = "/home/awurm/src/llama.cpp/build/bin/llama-server"

NEW_FIELDS = (
    "mmproj", "served_name", "context", "cache_type_k", "cache_type_v", "cache_ram",
    "api_key_file", "env", "config_ini", "effort",
)

# One valid value per llama.cpp-only field. api_key_file only validates with require_api_key, so
# the gate test sets it for that row; env and config_ini carry the empty value because the gate
# counts only None as unset.
GATE_VALUES = {
    "mmproj": "/x/p.gguf",
    "served_name": "a",
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "cache_ram": 0,
    "api_key_file": "/x/k",
    "env": {"LLAMA_ARG_N_GPU_LAYERS": "99"},
    "config_ini": "",
}


def _llama(**kw):
    return ManagedServerConfig(runtime="llamacpp", binary=LLAMA_SERVER, model="/x/m.gguf", **kw)


def _spec_config() -> dict:
    """Spec section 5, as the global config.yaml carries it."""
    return {
        "server": {
            "runtime": "llamacpp",
            "binary": LLAMA_SERVER,
            "model": GGUF,
            "mmproj": "~/models/gguf/mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf",
            "served_name": "qwen-27b-swift-1.5",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "effort": "high",
            "port": 8080,
            "extra_args": ["-fa", "on", "--jinja", "--spec-type", "draft-mtp",
                           "--spec-draft-n-max", "3"],
        },
        "extra_endpoints": [
            {
                "name": "flash-next",
                "base_url": "http://127.0.0.1:8082/v1",
                "provider_type": "llamacpp",
                "gpu": True,
                "lifecycle": {
                    "gpu": True,
                    "runtime": "llamacpp",
                    "model": "~/models/gguf/flash-next.gguf",
                    "served_name": "flash-next",
                    "context": 32768,
                    "effort": "max",
                    "port": 8082,
                },
            },
            {
                "name": "qwen-vllm",
                "base_url": "http://127.0.0.1:8081/v1",
                "provider_type": "vllm",
                "gpu": True,
                "lifecycle": {
                    "gpu": True,
                    "runtime": "vllm",
                    "launch": "docker",
                    # The spec's example omits docker_image, which the validator requires.
                    "docker_image": "vllm/vllm-openai:latest",
                    "model": "Qwen/Qwen3.6-27B",
                    "extra_args": ["--max-model-len", "131072"],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# The tracer: the spec's record loads through the real loader
# ---------------------------------------------------------------------------


def test_spec_record_loads_through_the_loader(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg=_spec_config())
    cfg = loader.load_harness()
    srv = cfg.server
    assert srv.runtime == "llamacpp"
    assert srv.binary == LLAMA_SERVER
    assert srv.model == GGUF
    assert srv.mmproj == "~/models/gguf/mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf"
    assert srv.served_name == "qwen-27b-swift-1.5"
    assert (srv.cache_type_k, srv.cache_type_v) == ("q8_0", "q8_0")
    assert srv.effort == "high"
    assert srv.port == 8080
    assert srv.extra_args[:2] == ["-fa", "on"]
    assert (srv.context, srv.cache_ram, srv.api_key_file, srv.env, srv.config_ini) == (
        None, None, None, None, None)
    flash, vllm = cfg.extra_endpoints
    assert flash.lifecycle.served_name == "flash-next"
    assert flash.lifecycle.context == 32768
    assert flash.lifecycle.effort == "max"
    assert flash.lifecycle.port == 8082
    # The binary-less peer took the primary's at load.
    assert flash.lifecycle.binary == LLAMA_SERVER
    assert vllm.lifecycle.runtime == "vllm"
    assert vllm.lifecycle.binary is None


def test_all_ten_fields_survive_the_loader(tmp_path):
    cfg_in = {"server": {
        "runtime": "llamacpp", "binary": LLAMA_SERVER, "model": "/x/m.gguf",
        "mmproj": "/x/p.gguf", "served_name": "alias", "context": 65536,
        "cache_type_k": "q8_0", "cache_type_v": "q4_0", "cache_ram": 4096,
        "api_key_file": "~/keys/llama", "require_api_key": True,
        "env": {"LLAMA_ARG_N_GPU_LAYERS": "99"},
        "config_ini": "[*]\nn-gpu-layers = 99\n", "effort": "low",
    }}
    loader, _ = _layers(tmp_path, g_cfg=cfg_in)
    srv = loader.load_harness().server
    for name, want in cfg_in["server"].items():
        assert getattr(srv, name) == want, name


# ---------------------------------------------------------------------------
# binary is optional on llama.cpp (declared, not launchable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binary", [None, ""])
def test_binary_less_llamacpp_record_validates(binary):
    srv = ManagedServerConfig(runtime="llamacpp", binary=binary, model="/x/m.gguf")
    assert not srv.binary


def test_vllm_binary_launch_still_requires_a_binary():
    with pytest.raises(ValueError, match="launch=binary requires"):
        ManagedServerConfig(runtime="vllm", model="org/repo")


def test_vllm_docker_launch_still_requires_an_image():
    with pytest.raises(ValueError, match="docker_image"):
        ManagedServerConfig(runtime="vllm", launch="docker", model="org/repo")


# ---------------------------------------------------------------------------
# The runtime gate
# ---------------------------------------------------------------------------


def _gate_kwargs(field: str) -> dict:
    kw = {field: GATE_VALUES[field]}
    if field == "api_key_file":
        kw["require_api_key"] = True
    return kw


def _other(runtime: str) -> dict:
    if runtime == "vllm":
        return {"runtime": "vllm", "launch": "docker", "docker_image": "img", "model": "org/repo"}
    return {"runtime": runtime, "model": "tag"}


@pytest.mark.parametrize("runtime", ["vllm", "ollama", "lmstudio"])
@pytest.mark.parametrize("field", LLAMACPP_ONLY_FIELDS)
def test_llamacpp_only_field_is_refused_on_other_runtimes(runtime, field):
    with pytest.raises(ValueError) as exc:
        ManagedServerConfig(**_other(runtime), **_gate_kwargs(field))
    msg = str(exc.value)
    assert field in msg
    assert "llama.cpp-only" in msg


@pytest.mark.parametrize("runtime", ["vllm", "ollama", "lmstudio"])
def test_gate_names_every_offending_field(runtime):
    with pytest.raises(ValueError) as exc:
        ManagedServerConfig(**_other(runtime), mmproj="/x", served_name="a")
    msg = str(exc.value)
    assert "mmproj" in msg and "served_name" in msg and "are llama.cpp-only" in msg


def test_gate_message_is_singular_for_one_field():
    with pytest.raises(ValueError, match="mmproj is llama.cpp-only"):
        ManagedServerConfig(**_other("vllm"), mmproj="/x")


@pytest.mark.parametrize("runtime", ["vllm", "ollama", "lmstudio"])
def test_context_and_effort_are_accepted_on_every_runtime(runtime):
    srv = ManagedServerConfig(**_other(runtime), context=4096, effort="max")
    assert (srv.context, srv.effort) == (4096, "max")


@pytest.mark.parametrize("field", LLAMACPP_ONLY_FIELDS)
def test_unset_llamacpp_only_field_validates_on_vllm(field):
    srv = ManagedServerConfig(**_other("vllm"), **{field: None})
    assert getattr(srv, field) is None


def test_llamacpp_only_fields_are_the_eight():
    assert set(LLAMACPP_ONLY_FIELDS) == {
        "mmproj", "served_name", "cache_type_k", "cache_type_v", "cache_ram",
        "api_key_file", "env", "config_ini"}


# ---------------------------------------------------------------------------
# api_key_file implies require_api_key and must be absolute
# ---------------------------------------------------------------------------


def test_api_key_file_without_require_api_key_is_refused():
    with pytest.raises(ValueError, match="api_key_file requires require_api_key: true"):
        _llama(api_key_file="/x/k")


def test_api_key_file_with_require_api_key_false_is_refused():
    with pytest.raises(ValueError, match="api_key_file requires require_api_key: true"):
        _llama(api_key_file="/x/k", require_api_key=False)


def test_relative_api_key_file_is_refused():
    with pytest.raises(ValueError, match="absolute path"):
        _llama(api_key_file="k", require_api_key=True)


def test_tilde_api_key_file_validates():
    srv = _llama(api_key_file="~/k", require_api_key=True)
    assert srv.api_key_file == "~/k"


# ---------------------------------------------------------------------------
# Adjacency, empty and encoding edges
# ---------------------------------------------------------------------------


def test_context_zero_is_refused_and_one_validates():
    with pytest.raises(ValueError):
        _llama(context=0)
    assert _llama(context=1).context == 1


def test_cache_ram_zero_validates_and_negative_is_refused():
    assert _llama(cache_ram=0).cache_ram == 0
    with pytest.raises(ValueError):
        _llama(cache_ram=-1)


@pytest.mark.parametrize("value", ["", "   ", "a\x00b", "a\nb", "a\rb"])
@pytest.mark.parametrize("field", ["served_name", "mmproj", "cache_type_k", "cache_type_v"])
def test_argv_bound_strings_must_be_clean_single_lines(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a non-empty single line"):
        _llama(**{field: value})


def test_api_key_file_must_be_a_clean_single_line():
    with pytest.raises(ValueError, match="api_key_file must be a non-empty single line"):
        _llama(api_key_file="/x/k\n", require_api_key=True)


def test_effort_outside_the_four_levels_is_refused():
    with pytest.raises(ValueError):
        _llama(effort="medium")
    for level in ("off", "low", "high", "max"):
        assert _llama(effort=level).effort == level


def test_effort_level_is_the_four_literals():
    assert set(EffortLevel.__args__) == {"off", "low", "high", "max"}


def test_empty_env_and_empty_config_ini_validate_on_llamacpp():
    srv = _llama(env={}, config_ini="")
    assert srv.env == {} and srv.config_ini == ""


def test_llamacpp_record_with_none_of_the_ten_matches_a_017_record():
    srv = _llama()
    assert all(getattr(srv, f) is None for f in NEW_FIELDS)
    assert srv.binary == LLAMA_SERVER and srv.launch == "binary"


# ---------------------------------------------------------------------------
# 0.17 records load unchanged
# ---------------------------------------------------------------------------


def test_a_017_vllm_binary_record_validates_with_new_fields_none():
    srv = ManagedServerConfig(
        binary="/opt/vllm", model="org/repo", port=8081,
        extra_args=["--max-model-len", "65536"], bind_all=True)
    assert all(getattr(srv, f) is None for f in NEW_FIELDS)
    assert srv.extra_args == ["--max-model-len", "65536"] and srv.bind_all is True


def test_a_017_vllm_docker_record_validates_with_new_fields_none():
    srv = ManagedServerConfig(launch="docker", docker_image="img", model="org/repo")
    assert all(getattr(srv, f) is None for f in NEW_FIELDS)


def test_an_attach_only_config_with_no_server_loads(tmp_path):
    loader, _ = _layers(tmp_path)
    cfg = loader.load_harness()
    assert cfg.server is None and cfg.extra_endpoints == []


# ---------------------------------------------------------------------------
# Peer binary inheritance (decided once, at load)
# ---------------------------------------------------------------------------

_PROVIDER = ProviderConfig(
    provider_type="vllm", base_url="http://127.0.0.1:9/v1", default_model="m")


def _peer(name="p", **lifecycle) -> EndpointRef:
    return EndpointRef(
        name=name, base_url="http://127.0.0.1:8082/v1", provider_type="llamacpp", gpu=True,
        lifecycle=ManagedServerConfig(runtime="llamacpp", model="/x/p.gguf", **lifecycle))


def _harness(server, *peers) -> HarnessConfig:
    return HarnessConfig(provider=_PROVIDER, server=server, extra_endpoints=list(peers))


def test_binary_less_peer_takes_a_llamacpp_primarys_binary():
    cfg = _harness(_llama(), _peer())
    assert cfg.extra_endpoints[0].lifecycle.binary == LLAMA_SERVER


def test_binary_less_peer_with_empty_binary_takes_the_primarys():
    cfg = _harness(_llama(), _peer(binary=""))
    assert cfg.extra_endpoints[0].lifecycle.binary == LLAMA_SERVER


@pytest.mark.parametrize("own", ["/other/llama-server", LLAMA_SERVER])
def test_a_peer_with_its_own_binary_keeps_it(own):
    cfg = _harness(_llama(), _peer(binary=own))
    assert cfg.extra_endpoints[0].lifecycle.binary == own


def test_a_vllm_primary_lends_no_binary():
    primary = ManagedServerConfig(binary="/opt/vllm", model="org/repo")
    cfg = _harness(primary, _peer())
    assert cfg.extra_endpoints[0].lifecycle.binary is None


def test_a_declared_only_primary_lends_nothing():
    primary = ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf")
    cfg = _harness(primary, _peer())
    assert cfg.extra_endpoints[0].lifecycle.binary is None


def test_no_primary_server_means_no_inheritance():
    cfg = _harness(None, _peer())
    assert cfg.extra_endpoints[0].lifecycle.binary is None


def test_a_vllm_peer_does_not_inherit_a_llamacpp_binary():
    peer = EndpointRef(
        name="v", base_url="http://127.0.0.1:8081/v1", provider_type="vllm", gpu=True,
        lifecycle=ManagedServerConfig(
            runtime="vllm", launch="docker", docker_image="img", model="org/repo"))
    cfg = _harness(_llama(), peer)
    assert cfg.extra_endpoints[0].lifecycle.binary is None


def test_an_attach_only_peer_is_untouched():
    peer = EndpointRef(name="a", base_url="http://127.0.0.1:8083/v1")
    cfg = _harness(_llama(), peer)
    assert cfg.extra_endpoints[0].lifecycle is None


def test_inheritance_assigns_in_place_so_identity_holds():
    peer = _peer()
    lifecycle = peer.lifecycle
    cfg = _harness(_llama(), peer)
    assert cfg.extra_endpoints[0].lifecycle is lifecycle
