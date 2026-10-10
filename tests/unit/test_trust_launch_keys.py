"""Every launch-affecting key of `server:` and of each peer `lifecycle:` is a snapshot entry by name.

D-01/D-02 of the machine-trust phase: the one key table (`LAUNCH_KEYS`), the model keys as their own
kind (`launch_model`), peers named by `name` and never by position, and a peer's secrets absent from
the record by construction.
"""
from __future__ import annotations

import json

from localharness.config import trust

from tests.unit.trust_support import by_name, g, names, write_yaml  # noqa: F401 — g is a fixture

SERVER = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
          "served_name": "a", "context": 8192, "cache_ram": 0,
          "env": {"LLAMA_ARG_N_GPU_LAYERS": "99"}, "local_models": [], "effort": "high"}


def _peer(**lifecycle) -> dict:
    return {"name": "qwen3.6", "base_url": "http://127.0.0.1:8081/v1", "gpu": True,
            "api_key": "SECRET", "extra_headers": {"X": "Y"},
            "lifecycle": {"runtime": "llamacpp", "binary": "/x/b", "model": "/m/b.gguf", **lifecycle}}


# ------------------------------------------------------------------ the tables


def test_the_launch_key_table_is_the_eleven_keys_in_order():
    assert trust.LAUNCH_KEYS == (
        "binary", "docker_image", "extra_args", "runtime", "launch", "bind_all", "require_api_key",
        "api_key_file", "env", "local_models", "config_ini")


def test_the_launch_model_table_and_kind():
    assert trust.LAUNCH_MODEL_KEYS == (
        "model", "mmproj", "served_name", "context", "cache_type_k", "cache_type_v", "cache_ram")
    assert trust.LAUNCH_MODEL_KIND == "launch_model"
    assert "launch_model" in trust.MACHINE_KINDS


def test_effort_is_in_neither_table():
    assert "effort" not in trust.LAUNCH_KEYS and "effort" not in trust.LAUNCH_MODEL_KEYS


# ------------------------------------------------------------------ the primary


def test_the_primary_server_section_yields_launch_and_launch_model_entries(g):
    write_yaml(g / "config.yaml", {"server": SERVER})

    snap = trust.machine_snapshot(g)

    assert {(e["kind"], e["name"]) for e in snap} == {
        ("launch", "server.binary"), ("launch", "server.runtime"), ("launch", "server.env"),
        ("launch_model", "server.model"), ("launch_model", "server.served_name"),
        ("launch_model", "server.context"), ("launch_model", "server.cache_ram")}
    assert by_name(snap, "server.env")["shown"] == '{"LLAMA_ARG_N_GPU_LAYERS":"99"}'
    assert by_name(snap, "server.context")["shown"] == "8192"
    assert by_name(snap, "server.cache_ram")["shown"] == "0", "a zero is present, not absent"
    assert by_name(snap, "server.binary")["shown"] == "/x/llama-server"
    assert all(e["file"] == "config.yaml" for e in snap)


def test_an_empty_local_models_list_and_effort_yield_nothing(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm", "local_models": [], "effort": "low"}})

    assert [e["name"] for e in trust.machine_snapshot(g)] == ["server.runtime"]


def test_a_non_empty_local_models_list_is_one_compact_entry(g):
    write_yaml(g / "config.yaml", {"server": {"local_models": [{"name": "a", "path": "/m/a"}]}})

    assert by_name(trust.machine_snapshot(g), "server.local_models")["shown"] == (
        '[{"name":"a","path":"/m/a"}]')


def test_a_false_value_is_an_entry_and_an_absent_key_is_not(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "ollama", "launch": False}})

    got = [e["name"] for e in trust.machine_snapshot(g)]

    assert "server.launch" in got
    assert "server.binary" not in got and "server.docker_image" not in got
    assert by_name(trust.machine_snapshot(g), "server.launch")["shown"] == "false"


def test_a_config_with_no_server_section_yields_no_launch_entries(g):
    write_yaml(g / "config.yaml", {"org": {"permissions": {"mode": "default"}}})
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [{"name": "p", "base_url": "http://x/v1"}]})

    assert [e for e in trust.machine_snapshot(g) if e["kind"] in ("launch", "launch_model")] == []


# ------------------------------------------------------------------ peers


def test_a_peer_lifecycle_is_named_by_the_peer_and_carries_its_file(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [_peer()]})

    snap = trust.machine_snapshot(g)

    got = {(e["file"], e["kind"], e["name"]) for e in snap}
    assert got == {
        ("overrides.yaml", "launch", "extra_endpoints.qwen3.6.lifecycle.binary"),
        ("overrides.yaml", "launch", "extra_endpoints.qwen3.6.lifecycle.runtime"),
        ("overrides.yaml", "launch_model", "extra_endpoints.qwen3.6.lifecycle.model")}


def test_a_peers_secrets_never_reach_the_record(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [_peer()]})
    write_yaml(g / "config.yaml", {"extra_endpoints": [_peer()]})

    blob = json.dumps(trust.machine_snapshot(g))

    assert "SECRET" not in blob
    assert '"X"' not in blob and '"Y"' not in blob and "extra_headers" not in blob
    assert "api_key" not in blob.replace("api_key_file", "")


def test_a_peer_without_a_lifecycle_yields_no_launch_entry(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        {"name": "p", "base_url": "http://x/v1", "api_key": "k"}, "junk", {"lifecycle": {"model": "/m"}}]})

    assert [e for e in trust.machine_snapshot(g) if e["kind"].startswith("launch")] == []


def test_a_peer_name_containing_the_lifecycle_marker_is_embedded_verbatim(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        {"name": "a.lifecycle.b", "lifecycle": {"runtime": "vllm"}}]})

    assert [e["name"] for e in trust.machine_snapshot(g)] == [
        "extra_endpoints.a.lifecycle.b.lifecycle.runtime"]


def test_peers_are_named_by_name_not_by_position(g):
    one = {"name": "p", "lifecycle": {"binary": "/x/p"}}
    two = {"name": "q", "lifecycle": {"binary": "/x/q"}}
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [one, two]})
    first = trust.machine_snapshot(g)
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [two, one]})

    assert trust.machine_snapshot(g) == first


# ------------------------------------------------------------------ ordering


def test_the_snapshot_is_sorted_and_its_fingerprint_is_stable(g):
    write_yaml(g / "config.yaml", {"server": SERVER})
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [_peer()], "server": {"model": "/m/z.gguf"}})

    first, second = trust.machine_snapshot(g), trust.machine_snapshot(g)

    assert first == second and trust.fingerprint(first) == trust.fingerprint(second)
    assert names(first) == sorted(names(first)), "sorted by (file, kind, name)"
