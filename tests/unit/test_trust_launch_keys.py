"""Every launch-affecting key of `server:` and of each peer `lifecycle:` is a snapshot entry by name.

D-01/D-02 of the machine-trust phase: the one key table (`LAUNCH_KEYS`), the model keys as their own
kind (`launch_model`), peers named by `name` and never by position, and a peer's secrets absent from
the record by construction.
"""
from __future__ import annotations

import json
import shutil

import pytest

from localharness.config import trust
from tests.unit.launch_support import record_opens
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


# ------------------------------------------------------------------ posture entries (D-03, D-04)


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    for var in ("LLAMA_API_KEY", "VLLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _posture(snap: list[dict], key: str) -> list[dict]:
    return [e for e in snap if e["name"] == f"server.{key}"]


def test_a_posture_entry_is_emitted_once_from_the_merged_view(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "llamacpp", "binary": "/x", "model": "/m",
                                              "bind_all": True}})
    write_yaml(g / "overrides.yaml", {"server": {"model": "/m2"}})

    entry = by_name(trust.machine_snapshot(g), "server.bind_all")

    assert entry == {"file": "config.yaml", "kind": "launch", "name": "server.bind_all",
                     "shown": "llamacpp:true:open"}


def test_the_file_whose_value_wins_owns_the_posture_entry(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "llamacpp", "bind_all": True}})
    write_yaml(g / "overrides.yaml", {"server": {"bind_all": False}})

    entry = by_name(trust.machine_snapshot(g), "server.bind_all")

    assert entry["file"] == "overrides.yaml" and entry["shown"] == "llamacpp:false:open"


def test_the_merged_runtime_is_the_one_shown(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "llamacpp", "bind_all": True}})
    write_yaml(g / "overrides.yaml", {"server": {"runtime": "vllm"}})

    entry = by_name(trust.machine_snapshot(g), "server.bind_all")

    assert entry["file"] == "config.yaml" and entry["shown"] == "vllm:true:open"


def test_the_runtime_defaults_to_vllm(g):
    write_yaml(g / "config.yaml", {"server": {"bind_all": True}})

    assert by_name(trust.machine_snapshot(g), "server.bind_all")["shown"] == "vllm:true:open"


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio"])
def test_an_ollama_or_lmstudio_bind_all_is_no_posture_entry(g, runtime):
    write_yaml(g / "config.yaml", {"server": {"runtime": runtime, "binary": "/x", "bind_all": True,
                                              "require_api_key": True}})

    got = [e["name"] for e in trust.machine_snapshot(g)]

    assert got == ["server.binary", "server.runtime"]


def test_require_api_key_is_its_own_posture_entry_with_the_same_state(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm", "bind_all": True,
                                              "require_api_key": True, "api_key_file": "/k"}})

    snap = trust.machine_snapshot(g)

    assert by_name(snap, "server.require_api_key")["shown"] == "vllm:true:keyed"
    assert by_name(snap, "server.bind_all")["shown"] == "vllm:true:keyed"
    assert by_name(snap, "server.api_key_file")["shown"] == "/k"


def test_a_provider_key_in_either_file_makes_it_keyed(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "llamacpp", "require_api_key": True}})
    write_yaml(g / "overrides.yaml", {"provider": {"api_key": "abc"}})

    assert by_name(trust.machine_snapshot(g), "server.require_api_key")["shown"] == "llamacpp:true:keyed"

    write_yaml(g / "overrides.yaml", {})
    write_yaml(g / "config.yaml", {"provider": {"api_key": "abc"},
                                   "server": {"runtime": "llamacpp", "require_api_key": True}})

    assert by_name(trust.machine_snapshot(g), "server.require_api_key")["shown"] == "llamacpp:true:keyed"


def test_a_provider_key_of_none_is_open(g):
    write_yaml(g / "config.yaml", {"provider": {"api_key": "none"},
                                   "server": {"runtime": "llamacpp", "require_api_key": True}})

    assert by_name(trust.machine_snapshot(g), "server.require_api_key")["shown"] == "llamacpp:true:open"


def test_the_runtimes_key_variable_in_the_environment_makes_it_keyed(g, monkeypatch):
    write_yaml(g / "config.yaml", {"server": {"runtime": "llamacpp", "require_api_key": True}})

    monkeypatch.setenv("LLAMA_API_KEY", "k")
    assert by_name(trust.machine_snapshot(g), "server.require_api_key")["shown"] == "llamacpp:true:keyed"

    monkeypatch.delenv("LLAMA_API_KEY")
    assert by_name(trust.machine_snapshot(g), "server.require_api_key")["shown"] == "llamacpp:true:open"


def test_a_key_source_without_require_api_key_is_still_open(g):
    write_yaml(g / "config.yaml", {"server": {"runtime": "llamacpp", "bind_all": True,
                                              "require_api_key": False, "api_key_file": "/k"},
                                   "provider": {"api_key": "abc"}})

    assert by_name(trust.machine_snapshot(g), "server.bind_all")["shown"] == "llamacpp:true:open"


def test_the_snapshot_never_opens_a_key_file(g, tmp_path):
    key = tmp_path / "key"
    key.write_text("topsecret", encoding="utf-8")
    write_yaml(g / "config.yaml", {"server": {"runtime": "llamacpp", "bind_all": True,
                                              "require_api_key": True, "api_key_file": str(key)}})

    with record_opens() as opened:
        snap = trust.machine_snapshot(g)

    assert str(key) not in opened
    assert by_name(snap, "server.bind_all")["shown"] == "llamacpp:true:keyed"
    assert "topsecret" not in json.dumps(snap)


def _lifecycle_peer(name="p", api_key=None, **lifecycle) -> dict:
    peer = {"name": name, "base_url": "http://127.0.0.1:8081/v1", "gpu": True,
            "lifecycle": {"runtime": "vllm", **lifecycle}}
    if api_key is not None:
        peer["api_key"] = api_key
    return peer


def test_a_peers_posture_comes_from_the_list_that_wins(g):
    write_yaml(g / "config.yaml", {"extra_endpoints": [_lifecycle_peer(bind_all=False)]})
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        _lifecycle_peer(api_key="k", bind_all=True, require_api_key=True)]})

    snap = trust.machine_snapshot(g)

    entry = by_name(snap, "extra_endpoints.p.lifecycle.bind_all")
    assert entry == {"file": "overrides.yaml", "kind": "launch",
                     "name": "extra_endpoints.p.lifecycle.bind_all", "shown": "vllm:true:keyed"}
    assert "k" not in entry["shown"].split(":")


def test_a_peers_posture_falls_back_to_the_config_list(g):
    write_yaml(g / "config.yaml", {"extra_endpoints": [_lifecycle_peer(bind_all=True)]})
    write_yaml(g / "overrides.yaml", {"server": {"runtime": "ollama"}})

    entry = by_name(trust.machine_snapshot(g), "extra_endpoints.p.lifecycle.bind_all")

    assert entry["file"] == "config.yaml" and entry["shown"] == "vllm:true:open"


def test_a_peer_with_the_default_key_and_no_key_file_is_open(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        _lifecycle_peer(api_key="none", bind_all=True, require_api_key=True)]})

    assert by_name(trust.machine_snapshot(g),
                   "extra_endpoints.p.lifecycle.bind_all")["shown"] == "vllm:true:open"


def test_a_peer_does_not_borrow_the_primarys_provider_key(g):
    write_yaml(g / "config.yaml", {"provider": {"api_key": "abc"}})
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        _lifecycle_peer(bind_all=True, require_api_key=True)]})

    assert by_name(trust.machine_snapshot(g),
                   "extra_endpoints.p.lifecycle.bind_all")["shown"] == "vllm:true:open"


def test_a_peer_with_a_key_file_is_keyed(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        _lifecycle_peer(bind_all=True, require_api_key=True, api_key_file="/k")]})

    assert by_name(trust.machine_snapshot(g),
                   "extra_endpoints.p.lifecycle.bind_all")["shown"] == "vllm:true:keyed"


def test_an_ollama_peer_emits_no_posture_entry(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        _lifecycle_peer(runtime="ollama", bind_all=True)]})

    assert [e["name"] for e in trust.machine_snapshot(g)] == [
        "extra_endpoints.p.lifecycle.runtime"]


# ------------------------------------------------------------------ confirmed_at_upgrade (D-05)


def _entry(name: str, shown: str = "x", kind: str = "launch") -> dict:
    return {"file": "config.yaml", "kind": kind, "name": name, "shown": shown}


def test_confirmed_at_upgrade_selects_exactly_the_governing_launch_entries():
    snap = [
        _entry("server.runtime", "vllm"), _entry("server.launch", "docker"),
        _entry("server.bind_all", "vllm:true:keyed"), _entry("server.require_api_key", "vllm:true:keyed"),
        _entry("server.local_models", "[]"), _entry("server.binary", "/b"),
        _entry("server.model", "/m", kind="launch_model"),
        _entry("extra_endpoints.p.lifecycle.runtime", "llamacpp"),
        _entry("extra_endpoints.p.lifecycle.bind_all", "llamacpp:true:open"),
        _entry("server.env", "{}"), {"file": "x", "kind": "mcp_server", "name": "runtime", "shown": "r"},
    ]

    got = [e["name"] for e in trust.confirmed_at_upgrade(snap)]

    assert got == ["server.runtime", "server.launch", "server.bind_all", "server.require_api_key",
                   "server.local_models", "extra_endpoints.p.lifecycle.runtime"]


def test_a_llamacpp_bind_all_false_is_still_confirmed():
    got = trust.confirmed_at_upgrade([_entry("server.bind_all", "llamacpp:false:open")])

    assert [e["name"] for e in got] == ["server.bind_all"]


def test_confirmed_at_upgrade_is_pure_and_reads_no_disk(g):
    snap = [_entry("server.runtime", "vllm"), _entry("server.bind_all", "vllm:true:open")]
    shutil.rmtree(g)

    assert trust.confirmed_at_upgrade(snap) == trust.confirmed_at_upgrade(snap) == snap
    assert not g.exists()
