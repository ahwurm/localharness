"""A withheld launch entry is enforced by the loader for both kinds, by the entry's name.

D-12/D-13/D-14 of the machine-trust phase: a withheld `server.bind_all` / `server.require_api_key`
narrows the launch (loopback, keyed) instead of dropping `server:`; every other withheld `server.*`
entry (kind `launch` or `launch_model`) still drops `server:`; a withheld
`extra_endpoints.<peer>.lifecycle.<key>` leaves every peer of that name attach-only (or narrowed for
the two posture keys). The raw dicts the loader was handed are never mutated.
"""
from __future__ import annotations

import copy

import pytest

from localharness.config import loader as loader_mod
from localharness.config.loader import ConfigLoader
from tests.unit.trust_support import g, write_yaml  # noqa: F401 — g is a fixture

PROVIDER = {"base_url": "http://127.0.0.1:8080/v1", "provider_type": "llamacpp", "default_model": "m"}
SERVER = {"runtime": "llamacpp", "binary": "/x", "model": "/m", "bind_all": True}


def _cfg(server=None, provider=None, **extra) -> dict:
    cfg: dict = {"provider": provider or dict(PROVIDER), **extra}
    if server is not None:
        cfg["server"] = server
    return cfg


def _load(g, withheld):
    return ConfigLoader(config_dir=g, machine_withheld=withheld).load_harness()


def _peer(name, base_url="http://127.0.0.1:8081/v1", **lifecycle) -> dict:
    return {"name": name, "base_url": base_url, "gpu": True,
            "lifecycle": {"runtime": "llamacpp", "binary": "/x/b", "model": "/m/b.gguf", **lifecycle}}


def _by_name(cfg, name):
    return [p for p in cfg.extra_endpoints if p.name == name]


# ------------------------------------------------------------------ server.bind_all (D-12)


def test_a_withheld_bind_all_loads_on_loopback_and_keeps_the_server(g):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER)))

    cfg = _load(g, {"config.yaml": {("launch", "server.bind_all")}})

    assert cfg.server is not None and cfg.server.bind_all is False
    assert cfg.server.binary == "/x"
    assert cfg.server.model == "/m"


def test_a_withheld_bind_all_on_a_server_that_lacks_the_key_still_loads_false(g):
    server = {k: v for k, v in SERVER.items() if k != "bind_all"}
    write_yaml(g / "config.yaml", _cfg(server))

    cfg = _load(g, {"config.yaml": {("launch", "server.bind_all")}})

    assert cfg.server is not None and cfg.server.bind_all is False


def test_the_overrides_file_is_narrowed_in_place(g):
    write_yaml(g / "config.yaml", _cfg({"runtime": "llamacpp", "binary": "/x", "model": "/m"}))
    write_yaml(g / "overrides.yaml", {"server": {"bind_all": True}})

    cfg = _load(g, {"overrides.yaml": {("launch", "server.bind_all")}})

    assert cfg.server is not None and cfg.server.bind_all is False
    assert cfg.server.binary == "/x"


def test_a_withheld_bind_all_in_the_overrides_narrows_a_true_in_the_config(g):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER)))
    write_yaml(g / "overrides.yaml", {"server": {"bind_all": True}})

    cfg = _load(g, {"overrides.yaml": {("launch", "server.bind_all")}})

    assert cfg.server is not None
    assert cfg.server.bind_all is False, "the overrides.yaml put wins the merge"


# ------------------------------------------------------------------ server.require_api_key (D-12)


def test_a_withheld_require_api_key_loads_true_and_a_missing_key_source_closes_bind_all(g):
    write_yaml(g / "config.yaml", _cfg({**SERVER, "require_api_key": False}))

    cfg = _load(g, {"config.yaml": {("launch", "server.require_api_key")}})

    assert cfg.server is not None
    assert cfg.server.require_api_key is True
    assert cfg.server.bind_all is False, "keyed with no key source: loopback, not every interface"
    assert cfg.server.binary == "/x"


def test_a_withheld_require_api_key_keeps_bind_all_when_a_provider_key_exists(g):
    write_yaml(g / "config.yaml", _cfg({**SERVER, "require_api_key": False},
                                       provider={**PROVIDER, "api_key": "abc"}))

    cfg = _load(g, {"config.yaml": {("launch", "server.require_api_key")}})

    assert cfg.server is not None
    assert cfg.server.require_api_key is True
    assert cfg.server.bind_all is True


def test_a_withheld_require_api_key_keeps_bind_all_when_a_key_file_exists(g):
    write_yaml(g / "config.yaml", _cfg({**SERVER, "require_api_key": True, "api_key_file": "/k"}))

    cfg = _load(g, {"config.yaml": {("launch", "server.require_api_key")}})

    assert cfg.server is not None and cfg.server.require_api_key is True
    assert cfg.server.bind_all is True


def test_a_withheld_require_api_key_keeps_bind_all_when_the_runtime_key_variable_is_set(
        g, monkeypatch):
    from localharness.provider.server import KEY_ENV

    monkeypatch.setenv(KEY_ENV["llamacpp"], "from-the-env")
    write_yaml(g / "config.yaml", _cfg({**SERVER, "require_api_key": False}))

    cfg = _load(g, {"config.yaml": {("launch", "server.require_api_key")}})

    assert cfg.server is not None and cfg.server.bind_all is True


def test_the_other_files_bind_all_true_is_closed_too_when_there_is_no_key_source(g):
    write_yaml(g / "config.yaml", _cfg({**SERVER, "require_api_key": False}))
    write_yaml(g / "overrides.yaml", {"server": {"bind_all": True}})

    cfg = _load(g, {"config.yaml": {("launch", "server.require_api_key")}})

    assert cfg.server is not None and cfg.server.require_api_key is True
    assert cfg.server.bind_all is False, "a true in the winning file must not open every interface"


def test_the_loader_never_mutates_the_dicts_the_yaml_reader_returned(g, monkeypatch):
    path = write_yaml(g / "config.yaml", _cfg({**SERVER, "require_api_key": False}))
    cache: dict[str, dict] = {}
    real = loader_mod._load_yaml_file

    def cached(p):
        return cache.setdefault(str(p), real(p))

    monkeypatch.setattr(loader_mod, "_load_yaml_file", cached)
    before = copy.deepcopy(real(path))

    _load(g, {"config.yaml": {("launch", "server.bind_all"), ("launch", "server.require_api_key")}})

    assert cache[str(path)] == before


# ------------------------------------------------------------------ every other launch entry drops server: (D-14)


@pytest.mark.parametrize("kind,name", [
    ("launch_model", "server.model"),
    ("launch", "server.env"),
    ("launch", "server.runtime"),
    ("launch", "server.binary"),
    ("launch", "server.docker_image"),
    ("launch", "server.extra_args"),
    ("launch", "server.api_key_file"),
    ("launch", "server.launch"),
    ("launch", "server.local_models"),
    ("launch", "server.config_ini"),
    ("launch_model", "server.served_name"),
])
def test_every_other_withheld_server_entry_drops_the_section(g, kind, name):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER)))

    assert _load(g, {"config.yaml": {(kind, name)}}).server is None


def test_a_withheld_launch_model_entry_in_the_overrides_drops_the_section_of_both_files(g):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER)))
    write_yaml(g / "overrides.yaml", {"server": {"model": "/planted"}})

    assert _load(g, {"overrides.yaml": {("launch_model", "server.model")}}).server is None


def test_a_withheld_posture_entry_beside_a_dropped_key_still_drops_the_section(g):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER)))

    cfg = _load(g, {"config.yaml": {("launch", "server.bind_all"), ("launch", "server.binary")}})

    assert cfg.server is None


def test_nothing_withheld_leaves_the_record_as_written(g):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER)))

    cfg = _load(g, {})

    assert cfg.server is not None and cfg.server.bind_all is True


# ------------------------------------------------------------------ per-peer lifecycle (D-13)


def test_a_withheld_peer_model_removes_exactly_that_peers_lifecycle(g):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        _peer("qwen3.6", "http://127.0.0.1:8081/v1"), _peer("qwen3", "http://127.0.0.1:8082/v1")]})
    write_yaml(g / "config.yaml", _cfg(dict(SERVER)))

    cfg = _load(g, {"overrides.yaml": {("launch_model", "extra_endpoints.qwen3.6.lifecycle.model")}})

    (dotted,) = _by_name(cfg, "qwen3.6")
    (plain,) = _by_name(cfg, "qwen3")
    assert dotted.lifecycle is None and dotted.gpu is True
    assert plain.lifecycle is not None and plain.lifecycle.model == "/m/b.gguf"
    assert cfg.server is not None, "a peer entry never drops the primary's server:"


def test_two_peers_sharing_a_name_both_lose_their_lifecycle(g):
    write_yaml(g / "config.yaml", _cfg(extra_endpoints=[
        _peer("p", "http://127.0.0.1:8081/v1"), _peer("p", "http://127.0.0.1:8082/v1")]))

    cfg = _load(g, {"config.yaml": {("launch", "extra_endpoints.p.lifecycle.binary")}})

    assert [p.lifecycle for p in cfg.extra_endpoints] == [None, None]


@pytest.mark.parametrize("key,value", [("bind_all", False), ("require_api_key", True)])
def test_a_withheld_peer_posture_key_is_narrowed_by_name_and_the_lifecycle_survives(g, key, value):
    write_yaml(g / "config.yaml", _cfg(extra_endpoints=[
        _peer("p", "http://127.0.0.1:8081/v1", bind_all=True, require_api_key=False),
        _peer("p", "http://127.0.0.1:8082/v1", bind_all=True, require_api_key=False),
        _peer("other", "http://127.0.0.1:8083/v1", bind_all=True, require_api_key=False)]))

    cfg = _load(g, {"config.yaml": {("launch", f"extra_endpoints.p.lifecycle.{key}")}})

    for peer in _by_name(cfg, "p"):
        assert peer.lifecycle is not None and getattr(peer.lifecycle, key) is value
        assert peer.lifecycle.binary == "/x/b"
    (other,) = _by_name(cfg, "other")
    assert other.lifecycle is not None
    assert other.lifecycle.bind_all is True and other.lifecycle.require_api_key is False


def test_a_peer_withheld_under_the_shadowed_file_still_applies_to_the_winning_list(g):
    write_yaml(g / "config.yaml", _cfg(extra_endpoints=[_peer("p")]))
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        _peer("p", "http://127.0.0.1:9000/v1", bind_all=True), _peer("q", "http://127.0.0.1:9001/v1")]})

    cfg = _load(g, {"config.yaml": {("launch_model", "extra_endpoints.p.lifecycle.model"),
                                    ("launch", "extra_endpoints.q.lifecycle.bind_all")}})

    (p,) = _by_name(cfg, "p")
    (q,) = _by_name(cfg, "q")
    assert p.base_url == "http://127.0.0.1:9000/v1", "the overrides list is the one that loaded"
    assert p.lifecycle is None
    assert q.lifecycle is not None and q.lifecycle.bind_all is False


def test_a_peer_name_holding_dot_lifecycle_dot_is_recovered_from_the_last_occurrence(g):
    write_yaml(g / "config.yaml", _cfg(extra_endpoints=[
        _peer("a.lifecycle.b", "http://127.0.0.1:8081/v1"), _peer("a", "http://127.0.0.1:8082/v1")]))

    cfg = _load(g, {"config.yaml": {("launch_model", "extra_endpoints.a.lifecycle.b.lifecycle.model")}})

    (odd,) = _by_name(cfg, "a.lifecycle.b")
    (plain,) = _by_name(cfg, "a")
    assert odd.lifecycle is None
    assert plain.lifecycle is not None


def test_a_withheld_peer_entry_does_not_drop_the_primary_server(g):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER), extra_endpoints=[_peer("p")]))

    cfg = _load(g, {"config.yaml": {("launch", "extra_endpoints.p.lifecycle.extra_args")}})

    assert cfg.server is not None and cfg.server.bind_all is True
    assert cfg.extra_endpoints[0].lifecycle is None


def test_a_withheld_peer_that_is_absent_changes_nothing(g):
    write_yaml(g / "config.yaml", _cfg(extra_endpoints=[_peer("p")]))

    cfg = _load(g, {"config.yaml": {("launch", "extra_endpoints.gone.lifecycle.binary")}})

    assert cfg.extra_endpoints[0].lifecycle is not None


def test_a_malformed_peer_name_without_the_lifecycle_marker_is_ignored(g):
    write_yaml(g / "config.yaml", _cfg(dict(SERVER), extra_endpoints=[_peer("p")]))

    cfg = _load(g, {"config.yaml": {("launch", "extra_endpoints.p")}})

    assert cfg.server is not None
    assert cfg.extra_endpoints[0].lifecycle is not None


def test_peer_withholding_never_mutates_the_dicts_the_yaml_reader_returned(g, monkeypatch):
    path = write_yaml(g / "config.yaml", _cfg(extra_endpoints=[_peer("p", bind_all=True)]))
    cache: dict[str, dict] = {}
    real = loader_mod._load_yaml_file

    def cached(p):
        return cache.setdefault(str(p), real(p))

    monkeypatch.setattr(loader_mod, "_load_yaml_file", cached)
    before = copy.deepcopy(real(path))

    _load(g, {"config.yaml": {("launch", "extra_endpoints.p.lifecycle.bind_all"),
                              ("launch_model", "extra_endpoints.p.lifecycle.model")}})

    assert cache[str(path)] == before
