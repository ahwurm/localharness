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
