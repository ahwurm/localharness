"""The loader's one `extra_endpoints` shadow line (Phase 5 D-40).

A global overrides.yaml `extra_endpoints` replaces config.yaml's list wholesale; when that hides a
NON-EMPTY list the loader says so once through `harness_warnings` (the path `start` prints). The
`[]` an `init`-written config.yaml holds shadows nothing and stays silent.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from localharness.config.loader import ConfigLoader

PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m"}


def _peer(name: str) -> dict:
    return {"name": name, "base_url": "http://127.0.0.1:8001/v1", "provider_type": "llamacpp"}


def _loader(tmp_path: Path, *, config_extra: dict, overrides: dict | None) -> ConfigLoader:
    g = tmp_path / "global"
    g.mkdir()
    (g / "config.yaml").write_text(
        yaml.safe_dump({"version": "1", "provider": PROVIDER, **config_extra}), encoding="utf-8")
    if overrides is not None:
        (g / "overrides.yaml").write_text(yaml.safe_dump(overrides), encoding="utf-8")
    return ConfigLoader(config_dir=g)


LINE = "overrides.yaml extra_endpoints replaces the {n} peers in config.yaml"


def test_a_hidden_non_empty_list_is_said_once(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a"), _peer("b")]},
                     overrides={"extra_endpoints": [_peer("c")]})
    loader.load_harness()
    assert loader.harness_warnings == [LINE.format(n=2)]


def test_the_count_is_the_config_yaml_peers(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a")]},
                     overrides={"extra_endpoints": [_peer("c"), _peer("d"), _peer("e")]})
    loader.load_harness()
    assert loader.harness_warnings == [LINE.format(n=1)]


def test_the_count_is_the_peers_that_vanish_by_name(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a"), _peer("b"), _peer("c")]},
                     overrides={"extra_endpoints": [_peer("b"), _peer("z")]})
    loader.load_harness()
    assert loader.harness_warnings == [LINE.format(n=2)]


def test_an_overrides_list_that_carries_every_config_peer_is_silent(tmp_path):
    """WR-08: the peer-posture flow and init's import copy config.yaml's list; nothing vanishes."""
    for over in ([_peer("a"), _peer("b")], [_peer("b"), _peer("a"), _peer("extra")]):
        (tmp_path / str(len(over))).mkdir()
        loader = _loader(tmp_path / str(len(over)), config_extra={"extra_endpoints": [_peer("a"), _peer("b")]},
                         overrides={"extra_endpoints": over})
        loader.load_harness()
        assert loader.harness_warnings == []


def test_an_empty_overrides_list_hides_every_config_peer(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a")]}, overrides={"extra_endpoints": []})
    loader.load_harness()
    assert loader.harness_warnings == [LINE.format(n=1)]


def test_an_empty_config_list_is_silent(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": []},
                     overrides={"extra_endpoints": [_peer("c")]})
    loader.load_harness()
    assert loader.harness_warnings == []


def test_no_config_list_is_silent(tmp_path):
    loader = _loader(tmp_path, config_extra={}, overrides={"extra_endpoints": [_peer("c")]})
    loader.load_harness()
    assert loader.harness_warnings == []


def test_overrides_without_the_key_is_silent(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a"), _peer("b")]},
                     overrides={"agent": {}, "org": {}})
    loader.load_harness()
    assert loader.harness_warnings == []


def test_no_overrides_file_is_silent(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a")]}, overrides=None)
    loader.load_harness()
    assert loader.harness_warnings == []


def test_a_second_load_does_not_double_the_line(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a"), _peer("b")]},
                     overrides={"extra_endpoints": [_peer("c")]})
    first = loader.load_harness()
    assert loader.load_harness() is first
    assert loader.harness_warnings == [LINE.format(n=2)]


def test_the_overrides_list_wins(tmp_path):
    loader = _loader(tmp_path, config_extra={"extra_endpoints": [_peer("a"), _peer("b")]},
                     overrides={"extra_endpoints": [_peer("c")]})
    assert [p.name for p in loader.load_harness().extra_endpoints] == ["c"]
