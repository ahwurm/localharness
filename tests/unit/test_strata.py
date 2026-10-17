"""A Strata record: config vocabulary and one row of the runtime table; nothing here runs Strata."""
from __future__ import annotations

import yaml

from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig
from localharness.provider import tuning
from localharness.provider.autosense import FIT_KEYS, fit_header
from localharness.provider.gguf_header import read_header
from localharness.provider.runtimes import RUNTIMES
from localharness.registry.catalogue import build_catalogue
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_tree

PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m"}
PEER = "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-strata"


def _load(tmp_path, shard, mmproj, engine):
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "config.yaml").write_text(yaml.dump({"version": "1", "provider": PROVIDER}), encoding="utf-8")
    peer = {"name": PEER, "base_url": "http://127.0.0.1:8083/v1", "provider_type": "strata", "gpu": True,
            "lifecycle": {"runtime": "strata", "gpu": True, "engine_dir": str(engine),
                          "engine_config": CONFIG_NAME, "model": str(shard), "mmproj": str(mmproj),
                          "served_name": "swift-1.5-iq3_xxs", "port": 8083}}
    (global_dir / "overrides.yaml").write_text(yaml.dump({"extra_endpoints": [peer]}), encoding="utf-8")
    return ConfigLoader(config_dir=global_dir).load_harness()


def test_a_strata_record_loads_through_the_loader_into_its_row(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    shard, mmproj = flash_next(models)
    engine = strata_tree(tmp_path, model=shard)
    harness = _load(tmp_path, shard, mmproj, engine)

    peer = harness.extra_endpoints[0]
    assert (peer.name, peer.provider_type) == (PEER, "strata")
    assert RUNTIMES[peer.lifecycle.runtime].name == "Strata"
    header = fit_header(read_header(shard, FIT_KEYS))
    assert tuning.derive(peer.lifecycle, header, AutosenseConfig()) == tuning.NOTHING_DERIVED
    entries = build_catalogue(harness, overlays={}, tool_registry=None)
    assert {"server.engine_dir", "server.engine_config"} <= set(entries)
