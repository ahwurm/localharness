"""A Strata record: config vocabulary and one row of the runtime table; nothing here runs Strata."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import tuning
from localharness.provider.autosense import FIT_KEYS, fit_header
from localharness.provider.gguf_header import read_header
from localharness.provider.runtimes import RUNTIMES
from localharness.registry.catalogue import build_catalogue
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_record, strata_tree

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


def _refusal(**over) -> str:
    with pytest.raises(ValidationError) as caught:
        strata_record(Path("/e/Strata"), Path("/m/a.gguf"), Path("/m/p.gguf"), **over)
    return str(caught.value)


def _bare(**over) -> dict:
    return {"runtime": "strata", "model": "/m/a.gguf", "port": 8083, "engine_dir": "/e/Strata",
            "engine_config": CONFIG_NAME, "served_name": "swift", **over}


def test_a_llamacpp_record_keeps_mmproj_and_served_name():
    record = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/m/a.gguf",
                                 mmproj="/m/p.gguf", served_name="swift")
    assert (record.mmproj, record.served_name) == ("/m/p.gguf", "swift")


def test_a_vllm_record_with_mmproj_is_refused_with_the_llamacpp_only_message():
    with pytest.raises(ValidationError, match=r"runtime=vllm: mmproj is llama.cpp-only \(runtime: llamacpp\)"):
        ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model="org/m", mmproj="/m/p.gguf")


def test_a_strata_record_with_bind_all_is_refused():
    assert "bind_all and extra_args do not apply" in _refusal(bind_all=True)


def test_a_strata_record_with_extra_args_is_refused():
    assert "bind_all and extra_args do not apply" in _refusal(extra_args=["--x"])


@pytest.mark.parametrize("missing,said", [
    (("engine_dir",), "engine_dir"), (("engine_config",), "engine_config"), (("served_name",), "served_name"),
    (("engine_dir", "served_name"), "engine_dir and served_name"),
    (("engine_dir", "engine_config", "served_name"), "engine_dir, engine_config and served_name")])
def test_a_strata_record_without_its_required_fields_is_refused_naming_only_them(missing, said):
    fields = {k: v for k, v in _bare().items() if k not in missing}
    with pytest.raises(ValidationError) as caught:
        ManagedServerConfig(**fields)
    assert f"runtime=strata requires {said}" in str(caught.value)


@pytest.mark.parametrize("config", ["/etc/strata.json", "../strata.json", "a/../../strata.json"])
def test_a_strata_engine_config_outside_engine_dir_is_refused(config):
    with pytest.raises(ValidationError, match="engine_config must be a path inside engine_dir"):
        ManagedServerConfig(**_bare(engine_config=config))


def test_a_strata_record_takes_no_llamacpp_only_field():
    with pytest.raises(ValidationError, match=r"cache_type_k is llama.cpp-only \(runtime: llamacpp\)"):
        ManagedServerConfig(**_bare(cache_type_k="q8_0"))


def test_a_llamacpp_record_with_engine_dir_is_refused_as_strata_only():
    with pytest.raises(ValidationError, match=r"runtime=llamacpp: engine_dir is Strata-only \(runtime: strata\)"):
        ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/m/a.gguf", engine_dir="/e")
