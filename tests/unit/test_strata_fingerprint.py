"""A Strata record's config and code are `launch` entries, hashed and never run, re-hashed only when a stat triple moves."""
from __future__ import annotations

import pytest

from localharness.cli.workspace import decide_machine_trust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_tree
from tests.unit.trust_support import g, write_yaml  # noqa: F401 — g is a fixture

PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m"}
PEER = "strata-peer"
CODE = f"extra_endpoints.{PEER}.lifecycle.strata_code"


@pytest.fixture
def engine(tmp_path, g):
    models = tmp_path / "models"
    models.mkdir()
    shard, mmproj = flash_next(models)
    root = strata_tree(tmp_path, model=shard)
    lifecycle = {"runtime": "strata", "gpu": True, "engine_dir": str(root), "engine_config": CONFIG_NAME,
                 "model": str(shard), "mmproj": str(mmproj), "served_name": "swift-1.5-iq3_xxs", "port": 8083}
    write_yaml(g / "config.yaml", {"version": "1", "provider": PROVIDER})
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        {"name": PEER, "base_url": "http://127.0.0.1:8083/v1", "provider_type": "strata", "gpu": True,
         "lifecycle": lifecycle}]})
    return root


def _code(snap: list[dict]) -> list[dict]:
    return [e for e in snap if e["name"] == CODE]


def _record(g) -> list[dict]:
    snap = trust.machine_snapshot(g)
    trust.record_machine(g, snap)
    return snap


def test_a_changed_server_py_withholds_the_strata_peer(g, engine):
    _record(g)
    (engine / "serve" / "server.py").write_text("print('changed')\n", encoding="utf-8")

    got = decide_machine_trust(g, ask=False)
    peer = ConfigLoader(config_dir=g, machine_withheld=got.withheld).load_harness().extra_endpoints[0]

    assert ("launch", CODE) in got.withheld["overrides.yaml"]
    assert peer.name == PEER and peer.lifecycle is None


def test_an_unchanged_engine_asks_nothing(g, engine):
    _record(g)

    got = decide_machine_trust(g, ask=False)

    assert got.withheld == {}
    assert ConfigLoader(config_dir=g, machine_withheld=got.withheld).load_harness().extra_endpoints[0].lifecycle
