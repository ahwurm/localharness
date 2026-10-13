"""The selection ladder (spec 20 §8.1, D-01..D-05): `--preset` > `pending_preset` > `unit-target` >
`active_endpoint` > `server:`, each rung read in its rank and always handing back the record the
loaded config holds.

Offline: harnesses come from `ConfigLoader(config_dir=...)` over a tmp dir, trust values are
constructed, nothing is started and no service manager is touched.
"""
from __future__ import annotations

import pytest
import yaml

from localharness.cli import model_ops
from localharness.cli.model_ops import (
    RUNG_ACTIVE,
    RUNG_DEFAULT,
    RUNG_PENDING,
    RUNG_PRESET,
    RUNG_UNIT,
    Serving,
    ServingRefusal,
    resolve_serving,
    serving_records,
)
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.provider import units

PRIMARY = "qwen-27b-swift-1.5"


def _peer(name, port, *, served_name=None, extra_args=None, lifecycle=True) -> dict:
    ep = {"name": name, "base_url": f"http://127.0.0.1:{port}/v1", "provider_type": "llamacpp"}
    if lifecycle:
        lc = {"gpu": True, "runtime": "llamacpp", "model": f"/m/{name}.gguf", "port": port}
        if served_name:
            lc["served_name"] = served_name
        if extra_args:
            lc["extra_args"] = list(extra_args)
        ep["gpu"] = True
        ep["lifecycle"] = lc
    return ep


def _seed(home, *, peers=None, server=True, overrides=None) -> None:
    data = {
        "version": "1",
        "provider": {
            "provider_type": "llamacpp", "base_url": "http://127.0.0.1:8080/v1",
            "default_model": PRIMARY, "available_models": [PRIMARY],
        },
        "org": {"default_model": PRIMARY, "audit_log_path": str(home / "audit.jsonl")},
        "extra_endpoints": peers if peers is not None else [
            _peer("flash-next", 8081), _peer("gemma", 8082)],
    }
    if server:
        data["server"] = {"runtime": "llamacpp", "binary": "/x/llama-server",
                          "model": "/m/a.gguf", "served_name": PRIMARY, "port": 8080}
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    overlay = home / "overrides.yaml"
    if overrides is None:
        overlay.unlink(missing_ok=True)
    else:
        overlay.write_text(yaml.safe_dump(overrides), encoding="utf-8")


def _load(home, *, machine_withheld=None):
    return ConfigLoader(config_dir=home, machine_withheld=machine_withheld).load_harness()


def _active(peer_url="http://127.0.0.1:8081/v1", model="flash-next") -> dict:
    return {"name": "flash-next", "base_url": peer_url, "provider_type": "llamacpp", "model": model}


@pytest.fixture
def home(tmp_path):
    h = tmp_path / ".localharness"
    h.mkdir()
    return h


def _serve(home, preset=None, trust=None, **kw):
    return resolve_serving(_load(home, **kw), trust or MachineTrust(), preset, config_dir=home)


def test_the_tracer_walk_takes_one_rung_at_a_time(home):
    _seed(home, overrides={"pending_preset": "gemma", "active_endpoint": _active()})
    units.write_target(home, "flash-next")

    got = _serve(home)
    assert isinstance(got, Serving)
    assert (got.record.name, got.rung, got.explicit) == ("gemma", RUNG_PENDING, True)

    _seed(home, overrides={"active_endpoint": _active()})
    got = _serve(home)
    assert (got.record.name, got.rung, got.explicit) == ("flash-next", RUNG_UNIT, False)

    units.remove_target(home)
    got = _serve(home)
    assert (got.record.name, got.rung, got.explicit) == ("flash-next", RUNG_ACTIVE, False)

    _seed(home)
    got = _serve(home)
    assert (got.record.name, got.rung, got.explicit) == (PRIMARY, RUNG_DEFAULT, False)
    assert got.record.peer is None


def test_an_explicit_preset_outranks_every_store(home):
    _seed(home, overrides={"pending_preset": "flash-next", "active_endpoint": _active()})
    units.write_target(home, "flash-next")
    got = _serve(home, "gemma")
    assert (got.record.name, got.rung, got.explicit) == ("gemma", RUNG_PRESET, True)


def test_the_active_endpoint_rung_hands_back_the_peers_own_base_url(home):
    _seed(home, overrides={"active_endpoint": _active(peer_url="http://127.0.0.1:8081/v1/")})
    got = _serve(home)
    assert got.rung == RUNG_ACTIVE
    assert got.record.base_url == "http://127.0.0.1:8081/v1"
    assert got.record.peer is not None and got.record.peer.name == "flash-next"


def test_the_rung_names_are_the_store_names():
    assert (RUNG_PRESET, RUNG_PENDING, RUNG_UNIT, RUNG_ACTIVE, RUNG_DEFAULT) == (
        "--preset", "pending_preset", units.TARGET_NAME, "active_endpoint", "server:")


def test_serving_records_lists_the_primary_first_then_each_peer(home):
    _seed(home, peers=[_peer("flash-next", 8081), _peer("cpu", 11434, lifecycle=False)])
    records = serving_records(_load(home))
    assert [r.name for r in records] == [PRIMARY, "flash-next", "cpu"]
    assert records[0].prefix == "server" and records[0].peer is None and records[0].spec is not None
    assert records[1].prefix == "extra_endpoints.flash-next.lifecycle"
    assert records[2].spec is None


def test_a_store_contributes_a_name_not_a_url(home):
    # an active_endpoint row with a base_url nobody configured matches nothing by URL, and a
    # model no record serves matches nothing by model: the rung is passed over, never obeyed
    _seed(home, overrides={"active_endpoint": _active(peer_url="http://evil.example/v1", model="zzz")})
    got = _serve(home)
    assert (got.rung, got.record.name) == (RUNG_DEFAULT, PRIMARY)


def test_the_active_endpoint_falls_back_to_a_unique_served_model(home):
    _seed(home, overrides={"active_endpoint": _active(peer_url="http://moved.example/v1")})
    got = _serve(home)
    assert (got.rung, got.record.name) == (RUNG_ACTIVE, "flash-next")


def test_the_resolver_is_the_module_attribute_surfaces_import():
    assert model_ops.resolve_serving is resolve_serving
    assert not isinstance(ServingRefusal("x", "unknown"), Serving)
