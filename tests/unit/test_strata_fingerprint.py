"""A Strata record's config and code are `launch` entries, hashed and never run, re-hashed only when a stat triple moves."""
from __future__ import annotations

import os

import pytest
import yaml

from localharness.cli.library import _adopt_written
from localharness.cli.workspace import decide_machine_trust, uncovered
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


def _uncovered(g) -> list[dict]:
    return uncovered(trust.recorded_machine(g)["entries"], trust.machine_snapshot(g))


def test_a_touch_of_the_engine_files_asks_nothing(g, engine):
    _record(g)
    for rel in trust.STRATA_CODE_FILES:
        os.utime(engine / rel, ns=(1, 1))

    assert _uncovered(g) == []


def test_a_file_whose_triple_is_unchanged_is_not_rehashed(g, engine, monkeypatch):
    _record(g)

    def refuse(path):
        raise AssertionError(f"re-hashed {path}")

    monkeypatch.setattr(trust, "_sha256_file", refuse)

    assert _uncovered(g) == []


def test_a_replaced_file_with_equal_size_and_mtime_is_rehashed(g, engine, monkeypatch):
    target = engine / "engine" / "strata"
    target.write_bytes(b"abcd")
    _record(g)
    old = target.stat()
    hashed: list = []
    real = trust._sha256_file
    monkeypatch.setattr(trust, "_sha256_file", lambda p: hashed.append(p) or real(p))

    def replace(data: bytes) -> None:
        fresh = target.with_name("fresh")
        fresh.write_bytes(data)
        os.utime(fresh, ns=(old.st_atime_ns, old.st_mtime_ns))
        os.replace(fresh, target)

    replace(target.read_bytes())
    assert _uncovered(g) == [] and hashed == [target]

    replace(b"x" * old.st_size)
    (changed,) = _uncovered(g)
    assert changed["name"] == CODE and changed["shown"].endswith("(engine/strata)")


def test_an_in_place_rewrite_with_the_mtime_restored_is_rehashed(g, engine):
    target = engine / "engine" / "strata"
    target.write_bytes(b"AAAA")
    held = {e["sha256"] for e in _record(g) if e.get("sha256")}
    old = target.stat()

    with open(target, "r+b") as fh:
        fh.write(b"BBBB")
    os.utime(target, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert target.stat().st_mtime_ns == old.st_mtime_ns and target.stat().st_ino == old.st_ino

    (changed,) = _uncovered(g)
    assert changed["name"] == CODE and changed["sha256"] not in held


def test_a_changed_config_is_an_uncovered_engine_config_entry(g, engine):
    _record(g)
    (engine / CONFIG_NAME).write_text('{"model_name": "other"}', encoding="utf-8")

    (changed,) = _uncovered(g)
    assert changed["name"] == f"extra_endpoints.{PEER}.lifecycle.engine_config" and "sha256" in changed


def test_a_missing_engine_file_is_unreadable_and_uncovered(g, engine):
    _record(g)
    (engine / "engine" / "strata").unlink()

    (gone,) = _uncovered(g)
    assert gone["shown"] == "unreadable (engine/strata)" and "sha256" not in gone


def test_a_symlinked_engine_file_is_unreadable(g, engine):
    target = engine / "serve" / "server.py"
    target.rename(engine / "serve" / "real.py")
    target.symlink_to(engine / "serve" / "real.py")

    assert "unreadable (serve/server.py)" in {e["shown"] for e in _code(trust.machine_snapshot(g))}


@pytest.mark.parametrize("runtime", ["llamacpp", "vllm"])
def test_other_runtimes_yield_no_strata_entry(g, runtime):
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [
        {"name": "p", "base_url": "http://127.0.0.1:8081/v1", "gpu": True,
         "lifecycle": {"runtime": runtime, "binary": "/x/b", "model": "/m/b.gguf"}}]})

    assert not [e for e in trust.machine_snapshot(g) if e["name"].endswith(("strata_code", "engine_config"))]


def test_a_set_up_strata_peer_is_covered_at_the_next_start(g, engine):
    _record(g)
    peer = {"name": "second-peer", "base_url": "http://127.0.0.1:8083/v1", "provider_type": "strata", "gpu": True,
            "lifecycle": {"runtime": "strata", "engine_dir": str(engine), "engine_config": CONFIG_NAME,
                          "model": "/m/a.gguf", "served_name": "s", "port": 8084}}
    overrides = yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))
    write_yaml(g / "overrides.yaml", {"extra_endpoints": [*overrides["extra_endpoints"], peer]})
    assert any(e["name"].endswith("strata_code") for e in _uncovered(g))

    _adopt_written(g, "overrides.yaml", None, [peer])

    assert _uncovered(g) == []
