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
    GoneSelection,
    RUNG_ACTIVE,
    RUNG_DEFAULT,
    RUNG_PENDING,
    RUNG_PRESET,
    RUNG_UNIT,
    Serving,
    ServingRefusal,
    clear_gone,
    record_withheld,
    resolve_preset,
    resolve_serving,
    serving_records,
)
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import load_overlay
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


# ----------------------------------------------------------------------------- name tiers (D-03)


def _name(home, name, **kw):
    return resolve_preset(_load(home, **kw), MachineTrust(), name)


def test_exact_name_resolves(home):
    _seed(home)
    assert _name(home, "flash-next").name == "flash-next"


def test_casefold_name_resolves(home):
    _seed(home)
    assert _name(home, "FLASH-NEXT").name == "flash-next"


def test_unique_prefix_resolves(home):
    _seed(home)
    assert _name(home, "fla").name == "flash-next"
    assert _name(home, "qwen-27b").name == PRIMARY


def test_ambiguous_prefix_is_refused_naming_both(home):
    _seed(home, server=False, peers=[_peer("qwen-a", 8081), _peer("qwen-b", 8082)])
    got = _name(home, "qwen")
    assert isinstance(got, ServingRefusal) and got.kind == "ambiguous" and got.exit_code == 2
    assert got.line == "preset 'qwen' is ambiguous — it matches qwen-a and qwen-b"


def test_an_exact_name_beats_a_longer_prefix_match(home):
    _seed(home, server=False, peers=[_peer("qwen", 8081), _peer("qwen-b", 8082)])
    assert _name(home, "qwen").name == "qwen"


def test_the_casefold_tier_runs_before_the_prefix_tier(home):
    _seed(home, server=False, peers=[_peer("Gem", 8081), _peer("gemma", 8082)])
    assert _name(home, "gem").name == "Gem"


def test_unknown_name_lists_the_presets_sorted(home):
    _seed(home)
    got = _name(home, "nope")
    assert isinstance(got, ServingRefusal) and got.kind == "unknown" and got.exit_code == 2
    assert got.line == "unknown preset 'nope' — presets: flash-next, gemma, qwen-27b-swift-1.5"


def test_a_served_name_resolves_too(home):
    _seed(home, peers=[_peer("flash-next", 8081, extra_args=["-a", "swift"]), _peer("gemma", 8082)])
    got = _name(home, "swift")
    assert got.name == "flash-next" and got.served == "swift"


def test_an_attach_only_peer_is_not_a_preset(home):
    _seed(home, peers=[_peer("flash-next", 8081), _peer("cpu", 11434, lifecycle=False)])
    got = _name(home, "cpu")
    assert isinstance(got, ServingRefusal) and got.kind == "unknown"
    assert "cpu" not in got.line.split("presets:")[1]
    assert [r.name for r in serving_records(_load(home))][-1] == "cpu"


# ----------------------------------------------------------------------- withheld vs gone (D-04)

PEER_WITHHELD = {"config.yaml": frozenset({("launch_model", "extra_endpoints.flash-next.lifecycle.model")})}
LINE = "Not applying 1 change(s) in /g that you have not confirmed (x): you said No"


def _withheld_trust(withheld=None):
    return MachineTrust(withheld=withheld or PEER_WITHHELD, line=LINE)


def test_withheld_stored_pending_preset_is_refused_and_clears_nothing(home):
    _seed(home, overrides={"pending_preset": "flash-next"})
    before = (home / "overrides.yaml").read_bytes()
    got = resolve_serving(_load(home, machine_withheld=PEER_WITHHELD), _withheld_trust(), None,
                          config_dir=home)
    assert isinstance(got, ServingRefusal) and (got.kind, got.line) == ("withheld", LINE)
    assert (home / "overrides.yaml").read_bytes() == before


def test_withheld_explicit_preset_is_refused(home):
    _seed(home)
    got = resolve_serving(_load(home, machine_withheld=PEER_WITHHELD), _withheld_trust(),
                          "flash-next", config_dir=home)
    assert isinstance(got, ServingRefusal) and (got.kind, got.line) == ("withheld", LINE)


def test_withheld_unit_target_is_refused_and_the_target_stays(home):
    _seed(home)
    units.write_target(home, "flash-next")
    got = resolve_serving(_load(home, machine_withheld=PEER_WITHHELD), _withheld_trust(), None,
                          config_dir=home)
    assert isinstance(got, ServingRefusal) and got.kind == "withheld"
    assert units.read_target(home) == "flash-next"


def test_withheld_active_endpoint_is_refused_and_stays_on_disk(home):
    _seed(home, overrides={"active_endpoint": _active()})
    got = resolve_serving(_load(home, machine_withheld=PEER_WITHHELD), _withheld_trust(), None,
                          config_dir=home)
    assert isinstance(got, ServingRefusal) and got.kind == "withheld"
    assert "active_endpoint" in load_overlay(home / "overrides.yaml")


def test_a_withheld_refusal_without_a_composed_line_builds_spec_16s_line(home):
    _seed(home)
    trust = MachineTrust(withheld=PEER_WITHHELD)
    got = resolve_serving(_load(home, machine_withheld=PEER_WITHHELD), trust, "flash-next",
                          config_dir=home)
    assert got.kind == "withheld"
    assert got.line.startswith("Not applying 1 change(s) in ")
    assert "extra_endpoints.flash-next.lifecycle.model (config.yaml)" in got.line
    assert "run `localharness start` on a terminal to review them" in got.line


def test_a_withheld_bind_all_alone_does_not_withhold_the_record(home):
    _seed(home)
    withheld = {"config.yaml": frozenset({("launch", "extra_endpoints.flash-next.lifecycle.bind_all")})}
    trust = MachineTrust(withheld=withheld, line=LINE)
    harness = _load(home, machine_withheld=withheld)
    flash = next(r for r in serving_records(harness) if r.name == "flash-next")
    assert record_withheld(trust, flash) is False
    got = resolve_serving(harness, trust, "flash-next", config_dir=home)
    assert isinstance(got, Serving) and got.record.name == "flash-next"


def test_a_withheld_primary_is_refused_never_gone(home):
    _seed(home, overrides={"pending_preset": PRIMARY})
    withheld = {"config.yaml": frozenset({("launch", "server.model")})}
    harness = _load(home, machine_withheld=withheld)
    assert harness.server is None
    got = resolve_serving(harness, MachineTrust(withheld=withheld, line=LINE), None, config_dir=home)
    assert isinstance(got, ServingRefusal) and got.kind == "withheld"
    assert load_overlay(home / "overrides.yaml")["pending_preset"] == PRIMARY


def test_a_withheld_primary_whose_alias_cannot_be_read_still_refuses_not_gone(home):
    _seed(home, overrides={"pending_preset": "an-alias-the-dropped-server-held"})
    withheld = {"config.yaml": frozenset({("launch", "server.model")})}
    got = resolve_serving(_load(home, machine_withheld=withheld),
                          MachineTrust(withheld=withheld, line=LINE), None, config_dir=home)
    assert isinstance(got, ServingRefusal) and got.kind == "withheld"


def test_a_pending_preset_naming_nothing_is_gone_and_the_next_rung_serves(home):
    _seed(home, overrides={"pending_preset": "deleted-peer", "active_endpoint": _active()})
    got = _serve(home)
    assert isinstance(got, Serving) and (got.rung, got.record.name) == (RUNG_ACTIVE, "flash-next")
    assert got.gone == (GoneSelection(
        "pending_preset", "deleted-peer",
        "pending_preset names deleted-peer, which is no longer a preset — removed"),)
    assert load_overlay(home / "overrides.yaml")["pending_preset"] == "deleted-peer"  # resolve writes nothing


def test_a_unit_target_naming_nothing_is_gone(home):
    _seed(home)
    units.write_target(home, "deleted-peer")
    got = _serve(home)
    assert got.rung == RUNG_DEFAULT
    assert [(g.store, g.name) for g in got.gone] == [(RUNG_UNIT, "deleted-peer")]


def test_clear_gone_removes_exactly_those_stores_and_returns_the_lines(home):
    _seed(home, overrides={"pending_preset": "deleted-peer", "active_endpoint": _active()})
    units.write_target(home, "also-gone")
    harness = _load(home)
    got = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    lines = clear_gone(harness, got.gone, config_dir=home)
    assert lines == [g.line for g in got.gone] and len(lines) == 2
    overlay = load_overlay(home / "overrides.yaml")
    assert "pending_preset" not in overlay and "active_endpoint" in overlay
    assert units.read_target(home) is None and harness.pending_preset is None


def test_an_active_endpoint_naming_no_record_is_passed_over_and_left_on_disk(home):
    _seed(home, overrides={"active_endpoint": _active(peer_url="http://x.example/v1", model="zzz")})
    got = _serve(home)
    assert (got.rung, got.gone) == (RUNG_DEFAULT, ())
    assert "active_endpoint" in load_overlay(home / "overrides.yaml")


def test_an_ambiguous_stored_name_is_refused_and_kept(home):
    _seed(home, server=False, peers=[_peer("qwen-a", 8081), _peer("qwen-b", 8082)],
          overrides={"pending_preset": "qwen"})
    got = _serve(home)
    assert isinstance(got, ServingRefusal) and got.kind == "ambiguous"
    assert load_overlay(home / "overrides.yaml")["pending_preset"] == "qwen"
