"""The `server:` token of `unit-target` and the round trip of every record the ladder can land on
(spec 20 §8.1, R-14-03, R-14-38).

`UnitStrategy` names the primary record with the reserved token `units.PRIMARY_TARGET` (the provider layer cannot
compute the primary's name) and a peer with its own name. `resolve_serving` maps the token back on BOTH
`unit-target` rungs, the plain one and the in-flight one, and what the strategy writes always resolves to the
record it was written for.
"""
from __future__ import annotations

import pytest

from localharness.cli import model_ops
from localharness.cli.model_ops import RUNG_UNIT, Serving, ServingRefusal
from localharness.cli.workspace import MachineTrust
from localharness.provider import units
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_unit_target_in_flight import fresh, ladder, own, plant, set_pending
from tests.unit.unit_strategy_support import PRIMARY, unit_world

PRIMARY_WITHHELD = {"config.yaml": frozenset({("launch_model", "server.model")})}
LINE = "Not applying 1 change(s) in /g that you have not confirmed (x): you said No"


@pytest.fixture
def world(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    return unit_world(tmp_path, monkeypatch, fake_proc)


def test_the_token_is_the_reserved_name_and_fullmatches_the_target_pattern():
    assert units.PRIMARY_TARGET == "server:"
    assert units.TARGET_NAME_PATTERN.fullmatch(units.PRIMARY_TARGET)


def test_a_plain_server_token_is_the_primary_on_the_unit_target_rung(world):
    plant(world.g, "server:\n")
    got = ladder(world.g)
    assert isinstance(got, Serving) and got.record.name == PRIMARY and got.record.peer is None
    assert got.rung == RUNG_UNIT and got.explicit is False and got.gone == ()


def test_a_withheld_primary_is_refused_as_withheld_not_gone_on_the_plain_rung(world):
    plant(world.g, "server:\n")
    got = ladder(world.g, MachineTrust(withheld=PRIMARY_WITHHELD, line=LINE))
    assert isinstance(got, ServingRefusal) and got.kind == "withheld" and got.line == LINE


def test_a_withheld_primary_is_refused_on_the_in_flight_rung_too(world):
    plant(world.g, f"server: {own()}\n")
    got = ladder(world.g, MachineTrust(withheld=PRIMARY_WITHHELD, line=LINE))
    assert isinstance(got, ServingRefusal) and got.kind == "withheld"


def test_a_live_in_flight_server_token_outranks_a_pending_preset_for_another_record(world):
    set_pending(world, "flash-next")
    plant(world.g, f"server: {own()}\n")
    got = ladder(world.g)
    assert got.record.name == PRIMARY and got.rung == RUNG_UNIT


def test_every_record_the_strategy_can_write_resolves_back_to_itself(world):
    """What `UnitStrategy` writes (`PRIMARY_TARGET` for the primary, the peer's name otherwise) resolves to the
    same record on the `unit-target` rung: plain with no `pending_preset`, and in flight beside a
    `pending_preset` naming another record."""
    records = model_ops.serving_records(fresh(world.g))
    assert [r.name for r in records] == [PRIMARY, "flash-next", "qwen-27b"]
    for record in records:
        written = units.PRIMARY_TARGET if record.peer is None else record.peer.name
        other = next(r.name for r in records if r.name != record.name)
        units.write_target(world.g, written)
        plain = ladder(world.g)
        assert plain.record.name == record.name and plain.rung == RUNG_UNIT
        set_pending(world, other)
        units.write_target(world.g, written, in_flight=True)
        flying = ladder(world.g)
        assert flying.record.name == record.name and flying.rung == RUNG_UNIT
        model_ops.clear_pending_preset(world.harness, config_dir=world.g)
