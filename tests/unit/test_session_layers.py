"""A private run holds its answers and records in memory; a normal run writes them as before."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import Decision, ToolMeta
from localharness.config import project_records as pr
from localharness.config import trust
from localharness.config.grants import GrantStore, new_grant, new_refusal
from localharness.core.events import PermissionResolved
from localharness.core.run_folder import begin, end, hold, shadow

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")

WRITE = ToolMeta(destructive=True, group="fs.write")


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    cfg = home / ".localharness"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(cfg))
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    ws = home / "proj"
    ws.mkdir()
    runs = []

    def start(private: bool):
        runs.append(begin(cfg, private=private))
        return runs[-1]

    yield type("World", (), {"home": home, "cfg": cfg, "ws": ws, "start": staticmethod(start)})
    for run in runs:
        end(run)


class Asker:
    def __init__(self, kind: str):
        self.kind, self.calls = kind, []

    async def __call__(self, request):
        self.calls.append(request)
        return Decision(kind=self.kind)


class Bus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


def gate_for(world, run, asker, bus=None) -> PermissionGate:
    return PermissionGate(
        boundary=world.ws, workspace=world.ws, grants=GrantStore(world.cfg / "grants.yaml"), mode="guarded",
        asker=asker, channel_name="test", has_review_surface=True, config_dir=world.cfg, run=run,
        scratch=run.scratch, bus=bus)


async def write_outside(gate, path: Path):
    return await gate.check("write", {"path": str(path), "content": "x"}, WRITE, agent_id="a", session_id="s")


def test_hold_is_a_noop_without_a_private_run_except_for_scratch(world):
    assert hold("grants", "k", 1) is False and shadow("grants") == {}
    run = world.start(private=False)
    assert hold("grants", "k", 1) is False and shadow("grants") == {}
    assert hold("grants", "s", 2, scratch=True) is True and shadow("grants") == {"s": 2}
    assert run.held == {("grants", "s"): 2}


async def test_an_always_in_a_private_run_holds_through_a_rebuilt_gate_and_writes_nothing(world):
    run = world.start(private=True)
    grants = world.cfg / "grants.yaml"
    other = world.home / "elsewhere"
    asker, bus = Asker("allow_always"), Bus()
    assert (await write_outside(gate_for(world, run, asker, bus), other / "a")).allowed
    assert len(asker.calls) == 1
    assert (await write_outside(gate_for(world, run, asker, bus), other / "b")).allowed
    assert len(asker.calls) == 1
    assert not grants.exists()
    resolved = [e for e in bus.events if isinstance(e, PermissionResolved)]
    assert [e.wrote_grant for e in resolved] == [False]


async def test_an_always_in_a_normal_run_is_written_and_a_scratch_key_is_not(world):
    run = world.start(private=False)
    asker, bus = Asker("allow_always"), Bus()
    assert (await write_outside(gate_for(world, run, asker, bus), world.home / "elsewhere" / "a")).allowed
    assert (world.cfg / "grants.yaml").exists()
    assert [e.wrote_grant for e in bus.events if isinstance(e, PermissionResolved)] == [True]
    before = (world.cfg / "grants.yaml").read_bytes()
    assert (await write_outside(gate_for(world, run, asker, bus), run.scratch / "x")).allowed
    assert (world.cfg / "grants.yaml").read_bytes() == before
    assert [e.wrote_grant for e in bus.events if isinstance(e, PermissionResolved)] == [True, False]


async def test_a_refusal_another_session_writes_mid_run_is_honoured(world):
    run = world.start(private=True)
    asker = Asker("allow_always")
    other = world.home / "elsewhere"
    assert (await write_outside(gate_for(world, run, asker), other / "a")).allowed
    GrantStore(world.cfg / "grants.yaml").add_refusal(new_refusal(
        key=str(other.resolve()), klass="edit-outside", workspace=world.ws, channel="x", session_id="o"))
    assert not (await write_outside(gate_for(world, run, asker), other / "a")).allowed
    assert len(asker.calls) == 1


def test_a_private_grant_and_refusal_are_read_back_and_never_written(world):
    run = world.start(private=True)
    store = GrantStore(world.cfg / "grants.yaml")
    grant = new_grant(key="ls", klass="shell", workspace=world.ws, channel="t", session_id="s")
    assert store.add(grant) is False
    assert store.lookup(world.ws / "sub", "shell", "ls").key == "ls"
    assert store.refused(world.ws, "shell", "ls") is None
    assert store.add_refusal(new_refusal(key="rm", klass="shell", workspace=world.ws, channel="t", session_id="s")) is False
    assert store.refused(world.ws, "shell", "rm").key == "rm"
    assert not (world.cfg / "grants.yaml").exists() and run.held


def test_a_normal_grant_is_written(world):
    world.start(private=False)
    store = GrantStore(world.cfg / "grants.yaml")
    assert store.add(new_grant(key="ls", klass="shell", workspace=world.ws, channel="t", session_id="s")) is True
    assert store.lookup(world.ws, "shell", "ls").key == "ls"


def snapshot(*paths: Path) -> dict[Path, bytes | None]:
    return {p: p.read_bytes() if p.exists() else None for p in paths}


def test_trust_executables_offers_and_records_read_back_but_stay_off_disk(world):
    world.start(private=True)
    store_paths = (trust.trust_store_path(), trust.declined_offers_path(), pr.store_path(None))
    before = snapshot(*store_paths)
    assert trust.is_trusted(world.ws) is None
    trust.record_trust(world.ws, True, unseen_executables=True)
    assert trust.is_trusted(world.ws) is True and trust.is_trusted_tree(world.ws / "sub") is True
    trust.record_executables(world.ws, [{"kind": "mcp", "name": "a"}])
    assert trust.recorded_executables(world.ws)["servers"] == [{"kind": "mcp", "name": "a"}]
    assert trust.is_trusted(world.ws) is True
    assert not trust.offer_was_declined(world.ws)
    trust.record_offer_decline(world.ws)
    assert trust.offer_was_declined(world.ws)
    pr.write_review_level(world.ws, 2)
    assert pr.read_review_level(world.ws, 5) == 2
    record = pr.InstructionsRecord(path=str(world.ws / "CLAUDE.md"), name="CLAUDE.md", digest="ab" * 32, size=1, state="yes")
    pr.write_instructions(world.ws, record)
    assert pr.read_instructions(world.ws).digest == "ab" * 32
    assert snapshot(*store_paths) == before


def test_machine_rows_and_path_free_declines_still_reach_disk_in_a_private_run(world):
    world.start(private=True)
    key = trust.decline_key("<binary>", "/usr/bin/x")
    trust.record_offer_decline(key)
    assert trust.offer_was_declined(key)
    assert yaml.safe_load(trust.declined_offers_path().read_text())[key] == {"declined": True}
    trust.record_machine(world.cfg, [])
    assert trust.recorded_machine(world.cfg) is not None
    assert trust.trust_store_path().exists()


def test_a_normal_run_writes_trust_and_records(world):
    world.start(private=False)
    trust.record_trust(world.ws, True)
    trust.record_offer_decline(world.ws)
    pr.write_review_level(world.ws, 2)
    assert yaml.safe_load(trust.trust_store_path().read_text())[str(world.ws)]["trusted"] is True
    assert trust.offer_was_declined(world.ws) and pr.read_review_level(world.ws, 5) == 2
    assert pr.store_path(None).exists()


def test_a_held_record_removal_hides_the_stored_one(world):
    normal = world.start(private=False)
    pr.write_review_level(world.ws, 2)
    end(normal)
    world.start(private=True)
    assert pr.read_review_level(world.ws, 5) == 2
    key = str(world.ws.resolve())
    with pr._store_transaction() as data:
        del data["roots"][key]
    assert pr.read_review_level(world.ws, 5) == 5
    assert yaml.safe_load(pr.store_path(None).read_text())["roots"][key]["review"] == 2
