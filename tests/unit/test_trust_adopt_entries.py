"""`adopt_entries` re-records an act's own write; a No can be remembered under a keyed name.

D-18 / D-25 of the machine-trust phase. Adoption works from the passed entries ONLY - it never
re-snapshots the disk, so an agent's write made in the meantime is not adopted with it.
"""
from __future__ import annotations

import dataclasses

import pytest

from localharness.cli.workspace import MachineTrust, decide_machine_trust
from localharness.config import trust
from tests.unit.trust_support import g, never_asked, write_yaml  # noqa: F401 — g is a fixture


def _e(file: str, kind: str, name: str, shown: str) -> dict:
    return {"file": file, "kind": kind, "name": name, "shown": shown}


A = _e("config.yaml", "launch_model", "server.model", "m1")
B = _e("config.yaml", "launch", "server.runtime", "vllm")
KINDS = frozenset(trust.MACHINE_KINDS)


def _seed(g, entries, kinds=KINDS):
    trust.record_machine(g, sorted(entries, key=trust._entry_key), kinds=kinds)


# ------------------------------------------------------------------ adopt_entries


def test_with_no_record_nothing_is_written(g):
    before = trust.trust_store_path().exists()

    trust.adopt_entries(g, [A])

    assert trust.recorded_machine(g) is None
    assert trust.trust_store_path().exists() == before


def test_a_new_entry_is_added_sorted_and_refingerprinted(g):
    _seed(g, [A, B], kinds=frozenset({"launch", "launch_model"}))
    new = _e("overrides.yaml", "launch_model", "server.model", "m2")

    trust.adopt_entries(g, [new])

    rec = trust.recorded_machine(g)
    assert rec["entries"] == sorted([A, B, new], key=trust._entry_key)
    assert [e["file"] for e in rec["entries"]] == ["config.yaml", "config.yaml", "overrides.yaml"]
    assert set(rec["kinds"]) == {"launch", "launch_model"}
    assert rec["fingerprint"] == trust.fingerprint(rec["entries"])


def test_the_same_file_kind_name_is_replaced(g):
    _seed(g, [A, B])
    m3 = _e("config.yaml", "launch_model", "server.model", "m3")

    trust.adopt_entries(g, [m3])

    entries = trust.recorded_machine(g)["entries"]
    assert m3 in entries and A not in entries and B in entries and len(entries) == 2


def test_adopting_twice_is_idempotent(g):
    _seed(g, [A, B])
    m3 = _e("config.yaml", "launch_model", "server.model", "m3")

    trust.adopt_entries(g, [m3])
    first = trust.recorded_machine(g)
    trust.adopt_entries(g, [m3])

    assert trust.recorded_machine(g) == first
    assert len([e for e in first["entries"] if e["name"] == "server.model"]) == 1


def test_the_store_stays_private_and_atomic(g):
    _seed(g, [A])

    trust.adopt_entries(g, [B])

    assert trust.trust_store_path().stat().st_mode & 0o777 == 0o600


def test_adopting_what_the_disk_holds_leaves_nothing_to_ask(g, monkeypatch):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm", "model": "m1"}})
    trust.record_machine(g, [], kinds=KINDS)
    never_asked(monkeypatch)
    trust.adopt_entries(g, trust.machine_snapshot(g))

    assert decide_machine_trust(g, ask=True) == MachineTrust()


def test_adopting_a_value_that_differs_from_the_disk_leaves_the_disk_a_change(g, monkeypatch):
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm", "model": "m1"}})
    trust.record_machine(g, trust.machine_snapshot(g), kinds=KINDS)
    # the act wrote m2; an agent then wrote m3 before the adoption ran
    write_yaml(g / "config.yaml", {"server": {"runtime": "vllm", "model": "m3"}})
    trust.adopt_entries(g, [_e("config.yaml", "launch_model", "server.model", "m2")])
    asked: list[str] = []
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda prompt, *a, **kw: asked.append(prompt) or False)

    decide_machine_trust(g, ask=True)

    assert len(asked) == 1, "the on-disk m3 is not what was adopted, so it is still a question"


def test_adopt_entries_never_snapshots_the_disk(g, monkeypatch):
    _seed(g, [A])

    def _boom(*_a, **_kw):
        raise AssertionError("adopt_entries re-snapshotted the disk")

    monkeypatch.setattr(trust, "machine_snapshot", _boom)
    trust.adopt_entries(g, [B])

    assert B in trust.recorded_machine(g)["entries"]


# ------------------------------------------------------------------ the keyed decline store


def test_decline_key_joins_its_parts():
    assert trust.decline_key("<binary>", "/usr/bin/llama-server") == "<binary>/usr/bin/llama-server"
    assert trust.decline_key("<peer>", "extra_endpoints.p.lifecycle.bind_all", "llamacpp:true:open") == (
        "<peer>extra_endpoints.p.lifecycle.bind_allllamacpp:true:open")
    assert trust.decline_key("<config.ini>", "abc123") == "<config.ini>abc123"


def test_a_keyed_decline_is_remembered_only_for_that_key(g):
    key = trust.decline_key("<peer>", "extra_endpoints.p.lifecycle.bind_all", "llamacpp:true:open")

    assert trust.offer_was_declined(key) is False
    trust.record_offer_decline(key)

    assert trust.offer_was_declined(key) is True
    assert trust.offer_was_declined(trust.decline_key(
        "<peer>", "extra_endpoints.p.lifecycle.bind_all", "llamacpp:true:keyed")) is False


def test_a_keyed_decline_lives_in_the_declined_offers_file(g):
    trust.record_offer_decline(trust.decline_key("<config.ini>", "d1"))

    assert "<config.ini>d1" in trust.declined_offers_path().read_text(encoding="utf-8")


def test_the_path_form_still_works_and_is_independent(g, tmp_path):
    trust.record_offer_decline(tmp_path)

    assert trust.offer_was_declined(tmp_path) is True
    assert trust.offer_was_declined(trust.decline_key("<binary>", str(tmp_path))) is False


def test_a_corrupt_declined_offers_file_reads_as_not_declined(g):
    trust.declined_offers_path().parent.mkdir(parents=True, exist_ok=True)
    trust.declined_offers_path().write_text("{not yaml: [", encoding="utf-8")

    assert trust.offer_was_declined(trust.decline_key("<binary>", "/x")) is False


# ------------------------------------------------------------------ MachineTrust


def test_machine_trust_new_fields_default_empty_and_keep_equality():
    assert MachineTrust().quiet == frozenset() and MachineTrust().posture_notes == ()
    assert MachineTrust() == MachineTrust()


def test_machine_trust_with_a_quiet_entry_is_frozen_and_unequal():
    quiet = MachineTrust(quiet=frozenset({("config.yaml", "launch", "server.bind_all")}))

    assert quiet != MachineTrust()
    with pytest.raises(dataclasses.FrozenInstanceError):
        quiet.quiet = frozenset()  # type: ignore[misc]


def test_machine_trust_field_order_keeps_existing_positional_calls_valid():
    assert [f.name for f in dataclasses.fields(MachineTrust)] == [
        "withheld", "line", "quiet", "posture_notes"]
    assert MachineTrust({}, "line").line == "line"
