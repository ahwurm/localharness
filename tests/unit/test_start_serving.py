"""`cli/start_serving.py`: what `start` decides once it knows what serves (spec 20 §8.2, §1 item 3, §9.3).

Offline: `list_live_models` is faked (`Probes`), harnesses come from `ConfigLoader` over a tmp dir,
the `/proc` side is the fake tree of `fake_proc`, the unit's state is patched, the server boundaries
a swap crosses are recorded fakes. No socket, no process, no service manager (D-30, D-32). Test
fakes take `**kw`.
"""
from __future__ import annotations

import pytest

from localharness.cli import model_ops, start_serving
from localharness.cli.model_ops import (
    RUNG_ACTIVE,
    RUNG_DEFAULT,
    RUNG_PENDING,
    RUNG_PRESET,
    Serving,
    launchable_loopback_others,
    resolve_serving,
    sense_serving,
    shares_base_url,
)
from localharness.cli.start_serving import (
    ATTACH_LINE,
    OTHER_SERVING_LINE,
    RESUME_CLAUSE,
    Decision,
    decide,
    selected_answers,
    settle_attach,
)
from localharness.cli.workspace import MachineTrust
from localharness.config.overlay import load_overlay
from localharness.provider import speed_stats
from tests.unit.test_sense_serving import (  # noqa: F401
    FLASH_ACTIVE,
    PEER_URL,
    PRIMARY,
    PRIMARY_URL,
    Probes,
    home,
    load,
    peer,
    probes,
    record,
    seed,
)


def ledger(home):
    return speed_stats.default_speed_stats_path(home)


def serve(home, preset=None, **kw):
    got = resolve_serving(load(home, **kw), MachineTrust(), preset, config_dir=home)
    assert isinstance(got, Serving), got
    return got


# ----------------------------------------------------------------------------------------- the tracer


@pytest.mark.asyncio
async def test_a_resumed_peer_answers_and_the_attach_lands_as_a_remembered_attach(
        home, probes, monkeypatch):
    seed(home, active=FLASH_ACTIVE, last_served=PRIMARY)
    harness = load(home)
    serving = resolve_serving(harness, MachineTrust(), None, config_dir=home)
    assert (serving.record.name, serving.rung) == ("flash-next", RUNG_ACTIVE)

    others = launchable_loopback_others(harness, MachineTrust(), serving.record)
    assert [r.name for r in others] == [PRIMARY]
    sense = await sense_serving(harness, others)
    assert sense.answering == () and probes.urls == [PRIMARY_URL]

    last = speed_stats.last_served(ledger(home))
    decision = decide(serving, selected_answers=True, sense=sense, last_served=last)
    assert (decision.kind, decision.record.name, decision.rung) == ("attach", "flash-next", RUNG_ACTIVE)
    assert decision.lines == ("Serving flash-next (your last /model choice)",)
    assert decision.lines == (ATTACH_LINE.format(name="flash-next", clause=RESUME_CLAUSE),)

    overrides = (home / "overrides.yaml").read_bytes()
    spied: list[dict] = []
    real = model_ops.persist_landed

    async def spy(harness, rec, **kw):
        spied.append({"record": rec.name, **kw})
        return await real(harness, rec, **kw)

    monkeypatch.setattr(model_ops, "persist_landed", spy)
    await settle_attach(harness, decision, config_dir=home)

    assert spied == [{"record": "flash-next", "outcome": "attach", "rung": RUNG_ACTIVE,
                      "config_dir": home}]
    assert speed_stats.last_served(ledger(home)) == "flash-next"
    assert (home / "overrides.yaml").read_bytes() == overrides  # no selection was written
    assert load_overlay(home / "overrides.yaml")["active_endpoint"] == FLASH_ACTIVE


def test_start_serving_does_not_import_start_cmd():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(start_serving.__file__).read_text(encoding="utf-8"))
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [n for n in names if "start_cmd" in n]
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    assert "start_cmd" not in imported
