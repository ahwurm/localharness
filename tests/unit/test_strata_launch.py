"""A Strata launch runs only what the machine record confirmed, is recognised by its proven process and stops parent-first."""
from __future__ import annotations

import json

import pytest

from localharness.config import trust
from localharness.provider import server, strata
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import CONFIG_NAME
from tests.unit.test_strata_activate import SENSED, SERVED, make_world, sensing  # noqa: F401

PEER = "strata-peer"
UNCONFIRMED = strata.UNCONFIRMED.format(name=SERVED)


def copy_file(world):
    return strata.copy_path(world.config_dir, 8083)


@pytest.mark.parametrize("peer", [PEER, None])
async def test_a_launch_whose_server_py_changed_after_the_record_is_refused_before_anything_starts(
        make_world, popen, peer):  # noqa: F811
    world = make_world(peer=peer)
    (world.engine / "serve" / "server.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == UNCONFIRMED
    assert popen == [] and not copy_file(world).exists()


@pytest.mark.parametrize("peer", [PEER, None])
async def test_a_launch_of_the_recorded_files_runs(make_world, popen, peer):  # noqa: F811
    world = make_world(peer=peer)
    await world.activate(autosense=None)
    assert len(popen) == 1 and copy_file(world).exists()


async def test_config_bytes_changed_between_the_record_and_the_launch_are_refused(make_world, popen):  # noqa: F811
    world = make_world(peer=PEER)
    path = world.engine / CONFIG_NAME
    config = json.loads(path.read_text("utf-8"))
    config["args"] += ["--prefill", "512"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == UNCONFIRMED
    assert popen == [] and not copy_file(world).exists()


async def test_a_file_swapped_after_start_checked_is_refused_at_the_launch(make_world, popen):  # noqa: F811
    world = make_world(peer=PEER)
    assert [e["name"] for e in trust.machine_snapshot(world.config_dir) if e["name"].endswith("strata_code")]
    (world.engine / "engine" / "strata").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(server.LaunchRefused):
        await world.activate(autosense=None)
    assert popen == []


async def test_a_launch_with_no_machine_record_is_refused(make_world, popen, monkeypatch):  # noqa: F811
    world = make_world()
    monkeypatch.setattr(trust, "recorded_machine", lambda config_dir: None)
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == UNCONFIRMED
    assert popen == []


@pytest.mark.parametrize("path", ["serve", CONFIG_NAME, "engine/strata"])
async def test_an_install_made_group_writable_since_the_set_up_is_refused(make_world, popen, path):  # noqa: F811
    world = make_world(peer=PEER)
    (world.engine / path).chmod(0o777)
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == strata.NOT_AN_INSTALL.format(name=SERVED, engine_dir=str(world.engine))
    assert popen == [] and not copy_file(world).exists()


async def test_the_planned_launch_is_checked_the_same_way(make_world, popen, sensing):  # noqa: F811
    world = make_world(peer=PEER)
    (world.engine / "serve" / "server.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=SENSED)
    assert str(refused.value) == UNCONFIRMED and popen == []
