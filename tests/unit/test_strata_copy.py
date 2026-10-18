"""The Strata config copy: only checked keys, flags and paths pass, and nothing planted redirects its write."""
from __future__ import annotations

import pytest

from localharness.provider import server, strata
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import strata_config
from tests.unit.test_strata_activate import SENSED, SERVED, make_world, sensing, with_args  # noqa: F401


def refusal(what: str) -> str:
    return strata.CONFIG_REFUSAL.format(name=SERVED, what=what)


@pytest.mark.parametrize("planned", [False, True])
async def test_an_env_key_refuses_the_launch_before_anything_starts(make_world, popen, sensing, planned):  # noqa: F811
    world = make_world(lambda shard: strata_config(shard, env={"X": "1"}))
    with pytest.raises(server.LaunchRefused, match="sets env"):
        await world.activate(autosense=SENSED if planned else None)
    assert popen == [] and not strata.copy_path(world.config_dir, 8083).exists()


async def test_a_clean_config_becomes_the_copy(make_world, popen):  # noqa: F811
    world = make_world(lambda shard: strata_config(shard, host="0.0.0.0", port=1, cwd="/", log="/x.log"))
    await world.activate(autosense=None)
    assert popen and world.copy["host"] == "127.0.0.1" and world.copy["port"] == 8083
    assert world.copy["cwd"] == str(world.engine) and world.copy["log"] == str(strata.log_path(world.config_dir, 8083))
    assert strata.copy_path(world.config_dir, 8083) == world.config_dir / "strata" / "8083.json"


async def test_one_refusal_names_every_key_and_flag(make_world, popen):  # noqa: F811
    world = make_world(lambda shard: with_args("--open", "-x")(shard) | {"mcp": {}, "api_key": "k", "env": {}})
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == refusal("sets api_key, env, mcp; passes --open, -x")
    assert popen == []
