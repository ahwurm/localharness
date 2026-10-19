"""`localharness model --script` for a Strata record prints the launch `activate` runs, copy and all; nothing is started."""
from __future__ import annotations

import httpx
from typer.testing import CliRunner

from localharness.agent import context as agent_context
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.provider import server, strata
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_tree
from tests.unit.test_serve_cmd import llama_peer, serve_world

NAME = "flash-strata"
URL = "http://127.0.0.1:8083/v1"
KEY = "k-held-7f3a91"


def strata_peer(engine, shard, projector, *, keyed: bool = False) -> dict:
    return {"name": NAME, "base_url": URL, "provider_type": "strata", "gpu": True, **({"api_key": KEY} if keyed else {}),
            "lifecycle": {"runtime": "strata", "engine_dir": str(engine), "engine_config": CONFIG_NAME,
                          "model": str(shard), "mmproj": str(projector), "served_name": NAME, "port": 8083, "gpu": True,
                          **({"require_api_key": True} if keyed else {})}}


def strata_world(tmp_path, monkeypatch, *, keyed: bool = False, with_llama: bool = False):
    shard, projector = flash_next(tmp_path)
    engine = strata_tree(tmp_path / "engine", model=shard)
    peers = [strata_peer(engine, shard, projector, keyed=keyed)]
    if with_llama:
        peer = llama_peer("flash", 8082)
        peer["lifecycle"].update(model=str(shard), mmproj=str(projector))
        peers.insert(0, peer)
    world = serve_world(tmp_path, monkeypatch, peers=peers)
    world.engine = engine
    return world


def script(world, *args: str):
    return CliRunner().invoke(app, ["model", "--script", *args, "--config-dir", str(world.g)])


def heredoc(text: str) -> str:
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.endswith("<<'STRATA_COPY'")) + 1
    return "\n".join(lines[start:lines.index(strata.COPY_DELIMITER, start)]) + "\n"


async def activate(world, popen, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(lambda request: httpx.Response(
        200, json={"loaded": True, "service": "strata"} if request.url.path == "/health" else {"data": [{"id": NAME}]})))
    monkeypatch.setattr(server, "server_pid", lambda config_dir: 4242 if popen else None)
    monkeypatch.setattr(agent_context, "probe_window", lambda *a, **k: agent_context.WindowProbe(None, None))
    harness = ConfigLoader(config_dir=world.g, local_config_dir=None).load_harness()
    peer = next(ep for ep in harness.extra_endpoints if ep.name == NAME)
    await strata.StrataStrategy().activate(
        peer.lifecycle, world.g, URL, peer=NAME, autosense=harness.autosense, api_key=KEY)
    return strata.copy_path(world.g, 8083).read_text("utf-8")


async def test_the_script_prints_the_launch_activate_runs_and_its_copy_is_the_one_activate_writes(
        tmp_path, monkeypatch, popen):  # noqa: F811
    world = strata_world(tmp_path, monkeypatch, keyed=True)
    result = script(world, NAME)

    assert result.exit_code == 0, result.stderr
    copy_file = strata.copy_path(world.g, 8083)
    assert f"cat > {copy_file} <<'STRATA_COPY'" in result.stdout.splitlines()
    assert result.stdout.splitlines()[-1] == (
        f"cd {world.engine} && exec .venv/bin/python serve/server.py --engine strata --config {copy_file} "
        "--port 8083 --host 127.0.0.1")
    assert KEY not in result.stdout and "# STRATA_API_KEY=<extra_endpoints.flash-strata.api_key>" in result.stdout
    assert not copy_file.exists()
    assert heredoc(result.stdout) == await activate(world, popen, monkeypatch)
