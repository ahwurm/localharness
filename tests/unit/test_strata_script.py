"""`localharness model --script` for a Strata record prints the launch `activate` runs, copy and all; nothing is started."""
from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from localharness.agent import context as agent_context
from localharness.cli import serve_cmd
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.provider import server, strata, units
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_record, strata_tree
from tests.unit.test_serve_cmd import _Exec, serve_world

NAME = "flash-strata"
URL = "http://127.0.0.1:8083/v1"
KEY = "k-held-7f3a91"


def strata_peer(engine, shard, projector, *, keyed: bool = False) -> dict:
    return {"name": NAME, "base_url": URL, "provider_type": "strata", "gpu": True, **({"api_key": KEY} if keyed else {}),
            "lifecycle": {"runtime": "strata", "engine_dir": str(engine), "engine_config": CONFIG_NAME,
                          "model": str(shard), "mmproj": str(projector), "served_name": NAME, "port": 8083, "gpu": True,
                          **({"require_api_key": True} if keyed else {})}}


def strata_world(tmp_path, monkeypatch, *, keyed: bool = False):
    shard, projector = flash_next(tmp_path)
    engine = strata_tree(tmp_path / "engine", model=shard)
    world = serve_world(tmp_path, monkeypatch, peers=[strata_peer(engine, shard, projector, keyed=keyed)])
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


def test_all_renders_a_strata_preset_as_a_shell_function_beside_the_others(tmp_path, monkeypatch):
    world = strata_world(tmp_path, monkeypatch)
    result = script(world, "--all")

    assert result.exit_code == 0, result.stderr
    blocks = result.stdout.split("\n\n")
    assert len(blocks) == 2 and all(block.split("\n")[0].endswith("() {") for block in blocks)
    assert "<<'STRATA_COPY'" not in blocks[0] and "<<'STRATA_COPY'" in blocks[1]
    assert "exec .venv/bin/python serve/server.py --engine strata" in blocks[1]


def served(world, monkeypatch, events: list):
    units.write_target(world.g, NAME)
    real_copy, previous = strata.write_copy, server.exec_server

    def write_copy(*args):
        events.append("copy")
        return real_copy(*args)

    def exec_server(*args):
        events.append("exec")
        return previous(*args)

    monkeypatch.setattr(strata, "write_copy", write_copy)
    monkeypatch.setattr(serve_cmd.os, "chdir", lambda path: events.append(("chdir", path)))
    monkeypatch.setattr(server, "exec_server", exec_server)
    return world.run()


def test_serve_writes_the_copy_then_changes_to_the_engine_dir_then_execs_the_venv_python(tmp_path, monkeypatch):
    world = strata_world(tmp_path, monkeypatch)
    events: list = []

    result = served(world, monkeypatch, events)

    assert isinstance(result.exception, _Exec), (result.exit_code, world.err)
    assert events == ["copy", ("chdir", str(world.engine)), "exec"]
    path, argv, _env = world.execs[0]
    copy_file = strata.copy_path(world.g, 8083)
    engine = str(world.engine)
    assert path == f"{engine}/.venv/bin/python" and argv == [
        f"{engine}/.venv/bin/python", f"{engine}/serve/server.py", "--engine", "strata", "--config", str(copy_file),
        "--port", "8083", "--host", "127.0.0.1"]
    assert copy_file.is_file() and world.pidfile_at_exec[0] is not None


def test_serve_refuses_files_changed_after_the_trust_check_before_anything_is_written(tmp_path, monkeypatch):
    world = strata_world(tmp_path, monkeypatch)
    real = strata.verify_recorded

    def changed_meanwhile(*args):
        (world.engine / "serve" / "server.py").write_text("# changed\n", encoding="utf-8")
        return real(*args)

    monkeypatch.setattr(strata, "verify_recorded", changed_meanwhile)
    events: list = []

    result = served(world, monkeypatch, events)

    assert result.exit_code == serve_cmd.EXIT_REFUSED and world.execs == [] and events == []
    assert strata.UNCONFIRMED.format(name=NAME) in world.said()
    assert not strata.copy_path(world.g, 8083).exists() and not world.pid_file.exists()


def test_the_exec_environment_drops_the_blocklisted_strata_names(tmp_path):
    names = ("STRATA_SKIP_SHA256", "STRATA_PREBUILT_URL", "STRATA_SOURCE", "STRATA_EXECV", "HF_ENDPOINT")
    spec = strata_record(tmp_path, tmp_path / "m.gguf", tmp_path / "p.gguf")

    env = serve_cmd._exec_environment(
        {"PATH": "/bin", **dict.fromkeys(names, "1")}, "/home/x", {"USER": "u"}, spec, KEY, None)

    assert not set(names) & set(env) and env["STRATA_API_KEY"] == KEY and env["PATH"] == "/bin"


@pytest.mark.parametrize("planned", [True, False])
def test_the_planned_argv_carries_the_copy_with_or_without_autosense_knobs(tmp_path, monkeypatch, planned):
    world = strata_world(tmp_path, monkeypatch)
    harness = ConfigLoader(config_dir=world.g, local_config_dir=None).load_harness()
    spec = next(ep for ep in harness.extra_endpoints if ep.name == NAME).lifecycle
    argv, _, _ = serve_cmd._planned_argv(
        spec, world.g, harness.autosense if planned else None, server.launch_decision(spec, False, peer=NAME),
        keyed=False, unit=None)

    assert argv[argv.index("--config") + 1] == str(strata.copy_path(world.g, 8083))
