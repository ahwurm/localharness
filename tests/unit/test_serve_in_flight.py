"""`localharness serve` over the in-flight rung (spec 20 §8.1, R-14-38, D-14, D-16, AUTO-07).

`serve` is unchanged: it resolves `resolve_serving(harness, trust, None)` and execs the record that names. These
tests run the REAL command in the world of test_serve_cmd.py (the exec is a recorder) with a llama.cpp primary and
two llama.cpp peers `a` and `b`, and put `unit-target` and `pending_preset` in each combination:

- `unit-target` = `server:` -> the primary;
- a LIVE in-flight `b` (this test process is the requester) beside `pending_preset: a` -> `b`, and `serve` leaves
  `pending_preset: a` in the overlay;
- a plain `b`, or an in-flight `b` whose requester is gone, beside `pending_preset: a` -> `a`.

The requester is this process, registered in the fake `/proc` (`fake_proc`) so `write_target(in_flight=True)`
and the ladder read it exactly as they read a real one.
"""
from __future__ import annotations

import os

import pytest
import yaml

from localharness.provider import procfs, server, units
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_serve_cmd import _launched, llama_peer, serve_world

OWN_START = 111


@pytest.fixture
def world(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    fake_proc.process(os.getpid(), argv=("python", "-m", "pytest"), exe="/usr/bin/python", start_time=OWN_START,
                      comm="python")
    binary = str(tmp_path / "bin" / "llama-server")
    return serve_world(tmp_path, monkeypatch,
                       peers=[llama_peer("a", 8081, binary=binary), llama_peer("b", 8082, binary=binary)])


def _model_of(argv: list[str]) -> str:
    return argv[argv.index("-m") + 1]


def _pending(world) -> str | None:
    path = world.g / "overrides.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("pending_preset") if path.exists() else None


def _start() -> int:
    own = procfs.read_process(os.getpid())
    assert isinstance(own, procfs.ProcRecord) and own.start_time == OWN_START
    return own.start_time


def _plant(world, body: str) -> None:
    directory = server.server_dir(world.g)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / units.TARGET_NAME).write_text(body, encoding="utf-8")


def _served(world, result, model: str, port: str, record: str | None = None) -> None:
    _path, argv, _env = _launched(world, result)
    assert _model_of(argv) == model and argv[argv.index("--port") + 1] == port
    marker = world.marker()
    assert marker is not None and marker.record == (record or f"llamacpp:{model}")  # the record `serve` execed


def test_serve_with_the_server_token_execs_the_primary(world):
    units.write_target(world.g, units.PRIMARY_TARGET)
    _served(world, world.run(), str(world.model), "8080", world.record)


def test_serve_with_an_in_flight_server_token_beside_a_pending_preset_execs_the_primary(world):
    world.set_overrides({"pending_preset": "a"})
    units.write_target(world.g, units.PRIMARY_TARGET, in_flight=True)
    _served(world, world.run(), str(world.model), "8080", world.record)
    assert _pending(world) == "a"


def test_serve_with_a_live_in_flight_target_execs_it_and_leaves_the_pending_preset(world):
    world.set_overrides({"pending_preset": "a"})
    units.write_target(world.g, "b", in_flight=True)
    _served(world, world.run(), "/m/b.gguf", "8082")
    assert _pending(world) == "a"                                              # serve never removes it (D-16)
    assert units.read_target(world.g) == "b"                                   # ... nor rewrites the target


def test_serve_with_a_plain_target_beside_a_pending_preset_execs_the_pending_preset(world):
    world.set_overrides({"pending_preset": "a"})
    units.write_target(world.g, "b")
    _served(world, world.run(), "/m/a.gguf", "8081")
    assert _pending(world) == "a"


def test_serve_with_an_in_flight_target_whose_requester_is_gone_execs_the_pending_preset(world):
    world.set_overrides({"pending_preset": "a"})
    _plant(world, f"b {os.getpid()} {_start() + 1}\n")                         # the same pid, another start time
    _served(world, world.run(), "/m/a.gguf", "8081")
    assert _pending(world) == "a"


def test_serve_with_a_requester_that_does_not_exist_execs_the_pending_preset(world):
    world.set_overrides({"pending_preset": "a"})
    _plant(world, "b 4194303 7\n")
    _served(world, world.run(), "/m/a.gguf", "8081")


def test_serve_with_no_pending_preset_execs_a_stale_target(world):
    _plant(world, f"b {os.getpid()} {_start() + 1}\n")
    _served(world, world.run(), "/m/b.gguf", "8082")
