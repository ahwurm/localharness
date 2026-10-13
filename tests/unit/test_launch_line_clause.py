"""TUNE-10 (spec 20 §6.5, D-16, D-17, D-23): `start`'s launch line carries what the launch derived.

The existing parenthesis gains the derived clause (`kv q8_0 · mtp (1 layer)`), never the fitted `-c`; a
retried launch says what it dropped and how to keep the full window; a remembered shape says when it was
learned. The line is printed from `activate`'s `on_plan` callback, so it names the plan that really
starts. Nothing here starts a process or reads the host: `start_server`, `wait_ready`, free memory, the
weights and the probe are faked (`Box`), the consoles are recorders, the ledger is a file in `tmp_path`.
"""
from __future__ import annotations

import pytest

from localharness.cli import start_cmd
from localharness.cli.start_cmd import _start_managed_server
from localharness.config.models import AutosenseConfig
from localharness.provider import server
from localharness.provider.autosense import FreeMemory
from localharness.provider.tuning import NOTHING_DERIVED, Tuned, derived_clause
from tests.unit.test_launch_fit_retry import BASE, Box
from tests.unit.test_launch_plan import SWIFT, _gguf_spec

GIB = 2**30
KNOBS = AutosenseConfig()
MTP = Tuned(cache_type="q8_0", spec_type="draft-mtp", draft_p_min=0.6, mtp_layers=1)


# ------------------------------------------------------------------- the clause helper


def test_a_full_shape_names_the_kv_type_and_the_mtp_layers():
    assert derived_clause(MTP) == "kv q8_0 · mtp (1 layer)"
    assert derived_clause(Tuned(cache_type="q8_0", spec_type="draft-mtp", mtp_layers=2)) == "kv q8_0 · mtp (2 layers)"
    assert derived_clause(Tuned(cache_type="q8_0")) == "kv q8_0"
    assert derived_clause(NOTHING_DERIVED) == ""


def test_the_full_shape_clause_never_carries_the_fitted_window():
    assert "-c" not in derived_clause(MTP, context=65536)


# ------------------------------------------------------------------- start's launch line


class _World:
    """The launch's consoles and `start_server` recorded in the one ordering log `Box` already keeps."""

    def __init__(self, monkeypatch, box: Box):
        self.box = box
        self.cmds: list[list[str]] = []

        def printer(tag):
            def _print(*a, **k):
                assert k.get("markup") is False
                box.events.append(f"{tag}:{a[0] if a else ''}")
            return _print

        def start(config_dir, cmd, **kw):
            self.cmds.append(list(cmd))
            box.events.append("start_server")
            return 4321

        monkeypatch.setattr(start_cmd.console, "print", printer("out"))
        monkeypatch.setattr(start_cmd.err_console, "print", printer("err"))
        monkeypatch.setattr(server, "server_pid", lambda _d: None)
        monkeypatch.setattr(server, "docker_container_running", lambda _n: False)
        monkeypatch.setattr(server, "start_server", start)

    @property
    def out(self) -> list[str]:
        return [e[4:] for e in self.box.events if e.startswith("out:")]

    @property
    def err(self) -> list[str]:
        return [e[4:] for e in self.box.events if e.startswith("err:")]


@pytest.fixture
def box(monkeypatch):
    world = Box(monkeypatch)
    world.mem = FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi")
    world.weights = 18 * GIB
    return world


@pytest.fixture
def world(monkeypatch, box):
    return _World(monkeypatch, box)


def _argv_c(cmd: list[str]) -> int:
    return int(cmd[cmd.index("-c") + 1])


async def _start(tmp_path, spec, **kw):
    kw.setdefault("autosense", KNOBS)
    return await _start_managed_server(spec, tmp_path, BASE, kw.pop("launch_key", {}), **kw)


async def test_the_swift_preset_line_names_kv_and_mtp_before_the_server_starts(tmp_path, world):
    assert await _start(tmp_path, _gguf_spec(tmp_path, SWIFT)) is None
    assert world.out == ["Managed llama.cpp is not running — starting it "
                         "(kv q8_0 · mtp (1 layer); model load can take several minutes)..."]
    events = world.box.events
    assert events.index(f"out:{world.out[0]}") < events.index("start_server")


async def test_the_line_never_carries_the_fitted_window(tmp_path, world):
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT))
    window = _argv_c(world.cmds[0])
    assert window > 1000
    assert " -c" not in world.out[0] and str(window) not in world.out[0]
