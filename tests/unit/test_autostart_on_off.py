"""`localharness model --autostart on` — gate, scope, capture, write, adopt, reload, re-check, enable (spec 20
§9.1, §9.2, §12; D-01..D-10, AUTO-01, AUTO-03..AUTO-06).

(`off` has its own module, test_autostart_off.py; `status` is tested with its own plan.)

`on_world` builds the one world every case shares: a global dir (conftest's hermetic `LOCALHARNESS_HOME`
renamed per case) with a llama.cpp `server:` record, a fake home, the unit file's directory under a redirected
`XDG_CONFIG_HOME`, a linger directory holding this user, a terminal operator, and a fake service manager
(`units_support.FakeManager`) that answers `daemon-reload`, the state `show` of the harness unit (loaded from
its own file, no drop-ins), `enable` and an empty `list-unit-files`. An argv nobody answered is the runner's
None, so a verb the code did not mean to issue fails the case.

Nothing here runs a service manager or a login manager, starts a unit or a server, or dials a port; the unit
file is written under tmp and linger is a stat under tmp.
"""
from __future__ import annotations

import errno
import fcntl
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import autostart, model_cmd
from localharness.cli.app import app
from localharness.cli.workspace import decide_machine_trust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle, server, units
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_serve_cmd import docker_peer, ollama_peer
from tests.unit.test_units_manager import answer_scan, show_block, unit_file
from tests.unit.trust_support import answers, never_asked
from tests.unit.unit_strategy_support import FakeServe
from tests.unit.units_support import FakeManager, fake_manager  # noqa: F401 — the fixture

REAL_OPERATOR_TEST = model_cmd._is_local_operator  # the real gate, kept before any case replaces it
SERVED = "qwen-27b-swift-1.5"
INTERPRETER = "/opt/lh/bin/python"
UNIT = units.HARNESS_UNIT
SHELL_PATH = "/usr/local/bin:/usr/bin:/bin"
RELOAD = ("systemctl", "--user", "daemon-reload")
STATE_SHOW = ("systemctl", "--user", "show", "-p", units._STATE_PROPS, "--", UNIT)
ENABLE_NOW = ("systemctl", "--user", "enable", "--now", UNIT)
ENABLE = ("systemctl", "--user", "enable", UNIT)
LIST_UNITS = ("systemctl", "--user", "list-unit-files")
TERMINAL_LINE = "autostart is set from a terminal: localharness model --autostart on"
DISPLAY = "~/.config/systemd/user/localharness-llm.service"


@dataclass
class World:
    tmp: Path
    g: Path
    home: Path
    linger: Path
    manager: FakeManager
    mp: pytest.MonkeyPatch

    def run(self, *args: str, mode: str = "on", config_dir: Path | None = None, before: tuple = ()):
        return CliRunner().invoke(
            app, [*before, "model", "--autostart", mode, "--config-dir", str(config_dir or self.g), *args])

    def manager_says(self, *, drop_ins: str = "", fragment: str | None = None, active: str = "inactive") -> None:
        text = "\n".join([
            f"FragmentPath={units.harness_unit_path() if fragment is None else fragment}",
            f"DropInPaths={drop_ins}", f"ActiveState={active}", "UnitFileState=disabled", "NRestarts=0",
            "ExecMainStatus=0", "MainPID=0", "InvocationID=", "Result=success", ""])
        self.manager.answer(lambda argv: argv[:3] == ("systemctl", "--user", "show") and "NRestarts" in argv[4],
                            out=text)

    @property
    def unit_path(self) -> Path:
        return units.harness_unit_path()

    @property
    def verbs(self) -> list[tuple[str, ...]]:
        return [c for c in self.manager.calls if c[2] not in ("show", "list-unit-files")]

    def body(self, captured: dict | None = None) -> str:
        return units.render_harness_unit(
            INTERPRETER, self.g, {"PATH": SHELL_PATH} if captured is None else captured,
            home=units.harness_home())

    def harness(self):
        return ConfigLoader(config_dir=self.g, local_config_dir=None).load_harness()


def write_config(g: Path, *, server_record: dict | None = None, peers=(), provider_key: str | None = None) -> None:
    record = server_record or {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                               "served_name": SERVED, "port": 8080}
    provider = {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8080/v1", "default_model": SERVED,
                "available_models": [SERVED]}
    if provider_key is not None:
        provider["api_key"] = provider_key
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1", "provider": provider,
        "org": {"default_model": SERVED, "audit_log_path": str(g / "audit.jsonl")},
        "extra_endpoints": list(peers), "server": record}), encoding="utf-8")


def on_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manager: FakeManager, *,
             dirname: str = ".localharness", peers=(), provider_key: str | None = None,
             linger: bool = True) -> World:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    g = tmp_path / dirname
    g.mkdir()
    monkeypatch.setenv("LOCALHARNESS_HOME", str(g))
    monkeypatch.setattr(sys, "executable", INTERPRETER)
    monkeypatch.setenv("PATH", SHELL_PATH)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    write_config(g, peers=peers, provider_key=provider_key)

    lingering = tmp_path / "linger"
    lingering.mkdir()
    if linger:
        (lingering / units.current_user()).write_text("", encoding="utf-8")
    monkeypatch.setattr(units, "LINGER_DIR", str(lingering))
    monkeypatch.setattr(units, "systemctl_path", lambda: "/usr/bin/systemctl")
    monkeypatch.setattr(model_cmd, "_is_local_operator", lambda: True)
    never_asked(monkeypatch)

    world = World(tmp_path, g, home, lingering, manager, monkeypatch)
    manager.answer(RELOAD, out="")
    manager.answer_prefix("systemctl", "--user", "enable", out="")
    manager.answer(LIST_UNITS + ("--type=service", "--plain", "--no-legend", "--no-pager"), out="")
    world.manager_says()
    return world


@pytest.fixture
def world(tmp_path, monkeypatch, fake_manager) -> World:
    return on_world(tmp_path, monkeypatch, fake_manager)


def receipt(captured: str = autostart.CAPTURED_TEXT, *, boot: bool = True, takes_over: bool = False) -> str:
    line = (autostart.RECEIPT_BOOT.format(path=DISPLAY, name=SERVED, captured=captured) if boot else
            autostart.RECEIPT_LOGIN.format(path=DISPLAY, name=SERVED, user=units.current_user(),
                                           captured=captured))
    return line + (autostart.TAKES_OVER_CLAUSE if takes_over else "")


def lock_is_free(g: Path) -> bool:
    fd = os.open(server.lock_path(g), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            return False
        raise
    finally:
        os.close(fd)
    return True


def stdout_lines(result) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.strip()]


# ===================================================================== the tracer


def test_on_from_a_terminal_writes_the_unit_adopts_its_entry_reloads_and_enables_it_now(world):
    result = world.run()

    assert result.exit_code == 0, (result.output, result.exception)
    assert world.unit_path.read_text(encoding="utf-8") == world.body()
    entry = trust.recorded_autostart_entry(world.g)
    assert entry is not None and entry == trust.autostart_entries(world.g)[0]
    assert world.manager.calls[:3] == [RELOAD, STATE_SHOW, ENABLE_NOW]
    assert world.manager.calls[3][:3] == LIST_UNITS  # the scan for a second launcher, after the receipt
    assert stdout_lines(result) == [
        f"autostart on: {DISPLAY} serves {SERVED} at login and boot (linger on; PATH captured from this shell)"]
    assert stdout_lines(result) == [receipt()]


def test_the_unit_file_is_the_harness_own_unit_by_all_three_tests(world):
    world.run()

    state = units.unit_runtime_state()

    assert state is not None
    assert units.harness_unit_failure(state, world.g) is None
