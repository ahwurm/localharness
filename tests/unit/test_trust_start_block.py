"""What `start` senses and writes before the one machine question (Phase 4, plan 05).

`_sense_before_trust` runs before `decide_machine_trust`: it says whether this start is about to
launch the primary (D-08, D-27), writes a `PATH`-found `llama-server` for a declared-only primary into
overrides.yaml so the question can show it (D-20), and converts a 0.17 `extra_args --host` (D-16).
Every write is one `workspace._write_overlay_validated` call, so each leaves one `ComponentMutated`
line (`actor="cli"`) in the audit log and a refused write leaves nothing (D-28).

The global dir is conftest's hermetic LOCALHARNESS_HOME; answers are `rich.prompt.Confirm.ask`; no
socket is ever opened (the one allowed connect is replaced) and no server is started.
"""
from __future__ import annotations

import json
import socket

import pytest
import yaml

from localharness.cli import start_cmd
from localharness.cli import workspace as ws_mod
from localharness.cli.start_cmd import (
    HOST_CONVERTED,
    SensedStart,
    _after_no_remove_path_binary,
    _sense_before_trust,
)
from localharness.cli.workspace import MachineTrust, decide_machine_trust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import server
from tests.unit.test_launch_paths_start import DECLARED_ONLY, _boot, _drive
from tests.unit.trust_support import answers, by_name, g, never_asked, write_yaml  # noqa: F401

PROVIDER = {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8081/v1", "default_model": "m"}
PATH_BIN = "/usr/bin/llama-server"
BINARY = ("launch", "server.binary")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    ws_mod._DECLINED.clear()
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    _Connect(monkeypatch, accepts=False)  # no test dials the port; one that cares replaces this
    yield
    ws_mod._DECLINED.clear()


def _config(g, **sections) -> None:
    write_yaml(g / "config.yaml", {"provider": PROVIDER, **sections})


def _record(g) -> None:
    """The machine already has a record, as every start after the first sees it."""
    trust.record_machine(g, trust.machine_snapshot(g))


def _overrides(g) -> dict:
    path = g / "overrides.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _audit(g) -> list[dict]:
    path = g / "audit.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("event_type") == "ComponentMutated"]


class _Connect:
    """Stands in for `socket.create_connection`: records each call and what it was closed with."""

    def __init__(self, monkeypatch, *, accepts: bool):
        self.calls: list[tuple[tuple, dict]] = []
        self.closed = 0
        owner = self

        class _Sock:
            def close(self) -> None:
                owner.closed += 1

        def create_connection(*a, **kw):
            self.calls.append((a, kw))
            if not accepts:
                raise ConnectionRefusedError("refused")
            return _Sock()

        monkeypatch.setattr(socket, "create_connection", create_connection)


def _err_lines(monkeypatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: seen.append(" ".join(map(str, a))))
    return seen


# ------------------------------------------------------------------ is this start about to launch?

LLAMACPP_OPEN = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                 "port": 8123, "bind_all": True}


def _uncovered_llamacpp(g) -> None:
    """A recorded machine whose files now hold a llama.cpp `bind_all: true` nobody confirmed."""
    _config(g, server={**LLAMACPP_OPEN, "bind_all": False})
    _record(g)
    _config(g, server=LLAMACPP_OPEN)


def test_a_live_pidfile_means_not_launching_and_no_connect_is_made(g, monkeypatch):
    _uncovered_llamacpp(g)
    monkeypatch.setattr(server, "server_pid", lambda _d: 4321)
    connect = _Connect(monkeypatch, accepts=True)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.launching is False and connect.calls == []


def test_an_accepted_connect_means_a_server_is_up_and_not_launching(g, monkeypatch):
    _uncovered_llamacpp(g)
    connect = _Connect(monkeypatch, accepts=True)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.launching is False
    assert connect.calls == [((("127.0.0.1", 8123),), {"timeout": 0.5})]
    assert connect.closed == 1


def test_a_refused_connect_means_launching(g, monkeypatch):
    _uncovered_llamacpp(g)
    connect = _Connect(monkeypatch, accepts=False)

    assert _sense_before_trust(g, can_ask=True, path_binary=None).launching is True
    assert len(connect.calls) == 1


def test_a_timed_out_connect_means_launching(g, monkeypatch):
    _uncovered_llamacpp(g)

    def slow(*a, **kw):
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", slow)

    assert _sense_before_trust(g, can_ask=True, path_binary=None).launching is True


def test_the_port_is_the_records_own_and_defaults_when_unset(g, monkeypatch):
    _config(g, server={k: v for k, v in LLAMACPP_OPEN.items() if k not in ("port", "bind_all")})
    _record(g)
    _config(g, server={k: v for k, v in LLAMACPP_OPEN.items() if k != "port"})
    connect = _Connect(monkeypatch, accepts=False)

    _sense_before_trust(g, can_ask=True, path_binary=None)

    from localharness.config.models import ManagedServerConfig
    assert connect.calls[0][0] == (("127.0.0.1", ManagedServerConfig.model_fields["port"].default),)


@pytest.mark.parametrize("server_block", [
    {**LLAMACPP_OPEN, "bind_all": False},            # nothing exposed
    {k: v for k, v in LLAMACPP_OPEN.items() if k != "bind_all"},   # bind_all unset
    {"runtime": "vllm", "binary": "/v/vllm", "model": "org/m", "bind_all": True},  # not llama.cpp
], ids=["bind_all-false", "bind_all-unset", "vllm"])
def test_any_other_record_is_launching_with_no_connect(g, monkeypatch, server_block):
    _config(g, server={**server_block, "bind_all": False} if "bind_all" in server_block else server_block)
    _record(g)
    _config(g, server=server_block)
    connect = _Connect(monkeypatch, accepts=True)

    assert _sense_before_trust(g, can_ask=True, path_binary=None).launching is True
    assert connect.calls == []


def test_a_covered_llamacpp_true_is_launching_with_no_connect(g, monkeypatch):
    _config(g, server=LLAMACPP_OPEN)
    _record(g)  # the record already covers the bind_all: true
    connect = _Connect(monkeypatch, accepts=True)

    assert _sense_before_trust(g, can_ask=True, path_binary=None).launching is True
    assert connect.calls == []


def test_a_machine_with_no_record_is_launching_with_no_connect(g, monkeypatch):
    _config(g, server=LLAMACPP_OPEN)
    connect = _Connect(monkeypatch, accepts=True)

    assert _sense_before_trust(g, can_ask=True, path_binary=None) == SensedStart(True, {})
    assert connect.calls == []


def test_the_connect_is_made_once_per_sense(g, monkeypatch):
    _uncovered_llamacpp(g)
    connect = _Connect(monkeypatch, accepts=False)

    _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)

    assert len(connect.calls) == 1


# ------------------------------------------------------------------ a PATH-found binary (D-20)

DECLARED = {"runtime": "llamacpp", "model": "/m/a.gguf", "port": 8123}


def test_a_path_binary_for_a_declared_only_primary_is_written_before_the_question(g, monkeypatch):
    _config(g, server=DECLARED)
    _record(g)
    err = _err_lines(monkeypatch)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)

    assert sensed.path_written == PATH_BIN
    assert _overrides(g) == {"server": {"binary": PATH_BIN}}
    assert len(err) == 1 and PATH_BIN in err[0] and "overrides.yaml" in err[0]


def test_the_path_write_is_audited_once_by_the_one_writer(g, monkeypatch):
    _config(g, server=DECLARED)
    _record(g)

    _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)

    (event,) = _audit(g)
    assert event["path"] == "server.binary" and event["before_value"] is None
    assert event["after_value"] == PATH_BIN and event["layer"] == "user" and event["actor"] == "cli"


@pytest.mark.parametrize("where", ["config.yaml", "overrides.yaml"])
def test_an_existing_binary_in_either_file_is_never_overwritten(g, where):
    _config(g, server={**DECLARED, **({"binary": "/mine/llama-server"} if where == "config.yaml" else {})})
    if where == "overrides.yaml":
        write_yaml(g / "overrides.yaml", {"server": {"binary": "/mine/llama-server"}})
    _record(g)
    before = _overrides(g)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)

    assert sensed.path_written is None and _overrides(g) == before and _audit(g) == []


def test_a_remembered_no_for_this_path_writes_nothing(g):
    _config(g, server=DECLARED)
    _record(g)
    trust.record_offer_decline(trust.decline_key("<binary>", PATH_BIN))

    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)

    assert sensed.path_written is None and not (g / "overrides.yaml").exists() and _audit(g) == []


def test_no_path_binary_and_no_declared_llamacpp_write_nothing(g):
    _config(g, server=DECLARED)
    _record(g)
    assert _sense_before_trust(g, can_ask=True, path_binary=None).path_written is None
    _config(g, server={"runtime": "vllm", "binary": "/v/vllm", "model": "org/m"})
    _record(g)
    assert _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN).path_written is None
    _config(g)
    _record(g)
    assert _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN).path_written is None
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_a_machine_with_no_record_is_never_written_to_from_the_path(g):
    """The first start adopts the disk unasked: a binary the harness found must not be adopted that way."""
    _config(g, server=DECLARED)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)

    assert sensed.path_written is None and not (g / "overrides.yaml").exists() and _audit(g) == []


def test_a_write_that_would_not_load_is_not_made(g):
    write_yaml(g / "config.yaml", {"server": DECLARED})  # no provider: block, the config would not load
    _record(g)

    assert _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN).path_written is None
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_the_question_lists_the_path_binary_and_a_yes_records_it(g, monkeypatch):
    _config(g, server=DECLARED)
    _record(g)
    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)
    asked = answers(monkeypatch, True)

    got = decide_machine_trust(g, ask=True, launching=sensed.launching, annotations=sensed.annotations)

    assert len(asked) == 1 and f"+ server.binary (overrides.yaml): {PATH_BIN}" in asked[0]
    assert got.withheld == {} and got.line == ""
    assert by_name(trust.recorded_machine(g)["entries"], "server.binary")["shown"] == PATH_BIN
    loaded = ConfigLoader(config_dir=g, machine_withheld=got.withheld).load_harness()
    assert loaded.server is not None and loaded.server.binary == PATH_BIN


def test_a_start_that_cannot_ask_withholds_it_and_the_server_is_dropped(g, monkeypatch):
    _config(g, server=DECLARED)
    _record(g)
    sensed = _sense_before_trust(g, can_ask=False, path_binary=PATH_BIN)
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=False, launching=sensed.launching, annotations=sensed.annotations)

    assert got.withheld == {"overrides.yaml": frozenset({BINARY})}
    assert "server.binary (overrides.yaml)" in got.line
    loaded = ConfigLoader(config_dir=g, machine_withheld=got.withheld).load_harness()
    assert loaded.server is None
    assert "server.binary" not in {e["name"] for e in trust.recorded_machine(g)["entries"]}


def test_a_no_removes_the_line_audits_it_and_remembers_the_path(g, monkeypatch):
    _config(g, server=DECLARED)
    _record(g)
    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)
    answers(monkeypatch, False)
    got = decide_machine_trust(g, ask=True, launching=sensed.launching, annotations=sensed.annotations)
    assert BINARY in got.withheld["overrides.yaml"]

    _after_no_remove_path_binary(g, sensed.path_written)

    assert _overrides(g) in ({}, None) or "server" not in _overrides(g)
    written, removed = _audit(g)
    assert (written["path"], written["after_value"]) == ("server.binary", PATH_BIN)
    assert removed["path"] == "server.binary" and removed["before_value"] == PATH_BIN
    assert removed["after_value"] is None and removed["layer"] == "user" and removed["actor"] == "cli"
    assert trust.offer_was_declined(trust.decline_key("<binary>", PATH_BIN))


def test_a_later_start_with_the_same_path_writes_nothing_and_asks_nothing(g, monkeypatch):
    _config(g, server=DECLARED)
    _record(g)
    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)
    answers(monkeypatch, False)
    decide_machine_trust(g, ask=True, launching=True, annotations={})
    _after_no_remove_path_binary(g, sensed.path_written)
    ws_mod._DECLINED.clear()  # the next process
    never_asked(monkeypatch)

    again = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)
    got = decide_machine_trust(g, ask=True, launching=again.launching, annotations=again.annotations)

    assert again.path_written is None and got.withheld == {} and len(_audit(g)) == 2


def test_a_different_path_writes_and_asks_again(g, monkeypatch):
    _config(g, server=DECLARED)
    _record(g)
    _after_no_remove_path_binary(g, PATH_BIN)
    ws_mod._DECLINED.clear()

    sensed = _sense_before_trust(g, can_ask=True, path_binary="/opt/bin/llama-server")
    asked = answers(monkeypatch, True)
    decide_machine_trust(g, ask=True, launching=True, annotations=sensed.annotations)

    assert sensed.path_written == "/opt/bin/llama-server"
    assert len(asked) == 1 and "/opt/bin/llama-server" in asked[0]


def test_a_changed_value_stays_and_only_the_no_is_remembered(g):
    _config(g, server=DECLARED)
    _record(g)
    write_yaml(g / "overrides.yaml", {"server": {"binary": "/mine/llama-server"}})

    _after_no_remove_path_binary(g, PATH_BIN)

    assert _overrides(g) == {"server": {"binary": "/mine/llama-server"}} and _audit(g) == []
    assert trust.offer_was_declined(trust.decline_key("<binary>", PATH_BIN))


def test_a_no_keeps_the_rest_of_the_servers_overlay(g):
    _config(g, server=DECLARED)
    _record(g)
    write_yaml(g / "overrides.yaml", {"server": {"binary": PATH_BIN, "port": 9000}, "keep": 1})

    _after_no_remove_path_binary(g, PATH_BIN)

    assert _overrides(g) == {"server": {"port": 9000}, "keep": 1}


# ------------------------------------------------------------------ through the real start


async def test_a_headless_start_withholds_the_path_binary_and_launches_nothing(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY, path_binary=PATH_BIN)
    _record(tmp_path)

    assert await _drive(tmp_path) == 1

    assert seen.started == []
    assert _overrides(tmp_path) == {"server": {"binary": PATH_BIN}}  # shown to the next terminal start
    assert any("server.binary (overrides.yaml)" in line for line in
               (" ".join(map(str, a)) for a, _ in seen.err))


async def test_a_terminal_yes_records_the_path_binary_and_launches_it(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY, path_binary=PATH_BIN)
    _record(tmp_path)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    asked = answers(monkeypatch, True)

    await _drive(tmp_path)

    machine_q = [q for q in asked if f"+ server.binary (overrides.yaml): {PATH_BIN}" in q]
    assert len(machine_q) == 1
    assert [cmd[0] for cmd, _ in seen.started] == [PATH_BIN]


async def test_a_terminal_no_removes_the_line_remembers_it_and_launches_nothing(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY, path_binary=PATH_BIN)
    _record(tmp_path)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    answers(monkeypatch, False)

    assert await _drive(tmp_path) == 1

    assert seen.started == []
    assert "server" not in _overrides(tmp_path)
    assert trust.offer_was_declined(trust.decline_key("<binary>", PATH_BIN))
    assert [e["after_value"] for e in _audit(tmp_path)] == [PATH_BIN, None]


# ------------------------------------------------------------------ a 0.17 extra_args --host (D-16)

LLAMACPP = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf", "port": 8123}
VLLM_BINARY = {"runtime": "vllm", "binary": "/v/vllm", "model": "org/m", "port": 8123}
OLD_LAUNCH_KEYS = ("binary", "docker_image", "extra_args")
BIND_ALL = ("overrides.yaml", "launch", "server.bind_all")


def _pre_upgrade(g) -> None:
    """The record 0.17 wrote: every entry the old tables knew, and no `launch_model` kind."""
    snap = trust.machine_snapshot(g)
    old = [e for e in snap if e["kind"] != trust.LAUNCH_MODEL_KIND
           and (e["kind"] != trust.LAUNCH_KIND or e["name"].rsplit(".", 1)[-1] in OLD_LAUNCH_KEYS)]
    trust.record_machine(g, old, kinds=trust.MACHINE_KINDS - {trust.LAUNCH_MODEL_KIND})


def _with_extra_args(g, extra_args, *, base=LLAMACPP, upgrade=True, **more) -> None:
    """A recorded machine whose server section carries ``extra_args`` (the record covers them)."""
    _config(g, server={**base, "extra_args": extra_args, **more})
    _pre_upgrade(g) if upgrade else _record(g)


def _audit_bind_all(g) -> list[dict]:
    return [e for e in _audit(g) if e["path"] == "server.bind_all"]


@pytest.mark.parametrize(("base", "extra_args"), [
    (LLAMACPP, ["--host", "0.0.0.0", "-ngl", "99"]),
    (LLAMACPP, ["--host=0.0.0.0"]),
    (LLAMACPP, ["--host", "127.0.0.1", "--host", "0.0.0.0"]),
    (LLAMACPP, ["--host", "0.0.0.0,0.0.0.0"]),
    (VLLM_BINARY, ["--host", "0.0.0.0"]),
], ids=["spaced", "equals-form", "last-wins", "list-of-the-same", "vllm-binary"])
@pytest.mark.parametrize("can_ask", [True, False], ids=["terminal", "headless"])
def test_a_0_17_all_interfaces_host_converts_at_the_upgrade_start_with_no_question(
        g, monkeypatch, base, extra_args, can_ask):
    _with_extra_args(g, extra_args, base=base)

    sensed = _sense_before_trust(g, can_ask=can_ask, path_binary=None)

    assert sensed.converted == HOST_CONVERTED == (
        "server.bind_all: true, from server.extra_args --host 0.0.0.0, which 0.17 applied")
    assert sensed.undo_line is None and sensed.annotations == {}
    assert _overrides(g) == {"server": {"bind_all": True}}
    runtime = base["runtime"]
    assert by_name(trust.recorded_machine(g)["entries"], "server.bind_all") == {
        "file": "overrides.yaml", "kind": "launch", "name": "server.bind_all",
        "shown": f"{runtime}:true:open"}
    never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=can_ask, launching=sensed.launching,
                                annotations=sensed.annotations) == MachineTrust()
    assert trust.LAUNCH_MODEL_KIND in trust.recorded_machine(g)["kinds"]


def test_the_conversion_is_audited_once_by_the_one_writer(g):
    _with_extra_args(g, ["--host", "0.0.0.0"])

    _sense_before_trust(g, can_ask=False, path_binary=None)

    (event,) = _audit(g)
    assert event["path"] == "server.bind_all" and event["before_value"] is None
    assert event["after_value"] is True and event["layer"] == "user" and event["actor"] == "cli"
    assert "--host 0.0.0.0" in event["actor_detail"]


def test_a_second_start_finds_the_conversion_recorded_and_converts_nothing_again(g):
    _with_extra_args(g, ["--host", "0.0.0.0"])
    _sense_before_trust(g, can_ask=True, path_binary=None)
    ws_mod._DECLINED.clear()
    decide_machine_trust(g, ask=True)

    again = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert again.converted is None and again.undo_line is None and again.annotations == {}
    assert len(_audit_bind_all(g)) == 1


def test_a_last_host_of_loopback_converts_nothing_and_says_nothing(g):
    _with_extra_args(g, ["--host", "0.0.0.0", "--host", "127.0.0.1"])

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert (sensed.converted, sensed.undo_line, sensed.annotations) == (None, None, {})
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_no_host_and_no_extra_args_convert_nothing(g):
    _with_extra_args(g, ["-ngl", "99"])
    assert _sense_before_trust(g, can_ask=True, path_binary=None) == SensedStart(True, {})
    _config(g, server=LLAMACPP)
    _pre_upgrade(g)
    assert _sense_before_trust(g, can_ask=True, path_binary=None) == SensedStart(True, {})
    assert _audit(g) == []


@pytest.mark.parametrize("bind_all", [True, False])
def test_a_bind_all_already_in_either_file_converts_nothing(g, bind_all):
    _with_extra_args(g, ["--host", "0.0.0.0"], bind_all=bind_all)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert (sensed.converted, sensed.undo_line) == (None, None) and _audit(g) == []
    _with_extra_args(g, ["--host", "0.0.0.0"])
    write_yaml(g / "overrides.yaml", {"server": {"bind_all": bind_all}})
    assert _sense_before_trust(g, can_ask=True, path_binary=None).converted is None
    assert _audit(g) == []


def test_a_record_that_already_knows_launch_model_converts_nothing_even_with_the_same_files(g):
    _with_extra_args(g, ["--host", "0.0.0.0"], upgrade=False)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.converted is None and sensed.annotations == {}
    assert sensed.undo_line is not None and "--host 0.0.0.0" in sensed.undo_line
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_the_overrides_files_extra_args_win_over_config_yamls(g):
    _config(g, server=LLAMACPP)
    write_yaml(g / "overrides.yaml", {"server": {"extra_args": ["--host", "0.0.0.0"]}})
    _pre_upgrade(g)

    sensed = _sense_before_trust(g, can_ask=False, path_binary=None)

    assert sensed.converted == HOST_CONVERTED
    assert _overrides(g)["server"]["bind_all"] is True


# another address: offered once, on a terminal


def test_another_address_is_written_and_offered_with_its_source_at_the_upgrade_start(g, monkeypatch):
    _with_extra_args(g, ["--host", "192.168.1.5"])

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    note = "(from server.extra_args --host 192.168.1.5, which 0.17 applied)"
    assert sensed.annotations == {BIND_ALL: note}
    assert sensed.converted is None and sensed.undo_line is None
    assert _overrides(g) == {"server": {"bind_all": True}}
    (event,) = _audit(g)
    assert (event["path"], event["before_value"], event["after_value"], event["actor"]) == (
        "server.bind_all", None, True, "cli")
    asked = answers(monkeypatch, True)
    got = decide_machine_trust(g, ask=True, launching=sensed.launching, annotations=sensed.annotations)
    assert len(asked) == 1 and got == MachineTrust()
    (line,) = [ln for ln in asked[0].splitlines() if "server.bind_all" in ln]
    assert line.endswith(f" {note}") and "never applied to llama.cpp" in line


def test_a_no_on_another_address_writes_false_and_remembers_it(g, monkeypatch):
    _with_extra_args(g, ["--host", "192.168.1.5"])
    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)
    answers(monkeypatch, False)

    decide_machine_trust(g, ask=True, launching=True, annotations=sensed.annotations)

    assert _overrides(g) == {"server": {"bind_all": False}}
    assert [e["after_value"] for e in _audit_bind_all(g)] == [True, False]
    ws_mod._DECLINED.clear()
    never_asked(monkeypatch)
    again = _sense_before_trust(g, can_ask=True, path_binary=None)
    assert again.annotations == {} and again.converted is None and again.undo_line is None
    assert decide_machine_trust(g, ask=True, launching=True) == MachineTrust()


def test_another_address_after_the_upgrade_start_has_no_which_0_17_applied(g):
    _with_extra_args(g, ["--host", "192.168.1.5"], upgrade=False)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.annotations == {BIND_ALL: "(from server.extra_args --host 192.168.1.5)"}


def test_another_address_on_a_headless_start_is_not_written_and_the_launch_says_how_to_restore_it(g):
    _with_extra_args(g, ["--host", "192.168.1.5"])

    sensed = _sense_before_trust(g, can_ask=False, path_binary=None)

    assert sensed.converted is None and sensed.annotations == {}
    assert sensed.undo_line == ("server.extra_args --host 192.168.1.5 is not applied — launched on "
                                "127.0.0.1 — server.bind_all: true restores it (asks once)")
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_a_vllm_binary_with_another_address_is_offered_like_a_llamacpp_one(g):
    _with_extra_args(g, ["--host", "10.0.0.7"], base=VLLM_BINARY, upgrade=False)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.annotations == {BIND_ALL: "(from server.extra_args --host 10.0.0.7)"}
    assert _overrides(g) == {"server": {"bind_all": True}}


# everything else: loopback and the undo line


@pytest.mark.parametrize("host", ["ip6-localhost", "/run/llama.sock", "my-box.lan"])
def test_a_hostname_or_a_socket_path_converts_nothing_and_the_launch_says_how_to_restore_it(g, monkeypatch, host):
    resolved: list = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: resolved.append(a) or (_ for _ in ()).throw(
        AssertionError("a --host value was resolved through DNS")))
    _with_extra_args(g, ["--host", host])

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.converted is None and sensed.annotations == {} and resolved == []
    assert sensed.undo_line.endswith("launched on 127.0.0.1 — server.bind_all: true restores it (asks once)")
    assert f"--host {host}" in sensed.undo_line
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_an_extra_args_entry_the_record_does_not_cover_is_never_converted(g):
    _config(g, server={**LLAMACPP, "extra_args": ["--host", "127.0.0.1"]})
    _pre_upgrade(g)  # the record covers ["--host", "127.0.0.1"] ...
    _config(g, server={**LLAMACPP, "extra_args": ["--host", "0.0.0.0"]})  # ... an agent then edited it

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.converted is None and sensed.annotations == {}
    assert sensed.undo_line is not None and "--host 0.0.0.0" in sensed.undo_line
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_a_machine_with_no_record_converts_nothing(g):
    _config(g, server={**LLAMACPP, "extra_args": ["--host", "0.0.0.0"]})

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.converted is None and sensed.undo_line is not None
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5"])
def test_a_vllm_docker_record_converts_nothing_and_says_nothing(g, host):
    docker = {"runtime": "vllm", "launch": "docker", "docker_image": "img:tag", "model": "org/m"}
    _with_extra_args(g, ["--host", host], base=docker)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed == SensedStart(True, {}) and sensed.undo_line is None
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_an_ollama_record_converts_nothing_and_says_nothing(g):
    _with_extra_args(g, ["--host", "0.0.0.0"], base={"runtime": "ollama", "model": "m"})

    assert _sense_before_trust(g, can_ask=True, path_binary=None) == SensedStart(True, {})


def test_a_write_that_would_not_load_converts_nothing_and_the_launch_is_on_loopback(g):
    write_yaml(g / "config.yaml", {"server": {**LLAMACPP, "extra_args": ["--host", "0.0.0.0"]}})  # no provider
    _pre_upgrade(g)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.converted is None and sensed.undo_line is not None
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_a_path_binary_and_a_conversion_in_one_start_are_two_audited_writes(g):
    _config(g, server={"runtime": "llamacpp", "model": "/m/a.gguf", "extra_args": ["--host", "0.0.0.0"]})
    _pre_upgrade(g)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=PATH_BIN)

    assert sensed.path_written == PATH_BIN and sensed.converted == HOST_CONVERTED
    assert [e["path"] for e in _audit(g)] == ["server.binary", "server.bind_all"]
    assert _overrides(g) == {"server": {"binary": PATH_BIN, "bind_all": True}}
