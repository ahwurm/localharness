"""`start` senses llama-server's own `config.ini` before the machine question (Phase 5, D-31, D-56, D-57).

The real `${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini` is masked by the one function `init` also
uses, written to overrides.yaml as `server.config_ini` through `workspace.write_overlay_validated` (never
by a writer of start's own), shown by the machine question as an ordinary uncovered launch entry, removed
and remembered on a No, and withheld on a headless start. Nothing here reads the owner's real file: every
test points `XDG_CONFIG_HOME` at its own temp directory, and nothing dials a port.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import socket
from pathlib import Path

import pytest
import yaml

from localharness.cli import start_cmd
from localharness.cli import workspace as ws_mod
from localharness.cli.start_cmd import (
    CONFIG_INI_REFUSED_LINE,
    CONFIG_INI_WRITTEN_LINE,
    _after_no_on_config_ini,
    _after_no_remove_config_ini,
    _sense_before_trust,
)
from localharness.cli.workspace import MachineTrust, decide_machine_trust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import server
from tests.unit.test_launch_paths_start import _boot, _drive
from tests.unit.trust_support import answers, by_name, g, never_asked, write_yaml  # noqa: F401

PROVIDER = {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8081/v1", "default_model": "m"}
LLAMACPP = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf", "port": 8123}
INI_ENTRY = ("launch", "server.config_ini")
INI = "ctx-size = 8192\napi-key = K\n"
KEPT = "ctx-size = 8192\n"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    ws_mod._DECLINED.clear()
    monkeypatch.setattr(server, "server_pid", lambda _d: None)

    def _refused(*_a, **_kw):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(socket, "create_connection", _refused)
    yield
    ws_mod._DECLINED.clear()


@pytest.fixture
def xdg(tmp_path, monkeypatch) -> Path:
    """A private `XDG_CONFIG_HOME`; the returned path is the `llama.cpp` directory inside it."""
    root = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root))
    return root / "llama.cpp"


def _ini(xdg: Path, text: str | bytes = INI) -> Path:
    xdg.mkdir(parents=True, exist_ok=True)
    path = xdg / "config.ini"
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


def _config(g, **sections) -> None:
    write_yaml(g / "config.yaml", {"provider": PROVIDER, **sections})


def _record(g) -> None:
    trust.record_machine(g, trust.machine_snapshot(g))


def _overrides(g) -> dict:
    path = g / "overrides.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}


def _audit(g) -> list[dict]:
    path = g / "audit.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("event_type") == "ComponentMutated"]


def _err_lines(monkeypatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: seen.append(" ".join(map(str, a))))
    return seen


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _declined(text: str) -> bool:
    return trust.offer_was_declined(trust.decline_key("<config.ini>", _digest(text)))


def _recorded_machine(g, **server_extra) -> None:
    _config(g, server={**LLAMACPP, **server_extra})
    _record(g)


# ------------------------------------------------------------------ the sensing write (D-31, D-57)


def test_a_changed_config_ini_is_written_masked_before_the_question(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    err = _err_lines(monkeypatch)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.ini_written == KEPT
    assert _overrides(g) == {"server": {"config_ini": KEPT}}
    assert len(err) == 1 and "overrides.yaml" in err[0] and "server.config_ini" in err[0]
    assert err[0] == CONFIG_INI_WRITTEN_LINE.format(dir=ws_mod.shown_path(xdg))


def test_the_key_never_reaches_overrides_or_the_audit_log(g, xdg):
    _ini(xdg, "api-key = SECRET-KEY-VALUE\nhf-token = hf_secret\nctx-size = 8192\n")
    _recorded_machine(g)

    _sense_before_trust(g, can_ask=True, path_binary=None)

    for name in ("overrides.yaml", "audit.jsonl"):
        text = (g / name).read_text(encoding="utf-8")
        assert "SECRET-KEY-VALUE" not in text and "hf_secret" not in text


def test_the_write_is_audited_once_by_the_one_writer(g, xdg):
    _ini(xdg)
    _recorded_machine(g)

    _sense_before_trust(g, can_ask=True, path_binary=None)

    (event,) = _audit(g)
    assert event["path"] == "server.config_ini" and event["before_value"] is None
    assert event["after_value"] == KEPT and event["layer"] == "user" and event["actor"] == "cli"


def test_start_defines_no_overlay_writer_of_its_own():
    source = Path(start_cmd.__file__).read_text(encoding="utf-8")
    assert "def write_overlay_validated" not in source and "def raw_settings" not in source


def test_the_question_lists_the_entry_and_a_yes_records_it(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)
    asked = answers(monkeypatch, True)

    got = decide_machine_trust(g, ask=True, launching=sensed.launching, annotations=sensed.annotations)

    assert len(asked) == 1 and "+ server.config_ini (overrides.yaml)" in asked[0]
    assert got.withheld == {} and got.line == ""
    assert by_name(trust.recorded_machine(g)["entries"], "server.config_ini")
    loaded = ConfigLoader(config_dir=g, machine_withheld=got.withheld).load_harness()
    assert loaded.server is not None and loaded.server.config_ini == KEPT


def test_the_next_start_with_the_same_file_writes_nothing_and_asks_nothing(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    first = _sense_before_trust(g, can_ask=True, path_binary=None)
    answers(monkeypatch, True)
    decide_machine_trust(g, ask=True, launching=first.launching, annotations=first.annotations)
    audited = len(_audit(g))
    never_asked(monkeypatch)

    again = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert again.ini_written is None and len(_audit(g)) == audited
    assert decide_machine_trust(g, ask=True, launching=again.launching,
                                annotations=again.annotations) == MachineTrust()


def test_an_older_value_in_config_yaml_is_overwritten_through_overrides(g, xdg):
    """Unlike the PATH binary, an existing `config_ini` is not respected: the file on disk wins."""
    _ini(xdg)
    _recorded_machine(g, config_ini="ctx-size = 4096\n")

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.ini_written == KEPT
    assert _overrides(g) == {"server": {"config_ini": KEPT}}
    (event,) = _audit(g)
    assert event["before_value"] == "ctx-size = 4096\n" and event["after_value"] == KEPT


def test_the_same_text_already_merged_writes_nothing(g, xdg):
    _ini(xdg)
    _recorded_machine(g, config_ini=KEPT)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.ini_written is None
    assert not (g / "overrides.yaml").exists() and _audit(g) == []


def test_no_machine_record_writes_nothing(g, xdg):
    _ini(xdg)
    _config(g, server=LLAMACPP)  # no _record: the first start adopts the disk unasked

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.ini_written is None and not (g / "overrides.yaml").exists()


def test_a_start_that_cannot_ask_writes_nothing(g, xdg):
    _ini(xdg)
    _recorded_machine(g)

    sensed = _sense_before_trust(g, can_ask=False, path_binary=None)

    assert sensed.ini_written is None and not (g / "overrides.yaml").exists() and _audit(g) == []


def test_a_start_that_only_attaches_writes_nothing(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    monkeypatch.setattr(start_cmd, "_sensed_launching", lambda *_a, **_k: False)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.launching is False and sensed.ini_written is None
    assert not (g / "overrides.yaml").exists()


def test_a_vllm_primary_writes_nothing(g, xdg):
    _ini(xdg)
    _config(g, server={"runtime": "vllm", "binary": "/v/vllm", "model": "org/m", "port": 8123})
    _record(g)

    assert _sense_before_trust(g, can_ask=True, path_binary=None).ini_written is None
    assert not (g / "overrides.yaml").exists()


def test_an_absent_file_writes_and_prints_nothing(g, xdg, monkeypatch):
    _recorded_machine(g)
    err = _err_lines(monkeypatch)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.ini_written is None and err == [] and not (g / "overrides.yaml").exists()


@pytest.mark.parametrize("text", ["", "api-key = K\n", "host = 0.0.0.0\nport = 9\n"])
def test_a_file_that_masks_to_nothing_writes_nothing(g, xdg, text):
    _ini(xdg, text)
    _recorded_machine(g)

    assert _sense_before_trust(g, can_ask=True, path_binary=None).ini_written is None
    assert not (g / "overrides.yaml").exists()


# ------------------------------------------------------------------ a No removes and remembers (D-31)


def _ask_no(g, monkeypatch, *, can_ask=True):
    sensed = _sense_before_trust(g, can_ask=can_ask, path_binary=None)
    answers(monkeypatch, False)
    got = decide_machine_trust(g, ask=True, launching=sensed.launching, annotations=sensed.annotations)
    assert INI_ENTRY in got.withheld["overrides.yaml"]
    return sensed, got


def test_a_no_removes_the_line_audits_it_and_remembers_the_digest(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    sensed, got = _ask_no(g, monkeypatch)

    _after_no_on_config_ini(g, got, can_ask=True, ini_written=sensed.ini_written)

    assert "server" not in _overrides(g)
    assert _declined(KEPT)
    assert [e["after_value"] for e in _audit(g)] == [KEPT, None]
    assert _audit(g)[-1]["actor"] == "cli" and _audit(g)[-1]["path"] == "server.config_ini"


def test_the_next_terminal_start_with_the_same_file_writes_and_asks_nothing(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    sensed, got = _ask_no(g, monkeypatch)
    _after_no_on_config_ini(g, got, can_ask=True, ini_written=sensed.ini_written)
    ws_mod._DECLINED.clear()  # the next process
    never_asked(monkeypatch)

    again = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert again.ini_written is None and _overrides(g) == {}
    assert decide_machine_trust(g, ask=True, launching=again.launching,
                                annotations=again.annotations).withheld == {}


def test_a_changed_file_writes_and_asks_again(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    sensed, got = _ask_no(g, monkeypatch)
    _after_no_on_config_ini(g, got, can_ask=True, ini_written=sensed.ini_written)
    _ini(xdg, "ctx-size = 16384\n")

    again = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert again.ini_written == "ctx-size = 16384\n"
    assert _overrides(g) == {"server": {"config_ini": "ctx-size = 16384\n"}}


def test_a_value_someone_changed_after_the_write_stays_and_only_the_no_is_remembered(g):
    _config(g, server=LLAMACPP)
    _record(g)
    write_yaml(g / "overrides.yaml", {"server": {"config_ini": "ctx-size = 1\n"}})

    _after_no_remove_config_ini(g, KEPT)

    assert _overrides(g) == {"server": {"config_ini": "ctx-size = 1\n"}}
    assert _declined(KEPT) and _audit(g) == []


def test_a_no_keeps_the_rest_of_the_servers_overlay(g):
    _config(g, server=LLAMACPP)
    _record(g)
    write_yaml(g / "overrides.yaml", {"server": {"config_ini": KEPT, "port": 8124}})

    _after_no_remove_config_ini(g, KEPT)

    assert _overrides(g) == {"server": {"port": 8124}}


def test_a_value_the_record_already_holds_is_never_removed(g):
    _config(g, server=LLAMACPP)
    write_yaml(g / "overrides.yaml", {"server": {"config_ini": KEPT}})
    _record(g)  # the person confirmed it: it is the record's, not an addition

    _after_no_remove_config_ini(g, KEPT)

    assert _overrides(g) == {"server": {"config_ini": KEPT}} and _declined(KEPT)


def test_a_line_an_earlier_unanswered_start_left_is_still_the_pending_entry(g, xdg, monkeypatch):
    """WR-02: the earlier start wrote the line and was interrupted before the answer; this start finds
    it already merged, still reports it as pending, and a No removes it."""
    _ini(xdg)
    _recorded_machine(g)
    _sense_before_trust(g, can_ask=True, path_binary=None)
    audited = len(_audit(g))

    sensed, got = _ask_no(g, monkeypatch)
    assert sensed.ini_written == KEPT and len(_audit(g)) == audited  # nothing written again
    _after_no_on_config_ini(g, got, can_ask=True, ini_written=sensed.ini_written)

    assert _overrides(g) == {} and _declined(KEPT)


def test_a_start_that_cannot_ask_removes_and_declines_nothing(g):
    _config(g, server=LLAMACPP)
    _record(g)
    write_yaml(g / "overrides.yaml", {"server": {"config_ini": KEPT}})
    withheld = MachineTrust(withheld={"overrides.yaml": frozenset({INI_ENTRY})})

    _after_no_on_config_ini(g, withheld, can_ask=False, ini_written=KEPT)
    _after_no_on_config_ini(g, MachineTrust(), can_ask=True, ini_written=KEPT)
    _after_no_on_config_ini(g, withheld, can_ask=True, ini_written=None)

    assert _overrides(g) == {"server": {"config_ini": KEPT}} and not _declined(KEPT)


# ------------------------------------------------------------------ headless withholds (D-31)


def test_a_headless_start_writes_nothing_and_withholds_a_line_an_earlier_terminal_start_left(g, xdg, monkeypatch):
    _ini(xdg)
    _recorded_machine(g)
    write_yaml(g / "overrides.yaml", {"server": {"config_ini": KEPT}})  # an earlier terminal start's
    audited = len(_audit(g))
    sensed = _sense_before_trust(g, can_ask=False, path_binary=None)
    never_asked(monkeypatch)

    got = decide_machine_trust(g, ask=False, launching=sensed.launching, annotations=sensed.annotations)

    assert sensed.ini_written is None and len(_audit(g)) == audited
    assert INI_ENTRY in got.withheld["overrides.yaml"]
    loaded = ConfigLoader(config_dir=g, machine_withheld=got.withheld).load_harness()
    assert loaded.server is None  # a withheld launch entry drops the whole server: block


# ------------------------------------------------------------------ the not-carried lines (D-56)


@pytest.mark.parametrize("line, key, category", [
    ("hf-repo = org/x", "hf-repo", "download"),
    ("ssl-key-file = /k", "ssl-key-file", "tls"),
])
def test_a_download_or_tls_key_prints_one_line_and_writes_nothing(g, xdg, monkeypatch, line, key, category):
    _ini(xdg, f"ctx-size = 8192\n{line}\napi-key = SECRET-KEY-VALUE\n")
    _recorded_machine(g)
    err = _err_lines(monkeypatch)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.ini_written is None and not (g / "overrides.yaml").exists() and _audit(g) == []
    assert err == [CONFIG_INI_REFUSED_LINE.format(dir=ws_mod.shown_path(xdg), key=key, category=category)]
    assert err[0] == (f"config.ini ({ws_mod.shown_path(xdg)}) sets {key} (a {category} option) — "
                      "not carried; a harness launch does not see it")
    assert "SECRET-KEY-VALUE" not in err[0]


def test_a_file_over_the_cap_prints_one_line_and_writes_nothing(g, xdg, monkeypatch):
    _ini(xdg, b"# " + b"x" * 65_535 + b"\n")  # 65 538 bytes
    _recorded_machine(g)
    err = _err_lines(monkeypatch)

    sensed = _sense_before_trust(g, can_ask=True, path_binary=None)

    assert sensed.ini_written is None and not (g / "overrides.yaml").exists()
    assert err == [f"config.ini ({ws_mod.shown_path(xdg)}) is over 65536 bytes — not carried"]


def test_the_directory_in_a_line_is_home_relative_and_control_bytes_are_escaped(g, tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / ".config" / "bad\x1bdir"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root))
    _ini(root / "llama.cpp", "hf-repo = org/x\n")
    _recorded_machine(g)
    err = _err_lines(monkeypatch)

    _sense_before_trust(g, can_ask=True, path_binary=None)

    (line,) = err
    assert "\x1b" not in line and "~/.config/bad\\x1bdir/llama.cpp" in line
    assert str(home) not in line


def test_an_absent_file_prints_nothing(g, xdg, monkeypatch):
    _recorded_machine(g)
    err = _err_lines(monkeypatch)

    _sense_before_trust(g, can_ask=True, path_binary=None)

    assert err == []


# ------------------------------------------------------------------ through the real start


def test_start_calls_the_no_hook_after_the_question_in_source_order():
    source = inspect.getsource(start_cmd._start_async)
    sense = source.index("_sense_before_trust(")
    decide = source.index("decide_machine_trust(cfg_path")
    hook = source.index("_after_no_on_config_ini(")
    assert sense < decide < hook


async def test_a_terminal_no_through_the_real_start_removes_the_line_and_launches_nothing(
        tmp_path, xdg, monkeypatch):
    block = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/x/m.gguf", "port": 8080}
    seen = _boot(tmp_path, monkeypatch, server_block=block)
    _ini(xdg)
    _record(tmp_path)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    asked = answers(monkeypatch, False)

    assert await _drive(tmp_path) == 1

    assert seen.started == []
    assert len([q for q in asked if "+ server.config_ini (overrides.yaml)" in q]) == 1
    assert "server" not in _overrides(tmp_path)
    assert _declined(KEPT)


async def test_a_headless_start_through_the_real_start_writes_nothing(tmp_path, xdg, monkeypatch):
    block = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/x/m.gguf", "port": 8080}
    _boot(tmp_path, monkeypatch, server_block=block)
    _ini(xdg)
    _record(tmp_path)

    await _drive(tmp_path)

    assert _overrides(tmp_path) == {} and _audit(tmp_path) == []
