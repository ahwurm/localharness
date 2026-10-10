"""`start` senses llama-server's own `config.ini` before the machine question (Phase 5, D-31, D-56, D-57).

The real `${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini` is masked by the one function `init` also
uses, written to overrides.yaml as `server.config_ini` through `workspace.write_overlay_validated` (never
by a writer of start's own), shown by the machine question as an ordinary uncovered launch entry, removed
and remembered on a No, and withheld on a headless start. Nothing here reads the owner's real file: every
test points `XDG_CONFIG_HOME` at its own temp directory, and nothing dials a port.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import yaml

from localharness.cli import start_cmd
from localharness.cli import workspace as ws_mod
from localharness.cli.start_cmd import CONFIG_INI_WRITTEN_LINE, _sense_before_trust
from localharness.cli.workspace import MachineTrust, decide_machine_trust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import server
from tests.unit.trust_support import answers, by_name, g, never_asked, write_yaml  # noqa: F401

PROVIDER = {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8081/v1", "default_model": "m"}
LLAMACPP = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf", "port": 8123}
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
