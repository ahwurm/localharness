"""INIT-12: an interactive `init` adopts exactly the launch entries it wrote and showed (Phase 5,
D-12, D-37, D-59, D-60).

What `init` writes and prints in front of a person is read and accepted, so the next start asks
nothing about it. Adoption is built from the dicts that went into config.yaml, never from a fresh
look at the disk, and only a run with a person behind it adopts: not `--no-input`, not a run
without a terminal, not a command the shell tool started (it carries LOCALHARNESS_COMMIT). An init
that wrote no record records nothing. The fake terminal is `init_cmd.sys` (P14).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import localharness.cli.init_cmd as init_cmd
from localharness.cli.workspace import decide_machine_trust
from localharness.config import trust
from tests.unit.test_init_core_setup import (
    _answers,
    _detect,
    _exited,
    _hermetic,  # noqa: F401  (autouse fixture: no real HTTP)
    _init,
    _seed,
    _terminal,
)
from tests.unit.test_init_discovery_records import (  # noqa: F401  (fixture)
    _written,
    box,
)
from tests.unit.test_init_writes_records import VLLM, guided  # noqa: F401  (fixture)
from tests.unit.trust_support import answers, never_asked

OTHER = {"file": "agents/x.yaml", "kind": "mcp_server", "name": "srv", "shown": "cmd", "env": [],
         "headers": []}


@pytest.fixture(autouse=True)
def _a_person_is_here(monkeypatch):
    """The shell tool's marker is absent unless a test sets it (P14)."""
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)


def _expected(cfg_dir: Path) -> list[dict]:
    """The entries an adoption must hold: the launch entries of the dicts config.yaml carries, plus
    the posture entries the machine's files give under those prefixes, in the record's order."""
    data = _written(cfg_dir)
    prefixes = {"server.": data["server"]}
    for peer in data["extra_endpoints"]:
        prefixes[f"extra_endpoints.{peer['name']}.lifecycle."] = peer["lifecycle"]
    entries = [e for prefix, mapping in prefixes.items()
               for e in trust._launch_entries("config.yaml", prefix, mapping)]
    posture = {prefix + key for prefix in prefixes for key in ("bind_all", "require_api_key")}
    entries += [e for e in trust.machine_snapshot(cfg_dir)
                if e["kind"] == trust.LAUNCH_KIND and e["name"] in posture]
    return sorted(entries, key=trust._entry_key)


def _names(entries: list[dict]) -> set[str]:
    return {e["name"] for e in entries}


# ------------------------------------------------------------------ an interactive init adopts


def test_an_interactive_discovery_init_records_exactly_what_it_wrote(box):
    _exited(_init(box.cfg), 0)

    rec = trust.recorded_machine(box.cfg)
    assert rec is not None and rec["kinds"] == ["launch", "launch_model"]
    assert rec["entries"] == _expected(box.cfg)
    names = _names(rec["entries"])
    assert {"server.binary", "server.model", "server.runtime", "server.bind_all",
            "server.require_api_key", "server.served_name",
            "extra_endpoints.Swift-27B-Q8.lifecycle.model", "extra_endpoints.Swift-27B-Q8.lifecycle.mmproj",
            "extra_endpoints.Swift-27B-Q8.lifecycle.bind_all"} <= names
    assert not any(n.endswith(".binary") for n in names if n.startswith("extra_endpoints."))


def test_the_next_start_asks_nothing_about_what_init_adopted(box, monkeypatch):
    _exited(_init(box.cfg), 0)

    never_asked(monkeypatch)
    decided = decide_machine_trust(box.cfg, ask=True, launching=True)

    assert decided.withheld == {} and decided.quiet == frozenset() and decided.line == ""


def test_an_existing_record_keeps_its_other_entries_and_takes_the_written_ones(box):
    stale = {"file": "config.yaml", "kind": "launch_model", "name": "server.model", "shown": "/old.gguf"}
    trust.record_machine(box.cfg, [OTHER, stale], kinds=trust.MACHINE_KINDS)

    _exited(_init(box.cfg), 0)

    rec = trust.recorded_machine(box.cfg)
    assert rec["kinds"] == sorted(trust.MACHINE_KINDS)
    assert OTHER in rec["entries"] and stale not in rec["entries"]
    written = _expected(box.cfg)
    assert all(e in rec["entries"] for e in written)
    assert rec["entries"] == sorted([OTHER, *written], key=trust._entry_key)


def test_the_guided_vllm_record_is_adopted_the_same_way(tmp_path, guided):  # noqa: F811
    _exited(_init(tmp_path), 0)

    rec = trust.recorded_machine(tmp_path)
    assert rec["kinds"] == ["launch", "launch_model"]
    assert rec["entries"] == _expected(tmp_path)
    by_name = {e["name"]: e["shown"] for e in rec["entries"]}
    assert by_name["server.binary"] == VLLM and by_name["server.model"] == guided
    assert by_name["server.runtime"] == "vllm"


def test_an_adoption_holds_no_key_and_no_audit_triple(box):
    _exited(_init(box.cfg), 0)

    text = yaml.safe_dump(trust.recorded_machine(box.cfg))
    assert "api_key" not in text.replace("require_api_key", "").replace("api_key_file", "")


# ------------------------------------------------------------------ nothing is adopted without a person


def _baseline(box) -> str:
    _exited(_init(box.cfg), 0)
    return (box.cfg / "config.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("how", ["no-input", "marker", "no-terminal"])
def test_without_a_person_the_same_config_is_written_and_nothing_is_recorded(box, monkeypatch, how):
    baseline = _baseline(box)
    for name in ("config.yaml", "config.yaml.before-init"):
        (box.cfg / name).unlink(missing_ok=True)
    flags: list[str] = []
    if how == "no-input":
        flags = ["--no-input"]
    elif how == "marker":
        monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    else:
        _terminal(monkeypatch, False)
    trust_file = trust.trust_store_path()
    trust_file.unlink()  # the adopting run above made it; this run starts from a machine with no record

    _exited(_init(box.cfg, *flags), 0)

    assert (box.cfg / "config.yaml").read_text(encoding="utf-8") == baseline
    assert trust.recorded_machine(box.cfg) is None


@pytest.mark.parametrize("how", ["no-input", "marker", "no-terminal"])
def test_without_a_person_the_next_terminal_start_still_asks_once(box, monkeypatch, how):
    trust.record_machine(box.cfg, [OTHER], kinds=trust.MACHINE_KINDS)
    if how == "no-input":
        flags = ["--no-input"]
    else:
        flags = []
        if how == "marker":
            monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
        else:
            _terminal(monkeypatch, False)

    _exited(_init(box.cfg, *flags), 0)

    assert trust.recorded_machine(box.cfg)["entries"] == [OTHER]
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    asked = answers(monkeypatch, True)
    decide_machine_trust(box.cfg, ask=True, launching=True)
    assert len(asked) == 1


def test_adopts_is_interactive_and_no_marker(monkeypatch):
    assert init_cmd._adopts(True) is True
    assert init_cmd._adopts(False) is False
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    assert init_cmd._adopts(True) is False
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "")
    assert init_cmd._adopts(True) is False  # present at all: the shell tool started this


# ------------------------------------------------------------------ an init that wrote no record


def test_attaching_to_a_running_server_records_nothing(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _terminal(monkeypatch)
    _answers(monkeypatch)

    _exited(_init(tmp_path), 0)

    assert trust.recorded_machine(tmp_path) is None and not trust.trust_store_path().exists()


def test_a_kept_config_records_nothing(tmp_path, monkeypatch):
    _seed(tmp_path)
    _detect(monkeypatch)
    _terminal(monkeypatch)
    _answers(monkeypatch, True)

    _exited(_init(tmp_path), 0)

    assert trust.recorded_machine(tmp_path) is None and not trust.trust_store_path().exists()


def test_declining_to_save_anything_records_nothing(tmp_path, monkeypatch):
    _detect(monkeypatch, found=False)
    _terminal(monkeypatch)
    _answers(monkeypatch, False)  # the guided setup is declined, and the skip asks for no address
    monkeypatch.setattr(init_cmd, "Prompt", type("P", (), {"ask": staticmethod(lambda *a, **k: "")}))

    init_cmd.core_setup(str(tmp_path), endpoint=None, model=None, force=False, interactive=True)

    assert trust.recorded_machine(tmp_path) is None and not trust.trust_store_path().exists()


# ------------------------------------------------------------------ adoption is exact


def test_adoption_is_built_from_the_dicts_given_not_from_the_disk(tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "server": {"runtime": "llamacpp", "model": "/planted.gguf", "binary": "/planted/llama-server",
                   "bind_all": True},
        "extra_endpoints": [{"name": "other", "base_url": "http://127.0.0.1:9/v1",
                             "lifecycle": {"runtime": "llamacpp", "model": "/o.gguf", "bind_all": True}}],
    }), encoding="utf-8")
    written = {"runtime": "llamacpp", "model": "/real.gguf", "binary": "/real/llama-server",
               "bind_all": False, "require_api_key": False}

    init_cmd._adopt_written(tmp_path, "config.yaml", written, [])

    entries = trust.recorded_machine(tmp_path)["entries"]
    by_name = {e["name"]: e["shown"] for e in entries}
    assert by_name["server.binary"] == "/real/llama-server" and by_name["server.model"] == "/real.gguf"
    assert not any(name.startswith("extra_endpoints.other.") for name in by_name)  # not written by this act
    assert "server.bind_all" not in by_name  # the disk says true where false was written: left to ask
    assert "/planted" not in yaml.safe_dump(entries)
