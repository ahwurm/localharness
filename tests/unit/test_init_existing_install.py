"""INIT-11 / INIT-12 / INIT-09 (the existing install): a config.yaml that sets no `server:` imports the
unit or process behind its provider into overrides.yaml, asked nothing, with config.yaml byte-identical
(Phase 5, plan 14; D-36, D-37, D-40, D-54, D-60).

The reference box's 0.17 install is a provider at `http://127.0.0.1:8080/v1` and a llama-server unit
on that port. One `init` writes `server:` (and discovery's peers while the peer list is empty) through
`workspace.write_overlay_validated`, adopts exactly what it wrote when a person is at the terminal, and
prints no `Keep it?`. A process that answers no HTTP, another user's process and a provider on another
machine are not imported. Everything outside the files is faked as in test_init_live_import.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

import localharness.cli.init_cmd as init_cmd
from localharness.cli.workspace import decide_machine_trust
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import speed_stats
from localharness.provider.detector import DetectorResult
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)
from tests.unit.test_init_core_setup import (
    _answers,
    _exited,
    _flat,
    _hermetic,  # noqa: F401  (autouse fixture; real sockets are refused by conftest)
    _init,
    _terminal,
)
from tests.unit.test_init_cmd import _make_capability_result
from tests.unit.test_init_live_import import (  # noqa: F401  (fixtures and helpers)
    INODE,
    UNIT,
    detected,
    gguf,
    lines_of,
    live,
)
from tests.unit.test_init_unit_import import _journal, _nothing_answers, _systemd
from tests.unit.trust_support import never_asked
from tests.unit.units_support import fake_systemd  # noqa: F401  (fixture)

SECRET = "sk-live-0123456789secret"
CONFIG = ("version: '1'\nprovider:\n  provider_type: llamacpp\n  base_url: http://127.0.0.1:8080/v1\n"
          "  default_model: qwen-27b\n  available_models: [qwen-27b]\nextra_endpoints: []\n")


@pytest.fixture(autouse=True)
def _a_person_is_here(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)


def seed(live, text: str = CONFIG) -> Path:  # noqa: F811
    live.cfg.mkdir(parents=True, exist_ok=True)
    path = live.cfg / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overrides(live) -> dict:  # noqa: F811
    return yaml.safe_load((live.cfg / "overrides.yaml").read_text(encoding="utf-8"))


def audit_lines(live) -> list[dict]:  # noqa: F811
    path = live.cfg / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def box(live, *serve: str, **kw) -> Path:  # noqa: F811
    """The reference box: the 0.17 config, two more GGUFs beside A, a unit serving A on :8080."""
    gguf(live.home / "models" / "gguf" / "C.gguf")
    live.serve(*serve, **kw)
    return seed(live)


def block_of(result) -> list[str]:
    """The shown block: from `serving ...` up to the line before `Updated overrides.yaml`."""
    lines = lines_of(result)
    start = next(i for i, line in enumerate(lines) if line.startswith("serving "))
    end = next(i for i, line in enumerate(lines) if line.startswith("✓ Updated "))
    return [line for line in lines[start:end] if line]


def never_imports(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("an existing-install import ran where today's flow keeps the config")

    monkeypatch.setattr(init_cmd, "_existing_install_import", _boom)


# ------------------------------------------------------------------ a running unit on the provider's port


def test_a_terminal_init_imports_the_running_unit_into_overrides_asking_nothing(live):  # noqa: F811
    config = box(live, "-c", "65536", "-ngl", "99")
    before = digest(config)
    _terminal(live.monkeypatch)

    result = _init(live.cfg)  # Confirm.ask raises: nothing may be asked

    _exited(result, 0)
    assert digest(config) == before
    assert "Keep it?" not in result.output
    data = overrides(live)
    server = data["server"]
    assert server["runtime"] == "llamacpp" and server["model"] == str(live.model_a)
    assert server["served_name"] == "qwen-27b" and server["port"] == 8080
    assert server["extra_args"] == ["-ngl", "99"] and None not in server.values()
    assert sorted(peer["name"] for peer in data["extra_endpoints"]) == ["B", "C"]
    assert set(data) == {"server", "extra_endpoints"}  # the provider block stays in config.yaml
    assert block_of(result)[0].startswith("serving qwen-27b (llama.cpp) ← ~/models/gguf/A.gguf")
    assert f"✓ Updated {live.cfg / 'overrides.yaml'} — your config.yaml is unchanged." in _flat(result)
    loaded = ConfigLoader(config_dir=live.cfg).load_harness()
    assert loaded.server.port == 8080 and sorted(p.name for p in loaded.extra_endpoints) == ["B", "C"]
    assert loaded.provider.default_model == "qwen-27b"


def test_the_audit_log_names_the_entries_written_as_the_cli(live):  # noqa: F811
    box(live, "-c", "65536", "-ngl", "99")

    _exited(_init(live.cfg, "--no-input"), 0)

    lines = audit_lines(live)
    assert lines and all(line["actor"] == "cli" and line["layer"] == "user" for line in lines)
    paths = {line["path"] for line in lines}
    assert {"server.model", "server.runtime", "server.extra_args",
            "extra_endpoints.B.lifecycle.model"} <= paths
    assert all(line["actor_detail"].startswith("localharness init: imported ") for line in lines)


def test_the_import_is_one_overlay_write_with_one_audit_batch(live, monkeypatch):  # noqa: F811
    from localharness.config import overlay

    box(live, "-c", "65536", "-ngl", "99")
    written: list[str] = []
    real = overlay.atomic_write_overlay
    monkeypatch.setattr(overlay, "atomic_write_overlay", lambda path, data: (written.append(Path(path).name), real(path, data))[1])

    _exited(_init(live.cfg, "--no-input"), 0)

    assert written == ["overrides.yaml"]
    paths = [line["path"] for line in audit_lines(live)]
    assert "server.model" in paths and len(paths) == len(set(paths))


def test_the_ledger_holds_the_weights_path_after_the_overlay_write(live):  # noqa: F811
    box(live, "-c", "65536")

    _exited(_init(live.cfg, "--no-input"), 0)

    assert speed_stats.model_path(live.ledger(), "llamacpp", "qwen-27b") == str(live.model_a)


def test_an_inline_key_is_written_as_provider_api_key_and_appears_nowhere_else(live):  # noqa: F811
    config = box(live, "--api-key", SECRET, "--host", "0.0.0.0", "-c", "65536")  # noqa: S104
    live.detect.return_value = detected(models=(), raw={"error": {"code": 401}})
    before = digest(config)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert digest(config) == before
    data = overrides(live)
    assert data["provider"] == {"api_key": SECRET}
    assert data["server"]["bind_all"] is True and data["server"]["require_api_key"] is True
    assert SECRET not in result.output
    assert SECRET not in (live.cfg / "audit.jsonl").read_text(encoding="utf-8")
    assert ("provider.api_key", "(set)") in {(e["path"], e["after_value"]) for e in audit_lines(live)}
    for path in live.cfg.rglob("*"):
        if path.is_file() and path.name != "overrides.yaml":
            assert SECRET not in path.read_text(encoding="utf-8", errors="replace"), path


# ------------------------------------------------------------------ who is not imported


def test_a_listener_that_answers_no_http_is_not_imported_and_keep_it_is_asked(live):  # noqa: F811
    config = box(live, "-c", "65536")
    live.detect.return_value = DetectorResult(found=False, provider_type="unknown", base_url="", models=[],
                                              suggested_model="", probe_duration_ms=0.0)
    _terminal(live.monkeypatch)
    confirm = _answers(live.monkeypatch, True)

    result = _init(live.cfg)

    _exited(result, 0)
    assert [c.args[0] for c in confirm.ask.call_args_list] == [
        "Config exists: qwen-27b at http://127.0.0.1:8080/v1. Keep it?"]
    assert not (live.cfg / "overrides.yaml").exists() and config.read_text(encoding="utf-8") == CONFIG


def test_another_users_process_on_the_port_is_not_imported(live, monkeypatch):  # noqa: F811
    config = seed(live)
    live.fake_proc.listen(8080, INODE, os.getuid() + 1)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert f"Kept {config}." in _flat(result) and not (live.cfg / "overrides.yaml").exists()


def test_a_provider_on_another_machine_is_not_imported(live):  # noqa: F811
    config = box(live, "-c", "65536")
    config.write_text(CONFIG.replace("127.0.0.1", "gpu-box"), encoding="utf-8")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert f"Kept {config}." in _flat(result) and not (live.cfg / "overrides.yaml").exists()


def test_a_config_that_already_sets_server_keeps_todays_flow(live, monkeypatch):  # noqa: F811
    config = box(live, "-c", "65536")
    text = CONFIG + "server:\n  runtime: llamacpp\n  model: /m/x.gguf\n  port: 8080\n"
    config.write_text(text, encoding="utf-8")
    never_imports_listener = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("imported"))  # noqa: E731
    monkeypatch.setattr(init_cmd, "_import_listener", never_imports_listener)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert f"Kept {config}." in _flat(result) and not (live.cfg / "overrides.yaml").exists()
    assert config.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("flags", [["--endpoint", "http://x:1/v1", "--model", "m"], ["--model", "m"]],
                         ids=["endpoint", "model"])
def test_an_explicit_endpoint_or_model_is_a_change_not_an_import(flags, live, monkeypatch):  # noqa: F811
    config = box(live, "-c", "65536")
    before = digest(config)
    never_imports(monkeypatch)
    monkeypatch.setattr(init_cmd, "_probe_capabilities", lambda result, model: _make_capability_result())
    monkeypatch.setattr(init_cmd, "_list_endpoint_models", lambda *_: ["m"])
    live.detect.return_value = detected(models=("m",))

    result = _init(live.cfg, "--no-input", *flags)

    _exited(result, 0)  # an import here would have raised from never_imports
    assert digest(config) == before
    assert "server" not in overrides(live) and overrides(live)["provider"]["default_model"] == "m"


def test_force_starts_over_instead_of_importing_into_overrides(live, monkeypatch):  # noqa: F811
    box(live, "-c", "65536")
    never_imports(monkeypatch)

    result = _init(live.cfg, "--no-input", "--force")

    _exited(result, 0)
    assert live.written()["server"]["port"] == 8080  # the fresh path's own live import
    assert set(overrides(live)) == {"extra_endpoints"}  # only the peers it found; no saved server or provider


# ------------------------------------------------------------------ attach-only


def test_an_attach_only_process_prints_its_one_line_and_keeps_the_config(live):  # noqa: F811
    config = box(live, "--host", "ip6-localhost", "-c", "65536")
    before = digest(config)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert digest(config) == before and not (live.cfg / "overrides.yaml").exists()
    assert any("attached only, no server record written" in line for line in lines_of(result))
    assert f"Kept {config}." in _flat(result)


# ------------------------------------------------------------------ the peers


def test_a_peer_list_in_config_yaml_is_copied_once_and_the_import_peers_follow_it(live):  # noqa: F811
    config = box(live, "-c", "65536")
    config.write_text(CONFIG.replace(
        "extra_endpoints: []", "extra_endpoints:\n  - name: mine\n    base_url: http://127.0.0.1:9000/v1\n"
        "    provider_type: llamacpp\n"), encoding="utf-8")
    before = digest(config)

    _exited(_init(live.cfg, "--no-input"), 0)

    data = overrides(live)
    assert set(data) == {"server", "extra_endpoints"} and digest(config) == before
    assert [p["name"] for p in data["extra_endpoints"]] == ["mine", "C", "B"]
    loader = ConfigLoader(config_dir=live.cfg)
    assert [p.name for p in loader.load_harness().extra_endpoints] == ["mine", "C", "B"]


def test_peers_gained_between_the_read_and_the_write_are_kept_and_the_import_peers_follow(live, monkeypatch):  # noqa: F811
    box(live, "-c", "65536")
    shown = init_cmd.show_records

    def show_then_a_peer_appears(recs):  # the block is printed after the read and before the write
        shown(recs)
        (live.cfg / "overrides.yaml").write_text(yaml.safe_dump({"extra_endpoints": [{
            "name": "late", "base_url": "http://127.0.0.1:9100/v1", "provider_type": "llamacpp"}]}),
            encoding="utf-8")

    monkeypatch.setattr(init_cmd, "show_records", show_then_a_peer_appears)

    _exited(_init(live.cfg, "--no-input"), 0)

    data = overrides(live)
    assert [peer["name"] for peer in data["extra_endpoints"]] == ["late", "C", "B"]
    assert data["server"]["port"] == 8080


def test_an_overrides_peer_list_already_there_is_kept_and_the_import_peers_follow(live):  # noqa: F811
    box(live, "-c", "65536")
    (live.cfg / "overrides.yaml").write_text(yaml.safe_dump({"extra_endpoints": [{
        "name": "mine", "base_url": "http://127.0.0.1:9100/v1", "provider_type": "llamacpp"}]}),
        encoding="utf-8")

    _exited(_init(live.cfg, "--no-input"), 0)

    assert [p["name"] for p in overrides(live)["extra_endpoints"]] == ["mine", "C", "B"]


# ------------------------------------------------------------------ nothing listens: a stopped unit


def _stopped(live) -> Path:  # noqa: F811
    _nothing_answers(live)
    _systemd(live, ("llm.service", "disabled", str(live.home / ".config/systemd/user/llm.service"), ""))
    _journal(live, extra=" -c 65536 -ngl 99", host="127.0.0.1")
    return seed(live)


def test_nothing_listening_and_a_stopped_unit_that_names_a_model_is_imported(live):  # noqa: F811
    config = _stopped(live)
    before = digest(config)
    _terminal(live.monkeypatch)

    result = _init(live.cfg)

    _exited(result, 0)
    assert digest(config) == before
    data = overrides(live)
    assert data["server"]["model"] == str(live.model_a) and data["server"]["served_name"] == "qwen-27b"
    assert [p["name"] for p in data["extra_endpoints"]] == ["B"]
    assert "Keep it?" not in result.output
    assert any(line.startswith("llm.service sets -c 65536;") for line in block_of(result))


def test_nothing_listening_and_no_unit_keeps_todays_flow(live):  # noqa: F811
    _nothing_answers(live)
    config = seed(live)
    _terminal(live.monkeypatch)
    confirm = _answers(live.monkeypatch, True)

    result = _init(live.cfg)

    _exited(result, 0)
    assert confirm.ask.call_count == 1 and not (live.cfg / "overrides.yaml").exists()
    assert config.read_text(encoding="utf-8") == CONFIG


def _todays_flow_keeps(live, config: Path) -> None:  # noqa: F811
    _terminal(live.monkeypatch)
    confirm = _answers(live.monkeypatch, True)

    result = _init(live.cfg)

    _exited(result, 0)
    assert confirm.ask.call_count == 1 and not (live.cfg / "overrides.yaml").exists()
    assert config.read_text(encoding="utf-8") == CONFIG or "ollama" in config.read_text(encoding="utf-8")


def test_a_stopped_unit_on_another_port_is_not_the_providers_server(live):  # noqa: F811
    """WR-05: an embedding server on :8090 is not what the provider at :8080 points at."""
    _nothing_answers(live)
    _systemd(live, ("embed.service", "disabled", str(live.home / ".config/systemd/user/embed.service"), ""))
    live.fake_systemd.journal([{"_EXE": str(live.bin), "_CMDLINE": f"llama-server -m {live.model_a} "
                                "--alias embed --port 8090"}], unit="embed.service")
    _todays_flow_keeps(live, seed(live))


def test_a_stopped_unit_is_not_imported_behind_a_provider_that_is_not_llama_cpp(live):  # noqa: F811
    _nothing_answers(live)
    _systemd(live, ("llm.service", "disabled", str(live.home / ".config/systemd/user/llm.service"), ""))
    _journal(live, extra=" -c 65536", host="127.0.0.1")
    _todays_flow_keeps(live, seed(live, CONFIG.replace("provider_type: llamacpp", "provider_type: ollama")))


# ------------------------------------------------------------------ adoption, and the quiet next starts


def test_an_interactive_import_adopts_overrides_yaml_and_the_next_two_starts_ask_nothing(live, monkeypatch):  # noqa: F811
    box(live, "-c", "65536", "-ngl", "99")
    _terminal(live.monkeypatch)

    _exited(_init(live.cfg), 0)

    recorded = trust.recorded_machine(live.cfg)
    assert recorded is not None
    launch = [e for e in recorded["entries"] if e["kind"] in (trust.LAUNCH_KIND, trust.LAUNCH_MODEL_KIND)]
    assert launch and {e["file"] for e in launch} == {"overrides.yaml"}
    assert {"server.model", "server.binary", "extra_endpoints.B.lifecycle.model"} <= {e["name"] for e in launch}
    never_asked(monkeypatch)
    for _ in range(2):
        decided = decide_machine_trust(live.cfg, ask=True, launching=True)
        assert decided.withheld == {} and decided.quiet == frozenset() and decided.line == ""
    loader = ConfigLoader(config_dir=live.cfg)
    loader.load_harness()
    assert not any("extra_endpoints replaces" in w for w in loader.harness_warnings)
    assert yaml.safe_load((live.cfg / "config.yaml").read_text(encoding="utf-8"))["extra_endpoints"] == []


@pytest.mark.parametrize("how", ["no-input", "shell-tool"])
def test_nothing_is_adopted_without_a_person(how, live, monkeypatch):  # noqa: F811
    box(live, "-c", "65536")
    if how == "shell-tool":
        _terminal(live.monkeypatch)
        monkeypatch.setenv("LOCALHARNESS_COMMIT", "abc")

    _exited(_init(live.cfg, *(["--no-input"] if how == "no-input" else [])), 0)

    assert trust.recorded_machine(live.cfg) is None
    assert overrides(live)["server"]["port"] == 8080


# ------------------------------------------------------------------ a write that cannot be made


def test_an_overlay_write_that_fails_keeps_the_config_and_says_nothing_was_written(live, monkeypatch):  # noqa: F811
    config = box(live, "-c", "65536")
    monkeypatch.setattr(init_cmd.workspace, "write_overlay_validated", lambda *a, **k: False)
    _terminal(live.monkeypatch)

    result = _init(live.cfg)

    _exited(result, 1)  # docs/specs/10-cli.md: a change that does not validate writes nothing and exits 1
    assert "Nothing was written" in _flat(result)
    assert trust.recorded_machine(live.cfg) is None and not (live.cfg / "overrides.yaml").exists()
    assert config.read_text(encoding="utf-8") == CONFIG
    assert speed_stats.model_path(live.ledger(), "llamacpp", "qwen-27b") is None


def test_the_carrier_of_an_inline_key_never_prints_it():
    """IN-04: a traceback, a debug log or a repr of the InitRecords must not show the key."""
    from pydantic import SecretStr

    recs = init_cmd.InitRecords(
        server=init_cmd.ManagedServerConfig(runtime="llamacpp", model="/m.gguf"), served="m",
        api_key=SecretStr(SECRET))
    assert SECRET not in repr(recs) and SECRET not in str(recs)
    assert recs.api_key.get_secret_value() == SECRET
    assert SECRET not in repr(init_cmd.InitRecords(
        server=recs.server, served="m", api_key=SECRET)), "a key handed over as text is wrapped too"
