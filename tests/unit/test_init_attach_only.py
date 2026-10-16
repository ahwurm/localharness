"""INIT-10: every import exit is an attach: `provider:` only, no `server:`, nothing relaunched,
nothing adopted, exit 0, and one line `<who> <reason> — attached only, no server record written`
(Phase 5, plan 12, D-25, D-26, D-53, D-54).

Also the listeners the import leaves to today's attach path (another uid answering 200, a user-owned
process that is not llama-server), a keyed listener of another uid (`set provider.api_key`), and a
keyed Ollama (`cannot carry a key`). The box is `test_init_live_import.live`: a `FakeProc` for
`/proc`, `FakeSystemd` for systemd, detection an `AsyncMock`; the capability probe, `Popen` and every
question raise if touched, except where a test says the probe runs.
"""
# ruff: noqa: F811  (fixtures are imported by name and requested by argument)
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import localharness.cli.init_cmd as init_cmd
from localharness.config import trust
from localharness.provider import model_library, procfs
from tests.unit.interview_support import scripted
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)
from tests.unit.test_init_cmd import _make_capability_result
from tests.unit.test_init_core_setup import (
    _exited,
    _hermetic,  # noqa: F401  (autouse fixture; real sockets are refused by conftest)
    _init,
    _terminal,
)
from tests.unit.test_init_live_import import (  # noqa: F401  (fixture and helpers)
    INODE,
    KEYED_401,
    PID,
    UNIT,
    detected,
    lines_of,
    live,
)
from tests.unit.units_support import fake_systemd  # noqa: F401  (fixture)

SUFFIX = "attached only, no server record written"


def attach_lines(result) -> list[str]:
    return [line for line in lines_of(result) if SUFFIX in line]


def assert_provider_only(live, default_model: str) -> dict:
    cfg = live.written()
    assert cfg.get("server") is None and cfg.get("extra_endpoints") == []
    provider = cfg["provider"]
    assert provider["base_url"] == "http://127.0.0.1:8080/v1"
    assert provider["default_model"] == default_model
    assert provider["available_models"] == ([default_model] if default_model else [])
    return cfg


# ------------------------------------------------------------------ every import exit


@pytest.mark.parametrize("argv, reason, model", [
    (("--host", "ip6-localhost"), "binds ip6-localhost (a host name)", "qwen-27b"),
    (("--host", "/run/llama.sock"), "binds /run/llama.sock (a UNIX socket)", "qwen-27b"),
    (("--ssl-key-file", "/k"), "serves TLS (--ssl-key-file)", "qwen-27b"),
    (("--api-key", "A,B"), "holds more than one --api-key", "qwen-27b"),
], ids=["host-name", "unix-socket", "tls", "two-keys"])
def test_an_import_exit_attaches_with_the_provider_only_and_one_line(live, argv, reason, model):
    live.serve(*argv)
    _terminal(live.monkeypatch)  # a person is here: still nothing is adopted

    result = _init(live.cfg)

    _exited(result, 0)
    assert attach_lines(result) == [f"{UNIT} {reason} — {SUFFIX}"]
    assert_provider_only(live, model)
    assert trust.recorded_machine(live.cfg) is None


def test_the_attach_names_the_model_from_the_alias_else_the_model_stem_else_the_detectors(live):
    live.serve("--ssl-key-file", "/k", base=("llama-server", "-m", str(live.model_a), "--port", "8080"))
    result = _init(live.cfg, "--no-input")
    _exited(result, 0)
    assert_provider_only(live, "A")  # the -m stem, the preset name rule

    (live.cfg / "config.yaml").unlink()
    live.serve(base=("llama-server", "-hf", "org/x", "-a", "hfname"))
    result = _init(live.cfg, "--no-input")
    _exited(result, 0)
    assert attach_lines(result) == [
        f"{UNIT} a record needs a local path; none mapped under ~/.cache/huggingface/hub — {SUFFIX}"]
    assert_provider_only(live, "hfname")

    (live.cfg / "config.yaml").unlink()
    live.serve(base=("llama-server", "-hf", "org/x"))
    _exited(_init(live.cfg, "--no-input"), 0)
    assert_provider_only(live, "qwen-27b")  # the detector's suggestion


def test_a_process_in_no_unit_is_named_by_its_pid_and_command(live):
    live.serve("--ssl-key-file", "/k", unit=None)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [f"pid {PID} (llama-server) serves TLS (--ssl-key-file) — {SUFFIX}"]


def test_a_command_line_over_the_cap_attaches(live, monkeypatch):
    monkeypatch.setattr(procfs, "CMDLINE_CAP", 64)
    live.serve("-ngl", "99", "--no-warmup", "--flash-attn", "on")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [f"{UNIT} has its command line over 64 bytes — {SUFFIX}"]
    assert_provider_only(live, "qwen-27b")


def test_an_environment_over_the_cap_attaches(live, monkeypatch):
    monkeypatch.setattr(procfs, "ENVIRON_CAP", 32)
    live.serve(environ={"LLAMA_ARG_THREADS": "8", "X": "y" * 100})

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [f"{UNIT} has its environment over 32 bytes — {SUFFIX}"]
    assert_provider_only(live, "qwen-27b")


def test_an_unreadable_environment_attaches(live, monkeypatch):
    live.serve()
    real = procfs._read_at

    def _read(name, dir_fd, cap):
        if name == "environ":
            raise PermissionError(name)
        return real(name, dir_fd, cap)

    monkeypatch.setattr(procfs, "_read_at", _read)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [f"{UNIT} has an unreadable environment — {SUFFIX}"]
    assert_provider_only(live, "qwen-27b")


def test_a_pid_that_is_gone_attaches_by_its_pid(live, monkeypatch):
    live.serve(unit=None)
    monkeypatch.setattr(procfs, "read_process", lambda pid: procfs.ProcRefusal("is gone"))

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [f"pid {PID} is gone — {SUFFIX}"]
    assert_provider_only(live, "qwen-27b")


def test_nothing_is_relaunched_or_asked_by_an_attach_exit(live):
    live.serve("--ssl-key-file", "/k")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)  # Popen, Confirm, Prompt and _probe_capabilities all raise if touched
    assert "serving " not in result.output


# ------------------------------------------------------------------ a keyed listener of another uid


def _foreign_uid(live, monkeypatch, *, name: str | None = "root", uid: int = 0) -> None:
    live.fake_proc.listen(8080, INODE, uid)
    live.detect.return_value = detected(models=(), raw=KEYED_401)
    import pwd

    if name is None:
        def _missing(_uid):
            raise KeyError(_uid)
        monkeypatch.setattr(pwd, "getpwuid", _missing)
    else:
        monkeypatch.setattr(pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_name=name))


def test_a_keyed_listener_of_another_uid_attaches_and_says_to_set_the_key(live, monkeypatch):
    _foreign_uid(live, monkeypatch)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)  # before this plan: exit 1, "No models available at detected endpoint."
    assert attach_lines(result) == [
        ":8080 is keyed and owned by uid 0 (root) — set provider.api_key; " + SUFFIX]
    assert "No models available" not in result.output
    assert_provider_only(live, "")


def test_core_setup_reports_no_server_ready_for_a_keyed_listener_of_another_uid(live, monkeypatch):
    _foreign_uid(live, monkeypatch)

    outcome = init_cmd.core_setup(str(live.cfg), endpoint=None, model=None, force=False,
                                  interactive=False)

    assert outcome.config_file == live.cfg / "config.yaml" and outcome.server_ready is False


def test_an_unresolvable_uid_reads_as_the_uid_twice(live, monkeypatch):
    _foreign_uid(live, monkeypatch, name=None, uid=4242)

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [
        ":8080 is keyed and owned by uid 4242 (4242) — set provider.api_key; " + SUFFIX]


def test_without_pwd_the_uid_alone_names_the_owner(live, monkeypatch):
    _foreign_uid(live, monkeypatch, uid=4242)
    monkeypatch.setitem(sys.modules, "pwd", None)  # `import pwd` raises ImportError

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [
        ":8080 is keyed and owned by uid 4242 (4242) — set provider.api_key; " + SUFFIX]


def test_pwd_is_not_imported_at_module_level():
    assert not hasattr(init_cmd, "pwd")  # D-71: function-local


def test_another_uid_answering_200_keeps_todays_attach_path(live, monkeypatch):
    live.fake_proc.listen(8080, INODE, 0)
    probed = []
    monkeypatch.setattr(init_cmd, "_probe_capabilities",
                        lambda result, model: probed.append(model) or _make_capability_result())

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert probed == ["qwen-27b"] and not attach_lines(result)
    assert live.written()["provider"]["default_model"] == "qwen-27b"
    assert live.written().get("server") is None


# ------------------------------------------------------------------ user-owned listeners that are not llama-server


@pytest.mark.parametrize("argv", [
    ("python3", "-m", "vllm.entrypoints.openai.api_server"),
    ("/usr/bin/vllm", "serve", "x"),
], ids=["python3", "vllm"])
def test_a_user_owned_listener_that_is_not_llama_server_keeps_todays_attach_path(live, monkeypatch, argv):
    live.serve(base=argv, unit=None, exe="/usr/bin/" + os.path.basename(argv[0]))
    probed = []
    monkeypatch.setattr(init_cmd, "_probe_capabilities",
                        lambda result, model: probed.append(model) or _make_capability_result())

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert probed == ["qwen-27b"] and not attach_lines(result)
    assert live.written().get("server") is None


def test_a_keyed_ollama_cannot_carry_a_key(live):
    live.serve(base=("ollama", "serve"), unit=None, exe="/usr/bin/ollama")
    live.detect.return_value = detected(models=(), raw=KEYED_401, provider_type="ollama")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [f"pid {PID} (ollama) runs Ollama, which cannot carry a key — {SUFFIX}"]
    assert live.written().get("server") is None


def test_a_keyed_lm_studio_cannot_carry_a_key(live):
    live.serve(base=("lms", "server", "start"), unit=None, exe="/usr/bin/lms")
    live.detect.return_value = detected(models=(), raw=KEYED_401, provider_type="lmstudio")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert attach_lines(result) == [
        f"pid {PID} (lms) runs LM Studio, which cannot carry a key — {SUFFIX}"]


def test_an_unkeyed_ollama_keeps_todays_attach_path(live, monkeypatch):
    live.serve(base=("ollama", "serve"), unit=None, exe="/usr/bin/ollama")
    probed = []
    monkeypatch.setattr(init_cmd, "_probe_capabilities",
                        lambda result, model: probed.append(model) or _make_capability_result())

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert probed and not attach_lines(result)


# ------------------------------------------------------------------ the library after a first run that attached


def _library(live) -> dict:
    return model_library.Library.load(model_library.library_path(live.cfg)).entries


def _every_file_seen(live) -> None:
    entries = _library(live)
    assert entries.keys() == {os.path.realpath(live.model_a), os.path.realpath(live.model_b)}
    assert not any(e.declined for e in entries.values())


def test_an_attached_first_run_baselines_the_library_for_a_person_and_asks_nothing(live):
    live.serve("--ssl-key-file", "/k")
    _terminal(live.monkeypatch)
    asked = scripted(live.monkeypatch)

    result = _init(live.cfg)

    _exited(result, 0)
    assert asked.asked == [] and attach_lines(result)
    _every_file_seen(live)


def test_an_attached_first_run_without_a_person_writes_no_library_file(live):
    live.serve("--ssl-key-file", "/k")

    _exited(_init(live.cfg, "--no-input"), 0)

    assert not model_library.library_path(live.cfg).exists()


def test_a_plain_detected_server_baselines_the_library_for_a_person_and_the_receipt_says_later(live, monkeypatch):
    live.fake_proc.listen(8080, INODE, 0)
    monkeypatch.setattr(init_cmd, "_probe_capabilities", lambda result, model: _make_capability_result())
    _terminal(live.monkeypatch)
    asked = scripted(live.monkeypatch)

    result = _init(live.cfg)

    _exited(result, 0)
    assert asked.asked == [] and "Later:" in result.output
    _every_file_seen(live)
