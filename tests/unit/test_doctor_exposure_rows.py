"""Doctor names every way the harness's own surfaces are reachable from beyond this terminal, each
with the one setting that closes it: a launched model server reachable from other machines
(`server.bind_all`), a launched server that does not require the API key you set
(`server.require_api_key`), and a remote channel that may switch a session to unattended or answer
"always" (`channels.remote_unattended`). Info rows, never failures; a key is never printed.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
import yaml

from localharness.cli import doctor_cmd
from tests.unit.launch_support import record_opens
from tests.unit.test_doctor_layer_report import _UNREACHABLE_BASE_URL, _layout, _run_doctor

VLLM = {"runtime": "vllm", "launch": "docker", "docker_image": "img", "model": "test-model", "port": 8081}
NETWORK_ROW = ("i Model server: reachable from other machines on port 8081 (server.bind_all: true) — "
               "set server.bind_all: false to keep it on this machine")
KEY_ROW = ("i Model server: launched without requiring your API key (provider.api_key is set) — "
           "set server.require_api_key: true to make it refuse requests without the key")
LLAMACPP = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/x/m.gguf", "port": 8080}
EVERY_INTERFACE = " — every interface, no key"
LLAMA_NETWORK_ROW = ("i Model server: reachable from other machines on port 8080 (server.bind_all: true) — "
                     "set server.bind_all: false to keep it on this machine")
REMOTE_ROW = ('i Remote channels on (mobile): a paired phone or an allowlisted chat account can switch a '
              'session to unattended and answer "always" — set channels.remote_unattended: false to '
              'keep both to this terminal')
SECRET = "sk-DOCTOR-0001"


def _doctor(tmp_path, monkeypatch, fake_home, *, key: str = "none", **sections) -> str:
    config = {"version": "1",
              "provider": {"provider_type": "vllm", "base_url": _UNREACHABLE_BASE_URL, "api_key": key,
                           "default_model": "test-model", "available_models": ["test-model"]},
              **sections}
    _layout(tmp_path, monkeypatch, fake_home, workspace=False,
            global_config=yaml.safe_dump(config, sort_keys=False))
    out = _run_doctor()
    assert SECRET not in out
    return "\n".join(" ".join(line.split()) for line in out.splitlines())


def _peer(**lifecycle) -> dict:
    return {"name": "peer-1", "base_url": "http://127.0.0.1:8001/v1", "provider_type": "vllm", "gpu": True,
            "api_key": SECRET, "lifecycle": {**VLLM, "port": 8001, **lifecycle}}


# ------------------------------------------------------------------------------- D6: the network


def test_a_server_reachable_from_other_machines_is_named_with_the_setting(tmp_path, monkeypatch, fake_home):
    assert NETWORK_ROW in _doctor(tmp_path, monkeypatch, fake_home, server={**VLLM, "bind_all": True})


@pytest.mark.parametrize("sections", [{"server": {**VLLM, "bind_all": False}}, {}],
                         ids=["loopback", "no server"])
def test_a_server_on_this_machine_or_none_has_no_network_row(tmp_path, monkeypatch, fake_home, sections):
    assert "reachable from other machines" not in _doctor(tmp_path, monkeypatch, fake_home, **sections)


def test_a_launched_peer_reachable_from_other_machines_gets_its_own_row(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, extra_endpoints=[_peer(bind_all=True)])
    assert ("i extra_endpoints[0] (peer-1): reachable from other machines on port 8001 "
            "(extra_endpoints[0].lifecycle.bind_all: true) — set extra_endpoints[0].lifecycle.bind_all: "
            "false to keep it on this machine") in out


# ------------------------------------------------------------------------------- R11: the key


def test_a_keyless_launch_with_a_key_set_is_named(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, key=SECRET, server={**VLLM, "bind_all": False})
    assert KEY_ROW in out


@pytest.mark.parametrize("key, server", [(SECRET, {**VLLM, "require_api_key": True}), ("none", dict(VLLM))],
                         ids=["required", "no key"])
def test_no_key_row_when_the_key_is_required_or_there_is_none(tmp_path, monkeypatch, fake_home, key, server):
    assert "launched without requiring" not in _doctor(tmp_path, monkeypatch, fake_home, key=key, server=server)


def test_a_launched_peer_with_a_key_and_no_requirement_gets_its_own_row(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, extra_endpoints=[_peer()])
    assert ("i extra_endpoints[0] (peer-1): launched without requiring its API key "
            "(extra_endpoints[0].api_key is set) — set extra_endpoints[0].lifecycle.require_api_key: true "
            "to make it refuse requests without the key") in out


# ------------------------------------------------------------------- D-12, D-21: llama.cpp rows


def test_an_unkeyed_llama_cpp_record_on_every_interface_says_so(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, server={**LLAMACPP, "bind_all": True})
    assert LLAMA_NETWORK_ROW + EVERY_INTERFACE in out


def test_a_loopback_llama_cpp_record_has_no_network_row(tmp_path, monkeypatch, fake_home):
    assert "reachable from other machines" not in _doctor(
        tmp_path, monkeypatch, fake_home, server={**LLAMACPP, "bind_all": False})


def test_the_unkeyed_vllm_row_keeps_its_old_text_as_a_prefix_and_gains_the_suffix(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, server={**VLLM, "bind_all": True})
    assert NETWORK_ROW in out
    assert NETWORK_ROW + EVERY_INTERFACE in out


def test_a_keyed_lan_llama_cpp_record_never_reads_as_loopback_nor_as_unkeyed(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, key=SECRET,
                  server={**LLAMACPP, "bind_all": True, "require_api_key": True})
    assert LLAMA_NETWORK_ROW in out
    assert "every interface, no key" not in out


def test_a_keyed_lan_vllm_record_keeps_the_old_row_without_the_suffix(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, key=SECRET,
                  server={**VLLM, "bind_all": True, "require_api_key": True})
    assert NETWORK_ROW in out
    assert "every interface, no key" not in out


def test_require_api_key_with_no_key_held_still_reads_every_interface_no_key(tmp_path, monkeypatch, fake_home):
    out = _doctor(tmp_path, monkeypatch, fake_home, key="none",
                  server={**LLAMACPP, "bind_all": True, "require_api_key": True})
    assert LLAMA_NETWORK_ROW + EVERY_INTERFACE in out


def test_a_regular_key_file_counts_as_keyed_and_is_never_opened(tmp_path, monkeypatch, fake_home):
    key_file = tmp_path / "api.key"
    key_file.write_text("file-key-DO-NOT-OPEN", encoding="utf-8")
    with record_opens() as opened:
        out = _doctor(tmp_path, monkeypatch, fake_home, key="none",
                      server={**LLAMACPP, "bind_all": True, "require_api_key": True,
                              "api_key_file": str(key_file)})
    assert LLAMA_NETWORK_ROW in out
    assert "every interface, no key" not in out
    assert "file-key-DO-NOT-OPEN" not in out
    assert str(key_file) not in opened


def test_a_fifo_key_file_never_blocks_doctor_and_narrows_the_launch(tmp_path, monkeypatch, fake_home):
    fifo = tmp_path / "api.fifo"
    os.mkfifo(fifo)
    result: dict = {}

    def run() -> None:
        try:
            with record_opens() as opened:
                result["out"] = _doctor(tmp_path, monkeypatch, fake_home, key="none",
                                        server={**LLAMACPP, "bind_all": True, "require_api_key": True,
                                                "api_key_file": str(fifo)})
            result["opened"] = list(opened)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "doctor blocked on the key file"
    assert "error" not in result, result.get("error")
    assert "reachable from other machines" not in result["out"]
    assert str(fifo) not in result["opened"]


def test_a_declared_only_llama_cpp_record_prints_no_posture_row_and_doctor_completes(
        tmp_path, monkeypatch, fake_home):
    declared = {k: v for k, v in LLAMACPP.items() if k != "binary"}
    out = _doctor(tmp_path, monkeypatch, fake_home, server={**declared, "bind_all": True})
    assert "reachable from other machines" not in out
    assert "launched without requiring" not in out
    assert "issue(s) found" in out  # the run reached its summary: doctor completed normally


@pytest.mark.parametrize("server", [{"runtime": "ollama", "model": "m", "bind_all": True, "port": 11434},
                                    {"runtime": "lmstudio", "model": "m", "bind_all": True, "port": 1234}],
                         ids=["ollama", "lmstudio"])
def test_ollama_and_lm_studio_print_no_posture_row(tmp_path, monkeypatch, fake_home, server):
    out = _doctor(tmp_path, monkeypatch, fake_home, server=server)
    assert "reachable from other machines" not in out


def test_a_llama_cpp_peer_on_every_interface_gets_its_own_row_with_the_suffix(tmp_path, monkeypatch, fake_home):
    peer = {"name": "peer-1", "base_url": "http://127.0.0.1:8001/v1", "provider_type": "llamacpp", "gpu": True,
            "api_key": "none",
            "lifecycle": {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/x/m.gguf",
                          "port": 8001, "bind_all": True}}
    out = _doctor(tmp_path, monkeypatch, fake_home, extra_endpoints=[peer])
    assert ("i extra_endpoints[0] (peer-1): reachable from other machines on port 8001 "
            "(extra_endpoints[0].lifecycle.bind_all: true) — set extra_endpoints[0].lifecycle.bind_all: "
            "false to keep it on this machine" + EVERY_INTERFACE) in out


def test_the_key_file_check_is_a_stat_and_never_an_open(tmp_path):
    regular = tmp_path / "k"
    regular.write_text("x", encoding="utf-8")
    fifo = tmp_path / "f"
    os.mkfifo(fifo)
    with record_opens() as opened:
        assert doctor_cmd._is_regular_file(str(regular)) is True
        assert doctor_cmd._is_regular_file(str(fifo)) is False
        assert doctor_cmd._is_regular_file(str(tmp_path)) is False
        assert doctor_cmd._is_regular_file(str(tmp_path / "missing")) is False
        assert doctor_cmd._is_regular_file(None) is False
        assert doctor_cmd._is_regular_file("bad\0path") is False
    assert str(regular) not in opened and str(fifo) not in opened


# ------------------------------------------------------------------------------- D7: the remote lock


# dispatch is on by default and counts as a remote channel wherever discord.py is installed (CI installs
# the extra); the row under test is mobile's, so dispatch is switched off explicitly in both tests.
@pytest.mark.plugin("mobile")
def test_a_remote_channel_with_the_lock_off_is_named(tmp_path, monkeypatch, fake_home):
    pytest.importorskip("starlette")
    assert REMOTE_ROW in _doctor(tmp_path, monkeypatch, fake_home, dispatch={"enabled": False})


@pytest.mark.plugin("mobile")
@pytest.mark.parametrize("sections", [{"channels": {"remote_unattended": False}}, {"mobile": {"enabled": False}}],
                         ids=["lock on", "no channel plugin on"])
def test_no_remote_row_with_the_lock_on_or_no_channel_on(tmp_path, monkeypatch, fake_home, sections):
    pytest.importorskip("starlette")
    out = _doctor(tmp_path, monkeypatch, fake_home, dispatch={"enabled": False}, **sections)
    assert "Remote channels on" not in out and "Plugins:" in out


def test_the_migration_state_block_keeps_its_one_call_site():
    """Doctor's migration-state block is a two-line revert (its definition and one call)."""
    text = Path(doctor_cmd.__file__).read_text(encoding="utf-8")
    assert text.count("_print_migration_state") == 2
