"""Doctor names every way the harness's own surfaces are reachable from beyond this terminal, each
with the one setting that closes it: a launched model server reachable from other machines
(`server.bind_all`), a launched server that does not require the API key you set
(`server.require_api_key`), and a remote channel that may switch a session to unattended or answer
"always" (`channels.remote_unattended`). Info rows, never failures; a key is never printed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.cli import doctor_cmd
from tests.unit.test_doctor_layer_report import _UNREACHABLE_BASE_URL, _layout, _run_doctor

VLLM = {"runtime": "vllm", "launch": "docker", "docker_image": "img", "model": "test-model", "port": 8081}
NETWORK_ROW = ("i Model server: reachable from other machines on port 8081 (server.bind_all: true) — "
               "set server.bind_all: false to keep it on this machine")
KEY_ROW = ("i Model server: launched without requiring your API key (provider.api_key is set) — "
           "set server.require_api_key: true to make it refuse requests without the key")
REMOTE_ROW = ('i Remote channels on (web): a paired phone or an allowlisted chat account can switch a '
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


# ------------------------------------------------------------------------------- D7: the remote lock


@pytest.mark.plugin("web")
def test_a_remote_channel_with_the_lock_off_is_named(tmp_path, monkeypatch, fake_home):
    pytest.importorskip("starlette")
    assert REMOTE_ROW in _doctor(tmp_path, monkeypatch, fake_home)


@pytest.mark.plugin("web")
@pytest.mark.parametrize("sections", [{"channels": {"remote_unattended": False}}, {"web": {"enabled": False}}],
                         ids=["lock on", "no channel plugin on"])
def test_no_remote_row_with_the_lock_on_or_no_channel_on(tmp_path, monkeypatch, fake_home, sections):
    pytest.importorskip("starlette")
    out = _doctor(tmp_path, monkeypatch, fake_home, **sections)
    assert "Remote channels on" not in out and "Plugins:" in out


def test_the_migration_state_block_keeps_its_one_call_site():
    """Doctor's migration-state block is a two-line revert (its definition and one call)."""
    text = Path(doctor_cmd.__file__).read_text(encoding="utf-8")
    assert text.count("_print_migration_state") == 2
