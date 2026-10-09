"""The one launch decision (spec §5): `launch_decision` judges a launch, `activate` enforces it.

- `launch_decision(spec, keyed, *, peer=None)` returns a frozen four-field `LaunchDecision`
  (refusal, loopback, key_file_ok, note), never a copy of the spec. It only ever `stat`s the key file.
- `SpawnedProcessStrategy.activate` and `ManagedVllmStrategy.activate` ask it once, raise its refusal
  as `LaunchRefused` (a RuntimeError) before anything starts, and carry its note on `LiveEndpoint`.
- The llama.cpp argv ends with `--host 127.0.0.1` whenever the launch binds loopback.

Every launch here is recorded or refused; no process is started and no key file is read.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import threading

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, server
from tests.unit.launch_support import PopenRecorder, popen, record_opens  # noqa: F401

URL = "http://127.0.0.1:8080/v1"


def _llama(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


def _keyed_llama(path, **over) -> ManagedServerConfig:
    return _llama(require_api_key=True, api_key_file=str(path), **over)


@pytest.fixture
def ready(monkeypatch):
    async def wait_ready(base_url, **kw):
        return ["m"]

    monkeypatch.setattr(server, "wait_ready", wait_ready)


def _note(path, reason) -> str:
    return f"--api-key-file {path} unreadable: {reason} — launched on 127.0.0.1 for this run"


# ------------------------------------------------------------------------------- the record


def test_the_decision_is_a_frozen_record_of_exactly_four_fields():
    assert [f.name for f in dataclasses.fields(server.LaunchDecision)] == [
        "refusal", "loopback", "key_file_ok", "note"]
    decision = server.launch_decision(_llama(), False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.loopback = False  # type: ignore[misc]


def test_a_refusal_is_a_runtime_error():
    assert issubclass(server.LaunchRefused, RuntimeError)


# ------------------------------------------------------------------------------- the truth table


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio"])
@pytest.mark.parametrize("bind_all", [False, True])
def test_ollama_and_lmstudio_are_always_loopback(runtime, bind_all):
    spec = ManagedServerConfig(runtime=runtime, model="m", port=11434, bind_all=bind_all)
    assert server.launch_decision(spec, False) == server.LaunchDecision(None, True, False, None)


def test_a_binary_less_llamacpp_primary_is_refused():
    decision = server.launch_decision(_llama(binary=None), False)
    assert decision == server.LaunchDecision("no llama-server — set server.binary", True, False, None)
    assert server.launch_decision(_llama(binary=""), True).refusal == "no llama-server — set server.binary"


def test_a_binary_less_llamacpp_peer_is_refused_naming_the_peer():
    decision = server.launch_decision(_llama(binary=None), False, peer="flash-next")
    assert decision.refusal == "no llama-server — set extra_endpoints[flash-next].lifecycle.binary"
    assert (decision.loopback, decision.key_file_ok, decision.note) == (True, False, None)


@pytest.mark.parametrize("name", ["[old] ünï", "a]b[c", "名前", "x y"])
def test_the_peer_name_is_embedded_verbatim(name):
    refusal = server.launch_decision(_llama(binary=None), False, peer=name).refusal
    assert refusal == f"no llama-server — set extra_endpoints[{name}].lifecycle.binary"


def test_a_set_binary_is_never_checked_on_disk():
    assert server.launch_decision(_llama(binary="/does/not/exist/llama-server"), False).refusal is None


@pytest.mark.parametrize("bind_all", [False, True])
@pytest.mark.parametrize("keyed", [False, True])
def test_no_key_file_binds_by_the_flag_and_has_no_note(bind_all, keyed):
    decision = server.launch_decision(_llama(bind_all=bind_all), keyed)
    assert decision == server.LaunchDecision(None, not bind_all, False, None)


def test_a_vllm_record_is_decided_the_same_way():
    spec = ManagedServerConfig(launch="binary", binary="/opt/vllm", model="org/repo", bind_all=True)
    assert server.launch_decision(spec, False) == server.LaunchDecision(None, False, False, None)


@pytest.mark.parametrize("bind_all", [False, True])
def test_a_regular_key_file_with_a_held_key_is_passed_to_the_server(tmp_path, bind_all):
    key = tmp_path / "key"
    key.write_text("K", encoding="utf-8")
    decision = server.launch_decision(_keyed_llama(key, bind_all=bind_all), True)
    assert decision == server.LaunchDecision(None, not bind_all, True, None)


def test_a_regular_key_file_without_a_held_key_is_left_off_argv_and_loopback(tmp_path):
    key = tmp_path / "key"
    key.write_text("K", encoding="utf-8")
    decision = server.launch_decision(_keyed_llama(key, bind_all=True), False)
    assert decision == server.LaunchDecision(
        None, True, False, _note(key, "the harness holds no key for it"))


def test_a_missing_key_file_is_withheld_with_a_note(tmp_path):
    key = tmp_path / "absent"
    decision = server.launch_decision(_keyed_llama(key, bind_all=True), True)
    assert decision == server.LaunchDecision(None, True, False, _note(key, "no such file"))


def test_a_directory_is_not_a_regular_file(tmp_path):
    decision = server.launch_decision(_keyed_llama(tmp_path, bind_all=True), True)
    assert decision == server.LaunchDecision(None, True, False, _note(tmp_path, "not a regular file"))


def test_a_symlink_to_a_regular_file_counts_as_regular(tmp_path):
    key = tmp_path / "key"
    key.write_text("K", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(key)
    assert server.launch_decision(_keyed_llama(link), True) == server.LaunchDecision(None, True, True, None)


def test_a_symlink_to_a_directory_gets_the_not_regular_note(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    decision = server.launch_decision(_keyed_llama(link, bind_all=True), True)
    assert decision.note == _note(link, "not a regular file") and decision.loopback is True


def test_a_tilde_key_file_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "key").write_text("K", encoding="utf-8")
    decision = server.launch_decision(_llama(require_api_key=True, api_key_file="~/key"), True)
    assert decision == server.LaunchDecision(None, True, True, None)


def test_an_unreadable_directory_entry_is_reported_with_its_reason(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    key = locked / "key"
    key.write_text("K", encoding="utf-8")
    locked.chmod(0)
    try:
        if os.access(locked, os.X_OK):  # running as root: permissions do not apply
            pytest.skip("directory permissions are not enforced for this user")
        decision = server.launch_decision(_keyed_llama(key, bind_all=True), True)
    finally:
        locked.chmod(0o700)
    assert decision.key_file_ok is False and decision.loopback is True
    assert decision.note is not None and decision.note.startswith(f"--api-key-file {key} unreadable: ")
    assert decision.note.endswith(" — launched on 127.0.0.1 for this run")


# ------------------------------------------------------------------------------- never open, never copy


def test_the_decision_never_opens_the_key_file(tmp_path):
    key = tmp_path / "key"
    key.write_text("K", encoding="utf-8")
    with record_opens() as opened:
        server.launch_decision(_keyed_llama(key), True)
        server.launch_decision(_keyed_llama(tmp_path / "absent"), True)
    assert str(key) not in opened


def test_a_fifo_key_file_neither_blocks_nor_is_opened(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    result: list[server.LaunchDecision] = []
    with record_opens() as opened:
        worker = threading.Thread(
            target=lambda: result.append(server.launch_decision(_keyed_llama(fifo, bind_all=True), True)),
            daemon=True)
        worker.start()
        worker.join(5)
    assert not worker.is_alive(), "launch_decision blocked on a FIFO key file"
    assert str(fifo) not in opened
    assert result[0] == server.LaunchDecision(None, True, False, _note(fifo, "not a regular file"))


def test_two_decisions_are_equal_and_leave_the_spec_unchanged(tmp_path):
    key = tmp_path / "key"
    key.write_text("K", encoding="utf-8")
    spec = _keyed_llama(key, bind_all=True)
    before = spec.model_dump()
    first, second = server.launch_decision(spec, True), server.launch_decision(spec, True)
    assert first == second
    assert spec.model_dump() == before
    assert not isinstance(first, ManagedServerConfig)


# ------------------------------------------------------------------------------- activate enforces it


async def test_activate_refuses_a_declared_only_record_before_anything_starts(
        tmp_path, monkeypatch, popen):  # noqa: F811
    started: list = []
    monkeypatch.setattr(server, "serve_command", lambda *a, **k: started.append("serve_command"))
    monkeypatch.setattr(server, "start_server", lambda *a, **k: started.append("start_server"))
    with pytest.raises(server.LaunchRefused) as raised:
        await lifecycle.SpawnedProcessStrategy().activate(_llama(binary=None), tmp_path, URL)
    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value) == "no llama-server — set server.binary"
    assert started == [] and popen == []


async def test_activate_names_the_peer_in_the_refusal(tmp_path, popen):  # noqa: F811
    with pytest.raises(server.LaunchRefused) as raised:
        await lifecycle.SpawnedProcessStrategy().activate(_llama(binary=None), tmp_path, URL,
                                                          peer="flash-next")
    assert str(raised.value) == "no llama-server — set extra_endpoints[flash-next].lifecycle.binary"
    assert popen == []


async def test_activate_embeds_an_awkward_peer_name_verbatim(tmp_path, popen):  # noqa: F811
    with pytest.raises(server.LaunchRefused, match=r"extra_endpoints\[\[old\] ünï\]\.lifecycle\.binary"):
        await lifecycle.SpawnedProcessStrategy().activate(_llama(binary=None), tmp_path, URL,
                                                          peer="[old] ünï")
    assert popen == []


async def test_a_launchable_llamacpp_record_ends_its_argv_with_the_loopback_host(
        tmp_path, popen, ready):  # noqa: F811
    spec = _llama(extra_args=["--host", "0.0.0.0", "-ngl", "99"])
    live = await lifecycle.SpawnedProcessStrategy().activate(spec, tmp_path, URL)
    (call,) = popen
    assert call["cmd"][-2:] == ["--host", "127.0.0.1"]
    assert call["cmd"][:5] == ["/x/llama-server", "-m", "/x/m.gguf", "--port", "8080"]
    assert (live.note, live.handle) == (None, 4242)


async def test_a_withheld_key_file_launches_on_loopback_and_carries_the_note(
        tmp_path, popen, ready):  # noqa: F811
    spec = _keyed_llama(tmp_path, bind_all=True)  # a directory: not a regular file
    live = await lifecycle.SpawnedProcessStrategy().activate(spec, tmp_path, URL)
    (call,) = popen
    assert call["cmd"][-2:] == ["--host", "127.0.0.1"]
    assert "--api-key-file" not in call["cmd"]
    assert live.note == _note(tmp_path, "not a regular file")


async def test_a_vllm_activation_asks_the_same_decision(tmp_path, popen, ready):  # noqa: F811
    spec = ManagedServerConfig(launch="binary", binary="/opt/vllm", model="org/repo", port=8080,
                               extra_args=["--host", "0.0.0.0"])
    live = await lifecycle.ManagedVllmStrategy().activate(spec, tmp_path, URL)
    assert popen[0]["cmd"][-2:] == ["--host", "127.0.0.1"] and live.note is None


async def test_activate_decides_once_and_passes_the_decision_to_the_command(tmp_path, monkeypatch):
    seen: dict = {}
    decisions: list = []
    real = server.launch_decision

    def spy(spec, keyed, *, peer=None):
        decisions.append((spec, keyed, peer))
        return real(spec, keyed, peer=peer)

    def serve_command(spec, **kw):
        seen.update(kw)
        return ["CMD"]

    async def wait_ready(base_url, **kw):
        return ["m"]

    monkeypatch.setattr(server, "launch_decision", spy)
    monkeypatch.setattr(server, "serve_command", serve_command)
    monkeypatch.setattr(server, "start_server", lambda *a, **k: 7)
    monkeypatch.setattr(server, "wait_ready", wait_ready)
    spec = _llama(bind_all=True)
    await lifecycle.SpawnedProcessStrategy().activate(spec, tmp_path, URL, api_key="sk-K", peer="p")
    assert decisions == [(spec, True, "p")]
    assert seen == {"keyed": True, "loopback": False, "key_file_ok": False,
                    "context": None, "cache_ram": None}


# ------------------------------------------------------------------------------- every strategy takes a peer


@pytest.mark.parametrize("cls", [lifecycle.LifecycleStrategy, lifecycle.ManagedVllmStrategy,
                                 lifecycle.SpawnedProcessStrategy, lifecycle.DaemonStrategy,
                                 lifecycle.LmsStrategy])
def test_every_activate_takes_a_keyword_only_peer(cls):
    param = inspect.signature(cls.activate).parameters["peer"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None


def test_live_endpoint_note_defaults_to_none():
    assert lifecycle.LiveEndpoint("u", ["m"], 1).note is None
