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
import socket
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


# ------------------------------------------------------------------------------- the literal host classifier


@pytest.mark.parametrize("value", ["127.0.0.1", "127.0.0.2", "127.0.0.1,::1", "localhost", "::ffff:127.0.0.1",
                                   "::1", " 127.0.0.1 , localhost ", "127.0.0.1,"])
def test_loopback_hosts_classify_as_loopback(value):
    assert server.classify_hosts(value) == "loopback"


@pytest.mark.parametrize("value", ["0.0.0.0", "192.168.1.5", "127.0.0.1,0.0.0.0", "::", "100.64.0.7",
                                   "::ffff:192.168.1.5"])
def test_other_addresses_classify_as_address(value):
    assert server.classify_hosts(value) == "address"


@pytest.mark.parametrize("value", ["ip6-localhost", "/run/llama.sock", "", " , ", "example.com",
                                   "127.0.0.1,ip6-localhost", "0.0.0.0,/run/x.sock"])
def test_names_paths_and_nothing_classify_as_unsupported(value):
    assert server.classify_hosts(value) == "unsupported"


def test_the_classifier_never_resolves_a_name(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("classify_hosts resolved a name")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "gethostbyname", boom)
    assert [server.classify_hosts(v) for v in ("localhost", "127.0.0.1", "0.0.0.0", "example.com", "")] == [
        "loopback", "loopback", "address", "unsupported", "unsupported"]


# ------------------------------------------------------------------------------- the effective posture


def _posture(spec, keyed=False, **kw):
    return server.effective_posture(spec, keyed=keyed, decision=server.launch_decision(spec, keyed, **kw))


def _vllm_binary(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"launch": "binary", "binary": "/opt/vllm", "model": "org/repo",
                                  "port": 8081, **over})


def _vllm_docker(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"launch": "docker", "docker_image": "img:tag", "model": "org/repo",
                                  "port": 8081, **over})


def test_the_exposure_phrase_is_the_contract():
    assert server.EVERY_INTERFACE_NO_KEY == "every interface, no key"


def test_a_bind_all_llama_cpp_launch_without_a_key_is_exposed_and_says_so():
    assert _posture(_llama(bind_all=True)) == (True, "every interface, no key")


def test_a_bind_all_llama_cpp_launch_with_a_held_key_is_exposed_without_the_phrase():
    assert _posture(_llama(bind_all=True), keyed=True) == (True, None)


def test_a_loopback_llama_cpp_launch_is_not_exposed():
    assert _posture(_llama()) == (False, None)
    assert _posture(_llama(), keyed=True) == (False, None)


def test_require_api_key_with_no_key_held_still_reads_every_interface_no_key():
    spec = _llama(bind_all=True, require_api_key=True)
    assert _posture(spec, keyed=False) == (True, "every interface, no key")


def test_a_regular_key_file_with_a_held_key_counts_as_keyed(tmp_path):
    key = tmp_path / "key"
    key.write_text("K", encoding="utf-8")
    with record_opens() as opened:
        assert _posture(_keyed_llama(key, bind_all=True), keyed=True) == (True, None)
    assert str(key) not in opened


def test_a_withheld_key_file_launches_loopback_and_is_not_exposed(tmp_path):
    assert _posture(_keyed_llama(tmp_path, bind_all=True), keyed=True) == (False, None)


def test_a_vllm_binary_binds_every_interface_by_default_when_bind_all():
    assert _posture(_vllm_binary(bind_all=True)) == (True, "every interface, no key")
    assert _posture(_vllm_binary()) == (False, None)


def test_a_vllm_binary_keyed_by_its_own_argv_is_not_flagged():
    spec = _vllm_binary(bind_all=True, extra_args=["--api-key", "K"])
    assert _posture(spec) == (True, None)
    assert _posture(_vllm_binary(bind_all=True, extra_args=["--api-key=K"])) == (True, None)


def test_the_effective_host_is_the_last_one_in_either_form():
    narrow = _llama(bind_all=True, extra_args=["--host=127.0.0.1"])
    assert _posture(narrow) == (False, None)
    wide = _vllm_binary(bind_all=True, extra_args=["--host", "127.0.0.1", "--host=0.0.0.0"])
    assert _posture(wide) == (True, "every interface, no key")
    assert _posture(_vllm_binary(bind_all=True, extra_args=["--host", "127.0.0.1"])) == (False, None)


def test_a_host_that_is_a_name_or_a_socket_path_counts_as_exposed():
    assert _posture(_llama(bind_all=True, extra_args=["--host", "/run/llama.sock"])) == (
        True, "every interface, no key")


def test_a_docker_launch_is_exposed_when_it_publishes_beyond_loopback():
    assert _posture(_vllm_docker(bind_all=True)) == (True, "every interface, no key")
    assert _posture(_vllm_docker(bind_all=True), keyed=True) == (True, None)
    assert _posture(_vllm_docker()) == (False, None)


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio"])
def test_ollama_and_lmstudio_are_never_exposed_and_build_no_command(runtime, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("serve_command must not be called")

    monkeypatch.setattr(server, "serve_command", boom)
    spec = ManagedServerConfig(runtime=runtime, model="m", bind_all=True)
    assert _posture(spec) == (False, None)


def test_a_refused_launch_is_not_exposed():
    assert _posture(_llama(binary=None, bind_all=True)) == (False, None)
