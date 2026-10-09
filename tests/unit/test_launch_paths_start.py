"""Launch path 1 of 5: `start`'s down-server branch (spec §5, D-08/D-09/D-13/D-20).

`_start_managed_server` asks `launch_decision` before it announces anything. A declared-only
llama.cpp record prints one refusal line to the error console and `start` exits 1 with no
`Error:` line and no `localharness doctor` pointer; a launchable one names its runtime and carries
the exposure phrase computed from the effective argv. No process is started and no socket opened.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import typer
import yaml

from localharness.cli import start_cmd
from localharness.cli.start_cmd import _start_async, _start_managed_server
from localharness.config.models import ManagedServerConfig
from localharness.provider import server
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries  # noqa: F401

URL = "http://127.0.0.1:8080/v1"
REFUSAL = "no llama-server — set server.binary"
PHRASE = "every interface, no key"


def _llama(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


def _write_config(tmp_path, *, server_block: dict | None) -> None:
    data: dict = {"version": "1",
                  "provider": {"provider_type": "llamacpp", "base_url": URL, "api_key": "none",
                               "default_model": "qwen"}}
    if server_block is not None:
        data["server"] = server_block
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _record_err(monkeypatch) -> list[tuple[tuple, dict]]:
    seen: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: seen.append((a, k)))
    return seen


def _err_text(seen) -> list[str]:
    return [" ".join(str(x) for x in a) for a, _ in seen]


def _boot(tmp_path, monkeypatch, *, server_block, path_binary=None, probe_ok=False):
    """The real `_start_async` with the boundaries stubbed, a failing probe and a start recorder."""
    _stub_start_boundaries(tmp_path, monkeypatch)
    _write_config(tmp_path, server_block=server_block)

    async def probe(llm, max_retries=3, delay=2.0):
        return (True, "native", 262_144, None) if probe_ok else (False, None, None, "Connection error.")

    monkeypatch.setattr(start_cmd, "_probe_llm", probe)
    monkeypatch.setattr(start_cmd, "which_outside_cwd", lambda name: path_binary)
    started: list[tuple] = []

    def start_server(config_dir, cmd, **kw):
        started.append((cmd, kw))
        return 4321

    monkeypatch.setattr(server, "start_server", start_server)
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    return SimpleNamespace(started=started, err=_record_err(monkeypatch),
                           out=_capture_start_console(monkeypatch))


async def _drive(tmp_path) -> int | None:
    try:
        await asyncio.wait_for(_start_async(None, False, False, str(tmp_path)), timeout=60)
    except typer.Exit as exc:
        return exc.exit_code
    return None


DECLARED_ONLY = {"runtime": "llamacpp", "model": "/x/m.gguf", "port": 8080}


# ------------------------------------------------------------------------------- tracer


async def test_a_declared_only_primary_prints_one_refusal_line_and_exits_1(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY)
    assert await _drive(tmp_path) == 1
    lines = _err_text(seen.err)
    assert [line for line in lines if "llama-server" in line] == [REFUSAL]
    assert not [line for line in lines if "Error:" in line or "localharness doctor" in line]
    assert seen.started == []


async def test_the_refusal_is_printed_verbatim_not_as_markup(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY, path_binary="/opt/[bold]x/llama-server")
    assert await _drive(tmp_path) == 1
    (args, kwargs), = [pair for pair in seen.err if "llama-server" in str(pair[0])]
    assert args == (f"{REFUSAL} (PATH has /opt/[bold]x/llama-server)",)
    assert kwargs.get("markup") is False and kwargs.get("soft_wrap") is True


async def test_a_path_binary_is_named_in_the_refusal_text_only(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY, path_binary="/opt/llama/llama-server")
    assert await _drive(tmp_path) == 1
    assert [line for line in _err_text(seen.err) if "llama-server" in line] == [
        "no llama-server — set server.binary (PATH has /opt/llama/llama-server)"]
    assert seen.started == []


async def test_a_refusal_announces_nothing_and_never_activates(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY)
    activated: list[int] = []
    from localharness.provider import lifecycle

    async def activate(self, *a, **k):
        activated.append(1)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)
    assert await _drive(tmp_path) == 1
    assert activated == []
    assert not [line for line in seen.out if "starting it" in line or "still loading" in line]


async def test_an_attach_only_config_keeps_the_unreachable_line_and_the_doctor_pointer(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=None)
    assert await _drive(tmp_path) == 1
    lines = _err_text(seen.err)
    assert any("model endpoint unreachable at" in line for line in lines)
    assert any("Run `localharness doctor` to diagnose" in line for line in lines)
    assert not [line for line in lines if REFUSAL in line]


async def test_a_declared_only_peer_is_not_touched_by_the_primary_sensing(tmp_path, monkeypatch):
    """The sensing line asks PATH once per start; it never launches what it finds."""
    asked: list[str] = []
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY)
    monkeypatch.setattr(start_cmd, "which_outside_cwd", lambda name: asked.append(name))
    assert await _drive(tmp_path) == 1
    assert asked == ["llama-server"]
    assert seen.started == []


def test_the_path_is_sensed_before_the_machine_trust_decision():
    text = open(start_cmd.__file__, encoding="utf-8").read()
    assert text.index('which_outside_cwd("llama-server")') < text.index("decide_machine_trust(cfg_path")


# ------------------------------------------------------------------------------- the helper


async def test_the_helper_returns_the_refusal_and_prints_and_activates_nothing(tmp_path, monkeypatch):
    out = _capture_start_console(monkeypatch)
    err = _record_err(monkeypatch)
    from localharness.provider import lifecycle

    activated: list[int] = []

    async def activate(self, *a, **k):
        activated.append(1)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    spec = ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", port=8080)
    assert await _start_managed_server(spec, tmp_path, URL, {}) == REFUSAL
    assert out == [] and err == [] and activated == []


async def test_the_helper_appends_the_path_only_for_llama_cpp(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    spec = ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", port=8080)
    assert await _start_managed_server(spec, tmp_path, URL, {}, path_binary="/p/llama-server") == (
        f"{REFUSAL} (PATH has /p/llama-server)")
