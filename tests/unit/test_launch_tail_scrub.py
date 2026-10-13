"""SEC-04, source half (D-22): no exception the provider layer raises carries a secret from the server log.

A server log can hold the key in the two forms llama-server writes (an `Authorization: Bearer <key>` header
line and its startup `api_keys: ****wxyz` line), a URL with credentials, and the key the caller holds.
Every message that quotes the log carries the tail scrubbed of all of them plus `(see log <path>, ref <id>)`;
the raw lines stay in `serve.log`, which gains one `=== ref <id>: <what> ===` line.

Hermetic: a planted `serve.log` in `tmp_path`, the REAL `wait_ready` / `ServerRecovery` with the pidfile
check faked dead (or a `/models` transport that is down); no server, no port, nothing under `~/.localharness`.
"""
from __future__ import annotations

import asyncio
import re

import httpx
import pytest

from localharness.config import redact
from localharness.provider import client as client_mod
from localharness.provider import server
from localharness.provider.lifecycle import ServerRecovery
from tests.unit.reconnect_support import (
    ModelsEndpoint,
    RecordingRelaunch,
    write_dead_pidfile,
    write_live_pidfile,
    write_serve_log,
)

BASE = "http://127.0.0.1:8080/v1"
URL = "http://127.0.0.1:9/v1"
HELD = "held-key-123"
PLANTED = (
    "boot: listening\n"
    "Authorization: Bearer sk-planted\n"
    "main: api_keys: ****abcd\n"
    "GET http://u:p@127.0.0.1:8080/v1 failed\n"
    f"request carried {HELD}\n"
    "CUDA out of memory\n"
)
SECRETS = ("sk-planted", "abcd", "u:p", HELD)
REF = r"\(see log (?P<path>.+), ref (?P<ref>[0-9a-f]{8})\)"


def _plant(tmp_path, text: str = PLANTED) -> str:
    write_serve_log(tmp_path, text)
    return text


def _clean(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, f"{secret!r} leaked into {text[:120]!r}"


def _assert_raw_only_in_log(tmp_path, planted: str, text: str, what: str) -> str:
    """The error ends with the reference line, serve.log still holds every raw line, and gained one
    `=== ref <same id>: <what> ===` line; returns the ref id."""
    end = re.search(REF + r"$", text)
    assert end is not None, text
    assert end["path"] == str(server.log_path(tmp_path))
    log = server.log_path(tmp_path).read_text(encoding="utf-8")
    assert log.startswith(planted)
    assert log[len(planted):].strip() == f"=== ref {end['ref']}: {what} ==="
    return end["ref"]


@pytest.fixture(autouse=True)
def _fast_visibility(monkeypatch):
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.02)


# ------------------------------------------------------------------------------- redact.scrub(patterns=)


def test_scrub_without_patterns_is_unchanged():
    assert redact.scrub("a sk-1 b", {"sk-1"}) == "a sk-1 b".replace("sk-1", redact.SECRET_MASK)
    assert redact.scrub("Bearer abc", ["x"]) == "Bearer abc"  # patterns are opt-in


def test_the_logged_key_patterns_mask_the_two_forms_llama_server_logs():
    got = redact.scrub("Authorization: Bearer abc\napi_keys: ****wxyz", (), patterns=redact.LOGGED_KEY_PATTERNS)
    assert got == f"Authorization: Bearer {redact.SECRET_MASK}\napi_keys: {redact.SECRET_MASK}"


def test_a_pattern_without_a_group_masks_its_whole_match():
    assert redact.scrub("x sk-abc y", (), patterns=[re.compile(r"sk-\w+")]) == f"x {redact.SECRET_MASK} y"


def test_literal_secrets_are_replaced_before_the_patterns_run():
    got = redact.scrub("Bearer held-key-123 ok", [HELD], patterns=redact.LOGGED_KEY_PATTERNS)
    assert HELD not in got and got.startswith("Bearer ") and got.endswith(" ok")


# ------------------------------------------------------------------------------- the tracer: wait_ready dies


async def test_a_server_that_dies_with_a_key_in_its_log_raises_a_clean_referenced_error(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    planted = _plant(tmp_path)
    with pytest.raises(RuntimeError) as died:
        await server.wait_ready(BASE, config_dir=tmp_path, runtime="llamacpp", api_key=HELD)
    text = str(died.value)
    assert text.startswith("llama.cpp exited during startup. Log tail:\n")
    _clean(text)
    assert "CUDA out of memory" in text and "boot: listening" in text  # the rest of the tail is still there
    _assert_raw_only_in_log(tmp_path, planted, text, "llama.cpp exited during startup")


async def test_the_held_key_is_masked_even_when_the_log_echoes_it_in_no_known_form(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    _plant(tmp_path, f"curl -H 'x-token: {HELD}' failed\n")
    with pytest.raises(RuntimeError) as died:
        await server.wait_ready(BASE, config_dir=tmp_path, api_key=HELD)
    assert HELD not in str(died.value) and redact.SECRET_MASK in str(died.value)


async def test_a_secret_wrapped_key_is_masked_too(tmp_path, monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    _plant(tmp_path, f"token {HELD}\n")
    with pytest.raises(RuntimeError) as died:
        await server.wait_ready(BASE, config_dir=tmp_path, api_key=SecretStr(HELD))
    assert HELD not in str(died.value)


def test_scrubbed_tail_masks_every_form_and_log_reference_is_a_pure_append(tmp_path):
    planted = _plant(tmp_path)
    tail = server.scrubbed_tail(tmp_path, secrets=(HELD,))
    _clean(tail)
    assert "Bearer " + redact.SECRET_MASK in tail
    ref = server.log_reference(tmp_path, "something")
    assert re.fullmatch(r"[0-9a-f]{8}", ref)
    assert server.log_path(tmp_path).read_text(encoding="utf-8") == planted + f"\n=== ref {ref}: something ===\n"


def test_log_reference_without_a_server_dir_is_best_effort(tmp_path):
    ref = server.log_reference(tmp_path / "nowhere", "x")  # the directory is absent: still returns an id
    assert re.fullmatch(r"[0-9a-f]{8}", ref)
