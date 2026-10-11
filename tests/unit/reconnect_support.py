"""Shared fakes for the reconnect tests (phase 7): a `/models` endpoint that is down for a while, a chat
endpoint whose stream drops mid-body, pidfile and log writers, a recording relaunch, and a key-leak check.

An importable helper module, like tests/unit/launch_support.py: its name does not start with `test_`, so
pytest collects nothing from it. A test module imports what it needs by name:

    from tests.unit.reconnect_support import ModelsEndpoint, RecordingRelaunch  # noqa: F401

Everything here is hermetic: the HTTP fakes are `httpx.MockTransport` handlers (no socket is opened),
the pidfile and log helpers write under a `tmp_path` the test owns, and nothing starts a server.
"""
from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from localharness.provider import lifecycle, server

_DROP = "peer closed connection without sending complete message body: incomplete chunked read"


class ModelsEndpoint:
    """A `GET .../models` handler for `httpx.MockTransport`.

    It answers 503 for the first `down_for` calls (or, when `down` is a zero-argument callable, for as
    long as it returns True) and then 200 `{"data": [{"id": ...}, ...]}` for the ids in `served`.
    `calls` counts every request, `urls` records each request URL and `auth` each request's
    Authorization header (None when it sent none)."""

    def __init__(self, down_for: int = 0, served: tuple[str, ...] = ("m",),
                 down: Callable[[], bool] | None = None) -> None:
        self.down_for = down_for
        self.served = served
        self.down = down
        self.calls = 0
        self.urls: list[str] = []
        self.auth: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.urls.append(str(request.url))
        self.auth.append(request.headers.get("authorization"))
        is_down = self.down() if self.down is not None else self.calls <= self.down_for
        if is_down:
            return httpx.Response(503, json={"error": "loading"})
        return httpx.Response(200, json={"data": [{"id": model} for model in self.served]})


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _usage_chunk(prompt: int = 3, completion: int = 1) -> dict:
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion}}


def text_answer(text: str) -> list[dict]:
    """The OpenAI streaming chunks of a plain text answer: the text, a `stop` finish, a usage chunk."""
    return [_chunk({"role": "assistant", "content": text}), _chunk({}, "stop"), _usage_chunk()]


def tool_call_answer(name: str, arguments_json: str, call_id: str = "call_1") -> list[dict]:
    """The chunks of an answer that calls one tool: the call, a `tool_calls` finish, a usage chunk."""
    call = {"index": 0, "id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments_json}}
    return [_chunk({"role": "assistant", "content": None, "tool_calls": [call]}),
            _chunk({}, "tool_calls"), _usage_chunk()]


def _sse(chunk: dict) -> bytes:
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


class _Stream(httpx.AsyncByteStream):
    """A response body of SSE chunks that may end in a transport error instead of `[DONE]`."""

    def __init__(self, chunks: list[dict], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    async def __aiter__(self):
        for chunk in self._chunks:
            yield _sse(chunk)
        if self._error is not None:
            raise self._error
        yield b"data: [DONE]\n\n"


class DroppingChat:
    """A `POST .../chat/completions` handler for `httpx.MockTransport` whose stream can drop.

    A call drops (200 headers, one SSE chunk, then `httpx.RemoteProtocolError`) when it is the first
    call and `drop_first`, or whenever `down` (a zero-argument callable) returns True. Any other call
    streams the next of `answers` (the last one repeats; default `text_answer("ok")`), ending with a
    usage chunk and `data: [DONE]`. Dropping after the headers keeps the openai SDK from retrying, so
    no test pays its backoff. `bodies` holds each request's raw body."""

    def __init__(self, answers: list[list[dict]] | None = None, drop_first: bool = True,
                 down: Callable[[], bool] | None = None) -> None:
        self.answers = answers if answers else [text_answer("ok")]
        self.drop_first = drop_first
        self.down = down
        self.bodies: list[bytes] = []
        self._served = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        first = len(self.bodies) == 1
        drops = (first and self.drop_first) or (self.down is not None and self.down())
        headers = {"content-type": "text/event-stream"}
        if drops:
            partial = [_chunk({"role": "assistant", "content": "par"})]
            return httpx.Response(200, headers=headers,
                                  stream=_Stream(partial, httpx.RemoteProtocolError(_DROP)))
        answer = self.answers[min(self._served, len(self.answers) - 1)]
        self._served += 1
        return httpx.Response(200, headers=headers, stream=_Stream(answer))


def write_live_pidfile(config_dir: Path) -> None:
    """Write a pidfile for this very process, with its `/proc` start time, so `server.server_pid` reports
    it alive. WARNING: never let a test run a real `stop_server` against this dir; it would signal the
    test runner's own process group (the `boundaries` fixtures stub `stop_server`)."""
    pid = os.getpid()
    start = server._proc_start_time(pid)
    server.server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    server.pid_path(config_dir).write_text(f"{pid}" if start is None else f"{pid} {start}", encoding="utf-8")


def _unused_pid() -> int:
    """A pid no process has right now (probed from the top of the pid space down)."""
    pid = 4_000_000
    while pid > 2:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            pass
        pid -= 1
    raise RuntimeError("no free pid to play a dead server")  # pragma: no cover


def write_dead_pidfile(config_dir: Path) -> None:
    """Write a pidfile whose process is not running, so `server.server_pid` reports None."""
    server.server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    server.pid_path(config_dir).write_text(str(_unused_pid()), encoding="utf-8")


def write_serve_log(config_dir: Path, text: str) -> None:
    """Write `serve.log` for a config dir, the file `server.log_tail` reads."""
    server.server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    server.log_path(config_dir).write_text(text, encoding="utf-8")


_DEFAULT = object()


class RecordingRelaunch:
    """An async zero-argument `relaunch` that records what happened to it.

    `calls` counts invocations and `budgets` holds the `lifecycle.relaunch_budget()` each one saw.
    `side_effect` may be a callable run on each call (e.g. writing a pidfile; if it returns an
    awaitable that is awaited), an exception (class or instance) to raise, or an awaitable to await.
    It returns a `LiveEndpoint` unless `returns` says otherwise (`returns=None` plays a relaunch that
    found another holder had already brought a server up)."""

    def __init__(self, side_effect: Any = None, returns: Any = _DEFAULT) -> None:
        self.side_effect = side_effect
        self.returns = returns
        self.calls = 0
        self.budgets: list[float] = []

    async def __call__(self) -> lifecycle.LiveEndpoint | None:
        self.calls += 1
        self.budgets.append(lifecycle.relaunch_budget())
        effect = self.side_effect
        if inspect.isclass(effect) and issubclass(effect, BaseException):
            raise effect()
        if isinstance(effect, BaseException):
            raise effect
        if callable(effect):
            effect = effect()
        if inspect.isawaitable(effect):
            await effect
        if self.returns is _DEFAULT:
            return lifecycle.LiveEndpoint("http://127.0.0.1:9/v1", ["m"], None)
        return self.returns


def assert_no_key_leak(sentinel: str, *texts: str, caplog: Any = None, capsys: Any = None) -> None:
    """Fail when `sentinel` appears in any of `texts`, in a captured log record (message or args), or in
    captured stdout/stderr."""
    for text in texts:
        assert sentinel not in text, f"key leaked into {text[:80]!r}"
    if caplog is not None:
        for record in caplog.records:
            assert sentinel not in record.getMessage(), f"key leaked into a log record: {record.name}"
            assert sentinel not in repr(record.args), f"key leaked into log args: {record.name}"
            assert sentinel not in (caplog.text or ""), "key leaked into the log text"
    if capsys is not None:
        out, err = capsys.readouterr()
        assert sentinel not in out and sentinel not in err, "key leaked to stdout/stderr"
