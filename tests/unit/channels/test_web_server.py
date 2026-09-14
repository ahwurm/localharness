"""The HTTP surface, driven in-process over ASGI — no socket, no model, no GPU.

`httpx.ASGITransport` is the web channel's equivalent of the ACP test's socket pair: the point is
that every assertion here speaks the real wire — real routes, real auth, real SSE framing —
rather than calling the server's own methods and proving nothing about what a phone would see.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from localharness.channels.web import auth
from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import WebServer
from localharness.core.bus import EventBus
from localharness.core.events import Action, Observation, TaskComplete

pytestmark = pytest.mark.asyncio

TOKEN = "test-token-not-a-real-one"
JSON = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
BEARER = {"Authorization": f"Bearer {TOKEN}"}


async def _stack(tmp_path, **kw):
    bus = EventBus(persist_path=tmp_path / "bus-events.jsonl")
    channel = WebChannel(bus=bus, config={})
    await channel.start()
    channel.bind_runtime(
        session_id="s1", agent_id="orchestrator",
        session_dir=tmp_path / "sessions", **kw.pop("runtime", {}),
    )
    server = WebServer(channel, token=TOKEN, **kw)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://web.test"
    )
    return bus, channel, server, client


READ_TIMEOUT_S = 5.0
"""A hard ceiling on any single stream read in this file.

The stream is infinite BY DESIGN — that is the feature — so a test that waits for a frame which
never arrives would hang the suite rather than fail it. Bounded here so the failure mode of a
broken assertion is a red test, not a wedged run.
"""


class _Hangup(Exception):
    """Thrown into the app once the test has the frames it came for, to unwind an endless stream."""


async def _read_frames(server, n, *, path="/api/stream", headers=None):
    """Collect `n` SSE frames by driving the ASGI app directly.

    NOT through `httpx.ASGITransport`: it buffers the whole response — it awaits the application
    to completion before returning a `Response` — so an endless SSE stream hangs it forever. That
    is a limitation of the test transport, not of the server, and driving ASGI by hand is the
    better test anyway: these assertions are on the exact bytes a phone would receive, framing
    included, rather than on something a client library reassembled.
    """
    raw = bytearray()
    query = path.partition("?")[2].encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "path": path.partition("?")[0], "raw_path": path.partition("?")[0].encode(),
        "query_string": query, "scheme": "http", "root_path": "",
        "server": ("web.test", 80), "client": ("127.0.0.1", 1),
        "headers": [(k.lower().encode(), v.encode())
                    for k, v in (headers if headers is not None else BEARER).items()],
    }
    started: dict = {}

    async def receive():
        await asyncio.sleep(3600)          # a browser holds the request open; so do we
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            started.update(status=message["status"], headers=dict(message.get("headers", [])))
        elif message["type"] == "http.response.body":
            raw.extend(message.get("body", b""))
            if raw.count(b"\n\n") >= n:
                raise _Hangup

    try:
        await asyncio.wait_for(server.app(scope, receive, send), timeout=READ_TIMEOUT_S)
    except _Hangup:
        pass
    except asyncio.TimeoutError:
        raise AssertionError(f"wanted {n} frames from {path}; got:\n{raw.decode()}") from None

    assert started.get("status") == 200, started
    assert started["headers"][b"content-type"].startswith(b"text/event-stream")

    frames: list[tuple[str, int | None, dict]] = []
    for block in raw.decode().split("\n\n"):
        event = seq = data = None
        for line in block.splitlines():
            if line.startswith("id: "):
                seq = int(line[4:])
            elif line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if data is not None:
            frames.append((event, seq, data))
    return frames


# ------------------------------------------------------------------ auth (WEBCH-13)

async def test_every_api_route_refuses_an_unauthenticated_caller(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    for path in ("/api/stream", "/api/health", "/api/protocol", "/api/schema", "/api/tools",
                 "/api/grants", "/api/permissions", "/api/sessions/s1/events"):
        assert (await client.get(path)).status_code == 401, path
    for path in ("/api/sessions/s1/message", "/api/sessions/s1/cancel",
                 "/api/sessions/s1/mode", "/api/sessions/s1/command",
                 "/api/permissions/x/answer", "/api/pending/1/approve", "/api/bringup/abort"):
        got = await client.post(path, json={})
        assert got.status_code == 401, path
    assert (await client.get("/api/health", headers=BEARER)).status_code == 200


async def test_a_wrong_token_is_refused(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    got = await client.get("/api/health", headers={"Authorization": "Bearer nearly-right"})
    assert got.status_code == 401


async def test_post_refuses_the_csrf_friendly_content_types(tmp_path):
    """§7.3: the classic bypass is a plain <form> or a text/plain fetch — a SIMPLE request that
    skips the preflight entirely. Requiring JSON is what makes the POST surface structurally
    CSRF-safe rather than leaning on the cookie's SameSite alone."""
    _, _, _, client = await _stack(tmp_path)
    for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        got = await client.post(
            "/api/sessions/s1/message", content=b'{"text":"hi"}',
            headers={"Content-Type": ctype, "Authorization": f"Bearer {TOKEN}"},
        )
        assert got.status_code == 415, ctype
        assert "preflight" in got.json()["error"]


async def test_enrolment_hands_back_a_samesite_strict_cookie(tmp_path):
    """`EventSource` cannot send an Authorization header, and the token is never put in a URL."""
    _, _, _, client = await _stack(tmp_path)
    got = await client.post("/api/auth/enroll", json={}, headers=JSON)
    assert got.status_code == 200
    raw = got.headers["set-cookie"].lower()
    assert auth.AUTH_COOKIE in raw and "httponly" in raw and "samesite=strict" in raw


async def test_the_stream_accepts_the_cookie_the_way_eventsource_would(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    client.cookies.set(auth.AUTH_COOKIE, TOKEN)
    got = await client.get("/api/health")
    assert got.status_code == 200


# ------------------------------------------------------------------ the stream

async def test_the_stream_opens_with_hello_and_stamps_the_bus_seq_as_the_sse_id(tmp_path):
    """WEBCH-03. And the Hello frame itself carries NO id, so the resume cursor always names a
    real, replayable event."""
    bus, channel, server, client = await _stack(tmp_path)

    async def _publish_soon():
        await asyncio.sleep(0.05)
        await bus.publish(Action(agent_id="a", session_id="s1", action_type="tool_call",
                                 tool_name="read", tool_params={"path": "x"},
                                 tool_call_id="c1"))

    task = asyncio.ensure_future(_publish_soon())
    frames = await _read_frames(server, 2)
    await task

    assert frames[0][0] == "Hello" and frames[0][1] is None
    assert frames[0][2]["protocol_version"] == 1
    assert frames[0][2]["session_id"] == "s1"
    assert frames[1][0] == "Action"
    assert frames[1][1] is not None and frames[1][1] == frames[1][2]["seq"]
    assert frames[1][2]["tool_call_id"] == "c1"


async def test_reconnect_backfills_from_the_cursor_then_tails_with_no_gaps_or_duplicates(tmp_path):
    """WEBCH-06 / WEBCH-33: replay-into-stream + live tail + seq de-dup at the seam."""
    bus, channel, server, client = await _stack(tmp_path)
    published = []
    for i in range(4):
        published.append(await bus.publish(Observation(
            agent_id="a", session_id="s1", observation_type="tool_result",
            tool_name="read", tool_call_id=f"c{i}", output=f"out-{i}",
        )))

    # Reconnect from the middle: everything at or after that seq, exactly once.
    frames = await _read_frames(server, 3, path=f"/api/stream?from={published[1].seq}")
    assert frames[0][0] == "Hello"
    seqs = [f[1] for f in frames[1:]]
    assert seqs == [published[1].seq, published[2].seq]
    assert len(seqs) == len(set(seqs))
    assert [f[2]["output"] for f in frames[1:]] == ["out-1", "out-2"]


async def test_the_backfill_matches_the_session_log_byte_for_byte(tmp_path):
    """The diff-against-the-JSONL check WEBCH-06 asks for, at the wire."""
    bus, _, _, client = await _stack(tmp_path)
    events = [await bus.publish(TaskComplete(
        agent_id="a", session_id="s1", success=True, summary=f"answer {i}",
        duration_seconds=1.0, iterations=1,
    )) for i in range(3)]

    got = await client.get("/api/sessions/s1/events?from=0", headers=BEARER)
    assert got.status_code == 200
    served = [json.loads(line) for line in got.text.splitlines()]
    on_disk = [json.loads(line) for line in
               (tmp_path / "sessions" / "s1.jsonl").read_text().splitlines()]
    assert served == on_disk
    assert [row["summary"] for row in served] == [e.summary for e in events]


async def test_a_persist_hole_surfaces_as_a_visible_gap(tmp_path):
    """WEBCH-33. The bus logs a persist failure and delivers anyway, so an event can be
    live-visible and permanently absent from the file a reconnecting client replays from."""
    bus, channel, server, client = await _stack(tmp_path)
    await bus.publish(TaskComplete(agent_id="a", session_id="s1", success=True, summary="kept",
                                   duration_seconds=1.0, iterations=1))
    # The channel forwarded further than the log can serve — exactly the swallowed-write shape.
    channel._forwarded_max["s1"] = 99

    frames = await _read_frames(server, 3, path="/api/stream?from=0")
    kinds = [f[0] for f in frames]
    assert "GapDetected" in kinds
    gap = next(f[2] for f in frames if f[0] == "GapDetected")
    assert gap["to_seq"] == 99 and "missing from the session log" in gap["detail"]


# ------------------------------------------------------------------ the verbs

async def test_a_message_between_turns_queues_for_the_repl(tmp_path):
    _, channel, _, client = await _stack(tmp_path)
    got = await client.post("/api/sessions/s1/message", json={"text": "hello"}, headers=JSON)
    assert got.json()["status"] == "queued"
    await channel.start()
    assert await asyncio.wait_for(channel.read_input(), timeout=1) == "hello"


async def test_mid_turn_an_omitted_intent_queues_rather_than_spending_the_classifier(tmp_path):
    """§4.4.1: tier-2's budget is permit_wait + timeout = ~35s on a capacity-1 inference gate,
    and it resolves to QUEUE anyway. Paying a model call to reach the answer we would have
    chosen for free is the worst available trade — bug #92 was exactly this."""
    _, channel, _, client = await _stack(tmp_path)
    channel._turn_running = True
    got = await client.post("/api/sessions/s1/message", json={"text": "also check the logs"},
                            headers=JSON)
    assert got.json() == {"status": "queued", "intent": "queue"}
    assert await asyncio.wait_for(channel.read_input(), timeout=1) == "also check the logs"


async def test_an_explicit_nudge_reaches_the_running_turn(tmp_path):
    """WEBCH-23: the human already said which they meant, so no classifier is spent."""
    _, channel, _, client = await _stack(tmp_path)
    channel._turn_running = True
    seen: list[str] = []

    async def _nudge(text: str, intent: str) -> bool:
        seen.append(text)
        return True

    channel._nudge_resolver = _nudge
    got = await client.post("/api/sessions/s1/message",
                            json={"text": "actually stop at step 2", "intent": "nudge"},
                            headers=JSON)
    assert got.json() == {"status": "nudged", "intent": "nudge"}
    assert seen == ["actually stop at step 2"]


async def test_a_nudge_with_no_handle_says_so_rather_than_doing_nothing(tmp_path):
    """Discord's PENDING_NO_RESOLVER_LINE precedent: a button that silently does nothing is the
    defect that teaches somebody the feature is broken."""
    _, channel, _, client = await _stack(tmp_path)
    channel._turn_running = True
    got = await client.post("/api/sessions/s1/message", json={"text": "x", "intent": "nudge"},
                            headers=JSON)
    assert got.status_code == 409 and got.json()["status"] == "no_resolver"


async def test_an_unknown_intent_is_refused(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    got = await client.post("/api/sessions/s1/message", json={"text": "x", "intent": "guess"},
                            headers=JSON)
    assert got.status_code == 400


async def test_a_slash_command_goes_through_the_repls_own_dispatcher(tmp_path):
    """WEBCH-19: not re-implemented here. ACP had to re-implement /pending, /approve and /deny
    because it took the self-driving shape, and that drift is what this channel avoids."""
    _, channel, _, client = await _stack(tmp_path)
    got = await client.post("/api/sessions/s1/command", json={"text": "mode guarded"}, headers=JSON)
    assert got.json() == {"status": "queued", "command": "/mode"}
    assert await asyncio.wait_for(channel.read_input(), timeout=1) == "/mode guarded"


async def test_cancel_reports_honestly_when_nothing_is_running(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    got = await client.post("/api/sessions/s1/cancel", json={}, headers=JSON)
    assert got.json() == {"status": "nothing_to_cancel"}


async def test_mode_is_validated_by_the_gate_not_by_the_route(tmp_path):
    class _Gate:
        mode = "auto"

        def set_mode(self, name, *, from_channel=False):
            if name not in ("auto", "guarded"):
                raise ValueError(f"unknown mode {name!r}; choose one of: auto, guarded")
            self.mode = name
            return name

    _, channel, _, client = await _stack(tmp_path, runtime={"gate": _Gate()})
    ok = await client.post("/api/sessions/s1/mode", json={"mode": "guarded"}, headers=JSON)
    assert ok.json() == {"status": "set", "mode": "guarded"}
    bad = await client.post("/api/sessions/s1/mode", json={"mode": "yolo"}, headers=JSON)
    assert bad.status_code == 400 and "unknown mode" in bad.json()["error"]


async def test_answering_a_blocking_ask_over_the_wire_needs_two_posts_for_always(tmp_path):
    """WEBCH-10's acceptance criterion, at the wire: a SINGLE curl of `allow_always` writes no
    grant, which is what proves the confirm is not client-side decoration."""
    from localharness.agent.gate_types import PermissionRequest

    _, channel, _, client = await _stack(tmp_path)
    task = asyncio.ensure_future(channel.ask_permission(PermissionRequest(
        tool_name="bash_exec", tool_params={"command": "ls"}, klass="shell-new", key="bash:ls",
        grantable=True, reason="new command", display="bash_exec: ls",
    )))
    await asyncio.sleep(0)
    request_id = next(iter(channel._open_asks))

    first = await client.post(f"/api/permissions/{request_id}/answer",
                              json={"kind": "allow_always"}, headers=JSON)
    assert first.status_code == 409 and first.json()["status"] == "confirm_required"
    assert not task.done(), "one request must never produce a permanent global grant"

    second = await client.post(
        f"/api/permissions/{request_id}/answer",
        json={"kind": "allow_always", "confirm_token": first.json()["confirm_token"]},
        headers=JSON,
    )
    assert second.json() == {"status": "recorded", "decision": "allow_always"}
    assert (await task).kind == "allow_always"


async def test_open_questions_are_served_as_state(tmp_path):
    """WEBCH-09: the ask survives a reconnect because it is state, not a message."""
    from localharness.agent.gate_types import PermissionRequest

    _, channel, _, client = await _stack(tmp_path)
    task = asyncio.ensure_future(channel.ask_permission(PermissionRequest(
        tool_name="write", tool_params={"path": "/etc/hosts"}, klass="protected-path", key=None,
        grantable=False, reason="outside the workspace", display="write: /etc/hosts",
    )))
    await asyncio.sleep(0)
    body = (await client.get("/api/permissions", headers=BEARER)).json()
    assert len(body["blocking"]) == 1
    assert body["blocking"][0]["tool_name"] == "write"
    assert body["parked"] == []
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_the_parked_queue_is_served_beside_the_blocking_one(tmp_path):
    """Without the parked half a cold client would have to fold every PermissionStaged not yet
    matched by a PermissionResolved out of the whole session log by hand."""
    class _Pending:
        id, rendering, agent_label, session_id, created_at = 2, "bash: rm -rf x", "", "s1", 1.0
        request = type("R", (), {"tool_name": "bash_exec", "klass": "shell-destructive"})()

    class _Gate:
        mode = "auto"
        pending = {2: _Pending()}

    _, _, _, client = await _stack(tmp_path, runtime={"gate": _Gate()})
    body = (await client.get("/api/permissions", headers=BEARER)).json()
    assert [p["id"] for p in body["parked"]] == [2]
    assert body["parked"][0]["rendering"] == "bash: rm -rf x"


async def test_answering_a_parked_call_says_it_runs_when_the_model_reissues_it(tmp_path):
    """`gate.approve` records the answer and dispatches NOTHING. A UI that says "ran" is lying."""
    _, channel, _, client = await _stack(tmp_path)
    answered: list[tuple[str, int]] = []

    async def _resolve(action: str, pending_id: int) -> None:
        answered.append((action, pending_id))

    channel._pending_resolver = _resolve
    got = await client.post("/api/pending/3/approve", json={}, headers=JSON)
    assert answered == [("approve", 3)]
    assert "re-issues" in got.json()["note"]
    assert (await client.post("/api/pending/3/shrug", json={}, headers=JSON)).status_code == 404


# ------------------------------------------------------------------ the describing endpoints

async def test_protocol_serves_the_commands_and_modes_from_their_own_sources(tmp_path):
    """§8: a mode chip or command menu that duplicates those lists client-side is the same drift
    the generated schema exists to prevent."""
    from localharness.agent.gate import MODE_STRICTNESS
    from localharness.cli.slash_commands import SLASH_COMMANDS

    _, _, _, client = await _stack(tmp_path)
    body = (await client.get("/api/protocol", headers=BEARER)).json()
    assert body["protocol_version"] == 1
    assert [c["name"] for c in body["commands"]] == [n for n, _ in SLASH_COMMANDS]
    assert set(body["modes"]) == set(MODE_STRICTNESS)
    assert body["default_mid_turn_intent"] == "queue"
    assert set(body["collapsible_groups"]) == {"fs.read", "web", "memory"}
    dead = {e["name"] for e in body["events"] if e["never_fires"]}
    assert "DelegationRequest" in dead and "Action" not in dead
    assert "TokenDelta" in body["sse_only"] and "BlockingAsk" in body["sse_only"]
    assert any("must not be rendered" in rule.lower() for rule in body["transcript_rules"])


async def test_schema_is_generated_from_the_models(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    body = (await client.get("/api/schema", headers=BEARER)).json()
    assert body["events"]["Action"]["properties"]["has_tool_calls"]["type"] == "boolean"
    assert "description" in body["sse_only"]["TokenDelta"]["properties"]["stream_id"]


async def test_tools_serves_group_and_destructive_for_the_collapse_rule(tmp_path):
    """WEBCH-32: adding a new dangerous tool server-side must not require a client edit to keep
    it itemized."""
    class _Schema:
        name, group, destructive, description = "bash_exec", "shell", True, "run a command"

    class _Tool:
        def info(self):
            return _Schema()

    class _Registry:
        _tools = {"global": {"bash_exec": _Tool()}, "mcp": {}}

    _, _, _, client = await _stack(tmp_path, runtime={"tool_registry": _Registry()})
    body = (await client.get("/api/tools", headers=BEARER)).json()
    assert body["tools"] == [{
        "name": "bash_exec", "group": "shell", "destructive": True,
        "description": "run a command",
    }]
    assert "shell" not in body["collapsible_groups"]


async def test_health_distinguishes_cold_from_unreachable(tmp_path):
    """WEBCH-26: neither is an indefinite spinner, and they are different problems."""
    _, channel, _, client = await _stack(tmp_path)
    channel._model_reachable = False
    assert (await client.get("/api/health", headers=BEARER)).json()["model_state"] == "unreachable"
    channel._model_reachable = True
    body = (await client.get("/api/health", headers=BEARER)).json()
    assert body["model_state"] == "ready" and body["session_live"] is True


async def test_grants_are_listed_read_only_and_say_they_cannot_be_revoked(tmp_path):
    """WEBCH-41 / §5.3b: a fat-thumbed tap is forever, so the least this owes is visibility."""
    _, _, _, client = await _stack(tmp_path)
    body = (await client.get("/api/grants", headers=BEARER)).json()
    assert body["revocable"] is False
    assert "no revoke command" in body["note"]
    assert isinstance(body["grants"], list)


async def test_a_missing_tool_result_says_why_rather_than_implying_it_exists(tmp_path):
    """§4.3: for a cap-truncated result, the output no longer exists ANYWHERE."""
    _, _, _, client = await _stack(tmp_path)
    got = await client.get("/api/tool-results/nope", headers=BEARER)
    assert got.status_code == 404
    assert "no longer exists anywhere" in got.json()["detail"]


# ------------------------------------------------------------------ path confinement (WEBCH-40)

async def test_the_ui_directory_is_realpath_confined(tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<p>hi</p>")
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")
    (ui / "escape.txt").symlink_to(secret)

    _, _, _, client = await _stack(tmp_path, ui_dir=ui)
    assert (await client.get("/")).text == "<p>hi</p>"
    # A symlink out of the root resolves outside it and is refused; a string-prefix check would
    # have served it.
    assert (await client.get("/escape.txt")).status_code == 404
    assert (await client.get("/../secret.txt")).status_code in (301, 307, 404)
    assert (await client.get("/%2e%2e/secret.txt")).status_code == 404


async def test_the_packaged_reference_page_is_served_by_default(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    body = (await client.get("/")).text
    assert "localharness" in body
    assert "/api/stream" in body, "the page must actually consume the wire it documents"


async def test_replay_refuses_a_path_that_is_not_a_session_log(tmp_path):
    from localharness.channels.web.replay import ReplayDriver

    secret = tmp_path / "secrets.env"
    secret.write_text("TOKEN=hunter2")
    with pytest.raises(ValueError):
        ReplayDriver.resolve(str(secret))
    with pytest.raises(ValueError):
        ReplayDriver.resolve(str(tmp_path / "does-not-exist.jsonl"))
    not_a_log = tmp_path / "nope.jsonl"
    not_a_log.write_text('{"hello":"world"}\n')
    with pytest.raises(ValueError):
        ReplayDriver.resolve(str(not_a_log))


# ------------------------------------------------------------------ bind + token (§7)

async def test_a_non_loopback_bind_is_refused_without_an_explicit_override():
    assert auth.check_bind("127.0.0.1") == "127.0.0.1"
    assert auth.check_bind("::1") == "::1"
    with pytest.raises(ValueError) as exc:
        auth.check_bind("0.0.0.0")
    assert "runs shell commands" in str(exc.value)
    assert auth.check_bind("0.0.0.0", allow_unsafe=True) == "0.0.0.0"


async def test_the_token_is_generated_once_stored_0600_and_rotatable(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    first, created = auth.load_or_create_token()
    assert created is True and len(first) > 30
    again, created_again = auth.load_or_create_token()
    assert again == first and created_again is False
    path = auth.token_path()
    assert (path.stat().st_mode & 0o777) == 0o600
    assert auth.rotate_token() not in (first, "")
    assert auth.load_or_create_token()[0] != first


async def test_an_event_published_during_the_backfill_is_neither_dropped_nor_duplicated(tmp_path):
    """The off-by-one race that sits next to the reconnect seam (§4.2.3), driven deliberately.

    The composition is subscribe-FIRST, then backfill, then drain with a `seq` de-dup — an order
    that can duplicate but can never drop, which is the safe side. This publishes a new event
    halfway through consuming the backfill, which is precisely the window, and asserts the seam
    serves it exactly once.
    """
    bus, channel, server, _ = await _stack(tmp_path)
    pub = [await bus.publish(Observation(
        agent_id="a", session_id="s1", observation_type="tool_result",
        tool_name="read", tool_call_id=f"c{i}", output=f"o{i}",
    )) for i in range(6)]

    client = channel.attach_client()
    seen: list[int] = []
    served: int | None = None
    injected = False
    async for _, seq, _line in server._backfill("s1", pub[2].seq):
        seen.append(seq)
        served = seq
        if not injected:
            injected = True
            pub.append(await bus.publish(Observation(
                agent_id="a", session_id="s1", observation_type="tool_result",
                tool_name="read", tool_call_id="mid", output="mid",
            )))

    while not client.queue.empty():                    # the SSE loop's own de-dup, applied here
        _, seq, _payload = client.queue.get_nowait()
        if seq is not None and served is not None and seq <= served:
            continue
        seen.append(seq)
        served = seq

    expected = {e.seq for e in pub if e.seq >= pub[2].seq}
    assert seen == sorted(set(seen)), f"duplicated or out of order: {seen}"
    assert set(seen) == expected, f"missing {expected - set(seen)}, extra {set(seen) - expected}"


async def test_connecting_does_not_start_a_cold_model_server(tmp_path):
    """§6.1's guardrail: a phone in a pocket must not be able to spin the GPU.

    A connect is a much weaker signal of intent than a message — a backgrounded page reconnects on
    its own — and bring-up starts a harness-managed model server when one is configured. So the
    connect path pre-warms only when the provider already answers, which is the normal case on a
    warm box and free there. On this machine that is not a style preference: concurrent prefills
    caused a hard system freeze.
    """
    _, channel, server, _ = await _stack(tmp_path)
    channel.session_id = None
    started: list[str] = []
    server.on_first_message = lambda: started.append("bring-up")

    channel._model_reachable = False
    server.channel.probe_model = _fixed(False)        # type: ignore[method-assign]
    await server._maybe_prewarm()
    assert started == [], "a connect started bring-up against a cold model server"

    # ...but an actual message always does, because a human asked for something.
    assert server._ensure_session() is True
    assert started == ["bring-up"]


async def test_connecting_does_prewarm_a_warm_box(tmp_path):
    _, channel, server, _ = await _stack(tmp_path)
    channel.session_id = None
    started: list[str] = []
    server.on_first_message = lambda: started.append("bring-up")
    server.channel.probe_model = _fixed(True)         # type: ignore[method-assign]
    await server._maybe_prewarm()
    assert started == ["bring-up"]


def _fixed(value):
    async def _probe():
        return value
    return _probe


async def test_the_collapse_rule_itemizes_the_measured_tool_mix_on_the_REAL_registry(tmp_path):
    """WEBCH-31/04 against the real thing, not a fake.

    Every other test here builds a stub registry, which proves the plumbing and nothing about the
    rule. The measured corpus says 60.8% of real tool calls MUTATE and `bash_exec` alone is 45.5%
    — so what has to hold is that the actual built-in tools land outside the collapsible
    allowlist. A fake registry cannot tell you that; a real one that grew a new dangerous tool in
    a collapsible group would fail here the day it landed.
    """
    from localharness.channels.web.protocol import COLLAPSIBLE_GROUPS
    from localharness.channels.web.server import _tool_rows
    from localharness.tools.builtin import register_builtin_tools
    from localharness.tools.registry import ToolRegistry

    registry = ToolRegistry()
    await register_builtin_tools(registry)
    rows = _tool_rows(registry)
    assert rows, "the real registry produced no tools — the walk is reading the wrong buckets"

    itemized = {r["name"] for r in rows if r["group"] not in COLLAPSIBLE_GROUPS}
    assert {"bash_exec", "edit", "write"} <= itemized, (
        f"a mutating built-in fell into a collapsible group; itemized = {sorted(itemized)}"
    )
    # Every destructive tool is itemized, whatever it is called.
    assert {r["name"] for r in rows if r["destructive"]} <= itemized
    assert {"read", "grep", "glob"} <= {r["name"] for r in rows}


async def test_last_event_id_resumes_AFTER_the_event_it_names(tmp_path):
    """The commonest reconnect there is, and it duplicated an event.

    `?from=N` is the page's own cursor and already means "from N inclusive" — it computes it as
    `last_seen + 1`. `Last-Event-ID: N` is the BROWSER's, attached automatically on
    `EventSource`'s auto-reconnect after any wifi blip with no app code involved, and names the
    last event RECEIVED. Treating both as inclusive re-delivered one event on every such
    reconnect — and the client is not idempotent about it: a repeated `Action(tool_call)`
    overwrites the call map, orphaning the first row at "waiting…" forever.

    Invisible to the cold-relaunch test, which sets `?from=`. Found by adversarial review.
    """
    bus, channel, server, _ = await _stack(tmp_path)
    pub = [await bus.publish(Observation(
        agent_id="a", session_id="s1", observation_type="tool_result",
        tool_name="read", tool_call_id=f"c{i}", output=f"o{i}",
    )) for i in range(4)]

    frames = await _read_frames(server, 2, headers={
        "Authorization": f"Bearer {TOKEN}", "Last-Event-ID": str(pub[1].seq),
    })
    seqs = [f[1] for f in frames if f[1] is not None]
    assert pub[1].seq not in seqs, "re-served the event the browser said it already had"
    assert seqs[0] == pub[2].seq

    # ...and an explicit ?from= is still INCLUSIVE, because the page already added the one.
    frames = await _read_frames(server, 2, path=f"/api/stream?from={pub[1].seq}")
    assert [f[1] for f in frames if f[1] is not None][0] == pub[1].seq


async def test_every_response_refuses_to_be_framed(tmp_path):
    """The shell is unauthenticated by design and inert without a token — but `localStorage` is
    scoped to the ORIGIN, not to the frame embedding it. Without a framing rule a malicious page
    can iframe this origin, inherit an enrolled session, and UI-redress BOTH taps of the
    `_always` confirm: defeating by clicks the one control built so a single request cannot write
    a permanent, global, unrevokable grant."""
    _, _, _, client = await _stack(tmp_path)
    for path, headers in (("/", {}), ("/api/health", BEARER), ("/api/nope", BEARER)):
        got = await client.get(path, headers=headers)
        assert got.headers["x-frame-options"] == "DENY", path
        assert "frame-ancestors 'none'" in got.headers["content-security-policy"], path
        assert got.headers["x-content-type-options"] == "nosniff", path
        assert got.headers["referrer-policy"] == "no-referrer", path
