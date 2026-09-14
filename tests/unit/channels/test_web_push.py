"""Web Push: the keys, the subscription store, the policy, and the one guard that matters.

Nothing here sends a real push. The transport is `httpx.MockTransport`, which is the point:
a test that reached Apple or Mozilla would be a test that fails on a plane, and a test that
"passes" by silently not sending would be worse. So the sender is driven for real and the
socket is the only thing faked — the ciphertext, the VAPID header and the pruning of a dead
endpoint are all asserted against what actually went out.
"""
from __future__ import annotations

import base64
import json

import httpx

from localharness.channels.web import auth, push
from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import WebServer
from localharness.core.bus import EventBus

TOKEN = "test-token-not-a-real-one"
JSON = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
BEARER = {"Authorization": f"Bearer {TOKEN}"}


def _subscription(endpoint: str = "https://push.example/aaa") -> dict:
    """A PushSubscription shaped the way `pushManager.subscribe().toJSON()` shapes one."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    priv = ec.generate_private_key(ec.SECP256R1())
    raw = priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()  # noqa: E731
    return {
        "endpoint": endpoint,
        "keys": {"p256dh": b64(raw), "auth": b64(b"0123456789abcdef")},
    }


async def _stack(tmp_path, **kw):
    bus = EventBus(persist_path=tmp_path / "bus-events.jsonl")
    channel = WebChannel(bus=bus, config={})
    await channel.start()
    channel.bind_runtime(session_id="s1", agent_id="orchestrator", session_dir=tmp_path / "s")
    server = WebServer(channel, token=TOKEN, config_dir=tmp_path, **kw)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://web.test"
    )
    return bus, channel, server, client


# ---------------------------------------------------------------- the guard

async def test_a_push_endpoint_cannot_be_registered_without_the_app_token(tmp_path):
    """THE load-bearing guard. An unauthenticated subscribe is a stranger asking to be told,
    forever, every time this box needs a human — with a deep link to the thing it should
    approve. It is a notification channel into the owner's lock screen, so it is gated exactly
    like every other verb."""
    _, _, server, client = await _stack(tmp_path)
    sub = _subscription()

    no_creds = await client.post("/api/push/subscribe", json=sub)
    assert no_creds.status_code == 401

    wrong = await client.post(
        "/api/push/subscribe", json=sub,
        headers={"Content-Type": "application/json", "Authorization": "Bearer not-the-token"},
    )
    assert wrong.status_code == 401

    # A cross-origin <form> can send this content type with no preflight; the bearer alone must
    # not be enough to make it a valid subscribe.
    form = await client.post(
        "/api/push/subscribe", content=json.dumps(sub),
        headers={"Content-Type": "text/plain", "Authorization": f"Bearer {TOKEN}"},
    )
    assert form.status_code == 415

    # And nothing was written by any of the three.
    assert push.SubscriptionStore(tmp_path).all() == []

    ok = await client.post("/api/push/subscribe", json=sub, headers=JSON)
    assert ok.status_code == 200
    assert [s["endpoint"] for s in push.SubscriptionStore(tmp_path).all()] == [sub["endpoint"]]


async def test_the_application_server_key_is_served_only_to_an_authenticated_caller(tmp_path):
    _, _, server, client = await _stack(tmp_path)
    assert (await client.get("/api/push/key")).status_code == 401
    body = (await client.get("/api/push/key", headers=BEARER)).json()
    # A raw uncompressed P-256 point, base64url, unpadded — what `applicationServerKey` takes.
    assert len(base64.urlsafe_b64decode(body["application_server_key"] + "==")) == 65


async def test_subscribing_twice_from_one_phone_stores_one_row(tmp_path):
    """A PWA re-subscribes on every launch it thinks its subscription may have changed. Keyed
    on the endpoint, that is an update, not a second device."""
    _, _, server, client = await _stack(tmp_path)
    sub = _subscription()
    await client.post("/api/push/subscribe", json=sub, headers=JSON)
    await client.post("/api/push/subscribe", json=sub, headers=JSON)
    assert len(push.SubscriptionStore(tmp_path).all()) == 1

    await client.post("/api/push/subscribe", json=_subscription("https://push.example/bbb"),
                      headers=JSON)
    assert len(push.SubscriptionStore(tmp_path).all()) == 2


async def test_a_subscription_missing_its_keys_is_refused(tmp_path):
    """Without `p256dh` and `auth` there is nothing to encrypt to, so storing it would be
    storing a row that can only ever fail at send time."""
    _, _, server, client = await _stack(tmp_path)
    bad = await client.post("/api/push/subscribe", json={"endpoint": "https://push.example/x"},
                            headers=JSON)
    assert bad.status_code == 400
    assert push.SubscriptionStore(tmp_path).all() == []


# ---------------------------------------------------------------- the keys

def test_the_harness_generates_its_own_vapid_keypair(tmp_path):
    """WEBCH-27: a new user is never sent to an external site to mint push keys."""
    keys = push.load_or_create_vapid(tmp_path)
    assert push.vapid_path(tmp_path).exists()
    assert len(base64.urlsafe_b64decode(keys.application_server_key + "==")) == 65
    # Stable across restarts — a regenerated key silently invalidates every enrolled phone.
    assert push.load_or_create_vapid(tmp_path).application_server_key == keys.application_server_key


def test_the_vapid_private_key_is_not_world_readable(tmp_path):
    push.load_or_create_vapid(tmp_path)
    mode = push.vapid_path(tmp_path).stat().st_mode & 0o777
    assert mode == auth.TOKEN_FILE_MODE


# ---------------------------------------------------------------- the policy

def _policy() -> push.PushPolicy:
    return push.PushPolicy()


def test_a_finishing_turn_does_not_buzz_a_phone_that_is_watching():
    """WEBCH-16: with the page open and visible, a finishing turn produces no push."""
    p = _policy()
    p.turn_started("s1", now=0.0)
    assert p.turn_finished("s1", clients_attached=1, now=1000.0) is None


def test_a_finishing_turn_buzzes_a_phone_that_is_away():
    p = _policy()
    p.turn_started("s1", now=0.0)
    out = p.turn_finished("s1", clients_attached=0, now=1000.0)
    assert out is not None
    assert out.data["session_id"] == "s1"
    assert out.data["class"] == push.CLASS_TURN_FINISHED
    assert out.alert is True


def test_a_turn_shorter_than_the_threshold_is_not_worth_a_notification():
    """A turn you could have waited for is not news. The threshold is the measured median turn,
    not a number someone liked the look of."""
    p = _policy()
    p.turn_started("s1", now=0.0)
    assert p.turn_finished("s1", clients_attached=0, now=push.TURN_PUSH_MIN_S - 1) is None
    p.turn_started("s2", now=0.0)
    assert p.turn_finished("s2", clients_attached=0, now=push.TURN_PUSH_MIN_S + 1) is not None


def test_a_turn_that_was_never_seen_to_start_does_not_push():
    """No start means no duration, and a push whose "while you were away" is unmeasured is a
    push that fires on process restart."""
    assert _policy().turn_finished("s1", clients_attached=0, now=9999.0) is None


def test_the_first_thing_needing_a_human_buzzes():
    p = _policy()
    out = p.needs_you("s1", kind="parked", now=0.0, pending_id=7)
    assert out.alert is True
    assert out.badge == 1
    assert out.data["pending_id"] == 7
    assert out.data["class"] == push.CLASS_NEEDS_YOU


def test_three_parked_calls_in_ten_minutes_buzz_once_and_the_badge_carries_the_rest():
    """§5.5's own worked example. A phone that buzzes three times is a phone whose owner turns
    notifications off — and then the permission badge reaches nobody."""
    p = _policy()
    first = p.needs_you("s1", kind="parked", now=0.0, pending_id=1)
    second = p.needs_you("s1", kind="parked", now=60.0, pending_id=2)
    third = p.needs_you("s1", kind="parked", now=120.0, pending_id=3)

    assert [x.alert for x in (first, second, third)] == [True, False, False]
    assert [x.badge for x in (first, second, third)] == [1, 2, 3]
    # One notification on the lock screen, not three: same tag, and only the first re-alerts.
    assert len({x.tag for x in (first, second, third)}) == 1


def test_the_cooldown_reopens_once_the_long_turn_horizon_has_passed():
    p = _policy()
    p.needs_you("s1", kind="parked", now=0.0, pending_id=1)
    later = p.needs_you("s1", kind="parked", now=push.NEEDS_YOU_COOLDOWN_S + 1, pending_id=2)
    assert later.alert is True


def test_two_sessions_do_not_share_a_cooldown():
    p = _policy()
    assert p.needs_you("s1", kind="parked", now=0.0, pending_id=1).alert is True
    assert p.needs_you("s2", kind="blocking", now=1.0, request_id="r1").alert is True


def test_resolving_everything_clears_the_badge_and_lets_the_next_ask_buzz():
    """The badge counts what is OUTSTANDING, not how many pushes were sent. An owner who
    answered everything and walks away must be buzzable again."""
    p = _policy()
    p.needs_you("s1", kind="parked", now=0.0, pending_id=1)
    p.needs_you("s1", kind="parked", now=1.0, pending_id=2)
    p.resolved("s1", pending_id=1)
    p.resolved("s1", pending_id=2)
    assert p.outstanding("s1") == 0
    nxt = p.needs_you("s1", kind="parked", now=2.0, pending_id=3)
    assert nxt.badge == 1


def test_an_escalation_is_a_needs_you():
    """`Escalation` is the harness's own "stuck, needs a human" signal. A stuck turn burning
    time silently until the owner happens to look is precisely what push exists to prevent."""
    out = _policy().needs_you("s1", kind="escalation", now=0.0)
    assert out is not None and out.data["class"] == push.CLASS_NEEDS_YOU


def test_every_payload_deep_links_to_the_thing_it_is_about():
    """WEBCH-44. A notification that drops you on a generic screen reintroduces the friction it
    was sent to remove."""
    p = _policy()
    parked = p.needs_you("s1", kind="parked", now=0.0, pending_id=7)
    assert parked.data["url"].startswith("/?")
    assert "session=s1" in parked.data["url"] and "pending=7" in parked.data["url"]

    blocking = p.needs_you("s2", kind="blocking", now=0.0, request_id="req-9")
    assert "session=s2" in blocking.data["url"] and "ask=req-9" in blocking.data["url"]

    p.turn_started("s3", now=0.0)
    done = p.turn_finished("s3", clients_attached=0, now=1000.0)
    assert "session=s3" in done.data["url"]


# ---------------------------------------------------------------- the sender

async def test_the_sender_encrypts_to_the_subscription_and_signs_with_vapid(tmp_path):
    """Drive the real encoder; fake only the socket. The browser is simulated with the
    subscription's own private key, so a payload the browser could not decrypt fails here."""
    import http_ece
    from cryptography.hazmat.primitives.asymmetric import ec

    from cryptography.hazmat.primitives import serialization

    browser_priv = ec.generate_private_key(ec.SECP256R1())
    raw = browser_priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()  # noqa: E731
    auth_secret = b"0123456789abcdef"
    sub = {"endpoint": "https://push.example/aaa",
           "keys": {"p256dh": b64(raw), "auth": b64(auth_secret)}}

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201)

    keys = push.load_or_create_vapid(tmp_path)
    sender = push.PushSender(keys, transport=httpx.MockTransport(handler))
    payload = push.Push(title="done", body="a turn finished", tag="t", badge=1,
                        data={"session_id": "s1", "url": "/?session=s1"}, alert=True)
    assert await sender.send(sub, payload) is True

    assert len(seen) == 1
    sent = seen[0]
    assert str(sent.url) == sub["endpoint"]
    assert sent.headers["content-encoding"] == "aes128gcm"
    assert sent.headers["authorization"].startswith("vapid t=")
    assert f"k={keys.application_server_key}" in sent.headers["authorization"]
    assert sent.headers["ttl"] == str(push.PUSH_TTL_S)
    # Every iOS push must be user-visible; nothing here is silent.
    assert sent.headers["urgency"] == push.PUSH_URGENCY

    body = http_ece.decrypt(sent.content, private_key=browser_priv,
                            auth_secret=auth_secret, version="aes128gcm")
    decoded = json.loads(body)
    assert decoded["title"] == "done"
    assert decoded["data"]["url"] == "/?session=s1"


async def test_a_dead_endpoint_is_pruned_rather_than_retried_forever(tmp_path):
    """404/410 is the push service saying the subscription is gone — an uninstalled app, a
    reset phone. Keeping it means signing and encrypting for a corpse on every turn."""
    store = push.SubscriptionStore(tmp_path)
    sub = _subscription()
    store.add(sub)
    keys = push.load_or_create_vapid(tmp_path)
    sender = push.PushSender(keys, transport=httpx.MockTransport(lambda r: httpx.Response(410)))

    service = push.PushService(store=store, sender=sender, policy=push.PushPolicy())
    await service.deliver(push.Push(title="x", body="y", tag="t", badge=1, data={}, alert=True))

    assert store.all() == []


async def test_a_transient_push_failure_keeps_the_subscription(tmp_path):
    """A 500 from the push service, or no network at all, is not evidence the phone is gone."""
    store = push.SubscriptionStore(tmp_path)
    store.add(_subscription())
    keys = push.load_or_create_vapid(tmp_path)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    sender = push.PushSender(keys, transport=httpx.MockTransport(boom))
    service = push.PushService(store=store, sender=sender, policy=push.PushPolicy())
    await service.deliver(push.Push(title="x", body="y", tag="t", badge=1, data={}, alert=True))

    assert len(store.all()) == 1


async def test_delivery_with_no_subscriptions_is_a_no_op_not_a_crash(tmp_path):
    """The normal state of a fresh install. Firing a trigger before any phone enrolled must not
    take the turn down with it."""
    keys = push.load_or_create_vapid(tmp_path)
    calls: list[httpx.Request] = []
    sender = push.PushSender(keys, transport=httpx.MockTransport(
        lambda r: calls.append(r) or httpx.Response(201)))
    service = push.PushService(store=push.SubscriptionStore(tmp_path), sender=sender,
                               policy=push.PushPolicy())
    await service.deliver(push.Push(title="x", body="y", tag="t", badge=0, data={}, alert=True))
    assert calls == []


# ---------------------------------------------------------------- the wiring
# "Done" is a push that a real bus event actually produces. Everything above this line could
# pass with the trigger sites unwritten, which is the failure mode these exist to close.

class _Recorder:
    """Stands in for the sender: records what would have gone out, sends nothing."""

    def __init__(self) -> None:
        self.sent: list[push.Push] = []

    async def send(self, subscription, message):
        self.sent.append(message)
        return True


async def _pushable(tmp_path):
    bus = EventBus(persist_path=tmp_path / "bus-events.jsonl")
    channel = WebChannel(bus=bus, config={})
    await channel.start()
    channel.bind_runtime(session_id="s1", agent_id="orchestrator", session_dir=tmp_path / "s")
    store = push.SubscriptionStore(tmp_path)
    store.add(_subscription())
    recorder = _Recorder()
    channel.set_push(push.PushService(store=store, sender=recorder, policy=push.PushPolicy()))
    return bus, channel, recorder


async def test_a_parked_call_on_the_bus_reaches_the_push_sender(tmp_path):
    """The whole point, end to end: the gate stages a call, and a phone in a pocket hears
    about it. A test that stopped at the policy would prove nothing about whether anything
    calls it."""
    from localharness.agent.gate_types import PendingCall
    from localharness.core.events import PermissionStaged

    bus, channel, recorder = await _pushable(tmp_path)
    pending = PendingCall(
        id=2, request=_request(), rendering="bash_exec: rm -rf ~/old-notes",
        agent_label="", session_id="s1", created_at=0.0,
    )
    await bus.publish(PermissionStaged(session_id="s1", agent_id="orchestrator", pending=pending, total=1,
                                    channel="web"))
    await channel.flush_push()

    assert len(recorder.sent) == 1
    sent = recorder.sent[0]
    assert sent.data["class"] == push.CLASS_NEEDS_YOU
    assert sent.data["pending_id"] == 2
    assert "pending=2" in sent.data["url"]
    assert "rm -rf" in sent.body


async def test_an_escalation_on_the_bus_reaches_the_push_sender(tmp_path):
    from localharness.core.events import Escalation

    bus, channel, recorder = await _pushable(tmp_path)
    await bus.publish(Escalation(session_id="s1", agent_id="orchestrator", reason="no progress for 6 iterations",
                              detail="same tool, same args, six times", iteration_at_escalation=6))
    await channel.flush_push()

    assert len(recorder.sent) == 1
    assert recorder.sent[0].data["class"] == push.CLASS_NEEDS_YOU


async def test_a_long_turn_finishing_with_nobody_attached_pushes(tmp_path):
    from localharness.core.events import TurnCompleted, TurnStarted

    bus, channel, recorder = await _pushable(tmp_path)
    await bus.publish(TurnStarted(session_id="s1", agent_id="orchestrator", task_summary="t", budget={"max_actions": 5}))
    await bus.publish(TurnCompleted(
        session_id="s1", agent_id="orchestrator", iterations=3, elapsed_tokens=100,
        duration_seconds=push.TURN_PUSH_MIN_S + 30, summary="fixed the parser",
    ))
    await channel.flush_push()

    assert len(recorder.sent) == 1
    assert recorder.sent[0].data["class"] == push.CLASS_TURN_FINISHED
    assert "fixed the parser" in recorder.sent[0].body


async def test_a_turn_finishing_while_the_phone_is_watching_pushes_nothing(tmp_path):
    """The presence gate, through the real attach path — not a hand-set integer."""
    from localharness.core.events import TurnCompleted, TurnStarted

    bus, channel, recorder = await _pushable(tmp_path)
    channel.attach_client()
    await bus.publish(TurnStarted(session_id="s1", agent_id="orchestrator", task_summary="t", budget={"max_actions": 5}))
    await bus.publish(TurnCompleted(session_id="s1", agent_id="orchestrator", iterations=1, elapsed_tokens=9,
                                 duration_seconds=9999.0, summary="done"))
    await channel.flush_push()

    assert recorder.sent == []


async def test_a_subagents_turn_finishing_does_not_push(tmp_path):
    """45% of real sessions delegate. A child turn completing mid-task is not "your task is
    done" — the same root-only rule the instrument cluster already learned the hard way."""
    from localharness.core.events import TurnCompleted, TurnStarted

    bus, channel, recorder = await _pushable(tmp_path)
    await bus.publish(TurnStarted(session_id="s1", agent_id="orchestrator", task_summary="t", budget={"max_actions": 5}))
    await bus.publish(TurnCompleted(session_id="s1", parent_id="root-1", agent_id="researcher", iterations=1,
                                 elapsed_tokens=9, duration_seconds=9999.0, summary="child"))
    await channel.flush_push()

    assert recorder.sent == []


async def test_answering_in_another_surface_clears_the_badge_here(tmp_path):
    """WEBCH-08's cross-surface rule, seen from the push side: answering in Discord publishes
    `PermissionResolved`, and the phone's badge has to come down with it."""
    from localharness.agent.gate_types import PendingCall
    from localharness.core.events import PermissionResolved, PermissionStaged

    bus, channel, recorder = await _pushable(tmp_path)
    pending = PendingCall(id=3, request=_request(), rendering="bash_exec: ls", agent_label="",
                          session_id="s1", created_at=0.0)
    await bus.publish(PermissionStaged(session_id="s1", agent_id="orchestrator", pending=pending, total=1,
                                    channel="web"))
    await bus.publish(PermissionResolved(
        session_id="s1", agent_id="orchestrator", pending_id=3, decision="allow_once",
        klass="shell", key="k", tool_name="bash_exec",
    ))
    await channel.flush_push()

    assert channel._push.policy.outstanding("s1") == 0


async def test_a_channel_with_no_push_service_still_runs_a_turn(tmp_path):
    """Push is optional — `--replay`, a box with no phone enrolled, a plain terminal user. The
    trigger sites must be inert without it rather than raising into the bus."""
    from localharness.core.events import TurnCompleted, TurnStarted

    bus = EventBus(persist_path=tmp_path / "bus-events.jsonl")
    channel = WebChannel(bus=bus, config={})
    await channel.start()
    channel.bind_runtime(session_id="s1", agent_id="orchestrator", session_dir=tmp_path / "s")
    await bus.publish(TurnStarted(session_id="s1", agent_id="orchestrator", task_summary="t", budget={"max_actions": 5}))
    await bus.publish(TurnCompleted(session_id="s1", agent_id="orchestrator", iterations=1, elapsed_tokens=9,
                                 duration_seconds=9999.0, summary="done"))
    await channel.flush_push()   # a no-op, and must not raise


async def test_a_failing_push_never_takes_the_turn_down(tmp_path):
    """A push service having a bad day is a convenience failing, not a turn failing."""
    from localharness.core.events import TurnCompleted, TurnStarted

    bus, channel, recorder = await _pushable(tmp_path)

    class _Broken:
        async def send(self, subscription, message):
            raise RuntimeError("push service on fire")

    channel._push.sender = _Broken()
    await bus.publish(TurnStarted(session_id="s1", agent_id="orchestrator", task_summary="t", budget={"max_actions": 5}))
    await bus.publish(TurnCompleted(session_id="s1", agent_id="orchestrator", iterations=1, elapsed_tokens=9,
                                 duration_seconds=9999.0, summary="done"))
    await channel.flush_push()   # must not raise


def _request():
    from localharness.agent.gate_types import PermissionRequest

    return PermissionRequest(
        tool_name="bash_exec", tool_params={"command": "rm -rf ~/old-notes"},
        klass="shell-destructive", key=None, grantable=False, reason="destructive",
        display="bash_exec: rm -rf ~/old-notes",
    )


async def test_health_carries_what_a_late_client_could_not_have_been_told(tmp_path):
    """Two facts a one-shot line on the wire cannot deliver to a phone that connects afterwards:
    how many devices are enrolled for push, and who else is driving this agent (WEBCH-29)."""
    from localharness.config.session_presence import LiveSession

    _, channel, _, client = await _stack(tmp_path)
    channel.set_co_tenants([LiveSession(
        pid=4242, agent="orchestrator", channel="terminal", workspace="/home/x/proj",
        session_id="s0", started_at=0.0,
    )])
    push.SubscriptionStore(tmp_path).add(_subscription())

    body = (await client.get("/api/health", headers=BEARER)).json()
    assert body["push_enrolled"] == 1
    assert body["other_sessions"][0]["channel"] == "terminal"
    assert body["other_sessions"][0]["pid"] == 4242


def test_escalations_do_not_inflate_the_badge_forever():
    """Found by self-critique. A parked call and a blocking ask both get a `resolved()` — an
    Escalation gets none, because nothing publishes "the stuck turn is unstuck". Keyed by the
    clock, every escalation would add a badge unit that could never come off, so a box that
    escalates occasionally ends up permanently showing "7 things need you" with nothing behind
    them. A new turn starting is the signal that the stuck one is over."""
    p = push.PushPolicy()
    p.needs_you("s1", kind="escalation", now=0.0)
    p.needs_you("s1", kind="escalation", now=1.0)
    assert p.outstanding("s1") == 2

    p.turn_started("s1", now=10.0)
    assert p.outstanding("s1") == 0


def test_a_new_turn_does_not_clear_a_parked_call():
    """The other half: parked calls OUTLIVE the turn that raised them — that is the whole point
    of parking — so the same sweep must not take them with it."""
    p = push.PushPolicy()
    p.needs_you("s1", kind="parked", now=0.0, pending_id=1)
    p.needs_you("s1", kind="escalation", now=1.0)
    p.turn_started("s1", now=10.0)
    assert p.outstanding("s1") == 1
