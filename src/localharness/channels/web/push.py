"""Web Push (RFC 8291/8292): the keys, the subscriptions, the policy, and the sender.

**Why Web Push and not a notification service.** iOS grants push to a Home-Screen-installed PWA
and shows it on the lock screen with the app closed, and a private tailnet origin does not break
that: the subscription is negotiated by the BROWSER directly with Apple over the phone's own
internet path, and sending is outbound-only from this box to the push service. Nothing here ever
needs to be publicly reachable. Self-hosted ntfy, the obvious alternative, still relays through
ntfy.sh to reach APNs — so the "avoid a third party" argument actually favours Web Push, whose
payloads are end-to-end encrypted to the subscription endpoint and unreadable by the relay.

**Why these two libraries and not `pywebpush`.** `py_vapid` and `http_ece` ARE the crypto core of
pywebpush — it imports both. What pywebpush adds on top is a transport, and it brings `requests`
AND `aiohttp` to provide it: 21 installed packages here against 6. This harness already
standardized on httpx, which is async (this is called from the event loop, mid-turn) and which
tests drive through `httpx.MockTransport` — so we take the encoder and keep our own transport.

**What is NOT here, deliberately.** No silent push (iOS forbids it and every payload below shows
something), and no inline notification actions: whether iOS Web Push supports them is unverified
and no surveyed product ships reliable approve-from-the-notification. A push says "come back and
look" and carries WHERE to look; the approving happens in the app.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

import structlog

from localharness.channels.web import auth

log = structlog.get_logger(__name__)


# --------------------------------------------------------------------- constants

MEASURED_MEDIAN_TURN_S = 107.0
"""The owner's measured median turn, from the corpus study in the PRD's §2.4.

Recorded as a measurement with a date rather than a tuning knob: every threshold below derives
from it, so when the box or the model changes, ONE number moves and the rest follow.
"""

MEASURED_LONG_TURN_S = 600.0
"""Ten minutes — the tail of the same measurement: 15.1% of real turns run past it."""

TURN_PUSH_MIN_S = MEASURED_MEDIAN_TURN_S
"""Below this, a finished turn is not worth a notification.

A turn shorter than the median is one you could have stood and waited for; buzzing for it is how
a phone earns its notifications being switched off — and then the permission badge reaches
nobody, which is the failure §5.5 exists to prevent.
"""

NEEDS_YOU_COOLDOWN_S = MEASURED_LONG_TURN_S
"""One alerting "needs you" per session per window.

Derived, not picked: the window is the length of a LONG turn, so a single turn can never buzz
twice no matter how many calls it parks. §5.5's worked example is exactly this — three parked
calls in ten minutes must be one buzz with a badge of three.
"""

PUSH_TTL_S = int(MEASURED_LONG_TURN_S)
"""How long the push service should hold an undelivered message.

Derived from the long-turn horizon for a reason: this is a "come back and look" notification
about a specific live moment. A phone that was off for an hour should be told what is true now,
by opening the app, not handed a buzz about a question that timed out forty minutes ago.
"""

PUSH_URGENCY = "high"
"""RFC 8030 urgency. Both classes are things a person is waiting on — a finished turn or a
blocked one — and `high` is what asks the push service not to batch it behind a power-saving
window. Not `very-low`: nothing here is a background sync."""

PUSH_TIMEOUT_S = 10.0
"""Socket budget for one push. It runs on the event loop during a turn, so it gets a bound; a
push service that hangs must not hold up the harness."""

MAX_BODY_CHARS = 160
"""Notification body length. Lock screens truncate around here anyway, and the encrypted record
below has a fixed size — a 40 KB tool error pasted into a notification body would simply fail to
encode, so it is cut where it is composed instead of failing at the transport."""

MAX_SUBSCRIPTIONS = 16
"""How many devices may be enrolled at once, oldest evicted first.

There is no per-device revoke (§7.2's named gap), so without a cap a store grows one dead row per
reinstall forever, and every push pays for all of them. Sixteen is well past one household's
phones and tablets.
"""

VAPID_SUBJECT = "https://localharness.dev"
"""The `sub` claim of the VAPID JWT — RFC 8292 requires a `mailto:` or `https:` URI that a push
service operator could use to contact whoever is sending. The project's own homepage is the
honest answer; inventing a `mailto:` for a local box would be putting a fake address in a header
that exists to be real."""

SUBSCRIPTION_ERROR = (
    "not a usable PushSubscription: it needs an https endpoint plus both keys (`p256dh`, the "
    "P-256 point the payload is encrypted to, and `auth`, the secret mixed into that "
    "derivation). Post what `pushManager.subscribe().toJSON()` returns."
)

CLASS_TURN_FINISHED = "turn_finished"
CLASS_NEEDS_YOU = "needs_you"

# The two — and only two — classes that may reach a lock screen. A phone that buzzes on every
# tool call gets notifications turned off. Kept as a tuple so a third one cannot be added by
# accident somewhere else in the file.
PUSH_CLASSES: tuple[str, ...] = (CLASS_TURN_FINISHED, CLASS_NEEDS_YOU)


def vapid_path(config_dir: Optional[str | Path] = None) -> Path:
    """`<global config dir>/web/vapid.pem` — beside the app token, and global for the same
    reason: one listener per machine, and the phone enrolled against it does not know which
    project directory the process was started in."""
    from localharness.config.paths import global_config_dir

    return global_config_dir(config_dir) / "web" / "vapid.pem"


def subscriptions_path(config_dir: Optional[str | Path] = None) -> Path:
    """`<global config dir>/web/push-subscriptions.json`."""
    from localharness.config.paths import global_config_dir

    return global_config_dir(config_dir) / "web" / "push-subscriptions.json"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


# --------------------------------------------------------------------- the keys


@dataclass(frozen=True)
class VapidKeys:
    """The server's identity to the push services. Generated here, never fetched."""

    private_pem: bytes
    application_server_key: str
    """The public key as a raw uncompressed P-256 point, base64url — the exact shape
    `pushManager.subscribe({applicationServerKey})` takes in the browser."""


def load_or_create_vapid(config_dir: Optional[str | Path] = None) -> VapidKeys:
    """The VAPID keypair, generated on first use (WEBCH-27).

    Generated by the harness rather than configured, because a setup step that sends a new user
    to an external site to mint push keys is a setup step that does not happen. Written `0600`
    through a mode-restricted descriptor for the same reason the app token is: every local
    process on this box can reach the loopback port.

    **Stability is the contract.** Every enrolled phone's subscription is bound to this public
    key, so regenerating it silently breaks push for every device — hence read-then-create, never
    create-then-overwrite.
    """
    from py_vapid import Vapid02

    path = vapid_path(config_dir)
    if path.exists():
        try:
            return _keys_from_pem(path.read_bytes())
        except Exception:  # noqa: BLE001 — a corrupt key file is replaced, not fatal
            log.warning("web_vapid_unreadable", path=str(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    vapid = Vapid02()
    vapid.generate_keys()
    pem = vapid.private_pem()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, auth.TOKEN_FILE_MODE)
    try:
        os.write(fd, pem)
    finally:
        os.close(fd)
    return _keys_from_pem(pem)


def _keys_from_pem(pem: bytes) -> VapidKeys:
    from cryptography.hazmat.primitives import serialization
    from py_vapid import Vapid02

    vapid = Vapid02.from_pem(pem)
    raw = vapid.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return VapidKeys(private_pem=pem, application_server_key=_b64(raw))


# --------------------------------------------------------------------- the store


class SubscriptionStore:
    """The enrolled devices, as a JSON file beside the token.

    A file rather than a table in the memory DB: this is deployment state (which phones has this
    machine been paired with), not session knowledge, and it must be readable and deletable by a
    person who has decided to un-enrol everything with `rm`.
    """

    def __init__(self, config_dir: Optional[str | Path] = None) -> None:
        self.path = subscriptions_path(config_dir)

    def all(self) -> list[dict]:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            log.warning("web_push_subscriptions_unreadable", path=str(self.path))
            return []
        return [r for r in rows if isinstance(r, dict) and r.get("endpoint")] if isinstance(
            rows, list) else []

    def add(self, subscription: dict) -> None:
        """Upsert on the endpoint. A PWA re-subscribes on launch whenever it suspects its
        subscription changed, and that is an update, not a second device."""
        endpoint = subscription.get("endpoint")
        if not endpoint:
            raise ValueError("a subscription needs an endpoint")
        rows = [r for r in self.all() if r.get("endpoint") != endpoint]
        rows.append({
            "endpoint": endpoint,
            "keys": subscription.get("keys") or {},
            "created_at": time.time(),
        })
        self._write(rows[-MAX_SUBSCRIPTIONS:])

    def remove(self, endpoint: str) -> None:
        rows = [r for r in self.all() if r.get("endpoint") != endpoint]
        self._write(rows)

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, auth.TOKEN_FILE_MODE)
        try:
            os.write(fd, json.dumps(rows, indent=1).encode("utf-8"))
        finally:
            os.close(fd)


def valid_subscription(body: Any) -> Optional[dict]:
    """The posted body as a storable subscription, or None if it could never be pushed to.

    Both keys are load-bearing: `p256dh` is what the payload is encrypted TO and `auth` is the
    shared secret mixed into that derivation. Storing a row without them is storing a row that
    can only fail at send time, on some later turn, invisibly.
    """
    if not isinstance(body, dict):
        return None
    endpoint = body.get("endpoint")
    keys = body.get("keys")
    if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
        return None
    if not isinstance(keys, dict):
        return None
    p256dh, secret = keys.get("p256dh"), keys.get("auth")
    if not isinstance(p256dh, str) or not isinstance(secret, str) or not p256dh or not secret:
        return None
    try:
        if len(_unb64(p256dh)) != 65:
            return None
        _unb64(secret)
    except Exception:  # noqa: BLE001 — anything unparseable is simply not a subscription
        return None
    return {"endpoint": endpoint, "keys": {"p256dh": p256dh, "auth": secret}}


# --------------------------------------------------------------------- the payload


@dataclass(frozen=True)
class Push:
    """One notification, already decided on. What the service worker receives, verbatim."""

    title: str
    body: str
    tag: str
    badge: int
    data: dict
    alert: bool
    """Whether this one should re-alert, or quietly REPLACE the notification already on the lock
    screen. Same `tag` plus `renotify: false` is how one buzz ends up carrying a badge of three
    (§5.5): the OS swaps the visible notification without making a sound."""

    def to_json(self) -> bytes:
        return json.dumps({
            "title": self.title,
            "body": self.body[:MAX_BODY_CHARS],
            "tag": self.tag,
            "badge": self.badge,
            "renotify": self.alert,
            "data": self.data,
        }).encode("utf-8")


def deep_link(session_id: Optional[str], **item: Any) -> str:
    """Where tapping this notification lands (WEBCH-44).

    By the time push exists there is a chat list to get lost in, so a notification that drops you
    on a generic screen reintroduces the friction it was sent to remove.
    """
    params = {"session": session_id or ""}
    params.update({k: str(v) for k, v in item.items() if v is not None})
    return "/?" + urlencode(params)


# --------------------------------------------------------------------- the policy


@dataclass
class _SessionState:
    turn_started_at: Optional[float] = None
    last_alert_at: Optional[float] = None
    outstanding: set = field(default_factory=set)


class PushPolicy:
    """WHEN to push, and with what badge. No I/O, so it is testable without a network.

    Two classes reach a lock screen and nothing else does: a turn that finished while you were
    away, and something that needs you — a parked call, a blocking ask, or an `Escalation`. That
    last one is in the list because it is literally the harness's own "stuck, needs a human"
    signal, and a stuck turn burning time until the owner happens to look is exactly what push
    exists to prevent on a box where the median turn is nearly two minutes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}

    def _state(self, session_id: Optional[str]) -> _SessionState:
        return self._sessions.setdefault(session_id or "", _SessionState())

    # -- class 1: a turn finished while you were away ----------------------

    def turn_started(self, session_id: Optional[str], *, now: Optional[float] = None) -> None:
        state = self._state(session_id)
        state.turn_started_at = time.monotonic() if now is None else now
        # An escalation is the only needs-you class with NO resolution event — nothing ever
        # publishes "the stuck turn is unstuck" — so its badge unit could never come off, and a
        # box that escalates occasionally would end up permanently claiming that seven things
        # need a human. A new turn starting IS that signal. Parked calls are deliberately not
        # swept: outliving the turn that raised them is the entire point of parking one.
        state.outstanding = {key for key in state.outstanding if key[0] != "escalation"}

    def turn_finished(
        self, session_id: Optional[str], *, clients_attached: int, now: Optional[float] = None,
        summary: str = "", duration: Optional[float] = None,
    ) -> Optional[Push]:
        """A push, or None — and None is the common answer.

        Three gates, each of which has to hold:

        1. **Presence.** If a stream is attached, the owner is looking at this. The page closes
           its own stream when it goes to the background precisely so that this signal means
           "watching" rather than "has the tab open somewhere".
        2. **Duration.** A turn shorter than the measured median is one you could have waited for.
        3. **A duration we actually have.** `TurnCompleted` carries the harness's own
           `duration_seconds` and that is preferred, because this object cannot see a turn that
           began before the page connected. Failing that, the start we timed ourselves. Failing
           both, no push: an unmeasured "while you were away" is a push that fires on restart.
        """
        state = self._state(session_id)
        started, state.turn_started_at = state.turn_started_at, None
        if clients_attached > 0:
            return None
        moment = time.monotonic() if now is None else now
        elapsed = duration if duration is not None else (
            None if started is None else moment - started)
        if elapsed is None or elapsed < TURN_PUSH_MIN_S:
            return None
        return Push(
            title="turn finished",
            body=summary or f"{int(elapsed)}s of work is waiting for you.",
            tag=f"{CLASS_TURN_FINISHED}:{session_id or ''}",
            badge=len(state.outstanding),
            data={
                "class": CLASS_TURN_FINISHED,
                "session_id": session_id,
                "url": deep_link(session_id),
            },
            alert=True,
        )

    # -- class 2: something needs you --------------------------------------

    def needs_you(
        self, session_id: Optional[str], *, kind: str, now: Optional[float] = None,
        pending_id: Any = None, request_id: Optional[str] = None, detail: str = "",
    ) -> Push:
        """Always returns a push; `alert` says whether it makes a sound.

        The honest statement of the coalescing: ONE BUZZ per session per cooldown, not one
        message. A later item in the same window still goes out, with the same `tag` and
        `renotify` false, so the lock screen shows one notification whose badge has moved from
        one to three rather than three notifications — which is the behavior §5.5 asks for and
        the one that does not train an owner to disable notifications.
        """
        state = self._state(session_id)
        moment = time.monotonic() if now is None else now
        key = ("ask", request_id) if request_id is not None else ("pending", pending_id) \
            if pending_id is not None else ("escalation", moment)
        state.outstanding.add(key)

        last = state.last_alert_at
        alert = last is None or (moment - last) >= NEEDS_YOU_COOLDOWN_S
        if alert:
            state.last_alert_at = moment

        count = len(state.outstanding)
        return Push(
            title=_NEEDS_YOU_TITLES.get(kind, "needs you"),
            body=detail or _NEEDS_YOU_BODIES.get(kind, "the harness is waiting on you."),
            tag=f"{CLASS_NEEDS_YOU}:{session_id or ''}",
            badge=count,
            data={
                "class": CLASS_NEEDS_YOU,
                "kind": kind,
                "session_id": session_id,
                "pending_id": pending_id,
                "request_id": request_id,
                "url": deep_link(session_id, pending=pending_id, ask=request_id),
            },
            alert=alert,
        )

    def resolved(self, session_id: Optional[str], *, pending_id: Any = None,
                 request_id: Optional[str] = None) -> None:
        """Something was answered — from ANY surface. The badge counts what is outstanding, not
        how many pushes were sent, so answering in Discord has to clear it here too."""
        state = self._state(session_id)
        state.outstanding.discard(("ask", request_id) if request_id is not None
                                  else ("pending", pending_id))

    def outstanding(self, session_id: Optional[str]) -> int:
        return len(self._state(session_id).outstanding)


_NEEDS_YOU_TITLES = {
    "parked": "a call is parked",
    "blocking": "permission needed",
    "escalation": "stuck — needs you",
}

_NEEDS_YOU_BODIES = {
    "parked": "a tool call is waiting for your approval.",
    "blocking": "the turn is blocked on a question.",
    "escalation": "the harness asked for a human.",
}


# --------------------------------------------------------------------- the sender


class PushSender:
    """Encrypt to one subscription and POST it, over httpx.

    Split from :class:`PushService` so the crypto can be driven in a test with a
    `MockTransport`: the tests simulate the BROWSER with the subscription's own private key and
    decrypt what went out, so a payload no browser could read fails here rather than on a phone.
    """

    def __init__(self, keys: VapidKeys, *, transport: Any = None,
                 subject: str = VAPID_SUBJECT) -> None:
        self.keys = keys
        self.subject = subject
        self._transport = transport

    async def send(self, subscription: dict, message: Push) -> Optional[bool]:
        """True on delivery, False if the endpoint is GONE and should be pruned, None if the
        attempt failed in a way that says nothing about the subscription."""
        import httpx

        endpoint = subscription["endpoint"]
        try:
            body = self._encrypt(subscription, message.to_json())
            headers = self._headers(endpoint, len(body))
        except Exception:  # noqa: BLE001 — a bad stored subscription must not kill the turn
            log.warning("web_push_encode_failed", endpoint=endpoint[:60], exc_info=True)
            return False
        try:
            async with httpx.AsyncClient(transport=self._transport,
                                         timeout=PUSH_TIMEOUT_S) as client:
                response = await client.post(endpoint, content=body, headers=headers)
        except Exception:  # noqa: BLE001 — no network is not evidence the phone is gone
            log.info("web_push_send_failed", endpoint=endpoint[:60], exc_info=True)
            return None
        if response.status_code in (404, 410):
            return False
        if response.status_code >= 400:
            log.info("web_push_rejected", status=response.status_code, endpoint=endpoint[:60])
            return None
        return True

    def _encrypt(self, subscription: dict, payload: bytes) -> bytes:
        """RFC 8291 aes128gcm. A FRESH ephemeral key per message, which the standard requires and
        which is why this is not hoisted out of the send path."""
        import http_ece
        from cryptography.hazmat.primitives.asymmetric import ec

        keys = subscription.get("keys") or {}
        return http_ece.encrypt(
            payload,
            salt=None,
            private_key=ec.generate_private_key(ec.SECP256R1()),
            dh=_unb64(keys["p256dh"]),
            auth_secret=_unb64(keys["auth"]),
            version="aes128gcm",
        )

    def _headers(self, endpoint: str, length: int) -> dict[str, str]:
        from py_vapid import Vapid02

        parsed = urlparse(endpoint)
        vapid = Vapid02.from_pem(self.keys.private_pem)
        signed = vapid.sign({
            "aud": f"{parsed.scheme}://{parsed.netloc}",
            "sub": self.subject,
        })
        headers = {
            "Content-Encoding": "aes128gcm",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(length),
            "TTL": str(PUSH_TTL_S),
            "Urgency": PUSH_URGENCY,
        }
        headers.update(signed)
        return headers


class PushService:
    """Store + sender + policy, and the one method the channel calls.

    Deliberately forgiving: every failure path logs and returns. Push is a convenience on top of
    a turn, and a push service having a bad day must never be able to take a turn down with it.
    """

    def __init__(self, *, store: SubscriptionStore, sender: PushSender,
                 policy: Optional[PushPolicy] = None) -> None:
        self.store = store
        self.sender = sender
        self.policy = policy or PushPolicy()

    @property
    def enrolled(self) -> int:
        return len(self.store.all())

    async def deliver(self, message: Optional[Push]) -> int:
        """Fan one decided push out to every enrolled device. Returns how many were delivered."""
        if message is None:
            return 0
        sent = 0
        for subscription in self.store.all():
            outcome = await self.sender.send(subscription, message)
            if outcome is True:
                sent += 1
            elif outcome is False:
                # The push service said this subscription is gone — an uninstalled app, a reset
                # phone. Keeping it means signing and encrypting for a corpse on every turn.
                self.store.remove(subscription["endpoint"])
                log.info("web_push_pruned", endpoint=subscription["endpoint"][:60])
        return sent

    @classmethod
    def build(cls, config_dir: Optional[str | Path] = None,
              policy: Optional[PushPolicy] = None) -> "PushService":
        keys = load_or_create_vapid(config_dir)
        return cls(store=SubscriptionStore(config_dir), sender=PushSender(keys), policy=policy)
