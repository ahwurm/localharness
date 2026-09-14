"""The app half: manifest, service worker, install sequencing and the QR pairing fragment.

The page assertions are string checks against the shipped file, in the style of
`test_web_reference_page.py`. There is no browser here, so what these can prove is that the
GUARDS are present and phrased the only way that works — which is precisely the class of bug
that ships silently: a service worker registered on plain HTTP, a notification permission asked
before install (iOS then grants a subscription that can never deliver), or a token handed to an
unauthenticated caller.
"""
from __future__ import annotations

import json

import httpx
import pytest

from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import (
    MANIFEST_CONTENT_TYPE,
    PACKAGED_UI_DIR,
    TOKEN_FRAGMENT_KEY,
    WebServer,
)
from localharness.core.bus import EventBus

TOKEN = "test-token-not-a-real-one"
JSON = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
BEARER = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def page() -> str:
    return (PACKAGED_UI_DIR / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def worker() -> str:
    return (PACKAGED_UI_DIR / "sw.js").read_text(encoding="utf-8")


def body_of(source: str, signature: str) -> str:
    """One top-level function's body, so a guard can be shown to DOMINATE a call rather than
    merely appear somewhere above it in the file.

    Written after a mutation test walked straight through the weaker version: moving
    `serviceWorker.register` out from behind the secure-context check left `isSecureContext`
    still sitting earlier in the file, and an assertion about ordering happily passed a page
    that would now register a worker over plain HTTP.
    """
    start = source.index(signature)
    end = source.index("\n}", start)
    return source[start:end]


async def _client(tmp_path):
    bus = EventBus(persist_path=tmp_path / "bus-events.jsonl")
    channel = WebChannel(bus=bus, config={})
    await channel.start()
    channel.bind_runtime(session_id="s1", agent_id="orchestrator", session_dir=tmp_path / "s")
    server = WebServer(channel, token=TOKEN, config_dir=tmp_path)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://web.test"
    )


# ---------------------------------------------------------------- the manifest

async def test_the_manifest_makes_the_page_installable(tmp_path):
    client = await _client(tmp_path)
    response = await client.get("/manifest.webmanifest")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(MANIFEST_CONTENT_TYPE)
    body = response.json()
    # `standalone` is the whole point: a home-screen icon, not a Safari tab with a URL bar.
    assert body["display"] == "standalone"
    assert body["scope"] == "/"
    assert {icon["sizes"] for icon in body["icons"]} >= {"192x192", "512x512"}


async def test_an_unauthenticated_manifest_never_carries_the_token(tmp_path):
    """THE load-bearing guard on this route. The manifest is fetched by the browser without
    credentials unless the page asks otherwise, and it is reachable by anything that can reach
    the port — so a `start_url` with the token in it would publish the credential to exactly the
    caller the token exists to stop."""
    client = await _client(tmp_path)
    response = await client.get("/manifest.webmanifest")
    assert TOKEN not in response.text
    assert response.json()["start_url"] == "/"


async def test_an_authenticated_manifest_pairs_the_installed_app(tmp_path):
    """iOS gives a home-screen app a DIFFERENT storage jar from Safari's, so an app installed
    from a paired tab starts out knowing nothing. A credentialed manifest fetch is the one hook
    that can carry the token across that boundary."""
    client = await _client(tmp_path)
    response = await client.get("/manifest.webmanifest", headers=BEARER)
    assert response.json()["start_url"] == f"/#{TOKEN_FRAGMENT_KEY}={TOKEN}"
    # And it must never be cached anywhere on the way.
    assert response.headers["cache-control"] == "no-store"


async def test_the_page_asks_for_the_manifest_with_credentials(page):
    """Without `crossorigin="use-credentials"` the browser fetches the manifest anonymously and
    the pairing `start_url` above can never be served — the mechanism would be dead code."""
    assert 'rel="manifest"' in page
    link = page[page.index('rel="manifest"') - 200:page.index('rel="manifest"') + 200]
    assert 'crossorigin="use-credentials"' in link


async def test_the_icons_and_worker_are_served(tmp_path):
    client = await _client(tmp_path)
    for name in ("icon-192.png", "icon-512.png", "icon-180.png"):
        response = await client.get(f"/{name}")
        assert response.status_code == 200
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert (await client.get("/icon.svg")).status_code == 200
    assert (await client.get("/sw.js")).status_code == 200


# ---------------------------------------------------------------- HTTPS gating

async def test_the_service_worker_registers_only_in_a_secure_context(page):
    """Plain HTTP on a LAN address gets no service worker, no install and no push — the browser
    refuses, and a page that tries anyway throws an unhandled rejection instead of saying so.
    The degraded topology is documented (§7.1) and this is it, said out loud in the UI."""
    assert "window.isSecureContext" in page
    # The guard must be INSIDE the registering function and must return before it registers —
    # not merely appear earlier in the file.
    register = body_of(page, "async function registerWorker")
    assert "serviceWorker.register" in register
    guard, call = register.index("secure"), register.index("serviceWorker.register")
    assert guard < call
    assert "return" in register[guard:call], "the secure-context check must bail out, not warn"

    # And the page must tell the owner WHY, rather than silently lacking a button.
    reason = body_of(page, "function pushReason")
    assert "secure" in reason and "https" in reason.lower()


async def test_the_worker_does_not_cache_the_shell(worker):
    """WIN-B (WEBCH-20) is 'edit the page, pull to refresh, see it'. A precaching service worker
    breaks that silently, and it is the single commonest thing a service worker is copied in to
    do — so the absence is asserted, not just intended."""
    assert "caches.open" not in worker
    assert "addEventListener(\"fetch\"" not in worker
    assert "addEventListener('fetch'" not in worker


# ---------------------------------------------------------------- push enrolment order

async def test_notification_permission_cannot_be_requested_before_install(page):
    """WEBCH-16, and the acceptance criterion says *by construction*. iOS grants push only to a
    PWA already on the Home Screen; asking first produces a subscription that can never deliver
    and burns the one permission prompt the owner will ever see."""
    assert "requestPermission" in page
    # The ONLY place permission is requested, and the installed check has to dominate it inside
    # that same function — a check that merely exists elsewhere in the file is decoration.
    assert page.count("Notification.requestPermission") == 1
    enable = body_of(page, "async function enablePush")
    assert "requestPermission" in enable
    gate, ask = enable.index("pushReason()"), enable.index("requestPermission")
    assert gate < ask
    assert "return" in enable[gate:ask], "a blocked reason must bail out before asking"

    reason = body_of(page, "function pushReason")
    assert "installed()" in reason
    assert "return" in reason[reason.index("installed()"):], "the install check must bail out"
    assert "display-mode: standalone" in page and "navigator.standalone" in page


async def test_the_page_subscribes_with_the_key_the_server_generated(page):
    """Never a key pasted into the page: the harness generates the VAPID pair and serves the
    public half, so rotating it is a server-side act."""
    assert "/api/push/key" in page
    assert "applicationServerKey" in page
    assert "/api/push/subscribe" in page


# ---------------------------------------------------------------- QR pairing

async def test_the_page_pairs_from_the_url_fragment_and_then_erases_it(page):
    """The QR encodes `https://host/#t=<token>`. A fragment reaches no server and lands in no
    log — but it would sit in the address bar and in history, so it is read once and stripped."""
    assert f"#{TOKEN_FRAGMENT_KEY}=" in page or f'"{TOKEN_FRAGMENT_KEY}"' in page
    assert "location.hash" in page
    assert "replaceState" in page
    # Stripping has to happen in the same breath as reading it.
    read = page.index("location.hash")
    assert "replaceState" in page[read:read + 900]


async def test_a_deep_link_is_honoured_on_open(page):
    """Tapping a push lands on `/?session=…&pending=…`; the page has to READ those parameters
    and open the thing they name, or the deep link is decoration on a generic screen."""
    assert "URLSearchParams" in page
    where = page.index("function followDeepLink")
    window = page[where:where + 1400]
    assert "URLSearchParams" in page[where - 600:where + 1400] or "params.get" in window
    assert '"pending"' in window or "'pending'" in window
    assert '"ask"' in window or "'ask'" in window
    # It has to actually open something: the parked queue or the blocking-ask modal.
    assert "showQueue" in window or "showAsk" in window


# ---------------------------------------------------------------- cold and unreachable

async def test_the_unreachable_state_offers_a_way_forward(page):
    """WEBCH-26: neither 'cold' nor 'unreachable' may be an indefinite spinner, and 'unreachable'
    must not be a dead end either — nothing re-checks a TCP probe on its own."""
    assert "unreachable" in page
    # Saying "start it, then retry" while offering nothing to retry WITH is the dead end: the
    # model probe only re-runs when something asks it to, so the page needs a control that asks.
    assert 'id="recheck"' in page, "no control to re-run the probe with"
    handler = page.index('$("recheck")')
    assert "/api/health" in page[handler:handler + 700]


async def test_the_bringup_ticker_names_its_stage_and_can_be_abandoned(page):
    """WEBCH-43. A rising number reports elapsed time, not health."""
    assert "BringUpStage" in page
    where = page.index('case "BringUpStage"')
    window = page[where:where + 1200]
    assert "d.stage" in window
    assert "/api/bringup/abort" in window


async def test_the_manifest_json_is_valid(tmp_path):
    """Cheap, and it catches the class of typo that makes iOS silently refuse to install."""
    client = await _client(tmp_path)
    body = json.loads((await client.get("/manifest.webmanifest")).text)
    assert body["name"] and body["short_name"]
    assert body["start_url"].startswith("/")


async def test_the_page_shows_a_co_tenant_session(page):
    """WEBCH-29's other half: a phone that connected after the warning was sent still has to
    learn that a terminal is driving the same agent."""
    assert "other_sessions" in page
    where = page.index("other_sessions")
    assert "another session is live" in page[where:where + 500]
