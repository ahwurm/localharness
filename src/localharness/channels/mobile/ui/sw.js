// The service worker. It exists for ONE reason: iOS delivers Web Push to a service worker or
// not at all. Everything else a service worker traditionally does is deliberately absent.
//
// IT DOES NOT CACHE. That is not an omission, it is the requirement: the reference page's whole
// promise is "edit the file, pull to refresh, see it" with no build step and no restart
// (WEBCH-20). A precaching worker would quietly break exactly that, and the failure mode —
// you edit the page, reload, and see yesterday's version — is the kind that costs an hour
// before anyone suspects the cache. An offline shell would also be of little use for a client
// whose entire job is talking to a server on the other end of a tailnet.
//
// So: no `caches`, ever. The ONE `fetch` handler below never stores a thing — on success it
// returns the live network response, so "edit, pull to refresh, see it" still holds. It exists
// only to turn a failed PAGE navigation (your box unreachable) into a readable line instead of
// iOS's blank white standalone screen. Push in, notification out; and never a white void.

// Show the notification, and let the page know if it happens to be open.
self.addEventListener("push", (event) => {
  let payload = {};
  try { payload = event.data ? event.data.json() : {}; } catch (e) { payload = {}; }

  const title = payload.title || "localharness";
  const options = {
    body: payload.body || "",
    // The same tag for every push about one session is half of the coalescing (§5.5): the OS
    // REPLACES the notification already on the lock screen instead of stacking a third one.
    tag: payload.tag || "localharness",
    // The other half. `renotify: false` replaces silently — no sound, no buzz — so three
    // parked calls in ten minutes are one buzz whose badge has climbed to three. The server
    // decides which one alerts; this line is what makes its decision visible.
    renotify: payload.renotify === true,
    data: payload.data || {},
    icon: "/icon-192.png",
    badge: "/icon-192.png",
    // Carried for a platform that renders it; harmless where it is ignored.
    ...(typeof payload.badge === "number" ? { badgeCount: payload.badge } : {}),
  };

  event.waitUntil((async () => {
    await self.registration.showNotification(title, options);
    // Tell an open page, so it can re-sync its badge — its event stream is closed while
    // backgrounded, so its counts are stale by definition. It deliberately does NOT navigate:
    // moving the view under a thumb that is mid-sentence is worse than a stale badge, and the
    // notification is still sitting there to be tapped.
    const open = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    for (const client of open) client.postMessage({ type: "push", payload });
    if (typeof navigator !== "undefined" && navigator.setAppBadge && payload.badge) {
      try { await navigator.setAppBadge(payload.badge); } catch (e) { /* not everywhere */ }
    }
  })());
});

// Tapping it lands on the thing it is about (WEBCH-44) — not on a generic screen, which would
// reintroduce the friction the notification was sent to remove.
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || "/";
  event.waitUntil((async () => {
    const open = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    for (const client of open) {
      if ("focus" in client) {
        // Same app, already running: steer it rather than opening a second window.
        client.postMessage({ type: "notificationclick", url });
        return client.focus();
      }
    }
    if (self.clients.openWindow) return self.clients.openWindow(url);
  })());
});

// Take over immediately rather than waiting for every tab to close. There is no cached content
// for a new worker to disagree with an old one about, so the usual reason for caution is absent.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

// A page navigation to an unreachable box is the white screen (WEBCH-20 keeps no offline shell).
// Only navigations are touched. We serve the box-down page when the proxy is up but the backend
// is not (502/504 — the mobile server itself never emits those) or when the fetch is rejected
// (tailscale itself unreachable). A real app status (401/404/500) passes through untouched, and
// nothing is ever cached. iOS standalone has no pull-to-refresh, so the page carries its own:
// a Retry button, the `online` event, and a quiet HEAD poll that reloads the instant the box
// answers again. Colors mirror index.html (--color-bg / --color-ink).
const BOX_DOWN = "<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
  + "<meta name=theme-color content=#0D0F15>"
  + "<body style='background:#0D0F15;color:#E9EBEF;font:16px/1.5 system-ui;display:grid;"
  + "place-items:center;min-height:100vh;margin:0;padding:24px;text-align:center'><div>"
  + "can’t reach your box — is <code>localharness mobile</code> running?<br><br>"
  + "<button onclick=location.reload() style='font:inherit;color:#0D0F15;background:#E9EBEF;"
  + "border:0;border-radius:9px;padding:10px 20px'>retry</button></div>"
  + "<script>addEventListener('online',()=>location.reload());"
  + "setInterval(()=>fetch(location.href,{method:'HEAD',cache:'no-store'})"
  + ".then(r=>r.ok&&location.reload()).catch(()=>{}),4000)</script>";
const boxDown = () => new Response(BOX_DOWN, { headers: { "Content-Type": "text/html; charset=utf-8" } });
self.addEventListener("fetch", (event) => {
  if (event.request.mode !== "navigate") return;
  event.respondWith(
    fetch(event.request)
      .then((r) => (r.status === 502 || r.status === 504) ? boxDown() : r)
      .catch(boxDown));
});
