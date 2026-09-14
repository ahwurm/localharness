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
// So: no `fetch` handler, no `caches`. Push in, notification out.

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
    // A page that IS open should move to the item too, rather than making the owner tap a
    // notification about a screen they are already looking at.
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
