# The browser plugin

Turning it on costs about 100 MB on disk (Playwright's headless Chromium, kept in Playwright's cache, `~/.cache/ms-playwright` on Linux) and several hundred MB of RAM while a page loads (one measurement put the driver and the browser together at about 700 MB on a trivial page). The browser starts at the first `browser_fetch` and is closed after `browser.idle_close_s` seconds without one (120 by default) and when the session ends. On a machine whose GPU shares system memory, that RAM is memory the model's cache cannot use while the browser is up.

The `browser` plugin gives the web researcher and the search verifier `browser_fetch`: an https page opened in a real, sandboxed headless Chromium with its JavaScript run, its text returned the way `web_fetch` returns a page (a window of it, the full text kept for `web_page_query`, the next window one `start_index` away). They hold it after `web_search`, `web_fetch` and `web_page_query`. The root agent never holds it, because it reads untrusted pages; an agent you define gets it with `tools.add: [browser_fetch]`. Its description tells the model to call it once, with the same URL, when `web_fetch` answered "low extractable text", and otherwise to use `web_fetch`. It ships with LocalHarness, is off until you turn it on, and its code is [`browser_plugin.py`](../../src/localharness/tools/builtin/browser_plugin.py), [`browser_tool.py`](../../src/localharness/tools/builtin/browser_tool.py) and [`browser_proxy.py`](../../src/localharness/tools/builtin/browser_proxy.py).

## Turn it on and off

```bash
uv tool install 'localharness[browser]'     # name the other extras you use too: a reinstall keeps only the ones it names
localharness plugins enable browser         # asks to download Playwright's headless Chromium, then checks it
localharness plugins disable browser
```

The enable asks "Download Playwright's headless Chromium now? About 100 MB from Playwright's CDN; it runs as a program on this machine (sandboxed by default).", runs Playwright's own installer with its progress on your terminal, then checks four `doctor` rows in this order: `browser-package`, `browser-binary`, `browser-launch` and `browser-sandbox`. It is done when all four pass. The plugin is for Linux and macOS, where the launch check can read the browser's command line; elsewhere `browser-launch` fails with that reason and `browser_fetch` stays off. Enabled before the extra is installed, it writes the enable, names the install line and downloads nothing, so run it again after installing.

In a running session, `/plugins enable browser` asks the same question and restarts the session with your conversation kept. With the plugin on and no Chromium downloaded, no browser tool exists and `localharness start` prints no warning, while `localharness doctor` and `localharness plugins list` say why.

On Linux, when `doctor` says `Host system is missing dependencies to run browsers`, run the install-deps command it prints (apt only), or set `browser.executable_path` to a system Chromium. When `browser-sandbox` fails (Ubuntu 23.10 and later restrict unprivileged user namespaces for a program without an AppArmor profile), add an AppArmor profile for the downloaded headless shell (on Ubuntu 26.04 with `kernel.apparmor_restrict_unprivileged_userns = 1`, a profile of the form `profile <name> <path to the shell> flags=(unconfined) { userns, }` is enough), or set `browser.sandbox: false` in your machine-level config, which accepts an unsandboxed browser. The profile attaches to the shell's path, which carries Playwright's build number (for example `chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell` under `~/.cache/ms-playwright`), so a `playwright install` that changes that folder needs the profile's path updated and the profile reloaded with `apparmor_parser -r`.

`plugins disable browser` removes the tool; the download stays in Playwright's cache.

## Settings

| Setting | Meaning |
|---|---|
| `browser.idle_close_s` | Seconds without a call before the browser process is closed. Default 120; 1 to 3600. |
| `browser.executable_path` | **Machine-level only.** Your own Chromium; empty uses the one Playwright installed. The way out on a distro without the apt libraries. |
| `browser.sandbox` | **Machine-level only.** `true` (default) launches Chromium sandboxed and refuses to run otherwise; `false` accepts an unsandboxed browser, a residual [SECURITY.md](../../SECURITY.md#what-a-web-fetch-may-reach) names. |

A project cannot set the two machine-level ones: they choose the program the harness starts and whether it is sandboxed. A project can turn the plugin on for itself, only against the Chromium and the sandbox setting your machine already has, and nothing downloads on a project's say-so. Timeouts and limits are constants, not settings.

## What it does not do

- No signed-in browsing: no cookies, profile or storage between calls. Each call gets a new browser context that is closed after it.
- No clicking, typing, forms, downloads or uploads.
- No screenshots.
- No plain-http pages: `browser_fetch` opens https only, and `web_fetch` reads http pages.
- No browser-backed search.
- No anti-bot stealth: its user agent is Chromium's own, which says HeadlessChrome, with LocalHarness and its version appended, so a site that refuses headless browsers refuses this one.
- No driving your own app under test; a config child with `bash_exec` does that.
- No reaching a private service (see the next section).

## What it reaches

Public addresses only, by the same rule as `web_fetch`, applied to every request the page makes: the page, each redirect, every script, image request, `fetch` and WebSocket leaves through a proxy the plugin runs on `127.0.0.1` for the life of the browser, with a user name and password made new at every launch. The proxy accepts only `CONNECT`, judges each one by the public-address rule and connects to the address it checked, never a fresh lookup. QUIC is off, WebRTC may not send UDP outside the proxy, and Chromium writes no crash dumps.

Every launch checks this before any page: a navigation to a loopback port must be refused by the proxy while a listener on that port sees nothing, and Chromium's command line must carry the proxy and the flags. A launch that fails the check keeps `browser_fetch` off, with "reach check failed" in `doctor`.

Images, media and fonts are not downloaded. A navigation waits for the page's `load` event and half a second more, at most 30 seconds. The page's text is cut inside the browser to 4,000,000 characters by default (`web_fetch`'s download budget, counted as the page's own string length counts characters) before it reaches the harness. At most two pages load at once and a third call waits.

`org.web_fetch_allow_private` is `web_fetch`'s alone. One GET with no origin can reach a service you named, but a page has an origin and runs code, so honouring the allowlist would let a public page rebind its own name to that service after loading, read its answers as same-origin and send it POSTs. Reading a private service stays `web_fetch`'s job.

See [SECURITY.md, What a web fetch may reach](../../SECURITY.md#what-a-web-fetch-may-reach).

## Not there yet

- The terminal prints `browser_fetch` as its own tool line rather than in the `web_search · web_fetch` counter, and in `--verbose` gives its result no "web results — UNTRUSTED" close note (the result still opens with the untrusted banner the model reads).
- A `browser_fetch` result takes the context store's generic large-result path, which evicts later than the early web-result eviction at half the window.
- The bench's ask-rate replay counts `browser_fetch` as an undeclared family.
- A config child that names the prefixed `plugin:browser.browser_fetch` keeps a copy not bound to its own store, so name it `browser_fetch`.

## More

- [localharness.dev/plugins/browser](https://localharness.dev/plugins/browser/)
- [SECURITY.md, What a web fetch may reach](../../SECURITY.md#what-a-web-fetch-may-reach)
- [Spec 09, the bundled plugins](../specs/09-hooks-plugins.md#the-bundled-plugins)
- [Write your own plugin](../../examples/plugin-template/README.md)
