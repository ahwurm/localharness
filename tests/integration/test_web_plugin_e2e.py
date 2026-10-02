"""Phase 46 (the web channel as a plugin), composed on the surfaces the rig can reach.

The unit tests of 46-02..46-07 prove each part; this file proves they compose through the real
entry points: the Typer app through CliRunner (`--help`, `web`, `plugins list|disable`, `start`);
the plugin resolver and its config layers; `_start_async` run in web mode with a real WebChannel
handed in; the WebServer over httpx's ASGI transport (no socket); the page's reducer run verbatim
under node, fed the exact bodies the server returned.

STUBBED: the LLM probe, the tokenizer and the REPL's read loop (`_stub_start_boundaries`; the
`drive` below stands in for the loop and does its HTTP work while the session is live); ComfyUI
(the image e2e's MockTransport — the image plugin is only here to bind an artifact root).

NOT proven here: the owner's phone (46-09's device checkpoint).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys

import httpx

import pytest
from typer.testing import CliRunner

from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import ARTIFACT_PAGE, MEMORY_OFF, WebServer
from localharness.cli.app import app
from localharness.cli.slash_commands import set_plugin_rows
from localharness.cli.web_cmd import MISSING_DEPENDENCY
from localharness.core.bus import EventBus
from tests.integration.test_image_plugin_e2e import _fake_comfy, _machine
from tests.unit.channels.test_web_artifacts import IMMUTABLE, PNG
from tests.unit.channels.test_web_reference_page import BUTTONS, PAGE, _drive
from tests.unit.channels.test_web_server import BEARER, TOKEN
from tests.unit.channels.test_web_server import JSON as JSON_POST
from tests.unit.test_plugin_cli_mount import _help_rows
from tests.unit.test_start_cmd import _capture_start_console

runner = CliRunner()


@pytest.fixture(autouse=True)
def _process_state():
    yield
    set_plugin_rows(())


def _invoke(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"`localharness {' '.join(args)}` raised {result.exception!r}\n{result.output}")
    return result


def _flat(text: str) -> str:
    """Rich soft-wraps; compare with the wrapping collapsed."""
    return " ".join(text.split())


# --- criterion 1 (CLI half): mounted from the plugin, gone when disabled, hint without the extra --


def test_web_is_a_plugin_command_on_by_default_and_gone_when_disabled(tmp_path, monkeypatch, fake_home):
    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    assert "web" in _help_rows(_invoke("--help").output)
    helped = _invoke("web", "--help")
    assert helped.exit_code == 0, helped.output
    assert "--incognito" in helped.output and "--rotate-token" in helped.output, helped.output
    listed = _invoke("plugins", "list").output
    assert any(line.split()[:1] == ["web"] for line in listed.splitlines()), listed

    disabled = _invoke("plugins", "disable", "web")
    assert disabled.exit_code == 0, disabled.output
    assert "web" not in _help_rows(_invoke("--help").output)
    gone = _invoke("web")
    # off bundled plugin: hint + exit 4, not Click's exit 2 (exit 2 is experiment run's reject-holdout verdict)
    assert gone.exit_code == 4 and "is provided by the web plugin, which is off" in gone.stderr, gone.output

    # the terminal is unaffected: a terminal session still boots with web off
    printed = _capture_start_console(monkeypatch)

    async def run() -> None:
        from localharness.cli.start_cmd import _start_async
        await _start_async(None, False, False, None)

    asyncio.run(run())
    assert any("startup)" in line for line in printed), printed
    assert not any("plugin web:" in line for line in printed), printed


def test_without_the_extra_web_prints_the_unchanged_hint(tmp_path, monkeypatch, fake_home):
    from localharness.plugins import resolve

    _machine(tmp_path, monkeypatch, fake_home)
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: False)
    monkeypatch.setitem(sys.modules, "starlette", None)
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert "web" in _help_rows(_invoke("--help").output)  # still mounted: the hint, not "No such command"
    ran = _invoke("web")
    assert ran.exit_code == 1, ran.output
    assert _flat(MISSING_DEPENDENCY) in _flat(ran.output), ran.output
    listed = _invoke("plugins", "list").output
    (row,) = [line for line in listed.splitlines() if line.split()[:1] == ["web"]]
    assert "on (install `localharness[web]` to use it)" in row, row


# --- criterion 2: a channel typo is refused before any plugin loads -------------------------------


def test_a_channel_typo_is_refused_before_any_plugin_loads(tmp_path, monkeypatch, fake_home):
    _machine(tmp_path, monkeypatch, fake_home)
    called: list = []

    def boom(*a, **k):
        called.append(a)
        raise AssertionError("resolve() must not run before the channel gate")

    monkeypatch.setattr("localharness.plugins.resolve.resolve", boom)
    ran = _invoke("start", "--channel", "wbe")
    assert ran.exit_code != 0, ran.output
    assert "unknown channel 'wbe'; choose one of: acp, discord, terminal, web" in _flat(ran.output), ran.output
    assert called == [], "resolve() ran before the channel name was checked"


# --- criteria 3-4 over a real web session: the memory screen behind the slot, the gallery ---------


def _web_session(monkeypatch, work, *, incognito: bool = False) -> dict:
    """One web-mode session through `_start_async`; `work(channel, client, out)` runs while it is
    live, against a WebServer over ASGI. Returns `out`; an error inside `work` fails the test."""
    out: dict = {"printed": _capture_start_console(monkeypatch)}

    async def drive(self):  # stands in for OrchestratorREPL.run
        await self._channel.start()
        try:
            server = WebServer(self._channel, token=TOKEN, incognito=incognito)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                         base_url="http://web.test") as client:
                await work(self._channel, client, out)
            out["ran"] = True
        except BaseException as exc:  # noqa: BLE001 — surfaced below, never swallowed
            out["error"] = exc
        finally:
            await self._channel.stop()

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)

    async def run() -> None:
        from localharness.cli.start_cmd import _start_async
        await _start_async(None, False, False, None, channel_mode="web",
                           web_channel=WebChannel(bus=EventBus(), config={}))

    asyncio.run(run())
    if "error" in out:
        raise out["error"]
    assert out.get("ran"), "the session never reached its REPL"
    return out


def _page(script: str, tmp_path):
    if shutil.which("node") is None:
        pytest.skip("no JS engine on this box: the page half of this test cannot run")
    return _drive(PAGE.read_text(encoding="utf-8"), BUTTONS + script, tmp_path)


def _serve(bodies: dict) -> str:
    """A page `fetch` answering each path with the exact body the server returned for it."""
    return f"const BODIES = {json.dumps(bodies)};\n" + (
        "globalThis.fetch = async (path) => ({json: async () => {\n"
        "  if (!(path in BODIES)) throw new Error('unexpected fetch ' + path);\n"
        "  return BODIES[path]; }});\n")


# The DOM shim predates the memory page and has no replaceChildren (every browser does): the one
# DOM call loadMemory needs, added for the two elements it touches.
REPLACE_CHILDREN = """
for (const id of ["memlist", "memdetail"]) $(id).replaceChildren = function (...ns) {
  for (const c of this.children) c.parent = null;
  this.children = []; for (const n of ns) this.appendChild(n); };
"""
FACT = "notes/phone"


def test_memory_on_the_screen_is_present_and_every_route_works_through_the_slot(
        tmp_path, monkeypatch, fake_home):
    _machine(tmp_path, monkeypatch, fake_home)

    async def work(ch, c, out):
        browse = ch.memory_slot().browse()  # the session's own store, behind the slot
        await browse._store.store_fact(key=FACT, value="the phone reads this", tags=["web"],
                                       source="remember")
        out["protocol"] = (await c.get("/api/protocol", headers=BEARER)).json()
        out["list"] = await c.get("/api/memory", headers=BEARER)
        out["fact"] = await c.get(f"/api/memory/fact?name={FACT}", headers=BEARER)
        out["edit"] = await c.post("/api/memory/edit", json={"name": FACT, "content": "edited on the phone"},
                                   headers=JSON_POST)
        out["edited"] = (await c.get(f"/api/memory/fact?name={FACT}", headers=BEARER)).json()
        out["forget"] = await c.post("/api/memory/forget", json={"name": FACT}, headers=JSON_POST)
        out["after_list"] = (await c.get("/api/memory", headers=BEARER)).json()
        out["after"] = await c.get(f"/api/memory/fact?name={FACT}", headers=BEARER)

    s = _web_session(monkeypatch, work)
    assert s["protocol"]["screens"]["memory"] is True, s["protocol"]["screens"]
    assert s["list"].status_code == 200
    assert [f["value"] for f in s["list"].json()["facts"] if f["name"] == FACT] == ["the phone reads this"]
    assert s["fact"].status_code == 200 and len(s["fact"].json()["history"]) == 1
    assert s["edit"].status_code == 200 and s["edit"].json()["status"] == "edited"
    assert s["edited"]["fact"]["value"] == "edited on the phone"
    assert s["edited"]["fact"]["provenance"].endswith(";web"), s["edited"]["fact"]
    assert "web" in s["edited"]["fact"]["tags"] and len(s["edited"]["history"]) == 2
    assert s["forget"].status_code == 200 and s["forget"].json()["status"] == "forgotten"
    assert not any(f["name"] == FACT for f in s["after_list"]["facts"])
    assert s["after"].status_code == 200 and s["after"].json()["history"], "forget destroyed history"


def test_memory_off_the_screen_is_absent_and_the_routes_404(tmp_path, monkeypatch, fake_home):
    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    with (global_dir / "config.yaml").open("a", encoding="utf-8") as f:
        f.write("org:\n  memory_enabled: false\n")

    async def work(ch, c, out):
        out["occupied"] = ch.memory_slot().occupied
        out["protocol"] = (await c.get("/api/protocol", headers=BEARER)).json()
        out["routes"] = [
            await c.get("/api/memory", headers=BEARER),
            await c.get("/api/memory/fact?name=x", headers=BEARER),
            await c.post("/api/memory/edit", json={"name": "x", "content": "y"}, headers=JSON_POST),
            await c.post("/api/memory/forget", json={"name": "x"}, headers=JSON_POST),
        ]

    s = _web_session(monkeypatch, work)
    assert s["occupied"] is False
    assert s["protocol"]["screens"]["memory"] is False, s["protocol"]["screens"]
    lst, fact, edit, forget = s["routes"]
    assert lst.status_code == 404 and lst.json() == MEMORY_OFF
    assert fact.status_code == 404 and fact.json() == MEMORY_OFF
    assert edit.status_code == 404 and edit.json() == MEMORY_OFF
    assert forget.status_code == 404 and forget.json() == MEMORY_OFF
    # a stale page that still opens the memory screen shows the server's own words (it used to see 409)
    got = _page(_serve({"/api/memory": lst.json()}) + REPLACE_CHILDREN + """
await loadMemory("");
console.log(JSON.stringify($("memlist").children.map((n) => n.textContent)));
""", tmp_path)
    assert got == [MEMORY_OFF["error"]], got


def test_the_page_shows_the_buttons_the_real_protocol_answer_names(tmp_path, monkeypatch, fake_home):
    """The drawer buttons, fed the exact /api/protocol bodies a memory-on and a memory-off session
    returned — both by `applyScreens(body.screens)` and through the page's own `refreshScreens()`."""
    bodies = {}
    for memory_on in (True, False):
        global_dir, _ = _machine(tmp_path / str(memory_on), monkeypatch, fake_home)
        if not memory_on:
            with (global_dir / "config.yaml").open("a", encoding="utf-8") as f:
                f.write("org:\n  memory_enabled: false\n")

        async def work(ch, c, out):
            out["protocol"] = (await c.get("/api/protocol", headers=BEARER)).json()

        bodies[memory_on] = _web_session(monkeypatch, work)["protocol"]
    got = bodies[True]["screens"]
    assert got == {"memory": True, "pictures": False, "incognito": False}, got
    got = bodies[False]["screens"]
    assert got == {"memory": False, "pictures": False, "incognito": False}, got
    for memory_on, body in bodies.items():
        got = _page(_serve({"/api/protocol": body}) + f"""
const out = [];
applyScreens({json.dumps(body)}.screens); out.push(btns());
applyScreens({{memory: !{json.dumps(memory_on)}, pictures: true}});   // flip, then re-read
await refreshScreens(); out.push(btns());
console.log(JSON.stringify(out));
""", tmp_path)
        want = {"mem": not memory_on, "pic": True}
        assert got == [want, want], (memory_on, got)


def _image_on(tmp_path, monkeypatch, fake_home):
    """The real image plugin on (it binds the session's artifact root); ComfyUI mocked, never dialed
    by these tests. Artifact ids minted one second apart so newest-first is deterministic."""
    from datetime import datetime, timedelta, timezone

    from localharness.core import artifacts

    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    enabled = _invoke("plugins", "enable", "image", "--set", "comfyui_url=http://comfy.test")
    assert enabled.exit_code == 0, enabled.output
    _fake_comfy(monkeypatch)
    real, t0, n = artifacts.mint_artifact_id, datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc), [0]

    def mint(now=None):
        n[0] += 1
        return real(now or t0 + timedelta(seconds=n[0]))

    monkeypatch.setattr(artifacts, "mint_artifact_id", mint)
    return global_dir


async def _write(ch, count: int) -> list[str]:
    from localharness.core.artifacts import write_artifact

    root = ch.artifact_roots()["image"]
    return [write_artifact(root, "image", PNG, "image/png").id for _ in range(count)]


def test_the_gallery_lists_the_sessions_pictures_and_incognito_turns_it_off(
        tmp_path, monkeypatch, fake_home):
    global_dir = _image_on(tmp_path, monkeypatch, fake_home)

    async def work(ch, c, out):
        out["roots"] = ch.artifact_roots()
        out["ids"] = ids = await _write(ch, 2)
        out["protocol"] = (await c.get("/api/protocol", headers=BEARER)).json()
        out["list"] = await c.get("/api/artifacts", headers=BEARER)
        out["items"] = [await c.get(f"/api/artifacts/image/{i}", headers=BEARER) for i in ids]

    s = _web_session(monkeypatch, work)
    assert s["roots"] == {"image": global_dir / "artifacts" / "image"}, s["roots"]
    old, new = s["ids"]
    assert s["protocol"]["screens"]["pictures"] is True, s["protocol"]["screens"]
    assert s["list"].status_code == 200
    assert s["list"].json() == {"items": [
        {"plugin": "image", "id": new, "mime": "image/png", "bytes": len(PNG)},
        {"plugin": "image", "id": old, "mime": "image/png", "bytes": len(PNG)}], "truncated": False}
    for got in s["items"]:
        assert got.status_code == 200 and got.content == PNG
        assert got.headers["cache-control"] == IMMUTABLE

    async def private(ch, c, out):
        out["ids"] = await _write(ch, 1)
        out["protocol"] = (await c.get("/api/protocol", headers=BEARER)).json()
        out["list"] = await c.get("/api/artifacts", headers=BEARER)
        out["item"] = await c.get(f"/api/artifacts/image/{out['ids'][0]}", headers=BEARER)

    p = _web_session(monkeypatch, private, incognito=True)
    assert p["list"].status_code == 404
    assert p["protocol"]["screens"]["pictures"] is False, p["protocol"]["screens"]
    assert p["item"].status_code == 200 and p["item"].content == PNG  # inline pictures still load…
    assert p["item"].headers["cache-control"] == "no-store"          # …fetched fresh each time
    # the page fed that --incognito protocol answer hides the Pictures button
    got = _page(f"applyScreens({json.dumps(p['protocol'])}.screens); console.log(JSON.stringify(btns()));",
                tmp_path)
    assert got["pic"] is True, got


def test_rapid_show_older_taps_never_duplicate_a_picture(tmp_path, monkeypatch, fake_home):
    """46-07's named gap, probed: two "show older" taps before the first answer lands. The page is
    fed the session's real listing pages (ARTIFACT_PAGE + 1 pictures → two pages)."""
    _image_on(tmp_path, monkeypatch, fake_home)

    async def work(ch, c, out):
        out["ids"] = await _write(ch, ARTIFACT_PAGE + 1)
        first = (await c.get("/api/artifacts", headers=BEARER)).json()
        before = first["items"][-1]["id"]
        out["bodies"] = {"/api/artifacts": first,
                         f"/api/artifacts?before={before}": (await c.get(
                             f"/api/artifacts?before={before}", headers=BEARER)).json()}

    s = _web_session(monkeypatch, work)
    first, older = s["bodies"].values()
    assert first["truncated"] is True and len(first["items"]) == ARTIFACT_PAGE
    assert older["truncated"] is False and len(older["items"]) == 1
    got = _page(_serve(s["bodies"]) + """
await loadPictures(false);
await Promise.all([loadPictures(true), loadPictures(true)]);
console.log(JSON.stringify({srcs: $("picgrid").children.map((e) => e.src), more: $("picmore").hidden}));
""", tmp_path)
    want = [f"/api/artifacts/image/{i}" for i in reversed(s["ids"])]
    assert got == {"srcs": want, "more": True}, (len(got["srcs"]), len(set(got["srcs"])))
