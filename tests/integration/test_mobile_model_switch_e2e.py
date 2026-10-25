"""A phone lists, switches and launches models over the real channel, server and REPL dispatch; only each endpoint's served list, the swap and the persists are fakes."""
from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

import httpx

from localharness.channels.mobile.channel import MobileChannel
from localharness.channels.mobile.server import MobileServer
from localharness.cli.model_switch import ModelSwitch, short_label
from localharness.core.bus import EventBus
from localharness.provider import lifecycle
from tests.unit.channels.test_mobile_server import BEARER, JSON, TOKEN
from tests.unit.launch_support import stub_liveness
from tests.unit.test_launch_paths_repl import PEER_URL, PRIMARY_URL, _launchable_peer, _session
from tests.unit.test_launch_paths_repl import _no_machine_probes as _probes  # noqa: F401 — autouse fixture
from tests.unit.test_model_swap_repl import _landing_swap, _logged_lock

FAILED = "the launch failed — llama-server exited"
COLD = {PRIMARY_URL: ["model-a"], PEER_URL: "unreachable"}


class Phone:
    def __init__(self, repl, channel, agent, client, http, swaps, live):
        self.repl, self.channel, self.agent, self.client, self.http = repl, channel, agent, client, http
        self.swaps, self.live = swaps, live
        self.seen: list[tuple[str, dict]] = []

    def drain(self) -> list[tuple[str, dict]]:
        while not self.client.queue.empty():
            name, _seq, payload = self.client.queue.get_nowait()
            self.seen.append((name, json.loads(payload)))
        return self.seen

    def of(self, name: str) -> list[dict]:
        return [p for n, p in self.drain() if n == name]

    def activity(self) -> list[str]:
        return [p["text"] for p in self.of("Activity")]

    async def until(self, ready) -> None:
        for _ in range(500):
            if ready(self.drain()):
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"timed out; frames so far: {self.seen}")

    async def ask(self) -> dict:
        await self.until(lambda seen: any(n == "BlockingAsk" for n, _ in seen))
        return self.of("BlockingAsk")[-1]

    async def post(self, text: str) -> None:
        response = await self.http.post("/api/sessions/s1/command", headers=JSON, json={"text": text})
        assert response.status_code == 200

    async def line(self, text: str) -> None:
        await self.post(text)
        await self.repl._dispatch_input(await self.channel.read_input())

    async def start(self, text: str) -> asyncio.Task:
        await self.post(text)
        return asyncio.ensure_future(self.repl._dispatch_input(await self.channel.read_input()))

    async def answer(self, ask: dict, kind: str, token: str | None = None) -> dict:
        body = {"kind": kind, **({"confirm_token": token} if token else {})}
        response = await self.http.post(f"/api/permissions/{ask['request_id']}/answer", headers=JSON, json=body)
        return {"http": response.status_code, **response.json()}

    async def models(self) -> dict:
        response = await self.http.get("/api/model", headers=BEARER)
        assert response.status_code == 200
        return response.json()


@contextlib.asynccontextmanager
async def _phone(tmp_path, monkeypatch, live, *, swap=None):
    stub_liveness(monkeypatch, False)
    harness, repl, _, agent = _session(tmp_path, _launchable_peer(), live=live)
    bus = EventBus()
    channel = MobileChannel(bus=bus, config={})
    await channel.start()
    channel.bind_runtime(session_id="s1", agent_id="root", session_dir=tmp_path / "sessions")
    agent.current_session_id = "s1"
    agent._config = SimpleNamespace(name="root")
    repl._bus, repl._channel = bus, channel
    repl._gate = SimpleNamespace(remote_unattended=True)
    repl._switch = ModelSwitch(agent, bus, channel, harness, repl._session_gate, tmp_path)

    async def persist_model(model):
        return False

    async def persist_endpoint(endpoint, model):
        return None

    repl._persist_default_model, repl._persist_active_endpoint = persist_model, persist_endpoint
    _logged_lock(repl, [])
    repl._subscribe_pending()
    channel.model_source = lambda: short_label(agent._llm.config.model)
    swaps = swap(monkeypatch, repl) if swap else _landing_swap(monkeypatch, repl)
    client = channel.attach_client()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=MobileServer(channel, token=TOKEN).app),
                             base_url="http://web.test")
    try:
        yield Phone(repl, channel, agent, client, http, swaps, live)
    finally:
        await http.aclose()
        await channel.stop()


def _failing_swap(monkeypatch, repl):
    calls: list = []

    async def swap(incumbent, target, *, reporter, **kw):
        calls.append(target)
        await reporter.freed()
        await reporter.line(FAILED)
        return lifecycle.SwapOutcome(None, None, None, RuntimeError(FAILED), "launch")

    monkeypatch.setattr(lifecycle, "swap", swap)
    return calls


def _hold_the_landing(repl):
    landing, release = asyncio.Event(), asyncio.Event()

    async def refresh(model, **kw):
        landing.set()
        await release.wait()
        return ""

    repl._refresh_token_counter = refresh
    return landing, release


async def test_a_phone_lists_and_hot_switches_a_model(tmp_path, monkeypatch):
    live = {PRIMARY_URL: ["model-a", "model-b"], PEER_URL: "unreachable"}
    async with _phone(tmp_path, monkeypatch, live) as p:
        assert (await p.channel.hello(p.client, None)).model == "model-a"
        listed = await p.models()
        assert listed["active"] == "model-a"
        assert [r["kind"] for r in listed["models"] if r["name"] == "model-b"] == ["hot"]

        await p.line("/model model-b")

        [switched] = p.of("ModelSwitched")
        assert (switched["model"], switched["endpoint"], switched["kind"]) == ("model-b", "primary", "hot")
        assert p.of("BlockingAsk") == [] and p.of("Activity") == []
        assert (await p.models())["active"] == "model-b"


async def test_a_phone_launches_a_cold_model_through_one_confirm(tmp_path, monkeypatch):
    async with _phone(tmp_path, monkeypatch, dict(COLD)) as p:
        landing, release = _hold_the_landing(p.repl)
        task = await p.start("/model qwen-gguf")
        ask = await p.ask()
        assert ask["klass"] == "picker"
        assert [(o["kind"], o["confirm_required"]) for o in ask["options"]] == [("Launch", True), ("Cancel", False)]

        first = await p.answer(ask, "Launch")
        assert first["status"] == "confirm_required" and first["detail"] == ask["reason"]
        assert await p.answer(ask, "Launch", first["confirm_token"]) == {
            "http": 200, "status": "recorded", "decision": "Launch"}

        await asyncio.wait_for(landing.wait(), 5)
        texts = p.activity()
        assert texts[0].startswith("launching qwen-gguf") and texts[-1] == ""
        assert p.of("ModelSwitched") == []
        assert (await p.models())["launching"] == "qwen-gguf"

        p.live[PEER_URL] = ["qwen-gguf"]
        release.set()
        await asyncio.wait_for(task, 5)

        [switched] = p.of("ModelSwitched")
        assert (switched["kind"], switched["model"], switched["endpoint"]) == ("cold", "qwen-gguf", "llamacpp-local")
        names = [n for n, _ in p.seen]
        assert names.index("ModelSwitched") < len(names) - 1 - names[::-1].index("Activity")
        assert p.activity()[-1] == ""
        landed = await p.models()
        assert landed["launching"] is None and landed["active"] == "qwen-gguf"
        assert len(p.swaps) == 1
