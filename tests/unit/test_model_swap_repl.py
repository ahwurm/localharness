"""In-REPL full model swap: /model <registry-name> restarts the managed server with the
named local checkpoint — stop → start(registry-resolved command) → readiness poll with a
live progress line → rebind + persist. The registry names appear in the listing and the
picker cache; a failed swap reports and leaves the session model untouched.

Offline: managed_server lifecycle functions are monkeypatched at the module the REPL
imports; serve_command stays REAL so the command the swap builds is asserted end-to-end.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace as NS

import pytest

from localharness.cli import model_ops
from localharness.cli.model_switch import (
    LAUNCH_ELSEWHERE,
    LAUNCH_LOCKED,
    LAUNCH_NO,
    LAUNCH_NO_PHONE,
    LAUNCH_STOPS,
    LAUNCH_STOPS_NONE,
    LAUNCH_YES,
    launch_holder,
)
from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import EndpointRef, HarnessConfig, ManagedServerConfig, OrgConfig, ProviderConfig
from localharness.core.bus import EventBus
from localharness.core.events import ModelSwitched
from tests.unit.launch_support import offer_primary_as, stub_liveness
from tests.unit.test_launch_paths_repl import PEER_URL, _launchable_peer, _session
from tests.unit.test_launch_paths_repl import _no_machine_probes as _probes  # noqa: F401 — autouse fixture
from tests.unit.test_launch_paths_repl import boundaries as _boundaries  # noqa: F401 — the fixture
from tests.unit.test_plan_entries import harness_of, twin_files
from tests.unit.test_serve_cmd import PRIMARY, PRIMARY_URL, serve_world


class _Chan:
    local_operator = True

    def __init__(self):
        self.msgs: list[str] = []
        self.activity: list[str | None] = []
        self.menu_opened = 0

    async def send_message(self, content, **kw):
        self.msgs.append(content)

    def box_activity(self, text):
        self.activity.append(text)

    def box_open_model_menu(self):
        self.menu_opened += 1


class _ListChan(_Chan):
    _model_list_resolver = None
    local_operator = True


def _managed() -> ManagedServerConfig:
    return ManagedServerConfig(
        launch="docker", docker_image="img:tag", model="qwen3.6-35b-a3b", port=8000,
        extra_args=["--kv-cache-dtype", "fp8"],
        local_models=[
            {"name": "qwen3.6-35b-a3b", "path": "/x/Qwen35",
             "extra_args": ["--moe-backend", "marlin"],
             "quant": "nvfp4 (modelopt)", "tps": 30.0},
            {"name": "qwen3.6-27b", "path": "/x/Qwen27",
             "quant": "nvfp4 (compressed-tensors)", "tps": 9.5},
        ],
    )


def _mk(tmp_path, managed, monkeypatch, *, bus=None, session_id=None, peers=(), live=None, chan=None):
    stub_liveness(monkeypatch, False)
    harness = HarnessConfig(
        provider=ProviderConfig(provider_type="vllm", base_url="http://x/v1",
                                default_model="qwen3.6-35b-a3b", available_models=["qwen3.6-35b-a3b"]),
        org=OrgConfig(default_model="qwen3.6-35b-a3b"), server=managed, extra_endpoints=list(peers))
    llm = NS(config=NS(base_url="http://x/v1", model="qwen3.6-35b-a3b"))

    def rebind(base_url, *, api_key=None, extra_headers=None, provider_type=None):
        llm.config.base_url = base_url

    async def caps():
        return NS(tool_call_mode="native")

    llm.detect_capabilities = caps
    llm.rebind_endpoint = rebind
    chan = chan or _Chan()
    agent = NS(_llm=llm, _config=NS(name="root"), current_session_id=session_id) if session_id else NS(_llm=llm)
    r = OrchestratorREPL(
        orchestrator=None, agent_loop=agent, channel=chan, bus=bus,
        config_dir=tmp_path, harness_config=harness,
    )
    served = live or {}

    async def fake_live(base_url):
        return (served.get(base_url, ["qwen3.6-35b-a3b"]), True)

    r._live_models = fake_live

    async def fake_refresh(model, **kw):
        return ""

    r._refresh_token_counter = fake_refresh
    persisted: list[str] = []

    async def fake_persist(model):
        persisted.append(model)

    r._persist_default_model = fake_persist
    return r, chan, llm, persisted


async def test_listing_names_the_serving_model_and_opens_the_picker(tmp_path, monkeypatch):
    r, chan, _llm, _p = _mk(tmp_path, _managed(), monkeypatch)
    await r._handle_model_cmd("")
    out = "\n".join(chan.msgs)
    assert "1. qwen3.6-35b-a3b  (serving)  ● [active]" in out
    assert "downloaded" not in out and "restarts the managed server" not in out
    assert r._model_cache == [("qwen3.6-35b-a3b", "serving now")]   # the menu offers what the list numbers
    assert chan.menu_opened == 1                       # bare /model auto-opens the picker menu


async def test_swap_to_registry_model_full_choreography(tmp_path, monkeypatch):
    from localharness.provider import server as ms

    r, chan, llm, persisted = _mk(tmp_path, _managed(), monkeypatch)
    offer_primary_as(monkeypatch, "qwen3.6-27b")
    calls: dict = {}

    def fake_stop(cfg, launch="binary", **kw):
        calls["stop"] = (cfg, launch)

    def fake_start(cfg, cmd, **kw):
        calls["cmd"] = cmd
        return 1

    async def fake_ready(base_url, config_dir=None, on_poll=None, **kw):
        if on_poll is not None:
            on_poll(1.0)
            on_poll(2.0)
        return ["qwen3.6-27b"]

    monkeypatch.setattr(ms, "stop_server", fake_stop)
    monkeypatch.setattr(ms, "start_server", fake_start)
    monkeypatch.setattr(ms, "wait_ready", fake_ready)

    await r._handle_model_cmd("qwen3.6-27b")

    assert calls["stop"][1] == "docker"
    joined = " ".join(calls["cmd"])                       # REAL serve_command output
    assert "-v /x/Qwen27:/models/serving:ro" in joined
    assert "--model /models/serving" in joined
    assert "--served-model-name qwen3.6-27b" in joined
    assert "--moe-backend" not in joined
    assert llm.config.model == "qwen3.6-27b"
    assert persisted == ["qwen3.6-27b"]
    notes = [a for a in chan.activity if a]
    assert notes and all("qwen3.6-27b" in n for n in notes)   # progress line while loading
    assert chan.activity[-1] is None                          # cleared when done


async def test_swap_failure_reports_and_keeps_current_model(tmp_path, monkeypatch):
    from localharness.provider import server as ms

    r, chan, llm, persisted = _mk(tmp_path, _managed(), monkeypatch)
    offer_primary_as(monkeypatch, "qwen3.6-27b")

    monkeypatch.setattr(ms, "stop_server", lambda cfg, launch="binary", **kw: None)
    monkeypatch.setattr(ms, "start_server", lambda cfg, cmd, **kw: 1)

    async def fake_ready(base_url, config_dir=None, on_poll=None, **kw):
        raise TimeoutError("vLLM not ready after 1800s.")

    monkeypatch.setattr(ms, "wait_ready", fake_ready)

    await r._handle_model_cmd("qwen3.6-27b")

    assert any("Model swap failed" in m for m in chan.msgs)
    assert llm.config.model == "qwen3.6-35b-a3b"          # session model untouched
    assert persisted == []
    assert chan.activity and chan.activity[-1] is None    # progress line cleared on failure too


async def test_a_hot_model_through_the_switch_publishes_one_hot_event(tmp_path, monkeypatch):
    bus = EventBus()
    r, _chan, llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=bus, session_id="s1",
                            live={"http://x/v1": ["qwen3.6-35b-a3b", "qwen3.6-27b"]})

    await r._handle_slash("/model qwen3.6-27b")
    [event] = bus.history(event_types=[ModelSwitched])

    assert (event.kind, event.endpoint, event.model, event.session_id) == ("hot", "primary", "qwen3.6-27b", "s1")
    assert llm.config.model == "qwen3.6-27b"


async def test_an_already_active_pick_through_the_switch_publishes_nothing(tmp_path, monkeypatch):
    bus = EventBus()
    r, _chan, _llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=bus, session_id="s1",
                             live={"http://x/v1": ["qwen3.6-35b-a3b", "qwen3.6-27b"]})

    await r._handle_slash("/model qwen3.6-35b-a3b")

    assert bus.history(event_types=[ModelSwitched]) == []


async def test_a_cross_model_through_the_switch_publishes_one_cross_event(tmp_path, monkeypatch):
    bus = EventBus()
    peer = EndpointRef(name="side", base_url="http://side/v1", provider_type="ollama", api_key="none")
    r, _chan, llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=bus, session_id="s1", peers=[peer],
                            live={"http://x/v1": ["qwen3.6-35b-a3b"], "http://side/v1": ["gpt-oss:20b"]})

    async def persist_active(endpoint, model):
        return None

    r._persist_active_endpoint = persist_active

    await r._handle_slash("/model gpt-oss:20b")
    [event] = bus.history(event_types=[ModelSwitched])

    assert (event.kind, event.endpoint, event.model) == ("cross", "side", "gpt-oss:20b")
    assert llm.config.base_url == "http://side/v1"


async def test_the_resolver_reports_the_model_a_hot_switch_landed_on(tmp_path, monkeypatch):
    chan = _ListChan()
    r, _chan, _llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=EventBus(), session_id="s1", chan=chan,
                             live={"http://x/v1": ["qwen3.6-35b-a3b", "qwen3.6-27b"]})
    r._subscribe_pending()

    await r._handle_slash("/model qwen3.6-27b")

    assert (await chan._model_list_resolver())["active"] == "qwen3.6-27b"


async def test_twins_every_listed_name_picks_its_own_entry(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, peers=twin_files(tmp_path))
    chan = _ListChan()
    llm = NS(config=NS(base_url=PRIMARY_URL, model=PRIMARY))
    r = OrchestratorREPL(
        orchestrator=None, agent_loop=NS(_llm=llm), channel=chan, bus=EventBus(), config_dir=world.g,
        harness_config=harness_of(world), machine_trust=model_ops.trust_as_recorded(world.g))

    async def live_models(url):
        return ([PRIMARY], True) if url == PRIMARY_URL else ([], False)

    r._live_models = live_models
    r._subscribe_pending()
    entries, _live, _bad = await r._plan_model_entries(llm)
    [cold] = [e for e in entries.numbered() if e.kind == "cold"]
    assert cold.others

    rows = [row for row in (await chan._model_list_resolver())["models"] if row["why"] is None]

    assert {row["name"] for row in rows} >= {cold.name}
    assert all(entries.pick(row["name"]).name == row["name"] for row in rows)


COLD_URL = "http://cold/v1"
COLD_SERVED = "qwen-gguf"


def _cold_peer() -> EndpointRef:
    return EndpointRef(
        name="cold", base_url=COLD_URL, provider_type="llamacpp", gpu=True, api_key="none",
        lifecycle=ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080,
                                      gpu=True, extra_args=["-a", COLD_SERVED]))


class _Remote(_Chan):
    local_operator = False

    def __init__(self, *, picker=True):
        super().__init__()
        self.picked: list = []
        if picker:
            self.can_run_picker = lambda: True

    async def run_picker(self, options, current):
        self.picked.append(options)
        return None


class _Anonymous:
    def __init__(self):
        self.msgs: list[str] = []
        self.styles: list[str] = []

    async def send_message(self, content, metadata=None):
        self.msgs.append(content)

    def box_activity(self, text):
        pass


def _remote_world(tmp_path, monkeypatch, chan, gate, *, bus=None, peers=(), served=("qwen3.6-35b-a3b",)):
    r, chan, llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=bus, session_id="s1" if bus else None,
                           peers=[_cold_peer(), *peers], chan=chan)

    async def live(base_url):
        return ([], False) if base_url == COLD_URL else (list(served), True)

    r._live_models = live
    r._gate = gate
    offer_primary_as(monkeypatch, "qwen3.6-27b")
    return r, chan, llm


def _watch_every_probe(r, monkeypatch):
    from localharness.provider import lifecycle

    touched: list[str] = []

    def spy(name):
        def inner(*a, **k):
            touched.append(name)
        return inner

    r._foreign_incumbent = spy("foreign")
    monkeypatch.setattr("localharness.cli.repl._peer_session_key", spy("key"))
    monkeypatch.setattr(lifecycle, "swap", spy("swap"))
    monkeypatch.setattr(lifecycle, "swap_refusal", spy("swap_refusal"))
    return touched


@pytest.mark.parametrize("target", [COLD_SERVED, "qwen3.6-27b"])
async def test_a_launch_from_a_locked_channel_is_refused_before_any_probe(tmp_path, monkeypatch, target):
    bus = EventBus()
    chan = _Remote()
    sent: list = []

    async def send(content, metadata=None):
        sent.append((content, metadata))

    chan.send_message = send
    r, chan, llm = _remote_world(tmp_path, monkeypatch, chan, NS(remote_unattended=False), bus=bus)
    touched = _watch_every_probe(r, monkeypatch)
    before = (r._active_heavy, dict(vars(llm.config)))

    await r._handle_slash(f"/model {target}")

    assert sent == [(LAUNCH_LOCKED, {"style": "system.info"})]
    assert touched == [] and chan.picked == []
    assert (r._active_heavy, dict(vars(llm.config))) == before
    assert bus.history(event_types=[ModelSwitched]) == []


async def test_a_launch_from_a_channel_with_no_operator_attribute_and_no_gate_is_locked(tmp_path, monkeypatch):
    chan = _Anonymous()
    r, chan, _llm = _remote_world(tmp_path, monkeypatch, chan, None)
    touched = _watch_every_probe(r, monkeypatch)

    await r._handle_slash(f"/model {COLD_SERVED}")

    assert chan.msgs == [LAUNCH_LOCKED] and touched == []


async def test_a_launch_from_a_channel_with_no_picker_is_sent_elsewhere(tmp_path, monkeypatch):
    chan = _Remote(picker=False)
    r, chan, _llm = _remote_world(tmp_path, monkeypatch, chan, NS(remote_unattended=True))
    touched = _watch_every_probe(r, monkeypatch)

    await r._handle_slash(f"/model {COLD_SERVED}")

    assert chan.msgs == [LAUNCH_ELSEWHERE.format(target=COLD_SERVED, phone="")] and touched == []


async def test_hot_and_cross_from_a_locked_channel_land_unasked(tmp_path, monkeypatch):
    bus = EventBus()
    side = EndpointRef(name="side", base_url="http://side/v1", provider_type="ollama", api_key="none")
    chan = _Remote()
    r, chan, llm = _remote_world(tmp_path, monkeypatch, chan, NS(remote_unattended=False), bus=bus, peers=[side],
                                served=("qwen3.6-35b-a3b", "qwen3.6-27b"))
    live = r._live_models

    async def live_with_side(base_url):
        return (["gpt-oss:20b"], True) if base_url == "http://side/v1" else await live(base_url)

    r._live_models = live_with_side

    async def persist_active(endpoint, model):
        return None

    r._persist_active_endpoint = persist_active

    await r._handle_slash("/model qwen3.6-27b")
    await r._handle_slash("/model gpt-oss:20b")

    assert [e.kind for e in bus.history(event_types=[ModelSwitched])] == ["hot", "cross"]
    assert chan.picked == [] and LAUNCH_LOCKED not in chan.msgs
    assert llm.config.base_url == "http://side/v1"


class _AskedNothing(_Chan):
    def __init__(self):
        super().__init__()
        self.picked: list = []
        self.can_run_picker = lambda: True

    async def run_picker(self, options, current):
        self.picked.append(options)
        return None


async def test_a_local_operator_is_never_asked_or_refused_a_cold_launch(tmp_path, request):
    boundaries = request.getfixturevalue("_boundaries")
    _, repl, channel, agent = _session(tmp_path, _launchable_peer())
    channel.local_operator = True
    channel.picked = []
    channel.can_run_picker = lambda: True
    channel.run_picker = lambda options, current: channel.picked.append(options)
    repl._gate = NS(remote_unattended=False)

    await repl._handle_slash("/model qwen-gguf")

    assert boundaries.cmds and agent._llm.rebinds[-1][0] == PEER_URL
    assert channel.picked == [] and LAUNCH_LOCKED not in channel.messages
    assert not any(m.startswith("qwen-gguf needs a launch") for m in channel.messages)


async def test_a_local_operator_is_never_asked_or_refused_a_managed_restart(tmp_path, monkeypatch):
    from localharness.provider import server as ms

    chan = _AskedNothing()
    r, chan, llm, _p = _mk(tmp_path, _managed(), monkeypatch, chan=chan)
    r._gate = NS(remote_unattended=False)
    offer_primary_as(monkeypatch, "qwen3.6-27b")
    calls: list = []

    async def fake_ready(base_url, config_dir=None, on_poll=None, **kw):
        return ["qwen3.6-27b"]

    monkeypatch.setattr(ms, "stop_server", lambda cfg, launch="binary", **kw: calls.append("stop"))
    monkeypatch.setattr(ms, "start_server", lambda cfg, cmd, **kw: calls.append("start") or 1)
    monkeypatch.setattr(ms, "wait_ready", fake_ready)

    await r._handle_slash("/model qwen3.6-27b")

    assert calls == ["stop", "start"] and llm.config.model == "qwen3.6-27b"
    assert chan.picked == [] and LAUNCH_LOCKED not in chan.msgs


class _Phone:
    def __init__(self, channel, *answers, attached=True, log=None):
        self.answers, self.attached = list(answers), attached
        self.log: list[str] = [] if log is None else log
        self.asked: list = []
        self.sent: list = []
        self.activity: list = []
        channel.local_operator = False
        channel.can_run_picker = lambda: self.attached
        channel.run_picker = self.run_picker
        channel.send_message = self.send_message
        channel.box_activity = self.box_activity

    async def run_picker(self, options, current, prompt=None, confirm=None):
        self.asked.append(NS(options=options, current=current, prompt=prompt, confirm=confirm))
        self.log.append("ask")
        return self.answers.pop(0) if self.answers else None

    async def send_message(self, content, metadata=None):
        self.sent.append((content, (metadata or {}).get("style")))
        self.log.append(f"send:{content}")

    def box_activity(self, text):
        self.activity.append(text)
        if text and text.startswith("launching "):
            self.log.append("chip")

    def steps(self) -> list[str]:
        return [e for e in self.log if not e.startswith("send:")]


def _logged_lock(repl, log):
    @contextlib.asynccontextmanager
    async def lock():
        log.append("lock")
        yield

    repl._launch_lock = lock


def _landing_swap(monkeypatch, repl, during=None, *, error=None, served="qwen-gguf"):
    from localharness.provider import lifecycle

    calls: list = []

    async def swap(incumbent, target, *, reporter, **kw):
        calls.append(target)
        if during is not None:
            during.append(repl._switch.launching)
        await reporter.freed()
        if error is not None:
            return lifecycle.SwapOutcome(None, None, None, error, "launch")
        live = lifecycle.LiveEndpoint(PEER_URL, [served], None)
        await reporter.launched(live)
        return lifecycle.SwapOutcome(live, None, None, None, None)

    monkeypatch.setattr(lifecycle, "swap", swap)
    return calls


def _phone_cold(tmp_path, monkeypatch, *answers, attached=True, peer=None, during=None, error=None, served="qwen-gguf"):
    stub_liveness(monkeypatch, False)
    harness, repl, channel, agent = _session(tmp_path, peer or _launchable_peer())
    phone = _Phone(channel, *answers, attached=attached)
    repl._gate = NS(remote_unattended=True)
    _logged_lock(repl, phone.log)
    swaps = _landing_swap(monkeypatch, repl, during, error=error, served=served)
    return harness, repl, phone, agent, swaps


def _refuse_from_foreign_call(monkeypatch, repl, n, module, name, result):
    real, seen = getattr(module, name), []

    def foreign(managed, target):
        seen.append(target)

    def wrapped(*a, **k):
        return result if len(seen) >= n else real(*a, **k)

    repl._foreign_incumbent = foreign
    monkeypatch.setattr(module, name, wrapped)


def _decision_refuses_from_foreign_call(monkeypatch, repl, n, line):
    from localharness.provider import server

    _refuse_from_foreign_call(monkeypatch, repl, n, server, "launch_decision", NS(refusal=line, note=None))


def _refusal_of(monkeypatch, repl, which):
    from localharness.cli import model_ops

    if which == "foreign":
        repl._foreign_incumbent = lambda managed, target: "SOMEONE ELSE HOLDS THE PORT"
        return "SOMEONE ELSE HOLDS THE PORT", "system.info"
    if which == "decision":
        _decision_refuses_from_foreign_call(monkeypatch, repl, 1, "DECISION REFUSED")
        return "DECISION REFUSED", "system.error"
    monkeypatch.setattr(model_ops, "resolve_serving", lambda *a, **k: model_ops.ServingRefusal("WITHHELD", "withheld"))
    return "WITHHELD", "system.error"


@pytest.mark.parametrize("which", ["foreign", "decision", "serving"])
async def test_phone_cold_peer_refusals_are_sent_as_today_and_never_asked(tmp_path, monkeypatch, which):
    _, repl, phone, agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES)
    line, style = _refusal_of(monkeypatch, repl, which)

    await repl._handle_slash("/model qwen-gguf")

    assert phone.sent == [(line, style)] and phone.asked == [] and swaps == []
    assert agent._llm.rebinds == []


async def test_phone_cold_peer_asks_once_naming_what_stops_and_takes_the_lock_after_the_chip(tmp_path, monkeypatch):
    harness, repl, phone, agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES)
    repl._active_heavy = (harness.server, agent._llm.config.base_url)
    holder = launch_holder(repl._active_heavy, harness)
    assert holder
    current = agent._llm.config.model

    await repl._handle_slash("/model qwen-gguf")

    [ask] = phone.asked
    assert ask.options == [LAUNCH_YES, LAUNCH_NO] and ask.confirm == LAUNCH_YES and ask.current == current
    assert "qwen-gguf" in ask.prompt and LAUNCH_STOPS.format(holder=holder) in ask.prompt
    assert len(swaps) == 1
    assert phone.steps()[:3] == ["ask", "chip", "lock"]


async def test_phone_cold_peer_with_nothing_to_stop_says_so(tmp_path, monkeypatch):
    _, repl, phone, _agent, _swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES)
    assert repl._active_heavy is None

    await repl._handle_slash("/model qwen-gguf")

    assert LAUNCH_STOPS_NONE in phone.asked[0].prompt


@pytest.mark.parametrize("answer", [LAUNCH_NO, None])
async def test_phone_cold_peer_cancel_or_no_answer_sends_nothing_and_changes_nothing(tmp_path, monkeypatch, answer):
    _, repl, phone, agent, swaps = _phone_cold(tmp_path, monkeypatch, answer)
    before = (dict(vars(agent._llm.config)), repl._active_heavy)

    await repl._handle_slash("/model qwen-gguf")

    assert len(phone.asked) == 1 and phone.sent == [] and swaps == []
    assert (dict(vars(agent._llm.config)), repl._active_heavy) == before
    assert repl._switch.launching is None and "chip" not in phone.log and "lock" not in phone.log


async def test_phone_cold_peer_with_the_app_closed_says_so_and_does_not_ask(tmp_path, monkeypatch):
    _, repl, phone, agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES, attached=False)

    await repl._handle_slash("/model qwen-gguf")

    expected = LAUNCH_NO_PHONE.format(target="qwen-gguf", current=agent._llm.config.model)
    assert phone.sent == [(expected, "system.info")] and phone.asked == [] and swaps == []


async def test_phone_cold_peer_foreign_holder_after_the_wait_is_sent_and_nothing_launches(tmp_path, monkeypatch):
    _, repl, phone, agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES)
    answers = iter([None, "A SERVER APPEARED ON THE PORT"])
    repl._foreign_incumbent = lambda managed, target: next(answers)

    await repl._handle_slash("/model qwen-gguf")

    assert len(phone.asked) == 1
    assert phone.sent == [("A SERVER APPEARED ON THE PORT", "system.info")]
    assert swaps == [] and agent._llm.rebinds == [] and "chip" not in phone.log


async def test_phone_cold_peer_decision_after_the_wait_is_masked_and_nothing_launches(tmp_path, monkeypatch):
    key = "sk-PEER-SECRET-0123456789"
    _, repl, phone, agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES, peer=_launchable_peer(api_key=key))
    _decision_refuses_from_foreign_call(monkeypatch, repl, 2, f"refused with {key} on argv")

    await repl._handle_slash("/model qwen-gguf")

    [(line, style)] = phone.sent
    assert len(phone.asked) == 1 and style == "system.error"
    assert line.startswith("refused with ") and key not in line
    assert swaps == [] and agent._llm.rebinds == [] and "chip" not in phone.log


async def test_phone_cold_peer_key_line_is_sent_once_after_the_yes(tmp_path, monkeypatch):
    monkeypatch.setattr("localharness.cli.repl._peer_session_key", lambda ep: (None, "KEY LINE"))
    _, repl, phone, _agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_NO, LAUNCH_YES)

    await repl._handle_slash("/model qwen-gguf")
    assert "send:KEY LINE" not in phone.log

    await repl._handle_slash("/model qwen-gguf")

    asks = [i for i, e in enumerate(phone.log) if e == "ask"]
    keys = [i for i, e in enumerate(phone.log) if e == "send:KEY LINE"]
    assert len(asks) == 2 and len(keys) == 1 and keys[0] > asks[1] and len(swaps) == 1


async def test_phone_cold_peer_build_note_is_computed_once_and_reaches_the_question_and_the_launch_line(
        tmp_path, monkeypatch):
    _, repl, phone, _agent, _swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES)
    notes: list = []

    async def build_note(spec):
        notes.append(spec)
        return " (first build x)"

    repl._first_build_note = build_note

    await repl._handle_slash("/model qwen-gguf")

    assert len(notes) == 1 and "(first build x)" in phone.asked[0].prompt
    [launch] = [m for m, _ in phone.sent if m.startswith("Launching ")]
    assert "the swap can take several minutes..." in launch and "(first build x)" in launch


async def test_phone_cold_peer_launching_is_the_target_during_the_swap_and_the_landing(tmp_path, monkeypatch):
    during: list = []
    _, repl, phone, _agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES, during=during)

    async def refresh(model, **kw):
        during.append(repl._switch.launching)
        return ""

    repl._refresh_token_counter = refresh

    await repl._handle_slash("/model qwen-gguf")

    assert during == ["qwen-gguf", "qwen-gguf"] and len(swaps) == 1
    assert repl._switch.launching is None and phone.activity[-1] is None


async def test_terminal_cold_peer_key_line_then_the_refusal_keeps_today_order(tmp_path, monkeypatch):
    from localharness.cli import model_ops

    stub_liveness(monkeypatch, False)
    monkeypatch.setattr("localharness.cli.repl._peer_session_key", lambda ep: (None, "KEY LINE"))
    monkeypatch.setattr(model_ops, "resolve_serving", lambda *a, **k: model_ops.ServingRefusal("WITHHELD", "withheld"))
    _, repl, channel, agent = _session(tmp_path, _launchable_peer())
    sent: list = []
    original = channel.send_message

    async def send(text, metadata=None):
        sent.append((text, (metadata or {}).get("style")))
        await original(text, metadata)

    channel.send_message = send
    swaps = _landing_swap(monkeypatch, repl)

    await repl._handle_slash("/model qwen-gguf")

    assert sent == [("KEY LINE", "system.info"), ("WITHHELD", "system.error")]
    assert swaps == [] and agent._llm.rebinds == [] and not hasattr(channel, "run_picker")


@pytest.fixture
def launches(request):
    return request.getfixturevalue("_boundaries")


def _phone_restart(tmp_path, monkeypatch, *answers, on_peer=False, attached=True):
    from tests.unit.test_launch_paths_repl import ON_PEER, PRIMARY_DOWN, _llama_primary, _on_the_cold_peer

    stub_liveness(monkeypatch, False)
    peer = _launchable_peer()
    harness, repl, channel, agent = _session(
        tmp_path, peer, server=_llama_primary(), live=ON_PEER if on_peer else PRIMARY_DOWN)
    if on_peer:
        _on_the_cold_peer(repl, agent, peer)
    phone = _Phone(channel, *answers, attached=attached)
    repl._gate = NS(remote_unattended=True)
    _logged_lock(repl, phone.log)
    return harness, repl, phone, agent, peer


PRIMARY_SERVED = "qwen-primary"
RESTART_KEY = "sk-PROVIDER-SECRET-0123456789"


def _restart_refusal(monkeypatch, repl, which, n, line):
    from localharness.provider import lifecycle
    from localharness.provider import server

    if which == "foreign":
        answers = iter([None] * (n - 1) + [line])
        repl._foreign_incumbent = lambda managed, target: next(answers, line)
        return line, "system.info"
    if which == "decision":
        _refuse_from_foreign_call(monkeypatch, repl, n, server, "launch_decision", NS(refusal=line, note=None))
        return line, "system.error"
    if which == "unit":
        _refuse_from_foreign_call(monkeypatch, repl, n, lifecycle, "unit_launch_refusal", line)
        return lifecycle.SWAP_REFUSED.format(name=PRIMARY_SERVED, why=line), "system.error"
    _refuse_from_foreign_call(monkeypatch, repl, n, lifecycle, "swap_refusal", line)
    return line, "system.error"


RESTART_REFUSALS = ["foreign", "decision", "unit", "swap"]


@pytest.mark.parametrize("which", RESTART_REFUSALS)
async def test_phone_restart_refusals_are_sent_as_today_and_never_asked(tmp_path, monkeypatch, launches, which):
    _, repl, phone, agent, peer = _phone_restart(tmp_path, monkeypatch, LAUNCH_YES, on_peer=True)
    line, style = _restart_refusal(monkeypatch, repl, which, 1, "NOT TODAY")

    await repl._handle_slash(f"/model {PRIMARY_SERVED}")

    assert phone.sent == [(line, style)] and phone.asked == []
    assert agent._llm.rebinds == [] and agent._llm.config.base_url == peer.base_url
    assert launches.stops == [] and launches.cmds == []


async def test_phone_restart_cancel_leaves_the_client_on_the_peer_with_no_rebind(tmp_path, monkeypatch, launches):
    _, repl, phone, agent, peer = _phone_restart(tmp_path, monkeypatch, LAUNCH_NO, on_peer=True)
    before = (dict(vars(agent._llm.config)), repl._active_heavy)

    await repl._handle_slash(f"/model {PRIMARY_SERVED}")

    assert len(phone.asked) == 1 and phone.sent == []
    assert agent._llm.rebinds == [] and agent._llm.config.base_url == peer.base_url
    assert (dict(vars(agent._llm.config)), repl._active_heavy) == before
    assert launches.stops == [] and launches.cmds == []
    assert repl._switch.launching is None and "chip" not in phone.log and "lock" not in phone.log


async def test_phone_restart_names_the_fallback_holder_when_nothing_is_tracked(tmp_path, monkeypatch, launches):
    harness, repl, phone, agent, _peer = _phone_restart(tmp_path, monkeypatch, LAUNCH_NO)
    assert repl._active_heavy is None
    holder = launch_holder(None, harness, (harness.server, agent._llm.config.base_url))
    assert holder

    await repl._handle_slash(f"/model {PRIMARY_SERVED}")

    [ask] = phone.asked
    assert ask.options == [LAUNCH_YES, LAUNCH_NO] and ask.confirm == LAUNCH_YES
    assert PRIMARY_SERVED in ask.prompt and LAUNCH_STOPS.format(holder=holder) in ask.prompt


@pytest.mark.parametrize("which", RESTART_REFUSALS)
async def test_phone_restart_refusal_after_the_wait_is_masked_and_nothing_moves(
        tmp_path, monkeypatch, launches, which):
    from pydantic import SecretStr

    harness, repl, phone, agent, peer = _phone_restart(tmp_path, monkeypatch, LAUNCH_YES, on_peer=True)
    harness.provider.api_key = SecretStr(RESTART_KEY)
    line, style = _restart_refusal(monkeypatch, repl, which, 2, f"changed, key {RESTART_KEY} in the tail")

    await repl._handle_slash(f"/model {PRIMARY_SERVED}")

    [(sent, sent_style)] = phone.sent
    assert len(phone.asked) == 1 and sent_style == style and sent
    assert RESTART_KEY not in sent and "changed, key" in sent
    assert agent._llm.rebinds == [] and agent._llm.config.base_url == peer.base_url
    assert launches.stops == [] and launches.cmds == [] and "chip" not in phone.log


async def test_phone_restart_asks_then_chips_then_locks_and_computes_the_build_note_once(
        tmp_path, monkeypatch, launches):
    _, repl, phone, agent, _peer = _phone_restart(tmp_path, monkeypatch, LAUNCH_YES)
    notes: list = []

    async def build_note(spec):
        notes.append(spec)
        return " (first build x)"

    repl._first_build_note = build_note

    await repl._handle_slash(f"/model {PRIMARY_SERVED}")

    assert phone.steps()[:3] == ["ask", "chip", "lock"] and len(launches.cmds) == 1
    assert len(notes) == 1 and "(first build x)" in phone.asked[0].prompt
    [line] = [m for m, _ in phone.sent if m.startswith("Restarting managed")]
    assert "model load can take several minutes... (first build x)" in line
    assert repl._switch.launching is None


def _publishing(repl, agent) -> EventBus:
    bus = EventBus()
    agent.current_session_id = "s1"
    agent._config = NS(name="root")
    repl._switch._bus = bus
    return bus


def _only_event(bus):
    [event] = bus.history(event_types=[ModelSwitched])
    return event


async def test_phone_cold_peer_lands_one_cold_event(tmp_path, monkeypatch):
    _, repl, _phone, agent, _swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES)
    bus = _publishing(repl, agent)

    await repl._handle_slash("/model qwen-gguf")

    event = _only_event(bus)
    assert (event.kind, event.endpoint, event.model, event.session_id) == ("cold", "llamacpp-local", "qwen-gguf", "s1")


async def test_phone_cold_peer_serving_the_name_the_session_has_lands_one_cold_event(tmp_path, monkeypatch):
    _, repl, _phone, agent, _swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_YES)
    agent._llm.config.model = "qwen-gguf"
    bus = _publishing(repl, agent)

    await repl._handle_slash("/model qwen-gguf")

    event = _only_event(bus)
    assert (event.kind, event.endpoint, event.model) == ("cold", "llamacpp-local", "qwen-gguf")
    assert agent._llm.config.base_url == PEER_URL


async def test_phone_restart_lands_one_cold_event_on_the_primary(tmp_path, monkeypatch, launches):
    _, repl, phone, agent, _peer = _phone_restart(tmp_path, monkeypatch, LAUNCH_YES)
    bus = _publishing(repl, agent)

    await repl._handle_slash(f"/model {PRIMARY_SERVED}")

    event = _only_event(bus)
    assert (event.kind, event.endpoint, event.model) == ("cold", "primary", "qwen-gguf")
    assert len(launches.cmds) == 1 and phone.activity[-1] is None


async def test_phone_restart_swap_back_to_the_name_the_peer_had_lands_one_cold_event(tmp_path, monkeypatch, launches):
    _, repl, _phone, agent, _peer = _phone_restart(tmp_path, monkeypatch, LAUNCH_YES, on_peer=True)
    assert agent._llm.config.model == "qwen-gguf"
    bus = _publishing(repl, agent)

    await repl._handle_slash(f"/model {PRIMARY_SERVED}")

    event = _only_event(bus)
    assert (event.kind, event.endpoint, event.model) == ("cold", "primary", "qwen-gguf")
    assert agent._llm.config.base_url != PEER_URL and len(launches.cmds) == 1


async def test_phone_typed_gguf_restart_lands_one_cold_event(tmp_path, monkeypatch, launches):
    from tests.unit.test_model_swap_back import _gguf, _llama, _on_a_peer, _ready_as

    path = _gguf(tmp_path)
    _, repl, channel, agent, _peer = _on_a_peer(tmp_path, _llama(served_name="qwen-27b"))
    phone = _Phone(channel, LAUNCH_YES)
    repl._gate = NS(remote_unattended=True)
    _ready_as(monkeypatch, "Other-Q4")
    bus = _publishing(repl, agent)

    await repl._handle_slash(f"/model {path}")

    event = _only_event(bus)
    assert (event.kind, event.endpoint, event.model) == ("cold", "primary", "Other-Q4")
    assert len(phone.asked) == 1 and len(launches.cmds) == 1


async def test_phone_cold_peer_cancelled_publishes_no_event(tmp_path, monkeypatch):
    _, repl, phone, agent, swaps = _phone_cold(tmp_path, monkeypatch, LAUNCH_NO)
    bus = _publishing(repl, agent)

    await repl._handle_slash("/model qwen-gguf")

    assert bus.history(event_types=[ModelSwitched]) == [] and swaps == [] and phone.activity == []


async def test_phone_cold_peer_failed_launch_publishes_no_event_and_clears_the_chip(tmp_path, monkeypatch):
    _, repl, phone, agent, swaps = _phone_cold(
        tmp_path, monkeypatch, LAUNCH_YES, error=RuntimeError("the launch failed"))
    bus = _publishing(repl, agent)

    await repl._handle_slash("/model qwen-gguf")

    assert len(swaps) == 1 and bus.history(event_types=[ModelSwitched]) == []
    assert agent._llm.config.base_url != PEER_URL
    assert phone.activity[0] is not None and phone.activity[-1] is None
