"""ModelSwitch on fakes: the one ModelSwitched per binding change, the launch chip, the strings."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from localharness.agent.gate import PermissionGate
from localharness.channels.mobile.auth import token_path
from localharness.cli import model_switch as ms
from localharness.cli.model_switch import ModelSwitch, launch_holder, short_label
from localharness.cli.workspace import visible_text
from localharness.config.grants import GrantStore
from localharness.config.paths import global_config_dir
from localharness.core.bus import EventBus
from localharness.core.events import ModelSwitched

PRIMARY = "http://primary:8000/v1"
PEER_B = "http://peer-b:8001/v1"
PEER_C = "http://peer-c:8002/v1"


class _Chan:
    def __init__(self):
        self.activity: list[str | None] = []

    def box_activity(self, text):
        self.activity.append(text)


class _Remote(_Chan):
    local_operator = False


class _Picker(_Remote):
    def __init__(self, answer=None, ready=True):
        super().__init__()
        self.answer, self.ready, self.asked = answer, ready, []

    def can_run_picker(self):
        return self.ready

    async def run_picker(self, options, current, **kw):
        self.asked.append((options, current, kw))
        return self.answer


class _Bus:
    def __init__(self, error: Exception | None = None):
        self.events: list = []
        self.error = error

    async def publish(self, event):
        if self.error:
            raise self.error
        self.events.append(event)


def _world(model="qwen3:32b", session="s1", bus=None, chan=None, with_llm=True, gate=lambda: None, config_dir=None, with_harness=True):
    llm = NS(config=NS(base_url=PRIMARY, model=model))
    agent = NS(_llm=llm if with_llm else None, _config=NS(name="root"), current_session_id=session)
    harness = NS(
        provider=NS(base_url=PRIMARY),
        extra_endpoints=[NS(name="peer-b", base_url=PEER_B), NS(name="twin", base_url=PRIMARY)],
        server=None,
    )
    bus = bus or _Bus()
    chan = chan or _Chan()
    return llm, ModelSwitch(agent, bus, chan, harness if with_harness else None, gate, config_dir), bus, chan


async def test_publish_hot_on_a_model_change_at_the_same_endpoint():
    llm, switch, bus, _ = _world()

    async def handler(arg):
        llm.config.model = arg

    await switch.run(handler, "new")
    [event] = bus.events
    assert (event.kind, event.endpoint, event.model, event.agent_id, event.session_id) == ("hot", "primary", "new", "root", "s1")


async def test_publish_cross_when_the_url_moves_even_with_the_same_model_name():
    llm, switch, bus, _ = _world()

    async def handler(arg):
        llm.config.base_url = PEER_B

    await switch.run(handler, "")
    [event] = bus.events
    assert (event.kind, event.endpoint, event.model) == ("cross", "peer-b", "qwen3:32b")


async def test_publish_cold_after_a_launch_and_the_chip_is_pushed_then_cleared():
    llm, switch, bus, chan = _world()
    seen = []

    async def handler(arg):
        switch.launch("big")
        seen.append(switch.launching)
        llm.config.model = "big"

    await switch.run(handler, "big")
    [event] = bus.events
    assert event.kind == "cold"
    assert seen == ["big"] and switch.launching is None
    assert chan.activity == [ms.LAUNCH_CHIP.format(target="big"), None]


async def test_chip_keeps_the_raw_target():
    _, switch, _, chan = _world()

    async def handler(arg):
        switch.launch("/home/u/w/big.gguf")
        assert switch.launching == "/home/u/w/big.gguf"

    await switch.run(handler, "")
    assert chan.activity[0] == "launching /home/u/w/big.gguf"


async def test_publish_nothing_when_the_binding_is_unchanged_and_the_chip_still_clears():
    _, switch, bus, chan = _world()

    async def handler(arg):
        switch.launch("big")

    await switch.run(handler, "")
    assert bus.events == [] and chan.activity == [ms.LAUNCH_CHIP.format(target="big"), None]


async def test_publish_nothing_without_a_session_id():
    llm, switch, bus, _ = _world(session=None)

    async def handler(arg):
        llm.config.model = "new"

    await switch.run(handler, "")
    assert bus.events == []


async def test_publish_nothing_and_no_raise_when_the_agent_has_no_llm():
    _, switch, bus, _ = _world(with_llm=False)
    ran = []

    async def handler(arg):
        ran.append(arg)

    await switch.run(handler, "x")
    assert ran == ["x"] and bus.events == []


async def test_run_reaches_the_handler_and_publishes_nothing_when_the_session_has_no_harness():
    llm, switch, bus, _ = _world(with_harness=False)
    ran = []

    async def handler(arg):
        ran.append(arg)
        llm.config.model = "new"

    await switch.run(handler, "x")
    assert ran == ["x"] and bus.events == []


async def test_publish_then_reraise_when_the_handler_raises_after_landing():
    llm, switch, bus, chan = _world()

    async def handler(arg):
        switch.launch("big")
        llm.config.model = "big"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await switch.run(handler, "")
    assert len(bus.events) == 1 and chan.activity[-1] is None


async def test_chip_clears_when_a_launched_handler_raises_without_landing():
    _, switch, bus, chan = _world()

    async def handler(arg):
        switch.launch("big")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await switch.run(handler, "")
    assert bus.events == [] and chan.activity == [ms.LAUNCH_CHIP.format(target="big"), None]


async def test_publish_nothing_on_a_cancel_and_the_chip_is_left():
    llm, switch, bus, chan = _world()

    async def handler(arg):
        switch.launch("big")
        llm.config.model = "big"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await switch.run(handler, "")
    assert bus.events == [] and chan.activity == [ms.LAUNCH_CHIP.format(target="big")]


async def test_chip_clears_when_publish_raises_and_the_error_propagates():
    llm, switch, _, chan = _world(bus=_Bus(error=OSError("closed")))

    async def handler(arg):
        switch.launch("big")
        llm.config.model = "big"

    with pytest.raises(OSError):
        await switch.run(handler, "")
    assert chan.activity == [ms.LAUNCH_CHIP.format(target="big"), None]


@pytest.mark.parametrize("change", ["landed", "unchanged", "raising"])
async def test_generation_grows_by_two_and_the_stored_payload_is_cleared(change):
    llm, switch, _, _ = _world()
    switch._stored = (0.0, {"rows": []})
    start = switch._generation

    async def handler(arg):
        if change != "unchanged":
            llm.config.model = "new"
        if change == "raising":
            raise RuntimeError("boom")

    if change == "raising":
        with pytest.raises(RuntimeError):
            await switch.run(handler, "")
    else:
        await switch.run(handler, "")
    assert switch._generation == start + 2 and switch._stored is None


async def test_publish_names_the_primary_when_a_peer_shares_its_url():
    llm, switch, bus, _ = _world()

    async def handler(arg):
        llm.config.model = "new"

    await switch.run(handler, "")
    assert bus.events[0].endpoint == "primary"


async def test_publish_shows_a_weights_path_as_its_file_name():
    llm, switch, bus, _ = _world()

    async def handler(arg):
        llm.config.model = "/home/u/w/x.gguf"

    await switch.run(handler, "")
    assert bus.events[0].model == "x.gguf"


async def test_publish_lands_one_event_on_a_real_bus():
    llm, _, _, chan = _world()
    real = EventBus()
    llm, switch, _, _ = _world(bus=real, chan=chan)

    async def handler(arg):
        llm.config.model = "new"

    await switch.run(handler, "")
    assert len(real.history(event_types=[ModelSwitched])) == 1


@pytest.mark.parametrize("raw", ["/a/b.gguf", "~/m/x", "./x", "../up/y", "models/x.gguf"])
def test_short_label_of_a_path_is_its_file_name(raw):
    assert short_label(raw) == raw.rsplit("/", 1)[-1]


@pytest.mark.parametrize("raw", ["Qwen/Qwen3-4B", "qwen3:32b"])
def test_short_label_leaves_a_served_name(raw):
    assert short_label(raw) == raw


def test_strings_equal_the_ruled_text():
    want = {
        "LAUNCH_SLOW": "can take several minutes",
        "LAUNCH_QUESTION": "Launch {target}? {stops}. A model server starts and can take several minutes{build}; once started it cannot be cancelled.",
        "LAUNCH_STOPS": "This stops {holder} for every client",
        "LAUNCH_STOPS_NONE": "Nothing the harness started is stopped",
        "LAUNCH_CHIP": "launching {target}",
        "LAUNCH_YES": "Launch",
        "LAUNCH_NO": "Cancel",
        "LAUNCH_NO_PHONE": "{target} not launched — the app was closed when asked; {current} still serves. Tap it again.",
        "LAUNCH_LOCKED": "Launching a model server from this channel is off — set channels.remote_unattended to true in the machine config to allow it.",
        "LAUNCH_ELSEWHERE": "{target} needs a launch this channel cannot confirm — /model {target} at the terminal launches it{phone}",
        "LAUNCH_ELSEWHERE_PHONE": " (or in the phone app)",
        "RESTART_DOWN_ON_PEER": " The managed server is down — you are still on the endpoint you were using; retry /model to relaunch.",
        "ROW_SETUP": "set up at the terminal",
        "ROW_FOREIGN": "held by another server",
        "ROW_LOCKED": "launch from the terminal",
        "ROW_DIGITS": "a bare number picks by list position — type /model list",
        "ROW_WITHHELD": "[path withheld]",
        "ROW_CUT": " … — type /model list for more",
        "ROW_WHY_MAX": 90,
        "MODEL_SETUP_FOOTER": "Terminal: localharness model --setup",
        "MODEL_FOREIGN_FOOTER": "Stop the other server at this machine; typed /model list shows the command.",
        "MODEL_LOCKED_FOOTER": "Remote launches are off (channels.remote_unattended: false).",
        "MODEL_NOT_HERE_FOOTER": "Type /model list for the path.",
        "MODEL_MALFORMED": "The bound server's reply wasn't understood — check its base_url at this machine, then reopen.",
        "MODEL_UNAVAILABLE": "Model switching is unavailable in this session.",
        "MODEL_LIST_REUSE_S": 15.0,
    }
    assert {name: getattr(ms, name) for name in want} == want


OPEN = NS(remote_unattended=True)
LOCK = NS(remote_unattended=False)


def _refusal(chan, gate=None, config_dir=None, target="mixtral"):
    _, switch, _, _ = _world(chan=chan, gate=lambda: gate, config_dir=config_dir)
    return switch.refusal(target)


def _local(picker=False):
    chan = _Picker() if picker else _Chan()
    chan.local_operator = True
    return chan


@pytest.mark.parametrize("gate", [LOCK, None, OPEN])
def test_refusal_none_for_a_local_operator_whatever_the_gate_says(gate):
    assert _refusal(_local(), gate) is None


def test_refusal_none_for_an_acp_shaped_channel_without_a_picker():
    chan = _local()
    assert not hasattr(chan, "can_run_picker") and _refusal(chan, LOCK) is None


@pytest.mark.parametrize("chan", [_Picker(), _Remote(), _Chan()])
@pytest.mark.parametrize("gate", [LOCK, NS(), None])
def test_locked_gate_object_without_the_lock_attribute_or_no_gate_refuses_a_launch(chan, gate):
    assert _refusal(chan, gate) == ms.LAUNCH_LOCKED


def test_locked_channel_without_local_operator_attribute_is_remote():
    assert not hasattr(_Chan(), "local_operator") and _refusal(_Chan(), LOCK) == ms.LAUNCH_LOCKED


def test_locked_gate_never_attached_still_refuses(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    gate = PermissionGate(boundary=workspace, workspace=workspace, grants=GrantStore(tmp_path / "grants.yaml"),
                          mode="guarded", remote_unattended=False)
    assert gate.channel_name == "none" and not gate._remote_locked()
    assert _refusal(_Picker(), gate) == ms.LAUNCH_LOCKED


@pytest.mark.parametrize("picker", ["missing", "none"])
def test_elsewhere_names_the_target_when_the_channel_cannot_confirm(picker):
    chan = _Remote()
    if picker == "none":
        chan.can_run_picker = None
    assert _refusal(chan, OPEN) == ms.LAUNCH_ELSEWHERE.format(target="mixtral", phone="")


def test_elsewhere_adds_the_phone_clause_when_the_app_is_enrolled(tmp_path):
    token_path(tmp_path).parent.mkdir(parents=True)
    token_path(tmp_path).write_text("t")
    line = _refusal(_Remote(), OPEN, config_dir=tmp_path)
    assert line == ms.LAUNCH_ELSEWHERE.format(target="mixtral", phone=ms.LAUNCH_ELSEWHERE_PHONE)
    assert line.endswith(" (or in the phone app)")


def test_elsewhere_shows_a_control_character_in_the_target_escaped():
    line = _refusal(_Remote(), OPEN, target="evil\x1bx")
    assert line == ms.LAUNCH_ELSEWHERE.format(target=visible_text("evil\x1bx"), phone="") and "\x1b" not in line


def test_elsewhere_phone_check_reads_the_app_token_path(tmp_path):
    assert ms._phone_enrolled(tmp_path) is False and ms._phone_enrolled(None) is False
    assert token_path(tmp_path) == global_config_dir(tmp_path) / "mobile" / "token"
    token_path(tmp_path).parent.mkdir(parents=True)
    token_path(tmp_path).write_text("t")
    assert ms._phone_enrolled(tmp_path) is True


@pytest.mark.parametrize("ready", [True, False])
def test_refusal_none_for_a_remote_channel_with_a_picker_and_the_lock_unset(ready):
    assert _refusal(_Picker(ready=ready), OPEN) is None


def test_never_gated_only_the_phone_channel_has_a_callable_picker():
    from localharness.channels.acp import AcpChannel
    from localharness.channels.mobile.channel import MobileChannel
    from localharness.channels.terminal import TerminalChannel

    dispatch = pytest.importorskip("localharness.dispatch.channel")
    for cls in (dispatch.DispatchChannel, AcpChannel, TerminalChannel):
        assert not callable(getattr(cls, "can_run_picker", None))
    assert callable(MobileChannel.can_run_picker)


@pytest.mark.parametrize("change", ["hot", "cross"])
async def test_never_gated_hot_and_cross_switches_run_from_a_locked_remote_channel(change, monkeypatch):
    def boom(self, target):
        raise AssertionError("refusal consulted")

    monkeypatch.setattr(ModelSwitch, "refusal", boom)
    llm, switch, bus, _ = _world(chan=_Picker(), gate=lambda: LOCK)

    async def handler(arg):
        if change == "hot":
            llm.config.model = "new"
        else:
            llm.config.base_url = PEER_B

    await switch.run(handler, "")
    [event] = bus.events
    assert event.kind == change


HELD = NS(served_name=None, model="/home/u/weights/held.gguf")
HARNESS = NS(provider=NS(base_url=PRIMARY, api_key=None), extra_endpoints=[])


def _asker(answer, ready=True, model="qwen3:32b"):
    chan = _Picker(answer, ready)
    llm, switch, _, _ = _world(model=model, chan=chan)
    return chan, switch


def _recheck(result=None):
    calls = []

    async def recheck():
        calls.append(1)
        return result

    return calls, recheck


async def test_confirm_asks_once_with_the_whole_question_and_returns_the_recheck():
    chan, switch = _asker(ms.LAUNCH_YES)
    calls, recheck = _recheck()
    assert await switch.confirm("mixtral", "qwen3:32b", " (first build x)", recheck) is None
    assert chan.asked == [([ms.LAUNCH_YES, ms.LAUNCH_NO], "qwen3:32b", {
        "prompt": "Launch mixtral? This stops qwen3:32b for every client. A model server starts and can take several minutes"
                  " (first build x); once started it cannot be cancelled.",
        "confirm": ms.LAUNCH_YES})]
    assert calls == [1]


async def test_confirm_without_a_holder_says_nothing_is_stopped():
    chan, switch = _asker(ms.LAUNCH_YES)
    await switch.confirm("mixtral", None, "", _recheck()[1])
    assert chan.asked[0][2]["prompt"] == ms.LAUNCH_QUESTION.format(target="mixtral", stops=ms.LAUNCH_STOPS_NONE, build="")


async def test_confirm_shows_control_characters_in_the_holder_escaped():
    chan, switch = _asker(ms.LAUNCH_YES)
    await switch.confirm("mixtral", "q\x1b[2J\nfake\u202e", "", _recheck()[1])
    prompt = chan.asked[0][2]["prompt"]
    assert "q\\x1b[2J\\x0afake\\u202e" in prompt
    assert not any(ord(c) < 0x20 or c == "\u202e" for c in prompt)


async def test_confirm_returns_a_refusal_that_appeared_during_the_wait_unchanged():
    _, switch = _asker(ms.LAUNCH_YES)
    _, recheck = _recheck(("refused now", "system.error"))
    assert await switch.confirm("mixtral", None, "", recheck) == ("refused now", "system.error")


@pytest.mark.parametrize("answer", [ms.LAUNCH_NO, None])
async def test_confirm_cancel_or_no_answer_stops_with_nothing_sent_and_no_recheck(answer):
    _, switch = _asker(answer)
    calls, recheck = _recheck()
    assert await switch.confirm("mixtral", None, "", recheck) == ("", "") and calls == []


async def test_confirm_a_closed_app_says_so_without_asking_or_rechecking():
    chan, switch = _asker(ms.LAUNCH_YES, ready=False)
    calls, recheck = _recheck()
    assert await switch.confirm("mixtral", None, "", recheck) == (
        ms.LAUNCH_NO_PHONE.format(target="mixtral", current="qwen3:32b"), "system.info")
    assert chan.asked == [] and calls == []


async def test_confirm_for_a_local_operator_neither_asks_nor_rechecks():
    chan = _local(picker=True)
    chan.answer = ms.LAUNCH_YES
    _, switch, _, _ = _world(chan=chan)
    calls, recheck = _recheck()
    assert await switch.confirm("mixtral", None, "", recheck) is None
    assert chan.asked == [] and calls == []


async def test_confirm_question_holds_no_directory_while_current_stays_raw():
    chan, switch = _asker(ms.LAUNCH_YES, model="/home/u/live/serving.gguf")
    holder = launch_holder((HELD, PRIMARY), HARNESS)
    await switch.confirm("/home/u/w/big.gguf", holder, "", _recheck()[1])
    [(_, current, kw)] = chan.asked
    assert "/home/u" not in kw["prompt"] and "big.gguf" in kw["prompt"] and "held.gguf" in kw["prompt"]
    assert current == "/home/u/live/serving.gguf"


async def test_confirm_closed_app_line_escapes_the_target_and_keeps_the_raw_current():
    _, switch = _asker(ms.LAUNCH_YES, ready=False, model="/home/u/live/serving.gguf")
    line, style = await switch.confirm("evil\x1bx", None, "", _recheck()[1])
    assert line == ms.LAUNCH_NO_PHONE.format(target=visible_text("evil\x1bx"), current="/home/u/live/serving.gguf")
    assert style == "system.info" and "\x1b" not in line


@pytest.mark.parametrize("args", [(None, HARNESS), (None, HARNESS, None)])
def test_holder_none_when_nothing_is_tracked(args):
    assert launch_holder(*args) is None


@pytest.mark.parametrize("model, name", [("/home/u/weights/held.gguf", "held.gguf"), ("models/x.gguf", "x.gguf")])
def test_holder_reduces_a_weights_path_to_its_file_name(model, name):
    assert launch_holder((NS(served_name=None, model=model), PRIMARY), HARNESS) == name


def test_holder_names_a_peer_by_its_peer_name():
    spec = NS(served_name=None, model="/m/p.gguf")
    harness = NS(provider=NS(base_url=PRIMARY, api_key=None), extra_endpoints=[NS(name="peer-c", base_url=PEER_C, lifecycle=spec)])
    assert launch_holder((spec, PEER_C), harness) == "peer-c"


def test_holder_prefers_the_active_heavy_over_the_fallback():
    fallback = (NS(served_name="srv", model="/m/x.gguf"), PRIMARY)
    assert launch_holder(None, HARNESS, fallback) == "srv"
    assert launch_holder((HELD, PRIMARY), HARNESS, fallback) == "held.gguf"


PATH_SHAPES = [
    "at /opt/x — fix", "(/opt/x) — fix", "'/opt/x' — fix", "dir=/opt/x — fix", "[/opt/x] — fix",
    "<path>/opt/x</path> — fix", "`/opt/x` — fix", "~/m — fix", "./rel/x — fix", "../up — fix",
    "at /home/u/My Engines/x no longer passes — fix", "at /home/o'brien/x — fix", "dir=/opt/x\nnext line — fix",
]


@pytest.mark.parametrize("text", PATH_SHAPES)
def test_path_token_withholds_the_path_and_keeps_the_fix(text):
    out = ms.PATH_TOKEN.sub(ms.ROW_WITHHELD, text)
    assert ms.ROW_WITHHELD in out and out.endswith(" — fix")
    for fragment in ("opt", "rel", "up", "My Engines", "brien", "home", "next line"):
        assert fragment not in out


def test_path_token_withholds_an_address():
    assert ms.PATH_TOKEN.sub(ms.ROW_WITHHELD, "see http://h:8000/v1 now") == "see [path withheld] now"


@pytest.mark.parametrize("text", ["(/proc/meminfo)", "read /sys/class/x — fix", "Qwen/Qwen3-4B"])
def test_path_token_leaves_proc_sys_and_hub_ids(text):
    assert ms.PATH_TOKEN.sub(ms.ROW_WITHHELD, text) == text


def test_cut_keeps_a_short_reason_and_a_long_one_with_no_dash():
    short, long_ = "x" * 30 + " — " + "y" * 30, "z" * 120
    assert (ms._why_text(short), ms._why_text(long_)) == (short, long_)


def test_cut_keeps_a_first_clause_over_the_limit_and_points_at_the_list():
    from localharness.cli.model_ops import not_offered_reason
    from localharness.provider.autosense import NGL_REFUSAL, RAM_UNKNOWN_REFUSAL

    ngl = not_offered_reason(NGL_REFUSAL.format(name="m"), "m")
    assert ms._why_text(ngl) == ngl.split(" — ")[0] + ms.ROW_CUT and len(ngl.split(" — ")[0]) > ms.ROW_WHY_MAX
    ram = not_offered_reason(RAM_UNKNOWN_REFUSAL.format(name="m", need=30), "m")
    assert ms._why_text(ram) == ram.split(" — ")[0] + ms.ROW_CUT and "(/proc/meminfo)" in ms._why_text(ram)


def test_cut_screens_a_raw_newline_before_escaping_and_escapes_control_characters():
    assert ms._why_text("dir=/opt/x\nnext line — fix") == "dir=[path withheld] — fix"
    assert "\x1b" not in ms._why_text("bad\x1bthing")


STRATA_DIRS = ("/home/u/My Engines/strata", "/home/o'brien/strata")
LIVE_PATH = "/home/u/live/serving.gguf"
INCUMBENT = "available once llm-service.service (pid 48213) is stopped"
KEY = "sk-held-secret"


def _strata_refusal(tmp_path, name, engine_dir):
    from localharness.provider import strata

    spec = NS(served_name=name, model="/w/x.gguf", engine_dir=engine_dir, engine_config=engine_dir + "/c.json")
    return strata.verify_recorded(tmp_path, spec, None, None)


def _payload_world(tmp_path, with_ckpt=True):
    from localharness.cli.model_ops import Entries, EntryGroup, PlanEntry, ServingRecord

    peer_b = NS(name="peer-b", base_url=PEER_B, api_key=KEY)
    peer_c = NS(name="peer-c", base_url=PEER_C, lifecycle=NS())
    held = NS(served_name=None, model="/home/u/weights/held.gguf")
    harness = NS(provider=NS(base_url=PRIMARY, api_key="none"), extra_endpoints=[peer_b, peer_c], server=held)

    def record(name, peer):
        return ServingRecord(name, name, PEER_C, "llamacpp", peer, None, "p")

    sentence = f"{INCUMBENT}: run systemctl --user stop llm-service.service"
    groups = (
        EntryGroup("llama.cpp · primary", (
            PlanEntry(LIVE_PATH, "live", "llamacpp", active=True),
            PlanEntry("llama3.1:8b", "live", "llamacpp", endpoint=peer_b)), True),
        EntryGroup("this box can launch", (
            PlanEntry("mixtral", "cold", "llamacpp", record=record("mixtral", NS(name="peer-c"))),
            PlanEntry("qwen-27b", "cold", "llamacpp", record=record("qwen-27b", None)),
            PlanEntry("Qwen/Qwen3-4B", "setup", "vllm"),
            PlanEntry("70", "cold", "llamacpp")), True),
        EntryGroup(INCUMBENT, (PlanEntry("held-model", "cold", "llamacpp", refusal=sentence),), False))
    refused = [PlanEntry(n, "cold", "strata", refusal=_strata_refusal(tmp_path, n, d))
               for n, d in zip(("strata-b", "strata-a"), STRATA_DIRS[::-1])]
    if with_ckpt:
        refused.append(PlanEntry("ckpt", "cold", "vllm", refusal="ckpt cannot run here: its checkpoint at /home/u/ckpt/model is not readable — fix its permissions."))
    entries = Entries(groups, not_offered=tuple(refused), notes=("skipped peer-d (http://10.0.0.9:8000/v1): unreachable",), foreign=sentence)
    llm = NS(config=NS(base_url=PRIMARY, model=LIVE_PATH))
    return entries, llm, harness, (held, PRIMARY)


def _payload(tmp_path, locked=False, launching=None, with_ckpt=True):
    entries, llm, harness, heavy = _payload_world(tmp_path, with_ckpt)
    return ms.list_payload(entries, llm=llm, harness=harness, active_heavy=heavy, locked=locked, launching=launching), harness, heavy


def test_payload_has_the_five_keys_and_seven_per_row(tmp_path):
    payload, _, _ = _payload(tmp_path)
    assert set(payload) == {"active", "endpoint", "launching", "models", "footer"}
    assert all(set(r) == {"name", "label", "kind", "endpoint", "active", "stops", "why"} for r in payload["models"])
    assert (payload["active"], payload["endpoint"], payload["launching"]) == ("serving.gguf", "primary", None)


def test_payload_rows_follow_display_order_then_the_not_offered_by_name(tmp_path):
    rows = _payload(tmp_path)[0]["models"]
    assert [r["name"] for r in rows] == [LIVE_PATH, "llama3.1:8b", "mixtral", "qwen-27b", "Qwen/Qwen3-4B", "70", "held-model", "ckpt", "strata-a", "strata-b"]
    assert [r["kind"] for r in rows] == ["hot", "cross", "cold", "cold", "setup", "cold", "cold", "cold", "cold", "cold"]
    assert [r["endpoint"] for r in rows][:6] == ["primary", "peer-b", "peer-c", "primary", "primary", "primary"]
    assert [r["active"] for r in rows] == [True] + [False] * 9
    assert rows[0]["label"] == "serving.gguf"


def test_payload_launching_is_the_raw_target(tmp_path):
    assert _payload(tmp_path, launching="/home/u/w/big.gguf")[0]["launching"] == "/home/u/w/big.gguf"


def test_payload_stops_names_the_holder_on_offered_cold_rows_only(tmp_path):
    payload, harness, heavy = _payload(tmp_path)
    stops = {r["name"]: r["stops"] for r in payload["models"]}
    assert stops["mixtral"] == launch_holder(heavy, harness) == "held.gguf"
    assert stops["qwen-27b"] == launch_holder(heavy, harness, (harness.server, PRIMARY)) == "held.gguf"
    for name in (LIVE_PATH, "llama3.1:8b", "Qwen/Qwen3-4B", "held-model", "ckpt", "strata-a", "strata-b"):
        assert stops[name] is None


def test_payload_stops_shows_control_and_bidi_characters_in_the_holder_escaped(tmp_path):
    entries, llm, harness, _ = _payload_world(tmp_path)
    held = NS(served_name=None, model="/w/a\x1b[2J\r\nb\u202e.gguf")
    payload = ms.list_payload(entries, llm=llm, harness=harness, active_heavy=(held, PRIMARY), locked=False, launching=None)
    stops = {r["name"]: r["stops"] for r in payload["models"]}
    for name in ("mixtral", "qwen-27b"):
        assert stops[name] == "a\\x1b[2J\\x0d\\x0ab\\u202e.gguf"
        assert not any(ord(c) < 0x20 or c == "\u202e" for c in stops[name])


def test_payload_why_follows_setup_digits_foreign_not_offered(tmp_path):
    why = {r["name"]: r["why"] for r in _payload(tmp_path)[0]["models"]}
    assert (why["Qwen/Qwen3-4B"], why["70"], why["held-model"]) == (ms.ROW_SETUP, ms.ROW_DIGITS, ms.ROW_FOREIGN)
    assert why["ckpt"] == "its checkpoint at [path withheld] — fix its permissions"
    for name in ("strata-a", "strata-b"):
        assert "strata" in why[name] and "owner and mode checks" in why[name]
    assert why["strata-a"].startswith("its Strata install at strata no longer passes the owner and mode checks")
    assert [why[n] for n in (LIVE_PATH, "llama3.1:8b", "mixtral", "qwen-27b")] == [None] * 4


def test_payload_locked_marks_offered_cold_rows_and_keeps_not_offered_reasons(tmp_path):
    unlocked = {r["name"]: r for r in _payload(tmp_path)[0]["models"]}
    rows = {r["name"]: r for r in _payload(tmp_path, locked=True)[0]["models"]}
    assert (rows["mixtral"]["why"], rows["qwen-27b"]["why"]) == (ms.ROW_LOCKED, ms.ROW_LOCKED)
    assert (rows["mixtral"]["stops"], rows["qwen-27b"]["stops"]) == ("held.gguf", "held.gguf")
    assert rows["70"]["why"] == ms.ROW_DIGITS
    for name in ("ckpt", "strata-a", "strata-b", "held-model"):
        assert rows[name] == unlocked[name]


def test_payload_footer_holds_each_line_once_in_order(tmp_path):
    assert _payload(tmp_path)[0]["footer"] == [ms.MODEL_SETUP_FOOTER, ms.MODEL_FOREIGN_FOOTER, ms.MODEL_NOT_HERE_FOOTER]
    assert _payload(tmp_path, locked=True)[0]["footer"] == [
        ms.MODEL_SETUP_FOOTER, ms.MODEL_FOREIGN_FOOTER, ms.MODEL_LOCKED_FOOTER, ms.MODEL_NOT_HERE_FOOTER]
    assert ms.MODEL_NOT_HERE_FOOTER not in _payload(tmp_path, with_ckpt=False)[0]["footer"]


FORBIDDEN = [
    PRIMARY, PEER_B, PEER_C, "10.0.0.9", "127.0.0.1", KEY, "48213", "llm-service", "systemctl",
    "/home/u/weights", "/home/u/My Engines", "o'brien", "available once", "turn_running",
]


@pytest.mark.parametrize("item", FORBIDDEN)
def test_payload_privacy_holds_no_address_key_pid_unit_or_directory(tmp_path, item):
    assert item not in json.dumps(_payload(tmp_path)[0])


def test_payload_privacy_holds_no_config_dir(tmp_path):
    assert str(tmp_path) not in json.dumps(_payload(tmp_path)[0])


def test_payload_privacy_shows_the_live_weights_directory_only_in_its_own_name(tmp_path):
    payload = _payload(tmp_path)[0]
    assert "/home/u/live" not in payload["active"]
    for r in payload["models"]:
        assert all("/home/u/live" not in str(v) for k, v in r.items() if k != "name")
    assert payload["models"][0]["name"] == LIVE_PATH


def test_payload_bidi_override_is_escaped_in_the_label_and_raw_in_the_name(tmp_path):
    from localharness.cli.model_ops import Entries, EntryGroup, PlanEntry

    entries = Entries((EntryGroup("g", (PlanEntry("evil‮name", "live", "llamacpp"),), True),))
    _, llm, harness, heavy = _payload_world(tmp_path)
    [row] = ms.list_payload(entries, llm=llm, harness=harness, active_heavy=heavy, locked=False, launching=None)["models"]
    assert "‮" not in row["label"] and row["name"] == "evil‮name"


def _list_entries():
    from localharness.cli.model_ops import Entries, EntryGroup, PlanEntry, ServingRecord

    peer_b = NS(name="peer-b", base_url=PEER_B)
    peer_c = NS(name="peer-c", base_url=PEER_C, lifecycle=NS())
    mixtral = ServingRecord("mixtral", "mixtral", PEER_C, "llamacpp", peer_c, None, "p")
    return Entries((EntryGroup("g", (
        PlanEntry("qwen3:32b", "live", "llamacpp", active=True),
        PlanEntry("llama3.1:8b", "live", "llamacpp", endpoint=peer_b),
        PlanEntry("mixtral", "cold", "llamacpp", record=mixtral)), True),))


class _Plan:
    def __init__(self, gates=(), error=None, malformed=()):
        self.calls, self.gates, self.error, self.malformed = 0, list(gates), error, set(malformed)

    async def __call__(self, llm):
        n = self.calls
        self.calls += 1
        if n < len(self.gates):
            await self.gates[n].wait()
        if self.error:
            raise self.error
        return _list_entries(), {}, self.malformed


async def _settle():
    for _ in range(8):
        await asyncio.sleep(0)


def _listing(chan=None, gate=lambda: None, tmp=None):
    return _world(chan=chan or _local(), gate=gate, config_dir=tmp)[1]


async def _noop(arg):
    return None


def _clock(monkeypatch, start=100.0):
    now = [start]
    monkeypatch.setattr(ms, "time", NS(monotonic=lambda: now[0]))
    return now


def _names(payload):
    return [r["name"] for r in payload["models"]]


@pytest.mark.parametrize("missing", ["llm", "harness", "config_dir"])
async def test_model_list_unavailable_answers_without_planning(tmp_path, missing):
    _, switch, _, _ = _world(chan=_local(), config_dir=None if missing == "config_dir" else tmp_path, with_llm=missing != "llm")
    if missing == "harness":
        switch._harness = None
    plan = _Plan()
    assert await switch.model_list(plan, None) == {"active": None, "models": [], "footer": [ms.MODEL_UNAVAILABLE]}
    assert plan.calls == 0


async def test_model_list_concurrent_calls_plan_once_and_agree(tmp_path):
    switch, plan = _listing(tmp=tmp_path), _Plan([asyncio.Event()])
    a, b = (asyncio.ensure_future(switch.model_list(plan, None)) for _ in range(2))
    await _settle()
    plan.gates[0].set()
    first, second = await asyncio.gather(a, b)
    assert plan.calls == 1 and first == second and _names(first) == ["qwen3:32b", "llama3.1:8b", "mixtral"]


async def test_model_list_reuses_inside_the_window_and_plans_again_at_its_edge(tmp_path, monkeypatch):
    now, switch, plan = _clock(monkeypatch), _listing(tmp=tmp_path), _Plan()
    first = await switch.model_list(plan, None)
    now[0] = 114.9
    assert await switch.model_list(plan, None) == first and plan.calls == 1
    now[0] = 115.0
    await switch.model_list(plan, None)
    assert plan.calls == 2


async def test_model_list_plans_again_after_a_run(tmp_path):
    switch, plan = _listing(tmp=tmp_path), _Plan()
    await switch.model_list(plan, None)
    await switch.run(_noop, "")
    await switch.model_list(plan, None)
    assert plan.calls == 2


async def test_model_list_during_a_launch_plans_nothing_and_serves_the_stored_rows(tmp_path):
    switch, plan, seen = _listing(tmp=tmp_path), _Plan(), []
    before = await switch.model_list(plan, None)

    async def handler(arg):
        switch.launch("/home/u/w/big.gguf")
        seen.append(await switch.model_list(plan, None))

    await switch.run(handler, "")
    [during] = seen
    assert plan.calls == 1 and during["launching"] == "/home/u/w/big.gguf"
    assert during["models"] == before["models"] and before["launching"] is None


async def test_model_list_during_a_launch_with_nothing_stored_has_the_label_and_no_rows(tmp_path):
    _, switch, _, _ = _world(model="/home/u/w/held.gguf", chan=_local(), config_dir=tmp_path)
    plan, seen = _Plan(), []

    async def handler(arg):
        switch.launch("/home/u/w/big.gguf")
        seen.append(await switch.model_list(plan, None))

    await switch.run(handler, "")
    assert plan.calls == 0
    assert seen == [{"active": "held.gguf", "endpoint": "primary", "launching": "/home/u/w/big.gguf", "models": [], "footer": []}]


async def test_model_list_straddling_a_run_is_returned_but_never_stored(tmp_path):
    switch, plan = _listing(tmp=tmp_path), _Plan([asyncio.Event(), asyncio.Event()])
    a = asyncio.ensure_future(switch.model_list(plan, None))
    await _settle()
    await switch.run(_noop, "")
    b = asyncio.ensure_future(switch.model_list(plan, None))
    await _settle()
    assert plan.calls == 2
    plan.gates[0].set()
    assert _names(await a) and switch._stored is None
    plan.gates[1].set()
    fresh = await b
    assert switch._stored is not None and switch._stored[1]["models"] == fresh["models"]


async def test_model_list_a_raising_plan_reaches_every_joiner_and_the_next_call_plans_again(tmp_path):
    switch, plan = _listing(tmp=tmp_path), _Plan([asyncio.Event()], error=RuntimeError("down"))
    calls = [asyncio.ensure_future(switch.model_list(plan, None)) for _ in range(2)]
    await _settle()
    plan.gates[0].set()
    results = await asyncio.gather(*calls, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
    assert switch._inflight is None and switch._stored is None
    plan.error = None
    assert _names(await switch.model_list(plan, None)) and plan.calls == 2


async def test_model_list_a_cancelled_joiner_leaves_the_shared_plan_running(tmp_path):
    switch, plan = _listing(tmp=tmp_path), _Plan([asyncio.Event()])
    a, b = (asyncio.ensure_future(switch.model_list(plan, None)) for _ in range(2))
    await _settle()
    a.cancel()
    await _settle()
    plan.gates[0].set()
    assert _names(await b) and plan.calls == 1
    with pytest.raises(asyncio.CancelledError):
        await a


async def test_model_list_reply_carries_the_launching_of_the_moment_without_touching_the_store(tmp_path):
    switch, plan = _listing(tmp=tmp_path), _Plan()
    await switch.model_list(plan, None)
    switch.launching = "x"
    assert (await switch.model_list(plan, None))["launching"] == "x"
    assert switch._stored[1]["launching"] is None


async def test_model_list_malformed_bound_server_keeps_only_the_cross_rows(tmp_path):
    switch, plan = _listing(tmp=tmp_path), _Plan(malformed={ms.model_ops._norm_url(PRIMARY)})
    payload = await switch.model_list(plan, None)
    assert _names(payload) == ["llama3.1:8b"] and [r["kind"] for r in payload["models"]] == ["cross"]
    assert payload["footer"] == [ms.MODEL_MALFORMED]
    assert PRIMARY not in json.dumps(payload) and "primary:8000" not in json.dumps(payload)
    assert switch._stored[1]["footer"] == [ms.MODEL_MALFORMED]


def test_malformed_line_names_no_address():
    assert "http" not in ms.MODEL_MALFORMED


async def test_model_list_malformed_naming_another_endpoint_changes_nothing(tmp_path):
    clean = await _listing(tmp=tmp_path).model_list(_Plan(), None)
    other = await _listing(tmp=tmp_path).model_list(_Plan(malformed={ms.model_ops._norm_url(PEER_B)}), None)
    assert other == clean and len(other["models"]) == 3


async def test_model_list_locked_marks_the_cold_rows_for_a_picker_channel_under_the_lock(tmp_path):
    _, switch, _, _ = _world(chan=_Picker(), gate=lambda: NS(remote_unattended=False), config_dir=tmp_path)
    payload = await switch.model_list(_Plan(), None)
    why = {r["name"]: r["why"] for r in payload["models"]}
    assert why == {"qwen3:32b": None, "llama3.1:8b": None, "mixtral": ms.ROW_LOCKED}
    assert payload["footer"] == [ms.MODEL_LOCKED_FOOTER]


async def test_model_list_locked_marks_the_cold_rows_for_a_channel_with_no_picker(tmp_path):
    _, switch, _, _ = _world(chan=_Remote(), gate=lambda: OPEN, config_dir=tmp_path)
    payload = await switch.model_list(_Plan(), None)
    assert [r["why"] for r in payload["models"]] == [None, None, ms.ROW_LOCKED]
    assert payload["footer"] == [ms.MODEL_LOCKED_FOOTER]


async def test_model_list_locked_is_absent_for_the_local_operator(tmp_path):
    _, switch, _, _ = _world(chan=_local(), gate=lambda: LOCK, config_dir=tmp_path)
    payload = await switch.model_list(_Plan(), None)
    assert [r["why"] for r in payload["models"]] == [None] * 3 and payload["footer"] == []


async def test_run_without_a_session_id_runs_the_handler_and_publishes_nothing():
    llm = NS(config=NS(base_url=PRIMARY, model="old"))
    harness = NS(provider=NS(base_url=PRIMARY), extra_endpoints=[], server=None)
    bus = _Bus()
    switch = ModelSwitch(NS(_llm=llm, _config=NS(name="root")), bus, _Chan(), harness, lambda: None, None)
    ran = []

    async def handler(arg):
        ran.append(arg)
        llm.config.model = "new"

    await switch.run(handler, "new")
    assert ran == ["new"] and bus.events == []
