"""ModelSwitch on fakes: the one ModelSwitched per binding change, the launch chip, the strings."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from localharness.cli import model_switch as ms
from localharness.agent.gate import PermissionGate
from localharness.channels.mobile.auth import token_path
from localharness.cli.model_switch import ModelSwitch, short_label
from localharness.cli.workspace import visible_text
from localharness.config.grants import GrantStore
from localharness.config.paths import global_config_dir
from localharness.core.bus import EventBus
from localharness.core.events import ModelSwitched

PRIMARY = "http://primary:8000/v1"
PEER_B = "http://peer-b:8001/v1"


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


def _world(model="qwen3:32b", session="s1", bus=None, chan=None, with_llm=True, gate=lambda: None, config_dir=None):
    llm = NS(config=NS(base_url=PRIMARY, model=model))
    agent = NS(_llm=llm if with_llm else None, _config=NS(name="root"), current_session_id=session)
    harness = NS(
        provider=NS(base_url=PRIMARY),
        extra_endpoints=[NS(name="peer-b", base_url=PEER_B), NS(name="twin", base_url=PRIMARY)],
        server=None,
    )
    bus = bus or _Bus()
    chan = chan or _Chan()
    return llm, ModelSwitch(agent, bus, chan, harness, gate, config_dir), bus, chan


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
