"""`lifecycle.swap`: the REPL's cold swap lifted into the provider layer (spec 20 §8.2, D-19..D-22).

The server boundaries are faked (`stop_server`, `start_server`, `wait_ready` record into one ordered
event list, `GPU_FREE_SETTLE_SECONDS` is 0), the harness is loaded from a tmp config dir, trust is a
constructed `MachineTrust`, and nothing is started or touched: no service manager, no real port.
"""
from __future__ import annotations

import asyncio

import pytest
import yaml

from localharness.cli import model_ops
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle, units
from localharness.provider import server as managed_server
from localharness.provider.lifecycle import Heavy, SwapOutcome, swap

PRIMARY = "qwen-27b-swift-1.5"
PRIMARY_URL = "http://127.0.0.1:8080/v1"
PEER_URL = "http://127.0.0.1:8081/v1"
PROVIDER_KEY = "sk-prov-7d3f9a"


def _seed(home, *, peer_lifecycle=True) -> None:
    peer = {"name": "flash-next", "base_url": PEER_URL, "provider_type": "llamacpp"}
    if peer_lifecycle:
        peer["gpu"] = True
        peer["lifecycle"] = {"gpu": True, "runtime": "llamacpp", "binary": "/x/llama-server",
                             "model": "/m/flash-next.gguf", "port": 8081}
    data = {
        "version": "1",
        "provider": {"provider_type": "llamacpp", "base_url": PRIMARY_URL, "default_model": PRIMARY,
                     "available_models": [PRIMARY], "api_key": PROVIDER_KEY},
        "org": {"default_model": PRIMARY, "audit_log_path": str(home / "audit.jsonl")},
        "extra_endpoints": [peer],
        "server": {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                   "served_name": PRIMARY, "port": 8080},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


class Rig:
    """The three process boundaries a swap crosses, recorded in order; `wait_raises` maps a base URL
    to the exception its readiness wait raises."""

    def __init__(self, home) -> None:
        self.home = home
        self.events: list[str] = []
        self.wait_raises: dict[str, BaseException] = {}
        self.stop_raises: list[BaseException] = []
        self.wait_kw: list[dict] = []
        _seed(home)
        self.harness = ConfigLoader(config_dir=home).load_harness()

    def serving(self, preset="flash-next"):
        got = model_ops.resolve_serving(self.harness, MachineTrust(), preset, config_dir=self.home)
        assert isinstance(got, model_ops.Serving), got
        return got

    def target(self) -> Heavy:
        return model_ops.heavy_for(self.serving(), self.harness)

    def incumbent(self) -> Heavy:
        return model_ops.incumbent_heavy((self.harness.server, self.harness.provider.base_url), self.harness)

    def redact(self, text: str) -> str:
        return model_ops.redact_for_channel(text, self.harness)


class Reporter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.lines: list[str] = []
        self.live = None
        self.restored_specs: list = []

    async def freed(self) -> None:
        self.calls.append("freed")

    async def launched(self, live) -> None:
        self.calls.append("launched")
        self.live = live

    async def restored(self, spec) -> None:
        self.calls.append("restored")
        self.restored_specs.append(spec)

    async def line(self, text: str) -> None:
        self.calls.append("line")
        self.lines.append(text)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    home = tmp_path / ".localharness"
    home.mkdir()
    r = Rig(home)
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])

    def fake_stop(cfg, launch="binary", **kw):
        r.events.append("stop")
        if r.stop_raises:
            raise r.stop_raises.pop(0)
        return True

    def fake_start(cfg, cmd, **kw):
        r.events.append("start:" + " ".join(cmd))
        return 4321

    async def fake_wait(base_url, config_dir=None, **kw):
        r.events.append(f"wait:{base_url}")
        r.wait_kw.append(kw)
        if base_url in r.wait_raises:
            raise r.wait_raises[base_url]
        return [PRIMARY if base_url == PRIMARY_URL else "flash-next"]

    monkeypatch.setattr(managed_server, "stop_server", fake_stop)
    monkeypatch.setattr(managed_server, "start_server", fake_start)
    monkeypatch.setattr(managed_server, "wait_ready", fake_wait)
    # a window probe / nvidia-smi is never asked: autosense is None in every swap here
    return r


async def _swap(rig: Rig, incumbent, target, reporter, **kw) -> SwapOutcome:
    return await swap(incumbent, target, config_dir=rig.home, autosense=None, reporter=reporter,
                      redact=rig.redact, **kw)


# --- the tracer ---------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_trust_resolved_target_swaps_in_incumbent_freed_target_launched_reporter_told(rig):
    incumbent, target, rec = rig.incumbent(), rig.target(), Reporter()
    assert target.peer == "flash-next" and incumbent.peer is None

    outcome = await _swap(rig, incumbent, target, rec)

    assert rig.events[0] == "stop"
    assert rig.events[1].startswith("start:") and "/m/flash-next.gguf" in rig.events[1]
    assert rig.events[2:] == [f"wait:{PEER_URL}"]
    assert rec.calls == ["freed", "launched"]
    assert outcome.live.base_url == PEER_URL and rec.live is outcome.live
    assert outcome.stopped == incumbent
    assert outcome.error is None and outcome.stage is None and outcome.restored is None


@pytest.mark.asyncio
async def test_with_nothing_to_stop_the_swap_still_reports_the_accelerator_free(rig):
    rec = Reporter()
    outcome = await _swap(rig, None, rig.target(), rec)
    assert [e.split(":")[0] for e in rig.events] == ["start", "wait"]
    assert rec.calls == ["freed", "launched"]
    assert outcome.stopped is None and outcome.error is None


def test_swap_without_redact_is_a_type_error(rig):
    with pytest.raises(TypeError):
        swap(rig.incumbent(), rig.target(), config_dir=rig.home, autosense=None, reporter=Reporter())


def test_heavy_for_a_record_with_no_lifecycle_raises_value_error(tmp_path):
    home = tmp_path / ".localharness"
    home.mkdir()
    _seed(home, peer_lifecycle=False)
    harness = ConfigLoader(config_dir=home).load_harness()
    record = next(r for r in model_ops.serving_records(harness) if r.name == "flash-next")
    assert record.spec is None
    with pytest.raises(ValueError):
        model_ops.heavy_for(model_ops.Serving(record, model_ops.RUNG_PRESET, True), harness)


def test_redact_for_channel_masks_a_bearer_token(rig):
    assert model_ops.redact_for_channel("Bearer abc", rig.harness) == "Bearer **********"


def test_heavy_names_a_peer_only_by_identity_and_keeps_its_key_out_of_the_repr(rig):
    spec_copy = rig.harness.extra_endpoints[0].lifecycle.model_copy()
    assert rig.incumbent().peer is None
    peer_heavy = model_ops.incumbent_heavy(
        (rig.harness.extra_endpoints[0].lifecycle, PEER_URL), rig.harness)
    assert peer_heavy.peer == "flash-next"
    assert model_ops.incumbent_heavy((spec_copy, PEER_URL), rig.harness).peer is None  # equal, not the same
    keyed = Heavy(spec_copy, PEER_URL, "flash-next", {"api_key": "k-secret-xyz"})
    assert "k-secret-xyz" not in repr(keyed)
    with pytest.raises(TypeError):
        keyed.launch_kw["api_key"] = "other"  # read-only


@pytest.mark.asyncio
async def test_the_targets_launch_key_reaches_the_launch_and_the_readiness_wait(rig):
    target = Heavy(rig.target().spec, PEER_URL, "flash-next", {"api_key": "k-launch-1"})
    await _swap(rig, None, target, Reporter())
    assert rig.wait_kw[-1].get("api_key") == "k-launch-1"


# --- failure, restore and cancellation contracts ------------------------------------------------- #

FLASH_FAILED = "llama.cpp exited during startup. Log tail:\nboom"
PLANTED = (f"stopped: Bearer sk-planted\napi_keys: ****abcd\nvia http://u:p@127.0.0.1:8081/v1 "
           f"with {PROVIDER_KEY}")
PLANTED_PARTS = ("sk-planted", "abcd", "u:p", PROVIDER_KEY)


def _launches(rig: Rig) -> list[str]:
    return [e for e in rig.events if e.startswith("start:")]


@pytest.mark.asyncio
async def test_a_free_failure_launches_nothing_and_says_the_current_model_still_serves(rig):
    rig.stop_raises.append(RuntimeError("won't die"))
    rec = Reporter()

    outcome = await _swap(rig, rig.incumbent(), rig.target(), rec)

    assert rig.events == ["stop"]
    assert rec.calls == ["line"]
    assert rec.lines == ["Could not free the GPU for flash-next: won't die. No change made — "
                         "the current model should still be serving."]
    assert outcome.stage == "free" and outcome.live is None and outcome.stopped is None
    assert isinstance(outcome.error, RuntimeError) and str(outcome.error) == "won't die"


@pytest.mark.asyncio
async def test_a_launch_failure_stops_the_target_then_restores_the_incumbent(rig):
    rig.wait_raises[PEER_URL] = RuntimeError(FLASH_FAILED)
    incumbent, rec = rig.incumbent(), Reporter()

    outcome = await _swap(rig, incumbent, rig.target(), rec)

    kinds = [e.split(":")[0] for e in rig.events]
    assert kinds == ["stop", "start", "wait", "stop", "start", "wait"]  # the target is torn down FIRST
    assert "/m/flash-next.gguf" in rig.events[1] and "/m/a.gguf" in rig.events[4]
    assert rig.events[5] == f"wait:{PRIMARY_URL}"
    assert rec.calls == ["freed", "restored", "line"]
    assert rec.lines == [f"Launch of flash-next failed: {FLASH_FAILED} Restored /m/a.gguf."]
    assert rec.restored_specs == [incumbent.spec]
    assert outcome.restored == incumbent and outcome.stopped == incumbent
    assert outcome.stage == "launch" and outcome.live is None
    assert isinstance(outcome.error, RuntimeError) and str(outcome.error) == FLASH_FAILED


@pytest.mark.asyncio
async def test_a_restore_that_fails_says_why_and_reports_no_restore(rig):
    rig.wait_raises[PEER_URL] = RuntimeError("target died")
    rig.wait_raises[PRIMARY_URL] = RuntimeError("port clash")
    rec = Reporter()

    outcome = await _swap(rig, rig.incumbent(), rig.target(), rec)

    assert rec.calls == ["freed", "line"]
    assert rec.lines == ["Launch of flash-next failed: target died Could not restore it: port clash"]
    assert outcome.restored is None and outcome.stopped is not None and outcome.stage == "launch"


@pytest.mark.asyncio
async def test_a_restore_the_launch_decision_refuses_names_the_refusal(rig):
    rig.wait_raises[PEER_URL] = RuntimeError("target died")
    declared_only = Heavy(rig.harness.server.model_copy(update={"binary": None}), PRIMARY_URL)
    rec = Reporter()

    outcome = await _swap(rig, declared_only, rig.target(), rec)

    assert rec.calls == ["freed", "line"]
    assert rec.lines[0].startswith("Launch of flash-next failed: target died Could not restore it: ")
    assert "binary" in rec.lines[0] and "NO model serving" not in rec.lines[0]
    assert outcome.restored is None
    assert _launches(rig) == [rig.events[1]]  # the refusal launched nothing: only the target started


@pytest.mark.asyncio
async def test_a_launch_failure_with_no_incumbent_has_no_restore_tail_and_no_nothing_serving_tail(rig):
    rig.wait_raises[PEER_URL] = RuntimeError("boom")
    rec = Reporter()

    outcome = await _swap(rig, None, rig.target(), rec)

    assert rec.lines == ["Launch of flash-next failed: boom"]
    assert rec.calls == ["freed", "line"]
    assert outcome.stopped is None and outcome.restored is None
    assert [e.split(":")[0] for e in rig.events] == ["start", "wait", "stop"]


@pytest.mark.asyncio
async def test_a_restore_with_nothing_to_relaunch_ends_with_the_nothing_serving_tail(rig, monkeypatch):
    rig.wait_raises[PEER_URL] = RuntimeError("boom")

    async def nothing(stopped, **kw):
        return None, None

    monkeypatch.setattr(lifecycle, "restore_incumbent", nothing)
    rec = Reporter()

    outcome = await _swap(rig, rig.incumbent(), rig.target(), rec)

    assert rec.lines == ["Launch of flash-next failed: boom The box now has NO model serving — "
                         "manual attention needed."]
    assert outcome.restored is None


@pytest.mark.asyncio
async def test_a_restore_note_follows_the_restored_label(rig, monkeypatch):
    rig.wait_raises[PEER_URL] = RuntimeError("boom")

    async def noted(stopped, **kw):
        return "/m/a.gguf", "the window was not widened"

    monkeypatch.setattr(lifecycle, "restore_incumbent", noted)
    rec = Reporter()

    await _swap(rig, rig.incumbent(), rig.target(), rec)

    assert rec.lines == ["Launch of flash-next failed: boom Restored /m/a.gguf. the window was not widened"]
    assert rec.calls == ["freed", "restored", "line"]


@pytest.mark.asyncio
async def test_a_cancelled_swap_stops_the_target_tells_the_reporter_and_restores_nothing(rig):
    rig.wait_raises[PEER_URL] = asyncio.CancelledError()
    rec = Reporter()

    with pytest.raises(asyncio.CancelledError):
        await _swap(rig, rig.incumbent(), rig.target(), rec)

    assert [e.split(":")[0] for e in rig.events] == ["stop", "start", "wait", "stop"]
    assert len(_launches(rig)) == 1  # the incumbent was not relaunched
    assert rec.calls == ["freed", "freed"]


@pytest.mark.asyncio
async def test_a_cancelled_swap_with_no_incumbent_still_stops_the_target(rig):
    rig.wait_raises[PEER_URL] = asyncio.CancelledError()
    rec = Reporter()

    with pytest.raises(asyncio.CancelledError):
        await _swap(rig, None, rig.target(), rec)

    assert [e.split(":")[0] for e in rig.events] == ["start", "wait", "stop"]
    assert rec.calls == ["freed", "freed"]


@pytest.mark.asyncio
async def test_a_cancelled_swap_reraises_even_when_the_targets_stop_fails(rig, monkeypatch):
    rig.wait_raises[PEER_URL] = asyncio.CancelledError()
    stops = {"n": 0}

    def second_stop_fails(cfg, launch="binary", **kw):
        stops["n"] += 1
        rig.events.append("stop")
        if stops["n"] == 2:  # the first frees the incumbent, the second tears the target down
            raise RuntimeError("stuck")
        return True

    monkeypatch.setattr(managed_server, "stop_server", second_stop_fails)
    rec = Reporter()

    with pytest.raises(asyncio.CancelledError):
        await _swap(rig, rig.incumbent(), rig.target(), rec)

    assert stops["n"] == 2 and rec.calls == ["freed", "freed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "free", "launch"])
async def test_swap_never_touches_the_unit_target_store(rig, monkeypatch, mode):
    def boom(*a, **kw):
        raise AssertionError("swap touched the unit-target store")

    monkeypatch.setattr(units, "write_target", boom)
    monkeypatch.setattr(units, "remove_target", boom)
    monkeypatch.setattr(model_ops, "remove_target", boom)
    if mode == "free":
        rig.stop_raises.append(RuntimeError("won't die"))
    elif mode == "launch":
        rig.wait_raises[PEER_URL] = RuntimeError("boom")
    rec = Reporter()

    outcome = await _swap(rig, rig.incumbent(), rig.target(), rec, unit=object())

    assert (outcome.error is None) == (mode == "success")
    assert not any(p.name == "unit-target" for p in rig.home.rglob("*"))


@pytest.mark.asyncio
async def test_the_unit_keyword_is_accepted_and_changes_nothing(rig):
    rig.wait_raises[PEER_URL] = RuntimeError("boom")
    with_unit, without = Reporter(), Reporter()

    a = await _swap(rig, rig.incumbent(), rig.target(), with_unit, unit=object())
    events_with = list(rig.events)
    rig.events.clear()
    b = await _swap(rig, rig.incumbent(), rig.target(), without, unit=None)

    assert events_with == rig.events and with_unit.lines == without.lines and with_unit.calls == without.calls
    assert (a.stage, a.restored) == (b.stage, b.restored)


@pytest.mark.asyncio
async def test_every_line_the_swap_says_goes_through_the_callers_redact(rig):
    rig.wait_raises[PEER_URL] = RuntimeError("boom")
    rec = Reporter()

    outcome = await swap(rig.incumbent(), rig.target(), config_dir=rig.home, autosense=None, reporter=rec,
                         redact=lambda t: t.replace("boom", "[redacted]"))

    assert rec.lines == ["Launch of flash-next failed: [redacted] Restored /m/a.gguf."]
    assert str(outcome.error) == "[redacted]"


# --- redaction: nothing the swap returns or says carries the raw tail ---------------------------- #


def _assert_clean(outcome: SwapOutcome, rec: Reporter, kind: type) -> None:
    error = outcome.error
    assert type(error) is kind
    for part in PLANTED_PARTS:
        assert part not in str(error), part
        assert part not in repr(outcome), part
        assert not any(part in line for line in rec.lines), part
    assert error.__cause__ is None and error.__context__ is None and error.__suppress_context__ is True
    assert error.__traceback__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [RuntimeError, TimeoutError])
async def test_a_planted_key_in_a_launch_failure_is_absent_from_the_error_and_every_line(rig, kind):
    rig.wait_raises[PEER_URL] = kind(PLANTED)
    rec = Reporter()

    outcome = await _swap(rig, rig.incumbent(), rig.target(), rec)

    _assert_clean(outcome, rec, kind)
    assert outcome.stage == "launch" and "Restored" in rec.lines[0]
    assert "**********" in str(outcome.error)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [RuntimeError, TimeoutError])
async def test_a_planted_key_in_a_free_failure_is_absent_from_the_error_and_the_line(rig, kind):
    rig.stop_raises.append(kind(PLANTED))
    rec = Reporter()

    outcome = await _swap(rig, rig.incumbent(), rig.target(), rec)

    _assert_clean(outcome, rec, kind)
    assert outcome.stage == "free"


@pytest.mark.asyncio
async def test_a_planted_key_in_a_restore_failure_never_reaches_the_line(rig):
    rig.wait_raises[PEER_URL] = RuntimeError("target died")
    rig.wait_raises[PRIMARY_URL] = RuntimeError(PLANTED)
    rec = Reporter()

    outcome = await _swap(rig, rig.incumbent(), rig.target(), rec)

    _assert_clean(outcome, rec, RuntimeError)
    assert "Could not restore it:" in rec.lines[0] and "**********" in rec.lines[0]
