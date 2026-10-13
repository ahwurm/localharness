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
