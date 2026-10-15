"""`model_ops.sense_serving` (spec 20 §8.2, D-12, R-13-27): the one sensor `start` and `/model` share.

It asks loopback addresses only, in parallel and off the event loop, and says which launch record
answers at its own address. Offline: `list_live_models` is faked at the module the sensor calls it
through, harnesses come from `ConfigLoader` over a tmp dir, trust values are constructed. No socket
is opened (the conftest guard holds) and nothing is started.
"""
from __future__ import annotations

import threading

import pytest
import yaml

from localharness.cli import model_ops
from localharness.cli.model_ops import (
    MalformedModelListError,
    ServingSense,
    is_loopback,
    launchable_loopback_others,
    sense_serving,
    serving_records,
    shares_base_url,
)
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader

PRIMARY = "qwen-27b-swift-1.5"
PRIMARY_URL = "http://127.0.0.1:8080/v1"
PEER_URL = "http://127.0.0.1:8081/v1"
PROVIDER_KEY = "sk-prov-7d3f9a"


def peer(name: str, port: int | None = 8081, *, url: str | None = None, lifecycle: bool = True,
         served_name: str | None = None, model: str | None = None) -> dict:
    ep = {"name": name, "base_url": url or f"http://127.0.0.1:{port}/v1", "provider_type": "llamacpp"}
    if lifecycle:
        lc = {"gpu": True, "runtime": "llamacpp", "binary": "/x/llama-server",
              "model": model or f"/m/{name}.gguf", "port": port or 8081}
        if served_name:
            lc["served_name"] = served_name
        ep["gpu"] = True
        ep["lifecycle"] = lc
    return ep


def seed(home, *, peers=None, active: dict | None = None, last_served: str | None = None,
         overrides: dict | None = None) -> None:
    """A llama.cpp primary on :8080 that requires a key, one peer `flash-next` on :8081 by default,
    optionally an `active_endpoint` in the overlay and a `last_served` in the ledger."""
    data = {
        "version": "1",
        "provider": {"provider_type": "llamacpp", "base_url": PRIMARY_URL, "default_model": PRIMARY,
                     "available_models": [PRIMARY], "api_key": PROVIDER_KEY},
        "org": {"default_model": PRIMARY, "audit_log_path": str(home / "audit.jsonl")},
        "extra_endpoints": peers if peers is not None else [peer("flash-next")],
        "server": {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                   "served_name": PRIMARY, "port": 8080, "require_api_key": True},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    overlay = dict(overrides or {})
    if active is not None:
        overlay["active_endpoint"] = active
    path = home / "overrides.yaml"
    if overlay:
        path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)
    if last_served is not None:
        from localharness.provider import speed_stats

        speed_stats.record_last_served(speed_stats.default_speed_stats_path(home), last_served)


def load(home, *, machine_withheld=None):
    return ConfigLoader(config_dir=home, machine_withheld=machine_withheld).load_harness()


FLASH_ACTIVE = {"name": "flash-next", "base_url": PEER_URL, "provider_type": "llamacpp",
                "model": "flash-next"}


class Probes:
    """A fake `list_live_models`: `answers` maps a base URL to `(ids, reachable)` or an exception to
    raise. Every call is recorded with the thread it ran on and the keyword arguments it carried."""

    def __init__(self) -> None:
        self.answers: dict[str, object] = {}
        self.calls: list[dict] = []

    def __call__(self, base_url, timeout=3.0, **kw):
        self.calls.append({"url": base_url, "timeout": timeout, "kw": kw,
                           "thread": threading.get_ident()})
        got = self.answers.get(base_url, ([], False))
        if isinstance(got, BaseException):
            raise got
        return got

    @property
    def urls(self) -> list[str]:
        return [c["url"] for c in self.calls]


@pytest.fixture
def probes(monkeypatch) -> Probes:
    fake = Probes()
    monkeypatch.setattr(model_ops, "list_live_models", fake)
    return fake


@pytest.fixture
def home(tmp_path):
    h = tmp_path / ".localharness"
    h.mkdir()
    return h


def record(harness, name):
    return next(r for r in serving_records(harness) if r.name == name)


# ----------------------------------------------------------------------------------------- the tracer


@pytest.mark.asyncio
async def test_the_sensor_probes_the_other_launchable_record_only_off_the_loop_with_its_launch_key(
        home, probes):
    seed(home, active=FLASH_ACTIVE)
    harness = load(home)
    selected = record(harness, "flash-next")
    others = launchable_loopback_others(harness, MachineTrust(), selected)
    assert [r.name for r in others] == [PRIMARY]

    sense = await sense_serving(harness, others)

    assert sense.answering == ()
    assert isinstance(sense, ServingSense) and dict(sense.ids) == {}
    assert probes.urls == [PRIMARY_URL]  # the selected record's own address is start's probe, not the sensor's
    assert probes.calls[0]["thread"] != threading.get_ident()
    assert probes.calls[0]["kw"] == {"api_key": PROVIDER_KEY}  # the primary's launch key
    assert probes.calls[0]["timeout"] == model_ops.SENSE_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------------------- loopback


@pytest.mark.parametrize("url, expected", [
    ("http://127.0.0.1:8080/v1", True),
    ("http://localhost:8080/v1", True),
    ("http://LOCALHOST:8080/v1", True),
    ("http://[::1]:8080/v1", True),
    ("http://10.0.0.5:8081/v1", False),
    ("http://example.com/v1", False),
    ("http://127.0.0.1.evil.example/v1", False),
    ("http://[::1/v1", False),
    ("", False),
])
def test_is_loopback_names_only_the_three_loopback_hosts(url, expected):
    assert is_loopback(url) is expected


def test_a_non_loopback_peer_is_never_in_the_probe_set(home):
    seed(home, peers=[peer("flash-next"), peer("remote", url="http://10.0.0.5:8081/v1")])
    harness = load(home)
    others = launchable_loopback_others(harness, MachineTrust(), record(harness, PRIMARY))
    assert [r.name for r in others] == ["flash-next"]


def test_an_attach_only_peer_is_not_in_the_probe_set(home):
    seed(home, peers=[peer("flash-next"), peer("attach-only", 8082, lifecycle=False)])
    harness = load(home)
    others = launchable_loopback_others(harness, MachineTrust(), record(harness, PRIMARY))
    assert [r.name for r in others] == ["flash-next"]


def test_a_withheld_peer_is_not_in_the_probe_set(home):
    seed(home, peers=[peer("flash-next"), peer("gemma", 8082)])
    trust = MachineTrust(withheld={
        "config.yaml": frozenset({("launch_model", "extra_endpoints.gemma.lifecycle.model")})})
    harness = load(home, machine_withheld=trust.withheld)
    others = launchable_loopback_others(harness, trust, record(harness, PRIMARY))
    assert [r.name for r in others] == ["flash-next"]


def test_the_selected_record_is_never_among_its_own_others(home):
    seed(home)
    harness = load(home)
    for selected in serving_records(harness):
        assert selected.name not in [r.name for r in launchable_loopback_others(
            harness, MachineTrust(), selected)]


@pytest.mark.asyncio
async def test_a_remote_record_handed_to_the_sensor_by_the_launchable_filter_is_never_probed(home, probes):
    seed(home, peers=[peer("remote", url="http://10.0.0.5:8081/v1")])
    harness = load(home)
    selected = record(harness, "remote")
    sense = await sense_serving(harness, launchable_loopback_others(harness, MachineTrust(), selected))
    assert probes.urls == [PRIMARY_URL]
    assert sense.answering == ()


# --------------------------------------------------------------------------------------- shared url


def test_shares_base_url_is_true_only_for_the_shared_case(home):
    seed(home, peers=[peer("flash-next"), peer("twin", url=PRIMARY_URL.rstrip("/") + "/", port=8080)])
    harness = load(home)
    primary, flash, twin = (record(harness, n) for n in (PRIMARY, "flash-next", "twin"))
    records = serving_records(harness)
    assert shares_base_url(primary, records) is True  # the twin is on 8080 too, trailing slash or not
    assert shares_base_url(twin, records) is True
    assert shares_base_url(flash, records) is False
    assert shares_base_url(flash, (flash,)) is False  # itself never counts


def _shared_harness(home, *, flash_served=None):
    seed(home, peers=[peer("flash-next", 8080, url=PRIMARY_URL, served_name=flash_served)])
    harness = load(home)
    return harness, record(harness, PRIMARY), record(harness, "flash-next")


@pytest.mark.asyncio
async def test_on_a_shared_address_the_record_whose_served_name_is_listed_answers(home, probes):
    harness, primary, flash = _shared_harness(home)
    probes.answers[PRIMARY_URL] = (["flash-next"], True)
    sense = await sense_serving(harness, (primary, flash))
    assert [r.name for r in sense.answering] == ["flash-next"]
    assert probes.urls == [PRIMARY_URL]  # one probe for the one address
    assert dict(sense.ids) == {PRIMARY_URL: ("flash-next",)}


@pytest.mark.asyncio
async def test_on_a_shared_address_ids_that_match_no_record_attribute_it_to_the_primary(home, probes):
    harness, primary, flash = _shared_harness(home)
    probes.answers[PRIMARY_URL] = (["something"], True)
    sense = await sense_serving(harness, (flash, primary))
    assert [r.name for r in sense.answering] == [PRIMARY]


@pytest.mark.asyncio
async def test_an_address_only_one_record_has_attributes_to_it_whatever_the_ids(home, probes):
    seed(home)
    harness = load(home)
    flash = record(harness, "flash-next")
    probes.answers[PEER_URL] = (["a-name-nobody-configured"], True)
    sense = await sense_serving(harness, (flash,))
    assert [r.name for r in sense.answering] == ["flash-next"]


@pytest.mark.asyncio
async def test_a_shared_address_with_no_primary_among_them_and_no_matching_id_is_unattributed(home, probes):
    seed(home, peers=[peer("one", 8081), peer("two", 8081)])
    harness = load(home)
    one, two = record(harness, "one"), record(harness, "two")
    probes.answers[PEER_URL] = (["something"], True)
    sense = await sense_serving(harness, (one, two))
    assert sense.answering == ()
    assert dict(sense.ids) == {PEER_URL: ("something",)}  # something answered; nobody is named for it


# --------------------------------------------------------------------------------------- failures


@pytest.mark.asyncio
async def test_a_reply_that_is_not_a_model_list_answers_with_no_ids(home, probes):
    seed(home)
    harness = load(home)
    flash = record(harness, "flash-next")
    probes.answers[PEER_URL] = MalformedModelListError("html")
    sense = await sense_serving(harness, (flash,))
    assert [r.name for r in sense.answering] == ["flash-next"]
    assert dict(sense.ids) == {PEER_URL: ()}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("refused"), TimeoutError("slow"), RuntimeError("odd"),
                                     ([], False)])
async def test_a_probe_that_fails_or_cannot_connect_answers_nothing(home, probes, failure):
    seed(home)
    harness = load(home)
    probes.answers[PEER_URL] = failure
    sense = await sense_serving(harness, (record(harness, "flash-next"),))
    assert sense.answering == () and dict(sense.ids) == {}


@pytest.mark.asyncio
async def test_every_address_is_probed_once_and_answers_keep_record_order(home, probes):
    seed(home, peers=[peer("flash-next"), peer("gemma", 8082)])
    harness = load(home)
    records = serving_records(harness)
    for url in (PRIMARY_URL, PEER_URL, "http://127.0.0.1:8082/v1"):
        probes.answers[url] = ([], True)
    sense = await sense_serving(harness, records)
    assert [r.name for r in sense.answering] == [PRIMARY, "flash-next", "gemma"]
    assert sorted(probes.urls) == sorted([PRIMARY_URL, PEER_URL, "http://127.0.0.1:8082/v1"])


@pytest.mark.asyncio
async def test_nothing_to_probe_probes_nothing(home, probes):
    seed(home)
    sense = await sense_serving(load(home), ())
    assert sense.answering == () and probes.calls == []


def test_a_peer_at_the_primarys_address_sends_its_own_key_for_its_own_spec():
    """`_launch_key` goes by the launch record when it is given: the peer's key and rule, never the
    primary's, though both stand at one address."""
    from types import SimpleNamespace

    from pydantic import SecretStr

    primary_spec = SimpleNamespace(require_api_key=True)
    peer_spec = SimpleNamespace(require_api_key=True)
    harness = SimpleNamespace(
        provider=SimpleNamespace(base_url=PRIMARY_URL, api_key=SecretStr("primary-key")),
        server=primary_spec,
        extra_endpoints=[SimpleNamespace(
            base_url=PRIMARY_URL, api_key=SecretStr("peer-key"), lifecycle=peer_spec)])

    assert model_ops._launch_key(harness, PRIMARY_URL) == {"api_key": "primary-key"}
    assert model_ops._launch_key(harness, PRIMARY_URL, primary_spec) == {"api_key": "primary-key"}
    assert model_ops._launch_key(harness, PRIMARY_URL, peer_spec) == {"api_key": "peer-key"}
