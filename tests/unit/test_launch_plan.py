# ruff: noqa: F811  (the `popen` fixture is imported by name, then taken as a parameter)
"""TUNE-06 / TUNE-03 / TUNE-04 (spec 20 §6.0, §6.4, D-14, D-15): `activate` is sense -> plan -> prepare ->
execute, and the argv it starts is exactly `plan.argv`.

The identical-argv test starts here, with `activate`; `serve`, `doctor` and `--script` join it in later
phases. Nothing here starts a process or reads the host: `subprocess.Popen` is the `PopenRecorder`, free
memory, `/proc/meminfo`, the weights and the probe are faked through their module seams (`Box`), and
`wait_ready` is the scripted fake.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from localharness.agent import gguf_header
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server, speed_stats, tuning, units
from localharness.provider.autosense import FIT_KEYS, FreeMemory, fit, fit_header
from tests.unit.gguf_support import write_gguf
from tests.unit.launch_support import PopenRecorder, popen, record_chain  # noqa: F401
from tests.unit.test_launch_fit_retry import BASE, Box, _argv, _has_pair
from tests.unit.test_tuning_derive import FLASH_NEXT, SWIFT

GIB = 2**30
KNOBS = AutosenseConfig()  # q8_0 KV, MTP drafting at p-min 0.6: what 0.19 launches with by default
PRE_TUNING = AutosenseConfig(kv_cache_type="f16", draft_mtp=False)  # the 0.18 shape
SHARE = 66 / 65


def _gguf_spec(tmp_path: Path, header: dict, **over) -> ManagedServerConfig:
    model = write_gguf(tmp_path / "model.gguf", header)
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": str(model), "port": 8080, **over})


def _vllm_spec(tmp_path: Path, **over) -> ManagedServerConfig:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    return ManagedServerConfig(**{"runtime": "vllm", "launch": "binary", "binary": "/x/vllm",
                                  "model": str(ckpt), **over})


@pytest.fixture
def box(monkeypatch):
    world = Box(monkeypatch)
    world.mem = FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi")
    world.weights = 18 * GIB
    return world


def _expected_fit(spec: ManagedServerConfig, box: Box, *, share: float) -> int | None:
    header = fit_header(gguf_header.read_header(spec.model, FIT_KEYS))
    return fit(header, box.mem.free_bytes, KNOBS, cache_type_k="q8_0", cache_type_v="q8_0",
               weights=box.weights, offload="gpu", draft_share=share)


async def _activate(tmp_path, spec, plans=None, **kw):
    strategy = lifecycle.ManagedVllmStrategy() if spec.runtime == "vllm" else lifecycle.SpawnedProcessStrategy()
    return await strategy.activate(
        spec, tmp_path, BASE, autosense=kw.pop("autosense", KNOBS),
        on_plan=(plans.append if plans is not None else None), **kw)


# ------------------------------------------------------------------- the identical-argv tracer


@pytest.mark.parametrize("kind", ["swift-gguf", "flash-next-gguf", "vllm-binary"])
async def test_activate_launches_exactly_the_planned_argv(tmp_path, popen, box, kind):
    spec = {"swift-gguf": lambda: _gguf_spec(tmp_path, SWIFT),
            "flash-next-gguf": lambda: _gguf_spec(tmp_path, FLASH_NEXT),
            "vllm-binary": lambda: _vllm_spec(tmp_path)}[kind]()
    plans: list = []
    seen_at_plan: list[int] = []

    def on_plan(plan):
        seen_at_plan.append(len(popen))  # how many processes had started when the plan was announced
        plans.append(plan)

    strategy = lifecycle.ManagedVllmStrategy() if spec.runtime == "vllm" else lifecycle.SpawnedProcessStrategy()
    live = await strategy.activate(spec, tmp_path, BASE, autosense=KNOBS, on_plan=on_plan)

    assert len(plans) == 1 and seen_at_plan == [0]  # once, before the first Popen
    plan = plans[0]
    assert len(popen) == 1 and tuple(_argv(popen)) == plan.argv
    assert live.tuned == plan.tuned

    if kind == "swift-gguf":
        argv = list(plan.argv)
        assert _has_pair(argv, "--cache-type-k", "q8_0") and _has_pair(argv, "--cache-type-v", "q8_0")
        assert _has_pair(argv, "--spec-type", "draft-mtp") and _has_pair(argv, "--spec-draft-p-min", "0.6")
        priced = _expected_fit(spec, box, share=SHARE)
        assert priced and priced < _expected_fit(spec, box, share=1.0)  # the draft context costs window
        assert _has_pair(argv, "-c", str(priced))
        assert plan.report.mode == "fit" and live.window.mode == "fit"
        assert plan.full_fit == priced
        assert box.events.count("free_memory") == 1
    elif kind == "flash-next-gguf":
        argv = list(plan.argv)
        assert _has_pair(argv, "--cache-type-k", "q8_0") and _has_pair(argv, "--cache-type-v", "q8_0")
        assert "--spec-type" not in argv and plan.tuned.spec_type is None
        assert box.events.count("free_memory") == 1
    else:
        argv = list(plan.argv)
        assert _has_pair(argv, "--reasoning-parser", "qwen3")
        assert _has_pair(argv, "--tool-call-parser", "qwen3_coder")
        assert "--enable-auto-tool-choice" in argv
        assert plan.report is None and live.window is None
        assert box.events.count("free_memory") == 0


async def test_an_untuned_launch_has_no_plan_and_todays_keywords(tmp_path, monkeypatch, box):
    calls = record_chain(monkeypatch)
    spec = _gguf_spec(tmp_path, SWIFT)
    plans: list = []
    live = await _activate(tmp_path, spec, plans, autosense=None)
    decision = server.launch_decision(spec, False)
    assert plans == [None]
    assert box.events == []  # nothing sensed: no free memory read, no probe
    assert calls["serve_command"] == {
        "keyed": False, "loopback": decision.loopback, "key_file_ok": decision.key_file_ok,
        "context": None, "cache_ram": None}
    assert live.tuned is None and live.window is None


async def test_a_launch_that_derived_nothing_calls_serve_command_with_the_0_18_keywords(tmp_path, popen, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    plans: list = []
    live = await _activate(tmp_path, spec, plans, autosense=PRE_TUNING)
    plan = plans[0]
    assert set(plan.serve_kwargs) == {"keyed", "loopback", "key_file_ok", "context", "cache_ram"}
    assert "--cache-type-k" not in plan.argv and "--spec-type" not in plan.argv
    assert live.tuned == plan.tuned == tuning.NOTHING_DERIVED


def test_the_0_18_parity_covers_the_keywords_and_an_env_cache_type_is_priced_as_intended(tmp_path, box):
    """IN-03: with f16 + draft_mtp false the keywords are 0.18's, and the fit of a record whose `env:`
    pins a cache type is priced at that type (0.18 priced f16 there): the documented, intended change."""
    spec = _gguf_spec(tmp_path, SWIFT, env={"LLAMA_ARG_CACHE_TYPE_K": "q8_0", "LLAMA_ARG_CACHE_TYPE_V": "q8_0"})
    plan = lifecycle.plan_launch(spec, _sense(spec, tmp_path), server.launch_decision(spec, False),
                                 autosense=PRE_TUNING, api_key_held=False)
    assert set(plan.serve_kwargs) == {"keyed", "loopback", "key_file_ok", "context", "cache_ram"}
    header = fit_header(gguf_header.read_header(spec.model, FIT_KEYS))
    q8 = fit(header, box.mem.free_bytes, PRE_TUNING, cache_type_k="q8_0", cache_type_v="q8_0",
             weights=box.weights, offload="gpu")
    f16 = fit(header, box.mem.free_bytes, PRE_TUNING, cache_type_k="f16", cache_type_v="f16",
              weights=box.weights, offload="gpu")
    assert plan.full_fit == q8 != f16


async def test_a_callback_that_raises_does_not_stop_the_launch(tmp_path, popen, box):
    spec = _gguf_spec(tmp_path, SWIFT)

    def printer(plan):
        raise ValueError("a broken printer")

    live = await lifecycle.SpawnedProcessStrategy().activate(
        spec, tmp_path, BASE, autosense=KNOBS, on_plan=printer)
    assert len(popen) == 1 and live.served_models == ["qwen"]


# ------------------------------------------------------------------- who is priced, who wins


async def test_a_hand_set_cache_type_is_priced_not_the_derived_one(tmp_path, popen, box):
    spec = _gguf_spec(tmp_path, SWIFT, extra_args=["-ctk", "f16", "-ctv", "f16"])
    plans: list = []
    await _activate(tmp_path, spec, plans)
    plan = plans[0]
    assert plan.tuned.cache_type is None and "--cache-type-k" not in plan.argv
    header = fit_header(gguf_header.read_header(spec.model, FIT_KEYS))
    assert plan.full_fit == fit(header, box.mem.free_bytes, KNOBS, cache_type_k="f16", cache_type_v="f16",
                                weights=box.weights, offload="gpu", draft_share=SHARE)


async def test_a_record_speculative_draft_mtp_is_priced_with_the_draft_share(tmp_path, popen, box):
    spec = _gguf_spec(tmp_path, SWIFT, speculative="draft-mtp")
    plans: list = []
    await _activate(tmp_path, spec, plans)
    plan = plans[0]
    assert plan.tuned.spec_type is None  # a record value, not a derivation
    assert _has_pair(list(plan.argv), "--spec-type", "draft-mtp")
    assert plan.full_fit == _expected_fit(spec, box, share=SHARE)


# ------------------------------------------------------------------- determinism and purity


def _sense(spec, tmp_path):
    return lifecycle.sense_launch(spec, tmp_path / "cfg", KNOBS)


def test_the_same_sensed_always_gives_an_equal_plan(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    sensed = _sense(spec, tmp_path)
    decision = server.launch_decision(spec, False)
    first = lifecycle.plan_launch(spec, sensed, decision, autosense=KNOBS, api_key_held=False)
    second = lifecycle.plan_launch(spec, sensed, decision, autosense=KNOBS, api_key_held=False)
    assert first == second
    assert first.retry() == second.retry() and first.retry() is not None


def test_plan_launch_does_no_io(tmp_path, box, monkeypatch):
    spec = _gguf_spec(tmp_path, SWIFT)
    sensed = _sense(spec, tmp_path)
    decision = server.launch_decision(spec, False)

    def boom(*a, **k):
        raise AssertionError("plan_launch reached for the world")

    for owner, name in [(autosense, "free_memory"), (autosense, "_read_meminfo"), (autosense, "weights_bytes"),
                        (gguf_header, "read_header"), (tuning, "read_model_config"),
                        (server, "prepare_config_ini_home"), (server, "start_server"),
                        (speed_stats, "_write")]:
        monkeypatch.setattr(owner, name, boom)
    one = lifecycle.plan_launch(spec, sensed, decision, autosense=KNOBS, api_key_held=False)
    two = lifecycle.plan_launch(spec, sensed, decision, autosense=KNOBS, api_key_held=False)
    assert one == two and one.argv
    assert one.retry() == two.retry()


def test_sense_launch_writes_nothing(tmp_path, box, monkeypatch):
    spec = _gguf_spec(tmp_path, SWIFT)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    def boom(*a, **k):
        raise AssertionError("sense_launch wrote")

    monkeypatch.setattr(server, "prepare_config_ini_home", boom)
    monkeypatch.setattr(speed_stats, "_write", boom)
    monkeypatch.setattr(units, "remove_marker", boom)
    sensed = lifecycle.sense_launch(spec, cfg, KNOBS)
    assert sensed.header and sensed.offload == "gpu" and sensed.memory is box.mem
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_sense_launch_of_a_runtime_with_nothing_to_sense_is_empty(tmp_path, box):
    spec = ManagedServerConfig(runtime="ollama", model="m")
    sensed = lifecycle.sense_launch(spec, tmp_path, KNOBS)
    assert sensed == lifecycle.Sensed("ollama", tmp_path)
    assert box.events == []


def test_the_config_ini_home_is_a_path_not_a_write(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    plan = lifecycle.plan_launch(spec, _sense(spec, tmp_path), server.launch_decision(spec, False),
                                 autosense=KNOBS, api_key_held=False)
    assert plan.config_ini_home == server.config_ini_home(tmp_path / "cfg", 8080)
    assert not plan.config_ini_home.exists()


# ------------------------------------------------------------------- the key is never in a plan


async def test_the_api_key_is_in_no_plan_field(tmp_path, popen, box):
    secret = "sk-SECRET-123"
    spec = _gguf_spec(tmp_path, SWIFT, env={"LLAMA_ARG_THREADS": "zzz-env-value"})
    plans: list = []
    await _activate(tmp_path, spec, plans, api_key=secret)
    plan = plans[0]
    assert secret not in repr(plan)
    assert all(secret not in token for token in plan.argv)
    assert all(secret not in name for name in plan.env_names)
    assert all(secret not in str(value) for value in plan.serve_kwargs.values())
    assert plan.serve_kwargs["keyed"] is True
    assert "LLAMA_API_KEY" in plan.env_names and "XDG_CONFIG_HOME" in plan.env_names
    assert "LLAMA_ARG_THREADS" in plan.env_names and "zzz-env-value" not in repr(plan)
    assert secret not in " ".join(_argv(popen))


async def test_a_keyless_plan_names_no_key_variable(tmp_path, popen, box):
    spec = _gguf_spec(tmp_path, SWIFT)
    plans: list = []
    await _activate(tmp_path, spec, plans)
    assert plans[0].env_names == ("XDG_CONFIG_HOME",)


# ------------------------------------------------------------------- the strategies' surface


def test_the_planning_strategies_say_so():
    assert lifecycle.SpawnedProcessStrategy.plans_launch is True
    assert lifecycle.ManagedVllmStrategy.plans_launch is True
    assert not getattr(lifecycle.DaemonStrategy, "plans_launch", False)
    assert not getattr(lifecycle.LmsStrategy, "plans_launch", False)


@pytest.mark.parametrize("cls", [lifecycle.DaemonStrategy, lifecycle.LmsStrategy])
def test_every_strategy_accepts_on_plan(cls):
    import inspect

    assert "on_plan" in inspect.signature(cls.activate).parameters


def test_the_live_endpoint_defaults_to_no_tuning():
    assert lifecycle.LiveEndpoint(base_url="u", served_models=[], handle=1).tuned is None


# ------------------------------------------------------------------- a vLLM parser retry (D-37, WR-08)

PARSER_FLAGS = ("--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice")
VLLM_DEATH = "vLLM exited during startup. Log tail:\nunknown parser"


def _vllm_plan(tmp_path, **over):
    spec = _vllm_spec(tmp_path, **over)
    return lifecycle.plan_launch(spec, _sense(spec, tmp_path), server.launch_decision(spec, False),
                                 autosense=KNOBS, api_key_held=False)


def test_a_vllm_plan_with_derived_parsers_retries_without_them(tmp_path, box):
    plan = _vllm_plan(tmp_path)
    assert all(f in plan.argv for f in PARSER_FLAGS)
    retry = plan.retry()
    assert retry is not None and retry.shape == "retry" and retry.cause == "death"
    assert retry.dropped == ("parsers",)
    assert not any(f in retry.argv for f in PARSER_FLAGS)
    assert retry.tuned.reasoning_parser is None and retry.tuned.tool_call_parser is None
    assert retry.dropped_parser_fields == ("reasoning_parser", "tool_call_parser")
    assert plan.retry() == retry and retry.retry() is None  # pure, and only once


def test_a_record_parser_is_never_dropped_by_the_retry(tmp_path, box):
    retry = _vllm_plan(tmp_path, tool_call_parser="hermes").retry()
    assert retry.dropped == ("parsers",) and retry.dropped_parser_fields == ("reasoning_parser",)
    argv = list(retry.argv)
    assert "--reasoning-parser" not in argv
    assert _has_pair(argv, "--tool-call-parser", "hermes") and "--enable-auto-tool-choice" in argv


def test_a_vllm_plan_that_derived_no_parser_has_no_retry(tmp_path, box):
    assert _vllm_plan(tmp_path, reasoning_parser="off", tool_call_parser="off").retry() is None
    assert _vllm_plan(tmp_path, extra_args=["--tool-call-parser", "x"]).retry() is None


async def test_a_vllm_death_with_derived_parsers_is_retried_once_without_them(tmp_path, popen, box):
    box.ready_script = [RuntimeError(VLLM_DEATH), ["qwen"]]
    plans: list = []
    live = await _activate(tmp_path, _vllm_spec(tmp_path), plans)
    assert [p.shape for p in plans] == ["full", "retry"]
    assert box.stops == [(tmp_path, "binary")]
    assert all(f in popen[0]["cmd"] for f in PARSER_FLAGS)
    assert not any(f in popen[1]["cmd"] for f in PARSER_FLAGS)
    assert live.served_models == ["qwen"] and live.tuned == plans[1].tuned
    assert speed_stats.last_retry(speed_stats.default_speed_stats_path(tmp_path),
                                  lifecycle.record_key(_vllm_spec(tmp_path))) is None  # nothing remembered


async def test_a_second_vllm_death_names_the_record_fields_that_pin_the_parsers_off(tmp_path, popen, box):
    box.ready_script = [RuntimeError(VLLM_DEATH), RuntimeError("vLLM exited during startup. Log tail:\nagain")]
    with pytest.raises(RuntimeError) as caught:
        await _activate(tmp_path, _vllm_spec(tmp_path))
    text = str(caught.value)
    assert "Log tail:\nagain" in text
    assert "server.reasoning_parser: off / server.tool_call_parser: off" in text
    assert len(box.stops) == 2 and len(popen) == 2


async def test_a_peer_parser_retry_names_the_peers_fields(tmp_path, popen, box):
    box.ready_script = [RuntimeError(VLLM_DEATH), RuntimeError("again")]
    with pytest.raises(RuntimeError, match=r"extra_endpoints\.deckard\.lifecycle\.tool_call_parser: off"):
        await _activate(tmp_path, _vllm_spec(tmp_path), peer="deckard")


async def test_a_vllm_timeout_or_an_untuned_death_is_not_retried(tmp_path, popen, box):
    box.ready_script = [TimeoutError("slow")]
    with pytest.raises(TimeoutError):
        await _activate(tmp_path, _vllm_spec(tmp_path))
    box.ready_script = [RuntimeError(VLLM_DEATH)]
    with pytest.raises(RuntimeError, match="unknown parser"):
        await _activate(tmp_path, _vllm_spec(tmp_path), autosense=None)
    assert box.stops == [] and len(popen) == 2


# ------------------------------------------------------------------- config_ini is a hand setting (CR-01)


async def test_a_config_ini_cache_type_is_priced_and_not_overridden(tmp_path, popen, box):
    spec = _gguf_spec(tmp_path, SWIFT, config_ini="cache-type-k = f16\nLLAMA_ARG_CACHE_TYPE_V = f16\n")
    plans: list = []
    await _activate(tmp_path, spec, plans)
    plan = plans[0]
    assert plan.tuned.cache_type is None and "--cache-type-k" not in plan.argv
    header = fit_header(gguf_header.read_header(spec.model, FIT_KEYS))
    assert plan.full_fit == fit(header, box.mem.free_bytes, KNOBS, cache_type_k="f16", cache_type_v="f16",
                                weights=box.weights, offload="gpu", draft_share=SHARE)


async def test_a_config_ini_flash_attention_off_launches_no_quantized_cache(tmp_path, popen, box):
    spec = _gguf_spec(tmp_path, SWIFT, config_ini="flash-attn = off\n")
    plans: list = []
    await _activate(tmp_path, spec, plans)
    assert plans[0].tuned.cache_type is None
    assert "--cache-type-k" not in plans[0].argv and "--cache-type-v" not in plans[0].argv


def test_a_config_ini_spec_type_draft_mtp_is_priced_with_the_draft_share(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT, config_ini="spec-type = draft-mtp\n")
    plan = lifecycle.plan_launch(spec, _sense(spec, tmp_path), server.launch_decision(spec, False),
                                 autosense=KNOBS, api_key_held=False)
    assert plan.tuned.spec_type is None and "--spec-type" not in plan.argv
    assert plan.full_fit == _expected_fit(spec, box, share=SHARE)
