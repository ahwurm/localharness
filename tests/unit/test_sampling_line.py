"""SAMPLE-04: the one sampling line (spec 20 §7 "What is said", D-27, D-31, R-07a..c).

`/props` is read by the window probe (one GET, faked here the way test_probe_window fakes it), the
pure `_sampling_line` decides whether there is anything to say, and `_say_sampling` prints it once on
change and records the facts. Nothing here touches a real server; the ledger lives in `tmp_path`.
"""
from __future__ import annotations

import httpx
import pytest

from localharness.agent.context import WindowProbe, probe_window
from localharness.cli.start_cmd import SAMPLING_LINE, _fmt_temp, _say_sampling, _sampling_line
from localharness.provider.speed_stats import SamplingRecord, last_sampling

LLAMA = "http://127.0.0.1:8080/v1"
PROPS = "http://127.0.0.1:8080/props"
META = {"data": [{"id": "qwen", "meta": {"n_ctx": 65536, "n_ctx_train": 262144}}]}
RUNG = "org.default_temperature"
LINE = ("sampling: temperature 0.6 from org.default_temperature "
        "(the model's own is 1.0 — unset org.default_temperature to use it)")


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _serve(monkeypatch, bodies: dict[str, object]) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def _get(url, timeout=0, **kw):
        calls.append((url, kw))
        body = bodies.get(url, {})
        if isinstance(body, Exception):
            raise body
        return _Resp(body)

    monkeypatch.setattr(httpx, "get", _get)
    return calls


def _props(temp_params=None, **dgs):
    body = dict(dgs)
    if temp_params is not None:
        body["params"] = {"temperature": temp_params}
    return {"default_generation_settings": body}


# --- the probe reads the server's default from the one /props response ---------------------------

def test_the_props_default_temperature_rides_the_window_probe(monkeypatch):
    _serve(monkeypatch, {f"{LLAMA}/models": META, PROPS: _props(1.0)})
    assert probe_window(LLAMA, "qwen", "llamacpp") == WindowProbe(65536, 262144, 1.0)


def test_positional_construction_keeps_working_and_sampling_defaults_to_none():
    assert WindowProbe(14950, 262144).sampling is None
    assert WindowProbe(1, 2, 0.7).sampling == 0.7


# --- the pure line --------------------------------------------------------------------------------

def test_the_line_text_is_exact():
    assert _sampling_line(None, temperature=0.6, rung=RUNG, server_default=1.0) == LINE
    assert SAMPLING_LINE.format(temp="0.6", rung=RUNG, own="1.0") == LINE


def test_the_same_facts_recorded_say_nothing():
    last = SamplingRecord(0.6, RUNG, 1.0, "2026-10-13T00:00:00+00:00")
    assert _sampling_line(last, temperature=0.6, rung=RUNG, server_default=1.0) is None


def test_temperatures_print_rounded():
    assert _fmt_temp(0.6) == "0.6"
    assert _fmt_temp(1.0) == "1.0"
    assert _fmt_temp(0.800000011920929) == "0.8"


# --- the wiring: print once, record every time --------------------------------------------------

def _say(tmp_path, probe, said, **kw):
    args = dict(temperature=0.6, rung=RUNG, ledger=tmp_path / "speed.json",
                provider_type="llamacpp", model="qwen", say=said.append)
    args.update(kw)
    _say_sampling(probe, **args)


def test_first_start_prints_and_records_second_is_silent(tmp_path):
    said: list[str] = []
    probe = WindowProbe(65536, 262144, 1.0)
    _say(tmp_path, probe, said)
    assert said == [LINE]
    rec = last_sampling(tmp_path / "speed.json", "llamacpp", "qwen")
    assert rec is not None and (rec.temperature, rec.rung, rec.server_default) == (0.6, RUNG, 1.0)
    first_at = rec.recorded_at
    _say(tmp_path, probe, said)
    assert said == [LINE]  # nothing new
    again = last_sampling(tmp_path / "speed.json", "llamacpp", "qwen")
    assert again is not None and again.recorded_at >= first_at  # recorded both times (R-07c)


def test_the_default_printer_prints_the_line_as_plain_text_never_markup(tmp_path, monkeypatch):
    """T-12-24: the rung label rides the line verbatim, so the console must not read it as markup."""
    from localharness.cli import start_cmd

    printed: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(start_cmd.console, "print", lambda *a, **k: printed.append((a, k)))
    _say_sampling(WindowProbe(65536, 262144, 1.0), temperature=0.6, rung="[bold]agents.x.temperature",
                  ledger=tmp_path / "speed.json", provider_type="llamacpp", model="qwen")
    assert len(printed) == 1
    (args, kwargs), = printed
    assert args[0].startswith("sampling: temperature 0.6 from [bold]agents.x.temperature")
    assert kwargs.get("markup") is False


def test_a_probe_that_reported_no_default_prints_and_records_nothing(tmp_path):
    said: list[str] = []
    _say(tmp_path, WindowProbe(65536, 262144), said)
    assert said == []
    assert last_sampling(tmp_path / "speed.json", "llamacpp", "qwen") is None


def test_an_unwritable_ledger_never_fails_the_start(tmp_path, monkeypatch):
    def _boom(*a, **kw):
        raise OSError("read-only")

    monkeypatch.setattr("localharness.provider.speed_stats.record_sampling", _boom)
    monkeypatch.setattr("localharness.provider.speed_stats.last_sampling", _boom)
    said: list[str] = []
    _say(tmp_path, WindowProbe(1, 2, 1.0), said)
    assert said == [LINE]


def test_unsetting_then_resetting_says_it_again(tmp_path):
    said: list[str] = []
    probe = WindowProbe(1, 2, 1.0)
    _say(tmp_path, probe, said)
    _say(tmp_path, probe, said, temperature=None, rung=None)  # unset: recorded, silent
    assert said == [LINE]
    _say(tmp_path, probe, said)  # set again
    assert said == [LINE, LINE]


# --- every no-line case, the labels, and every change that says it again -------------------------

@pytest.mark.parametrize("kw", [
    dict(temperature=None, rung=RUNG, server_default=1.0),
    dict(temperature=0.6, rung=None, server_default=1.0),
    dict(temperature=0.6, rung=RUNG, server_default=None),
    dict(temperature=0.6, rung=RUNG, server_default=0.6),
    dict(temperature=0.6, rung=RUNG, server_default=0.6004),
    dict(temperature=None, rung=None, server_default=None),
])
def test_no_line_cases(kw):
    assert _sampling_line(None, **kw) is None


@pytest.mark.parametrize("label", [
    "temperature in agents/orchestrator.yaml", "temperature in divisions/core.yaml",
    "agent.temperature in overrides.yaml"])
def test_the_rungs_own_label_names_the_value_and_the_undo(label):
    assert _sampling_line(None, temperature=0.6, rung=label, server_default=1.0) == (
        f"sampling: temperature 0.6 from {label} (the model's own is 1.0 — unset {label} to use it)")


@pytest.mark.parametrize("now", [
    dict(temperature=0.7, rung=RUNG, server_default=1.0),
    dict(temperature=0.6, rung="temperature in agents/orchestrator.yaml", server_default=1.0),
    dict(temperature=0.6, rung=RUNG, server_default=0.8),
])
def test_a_change_in_any_of_the_three_facts_says_it_again(now):
    last = SamplingRecord(0.6, RUNG, 1.0, "2026-10-13T00:00:00+00:00")
    assert _sampling_line(last, **now) is not None


def test_a_recorded_baseline_of_no_temperature_does_not_hide_a_new_one():
    last = SamplingRecord(None, None, 1.0, "2026-10-13T00:00:00+00:00")
    assert _sampling_line(last, temperature=0.6, rung=RUNG, server_default=1.0) == LINE
