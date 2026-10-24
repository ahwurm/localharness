"""A model that arrives after the first run is asked about once at the next terminal `start`, before the session
loads, and never twice; `start` walks, asks and writes nothing without a person, off the terminal channel, or when the
config directory cannot be written. Each GGUF is a real header under a faked HOME; `_start_async` is a recorder."""
from __future__ import annotations

import asyncio
import json
import os
import signal
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import library, plugins_cmd, start_cmd, workspace
from localharness.cli.app import app
from localharness.cli.session_resume import RELOAD_ACTION, Restart, Resume
from localharness.provider import model_library
from tests.unit import test_init_discovery_records as world
from tests.unit.gguf_support import write_gguf
from tests.unit.interview_support import CANCEL, scripted
from tests.unit.test_init_core_setup import _exited, _flat, _hermetic, _init  # noqa: F401  (autouse fixture)
from tests.unit.test_init_discovery_records import _model, _peers

box = world.box


def _add(box, name: str, mtime: int = 5_000_000):
    return _model(box.root / f"{name}.gguf", mtime=mtime)


def _library_text(box) -> dict:
    return json.loads(model_library.library_path(box.cfg).read_text(encoding="utf-8"))["entries"]


@pytest.fixture
def after_first_run(box, monkeypatch):
    """A config written by a first run over the two-model box; the interview is then scripted by each test."""
    _exited(_init(box.cfg), 0)
    return box


def _start(box, monkeypatch, *flags: str, inside=None, env: dict | None = None):
    """`localharness start` with `_start_async` a recorder; `inside` runs where the session would be built."""
    ran: list = []

    async def start_async(*args, **kw):
        ran.append(kw.get("preset"))
        if inside is not None:
            inside()

    monkeypatch.setattr(start_cmd, "_start_async", start_async)
    result = CliRunner().invoke(app, ["start", "--config-dir", str(box.cfg), *flags], env=env)
    _exited(result, 0)
    return SimpleNamespace(result=result, ran=ran, out=_flat(result))


def _no_walk(monkeypatch) -> None:
    def walk(*_a, **_k):
        raise AssertionError("walk_ggufs called: this start must walk nothing")

    monkeypatch.setattr(model_library, "walk_ggufs", walk)


def _syncs(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    real = library.sync

    def spy(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr(library, "sync", spy)
    return calls


def test_a_model_copied_in_is_asked_before_the_session_and_the_trust_decision_finds_it_covered(
        after_first_run, monkeypatch):
    box = after_first_run
    _add(box, "C")
    asked = scripted(monkeypatch, "set-up")
    decided: list = []

    started = _start(box, monkeypatch, inside=lambda: decided.append(workspace.decide_machine_trust(box.cfg, ask=False)))

    [question] = asked.asked
    assert question.question == "New model C — set it up?"
    assert [o.label for o in question.options] == ["Set it up", "Not now"] and question.default == 0
    assert sorted(p["name"] for p in _peers(box.cfg)) == ["C", "Swift-27B-Q8"]
    assert "/model C switches in a session; localharness start --preset C starts on it" in started.out
    [trust] = decided
    assert dict(trust.withheld) == {} and trust.line == ""


def test_not_now_keeps_both_seen_and_a_second_start_asks_nothing_and_writes_nothing(after_first_run, monkeypatch):
    box = after_first_run
    _add(box, "C", 5_000_000)
    _add(box, "D", 4_000_000)
    asked = scripted(monkeypatch, "not-now")

    started = _start(box, monkeypatch)

    [question] = asked.asked
    assert question.question == "2 new models — set them up?"
    assert [o.label for o in question.options] == ["Set them up", "Not now", "Choose which"]
    assert "Not set up: C, D — localharness model --setup lists them" in started.out
    assert len(_peers(box.cfg)) == 1
    kept = model_library.library_path(box.cfg).read_text(encoding="utf-8")
    assert not any(e["declined"] for e in json.loads(kept)["entries"].values())
    again = scripted(monkeypatch)

    second = _start(box, monkeypatch)

    assert again.asked == [] and "Not set up" not in second.out
    assert model_library.library_path(box.cfg).read_text(encoding="utf-8") == kept
    assert len(_peers(box.cfg)) == 1


def test_a_cancel_is_not_now(after_first_run, monkeypatch):
    box = after_first_run
    _add(box, "C")
    scripted(monkeypatch, CANCEL)

    started = _start(box, monkeypatch)

    assert "Not set up: C — localharness model --setup lists them" in started.out
    assert len(_peers(box.cfg)) == 1 and not any(e["declined"] for e in _library_text(box).values())


def test_no_first_sighting_prints_nothing_from_the_library_and_never_calls_the_interview(after_first_run, monkeypatch):
    asked = scripted(monkeypatch)

    started = _start(after_first_run, monkeypatch)

    assert asked.asked == [] and "model" not in started.out.lower().replace("localharness", "")
    assert started.ran == [None]


def test_a_missing_library_is_baselined_once_and_a_later_model_is_new(after_first_run, monkeypatch):
    box = after_first_run
    model_library.library_path(box.cfg).unlink()
    old = _add(box, "A")
    asked = scripted(monkeypatch)

    first = _start(box, monkeypatch)

    assert asked.asked == [] and "A" not in first.out
    assert os.path.realpath(old) in _library_text(box)
    _add(box, "C")
    later = scripted(monkeypatch, "not-now")

    _start(box, monkeypatch)

    [question] = later.asked
    assert question.question == "New model C — set it up?"


def test_a_quiet_change_is_kept_and_never_read_again(after_first_run, monkeypatch):
    box = after_first_run
    lora = write_gguf(box.root / "lora.gguf", {"general.architecture": "llama", "general.type": "adapter"})
    asked = scripted(monkeypatch)

    _start(box, monkeypatch)

    assert asked.asked == []
    assert _library_text(box)[os.path.realpath(lora)]["verdict"] == "other"

    def no_header(*_a, **_k):
        raise AssertionError("read_header called: a typed file is never read again")

    monkeypatch.setattr(model_library.gguf_header, "read_header", no_header)
    _start(box, monkeypatch)


def test_a_first_run_without_a_person_is_baselined_by_the_next_terminal_start(box, monkeypatch):
    scripted(monkeypatch)
    _exited(_init(box.cfg, "--no-input"), 0)
    assert not model_library.library_path(box.cfg).exists()

    _start(box, monkeypatch)

    assert box.asked.asked == [] and model_library.library_path(box.cfg).exists()
    _add(box, "C")
    later = scripted(monkeypatch, "not-now")

    _start(box, monkeypatch)

    assert [a.question for a in later.asked] == ["New model C — set it up?"]


def test_a_removed_peer_is_a_first_sighting_again(after_first_run, monkeypatch):
    box = after_first_run
    overrides = box.cfg / "overrides.yaml"
    raw = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    raw["extra_endpoints"] = []
    overrides.write_text(yaml.safe_dump(raw), encoding="utf-8")
    asked = scripted(monkeypatch, "not-now")

    _start(box, monkeypatch)

    assert [a.question for a in asked.asked] == ["New model Swift-27B-Q8 — set it up?"]


def _stdout_piped(monkeypatch):
    monkeypatch.setattr(workspace, "_stdout_is_a_terminal", lambda: False)
    return (), None


def _marker(monkeypatch):
    return (), {"LOCALHARNESS_COMMIT": "1"}


@pytest.mark.parametrize("how", [
    lambda mp: (("--no-input",), None),
    _stdout_piped,
    _marker,
    lambda mp: (("--channel", "acp"), None),
    lambda mp: (("--list-models",), None),
], ids=["no-input", "stdout-piped", "marker", "acp", "list-models"])
def test_a_start_without_a_person_or_off_the_terminal_channel_walks_asks_and_writes_nothing(
        after_first_run, monkeypatch, how):
    box = after_first_run
    _add(box, "C")
    before = model_library.library_path(box.cfg).read_text(encoding="utf-8")
    asked = scripted(monkeypatch)
    flags, env = how(monkeypatch)
    _no_walk(monkeypatch)

    async def start_async(*args, **kw):
        return None

    monkeypatch.setattr(start_cmd, "_start_async", start_async)
    result = CliRunner().invoke(app, ["start", "--config-dir", str(box.cfg), *flags], env=env)

    assert "walk_ggufs called" not in repr(result.exception)
    assert asked.asked == [] and len(_peers(box.cfg)) == 1
    assert model_library.library_path(box.cfg).read_text(encoding="utf-8") == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through a read-only directory")
def test_a_config_directory_that_cannot_be_written_walks_and_asks_nothing(after_first_run, monkeypatch):
    box = after_first_run
    _add(box, "C")
    asked = scripted(monkeypatch)
    _no_walk(monkeypatch)
    box.cfg.chmod(0o500)
    try:
        started = _start(box, monkeypatch)
    finally:
        box.cfg.chmod(0o700)

    assert asked.asked == [] and started.ran == [None]


def test_the_first_start_that_just_ran_the_setup_does_not_sync_a_second_time(box, monkeypatch):
    calls = _syncs(monkeypatch)

    started = _start(box, monkeypatch)

    assert [c.get("first_run") for c in calls] == [True] and started.ran == [None]
    assert (box.cfg / "config.yaml").exists()


def test_the_plugins_restart_pass_does_not_sync_again(after_first_run, monkeypatch):
    box = after_first_run
    calls = _syncs(monkeypatch)
    scripted(monkeypatch)
    answers = [Restart(("enable", "x"), Resume(("enable", "x"), "a", (), "", None, (), "default", "s")), None]
    seen: list = []

    async def start_async(*args, **kw):
        seen.append(kw.get("resume"))
        return answers.pop(0)

    monkeypatch.setattr(start_cmd, "_start_async", start_async)
    monkeypatch.setattr(plugins_cmd, "session_step",
                        lambda *a, **k: SimpleNamespace(failed_check="", skipped_check="", stopped=False))

    result = CliRunner().invoke(app, ["start", "--config-dir", str(box.cfg)])

    _exited(result, 0)
    assert len(seen) == 2 and seen[0] is None and seen[1] is not None
    assert len(calls) == 1 and calls[0]["scope"] == "new"


def _restart_pass(box, monkeypatch, first: Restart):
    """Run `start` with `_start_async` answering `first` then None; returns what the loop did around it."""
    import gc

    import localharness.provider.client as client

    scripted(monkeypatch)
    answers = [first, None]
    seen: list = []
    steps: list = []
    gates: list = []
    collects: list = []
    real_collect = gc.collect

    async def start_async(*args, **kw):
        seen.append(kw.get("resume"))
        return answers.pop(0)

    def step(*a, **k):
        steps.append(a)
        return SimpleNamespace(failed_check="boom", skipped_check="", stopped=False)

    def collect(*a, **k):
        collects.append(1)
        return real_collect(*a, **k)

    monkeypatch.setattr(start_cmd, "_start_async", start_async)
    monkeypatch.setattr(plugins_cmd, "session_step", step)
    monkeypatch.setattr(client, "reset_inference_gate", lambda: gates.append(1))
    monkeypatch.setattr(gc, "collect", collect)
    result = CliRunner().invoke(app, ["start", "--config-dir", str(box.cfg)])
    _exited(result, 0)
    return SimpleNamespace(seen=seen, steps=steps, gates=gates, collects=collects)


def test_a_reload_restart_skips_the_step_keeps_the_gate_reset_and_resumes_unchanged(after_first_run, monkeypatch):
    reload_resume = Resume(RELOAD_ACTION, "a", (), "", None, (), "default", "s")

    ran = _restart_pass(after_first_run, monkeypatch, Restart(RELOAD_ACTION, reload_resume))

    assert ran.steps == []
    assert len(ran.gates) == 1 and len(ran.collects) >= 2
    assert len(ran.seen) == 2 and ran.seen[0] is None and ran.seen[1] is reload_resume


def test_a_verb_restart_still_runs_its_step_and_carries_the_check_into_the_resume(after_first_run, monkeypatch):
    resume = Resume(("enable", "x"), "a", (), "", None, (), "default", "s")

    ran = _restart_pass(after_first_run, monkeypatch, Restart(("enable", "x"), resume))

    assert ran.steps == [(("enable", "x"), str(after_first_run.cfg))]
    assert len(ran.gates) == 1
    assert ran.seen[1].failed_check == "boom" and ran.seen[1].action == ("enable", "x")


@pytest.mark.usefixtures("default_sigint")
def test_a_real_sigint_in_the_session_says_goodbye_and_exits_zero(after_first_run, monkeypatch):
    scripted(monkeypatch)

    async def start_async(*args, **kw):
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGINT)
        await asyncio.sleep(5)

    monkeypatch.setattr(start_cmd, "_start_async", start_async)
    result = CliRunner().invoke(app, ["start", "--config-dir", str(after_first_run.cfg)])

    _exited(result, 0)
    assert "Goodbye." in _flat(result)


def test_a_library_that_fails_with_an_oserror_says_one_line_and_the_session_still_starts(after_first_run, monkeypatch):
    def broken(*_a, **_k):
        raise OSError("disk")

    monkeypatch.setattr(library, "sync", broken)

    started = _start(after_first_run, monkeypatch)

    assert "model library: disk — start goes on" in started.out and started.ran == [None]


def test_a_preset_the_person_asked_for_reaches_the_library_to_preselect_a_file(after_first_run, monkeypatch):
    box = after_first_run
    calls = _syncs(monkeypatch)
    scripted(monkeypatch)

    _start(box, monkeypatch, "--preset", "Swift-27B-Q8")

    assert calls[0]["preset"] == "Swift-27B-Q8" and calls[0]["person"] is True
