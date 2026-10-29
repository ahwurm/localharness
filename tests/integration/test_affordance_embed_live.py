"""The skill cue over the local embedding model alone: a fresh measurement, the three misses, the holdout, a stuffed description and named skills."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import shutil
import statistics
import threading
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from localharness.agent.affordance import Affordances
from localharness.agent.calibration import entry_state, labelled, measure, shipped_pins, write_entry
from localharness.agent.task_record import TaskState
from localharness.cli.skill_session import LINE
from localharness.config.skills import SkillSet
from localharness.memory.resonance import EMBED_MAX_TOKENS
from tests.integration.test_skills_model_live import HOLDOUT, KINDS, LABELLED, REPORTS, cell, git

pytestmark = pytest.mark.live_embed

DRIFT = "the labelled prompts moved: update MISSES"
MISSES = (
    (3, "What should I put in the history entry for these edits to the parser?"),
    (5, "Help me phrase the log line for this change; the team likes 'area: did thing'."),
    (6, "Is there a version of my history entry the hook will accept? It was bounced for having capitals."),
)


def sets(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["skills"]


def prompts(group: dict) -> list[tuple[str, str, str]]:
    return [(skill, kind, text) for skill, texts in group.items() for kind in KINDS for text in texts[kind]]


def skill_set(config_dir: Path) -> SkillSet:
    shutil.copytree(LABELLED / "skills", config_dir / "skills")
    return SkillSet.discover(config_dir, None)


def warmed(skills: SkillSet, engine, record_dir: Path) -> Affordances:
    holder = Affordances(skills, SimpleNamespace(embedder=lambda: engine), TaskState(), record_dir)
    holder.warm().join(timeout=300)
    assert holder._ready is not None, "the holder did not warm from the fresh measurement"
    return holder


async def outcome(holder: Affordances, text: str, named: str | None = None) -> str | None:
    holder.named = named
    cue = await holder.cue(text, [])
    return cue.name if cue else None


async def timed(holder: Affordances, text: str) -> float:
    start = time.perf_counter()
    await outcome(holder, text)
    return time.perf_counter() - start


@pytest.fixture(scope="module")
def ready(live_embedder, tmp_path_factory):
    """`measure` once with no record dir; the holders then read that entry from a scratch dir instead of measuring again."""
    engine, folder = live_embedder
    pins, idle = shipped_pins(engine.model_name, folder), threading.Event()
    idle.set()
    entry = measure(engine, pins, None, idle, None)
    state, threshold = entry_state(entry, pins, labelled()[1])
    assert state == "ready", threshold
    record_dir = tmp_path_factory.mktemp("affordance")
    write_entry(record_dir, engine.model_name, entry)
    skills = skill_set(tmp_path_factory.mktemp("cfg"))
    return engine, entry, threshold, skills, record_dir, warmed(skills, engine, record_dir)


async def test_every_calibration_prompt_picks_what_the_fresh_rows_give(ready):
    _, entry, threshold, _, _, holder = ready
    rows, asked = entry["rows"], prompts(sets(LABELLED / "prompts.yaml"))
    assert [tuple(row[:2]) for row in rows] == [(skill, kind) for skill, kind, _ in asked]
    expected = [top if score > threshold else None for _, _, top, score in rows]
    assert [await outcome(holder, text) for _, _, text in asked] == expected


async def test_the_three_commit_message_misses_are_cued(ready):
    matching = sets(LABELLED / "prompts.yaml")["commit-message"]["matching"]
    for index, text in MISSES:
        assert matching[index] == text, DRIFT
        assert await outcome(ready[5], text) == "commit-message", text


async def test_the_holdout_draws_no_false_cue(ready):
    false = [(skill, kind, text, cued) for skill, kind, text in prompts(sets(HOLDOUT))
             if (cued := await outcome(ready[5], text)) is not None and (kind != "matching" or cued != skill)]
    assert false == []


async def test_a_description_stuffed_with_its_matching_prompts_draws_no_holdout_false_cue(ready):
    engine, _, _, skills, record_dir, _ = ready
    calibration = sets(LABELLED / "prompts.yaml")
    barred = [text for _, kind, text in prompts(sets(HOLDOUT)) if kind != "matching"]
    counts = {}
    for name in calibration:
        stuffed = " ".join(calibration[name]["matching"])
        holder = warmed(dataclasses.replace(skills, skills=tuple(
            dataclasses.replace(s, description=stuffed) if s.name == name else s for s in skills.skills)), engine, record_dir)
        counts[name] = sum([await outcome(holder, text) == name for text in barred])
    assert counts == dict.fromkeys(calibration, 0)


async def test_named_skills_and_cue_time_are_measured_into_the_report(ready):
    engine, entry, threshold, _, _, holder = ready
    calibration, held = sets(LABELLED / "prompts.yaml"), sets(HOLDOUT)
    forms = []
    for name in calibration:
        for text in calibration[name]["matching"] + held[name]["matching"]:
            forms += [("/name with arguments", name, LINE.format(name=name, args=f" with: {text}"), name),
                      ("use the name skill with arguments", name, f"use the {name} skill: {text}", None)]
        forms += [("/name", name, LINE.format(name=name, args=""), name),
                  ("use the name skill", name, f"use the {name} skill", None)]
    off = [(form, name, cued) for form, name, text, named in forms
           if (cued := await outcome(holder, text, named)) not in (None, name)]
    times = [await timed(holder, text) for _, _, text in prompts(calibration) + prompts(held)]
    longest = await timed(holder, " ".join(["history entry"] * EMBED_MAX_TOKENS))
    day = date.today().isoformat()
    run, path = next((n, p) for n in itertools.count(1) if not (
        p := REPORTS / (f"affordance-gate-{day}.md" if n == 1 else f"affordance-gate-{day}-{n}.md")).exists())
    lines = [
        "# Affordance cue, embedder only", "",
        f"Date: {day} · Run: {run}",
        f"Commit: {git('rev-parse', 'HEAD')} ({'dirty' if git('status', '--porcelain') else 'clean'} tree)",
        f"Module sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        f"Embedder: {engine.model_name}",
        f"Threshold: {threshold} (measured on this machine on {entry['date'].isoformat()})", "",
        f"Named-skill row: {len(off)} cues naming another skill over {len(forms)} cues",
        *(["", "| form | named | cued |", "|---|---|---|", *(f"| {cell(f)} | {cell(n)} | {cell(c)} |" for f, n, c in off)]
          if off else []), "",
        f"cue() wall time: median {statistics.median(times):.3f} s, maximum {max(times):.3f} s over {len(times)} turns; "
        f"one {EMBED_MAX_TOKENS}-token turn: {longest:.3f} s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(path, flush=True)
    assert path.exists()
