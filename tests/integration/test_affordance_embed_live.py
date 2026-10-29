"""The skill cue over the local embedding model alone: the record, the three misses, the holdout, a stuffed description and named skills."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import shutil
import statistics
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from localharness.agent.affordance import CALIBRATION, Affordances, load_calibration, snapshot_files
from localharness.agent.task_record import TaskState
from localharness.cli.skill_session import LINE
from localharness.config.skills import SkillSet
from localharness.memory.resonance import DEFAULT_EMBEDDING_MODEL, EMBED_MAX_TOKENS
from tests.integration.test_skills_model_live import HOLDOUT, KINDS, LABELLED, REPORTS, cell, git

pytestmark = pytest.mark.live_embed

STALE = "calibration is stale: run python scripts/gen_calibration.py"
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


def record() -> dict:
    return yaml.safe_load(CALIBRATION.read_text(encoding="utf-8"))[DEFAULT_EMBEDDING_MODEL]


def warmed(skills: SkillSet, engine) -> Affordances:
    holder = Affordances(skills, SimpleNamespace(embedder=lambda: engine), TaskState())
    holder.warm().join(timeout=300)
    assert holder._ready is not None, "the holder did not warm: no calibration entry, a snapshot mismatch or a containment"
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
    engine, folder = live_embedder
    skills = skill_set(tmp_path_factory.mktemp("cfg"))
    return engine, folder, skills, warmed(skills, engine)


def test_the_record_pins_the_snapshot_the_engine_loaded(ready):
    _, folder, _, _ = ready
    assert record()["snapshot_files"] == snapshot_files(folder)


async def test_every_calibration_prompt_picks_what_the_record_rows_give(ready):
    engine, _, _, holder = ready
    threshold = load_calibration(engine.model_name).threshold
    rows, asked = record()["rows"], prompts(sets(LABELLED / "prompts.yaml"))
    assert [tuple(row[:2]) for row in rows] == [(skill, kind) for skill, kind, _ in asked], STALE
    expected = [top if score > threshold else None for _, _, top, score in rows]
    assert [await outcome(holder, text) for _, _, text in asked] == expected


async def test_the_three_commit_message_misses_are_cued(ready):
    matching = sets(LABELLED / "prompts.yaml")["commit-message"]["matching"]
    for index, text in MISSES:
        assert matching[index] == text, STALE
        assert await outcome(ready[3], text) == "commit-message", text


async def test_the_holdout_draws_no_false_cue(ready):
    false = [(skill, kind, text, cued) for skill, kind, text in prompts(sets(HOLDOUT))
             if (cued := await outcome(ready[3], text)) is not None and (kind != "matching" or cued != skill)]
    assert false == []


async def test_a_description_stuffed_with_its_matching_prompts_draws_no_holdout_false_cue(ready):
    engine, _, skills, _ = ready
    calibration = sets(LABELLED / "prompts.yaml")
    barred = [text for _, kind, text in prompts(sets(HOLDOUT)) if kind != "matching"]
    counts = {}
    for name in calibration:
        stuffed = " ".join(calibration[name]["matching"])
        holder = warmed(dataclasses.replace(skills, skills=tuple(
            dataclasses.replace(s, description=stuffed) if s.name == name else s for s in skills.skills)), engine)
        counts[name] = sum([await outcome(holder, text) == name for text in barred])
    assert counts == dict.fromkeys(calibration, 0)


async def test_named_skills_and_cue_time_are_measured_into_the_report(ready):
    engine, _, _, holder = ready
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
    calibrated = load_calibration(engine.model_name)
    lines = [
        "# Affordance cue, embedder only", "",
        f"Date: {day} · Run: {run}",
        f"Commit: {git('rev-parse', 'HEAD')} ({'dirty' if git('status', '--porcelain') else 'clean'} tree)",
        f"Module sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        f"Embedder: {engine.model_name}",
        f"Threshold: {calibrated.threshold} (derived from the record dated {calibrated.date.isoformat()})", "",
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
