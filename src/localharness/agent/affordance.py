"""The skill cue: the turn scored against the skill descriptions, and one skill named past a calibrated threshold."""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import math
import string
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from localharness.core.types import is_harness_message

if TYPE_CHECKING:
    from localharness.agent.task_record import TaskState
    from localharness.config.skills import Skill, SkillSet
    from localharness.plugins.api import QueryEmbedder
    from localharness.plugins.slot import MemorySlot

log = logging.getLogger(__name__)

CUE_LINE = "A skill may match this request: skill('{name}'){origin}. Call it if it applies; otherwise ignore this."
CALIBRATION = Path(__file__).resolve().parent.parent / "assets" / "affordance" / "calibration.yaml"
LABELS = ("matching", "near_miss", "unrelated")
UNREAD = ("README.md", ".gitattributes")


@dataclass(frozen=True)
class Cue:
    name: str
    origin: str
    line: str


@dataclass(frozen=True)
class Calibration:
    threshold: float
    snapshot_files: dict[str, str]
    date: datetime.date


def normalise(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim not in (1, 2) or not np.isfinite(a).all():
        raise ValueError(f"not a finite vector or matrix: shape {a.shape}")
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    if not (np.isfinite(norms) & (norms > 0)).all():
        raise ValueError("a zero or non-finite norm")
    return a / norms


def scores(matrix, vec) -> np.ndarray:
    matrix, vec = np.asarray(matrix), np.asarray(vec)
    if matrix.ndim != 2 or vec.ndim != 1 or matrix.shape[1] != vec.shape[0]:
        raise ValueError(f"cannot score a {vec.shape} vector against a {matrix.shape} matrix")
    return normalise(matrix) @ normalise(vec)


def pick(scores: np.ndarray, threshold: float) -> int | None:
    if not len(scores):
        return None
    i = int(scores.argmax())
    return i if scores[i] > threshold else None


def _line(skill: Skill) -> str:
    return CUE_LINE.format(name=skill.name, origin=" (a project skill)" if skill.origin == "project" else "")


def calibrate(rows) -> float:
    """The midpoint between the highest must-not-cue score and the lowest admitted score above it."""
    for row in rows:
        _, label, _, score = row
        if label not in LABELS or not math.isfinite(float(score)) or not -1 <= float(score) <= 1:
            raise ValueError(f"bad calibration row {row!r}")
    barred = [float(s) for item, label, top, s in rows if label != "matching" or top != item]
    if not barred:
        raise ValueError("no calibration row must not cue")
    f = max(barred)
    above = [float(s) for item, label, top, s in rows if label == "matching" and top == item and float(s) > f]
    return (f + min(above)) / 2 if above else f


def load_calibration(model_name: str) -> Calibration | None:
    record = yaml.safe_load(CALIBRATION.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{CALIBRATION} is not a mapping")
    entry = record.get(model_name)
    if entry is None:
        return None
    if not isinstance(entry["date"], datetime.date):
        raise ValueError(f"{CALIBRATION}: {model_name}'s date is not a date")
    return Calibration(calibrate([tuple(r) for r in entry["rows"]]), dict(entry["snapshot_files"]), entry["date"])


def snapshot_files(folder: Path) -> dict[str, str]:
    """Each file's relative path mapped to the name of what it links to: a cache snapshot's blob."""
    return {rel: p.resolve().name for p in Path(folder).rglob("*")
            if p.is_file() and (rel := p.relative_to(folder).as_posix()) not in UNREAD}


def descriptions_digest(skills) -> str:
    text = "".join(f"{s.name}\0{s.description}\n" for s in sorted(skills, key=lambda s: s.name))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _field(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _called_skill(call) -> str | None:
    fn = _field(call, "function")
    if fn is None or _field(fn, "name") != "skill":
        return None
    try:
        args = json.loads(_field(fn, "arguments"))
    except (TypeError, ValueError):
        return None
    name = args.get("name") if isinstance(args, dict) else None
    return name if isinstance(name, str) else None


def held_skills(messages) -> set[str]:
    """The names the model asked the skill tool for, whatever the call's outcome."""
    return {_called_skill(c) for m in messages if m.get("role") == "assistant"
            for c in m.get("tool_calls") or ()} - {None}


def cued_skills(messages, skills) -> set[str]:
    lines = {m.get("content") for m in messages
             if is_harness_message(m) and m["_lh"].get("subtype") == "affordance_cue"}
    return {s.name for s in skills if _line(s) in lines}


def task_skills(task_state: TaskState | None, skills) -> set[str]:
    rec = task_state._live() if task_state is not None else None
    sources = {r.source for r in rec.references} if rec is not None else set()
    return {s.name for s in skills if f"skill {s.name}" in sources}


def _named_in(turn: str, names) -> set[str]:
    return set(names) & {t.strip(string.punctuation).casefold() for t in turn.split()}


class Affordances:
    """The cue's holder for one sitting: description vectors warmed once from a local snapshot, at most one cue a turn."""

    def __init__(self, skill_set: SkillSet, memory_slot: MemorySlot, task_state: TaskState | None) -> None:
        self._skill_set, self._slot, self._task_state = skill_set, memory_slot, task_state
        self._skills: tuple[Skill, ...] = ()
        self._ready: tuple[QueryEmbedder, np.ndarray, float] | None = None
        self.named: str | None = None

    def warm(self) -> threading.Thread:
        self._skills = self._skill_set.skills
        thread = threading.Thread(target=self._warm, args=(self._skills,), name="affordance-warm", daemon=True)
        thread.start()
        return thread

    def _warm(self, skills: tuple[Skill, ...]) -> None:
        try:
            why = self._publish(skills)
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
        if why:
            log.info("skill cue off: %s", why)

    def _publish(self, skills: tuple[Skill, ...]) -> str | None:
        embedder = self._slot.embedder()
        if embedder is None:
            return "the memory slot offers no embedder"
        calibration = load_calibration(embedder.model_name)
        if calibration is None:
            return f"no calibration for {embedder.model_name}"
        folder = embedder.ensure_local()
        if folder is None:
            return "embedding model not on this machine yet"
        if snapshot_files(folder) != calibration.snapshot_files:
            return f"{folder} is not the snapshot the calibration was measured on"
        matrix = normalise(embedder.embed_docs([s.description for s in skills]))
        if matrix.ndim != 2 or len(matrix) != len(skills):
            return f"{matrix.shape} vectors for {len(skills)} descriptions"
        self._ready = (embedder, matrix, calibration.threshold)
        return None

    async def cue(self, turn: str, messages) -> Cue | None:
        try:
            return await self._cue(turn, messages)
        except Exception as exc:
            log.warning("skill cue off for this sitting: %s: %s", type(exc).__name__, exc)
            self._ready = None
            return None

    async def _cue(self, turn: str, messages) -> Cue | None:
        ready, skills = self._ready, self._skills
        if ready is None or not turn.strip():
            return None
        embedder, matrix, threshold = ready
        vec = await asyncio.to_thread(embedder.try_embed_query, turn)
        if vec is None:
            return None
        named = _named_in(turn, [s.name for s in skills]) | ({self.named} if self.named else set())
        keep = [i for i, s in enumerate(skills) if s.name not in named]
        i = pick(scores(matrix, vec)[keep], threshold)
        if i is None:
            return None
        skill = skills[keep[i]]
        in_view = held_skills(messages) | cued_skills(messages, skills) | task_skills(self._task_state, skills)
        return None if skill.name in in_view else Cue(skill.name, skill.origin, _line(skill))
