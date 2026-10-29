"""The skill cue: the turn scored against the skill descriptions, and one skill named past a calibrated threshold."""
from __future__ import annotations

import asyncio
import json
import logging
import string
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from localharness.core.types import is_harness_message

if TYPE_CHECKING:
    from localharness.agent.task_record import TaskState
    from localharness.config.skills import Skill, SkillSet
    from localharness.plugins.api import QueryEmbedder
    from localharness.plugins.slot import MemorySlot

log = logging.getLogger(__name__)

CUE_LINE = "A skill may match this request: skill('{name}'){origin}. Call it if it applies; otherwise ignore this."


@dataclass(frozen=True)
class Cue:
    name: str
    origin: str
    line: str


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

    def __init__(self, skill_set: SkillSet, memory_slot: MemorySlot, task_state: TaskState | None,
                 record_dir: Path | None = None) -> None:
        self._skill_set, self._slot, self._task_state, self._record_dir = skill_set, memory_slot, task_state, record_dir
        self._skills: tuple[Skill, ...] = ()
        self._ready: tuple[QueryEmbedder, np.ndarray, float] | None = None
        self.named: str | None = None
        self._turns, self._idle = 0, threading.Event()
        self._idle.set()

    def turn_started(self) -> None:
        self._turns += 1
        self._idle.clear()

    def turn_ended(self) -> None:
        self._turns -= 1
        if self._turns == 0:
            self._idle.set()

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
        from localharness.agent.calibration import entry_state, labelled, measure, read_entry, shipped_pins, usable_dir

        embedder = self._slot.embedder()
        if embedder is None:
            return "the memory slot offers no embedder"
        folder = embedder.ensure_local()
        if folder is None:
            return "embedding model not on this machine yet"
        pins, prompts, record_dir = shipped_pins(embedder.model_name, folder), labelled()[1], usable_dir(self._record_dir)
        entry = read_entry(record_dir, embedder.model_name) if record_dir is not None else None
        state, detail = entry_state(entry, pins, prompts)
        if state in ("stale", "partial"):
            entry = measure(embedder, pins, entry if state == "partial" else None, self._idle, record_dir)
            state, detail = entry_state(entry, pins, prompts)
        if state != "ready":
            return detail
        matrix = normalise(embedder.embed_docs([s.description for s in skills]))
        if matrix.ndim != 2 or len(matrix) != len(skills):
            return f"{matrix.shape} vectors for {len(skills)} descriptions"
        self._ready = (embedder, matrix, detail)
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
