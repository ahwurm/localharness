"""The cue's threshold measured on this machine from the shipped labelled set, and its record in the machine directory."""
from __future__ import annotations

import datetime
import hashlib
import logging
import math
import os
import stat
import time
from pathlib import Path
from uuid import uuid4

import yaml

from localharness.agent.affordance import normalise, scores
from localharness.config.skills import Skill, read_skill
from localharness.core.private_files import touch_private

log = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
PROMPTS = ASSETS / "affordance" / "prompts.yaml"
LABELS = ("matching", "near_miss", "unrelated")
UNREAD = ("README.md", ".gitattributes")
# bumped by hand when the engine's query or document encoding changes
REV = 1
BATCH = 8
RETRY_S = 0.05
BUSY_S = 60.0
MAX_RECORD = 64 * 1024
NO_SEPARATION = "this embedding model does not separate the skills well enough"


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


def snapshot_files(folder: Path) -> dict[str, str]:
    """Each file's relative path mapped to the name of what it links to: a cache snapshot's blob."""
    return {rel: p.resolve().name for p in Path(folder).rglob("*")
            if p.is_file() and (rel := p.relative_to(folder).as_posix()) not in UNREAD}


def descriptions_digest(skills) -> str:
    text = "".join(f"{s.name}\0{s.description}\n" for s in skills)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def labelled() -> tuple[list[Skill], list[tuple[str, str, str]]]:
    """The shipped skills in the prompts' order, and every (skill, label, prompt) in that order."""
    groups = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))["skills"]
    found = [read_skill(d if (d := ASSETS / "affordance" / "skills" / name).is_dir() else ASSETS / "skills" / name,
                        "bundled") for name in groups]
    if bad := [s for s in found if not isinstance(s, Skill)]:
        raise ValueError(f"unreadable labelled skills: {bad}")
    return found, [(name, label, text) for name in groups for label in LABELS for text in groups[name][label]]


def shipped_pins(model_name: str, folder: Path) -> dict:
    return {"model_name": model_name, "snapshot_files": snapshot_files(folder),
            "prompts_sha256": hashlib.sha256(PROMPTS.read_bytes()).hexdigest(),
            "descriptions_sha256": descriptions_digest(labelled()[0]), "rev": REV}


def _row_ok(row, prompt) -> bool:
    return (isinstance(row, list) and len(row) == 4 and row[:2] == list(prompt[:2]) and isinstance(row[2], str)
            and isinstance(row[3], float) and -1 <= row[3] <= 1)


def entry_state(entry, pins: dict, prompts) -> tuple[str, str | float]:
    """The record's state for these pins, with its reason, its "k of n" or its threshold."""
    rows = entry.get("rows") if isinstance(entry, dict) else None
    if (not isinstance(rows, list) or {k: entry.get(k) for k in pins} != pins or len(rows) > len(prompts)
            or not isinstance(entry.get("date"), datetime.date) or not all(map(_row_ok, rows, prompts))):
        return "stale", "no record for this model and labelled set"
    if "failed" in entry:
        return "failed", NO_SEPARATION
    if len(rows) < len(prompts):
        return "partial", f"{len(rows)} of {len(prompts)}"
    try:
        return "ready", calibrate(rows)
    except ValueError:
        return "failed", NO_SEPARATION


def record_path(record_dir: Path, model_name: str) -> Path:
    return record_dir / f"{hashlib.sha256(model_name.encode('utf-8')).hexdigest()[:16]}.yaml"


def usable_dir(record_dir: Path | None) -> Path | None:
    """`record_dir`, created; None when it is a symlink or not a directory, which is never replaced."""
    if record_dir is None:
        return None
    try:
        record_dir.mkdir(parents=True, exist_ok=True)
        return record_dir if stat.S_ISDIR(os.lstat(record_dir).st_mode) else None
    except OSError:
        return None


def read_entry(record_dir: Path, model_name: str) -> dict | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        if not stat.S_ISDIR(os.lstat(record_dir).st_mode):
            return None
        with os.fdopen(os.open(record_path(record_dir, model_name), flags), "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return None
            data = f.read(MAX_RECORD + 1)
        entry = yaml.safe_load(data) if len(data) <= MAX_RECORD else None
    except Exception:  # noqa: BLE001 — any unreadable record is stale and measured again
        return None
    return entry if isinstance(entry, dict) else None


def write_entry(record_dir: Path, model_name: str, entry: dict) -> None:
    path = record_path(record_dir, model_name)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid4().hex[:8]}.tmp")
    try:
        touch_private(tmp)
        tmp.write_text(yaml.safe_dump(entry, sort_keys=False, default_flow_style=None), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _written(record_dir: Path | None, model_name: str, entry: dict) -> Path | None:
    if record_dir is None:
        return None
    try:
        write_entry(record_dir, model_name, entry)
        return record_dir
    except OSError as exc:
        log.info("skill cue record not written, measured for this sitting only: %s", exc)
        return None


def _query(embedder, text: str):
    deadline = time.monotonic() + BUSY_S
    while (vec := embedder.try_embed_query(text)) is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"the embedder stayed busy for {BUSY_S:g} s")
        time.sleep(RETRY_S)
    return vec


def measure(embedder, pins: dict, entry: dict | None, idle, record_dir: Path | None) -> dict:
    """The labelled prompts scored from `entry`'s rows on, a batch at a time while `idle` is set, written after each."""
    skills, prompts = labelled()
    docs = normalise(embedder.embed_docs([s.description for s in skills]))
    rows = list(entry["rows"]) if entry else []
    for start in range(len(rows), len(prompts), BATCH):
        idle.wait()
        for item, label, text in prompts[start:start + BATCH]:
            s = scores(docs, _query(embedder, text))
            i = int(s.argmax())
            rows.append([item, label, skills[i].name, float(s[i])])
        entry = {**pins, "date": datetime.date.today(), "rows": rows}
        record_dir = _written(record_dir, embedder.model_name, entry)
    try:
        calibrate(rows)
    except ValueError as exc:
        entry = {**entry, "failed": str(exc)}
        _written(record_dir, embedder.model_name, entry)
    return entry
