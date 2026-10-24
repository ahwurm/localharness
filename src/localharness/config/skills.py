"""Skill directories on disk: origins, validation rules as data, discovery, per-read checks and the listing budget."""
from __future__ import annotations

import errno
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from localharness.core.content import MAX_REFERENCE_BYTES

ORIGINS = (
    ("project", ".localharness/skills"),
    ("project", ".agents/skills"),
    ("machine", "skills"),
    ("bundled", "assets/skills"),
)
PACKAGE_DIR = Path(__file__).resolve().parent.parent

MAX_ENTRIES = 256
MAX_PROJECT = 32
MAX_SKILLS = 64
MAX_NAME = 64
MAX_DESCRIPTION = 1024
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")

OVER_256 = ("<origin> holds more than 256 entries; the rest are ignored", "remove or move skills you do not need")
NOT_REGULAR = ("<path> is not a regular file", "replace it with a regular file")
OUTSIDE = ("<dir> resolves outside <origin>", "replace the link with a directory")
CONTROL_PATH = ("<path> holds a control character", "rename it without control characters")
TOO_LARGE = ("<path> is over 200 KiB", "move detail into files under references/")
UNREADABLE = ("<path> is not readable UTF-8 text", "save it as UTF-8")
NO_FRONTMATTER = ("<path> has no frontmatter", "begin it with a --- line, then name and description, then a --- line")
UNPARSEABLE = ("<path> frontmatter is not valid YAML", "fix the YAML between the --- lines")
ANCHORS = ("<path> frontmatter uses YAML anchors or aliases", "write each value out in full")
NOT_MAPPING = ("<path> frontmatter is not a mapping with string values", "write name: and description: as plain text")
NO_NAME = ("<path> has no name", "add a name: line equal to the directory name")
BAD_NAME = ("<path> name <name> is not lowercase letters and digits joined by single hyphens", "rename it, for example release-notes")
LONG_NAME = ("<path> name <name> is over 64 characters", "shorten the name and the directory to 64 characters")
NAME_DIR = ("<path> name <name> differs from its directory", "make the name and the directory name equal")
BUNDLED_NAME = ("<path> name <name> is a bundled skill's name", "rename the skill and its directory")
NO_DESCRIPTION = ("<path> has no description", "add a description: line saying when to use the skill")
LONG_DESCRIPTION = ("<path> description is over 1024 characters", "shorten it to 1024 characters")
MULTILINE_DESCRIPTION = ("<path> description is more than one line", "write it on one line")
CONTROL_DESCRIPTION = ("<path> description holds a control character", "remove the control characters")
CAP_PROJECT = ("<dir> is past the 32 project skills kept", "remove project skills you do not need")
CAP_TOTAL = ("<dir> is past the 64 skills kept", "remove skills you do not need")
SHADOWED = ("<dir> is shadowed by <other>", "rename yours to keep both")
RULES = (
    OVER_256, NOT_REGULAR, OUTSIDE, CONTROL_PATH, TOO_LARGE, UNREADABLE, NO_FRONTMATTER, UNPARSEABLE,
    ANCHORS, NOT_MAPPING, NO_NAME, BAD_NAME, LONG_NAME, NAME_DIR, BUNDLED_NAME, NO_DESCRIPTION,
    LONG_DESCRIPTION, MULTILINE_DESCRIPTION, CONTROL_DESCRIPTION, CAP_PROJECT, CAP_TOTAL, SHADOWED,
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    origin: str
    path: Path
    untrusted: bool


@dataclass(frozen=True)
class Skipped:
    reason: str
    fix: str
    path: Path


def skip_line(skipped: Skipped) -> str:
    return f"skipped: {skipped.reason}. {skipped.fix}."


def _skip(pair: tuple[str, str], at: Path, /, **fields: object) -> Skipped:
    reason, fix = pair
    return Skipped(re.sub(r"<(\w+)>", lambda m: repr(str(fields[m.group(1)])), reason), fix, at)


def _origin_dirs(config_dir, project_root) -> tuple[tuple[str, Path, Path], ...]:
    has_project = project_root is not None and (Path(project_root) / ".localharness").is_dir()
    bases = {"project": project_root if has_project else None, "machine": config_dir, "bundled": PACKAGE_DIR}
    found = []
    for label, rel in ORIGINS:
        if bases[label] is None:
            continue
        try:
            real_dir = Path(bases[label]).resolve() / rel
        except (OSError, RuntimeError, ValueError):
            continue
        found.append((label, Path(bases[label]) / rel, real_dir))
    return tuple(found)


def _entries(lexical_dir: Path) -> tuple[list[str], Skipped | None]:
    try:
        names = sorted(os.listdir(lexical_dir))
    except OSError:
        return [], None
    if len(names) <= MAX_ENTRIES:
        return names, None
    return names[:MAX_ENTRIES], _skip(OVER_256, lexical_dir, origin=lexical_dir)


def _read(path: Path) -> str | tuple[str, str]:
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return NOT_REGULAR if exc.errno == errno.ELOOP else UNREADABLE
    chunks: list[bytes] = []
    total = 0
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return NOT_REGULAR
        if info.st_size > MAX_REFERENCE_BYTES:
            return TOO_LARGE
        while total <= MAX_REFERENCE_BYTES:
            chunk = os.read(fd, min(1 << 16, MAX_REFERENCE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except OSError:
        return UNREADABLE
    finally:
        os.close(fd)
    if total > MAX_REFERENCE_BYTES:
        return TOO_LARGE
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        return UNREADABLE


def _control(text: str) -> bool:
    return any(unicodedata.category(c)[0] == "C" or unicodedata.category(c) in ("Zl", "Zp") for c in text)


def _inside(directory: Path, real_dir: Path) -> bool:
    try:
        return directory.resolve().is_relative_to(real_dir)
    except (OSError, RuntimeError, ValueError):
        return False


def _load(skill_md: Path, lexical_dir: Path, real_dir: Path) -> str | Skipped:
    try:
        info = os.lstat(skill_md)
    except OSError:
        info = None
    if info is None or not stat.S_ISREG(info.st_mode):
        result = NOT_REGULAR
    elif not _inside(skill_md.parent, real_dir):
        return _skip(OUTSIDE, skill_md.parent, dir=skill_md.parent, origin=lexical_dir)
    elif _control(str(skill_md)):
        result = CONTROL_PATH
    elif info.st_size > MAX_REFERENCE_BYTES:
        result = TOO_LARGE
    else:
        result = _read(skill_md)
    return _skip(result, skill_md.parent, path=skill_md) if isinstance(result, tuple) else result


def _parse(text: str) -> dict | tuple[str, str]:
    lines = text.split("\n")
    if lines[0].rstrip() != "---":
        return NO_FRONTMATTER
    close = next((i for i, line in enumerate(lines[1:], 1) if line.rstrip() == "---"), None)
    if close is None:
        return NO_FRONTMATTER
    front = "\n".join(lines[1:close])
    try:
        data = yaml.safe_load(front)
        # yaml.scan took 9.2 s on 40,000 open brackets where safe_load raised RecursionError on
        # 100,000 in 0.4 s (PyYAML 6.0.3, Python 3.12); safe_load shares aliased objects, so loading first expands nothing
        anchored = any(isinstance(t, (yaml.AnchorToken, yaml.AliasToken)) for t in yaml.scan(front))
    except Exception:
        return UNPARSEABLE
    if anchored:
        return ANCHORS
    if not isinstance(data, dict):
        return NOT_MAPPING
    if any(data.get(key) is not None and not isinstance(data[key], str) for key in ("name", "description")):
        return NOT_MAPPING
    return data


def _check(label: str, lexical_dir: Path, real_dir: Path, entry: str, bundled: frozenset[str]) -> Skill | Skipped | None:
    skill_md = lexical_dir / entry / "SKILL.md"
    if not os.path.lexists(skill_md):
        return None
    text = _load(skill_md, lexical_dir, real_dir)
    if isinstance(text, Skipped):
        return text
    data = _parse(text)
    if isinstance(data, tuple):
        return _skip(data, skill_md.parent, path=skill_md)
    name = data.get("name")
    if name is None:
        return _skip(NO_NAME, skill_md.parent, path=skill_md)
    cut = name[:MAX_NAME]
    if not NAME.fullmatch(name):
        return _skip(BAD_NAME, skill_md.parent, path=skill_md, name=cut)
    if len(name) > MAX_NAME:
        return _skip(LONG_NAME, skill_md.parent, path=skill_md, name=cut)
    if name != entry:
        return _skip(NAME_DIR, skill_md.parent, path=skill_md, name=cut)
    if label != "bundled" and name in bundled:
        return _skip(BUNDLED_NAME, skill_md.parent, path=skill_md, name=cut)
    description = (data.get("description") or "").removesuffix("\n")
    if description.strip() == "":
        return _skip(NO_DESCRIPTION, skill_md.parent, path=skill_md)
    if len(description) > MAX_DESCRIPTION:
        return _skip(LONG_DESCRIPTION, skill_md.parent, path=skill_md)
    if description.splitlines() != [description]:
        return _skip(MULTILINE_DESCRIPTION, skill_md.parent, path=skill_md)
    if _control(description):
        return _skip(CONTROL_DESCRIPTION, skill_md.parent, path=skill_md)
    return Skill(name, description, label, skill_md, label == "project")


def _scan(config_dir, project_root) -> tuple[tuple[str | None, Skill | Skipped], ...]:
    listed = [(label, lexical, real, *_entries(lexical)) for label, lexical, real in _origin_dirs(config_dir, project_root)]

    def outcomes(row, bundled):
        label, lexical, real, names, _ = row
        return [(e, r) for e in names if (r := _check(label, lexical, real, e, bundled)) is not None]

    done = {i: outcomes(row, frozenset()) for i, row in enumerate(listed) if row[0] == "bundled"}
    bundled = frozenset(e for found in done.values() for e, r in found if isinstance(r, Skill))
    rows: list[tuple[str | None, Skill | Skipped]] = []
    for i, row in enumerate(listed):
        rows += [(None, row[4])] if row[4] else []
        rows += done[i] if i in done else outcomes(row, bundled)
    kept: dict[str, Path] = {}
    project = 0
    scanned = []
    for entry, item in rows:
        if isinstance(item, Skill):
            here = item.path.parent
            if item.origin == "project" and project >= MAX_PROJECT:
                item = _skip(CAP_PROJECT, here, dir=here)
            elif len(kept) >= MAX_SKILLS:
                item = _skip(CAP_TOTAL, here, dir=here)
            elif entry in kept:
                item = _skip(SHADOWED, here, dir=here, other=kept[entry])
            else:
                kept[entry] = here
                project += item.origin == "project"
        scanned.append((entry, item))
    return tuple(scanned)


@dataclass
class SkillSet:
    skills: tuple[Skill, ...]
    skipped: tuple[Skipped, ...]
    config_dir: Path
    project_root: Path | None

    @classmethod
    def discover(cls, config_dir, project_root) -> SkillSet:
        scanned = [item for _, item in _scan(config_dir, project_root)]
        return cls(
            tuple(i for i in scanned if isinstance(i, Skill)),
            tuple(i for i in scanned if isinstance(i, Skipped)),
            Path(config_dir),
            project_root,
        )

    def read_body(self, record: Skill) -> str | Skipped:
        """Return the whole SKILL.md text after repeating every discovery check."""
        here = record.path.parent
        for _, lexical, real in _origin_dirs(self.config_dir, self.project_root):
            if lexical == here.parent:
                return _load(record.path, lexical, real)
        return _skip(OUTSIDE, here, dir=here, origin=here.parent)

    def find(self, name) -> Skill | Skipped | None:
        """Look a typed name up, rescanning with the start-time workspace decision; the one method that grows the set."""
        if not isinstance(name, str) or len(name) > MAX_NAME or not NAME.fullmatch(name):
            return None
        live = next((s for s in self.skills if s.name == name), None)
        if live is not None:
            return live
        hits = [item for entry, item in _scan(self.config_dir, self.project_root) if entry == name]
        record = next((i for i in hits if isinstance(i, Skill)), None)
        if record is None:
            return hits[0] if hits else None
        here = record.path.parent
        if record.origin == "project" and sum(s.origin == "project" for s in self.skills) >= MAX_PROJECT:
            return _skip(CAP_PROJECT, here, dir=here)
        if len(self.skills) >= MAX_SKILLS:
            return _skip(CAP_TOTAL, here, dir=here)
        self.skills = self.skills + (record,)
        return record


LISTING_BUDGET_CHARS = 1_300


def shown(skills: tuple[Skill, ...]) -> tuple[tuple[str, str, str | None, int], ...]:
    """Rows of (name, label, description or None, fit_len) under the listing budget, project descriptions dropped first."""
    total = sum(len(s.description) for s in skills)
    order = sorted(range(len(skills)), key=lambda i: (skills[i].origin != "project", -len(skills[i].description), skills[i].name, i))
    dropped = set()
    for i in order:
        if total <= LISTING_BUDGET_CHARS:
            break
        dropped.add(i)
        total -= len(skills[i].description)
    slack = LISTING_BUDGET_CHARS - total
    return tuple(
        (s.name, s.origin, None, slack) if i in dropped else (s.name, s.origin, s.description, slack + len(s.description))
        for i, s in enumerate(skills)
    )
