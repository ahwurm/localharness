"""The task flow's filer: the no-follow walk to an agent's task folders and every read and write of a task doc."""
from __future__ import annotations

import errno
import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from localharness.channels.base import one_line, sanitize_for_display
from localharness.core.caps import Caps
from localharness.core.private_files import PRIVATE_DIR_MODE, NotRegularFile, atomic_write_text, open_regular
from localharness.orchestrator.task_flow.record import Counts, Record, Row

if TYPE_CHECKING:
    from localharness.orchestrator.task_flow.panel import Finding

TRUNCATION_MARKER = "\n[truncated]\n"
SESSION_DOCS = ("jtbd.md", "reply-review.md")
_BOUND = Caps.DOC + len(TRUNCATION_MARKER.encode())
_DIR = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
_FOLDER = re.compile(r"t[0-9]+")
_TASK_DOC = re.compile(r"(?:(?:decisions|research|plan|step-[1-9][0-9]*)(?:-review)?|review)\.md")
_ATTEMPTS = 8
PLAN_SHAPE = ("plan.md needs `## Step N` headings numbered 1..n in order, each with `files:`, `depends_on:` "
              "(earlier steps only, or none) and `done:` lines")
NO_REVIEW = "no review yet"
_HEADING = re.compile(r"## Step ([1-9][0-9]*)(?:[:\s].*)?")
_FIELD = re.compile(r"(?:[-*] )?(files|depends_on|done):(.*)")
_SPLIT = re.compile(r"(?m)^(?=##(?!#))")
_STEP_KEY = re.compile(r"step-([1-9][0-9]*)")
_REVIEW_DOC = re.compile(r"(?:[a-z0-9-]+-)?review\.md")
_HEADING_LINE = re.compile(r"(?m)^#")
VERDICTS = ("pass", "blocked", "accepted", "unreviewed")
UNANSWERED = "unanswered"


class Refused(Exception):
    """A path the walk will not follow, or a doc the filer will not touch: one line for the task call."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def valid_component(name: str) -> bool:
    return bool(name) and "/" not in name and os.sep not in name and "\x00" not in name and name not in (".", "..")


def _open_dir(path: str | os.PathLike[str], nofollow: bool, dir_fd: int | None = None) -> int:
    try:
        return os.open(path, _DIR | (getattr(os, "O_NOFOLLOW", 0) if nofollow else 0), dir_fd=dir_fd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise Refused(f"{Path(path).name} is a link or not a folder; the filer does not follow it") from e
        raise


def _walk(root: Path, workspace_layer: bool, parts: tuple[str, ...], create: bool) -> int:
    """An open descriptor for `root/parts`, one component at a time. The root and `agents` follow a link
    only on the machine layer; every other component refuses one."""
    fd = _open_dir(root, workspace_layer)
    for i, name in enumerate(parts):
        try:
            if create:
                try:
                    os.mkdir(name, PRIVATE_DIR_MODE, dir_fd=fd)
                except FileExistsError:
                    pass
            nxt = _open_dir(name, workspace_layer or i > 0, fd)
        finally:
            os.close(fd)
        fd = nxt
    return fd


def _listing(fd: int) -> list[str]:
    return os.listdir(fd)


def _highest(fd: int) -> int:
    return max((int(n[1:]) for n in _listing(fd) if _FOLDER.fullmatch(n)), default=0)


def _folder(task: int, name: str) -> str:
    folder = f"t{task}"
    if not _FOLDER.fullmatch(folder) or not _TASK_DOC.fullmatch(name):
        raise Refused(f"{folder}/{name} is not a task doc")
    return folder


def _session(name: str) -> str:
    if name not in SESSION_DOCS:
        raise Refused(f"{name} is not a session doc")
    return name


def _cut(text: str) -> tuple[str, bool]:
    raw = text.encode()
    return (text, False) if len(raw) <= Caps.DOC else (raw[: Caps.DOC].decode(errors="ignore") + TRUNCATION_MARKER, True)


def _clip(text: str, limit: int) -> str:
    return text.encode()[:limit].decode(errors="ignore")


def _slurp(fd: int, name: str) -> str | None:
    """The doc `name` inside `fd`, or None when it is missing or longer than a doc can be."""
    try:
        with os.fdopen(open_regular(name, dir_fd=fd), "rb") as fh:
            raw = fh.read(_BOUND + 1)
    except FileNotFoundError:
        return None
    except NotRegularFile as e:
        raise Refused(f"{name} is not a regular file") from e
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise Refused(f"{name} is a link; the filer does not follow it") from e
        raise
    if len(raw) > _BOUND:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError as e:
        raise Refused(f"{name} is not UTF-8") from e


@dataclass(frozen=True)
class Section:
    n: int
    text: str  # as written; `digest` hashes the whitespace-normalised form
    files: tuple[str, ...]
    depends_on: tuple[int, ...]
    done: str
    digest: str


def _normal(chunk: str) -> list[str]:
    return [line for line in (" ".join(raw.split()) for raw in chunk.split("\n")) if line]


def _depends(value: str, step: int) -> tuple[int, ...] | None:
    if value.lower() == "none":
        return ()
    earlier = {str(k): k for k in range(1, step)}
    deps = tuple(dict.fromkeys(earlier.get(c.strip()) for c in value.split(",")))
    return None if None in deps else deps


def parse_plan(text: str) -> list[Section] | str:
    """The `## Step N` sections of a plan, or `PLAN_SHAPE`; text before the first heading is no section."""
    out: list[Section] = []
    for i, chunk in enumerate(_SPLIT.split(text)[1:], 1):
        head, *body = lines = _normal(chunk)
        heading, found = _HEADING.fullmatch(head), [m for line in body if (m := _FIELD.fullmatch(line))]
        if not heading or heading[1] != str(i) or sorted(m[1] for m in found) != ["depends_on", "done", "files"]:
            return PLAN_SHAPE
        fields = {m[1]: m[2].strip() for m in found}
        deps = _depends(fields["depends_on"], i)
        if deps is None:
            return PLAN_SHAPE
        out.append(Section(i, chunk.strip(), tuple(f.strip() for f in fields["files"].split(",") if f.strip()), deps,
                           fields["done"], digest("\n".join(lines))))
    return out or PLAN_SHAPE


def rerecord(record: Record, sections: list[Section]) -> list[int]:
    """A row per section, kept while its digest and every dependency are; returns the steps whose executed work is void."""
    kept: set[int] = set()
    reopened: list[int] = []
    for s in sections:
        key = ("plan.md", f"step-{s.n}")
        prior = record.filed.get(key)
        if prior and prior.digest == s.digest and kept.issuperset(s.depends_on):
            kept.add(s.n)
            continue
        record.filed[key] = Row(s.digest, "unreviewed", Counts())
        if record.filed.pop((f"step-{s.n}.md", None), None):
            reopened.append(s.n)
    live = {f"step-{s.n}" for s in sections}
    for key in [k for k in record.filed if k[0] == "plan.md" and k[1] and k[1] not in live]:
        del record.filed[key]
        record.filed.pop((f"{key[1]}.md", None), None)
    record.stopped_log = ""
    return reopened


def _flat(text: str, cap: int = Caps.LINE) -> str:
    return one_line(text)[:cap]


def _quote(text: str) -> str:
    """Child text sanitised for display and with its heading marks escaped, so it cannot open a section of the doc that holds it."""
    return _HEADING_LINE.sub(lambda _: "\\#", sanitize_for_display(text))


def render_section(round_n: int | None, findings: Iterable[Finding], verdict: str, blockers: int) -> str:
    if verdict not in VERDICTS:
        raise ValueError(f"{verdict!r} is not a review verdict")
    head = "## accepted by the person" if round_n is None else f"## round {round_n} · blockers {blockers}"
    rows = (f"- [{f.tag}] {_flat(' — '.join(p for p in (f.rule, f.evidence) if p), Caps.FINDING)}"
            + (f" ({_flat(f.critic)})" if f.critic else "") for f in findings)
    return "\n".join([head, *rows, f"verdict: {verdict}"])


def _sections(text: str) -> list[str]:
    return [p for p in _SPLIT.split(text) if p.startswith("##")]


def last_section(text: str) -> str:
    parts = _sections(text)
    return parts[-1].strip() if parts else NO_REVIEW


def _newest(sections: list[str]) -> str:
    """The sections joined, the oldest dropped until they fit a doc, so the latest verdict is never the cut."""
    while len("".join(sections).encode()) > Caps.DOC and len(sections) > 1:
        sections = sections[1:]
    return "".join(sections)


class StepLog:
    """One line per tool event, kept in memory until the executor returns or is cancelled."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str) -> None:
        self.lines.append(_flat(line))

    def render(self, final: str, runtime: dict, stopped: str = "") -> str:
        files, tools = [_flat(str(f)) for f in runtime.get("files", ())], runtime.get("tools", {})
        called = ", ".join(f"{_flat(str(k))} x{v}" for k, v in tools.items())
        tail = "\n".join([
            *(["## stopped", f"stopped at {self.lines[-1] if self.lines else 'the start'} · {_flat(stopped)}"] if stopped else []),
            *(["## final answer", _clip(_quote(final.strip()), Caps.BLOCK)] if final.strip() else []),
            "## runtime",
            f"files written: {len(files)}" + (f": {_clip(', '.join(files), Caps.BLOCK // 4)}" if files else ""),
            f"tools called: {_clip(called, Caps.BLOCK // 4) or 'none'}",
            f"compactions: {int(runtime.get('compactions', 0))}"])
        kept = self.lines
        while kept and len("\n".join(["## log", *kept, tail]).encode()) > Caps.DOC:
            kept = kept[1:]
        dropped = len(self.lines) - len(kept)
        return "\n".join(["## log", *([f"[{dropped} earlier events dropped]"] if dropped else []), *kept, tail])


def answered(answer: str) -> bool:
    return answer.strip().lower() not in ("", "none", "blocked")


def file_answer(text: str, n: int, question: str, explorer: str, answer: str, ended: str = "") -> str:
    """`research.md` with question `n`'s section written or replaced; `none`, `blocked` and empty leave it unanswered, an explorer that `ended` early files its reason and no answer."""
    body = f"stopped: {_flat(ended)}" if ended else _clip(_quote(answer.strip()), Caps.BLOCK // 4) if answered(answer) else UNANSWERED
    section = "\n".join([f"## q{n} · {_flat(question)}", body, f"explorer: {_flat(explorer)}"])
    parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    out = [section if p.startswith(f"## q{n} ·") else p for p in parts]
    return "\n\n".join(out if section in out else [*out, section]) + "\n"


def manifest(filed: Iterable[str], child: str = "", verdict: str = "", doc: str = "", findings: str = "",
             notes: Iterable[str] = (), next_line: str = "") -> str:
    """What the root reads from a unit: filed names, counts, verdicts and fenced findings, never a child's output."""
    names = ", ".join(filed)
    parts = (f"filed: {names}" if names else "", _flat(child, Caps.FINDING), _flat(verdict, Caps.FINDING),
             f"doc:\n{doc.strip()}" if doc.strip() else "", findings,
             *(f"note: {_flat(n, Caps.FINDING)}" for n in notes), next_line)
    return "\n".join(p for p in parts if p)


@dataclass
class Docs:
    state_dir: Path
    workspace_layer: bool
    agent: str
    adopted: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if not valid_component(self.agent):
            raise Refused(f"{self.agent!r} is not one folder name")

    @contextmanager
    def _dir(self, *parts: str, create: bool):
        fd = _walk(self.state_dir, self.workspace_layer, ("agents", self.agent, *parts), create)
        try:
            yield fd
        finally:
            os.close(fd)

    def new_task(self) -> int:
        with self._dir("tasks", create=True) as fd:
            n = _highest(fd) + 1
            for _ in range(_ATTEMPTS):
                try:
                    os.mkdir(f"t{n}", PRIVATE_DIR_MODE, dir_fd=fd)
                except FileExistsError:
                    n = max(n, _highest(fd)) + 1
                    continue
                self.adopted.add(n)
                return n
        raise Refused("no free task folder")

    def write(self, task: int, name: str, text: str) -> tuple[str, bool]:
        folder = _folder(task, name)
        if task not in self.adopted:
            raise Refused(f"{folder} was not made by this session; it is read-only")
        cut, truncated = _cut(text)
        with self._dir("tasks", folder, create=False) as fd:
            atomic_write_text(name, cut, dir_fd=fd)
        return digest(cut), truncated

    def read(self, task: int, name: str) -> str | None:
        folder = _folder(task, name)
        try:
            with self._dir("tasks", folder, create=False) as fd:
                return _slurp(fd, name)
        except FileNotFoundError:
            return None

    def read_session(self, name: str) -> str:
        try:
            with self._dir("sessions", create=False) as fd:
                return _slurp(fd, _session(name)) or ""
        except FileNotFoundError:
            return ""

    def write_session(self, name: str, text: str) -> None:
        with self._dir("sessions", create=True) as fd:
            atomic_write_text(_session(name), _cut(text)[0], dir_fd=fd)

    def reset_sessions(self) -> None:
        for name in SESSION_DOCS:
            self.write_session(name, "")

    def path(self, task: int, name: str) -> Path:
        return self.state_dir / "agents" / self.agent / "tasks" / f"t{task}" / name

    def append_section(self, task: int, doc: str, section: str) -> str:
        if not _REVIEW_DOC.fullmatch(doc):
            raise Refused(f"{doc} is not a review doc")
        text = "\n\n".join(p for p in ((self.read(task, doc) or "").strip(), section.strip()) if p)
        body = _newest([f"{p.strip()}\n\n" for p in _SPLIT.split(text) if p.strip()])
        return self.write(task, doc, body.rstrip() + "\n")[0]

    def write_step(self, task: int, n: int, log: StepLog, final: str, runtime: dict, stopped: str = "") -> tuple[str, bool]:
        return self.write(task, f"step-{n}.md", log.render(final, runtime, stopped))

    def verified(self, record: Record, key: tuple[str, str | None]) -> bool:
        """A doc's, or a plan section's, current text still has the digest it was filed with."""
        row = record.filed.get(key)
        try:
            text = self.read(record.id, key[0])
        except Refused:
            return False
        if row is None or text is None:
            return False
        if key[1] is None:
            return digest(text) == row.digest
        n, plan = _STEP_KEY.fullmatch(key[1]), parse_plan(text)
        return bool(n) and not isinstance(plan, str) and any(s.n == int(n[1]) and s.digest == row.digest for s in plan)
