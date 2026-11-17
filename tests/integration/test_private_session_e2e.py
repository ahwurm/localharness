"""A private session lands every write in its run folder, in a file the agent wrote, or in `KEPT`.

REAL: `_start_async` on a private run folder, the REPL's own handlers and the gate. Stubbed: the model
(scripted) and the person (one scripted asker). An audit hook records each in-process write while a row
runs; the config folder and the project are compared byte for byte before and after.
"""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import yaml

from localharness.agent.gate_types import Decision
from localharness.agent.loop import PRIVATE_PROMPT_LINE
from localharness.core.events import Observation
from localharness.cli.workspace import run_home
from localharness.core.run_folder import begin, end
from tests.conftest import FakeLLMResponse
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_memory_compat_baseline_e2e import _no_session_start_pass, _seed_store
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _stub_start_boundaries
from tests.unit.test_web_fetch_guard import _serve, _text, public_web

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")

KEPT = (
    ("config migration and backups, the config folder's mode, and the seeded script",
     ("config.yaml", "config.yaml.bak-*", "tools/design-screenshot.js", "tools")),
    ("the model server with its state and logs, and the model library",
     ("server", "server/*", "models/*", "launch/*")),
    ("the window floor, the speed ledger, the skill-cue calibration and the inference lock",
     ("server-floor.json", "speed_stats.json", "skill_cues.json", "inference.lock")),
    ("the machine trust record, and declined offers keyed by binary, digest, posture or host",
     ("trusted_workspaces.yaml", "declined_workspace_offers.yaml")),
    ("the grants.yaml.lock and project_records.yaml.lock sidecars",
     ("grants.yaml.lock", "project_records.yaml.lock")),
    ("the embedding model and tiktoken caches, and phone pairing",
     ("embeddings/*", "tiktoken/*", "mobile/*")),
    ("/model, /effort, /plugins and /mcp, with their audit rows",
     ("audit.jsonl", "plugins/*", "mcp/*")),
)

PROJECT_RUN = (".localharness/", ".localharness/run/", ".localharness/run/.gitignore")

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
SKIP = re.compile(r"^/(proc|dev|sys)/")


@dataclass
class Recorder:
    on: bool = False
    events: list = field(default_factory=list)


RECORD = Recorder()


def _where(path, dir_fd=None) -> tuple[str, bool] | None:
    """The absolute path an event names, and whether it is certain (a relative name with no folder fd is a guess)."""
    if isinstance(path, int):
        return None
    try:
        text = os.fsdecode(path)
    except TypeError:
        return None
    if os.path.isabs(text):
        return os.path.normpath(text), True
    if isinstance(dir_fd, int) and dir_fd >= 0:
        try:
            return os.path.normpath(os.path.join(os.readlink(f"/proc/self/fd/{dir_fd}"), text)), True
        except OSError:
            return None
    return os.path.normpath(os.path.join(os.getcwd(), text)), False


def _record(event: str, args: tuple) -> None:
    found: list = []
    if event == "open":
        path, mode, flags = args
        write = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(
            isinstance(flags, int) and flags & WRITE_FLAGS)
        hit = _where(path) if write else None
        found = [hit] if hit and not os.path.isdir(hit[0]) else []
    elif event == "os.mkdir":
        made = _where(args[0], args[2])
        found = [made] if made and not os.path.isdir(made[0]) else []
    elif event in ("os.remove", "os.rmdir", "shutil.rmtree"):
        found = [_where(args[0], args[1])]
    elif event in ("os.truncate", "tempfile.mkstemp", "tempfile.mkdtemp", "sqlite3.connect"):
        found = [_where(args[0])]
    elif event in ("os.rename", "os.link"):
        found = [_where(args[0], args[2]), _where(args[1], args[3])]
    elif event == "os.symlink":
        found = [_where(args[1], args[2])]
    elif event in ("shutil.copyfile", "shutil.copytree", "shutil.move", "shutil.copymode", "shutil.copystat"):
        found = [_where(args[1])]
    for hit in found:
        if hit and not SKIP.match(hit[0]):
            RECORD.events.append((event, *hit))


def _hook(event: str, args: tuple) -> None:
    if not RECORD.on:
        return
    try:
        _record(event, args)
    except Exception:  # noqa: BLE001 - an audit hook must never break the code it watches
        RECORD.events.append(("hook-error", f"{event} {args!r}", True))


sys.addaudithook(_hook)


class Screen:
    """A pty standing in for the terminal: isatty holds, and a thread drains what is drawn."""

    def __init__(self) -> None:
        self.master, slave = os.openpty()
        self.out = os.fdopen(slave, "w", buffering=1, encoding="utf-8", errors="replace")
        self.stdin = os.fdopen(os.dup(slave), "r")
        self.seen: list[bytes] = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            self.seen.append(data)

    def text(self) -> str:
        return b"".join(self.seen).decode("utf-8", "replace")

    def close(self) -> None:
        for f in (self.out, self.stdin):
            with contextlib.suppress(OSError):
                f.close()
        with contextlib.suppress(OSError):
            os.close(self.master)


TMP_NAME = re.compile(r"^\.?(?P<base>.+)\.[0-9a-z_-]+\.tmp$")


def snapshot(root: Path, skip: tuple[str, ...] = ()) -> dict[str, str]:
    found: dict[str, str] = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if Path(dirpath, d).relative_to(root).as_posix() not in skip]
        for d in dirs:
            found[Path(dirpath, d).relative_to(root).as_posix() + "/"] = "dir"
        for name in files:
            p = Path(dirpath, name)
            found[p.relative_to(root).as_posix()] = (
                "link" if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest())
    return found


def project_rows(cfg: Path) -> dict:
    """The path-keyed rows of the trust and declined-offer stores: what names a project, apart from the machine's own."""
    rows = {}
    for name in ("trusted_workspaces.yaml", "declined_workspace_offers.yaml"):
        data = yaml.safe_load((cfg / name).read_text(encoding="utf-8")) if (cfg / name).exists() else None
        rows[name] = {k: v for k, v in (data or {}).items() if str(k).startswith("/")}
    return rows


def changed(before: dict, after: dict) -> set[str]:
    return {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}


def in_kept(rel: str) -> str | None:
    """The KEPT bullet a path under the config folder belongs to; a temp file counts as the file it becomes."""
    base = TMP_NAME.sub(lambda m: str(Path(rel).parent / m.group("base")) if "/" in rel else m.group("base"), rel)
    return next((label for label, globs in KEPT if any(fnmatch.fnmatch(base, g) for g in globs)), None)


@dataclass
class World:
    home: Path
    cfg: Path
    project: Path
    elsewhere: Path
    asked: list = field(default_factory=list)
    answers: list = field(default_factory=list)
    model: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    observed: list = field(default_factory=list)
    actions: list = field(default_factory=list)


def root_agent(cfg: Path, **tools) -> None:
    (cfg / "agents").mkdir(exist_ok=True)
    (cfg / "agents" / "orchestrator.yaml").write_text(yaml.dump(
        {"name": "orchestrator", "role": "General-purpose assistant", "model": "inherit", **({"tools": tools} if tools else {})}),
        encoding="utf-8")


@pytest.fixture
def world(tmp_path, monkeypatch, fake_home):
    home = fake_home(tmp_path / "home")
    cfg = home / ".localharness"
    cfg.mkdir()
    project, elsewhere = home / "proj", home / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    (project / "CLAUDE.md").write_text("# House rules\nuse tabs\n", encoding="utf-8")
    (project / "README.md").write_text("hello\n", encoding="utf-8")
    monkeypatch.chdir(project)
    _stub_start_boundaries(cfg, monkeypatch)
    root_agent(cfg)
    _offline_provider(cfg)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    _no_session_start_pass(monkeypatch)
    monkeypatch.setattr("localharness.memory.resonance.ResonanceEngine.embed_docs",
                        lambda self, texts: np.ones((len(texts), 8), dtype=np.float32))
    monkeypatch.setattr("localharness.memory.resonance.ResonanceEngine.embed_query",
                        lambda self, text: np.ones(8, dtype=np.float32))
    w = World(home, cfg, project, elsewhere)

    async def model(self, messages, tools=None, on_token=None, **_):
        w.requests.append({"system": dict(messages[0]), "tools": sorted(t.name for t in tools or ())})
        if w.model and "remember" in w.requests[-1]["tools"]:
            return w.model.pop(0), None
        return FakeLLMResponse(content="Done."), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)

    async def ask(self, request):
        w.asked.append(request)
        return Decision(kind=w.answers.pop(0) if w.answers else "allow_once")

    monkeypatch.setattr("localharness.channels.terminal.TerminalChannel.ask_permission", ask)
    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL._use_input_box", lambda self: False)

    async def play(self):
        await self._channel.start()
        self._bus.subscribe(Observation, w.observed.append)
        try:
            await self._establish_workspace_trust()
            for line in w.lines:
                task = await self._dispatch_input(line)
                if task is not None:
                    self._turn_task = task
                    await task
                await self._settle_pending_coauthor()
            for action in w.actions:
                await action(self)
        finally:
            await self._channel.stop()

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", play)
    return w


def call(i: str, tool: str, **arguments) -> FakeLLMResponse:
    return FakeLLMResponse(tool_calls=[{"id": i, "type": "function",
                                        "function": {"name": tool, "arguments": json.dumps(arguments)}}])


async def session(w: World, monkeypatch, *, private: bool, record: bool = True) -> dict:
    """One real session on a pty; returns what changed in the config folder and the project, and the events."""
    from localharness.cli.start_cmd import _start_async

    screen = Screen()
    cfg_before, proj_before, rows_before = snapshot(w.cfg), snapshot(w.project), project_rows(w.cfg)
    RECORD.events.clear()
    try:
        with monkeypatch.context() as mp:
            mp.setattr(sys, "stdin", screen.stdin)
            mp.setattr(sys, "stdout", screen.out)
            mp.setattr(sys, "stderr", screen.out)
            RECORD.on = record
            run = begin(w.cfg, private, run_home(None))
            assert run is not None and run.private is private
            await _start_async(None, False, False, str(w.cfg), run=run)
            end(run)
    finally:
        RECORD.on = False
        screen.close()
    return {
        "run": run,
        "cfg": changed(cfg_before, snapshot(w.cfg)),
        "project": changed(proj_before, snapshot(w.project)) - set(PROJECT_RUN),
        "events": list(RECORD.events),
        "rows": (rows_before, project_rows(w.cfg)),
        "screen": screen.text(),
    }


async def warm_up(w: World, monkeypatch) -> None:
    """A normal session elsewhere: the machine's own upkeep (seeded files, migration, trust) is done before the private run."""
    monkeypatch.chdir(w.elsewhere)
    await session(w, monkeypatch, private=False, record=False)
    monkeypatch.chdir(w.project)
    w.asked.clear()


def leaks(w: World, out: dict) -> list:
    """Every recorded write that is not in the run folder, in a file the agent wrote, or in KEPT."""
    written = {str(w.project / rel.rstrip("/")) for rel in out["project"]}
    run_root, dot, cfg = str(out["run"].path.parent), str(out["run"].path.parent.parent), str(w.cfg)
    bad = []
    for event, path, certain in out["events"]:
        if path in (run_root, dot) or path.startswith(run_root + os.sep):
            continue
        if any(path == x or path.startswith(x + os.sep) for x in written):
            continue
        if not certain and not os.path.lexists(path):
            continue
        if path.startswith(cfg + os.sep) and in_kept(os.path.relpath(path, cfg)):
            continue
        bad.append((event, os.path.relpath(path, w.home)))
    return bad


def landed_in_state(out: dict, tail: str) -> bool:
    return any("/run/" in path and f"/state/{tail}" in path for _event, path, _certain in out["events"])


def assert_nothing_stays(w: World, out: dict) -> None:
    assert not out["run"].path.exists() and not list(out["run"].path.parent.glob(".deleting-*"))
    stray = sorted(rel for rel in out["cfg"] if not in_kept(rel))
    assert stray == []
    assert out["rows"][0] == out["rows"][1]
    assert {rel for rel in snapshot(w.project) if rel.startswith(".localharness")} == set(PROJECT_RUN)
    assert (w.project / PROJECT_RUN[2]).read_text() == "*\n"
    assert not (w.project / "reports").exists()
    assert leaks(w, out) == []


async def test_a_private_run_leaves_only_the_agents_files_and_kept_items(world, monkeypatch):
    await warm_up(world, monkeypatch)
    world.answers = ["allow_once"]
    world.lines = ["write notes then echo"]
    world.model = [call("w1", "write", path=str(world.project / "NOTES.md"), content="n\n"),
                   call("b1", "bash_exec", command="echo x > made.txt"),
                   call("r1", "remember", name="fav", content="the person likes tabs"),
                   FakeLLMResponse(content="Done.")]
    out = await session(world, monkeypatch, private=True)
    assert [r.klass for r in world.asked] == ["workspace-trust"]
    assert out["project"] == {"NOTES.md", "made.txt"}
    assert_nothing_stays(world, out)
    assert PRIVATE_PROMPT_LINE in world.requests[0]["system"]["content"]
    assert landed_in_state(out, "agents/orchestrator/memory.db")


async def test_the_same_turn_in_a_normal_session_does_leave_records(world, monkeypatch):
    await warm_up(world, monkeypatch)
    world.answers = ["allow_once"]
    world.lines = ["write notes then echo"]
    world.model = [call("r1", "remember", name="fav", content="the person likes tabs")]
    out = await session(world, monkeypatch, private=False)
    assert {"trusted_workspaces.yaml", "project_records.yaml"} <= out["cfg"]
    assert out["rows"][0] != out["rows"][1]
    assert [rel for rel in out["cfg"] if rel.startswith("agents/orchestrator/")]
    assert leaks(world, out)


async def test_a_guarded_turn_answered_always_and_never_writes_no_grants(world, monkeypatch):
    await warm_up(world, monkeypatch)
    world.answers = ["reject_once", "allow_always", "reject_always"]
    world.lines = ["do two things", "and again"]
    world.model = [call("b1", "bash_exec", command="touch b.txt"),
                   call("b2", "bash_exec", command="chmod 644 b.txt"),
                   FakeLLMResponse(content="Done."),
                   call("b3", "bash_exec", command="touch c.txt"),
                   call("b4", "bash_exec", command="chmod 644 c.txt")]
    out = await session(world, monkeypatch, private=True)
    assert [r.klass for r in world.asked] == ["workspace-trust", "shell-unfamiliar", "shell-unfamiliar"]
    assert out["project"] == {"b.txt", "c.txt"}
    assert not (world.cfg / "grants.yaml").exists()
    assert_nothing_stays(world, out)


async def test_a_review_off_turn_leaves_nothing(world, monkeypatch):
    await warm_up(world, monkeypatch)
    world.lines = ["/review off", "write a file"]
    world.model = [call("w1", "write", path=str(world.project / "out.md"), content="o\n")]
    out = await session(world, monkeypatch, private=True)
    assert out["project"] == {"out.md"}
    assert_nothing_stays(world, out)


async def test_a_one_line_ask_leaves_nothing(world, monkeypatch):
    await warm_up(world, monkeypatch)
    world.lines = ["in one line, what is in README.md?"]
    world.model = [call("r1", "read", path=str(world.project / "README.md")), FakeLLMResponse(content="hello")]
    out = await session(world, monkeypatch, private=True)
    assert out["project"] == set()
    assert_nothing_stays(world, out)


async def test_a_saved_page_lands_in_the_run_folder_and_goes_with_it(world, monkeypatch):
    from localharness.tools.builtin.web_tool import WebFetchTool
    await warm_up(world, monkeypatch)
    public_web(monkeypatch)
    _serve(monkeypatch, lambda request: _text("a saved page"))
    saved = []

    async def fetch(repl):
        saved.append(await repl._agent._registered(WebFetchTool).run(url="http://example.com/p", save_to="p.md"))

    world.actions = [fetch]
    out = await session(world, monkeypatch, private=True)
    assert saved[0].success, saved[0].error
    assert out["project"] == set()
    assert any("downloads" in path and path.endswith("p.md") and "/run/" in path for _e, path, _c in out["events"])
    assert_nothing_stays(world, out)


async def test_a_task_opened_and_cleared_leaves_nothing(world, monkeypatch):
    await warm_up(world, monkeypatch)
    world.lines = ["/task new count the files", "/task clear"]
    out = await session(world, monkeypatch, private=True)
    assert out["project"] == set()
    assert landed_in_state(out, "agents/orchestrator/tasks/t1")
    assert_nothing_stays(world, out)


async def test_a_private_session_recalls_none_of_the_projects_facts(world, monkeypatch):
    await warm_up(world, monkeypatch)
    await _seed_store(world.cfg)
    db = world.cfg / "agents" / "orchestrator" / "memory.db"
    before = db.read_bytes()
    world.lines = ["what do you know?"]
    normal = await session(world, monkeypatch, private=False)
    assert "user-editor" in world.requests[-1]["system"]["content"]
    db.write_bytes(before)
    world.requests.clear()
    out = await session(world, monkeypatch, private=True)
    assert "user-editor" not in world.requests[0]["system"]["content"]
    assert "Neovim" not in world.requests[0]["system"]["content"]
    assert db.read_bytes() == before
    assert_nothing_stays(world, out)
    assert normal["cfg"]


async def test_a_redirect_spelled_through_a_variable_or_beside_one_is_refused_and_leaves_nothing(world, monkeypatch):
    await warm_up(world, monkeypatch)
    other = world.cfg / "run" / "aaaaaaaaaaaa" / "scratch" / "s.sh"
    commands = ["echo x >> CLAUDE.md 2>$ERR", "cat README.md > CLAUDE.md 2> $LOCALHARNESS_SCRATCH/err",
                "echo x >> $PWD/CLAUDE.md", 'cat >> "$PWD/AGENTS.md" <<EOF\nx\nEOF',
                "echo x > $HOME/.localharness/agents/a.yaml", "echo x > $PWD/.agents/skills/s/SKILL.md",
                f"echo x > {other} 2>$TMPDIR/err", f"tee {other} $LOG"]
    world.lines = ["write the files"]
    world.model = [call(f"b{i}", "bash_exec", command=c) for i, c in enumerate(commands)] + [
        FakeLLMResponse(content="Done.")]
    out = await session(world, monkeypatch, private=True)
    assert [r.klass for r in world.asked] == ["workspace-trust"]
    refused = [o.output or "" for o in world.observed if o.tool_name == "bash_exec"]
    assert refused == ["[DENIED]"] * len(commands)
    assert out["project"] == set() and not other.exists()
    assert_nothing_stays(world, out)
