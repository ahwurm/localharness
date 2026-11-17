"""The gate around a session's run folder: other sessions' folders are denied, scratch is judged as an outside write and held for the session, a move into scratch asks."""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import Decision, GateSettings, Grant, ToolMeta, Verdict
from localharness.agent.shell_classify import copy_operands, moved_into, removal_operands
from localharness.agent.verdict import SCRATCH_OPTIONS_LEGEND, GateContext, evaluate
from localharness.config.grants import GrantStore
from localharness.core.run_folder import begin, end

SETTINGS = GateSettings()
WRITE = ToolMeta(destructive=True, group="fs.write")
SHELL = ToolMeta(destructive=True, group="shell")
MODES = ("guarded", "trusted", "auto", "unattended")
OTHER = "aaaaaaaaaaaa"
MOVES_INTO = "into this session's scratch, which is deleted when the session ends"


def no_grants(workspace: Path, klass: str, key: str) -> Grant | None:
    return None


class World:
    def __init__(self, home: Path):
        self.home = home
        self.cfg = home / ".localharness"
        self.ws = home / "proj"
        self.ws.mkdir()
        (self.cfg / "agents").mkdir(parents=True)
        (self.cfg / "config.yaml").write_text("{}\n")
        self.run = begin(self.cfg)
        assert self.run is not None and self.run.path.parent == self.cfg.resolve() / "run"
        self.scratch = self.run.scratch
        self.other = self.cfg.resolve() / "run" / OTHER / "scratch"
        self.other.mkdir(parents=True)
        (self.other / "secret").write_text("s")

    def ctx(self, **kw) -> GateContext:
        kw.setdefault("mode", "guarded")
        kw.setdefault("config_dir", self.cfg)
        kw.setdefault("run", self.run.path)
        kw.setdefault("scratch", self.scratch)
        return GateContext(boundary=self.ws, workspace=self.ws, grants=no_grants, has_review_surface=True, **kw)

    def write(self, path: Path | str, **kw):
        return evaluate("write", {"path": str(path), "content": "x"}, WRITE, self.ctx(**kw), SETTINGS)

    def shell(self, command: str, working_dir: Path | None = None, **kw):
        params = {"command": command, **({"working_dir": str(working_dir)} if working_dir else {})}
        return evaluate("bash_exec", params, SHELL, self.ctx(**kw), SETTINGS)


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    w = World(home)
    yield w
    end(w.run)


# ------------------------------------------------------------------------------- the deny row

@pytest.mark.parametrize("mode", MODES)
def test_another_sessions_scratch_is_denied_to_a_write_and_a_shell_write_in_every_mode(world, mode):
    for result in (world.write(world.other / "x", mode=mode), world.shell(f"echo x > {world.other}/x", mode=mode)):
        assert result.verdict is Verdict.DENY
        assert str(world.scratch) in result.reason


@pytest.mark.parametrize("mode", MODES)
def test_removing_run_or_writing_its_gitignore_is_denied_in_every_mode(world, mode):
    run_dir = world.cfg.resolve() / "run"
    for result in (world.shell(f"rm -rf {run_dir}", mode=mode), world.write(run_dir / ".gitignore", mode=mode),
                   world.shell(f"rm -rf {world.other.parent}", mode=mode)):
        assert result.verdict is Verdict.DENY


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("command", [
    f"ln -s ../../{OTHER}/scratch/secret link",
    f"ln -s ../../../{OTHER}/scratch/secret t3/link",
    f"ln -s ../../../{OTHER}/scratch/secret -t t3",
])
def test_a_relative_link_source_is_read_from_the_folder_the_link_sits_in(world, mode, command):
    (world.scratch / "t3").mkdir()
    assert world.shell(command, working_dir=world.scratch, mode=mode).verdict is Verdict.DENY


def test_a_link_to_a_file_in_the_project_is_not_denied(world):
    (world.scratch / "t3").mkdir()
    (world.ws / "a.txt").write_text("a")
    assert world.shell("ln -s ../../../../proj/a.txt t3/link", working_dir=world.scratch).verdict is not Verdict.DENY


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("command", [
    "echo x > {o}/s.sh 2>$TMPDIR/err", "cp a {o}/s.sh 2>$ERR", "tee {o}/s.sh $LOG", "echo x >> {o}/s.sh 2> $LOG",
])
def test_a_second_variable_spelled_target_does_not_lift_the_deny(world, mode, command):
    assert world.shell(command.format(o=world.other), mode=mode).verdict is Verdict.DENY


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("command", [
    "cp -l {o}/secret {s}/h", "cp -s {o}/secret {s}/h", "cp -a {o}/secret {s}/h", "cp --link {o}/secret {s}/h",
    "rsync --link-dest={o} a/ {s}/b", "rsync -a --link-dest {o} a/ {s}/b",
])
def test_a_linking_copy_from_another_sessions_folder_is_denied(world, mode, command):
    assert world.shell(command.format(o=world.other, s=world.scratch), mode=mode).verdict is Verdict.DENY


def test_a_plain_copy_out_of_another_sessions_folder_is_not_denied(world):
    assert world.shell(f"cp {world.other}/secret {world.scratch}/h").verdict is not Verdict.DENY


@pytest.mark.parametrize("mode", MODES)
def test_a_home_spelling_of_the_run_folder_is_refused_by_text(world, mode):
    for command in (f'echo x > "$HOME/.localharness/run/{OTHER}/scratch/y"',
                    f'rm -rf "$HOME/.LocalHarness/run/{OTHER}"',
                    f'cp a "${{HOME}}/.localharness/run/{OTHER}/scratch/y"'):
        assert world.shell(command, mode=mode).verdict is Verdict.DENY
    assert world.write(f"$HOME/.localharness/run/{OTHER}/scratch/y", mode=mode).verdict is Verdict.DENY


def test_a_read_of_a_home_spelling_is_not_refused(world):
    assert world.shell(f'cat "$HOME/.localharness/run/{OTHER}/scratch/secret"').verdict is not Verdict.DENY


def test_the_default_config_folder_is_guarded_when_the_session_has_another(world, tmp_path):
    custom = tmp_path / "elsewhere"
    custom.mkdir()
    default_run = world.cfg.resolve() / "run" / OTHER / "scratch" / "x"
    for path in (default_run, Path(str(default_run).replace("/run/", "/RUN/"))):
        assert world.write(path, config_dir=custom).verdict is Verdict.DENY


def test_a_session_without_scratch_is_told_so(world):
    result = world.write(world.other / "x", scratch=None)
    assert result.verdict is Verdict.DENY and "has no scratch" in result.reason


def test_writes_inside_this_sessions_scratch_are_not_denied(world):
    for path in (world.scratch / "a", world.scratch / "t3" / "b"):
        assert world.write(path).verdict is not Verdict.DENY
    assert world.shell(f"echo x > {world.scratch}/t3/b").verdict is not Verdict.DENY
    assert world.shell(f"rm -rf {world.scratch}/t3").verdict is not Verdict.DENY


# ------------------------------------------------------------------- scratch judged as an outside write

def test_a_scratch_write_is_silent_in_auto_and_trusted(world):
    for mode in ("auto", "trusted"):
        assert world.write(world.scratch / "a", mode=mode).verdict is Verdict.ALLOW
        assert world.shell(f"echo x > {world.scratch}/a", mode=mode).verdict is Verdict.ALLOW


@pytest.mark.parametrize("via", ["write", "shell"])
def test_a_scratch_write_asks_in_guarded_keyed_on_scratch_with_the_session_legend(world, via):
    target = world.scratch / "t3" / "a"
    result = world.write(target) if via == "write" else world.shell(f"echo x > {target}")
    assert result.verdict is Verdict.ASK
    request = result.request
    assert request.grantable and request.grant_keys == (("edit-outside", str(world.scratch)),)
    assert "this session's scratch (deleted when it ends)" in request.reason
    assert request.options_legend == SCRATCH_OPTIONS_LEGEND == (
        "[y]es once   [a]lways (this session)   [n]o   [N]ever (this session)")


@pytest.mark.parametrize("mode", ["guarded", "auto", "trusted"])
def test_a_scratch_write_is_a_severe_ask_once_saved_pages_are_read(world, mode):
    for result in (world.write(world.scratch / "a", mode=mode, tainted_by="a.md"),
                   world.shell(f"echo x > {world.scratch}/a", mode=mode, tainted_by="a.md")):
        assert result.verdict is Verdict.ASK
        assert "a.md" in result.request.reason


@pytest.mark.parametrize("target", ["$LOCALHARNESS_SCRATCH/out", "${LOCALHARNESS_SCRATCH}/t3/out", '"$TMPDIR"/o', "$TMPDIR/o"])
def test_a_scratch_write_spelled_by_its_variable_is_keyed_on_scratch(world, target):
    request = world.shell(f"echo x > {target}").request
    assert request.grant_keys == (("edit-outside", str(world.scratch)),)
    assert request.options_legend == SCRATCH_OPTIONS_LEGEND


@pytest.mark.parametrize("command", [
    "LOCALHARNESS_SCRATCH=$HOME/.ssh; echo k >> $LOCALHARNESS_SCRATCH/out", "export TMPDIR=~; echo k >> $TMPDIR/o",
    "n=TMP; declare ${n}DIR=~; echo k >> $TMPDIR/o", 'n=TMP; printf -v "${n}DIR" ~; echo k >> $TMPDIR/o',
    "eval \"$X\"; echo k >> $TMPDIR/o", "echo k >> '$TMPDIR'/o", "echo k >> $TMPDIR/../../../../o",
    "echo k >> ${TMPDIR:=~}/o",
])
def test_a_rebound_quoted_or_escaping_spelling_is_not_keyed_on_scratch(world, command):
    request = world.shell(command).request
    assert request is not None and ("edit-outside", str(world.scratch)) not in request.grant_keys


def test_an_outside_write_elsewhere_carries_no_session_legend(world, tmp_path):
    result = world.write(tmp_path / "elsewhere" / "a")
    assert result.verdict is Verdict.ASK and result.request.options_legend is None


# ----------------------------------------------------------------- one answer, held for the session

class Asker:
    def __init__(self, kind: str):
        self.kind, self.calls = kind, []

    async def __call__(self, request):
        self.calls.append(request)
        return Decision(kind=self.kind)


def gate_for(world, asker, tmp_path) -> PermissionGate:
    return PermissionGate(
        boundary=world.ws, workspace=world.ws, grants=GrantStore(tmp_path / "grants.yaml"), mode="guarded",
        asker=asker, channel_name="test", has_review_surface=True, config_dir=world.cfg,
        run=world.run, scratch=world.scratch)


async def check(gate, path: Path):
    return await gate.check("write", {"path": str(path), "content": "x"}, WRITE, agent_id="a", session_id="s")


@pytest.mark.parametrize("seeded", [False, True])
async def test_one_always_covers_scratch_and_every_task_folder_across_a_rebuilt_gate_and_writes_nothing(
        world, tmp_path, seeded):
    grants = tmp_path / "grants.yaml"
    if seeded:
        grants.write_bytes(b"version: 1\ngrants: []\n")
    before = grants.read_bytes() if grants.exists() else None
    asker = Asker("allow_always")
    gate = gate_for(world, asker, tmp_path)
    assert (await check(gate, world.scratch / "a")).allowed
    assert len(asker.calls) == 1 and asker.calls[0].options_legend == SCRATCH_OPTIONS_LEGEND
    assert (await check(gate, world.scratch / "t3" / "b")).allowed
    assert (await check(gate_for(world, asker, tmp_path), world.scratch / "t4" / "c")).allowed
    assert len(asker.calls) == 1
    assert (grants.read_bytes() if grants.exists() else None) == before


async def test_an_always_for_a_variable_spelled_scratch_write_is_held_and_a_rebinding_asks_again(world, tmp_path):
    asker = Asker("allow_always")
    gate = gate_for(world, asker, tmp_path)

    async def shell(command: str):
        return await gate.check("bash_exec", {"command": command}, SHELL, agent_id="a", session_id="s")

    assert (await shell("echo x > $LOCALHARNESS_SCRATCH/out")).allowed
    assert (await shell("echo x > $TMPDIR/o")).allowed and (await check(gate, world.scratch / "a")).allowed
    assert len(asker.calls) == 1 and not (tmp_path / "grants.yaml").exists()
    assert (await shell("LOCALHARNESS_SCRATCH=$HOME/.ssh; echo k >> $LOCALHARNESS_SCRATCH/out")).allowed
    assert len(asker.calls) == 2


async def test_a_never_is_held_the_same_way_and_never_written(world, tmp_path):
    asker = Asker("reject_always")
    gate = gate_for(world, asker, tmp_path)
    assert not (await check(gate, world.scratch / "a")).allowed
    assert not (await check(gate, world.scratch / "t3" / "b")).allowed
    assert not (await check(gate_for(world, asker, tmp_path), world.scratch / "c")).allowed
    assert len(asker.calls) == 1
    assert not (tmp_path / "grants.yaml").exists()


async def test_another_session_does_not_inherit_the_answer(world, tmp_path):
    first = Asker("allow_always")
    assert (await check(gate_for(world, first, tmp_path), world.scratch / "a")).allowed
    world.run.held.clear()
    second = Asker("allow_once")
    assert (await check(gate_for(world, second, tmp_path), world.scratch / "a")).allowed
    assert len(second.calls) == 1


# ------------------------------------------------------------------------- a move into the run asks

MOVES = [
    ("mv ~/x {s}/y", "~/x"),
    ('mv ~/x "$LOCALHARNESS_SCRATCH"/y', "~/x"),
    ("mv --target={s} ~/x", "~/x"),
    ("mv --target-dir={s} ~/x", "~/x"),
    ("mv --targ={s} ~/x", "~/x"),
    ("mv -t {s} ~/x", "~/x"),
    ("rsync --remove-sent-files ~/x {s}/", "~/x"),
    ("rsync --remove-source-files ~/x {s}/", "~/x"),
    ('cd "$TMPDIR" && mv ~/x .', "~/x"),
    ("cd $LOCALHARNESS_SCRATCH; mv ~/x .", "~/x"),
    ("find ~ -name a -exec mv {{}} {s} \\;", "{}"),
    ("tar --remove-files -cf {s}/a.tar ~/x", "~/x"),
    ("tar --remove-files --file={s}/a.tar ~/x", "~/x"),
    ("zip -m {s}/a.zip ~/x", "~/x"),
    ("mv ~/Documents {s}/d 2>$TMPDIR/err", "~/Documents"),
    ("mv ~/x {s}/y 2>$ERR", "~/x"),
]


@pytest.mark.parametrize("mode", ["auto", "guarded"])
@pytest.mark.parametrize("command,source", MOVES)
def test_a_move_into_scratch_asks_ungrantably_and_names_the_source(world, mode, command, source):
    result = world.shell(command.format(s=world.scratch), mode=mode)
    assert result.verdict is Verdict.ASK
    assert not result.request.grantable
    assert f"moves {source} {MOVES_INTO}" in result.request.reason


@pytest.mark.parametrize("mode", ["auto", "guarded"])
def test_a_move_into_a_fallback_run_asks_too(world, tmp_path, mode):
    fallback = tmp_path / "localharness-run-abc"
    result = world.shell(f"mv ~/x {fallback}/scratch/y", mode=mode, run=fallback, scratch=fallback / "scratch")
    assert result.verdict is Verdict.ASK and MOVES_INTO in result.request.reason


@pytest.mark.parametrize("mode", ["auto", "guarded"])
@pytest.mark.parametrize("command", ["mv {s}/a {s}/b", "cp ~/x {s}/y", "mv ~/x ~/y", "cd /tmp && mv ~/x ~/y"])
def test_a_move_inside_scratch_a_copy_in_or_a_move_elsewhere_does_not_raise_the_move_ask(world, mode, command):
    result = world.shell(command.format(s=world.scratch), mode=mode)
    assert MOVES_INTO not in (result.request.reason if result.request else "")


# -------------------------------------------------------------------------------- the removal facts

def test_the_removal_facts_know_the_extra_spellings():
    assert removal_operands("rsync", ("rsync", "--remove-sent-files", "A", "B")) == ["A"]
    assert removal_operands("rsync", ("rsync", "--remove-source-files", "A", "B")) == ["A"]
    assert removal_operands("tar", ("tar", "--remove-files", "-cf", "a.tar", "x", "y")) == ["x", "y"]
    assert removal_operands("zip", ("zip", "-m", "a.zip", "x")) == ["x"]
    assert removal_operands("zip", ("zip", "a.zip", "x")) == []
    assert removal_operands("tar", ("tar", "-cf", "a.tar", "x")) == []


@pytest.mark.parametrize("flag", ["--target", "--targ", "--t", "--target-dir", "--target-directory"])
def test_every_unambiguous_abbreviation_of_target_directory_names_the_destination(flag):
    ops = copy_operands([f"{flag}=D", "s"])
    assert (ops.sources, ops.destination, ops.into) == (("s",), "D", True)


def test_moved_into_reads_the_four_commands():
    assert moved_into(("mv", "a", "b", "d")) == (("a", "b"), "d")
    assert moved_into(("rsync", "--remove-source-files", "a", "d")) == (("a",), "d")
    assert moved_into(("tar", "--remove-files", "-czf", "t.tgz", "a")) == (("a",), "t.tgz")
    assert moved_into(("zip", "-mr", "z.zip", "a")) == (("a",), "z.zip")
    assert moved_into(("rsync", "a", "d")) is None and moved_into(("cp", "a", "d")) is None
