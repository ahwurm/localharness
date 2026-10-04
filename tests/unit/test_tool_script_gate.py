"""A tool script the agent wrote runs as an unconfirmed shell command until you confirm it at start.

Self-extension stays (owner ruling, Option 1): the agent may drop helper scripts into the machine's
tools folder (`~/.localharness/tools/`) and run them. What changes is what such a script is trusted
with. Until you confirm its current content at a `localharness start` on a terminal (the machine
record, config/trust.machine_snapshot), a shell command that runs it gets exactly what any shell
command gets in the current mode (orchestrator ruling R21): `guarded` asks, keyed on the script's
path and content, so no earlier "always" on `python3` or `node` covers it and an "always" given to
it covers that exact content only; `trusted`, `auto` and `unattended` run it as they run any
command. An "always" for a script counts only where the script IS what runs (R23).

The global dir is conftest's hermetic LOCALHARNESS_HOME and `~` is its parent, so
`~/.localharness/tools/` is the same folder. Start's question is `rich.prompt.Confirm.ask`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from importlib import resources
from pathlib import Path

import pytest

from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import (
    Decision,
    GateSettings,
    Grant,
    ShellSegment,
    ToolMeta,
    Verdict,
)
from localharness.agent.shell_classify import classify_shell
from localharness.agent.verdict import GateContext, evaluate
from localharness.cli import workspace as ws_mod
from localharness.cli.workspace import MachineTrust, decide_machine_trust
from localharness.config import trust
from localharness.config.grants import GrantStore

SETTINGS = GateSettings()
SHELL = ToolMeta(group="shell", destructive=True)
CLASS = "tool-script-unconfirmed"
DIGEST = "sha256:abc123abc123"
KINDS_BEFORE_SCRIPTS = frozenset({"mcp_server", "embedding_model", "permission"})
DEPENDENCY_DIRS = {"node_modules", ".venv", "venv", "__pycache__", ".git", "site-packages"}
NL = bytes([10])


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    ws_mod._DECLINED.clear()
    yield
    ws_mod._DECLINED.clear()


@pytest.fixture
def g(fake_home) -> Path:
    """The machine's config dir, with `~` pointed at its parent."""
    home = Path(os.environ["LOCALHARNESS_HOME"])
    fake_home(home.parent, clear_overrides=False)
    return home


@pytest.fixture
def proj(tmp_path) -> Path:
    root = (tmp_path / "proj").resolve()
    root.mkdir()
    return root


def _script(g: Path, rel: str, body: bytes = b"print(1)\n") -> Path:
    path = g / "tools" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _short(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()[:12]


def _packaged() -> bytes:
    return resources.files("localharness").joinpath("assets", "design-screenshot.js").read_bytes()


def _never_asked(monkeypatch) -> None:
    def _boom(*_a, **_kw):
        raise AssertionError("asked where the rule forbids asking")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _answers(monkeypatch, *answers: bool) -> list[str]:
    asked: list[str] = []
    queue = list(answers)

    def _ask(prompt, *_a, **_kw):
        asked.append(str(prompt))
        return queue.pop(0)

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def _adopt(g: Path, monkeypatch) -> None:
    """A start with no machine record: what is on disk becomes the record, nobody is asked."""
    _never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=True) == MachineTrust()


def _granting(*pairs: tuple[str, str]):
    def _lookup(workspace: Path, klass: str, key: str):
        if (klass, key) not in pairs:
            return None
        return Grant(key=key, klass=klass, granted_at="2026-10-03T00:00:00+00:00",
                     channel="terminal", session_id="s1", workspace=str(workspace))
    return _lookup


def _no_grants(workspace: Path, klass: str, key: str):
    return None


def _base(proj: Path, mode: str = "guarded", grants=_no_grants) -> GateContext:
    return GateContext(boundary=proj, workspace=proj, grants=grants, mode=mode,  # type: ignore[arg-type]
                       has_review_surface=True)


def _ctx(proj: Path, *, mode: str = "guarded", grants=_no_grants, pending=lambda p: DIGEST):
    return dataclasses.replace(_base(proj, mode, grants), script_pending=pending)


def _shell(command: str, ctx: GateContext):
    return evaluate("bash_exec", {"command": command}, SHELL, ctx, SETTINGS)


def _asked(result) -> tuple:
    return (result.verdict, result.request.klass, result.request.key) if result.request else (
        result.verdict,)


# --------------------------------------------------------------------------- the snapshot


def test_each_script_under_the_tools_folder_is_an_entry_with_its_hash(g):
    _script(g, "a.py", b"print(1)\n")
    _script(g, "sub/b.sh", b"echo b\n")

    assert trust.tool_script_entries(g) == [
        {"kind": "tool_script", "file": "tools/a.py", "name": "a.py",
         "shown": _short(b"print(1)\n"), "sha256": hashlib.sha256(b"print(1)\n").hexdigest()},
        {"kind": "tool_script", "file": "tools/sub/b.sh", "name": "sub/b.sh",
         "shown": _short(b"echo b\n"), "sha256": hashlib.sha256(b"echo b\n").hexdigest()},
    ]


def test_the_packaged_helper_is_never_listed_but_a_modified_copy_is(g):
    helper = _script(g, "design-screenshot.js", _packaged())
    assert trust.tool_script_entries(g) == []

    helper.write_bytes(_packaged() + b"\n// changed by the agent\n")
    assert [e["name"] for e in trust.tool_script_entries(g)] == ["design-screenshot.js"]


def test_a_symlink_in_the_tools_folder_is_listed_by_its_own_name_and_what_it_points_to(g, tmp_path):
    """A file is judged by its entry, not by where it links: skipping a link that leads outside let
    it run past the tool-script rule as an ordinary command (the re-review's symlink finding)."""
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"print('outside')\n")
    _script(g, "a.py")
    (g / "tools" / "link.py").symlink_to(outside)

    entries = {e["name"]: e["shown"] for e in trust.tool_script_entries(g)}
    assert entries == {"a.py": _short(b"print(1)\n"), "link.py": _short(b"print('outside')\n")}


def test_a_linked_tool_script_waits_for_its_yes_like_any_other(g, tmp_path, monkeypatch):
    _script(g, "a.py")
    _adopt(g, monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"print('outside')\n")
    link = g / "tools" / "link.py"
    link.symlink_to(outside)

    assert trust.tool_script_pending(link) == _short(b"print('outside')\n")
    _answers(monkeypatch, True)
    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert trust.tool_script_pending(link) is None, "confirmed at start"
    outside.write_bytes(b"print('changed elsewhere')\n")
    assert trust.tool_script_pending(link) == _short(b"print('changed elsewhere')\n")


def test_a_virtual_environment_under_any_name_is_never_listed_or_gated(g, tmp_path):
    """A venv not called `.venv` or `venv` (`python3 -m venv tools/pyenv`): its interpreter links
    and site-packages were listed as tool scripts. A folder holding pyvenv.cfg is skipped."""
    _script(g, "a.py")
    env = g / "tools" / "pyenv"
    (env / "bin").mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    real_python = tmp_path / "python3.12"
    real_python.write_bytes(b"#!/bin/sh\n")
    (env / "bin" / "python").symlink_to(real_python)
    _script(g, "pyenv/lib/python3.12/site-packages/mod.py")

    assert [e["name"] for e in trust.tool_script_entries(g)] == ["a.py"]
    assert trust.tool_script_pending(env / "bin" / "python") is None
    assert trust.tool_script_pending(env / "lib" / "python3.12" / "site-packages" / "mod.py") is None
    assert trust.tool_script_pending(g / "tools" / "a.py") is not None, "the rule still holds elsewhere"


def test_a_file_reached_through_a_symlinked_folder_is_judged_as_what_it_points_to(g, tmp_path):
    """Start never walks a symlinked folder (it could lead anywhere, or loop), so a file reached
    through one is never listed — and is judged where it resolves, as running it from there would be."""
    lib = tmp_path / "lib"
    (lib / "x.py").parent.mkdir()
    (lib / "x.py").write_bytes(b"print(2)\n")
    (g / "tools").mkdir(exist_ok=True)
    (g / "tools" / "lib").symlink_to(lib)

    assert trust.tool_script_entries(g) == []
    assert trust.tool_script_pending(g / "tools" / "lib" / "x.py") is None


@pytest.mark.parametrize("rel", [
    "node_modules/playwright/index.js", "x/.venv/bin/python", "venv/bin/activate",
    "__pycache__/a.cpython-312.pyc", ".git/config", "lib/python3.12/site-packages/m.py",
])
def test_dependency_and_cache_folders_are_never_walked(g, monkeypatch, rel):
    """`npm install playwright` (which the packaged helper asks for) puts ~1,500 files under
    tools/node_modules: a dependency tree is not a script and must never become a start question."""
    _script(g, "a.py")
    _script(g, rel, b"module.exports = 1\n")
    visited: list[str] = []
    real_walk = os.walk

    def _walk(top, *a, **kw):
        for dirpath, dirnames, filenames in real_walk(top, *a, **kw):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(os, "walk", _walk)

    assert [e["name"] for e in trust.tool_script_entries(g)] == ["a.py"]
    assert visited, "the tools folder was walked"
    assert not any(set(Path(d).parts) & DEPENDENCY_DIRS for d in visited)


def test_the_machine_snapshot_carries_the_scripts_as_a_known_kind(g):
    _script(g, "a.py")

    assert "tool_script" in trust.MACHINE_KINDS
    assert [e for e in trust.machine_snapshot(g) if e["kind"] == "tool_script"] == \
        trust.tool_script_entries(g)


# --------------------------------------------------------------------------- the start decision


def test_existing_scripts_are_adopted_silently(g, monkeypatch):
    """The first start with this release: the record predates tool scripts as a kind."""
    trust.record_machine(g, trust.machine_snapshot(g), kinds=KINDS_BEFORE_SCRIPTS)
    _script(g, "a.py", b"print(1)\n")
    _script(g, "sub/b.sh", b"echo b\n")
    _never_asked(monkeypatch)

    assert decide_machine_trust(g, ask=True) == MachineTrust()
    rec = trust.recorded_machine(g)
    assert "tool_script" in rec["kinds"]
    assert {e.get("sha256") for e in rec["entries"] if e["kind"] == "tool_script"} == {
        hashlib.sha256(b"print(1)\n").hexdigest(), hashlib.sha256(b"echo b\n").hexdigest()}


def test_a_script_added_afterwards_is_listed_once_and_a_yes_records_it(g, monkeypatch):
    _script(g, "a.py")
    _adopt(g, monkeypatch)
    _script(g, "a2.py", b"print(2)\n")
    asked = _answers(monkeypatch, True)

    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert len(asked) == 1
    assert f"+ a2.py (tools/a2.py): {_short(b'print(2)' + NL)}" in asked[0]
    assert "(tools/a.py)" not in asked[0], "one line per change, nothing already confirmed"
    _never_asked(monkeypatch)
    assert decide_machine_trust(g, ask=True) == MachineTrust()


def test_with_nobody_to_ask_one_line_names_the_script(g, monkeypatch):
    _script(g, "a.py")
    _adopt(g, monkeypatch)
    _script(g, "a2.py", b"print(2)\n")

    got = decide_machine_trust(g, ask=False)

    assert got.withheld == {"tools/a2.py": frozenset({("tool_script", "a2.py")})}
    assert "a2.py (tools/a2.py)" in got.line and "\n" not in got.line


def test_a_changed_script_reads_as_changed(g, monkeypatch):
    path = _script(g, "a.py")
    _adopt(g, monkeypatch)
    body = b"print('changed')\n"
    path.write_bytes(body)
    asked = _answers(monkeypatch, False)

    got = decide_machine_trust(g, ask=True)

    assert f"~ a.py (tools/a.py): {_short(body)}" in asked[0]
    assert got.withheld == {"tools/a.py": frozenset({("tool_script", "a.py")})}


# --------------------------------------------------------------------------- what the gate reads


def test_a_script_is_pending_from_its_change_until_a_yes_at_start(g, monkeypatch):
    path = _script(g, "a.py")
    _adopt(g, monkeypatch)
    assert trust.tool_script_pending(path) is None

    path.write_bytes(b"print('a longer body')\n")
    assert trust.tool_script_pending(path) == _short(b"print('a longer body')\n")

    _answers(monkeypatch, True)
    assert decide_machine_trust(g, ask=True) == MachineTrust()
    assert trust.tool_script_pending(path) is None


def test_pending_never_names_a_file_the_rule_does_not_cover(g, monkeypatch, tmp_path):
    _adopt(g, monkeypatch)
    outside = tmp_path / "x.py"
    outside.write_text("print(1)\n", encoding="utf-8")

    assert trust.tool_script_pending(outside) is None
    assert trust.tool_script_pending(_script(g, "design-screenshot.js", _packaged())) is None
    assert trust.tool_script_pending(_script(g, "node_modules/x.js", b"1\n")) is None
    assert trust.tool_script_pending(_script(g, "new.py")) == _short(b"print(1)\n")


def test_a_record_from_before_scripts_were_a_kind_reads_every_script_as_confirmed(g):
    """What start's decision counts as adopted the gate counts as adopted: until a start records
    the kind, every script there predates the rule."""
    trust.record_machine(g, [], kinds=KINDS_BEFORE_SCRIPTS)

    assert trust.tool_script_pending(_script(g, "a.py")) is None


def test_with_no_machine_record_every_tools_file_is_pending(g):
    path = _script(g, "a.py")

    assert trust.recorded_machine(g) is None
    assert trust.tool_script_pending(path) == _short(b"print(1)\n")


# --------------------------------------------------------------------------- the verdict


def test_guarded_asks_about_an_unconfirmed_script_by_path_and_content(g, proj):
    script = _script(g, "a.py").resolve()

    result = _shell(f"python3 {script}", _ctx(proj))

    assert result.verdict is Verdict.ASK
    request = result.request
    assert (request.klass, request.key, request.grantable) == (CLASS, f"{script}@{DIGEST}", True)
    assert str(script) in request.reason and "localharness start" in request.reason


def test_an_always_on_the_interpreter_never_covers_the_script(g, proj):
    script = _script(g, "a.py").resolve()
    ctx = _ctx(proj, grants=_granting(("shell-unfamiliar", "python3 <script>")))

    assert _asked(_shell(f"python3 {script}", ctx)) == (Verdict.ASK, CLASS, f"{script}@{DIGEST}")


def test_an_always_on_the_script_covers_that_exact_content_only(g, proj):
    script = _script(g, "a.py").resolve()
    grants = _granting((CLASS, f"{script}@{DIGEST}"))

    assert _shell(f"python3 {script}", _ctx(proj, grants=grants)).verdict is Verdict.ALLOW
    changed = _ctx(proj, grants=grants, pending=lambda p: "sha256:fedcba987654")
    assert _asked(_shell(f"python3 {script}", changed)) == (
        Verdict.ASK, CLASS, f"{script}@sha256:fedcba987654")


def test_direct_execution_and_a_tilde_path_are_recognised(g, proj):
    a_sh = _script(g, "a.sh", b"#!/bin/sh\necho a\n").resolve()
    b_js = _script(g, "b.js", b"console.log(1)\n").resolve()

    assert _asked(_shell(f"{a_sh} --x", _ctx(proj))) == (Verdict.ASK, CLASS, f"{a_sh}@{DIGEST}")
    assert _asked(_shell("node ~/.localharness/tools/b.js", _ctx(proj))) == (
        Verdict.ASK, CLASS, f"{b_js}@{DIGEST}")
    assert _asked(_shell("timeout 5 ~/.localharness/tools/a.sh", _ctx(proj))) == (
        Verdict.ASK, CLASS, f"{a_sh}@{DIGEST}")


def test_a_read_and_a_script_elsewhere_are_not_tool_scripts(g, proj, tmp_path):
    script = _script(g, "a.py").resolve()
    elsewhere = tmp_path / "elsewhere.py"
    elsewhere.write_text("print(1)\n", encoding="utf-8")

    assert _shell(f"cat {script}", _ctx(proj)).verdict is Verdict.ALLOW
    assert _asked(_shell(f"python3 {elsewhere}", _ctx(proj))) == (
        Verdict.ASK, "shell-unfamiliar", "python3 <script>")


@pytest.mark.parametrize("mode,expected", [
    ("auto", Verdict.ALLOW), ("unattended", Verdict.ALLOW), ("trusted", Verdict.ALLOW),
    ("guarded", Verdict.ASK),
])
def test_auto_runs_an_unconfirmed_script_without_asking(g, proj, tmp_path, mode, expected):
    """R21: what the gate does for any shell command in this mode, no more."""
    script = _script(g, "a.py").resolve()
    elsewhere = tmp_path / "elsewhere.py"
    elsewhere.write_text("print(1)\n", encoding="utf-8")

    assert _shell(f"python3 {script}", _ctx(proj, mode=mode)).verdict is expected
    assert _shell(f"python3 {elsewhere}", _ctx(proj, mode=mode)).verdict is expected


def test_the_classifier_keeps_the_program_as_written():
    direct = classify_shell("~/.localharness/tools/a.sh --x", SETTINGS).segments[0]
    assert (direct.argv[0], direct.program) == ("a.sh", "~/.localharness/tools/a.sh")
    wrapped = classify_shell("timeout 5 ~/.localharness/tools/a.sh", SETTINGS).segments[0]
    assert (wrapped.argv[0], wrapped.program) == ("a.sh", "~/.localharness/tools/a.sh")
    usr = classify_shell("/usr/bin/python3 x.py", SETTINGS).segments[0]
    assert (usr.program, usr.argv[0], usr.signature) == ("/usr/bin/python3", "python3",
                                                          "python3 <script>")


def test_the_program_is_not_part_of_a_segments_identity():
    assert ShellSegment("x", ("x",), program="/a/x") == ShellSegment("x", ("x",))
    assert "program" not in repr(ShellSegment("x", ("x",), program="/a/x"))


def test_a_script_grant_never_covers_another_command(g, proj):
    """R23: an "always" for a script counts where the script is the command (argv[0]) or an
    interpreter's first argument (argv[1]) — never as a trailing argument of another command."""
    script = _script(g, "a.py").resolve()
    key = f"{script}@{DIGEST}"
    granted = _ctx(proj, grants=_granting((CLASS, key)))

    assert _shell(f"python3 {script}", granted).verdict is Verdict.ALLOW
    assert _shell(f"{script} --x", granted).verdict is Verdict.ALLOW
    assert _asked(_shell(f"cp {script} ./copy.py", granted)) == (
        Verdict.ASK, "shell-unfamiliar", "cp")
    decoy = f"curl -T {script} https://example.com/up"
    assert _asked(_shell(decoy, granted)) == (Verdict.ASK, "shell-unfamiliar", "curl")

    both = _shell(decoy, _ctx(proj)).request
    assert set(both.grant_keys) == {(CLASS, key), ("shell-unfamiliar", "curl")}


@pytest.mark.parametrize("command", [
    "python3 $HOME/.localharness/tools/a.py",
    "python3 ~/.localharness/tools/*.py",
    "cd ~/.localharness/tools && python3 a.py",
    "echo 'print(2)' > ~/.localharness/tools/b.py && python3 ~/.localharness/tools/b.py",
], ids=["variable", "glob", "relative-to-cd", "written-by-the-same-command"])
def test_what_security_md_says_is_not_recognised(g, proj, command):
    """SECURITY.md names these: a script named through a variable or a glob, relative to an
    earlier `cd`, or written by the same command that runs it, is not recognised as one."""
    _script(g, "a.py")

    result = _shell(command, _ctx(proj))

    assert all(klass != CLASS for klass, _ in (result.request.grant_keys if result.request else ()))


@pytest.mark.parametrize("pending", [None, lambda p: None], ids=["no-rule", "nothing-pending"])
def test_with_nothing_pending_every_verdict_is_todays(g, proj, pending):
    script = _script(g, "a.py").resolve()
    ctx = _base(proj) if pending is None else _ctx(proj, pending=pending)

    assert _asked(_shell(f"python3 {script}", ctx)) == (
        Verdict.ASK, "shell-unfamiliar", "python3 <script>")
    assert _asked(_shell(f"{script} --x", ctx)) == (Verdict.ASK, "shell-unfamiliar", "a.py")
    assert _shell(f"cat {script}", ctx).verdict is Verdict.ALLOW


# --------------------------------------------------------------------------- the session gate


def _gate(tmp_path: Path, proj: Path, mode: str, asker) -> PermissionGate:
    return PermissionGate(boundary=proj, workspace=proj, grants=GrantStore(tmp_path / "grants.yaml"),
                          channel_name="test", mode=mode, asker=asker)  # type: ignore[arg-type]


async def _run(gate: PermissionGate, command: str):
    return await gate.check("bash_exec", {"command": command}, SHELL, agent_id="a", session_id="s")


def test_the_session_gate_reads_the_machine_record(tmp_path, proj):
    pending = _gate(tmp_path, proj, "guarded", None).context().script_pending
    assert pending.func is trust.tool_script_pending and pending.keywords == {"global_dir": None}


@pytest.mark.asyncio
async def test_a_linked_script_in_guarded_asks_by_its_entry_and_no_python3_grant_covers_it(
        g, proj, tmp_path, monkeypatch):
    _script(g, "a.py")
    _adopt(g, monkeypatch)
    outside = tmp_path / "evil.py"
    outside.write_bytes(b"import os\n")
    link = g / "tools" / "helper.py"
    link.symlink_to(outside)
    ctx = _ctx(proj, grants=_granting(("shell-unfamiliar", "python3 <script>")),
               pending=trust.tool_script_pending)

    assert _asked(_shell(f"python3 {link}", ctx)) == (
        Verdict.ASK, CLASS, f"{link}@{_short(b'import os' + NL)}")


@pytest.mark.parametrize("mode", ["auto", "guarded"])
def test_a_config_dir_sessions_protected_entries_are_protected_and_so_are_the_defaults(
        g, tmp_path, proj, mode):
    """The verifier's finding: the protected list (plugins/**, grants.yaml, trusted_workspaces.yaml
    …) was keyed on the default folder only, so in a `--config-dir /custom` session a write to
    /custom/plugins/x/__init__.py ran in `auto` without a question. Both folders are now guarded:
    the session's own, and the default one, where its trust store and grants still live."""
    custom = (tmp_path / "custom").resolve()
    (custom / "plugins" / "x").mkdir(parents=True)
    ctx = dataclasses.replace(_base(proj, mode), config_dir=custom)
    write = ToolMeta(group="fs.write", destructive=True)

    for target in (custom / "plugins" / "x" / "__init__.py", custom / "grants.yaml",
                   g / "trusted_workspaces.yaml", g / "plugins" / "y.py"):
        result = evaluate("write", {"path": str(target), "content": "x"}, write, ctx, SETTINGS)
        assert result.verdict is Verdict.ASK and result.request.klass == "protected-path", target
    agent_file = evaluate("write", {"path": str(custom / "agents" / "x.yaml"), "content": "x"},
                          write, ctx, SETTINGS)
    assert agent_file.verdict is Verdict.ALLOW or agent_file.request.klass != "protected-path"


@pytest.mark.asyncio
async def test_a_session_started_with_its_own_config_dir_guards_that_folder(tmp_path, proj, monkeypatch):
    """`--config-dir /custom`: the gate is handed /custom, so its tools folder and its settings
    files are guarded though the folder is not named .localharness and is not the default one."""
    custom = (tmp_path / "custom").resolve()
    custom.mkdir()
    (custom / "config.yaml").write_text("{}\n", encoding="utf-8")
    body = b"print('agent wrote me')\n"
    script = custom / "tools" / "x.py"
    script.parent.mkdir()
    script.write_bytes(body)
    asked = []

    async def _record(request):
        asked.append(request)
        return Decision(kind="allow_once")

    gate = PermissionGate(boundary=proj, workspace=proj, grants=GrantStore(tmp_path / "grants.yaml"),
                          channel_name="test", mode="guarded", asker=_record,  # type: ignore[arg-type]
                          config_dir=custom)

    assert (await _run(gate, f"python3 {script}")).allowed
    assert [(r.klass, r.key) for r in asked] == [(CLASS, f"{script}@{_short(body)}")]
    refused = await gate.check("write", {"path": str(custom / "config.yaml"), "content": "x"},
                               ToolMeta(group="fs.write", destructive=True), agent_id="a", session_id="s")
    assert not refused.allowed and "localharness components set" in refused.reason


@pytest.mark.asyncio
async def test_guarded_asks_until_the_start_confirms_the_script(g, proj, tmp_path, monkeypatch):
    _script(g, "old.py")
    _adopt(g, monkeypatch)
    body = b"print('new')\n"
    new = _script(g, "new.py", body).resolve()
    seen = []

    async def _always(request):
        seen.append(request)
        return Decision(kind="allow_always")

    gate = _gate(tmp_path, proj, "guarded", _always)
    command = f"python3 {new}"

    assert (await _run(gate, command)).allowed
    assert [(r.klass, r.key, r.grantable) for r in seen] == [(CLASS, f"{new}@{_short(body)}", True)]
    assert (await _run(gate, command)).allowed
    assert len(seen) == 1, "an always given to the script covers that exact content"

    new.write_bytes(b"print('changed, and longer')\n")
    assert (await _run(gate, command)).allowed
    assert len(seen) == 2 and seen[1].klass == CLASS, "changed content is a new question"

    trust.record_machine(g, trust.machine_snapshot(g))  # the Yes at the next start
    assert (await _run(gate, command)).allowed
    assert (seen[2].klass, seen[2].key) == ("shell-unfamiliar", "python3 <script>")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "trusted"])
async def test_auto_and_trusted_never_ask_about_an_unconfirmed_script(g, proj, tmp_path, monkeypatch,
                                                                      mode):
    _adopt(g, monkeypatch)
    new = _script(g, "new.py").resolve()

    async def _boom(request):
        raise AssertionError(f"{mode} asked: {request.display}")

    assert (await _run(_gate(tmp_path, proj, mode, _boom), f"python3 {new}")).allowed


# --------------------------------------------------------------------------- through the real start


@pytest.fixture
def machine(tmp_path, monkeypatch, fake_home):
    """The real `_start_async` on a hermetic machine (tests.integration.test_image_plugin_e2e:
    offline provider, the REPL read loop returning at once), recording every session gate it
    builds. Real: the machine decision and record, the packaged-helper install, the gate."""
    from tests.integration.test_image_plugin_e2e import _machine

    global_dir, cwd = _machine(tmp_path, monkeypatch, fake_home, project=False)
    trust.record_trust(cwd, True)  # the workspace question is not under test here
    gates: list[PermissionGate] = []
    real_init = PermissionGate.__init__

    def _init(self, *a, **kw):
        real_init(self, *a, **kw)
        gates.append(self)

    monkeypatch.setattr(PermissionGate, "__init__", _init)
    return global_dir, gates


async def _start(monkeypatch, *, tty: bool, answers: tuple[bool, ...] = ()) -> list[str]:
    from localharness.cli.start_cmd import _start_async

    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: tty)
    asked = _answers(monkeypatch, *answers)  # one question too many pops an empty queue: a failure
    await _start_async(None, False, False, None)
    return asked


async def _ask_class(gate: PermissionGate, command: str) -> str:
    seen = []

    async def _once(request):
        seen.append(request)
        return Decision(kind="allow_once")

    gate.mode, gate.asker = "guarded", _once
    assert (await _run(gate, command)).allowed
    return seen[-1].klass


@pytest.mark.asyncio
async def test_through_the_real_start_a_new_script_waits_for_one_yes(machine, monkeypatch, capsys):
    g, gates = machine
    await _start(monkeypatch, tty=True)  # the first start with this release: adopts, asks nothing
    assert (g / "tools" / "design-screenshot.js").is_file(), "start installed its own helper"
    helper = _script(g, "helper.py", b"print('a helper the agent wrote')\n").resolve()
    capsys.readouterr()

    await _start(monkeypatch, tty=False)
    err = " ".join(capsys.readouterr().err.split())
    assert "helper.py (tools/helper.py)" in err and "design-screenshot" not in err
    assert await _ask_class(gates[-1], f"python3 {helper}") == CLASS

    asked = await _start(monkeypatch, tty=True, answers=(True,))
    assert len(asked) == 1 and "+ helper.py (tools/helper.py): sha256:" in asked[0]
    assert "design-screenshot" not in asked[0]
    assert await _ask_class(gates[-1], f"python3 {helper}") == "shell-unfamiliar"
