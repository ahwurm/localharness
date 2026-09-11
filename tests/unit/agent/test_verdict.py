"""The pure verdict (PRD §3.1, §3.4): order, boundary, protected paths, modes, grants.

The shell branch of ``evaluate`` imports ``agent/shell_classify`` lazily, so almost every test
here builds its own ``ShellClassification`` and injects a stand-in module — the verdict's job is
what it DOES with a classification, and that stays testable whether or not A1 has landed. One
smoke test drives the real classifier under ``importorskip``.
"""
from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

from localharness.agent.gate_types import (
    GateSettings,
    Grant,
    PermissionRequest,
    Refusal,
    ShellClassification,
    ShellSegment,
    ToolMeta,
    Verdict,
)
from localharness.agent.permissions import PermissionResult
from localharness.agent.verdict import (
    READ_ONLY_DENY_REASON,
    REFUSAL_DENY_REASON,
    GateContext,
    derive_boundary,
    evaluate,
    narrow_boundary,
)

SETTINGS = GateSettings()
ALL_MODES = ("guarded", "trusted", "read-only", "unattended")
WRITE_META = ToolMeta(destructive=True, group="fs.write")
SHELL_META = ToolMeta(destructive=True, group="shell")
READ_META = ToolMeta(group="fs.read")


class LookupSpy:
    """A ``GrantLookup`` that records every call — the probe for "grants were not consulted"."""

    def __init__(self, grant: Grant | None = None) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.grant = grant

    def __call__(self, workspace: Path, key: str) -> Grant | None:
        self.calls.append((workspace, key))
        return self.grant


def no_grants(workspace: Path, key: str) -> Grant | None:
    return None


def a_grant(key: str = "k") -> Grant:
    return Grant(
        key=key, klass="shell-unfamiliar", granted_at="2026-09-11T00:00:00+00:00",
        channel="terminal", session_id="s1", workspace="/w",
    )


def a_refusal(key: str) -> Refusal:
    return Refusal(
        key=key, klass="shell-unfamiliar", refused_at="2026-09-11T00:00:00+00:00",
        channel="terminal", session_id="s1", workspace="/w",
    )


def refusing(*keys: str):
    """A ``RefusalLookup`` that refuses exactly these keys — nothing fuzzy, nothing globbed."""
    def _refused(workspace: Path, key: str) -> Refusal | None:
        return a_refusal(key) if key in keys else None
    return _refused


def make_ctx(workspace: Path, **kw) -> GateContext:
    kw.setdefault("boundary", workspace)
    kw.setdefault("grants", no_grants)
    kw.setdefault("has_review_surface", True)
    return GateContext(workspace=workspace, **kw)


def fake_shell(monkeypatch, *segments: ShellSegment) -> None:
    """Inject a stand-in ``agent/shell_classify`` returning a hand-built classification."""
    module = types.ModuleType("localharness.agent.shell_classify")
    module.classify_shell = lambda command, settings: ShellClassification(segments=tuple(segments))
    monkeypatch.setitem(sys.modules, "localharness.agent.shell_classify", module)


def seg(signature: str, **kw) -> ShellSegment:
    kw.setdefault("argv", tuple(signature.split()))
    return ShellSegment(signature=signature, **kw)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with its own HOME and GLOBAL config dir, so nothing touches the real ones."""
    home = (tmp_path / "home").resolve()
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    workspace = (home / "proj").resolve()
    workspace.mkdir()
    return workspace


# ------------------------------------------------------------------ the boundary

def test_derive_boundary_prefers_the_workspace_dir_then_git_then_cwd(tmp_path):
    home = tmp_path / "home"
    proj = tmp_path / "home" / "proj"
    (proj / ".localharness").mkdir(parents=True)
    assert derive_boundary(proj, proj / ".localharness", tmp_path, home) == proj.resolve()
    assert derive_boundary(proj, None, proj, home) == proj.resolve()
    assert derive_boundary(proj, None, None, home) == proj.resolve()


def test_home_and_root_boundaries_collapse_to_none(tmp_path):
    """PRD §3.1 / critic finding 1: a boundary containing your home directory is not one."""
    home = (tmp_path / "home").resolve()
    home.mkdir()
    assert derive_boundary(home, None, None, home) is None
    assert derive_boundary(home, home / ".localharness", None, home) is None
    assert derive_boundary(home, None, tmp_path, home) is None  # an ancestor of home
    assert derive_boundary(home, None, Path("/"), home) is None


def test_narrow_boundary_narrows_inside_and_ignores_outside(tmp_path):
    boundary = (tmp_path / "proj").resolve()
    inside = boundary / "src"
    inside.mkdir(parents=True)
    assert narrow_boundary(boundary, str(inside)) == (inside, None)
    assert narrow_boundary(boundary, None) == (boundary, None)
    effective, warning = narrow_boundary(boundary, str(tmp_path))
    assert effective == boundary and "outside" in warning


def test_narrow_boundary_cannot_invent_a_boundary_where_home_collapsed(tmp_path):
    effective, warning = narrow_boundary(None, str(tmp_path))
    assert effective is None and "no workspace boundary" in warning


# ------------------------------------------------------------------- deny is first

@pytest.mark.parametrize("mode", ALL_MODES)
def test_deny_beats_every_grant_and_every_mode(ws, mode):
    spy = LookupSpy(a_grant())
    ctx = make_ctx(ws, mode=mode, grants=spy, deny=lambda n, p: PermissionResult(True, "Matches deny pattern: write"))
    result = evaluate("write", {"path": str(ws / "f.py"), "content": "x"}, WRITE_META, ctx, SETTINGS)
    assert result.verdict is Verdict.DENY
    assert "deny pattern" in result.reason
    assert spy.calls == []


# ------------------------------------------------------------ write / edit tier

def test_in_workspace_edit_with_a_review_surface_never_asks(ws):
    result = evaluate("edit", {"path": str(ws / "a.py"), "old_string": "a", "new_string": "b"},
                      WRITE_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ALLOW


def test_in_workspace_edit_without_a_review_surface_asks_once_per_workspace(ws):
    ctx = make_ctx(ws, has_review_surface=False)
    result = evaluate("write", {"path": str(ws / "a.py"), "content": "x"}, WRITE_META, ctx, SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "edit-unreviewed"
    assert result.request.key == str(ws)
    assert result.request.grantable is True


def test_edit_unreviewed_is_silenced_by_its_grant(ws):
    ctx = make_ctx(ws, has_review_surface=False, grants=lambda w, k: a_grant(k) if k == str(ws) else None)
    result = evaluate("write", {"path": str(ws / "a.py"), "content": "x"}, WRITE_META, ctx, SETTINGS)
    assert result.verdict is Verdict.ALLOW


def test_a_write_outside_the_boundary_asks_with_the_parent_dir_as_key(ws, tmp_path):
    outside = (tmp_path / "elsewhere").resolve()
    outside.mkdir()
    result = evaluate("write", {"path": str(outside / "f.txt"), "content": "x"},
                      WRITE_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "edit-outside"
    assert result.request.key == str(outside)
    assert result.request.grantable is True


def test_an_edit_outside_grant_silences_that_directory(ws, tmp_path):
    outside = (tmp_path / "elsewhere").resolve()
    outside.mkdir()
    ctx = make_ctx(ws, grants=lambda w, k: a_grant(k) if k == str(outside) else None)
    result = evaluate("write", {"path": str(outside / "f.txt"), "content": "x"}, WRITE_META, ctx, SETTINGS)
    assert result.verdict is Verdict.ALLOW


def test_a_symlink_out_of_the_workspace_is_an_edit_outside(ws, tmp_path):
    """Realpath on both sides: the escape hatch is the symlink, so resolve before comparing."""
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    (ws / "link").symlink_to(outside, target_is_directory=True)
    result = evaluate("write", {"path": str(ws / "link" / "f.txt"), "content": "x"},
                      WRITE_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "edit-outside"
    assert result.request.key == str(outside)


def test_a_write_naming_no_target_asks_rather_than_passing(ws):
    result = evaluate("write", {"content": "x"}, WRITE_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "edit-outside"


# --------------------------------------------------------------- protected paths

@pytest.mark.parametrize("relative", [".git/config", ".env", "secrets/deploy.pem", "keys/id_rsa"])
def test_workspace_protected_names_match_at_any_depth(ws, relative):
    result = evaluate("write", {"path": str(ws / relative), "content": "x"},
                      WRITE_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "protected-path"
    assert result.request.grantable is False


def test_home_protected_paths_are_ungrantable(ws):
    home = Path(ws).parent
    result = evaluate("write", {"path": str(home / ".ssh" / "id_rsa"), "content": "x"},
                      WRITE_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "protected-path"
    assert result.request.grantable is False


def test_the_harness_config_dir_is_protected_but_its_runtime_store_is_exempt(tmp_path, monkeypatch):
    """PRD §3.1: ``~/.localharness`` except the harness's own runtime store."""
    home = (tmp_path / "home").resolve()
    workspace = home / "proj"
    store = workspace / "runtime"
    store.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(store))
    ctx = make_ctx(workspace)

    protected = evaluate("write", {"path": str(store / "config.yaml"), "content": "x"},
                         WRITE_META, ctx, SETTINGS)
    assert protected.verdict is Verdict.ASK and protected.request.klass == "protected-path"

    exempt = evaluate("write", {"path": str(store / "audit.jsonl"), "content": "x"},
                      WRITE_META, ctx, SETTINGS)
    assert exempt.verdict is Verdict.ALLOW


# ------------------------------------------------------------------- no boundary

@pytest.mark.parametrize(
    "tool_name,params",
    [
        ("write", {"path": "f.txt", "content": "x"}),
        ("edit", {"path": "f.txt", "old_string": "a", "new_string": "b"}),
    ],
)
def test_every_write_shaped_call_asks_when_there_is_no_boundary(ws, tool_name, params):
    ctx = make_ctx(ws, boundary=None)
    result = evaluate(tool_name, params, WRITE_META, ctx, SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "no-boundary"
    assert result.request.grantable is False


def test_no_boundary_also_catches_a_shell_write_target(ws, monkeypatch):
    fake_shell(monkeypatch, seg("tee", write_targets=("out.txt",)))
    ctx = make_ctx(ws, boundary=None)
    result = evaluate("bash_exec", {"command": "tee out.txt"}, SHELL_META, ctx, SETTINGS)
    assert result.verdict is Verdict.ASK and result.request.klass == "no-boundary"


# --------------------------------------------------------------- grant hygiene

@pytest.mark.parametrize(
    "tool_name,params,setup",
    [
        ("write", {"path": ".git/config", "content": "x"}, None),          # protected-path
        ("write", {"path": "f.txt", "content": "x"}, "no-boundary"),        # no-boundary
        ("bash_exec", {"command": "rm -rf build"}, "destructive"),          # shell-destructive
    ],
)
def test_ungrantable_classes_never_consult_the_grant_store(ws, monkeypatch, tool_name, params, setup):
    """PRD §3.1 / critic finding 12: the ungrantable tier runs BEFORE any grant lookup."""
    if setup == "destructive":
        fake_shell(monkeypatch, seg("rm -rf", destructive=True))
    spy = LookupSpy(a_grant())
    ctx = make_ctx(ws, grants=spy, boundary=None if setup == "no-boundary" else ws)
    if tool_name == "write" and not str(params["path"]).startswith("/"):
        params = dict(params, path=str(ws / params["path"]))
    result = evaluate(tool_name, params, WRITE_META if tool_name == "write" else SHELL_META, ctx, SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.grantable is False
    assert spy.calls == []


# ------------------------------------------------------------------------ modes

def test_trusted_allows_grantable_classes_but_still_asks_the_ungrantable_ones(ws, tmp_path):
    outside = (tmp_path / "elsewhere").resolve()
    outside.mkdir()
    ctx = make_ctx(ws, mode="trusted")
    grantable = evaluate("write", {"path": str(outside / "f.txt"), "content": "x"}, WRITE_META, ctx, SETTINGS)
    assert grantable.verdict is Verdict.ALLOW

    ungrantable = evaluate("write", {"path": str(ws / ".git" / "config"), "content": "x"},
                           WRITE_META, ctx, SETTINGS)
    assert ungrantable.verdict is Verdict.ASK
    assert ungrantable.request.klass == "protected-path"


@pytest.mark.parametrize(
    "tool_name,params,meta",
    [
        ("write", {"path": ".git/config", "content": "x"}, WRITE_META),
        ("python_exec", {"code": "print(1)"}, ToolMeta(destructive=True, group="code")),
        ("agent", {"agent_id": "researcher", "task": "go"}, ToolMeta(group="delegate")),
    ],
)
def test_unattended_turns_every_ask_into_allow(ws, tool_name, params, meta):
    """PRD §3.4: today's shipped behavior, named honestly — what bench and cron pin."""
    if "path" in params:
        params = dict(params, path=str(ws / params["path"]))
    result = evaluate(tool_name, params, meta, make_ctx(ws, mode="unattended"), SETTINGS)
    assert result.verdict is Verdict.ALLOW


@pytest.mark.parametrize(
    "tool_name,params,meta",
    [
        ("write", {"path": "a.py", "content": "x"}, WRITE_META),
        ("edit", {"path": "a.py", "old_string": "a", "new_string": "b"}, WRITE_META),
        ("python_exec", {"code": "print(1)"}, ToolMeta(destructive=True, group="code")),
        ("agent", {"agent_id": "researcher", "task": "go"}, ToolMeta(group="delegate")),
    ],
)
def test_read_only_soft_denies_everything_that_changes_state(ws, tool_name, params, meta):
    if "path" in params:
        params = dict(params, path=str(ws / params["path"]))
    result = evaluate(tool_name, params, meta, make_ctx(ws, mode="read-only"), SETTINGS)
    assert result.verdict is Verdict.DENY
    assert result.reason == READ_ONLY_DENY_REASON


def test_read_only_denies_a_non_read_only_shell_command(ws, monkeypatch):
    fake_shell(monkeypatch, seg("npm install"))
    result = evaluate("bash_exec", {"command": "npm install"}, SHELL_META,
                      make_ctx(ws, mode="read-only"), SETTINGS)
    assert result.verdict is Verdict.DENY and result.reason == READ_ONLY_DENY_REASON


def test_read_only_still_allows_reads(ws, monkeypatch):
    fake_shell(monkeypatch, seg("ls", read_only=True))
    assert evaluate("read", {"path": str(ws / "a.py")}, READ_META,
                    make_ctx(ws, mode="read-only"), SETTINGS).verdict is Verdict.ALLOW
    assert evaluate("bash_exec", {"command": "ls"}, SHELL_META,
                    make_ctx(ws, mode="read-only"), SETTINGS).verdict is Verdict.ALLOW


# ------------------------------------------------------------------------ shell

def test_read_only_shell_signatures_never_ask(ws, monkeypatch):
    fake_shell(monkeypatch, seg("ls", read_only=True), seg("git status", read_only=True))
    result = evaluate("bash_exec", {"command": "ls && git status"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ALLOW


def test_an_unfamiliar_signature_asks_once_and_the_grant_silences_it(ws, monkeypatch):
    fake_shell(monkeypatch, seg("npm install"))
    result = evaluate("bash_exec", {"command": "npm install"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert (result.request.klass, result.request.key) == ("shell-unfamiliar", "npm install")

    granted = make_ctx(ws, grants=lambda w, k: a_grant(k) if k == "npm install" else None)
    assert evaluate("bash_exec", {"command": "npm install"}, SHELL_META, granted, SETTINGS).verdict is Verdict.ALLOW


def test_inline_interpreters_get_their_own_class_and_key(ws, monkeypatch):
    fake_shell(monkeypatch, seg("python3 -c", inline_interpreter=True))
    result = evaluate("bash_exec", {"command": 'python3 -c "print(1)"'}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.request.klass == "interpreter-inline"
    assert result.request.key == "python3 -c"
    assert result.request.grantable is True


def test_a_destructive_segment_is_ungrantable_whatever_else_the_line_does(ws, monkeypatch):
    fake_shell(monkeypatch, seg("cd", read_only=True), seg("rm -rf", destructive=True))
    result = evaluate("bash_exec", {"command": "cd build && rm -rf ."}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "shell-destructive"
    assert result.request.grantable is False


def test_a_shell_write_target_outside_the_boundary_asks_before_the_signature_does(ws, monkeypatch, tmp_path):
    outside = (tmp_path / "elsewhere").resolve()
    outside.mkdir()
    fake_shell(monkeypatch, seg("tee", write_targets=(str(outside / "f.txt"),)))
    result = evaluate("bash_exec", {"command": f"tee {outside}/f.txt"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.request.klass == "edit-outside"
    assert result.request.key == str(outside)


def test_an_unresolvable_shell_write_target_counts_as_outside(ws, monkeypatch):
    fake_shell(monkeypatch, seg("tee", write_targets=("$OUT/f.txt",), unresolvable_write=True))
    result = evaluate("bash_exec", {"command": "tee $OUT/f.txt"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.request.klass == "edit-outside"
    assert result.request.key == "$OUT/f.txt"


def test_an_in_workspace_shell_write_target_does_not_add_an_ask(ws, monkeypatch):
    fake_shell(monkeypatch, seg("ls", read_only=True, write_targets=(str(ws / "out.txt"),)))
    result = evaluate("bash_exec", {"command": "ls > out.txt"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ALLOW


# ------------------------------------------- the shell call's own working directory

def test_a_relative_shell_target_lands_where_working_dir_points_not_in_the_workspace(ws, monkeypatch):
    """R2b: ``bash_exec`` takes a ``working_dir``; a relative target resolves against THAT.

    Anchoring at the workspace instead read this call as an in-workspace write and allowed it.
    """
    fake_shell(monkeypatch, seg("echo", read_only=True, write_targets=("authorized_keys",)))
    result = evaluate(
        "bash_exec", {"command": "echo k >> authorized_keys", "working_dir": "~/.ssh"},
        SHELL_META, make_ctx(ws), SETTINGS,
    )
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "protected-path"
    assert result.request.grantable is False
    assert result.request.key == str(Path("~/.ssh/authorized_keys").expanduser())


def test_a_working_dir_outside_the_boundary_puts_every_relative_target_outside(ws, monkeypatch, tmp_path):
    outside = (tmp_path / "x").resolve()
    outside.mkdir()
    fake_shell(monkeypatch, seg("echo", read_only=True, write_targets=("f",)))
    result = evaluate(
        "bash_exec", {"command": "echo k > f", "working_dir": str(outside)},
        SHELL_META, make_ctx(ws), SETTINGS,
    )
    assert result.request.klass == "edit-outside"
    assert result.request.key == str(outside)


def test_without_a_working_dir_relative_targets_still_anchor_at_the_workspace(ws, monkeypatch):
    fake_shell(monkeypatch, seg("echo", read_only=True, write_targets=("out.txt",)))
    for params in ({"command": "echo k > out.txt"}, {"command": "echo k > out.txt", "working_dir": "."}):
        assert evaluate("bash_exec", params, SHELL_META, make_ctx(ws), SETTINGS).verdict is Verdict.ALLOW


def test_an_absolute_target_ignores_the_working_dir(ws, monkeypatch):
    """The classifier may join a ``cd`` onto a later target, so targets arrive ``~``/``/``-prefixed."""
    fake_shell(monkeypatch, seg("echo", read_only=True, write_targets=(str(ws / "out.txt"),)))
    result = evaluate(
        "bash_exec", {"command": "cd /etc && echo k > out.txt", "working_dir": "/etc"},
        SHELL_META, make_ctx(ws), SETTINGS,
    )
    assert result.verdict is Verdict.ALLOW


def test_an_unresolvable_working_dir_makes_relative_targets_outside(ws, monkeypatch):
    fake_shell(monkeypatch, seg("echo", read_only=True, write_targets=("f",)))
    result = evaluate(
        "bash_exec", {"command": "echo k > f", "working_dir": "$DEST"},
        SHELL_META, make_ctx(ws), SETTINGS,
    )
    assert result.request.klass == "edit-outside"
    assert result.request.key == "f"


@pytest.mark.parametrize("signature,command", [
    ("$__lh_subst__", "$(echo rm) -rf build"),
    ("$RM", '"$RM" -rf build'),
    ("`echo rm`", "`echo rm` -rf build"),
])
def test_a_computed_command_name_asks_every_time_and_cannot_be_remembered(ws, monkeypatch, signature, command):
    """PRD §3.2 residual gap: one "always" on ``$RM`` would cover every future built command."""
    fake_shell(monkeypatch, seg(signature))
    spy = LookupSpy(a_grant(signature))
    result = evaluate("bash_exec", {"command": command}, SHELL_META, make_ctx(ws, grants=spy), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "shell-unfamiliar"
    assert result.request.grantable is False
    assert result.request.key is None
    assert "computed at runtime" in result.request.reason
    assert spy.calls == []


@pytest.mark.parametrize("signature", ["$__lh_subst__", "$RM"])
def test_a_computed_command_name_still_asks_under_trusted(ws, monkeypatch, signature):
    fake_shell(monkeypatch, seg(signature))
    spy = LookupSpy(a_grant(signature))
    result = evaluate("bash_exec", {"command": f"{signature} -rf build"}, SHELL_META,
                      make_ctx(ws, mode="trusted", grants=spy), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.grantable is False
    assert spy.calls == []


@pytest.mark.parametrize("signature", ["$__lh_subst__", "$RM"])
def test_a_computed_command_name_is_allowed_under_unattended(ws, monkeypatch, signature):
    fake_shell(monkeypatch, seg(signature))
    result = evaluate("bash_exec", {"command": f"{signature} -rf build"}, SHELL_META,
                      make_ctx(ws, mode="unattended"), SETTINGS)
    assert result.verdict is Verdict.ALLOW


def test_an_ordinary_signature_is_still_grantable(ws, monkeypatch):
    """The dynamic-name rule must not swallow the normal shell-unfamiliar path."""
    fake_shell(monkeypatch, seg("npm install"))
    result = evaluate("bash_exec", {"command": "npm install"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.request.grantable is True and result.request.key == "npm install"


def test_an_empty_command_is_nothing_to_classify(ws):
    assert evaluate("bash_exec", {"command": "   "}, SHELL_META, make_ctx(ws), SETTINGS).verdict is Verdict.ALLOW


def test_the_real_classifier_agrees_that_rm_rf_is_destructive(ws):
    """Integration smoke against A1; skipped until ``agent/shell_classify`` lands."""
    pytest.importorskip("localharness.agent.shell_classify")
    result = evaluate("bash_exec", {"command": "rm -rf build"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "shell-destructive"
    assert result.request.grantable is False


# ------------------------------------------------------- code, delegate, mcp, web

@pytest.mark.parametrize(
    "tool_name,klass,meta",
    [
        ("python_exec", "code-exec", ToolMeta(destructive=True, group="code")),
        ("cruncher_exec", "code-exec", ToolMeta(destructive=True, group="code")),
        ("agent", "delegate", ToolMeta(group="delegate")),
    ],
)
def test_code_and_delegate_ask_once_per_tool_name(ws, tool_name, klass, meta):
    result = evaluate(tool_name, {"code": "print(1)", "task": "go"}, meta, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert (result.request.klass, result.request.key) == (klass, tool_name)
    assert result.request.grantable is True

    granted = make_ctx(ws, grants=lambda w, k: a_grant(k) if k == tool_name else None)
    assert evaluate(tool_name, {"code": "print(1)"}, meta, granted, SETTINGS).verdict is Verdict.ALLOW


def test_mcp_tools_ask_once_per_server_and_tool(ws):
    meta = ToolMeta(destructive=True, group="mcp/linear", is_mcp=True, mcp_server="linear")
    result = evaluate("linear__create_issue", {"title": "x"}, meta, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert (result.request.klass, result.request.key) == ("mcp", "mcp/linear/create_issue")


def test_a_trusted_mcp_server_skips_the_ask(ws):
    meta = ToolMeta(destructive=True, group="mcp/linear", is_mcp=True, mcp_server="linear")
    settings = dataclasses.replace(SETTINGS, mcp_trusted_servers=frozenset({"linear"}))
    assert evaluate("linear__create_issue", {"title": "x"}, meta, make_ctx(ws), settings).verdict is Verdict.ALLOW


def test_network_reads_are_silent_by_default_and_per_host_when_asked_for(ws):
    meta = ToolMeta(group="web")
    params = {"url": "https://example.com/a/b"}
    assert evaluate("web_fetch", params, meta, make_ctx(ws), SETTINGS).verdict is Verdict.ALLOW

    settings = dataclasses.replace(SETTINGS, ask_network_hosts=True)
    result = evaluate("web_fetch", params, meta, make_ctx(ws), settings)
    assert (result.request.klass, result.request.key) == ("network-host", "example.com")

    granted = make_ctx(ws, grants=lambda w, k: a_grant(k) if k == "example.com" else None)
    assert evaluate("web_fetch", params, meta, granted, settings).verdict is Verdict.ALLOW


def test_a_search_that_names_no_host_never_asks(ws):
    settings = dataclasses.replace(SETTINGS, ask_network_hosts=True)
    result = evaluate("web_search", {"query": "acp"}, ToolMeta(group="web"), make_ctx(ws), settings)
    assert result.verdict is Verdict.ALLOW


def test_read_tier_tools_are_allowed_without_a_grant_lookup(ws):
    spy = LookupSpy()
    ctx = make_ctx(ws, grants=spy)
    for name in ("read", "glob", "grep", "memory_search", "chunk"):
        assert evaluate(name, {"path": str(ws / "a.py")}, READ_META, ctx, SETTINGS).verdict is Verdict.ALLOW
    assert spy.calls == []


def test_an_unknown_tool_is_classified_by_its_group(ws, tmp_path):
    outside = (tmp_path / "elsewhere").resolve()
    outside.mkdir()
    meta = ToolMeta(destructive=True, group="fs.write")
    result = evaluate("plugin_writer", {"file_path": str(outside / "f.txt")}, meta, make_ctx(ws), SETTINGS)
    assert result.request.klass == "edit-outside"


# ---------------------------------------------------------------------- display

def test_the_request_renders_as_one_readable_line(ws, monkeypatch):
    fake_shell(monkeypatch, seg("rm -rf", destructive=True))
    request = evaluate("bash_exec", {"command": "rm -rf build"}, SHELL_META, make_ctx(ws), SETTINGS).request
    assert isinstance(request, PermissionRequest)
    assert "\n" not in request.display
    assert request.display.startswith("bash_exec: rm -rf build")
    assert "shell-destructive" in request.display


def test_a_long_argument_is_truncated_to_one_line(ws):
    long_path = str(ws / ("d" * 200) / "f.txt")
    request = evaluate("write", {"path": long_path, "content": "x"}, WRITE_META,
                       make_ctx(ws, boundary=None), SETTINGS).request
    assert len(request.display.split("  (")[0]) <= len("write: ") + 80
    assert request.tool_params["path"] == long_path


# ------------------------------------------------------- one call, one question (D1)

def test_one_call_collects_every_ask_into_one_request(ws, monkeypatch, tmp_path):
    """PRD §7: "'always' → the same command never asks again". Verification A defect D1: the
    shipped verdict returned the FIRST unsatisfied class, so a two-verb command asked once per
    class, each prompt costing a whole turn."""
    outside = (tmp_path / "elsewhere").resolve()
    fake_shell(
        monkeypatch,
        seg("mkdir", write_targets=(str(outside / "y"),)),
        seg("touch", write_targets=(str(outside / "y" / "f"),)),
    )
    result = evaluate(
        "bash_exec", {"command": "mkdir -p x ; touch y"}, SHELL_META, make_ctx(ws), SETTINGS
    )
    assert result.verdict is Verdict.ASK
    request = result.request
    assert request.grantable is True
    assert set(request.grant_keys) == {
        ("edit-outside", str(outside)),
        ("edit-outside", str(outside / "y")),
        ("shell-unfamiliar", "mkdir"),
        ("shell-unfamiliar", "touch"),
    }
    assert request.klass == "edit-outside", "the most severe ask names the request"
    assert "\n" not in request.display
    assert "mkdir, touch" in request.display, request.display


def test_the_verifiers_repro_asks_once_with_both_signatures(ws, tmp_path):
    """The exact command from VERIFICATION-A, through the REAL classifier."""
    pytest.importorskip("localharness.agent.shell_classify")
    outside = (tmp_path / "x").resolve()
    command = f"mkdir -p {outside}/y ; touch {outside}/y/f"
    request = evaluate("bash_exec", {"command": command}, SHELL_META, make_ctx(ws), SETTINGS).request
    assert request.grantable is True
    signatures = {k for klass, k in request.grant_keys if klass == "shell-unfamiliar"}
    directories = {k for klass, k in request.grant_keys if klass == "edit-outside"}
    assert signatures == {"mkdir", "touch"}
    assert str(outside) in directories


def test_one_ungrantable_ask_makes_the_whole_request_ungrantable(ws, monkeypatch, tmp_path):
    """Precedence is unchanged: a destructive segment bundled with a benign first-exposure one
    must not be quietly remembered by a single "always here"."""
    outside = (tmp_path / "elsewhere").resolve()
    fake_shell(
        monkeypatch,
        seg("rm -rf", destructive=True),
        seg("cp", write_targets=(str(outside / "f"),)),
    )
    request = evaluate(
        "bash_exec", {"command": "rm -rf build ; cp a b"}, SHELL_META, make_ctx(ws), SETTINGS
    ).request
    assert request.klass == "shell-destructive" and request.grantable is False
    assert "edit-outside" in request.display


def test_trusted_mode_needs_every_ask_grantable(ws, monkeypatch):
    """PRD §3.4: trusted allows the grantable classes; one ungrantable ask still asks."""
    fake_shell(monkeypatch, seg("rm -rf", destructive=True), seg("npm install"))
    result = evaluate(
        "bash_exec", {"command": "rm -rf build ; npm install"}, SHELL_META,
        make_ctx(ws, mode="trusted"), SETTINGS,
    )
    assert result.verdict is Verdict.ASK and result.request.grantable is False


def test_a_granted_key_drops_out_of_the_collected_asks(ws, monkeypatch, tmp_path):
    outside = (tmp_path / "elsewhere").resolve()

    def granted(workspace: Path, key: str):
        return a_grant(key) if key == "mkdir" else None

    fake_shell(monkeypatch, seg("mkdir", write_targets=(str(outside / "y"),)), seg("touch"))
    request = evaluate(
        "bash_exec", {"command": "mkdir -p y ; touch f"}, SHELL_META,
        make_ctx(ws, grants=granted), SETTINGS,
    ).request
    assert ("shell-unfamiliar", "mkdir") not in request.grant_keys
    assert ("shell-unfamiliar", "touch") in request.grant_keys


# --------------------------------------------- a directory grant covers its subtree (D4)

def test_a_grant_on_a_directory_covers_a_deeper_target(ws, tmp_path):
    """Verification A defect D4: `grant '/tmp'` did not cover `touch /tmp/sub/x`, so every
    fresh subdirectory under an approved root paid a first-exposure prompt (PRD §8's "growing
    vocabularies", arriving through paths)."""
    outside = (tmp_path / "elsewhere").resolve()

    def granted(workspace: Path, key: str):
        return a_grant(key) if key == str(outside) else None

    deep = outside / "a" / "b" / "f.txt"
    result = evaluate("write", {"path": str(deep), "content": "x"}, WRITE_META,
                      make_ctx(ws, grants=granted), SETTINGS)
    assert result.verdict is Verdict.ALLOW


def test_a_grant_on_a_sibling_directory_covers_nothing(ws, tmp_path):
    """The walk goes UP, never sideways or down."""
    outside = (tmp_path / "elsewhere").resolve()

    def granted(workspace: Path, key: str):
        return a_grant(key) if key == str(outside / "a") else None

    result = evaluate("write", {"path": str(outside / "b" / "f.txt"), "content": "x"}, WRITE_META,
                      make_ctx(ws, grants=granted), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "edit-outside"
    assert result.request.key == str(outside / "b")


def test_a_grant_on_home_does_not_cover_a_protected_path(ws, tmp_path):
    """Protected paths are classified before any grant is consulted (PRD §3.1, finding 12), so
    widening the grant walk cannot open `~/.ssh`."""
    home = (tmp_path / "home").resolve()
    (home / ".ssh").mkdir(parents=True, exist_ok=True)

    def granted(workspace: Path, key: str):
        return a_grant(key)  # every key is granted — the widest possible grant

    result = evaluate("write", {"path": str(home / ".ssh" / "authorized_keys"), "content": "x"},
                      WRITE_META, make_ctx(ws, grants=granted), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.klass == "protected-path" and result.request.grantable is False


def test_the_subtree_grant_reaches_shell_write_targets(ws, monkeypatch, tmp_path):
    """The verifier's probe: `grant '/tmp'` vs `touch /tmp/sub/x` — the shell path uses the same
    target check, so it inherits the walk."""
    outside = (tmp_path / "elsewhere").resolve()

    def granted(workspace: Path, key: str):
        return a_grant(key) if key in (str(outside), "touch") else None

    fake_shell(monkeypatch, seg("touch", write_targets=(str(outside / "sub" / "x"),)))
    result = evaluate("bash_exec", {"command": "touch sub/x"}, SHELL_META,
                      make_ctx(ws, grants=granted), SETTINGS)
    assert result.verdict is Verdict.ALLOW


# ------------------------------------------------------------------- refusals

def test_a_refused_signature_denies_without_a_request(ws, monkeypatch):
    """PRD §3.3: a "never here" denies and asks no more — no prompt is rendered at all."""
    fake_shell(monkeypatch, seg("cargo publish"))
    result = evaluate("bash_exec", {"command": "cargo publish"}, SHELL_META,
                      make_ctx(ws, refusals=refusing("cargo publish")), SETTINGS)
    assert result.verdict is Verdict.DENY
    assert result.request is None
    assert REFUSAL_DENY_REASON in result.reason


def test_a_refusal_matches_its_key_and_not_a_neighbour(ws, monkeypatch):
    """The defect: a refusal on ``cp`` must not reach ``scp``, ``cpio`` or a path holding "cp"."""
    fake_shell(monkeypatch, seg("scp"), seg("cpio"))
    result = evaluate("bash_exec", {"command": "scp a h:b && cpio -o < /srv/backup-cp/list"},
                      SHELL_META, make_ctx(ws, refusals=refusing("cp")), SETTINGS)
    assert result.verdict is Verdict.ASK


def test_a_refusal_beats_a_grant_on_the_same_key(ws, monkeypatch):
    """A tightening always wins, whichever answer was recorded last (PRD §3.3)."""
    fake_shell(monkeypatch, seg("cargo publish"))
    result = evaluate("bash_exec", {"command": "cargo publish"}, SHELL_META,
                      make_ctx(ws, grants=lambda w, k: a_grant(k),
                               refusals=refusing("cargo publish")), SETTINGS)
    assert result.verdict is Verdict.DENY


@pytest.mark.parametrize("mode", ALL_MODES)
def test_no_mode_overrides_a_refusal(ws, monkeypatch, mode):
    """A refusal is a DENY, and DENY ignores modes — unattended included (PRD §3.4)."""
    fake_shell(monkeypatch, seg("cargo publish"))
    result = evaluate("bash_exec", {"command": "cargo publish"}, SHELL_META,
                      make_ctx(ws, mode=mode, refusals=refusing("cargo publish")), SETTINGS)
    assert result.verdict is Verdict.DENY


def test_a_directory_refusal_covers_the_subtree(ws, tmp_path):
    """Directory refusals reach down exactly as directory grants do (PRD §3.1 edit-outside)."""
    outside = (tmp_path / "elsewhere").resolve()
    result = evaluate("write", {"path": str(outside / "y" / "f"), "content": "x"}, WRITE_META,
                      make_ctx(ws, refusals=refusing(str(outside))), SETTINGS)
    assert result.verdict is Verdict.DENY
    assert REFUSAL_DENY_REASON in result.reason


def test_one_refused_key_denies_the_whole_call(ws, monkeypatch, tmp_path):
    """One prompt covers every key of a call, so one refused key denies all of it (PRD §3.3)."""
    outside = (tmp_path / "elsewhere").resolve()
    fake_shell(monkeypatch, seg("cargo publish"), seg("touch", write_targets=(str(outside / "f"),)))
    result = evaluate("bash_exec", {"command": "cargo publish && touch f"}, SHELL_META,
                      make_ctx(ws, refusals=refusing("cargo publish")), SETTINGS)
    assert result.verdict is Verdict.DENY


def test_without_a_refusal_lookup_nothing_is_refused(ws, monkeypatch):
    """``refusals=None`` (a caller that cannot read them) asks; it never silently allows."""
    fake_shell(monkeypatch, seg("cargo publish"))
    result = evaluate("bash_exec", {"command": "cargo publish"}, SHELL_META,
                      make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
