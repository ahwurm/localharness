"""Issue #15 — SHIPPED-DEFAULT permission hardening + opt-in workspace_root confinement.

Two distinct surfaces, one root cause (the 2026-07-11 incident where the subject model's
bash_exec issued `docker stop`/`docker kill` against its OWN vLLM server and the harness
allowed it, and model-authored files landed in the harness repo checkout):

  (a) PermissionConfig().deny_patterns — the SHIPPED DEFAULT must deny destructive
      service/process ops out of the box (docker stop/kill/rm, systemctl stop/disable,
      pkill/killall/kill, shutdown/reboot) while leaving read-only ops (`docker ps`,
      `systemctl status`, `journalctl`) untouched.

  (b) permissions.workspace_root — an OPT-IN confinement: when set, write/edit targets and
      bash_exec working_dir must resolve inside it; default None stays FULLY unconfined
      (file-write power is a core product capability — owner ruling).

This is the shipped-DEFAULT surface. It is DISTINCT from the sema05 EVAL policy (which
appends its own broader `*docker*`/`*vllm*` denies) covered by
tests/unit/test_sema05_designed_budget.py::test_root_agent_config_denies_subject_ops_commands.
"""
import os

import pytest
from pathlib import Path

# git-bash's `pwd` prints the MSYS mount form (/tmp/...), which never string-matches a
# Windows Path; `pwd -W` prints the real Windows path with forward slashes, which Path()
# normalizes. POSIX bash keeps plain `pwd`.
_PWD = "pwd -W" if os.name == "nt" else "pwd"

from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, PermissionConfig
from localharness.core.types import ToolCall


# --- (a) shipped-default deny coverage --------------------------------------

# The real 2026-07-11 casualty class + neighbours: destructive service/process ops.
DENY_COMMANDS = [
    "docker stop vllm-server",
    "docker kill abc123",
    "docker rm -f abc123",
    "docker compose down",
    "systemctl stop vllm",
    "systemctl disable docker",
    "pkill -f vllm",
    "killall python",
    "kill -9 4242",
    "sudo systemctl stop docker",
    "nohup shutdown -h now",
    "reboot",
]

# False-positive guards: read-only / benign ops that MUST stay allowed by default.
ALLOW_COMMANDS = [
    "docker ps",
    "docker logs vllm-server",
    "docker images",
    "systemctl status vllm",
    "journalctl -u docker",
    "grep skill notes.md",   # 'skill' must not trip the kill pattern
    "echo docker is fine",
    "ls -la",
    "python train.py",
]


def _denied(cmd: str, perms) -> bool:
    ev = PermissionEvaluator()
    return ev.evaluate(ToolCall(name="bash_exec", arguments={"command": cmd}), perms).denied


@pytest.mark.parametrize("cmd", DENY_COMMANDS)
def test_shipped_default_denies_destructive_service_ops(cmd):
    assert _denied(cmd, PermissionConfig()), f"default deny list must block: {cmd!r}"


@pytest.mark.parametrize("cmd", ALLOW_COMMANDS)
def test_shipped_default_allows_readonly_ops(cmd):
    assert not _denied(cmd, PermissionConfig()), f"default deny list wrongly blocked: {cmd!r}"


def test_agentconfig_permissions_carry_the_default_deny_list():
    """AgentConfig().permissions must carry the shipped defaults (not an empty list)."""
    perms = AgentConfig(name="x", role="y").permissions
    assert _denied("docker kill x", perms)


def test_sudo_pattern_matches_real_sudo_commands():
    """REGRESSION: the shipped glob was `bash_exec(sudo:*)` — fnmatch needs a literal colon
    after 'sudo', so it NEVER matched a real `sudo <cmd>` (only the bogus form `sudo:...`).
    The fixed pattern must catch actual sudo invocations."""
    perms = PermissionConfig()
    for cmd in ("sudo rm -rf /", "sudo systemctl stop docker", "sudo reboot"):
        assert _denied(cmd, perms), f"sudo command must be denied: {cmd!r}"


def test_embedded_rm_rf_is_denied():
    """Bare `rm -rf *` matched only when the command STARTS with `rm -rf `; embedded forms
    (`cd /tmp && rm -rf x`) slipped through. Growth adds the embedded form."""
    assert _denied("cd /tmp && rm -rf x", PermissionConfig())


# --- (b) opt-in workspace_root confinement ----------------------------------


@pytest.mark.asyncio
async def test_write_denied_outside_workspace_root(tmp_path: Path):
    from localharness.tools.builtin.write_tool import WriteTool

    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    tool = WriteTool(workspace_root=str(root))
    result = await tool.run(path=str(outside), content="x")
    assert result.success is False
    assert result.error_type == "permission_denied"
    assert not outside.exists()


@pytest.mark.asyncio
async def test_write_allowed_inside_workspace_root(tmp_path: Path):
    from localharness.tools.builtin.write_tool import WriteTool

    root = tmp_path / "ws"
    root.mkdir()
    inside = root / "sub" / "note.txt"
    tool = WriteTool(workspace_root=str(root))
    result = await tool.run(path=str(inside), content="hello")
    assert result.success is True
    assert inside.read_text() == "hello"


@pytest.mark.asyncio
async def test_edit_denied_outside_workspace_root(tmp_path: Path):
    from localharness.tools.builtin.edit_tool import EditTool

    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "f.txt"
    outside.write_text("aaa")
    tool = EditTool(workspace_root=str(root))
    result = await tool.run(path=str(outside), old_string="aaa", new_string="bbb")
    assert result.success is False
    assert result.error_type == "permission_denied"
    assert outside.read_text() == "aaa"  # untouched


@pytest.mark.asyncio
async def test_bash_working_dir_outside_workspace_root_denied(tmp_path: Path):
    from localharness.tools.builtin.bash_tool import BashExecTool

    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    tool = BashExecTool(workspace_root=str(root))
    result = await tool.run(command="echo hi", working_dir=str(outside))
    assert result.success is False
    assert result.error_type == "permission_denied"


@pytest.mark.asyncio
async def test_bash_working_dir_inside_workspace_root_runs(tmp_path: Path):
    from localharness.tools.builtin.bash_tool import BashExecTool

    root = tmp_path / "ws"
    root.mkdir()
    tool = BashExecTool(workspace_root=str(root))
    result = await tool.run(command="echo hi", working_dir=str(root))
    assert result.success is True
    assert "hi" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("call_kwargs", [{}, {"working_dir": "."}], ids=["omitted", "dot"])
async def test_confined_bash_default_working_dir_is_workspace_root(tmp_path: Path, call_kwargs):
    """Resting behavior: confined means 'your default cwd IS the workspace', not 'your default
    bash call errors'. An untouched working_dir (omitted or the schema default '.') must run
    with cwd == workspace_root, not be denied for resolving against the ambient harness CWD."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    root = tmp_path / "ws"
    root.mkdir()
    tool = BashExecTool(workspace_root=str(root))
    result = await tool.run(command=_PWD, **call_kwargs)
    assert result.success is True, f"default working_dir must run confined, got: {result.error}"
    assert Path(result.output.strip()) == root.resolve()


@pytest.mark.asyncio
async def test_confined_bash_relative_working_dir_anchors_at_root(tmp_path: Path):
    """Explicit relative working_dir resolves AGAINST workspace_root when confined
    (working_dir='sub' -> <root>/sub), not against the ambient harness CWD."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    tool = BashExecTool(workspace_root=str(root))
    result = await tool.run(command=_PWD, working_dir="sub")
    assert result.success is True, f"relative working_dir must anchor at root, got: {result.error}"
    assert Path(result.output.strip()) == (root / "sub").resolve()


@pytest.mark.asyncio
async def test_confined_bash_relative_escape_still_denied(tmp_path: Path):
    """A relative working_dir that escapes the root after resolve() ('../evil') stays denied."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "evil").mkdir()
    tool = BashExecTool(workspace_root=str(root))
    result = await tool.run(command="pwd", working_dir="../evil")
    assert result.success is False
    assert result.error_type == "permission_denied"


@pytest.mark.asyncio
async def test_symlink_escape_is_blocked(tmp_path: Path):
    """resolve() must defeat a symlink inside the workspace that points outside it."""
    from localharness.tools.builtin.write_tool import WriteTool

    root = tmp_path / "ws"
    root.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    try:
        (root / "link").symlink_to(secret)  # ws/link -> ../secret
    except OSError as exc:  # Windows: symlink creation needs elevation or Developer Mode
        pytest.skip(f"cannot create symlinks in this environment: {exc}")
    tool = WriteTool(workspace_root=str(root))
    result = await tool.run(path=str(root / "link" / "escaped.txt"), content="x")
    assert result.success is False
    assert result.error_type == "permission_denied"
    assert not (secret / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_write_unconfined_by_default(tmp_path: Path):
    """DEFAULT (workspace_root=None) MUST stay fully unconfined — writing anywhere the OS
    user can reach still works. File-write power is a core product capability (owner ruling)."""
    from localharness.tools.builtin.write_tool import WriteTool

    outside = tmp_path / "anywhere" / "f.txt"
    tool = WriteTool()  # no workspace_root
    result = await tool.run(path=str(outside), content="ok")
    assert result.success is True
    assert outside.read_text() == "ok"
