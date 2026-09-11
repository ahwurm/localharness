"""The per-tool confinement leash acts ONLY when a human configured it (v0.14.1).

A live end-to-end run against the real model, in `auto` mode, produced this pair in one session:

    bash_exec("cat > /tmp/lh-e2e-notes/notes.md")   ran
    write(path="/tmp/lh-e2e-notes/notes.md")        permission_denied

Same file, same session, opposite answers. The refusal came from `tools/base.Tool.
_outside_workspace`, a second boundary that predates the gate, switched on by the loader's step
5c — which auto-filled `permissions.workspace_root` with the project folder whenever a workspace
layer applied. Nobody had asked for it; the shell tool's own command string never reached it; and
`auto`'s contract is that a write outside the project runs unless it lands somewhere protected.

So 5c is gone. The gate owns the boundary: it DERIVES it from where you stand, applies it to
every tool the same way, and is the one thing a mode can talk about. An EXPLICIT
`permissions.workspace_root` is untouched and is still a hard confinement — no prompt, no mode
that overrides it — which is what harness-run evals set.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader

_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "m",
    },
}


def _layers(tmp_path: Path, agent: dict) -> tuple[Path, Path]:
    global_dir = tmp_path / "global"
    (global_dir / "agents").mkdir(parents=True)
    (global_dir / "config.yaml").write_text(yaml.dump(_MINIMAL), encoding="utf-8")
    (global_dir / "agents" / "worker.yaml").write_text(yaml.dump(agent), encoding="utf-8")
    workspace = tmp_path / "proj" / ".localharness"
    workspace.mkdir(parents=True)
    return global_dir, workspace


def _permissions(global_dir: Path, workspace: Path):
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace)
    return loader.load_agent("worker").permissions


def test_a_workspace_layer_no_longer_invents_a_confinement(tmp_path):
    """Step 5c's auto-fill is gone. UNCONFINED is what `workspace_root: None` has always meant
    (`config/models.PermissionConfig`), and it is now what a workspace session actually gets."""
    global_dir, workspace = _layers(tmp_path, {"name": "worker", "role": "Work"})
    assert _permissions(global_dir, workspace).workspace_root is None


def test_an_explicitly_configured_root_still_confines(tmp_path):
    """The knob a human writes down is untouched — evals depend on it, and it is the only way
    the per-tool leash switches on now."""
    root = tmp_path / "scratch"
    global_dir, workspace = _layers(tmp_path, {
        "name": "worker", "role": "Work", "permissions": {"workspace_root": str(root)},
    })
    assert _permissions(global_dir, workspace).workspace_root == str(root)


@pytest.mark.asyncio
async def test_a_write_outside_the_project_is_not_refused_by_the_tool(tmp_path):
    """The end-to-end symptom, at the seam that produced it: with no configured root, the write
    tool has no opinion about where the file goes. Whether it SHOULD go there is the gate's
    question, asked once, in one place, for every tool."""
    from localharness.tools.builtin.write_tool import WriteTool

    outside = tmp_path / "elsewhere" / "notes.md"
    outside.parent.mkdir(parents=True)
    result = await WriteTool(workspace_root=None).run(path=str(outside), content="hi\n")

    assert result.success, result.error
    assert result.error_type is None
    assert outside.read_text(encoding="utf-8") == "hi\n"


@pytest.mark.asyncio
async def test_a_configured_root_still_blocks_the_same_write(tmp_path):
    """The other half, so the removal above cannot be read as "confinement stopped working"."""
    from localharness.tools.builtin.write_tool import WriteTool

    project = tmp_path / "proj"
    project.mkdir()
    outside = tmp_path / "elsewhere" / "notes.md"
    outside.parent.mkdir(parents=True)
    result = await WriteTool(workspace_root=str(project)).run(path=str(outside), content="hi\n")

    assert not result.success
    assert result.error_type == "permission_denied"
    assert "workspace_root" in (result.error or "")
    assert not outside.exists()
