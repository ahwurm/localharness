"""MERG-03 — the provider is HARDWARE truth, so nothing writes a provider block to a workspace.

Two halves, both of them carve-outs that say "this does NOT follow the workspace":

* A workspace MAY override `provider:` explicitly in its own config.yaml, and that is fine — the
  read side is proved in `tests/unit/test_config_merge_four_source.py`. What must never happen is
  a WRITE: `localharness model <name>` run from inside a project folder must land in the global
  `~/.localharness/overrides.yaml`, never in `<workspace>/overrides.yaml`. There is ONE physical
  GPU daemon on the machine; two workspaces must never each believe they own its model, because
  `server.model` is persisted alongside the default and the next cold start rebuilds
  `vllm serve <model>` from it (critique amendment #2, BLOCKER-level).
* `init` is the only command that SCAFFOLDS a `provider:` block, and it cannot address a workspace
  at all — proved here by a source scan rather than by a comment.

The write-side pin itself landed in 38-03 (`model_ops` names `global_config_dir(config_dir)`); what
this file adds is the missing proof: a swap driven from a real workspace session through the real
Typer app, with the files that changed counted before and after.

Phase 41 (MEMS-04) amended the measured set: the swap's AUDIT record now lands in the workspace,
because the log follows the work. The OVERLAY half did not move and never will. The distinction is
the whole point of the amendment — see
`test_model_swap_in_a_workspace_session_writes_only_the_global_overlay`'s docstring.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from localharness.config.paths import discover_workspace_dir, workspace_is_within_repo

runner = CliRunner()

# The repo root, derived from THIS file so the source scans below cannot silently scan zero files
# no matter what CWD pytest was launched from (39-01's rule).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# An unreachable base_url is load-bearing, not laziness: with no server to ask, `localharness model`
# takes its documented degrade path ("persisting it as the default UNVERIFIED") and persists anyway,
# which is what makes an offline end-to-end write test possible at all. Port 9 is `discard`.
_UNREACHABLE = "http://127.0.0.1:9/v1"

_GLOBAL_CONFIG_YAML = (
    "version: '1'\n"
    "provider:\n"
    "  provider_type: vllm\n"
    f"  base_url: {_UNREACHABLE}\n"
    "  default_model: global-model\n"
    "  available_models:\n"
    "    - global-model\n"
    "    - other-model\n"
)


# --------------------------------------------------------------------------------- helpers


def _hermetic(monkeypatch, home: Path) -> Path:
    """A fake `$HOME` holding the GLOBAL layer, with both env overrides cleared.

    All three moves are required. Both walks stop at `$HOME`, so a real one would let the
    developer's own `~/.localharness` answer these tests; and the conftest fixtures set
    `LOCALHARNESS_HOME`, which the resolver counts as an EXPLICIT selection — leaving it set would
    switch discovery off and every test below would pass for the wrong reason (there would be no
    workspace session to carve out of). Copied from
    `tests/integration/test_workspace_discovery_e2e.py` rather than imported, so this file stays
    readable as the whole of what the global layer contains.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True, exist_ok=True)
    (global_dir / "config.yaml").write_text(_GLOBAL_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    return global_dir


def _workspace_session(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """A real workspace session: fake `$HOME`, a project inside it, CWD in the project.

    The `.git` marker makes `proj/` the project the caller is standing in, so the workspace is
    IN-project and loads with no trust prompt — an unanswered prompt would block the runner and a
    recorded verdict would put a third file in the diff. The project lives UNDER the fake `$HOME`
    on purpose: it puts the global layer and the workspace in one `rglob` snapshot, so "which files
    did the swap touch" is one question with one answer.

    Returns (home, global_dir, workspace_dir).
    """
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, home)

    proj = home / "proj"
    ws_dir = proj / ".localharness"
    ws_dir.mkdir(parents=True)
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)

    def _boom(*args, **kwargs):
        raise AssertionError("prompted for trust on an IN-PROJECT workspace, which must not ask")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    # Guards: if either flipped, every assertion below would pass for the wrong reason — a session
    # with no workspace cannot prove that the write avoided one.
    assert discover_workspace_dir() == ws_dir.resolve()
    assert workspace_is_within_repo(ws_dir.resolve(), proj)
    return home, global_dir, ws_dir


def _file_snapshot(root: Path) -> dict[Path, tuple[int, int]]:
    """Every file under `root`, keyed by path, valued by (size, mtime_ns).

    `st_mtime_ns` rather than `st_mtime`: a swap finishes in milliseconds, and second-resolution
    mtimes would report an in-place rewrite as unchanged.
    """
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def _changed(before: dict, after: dict) -> set[Path]:
    """Paths created or modified between the two snapshots. Deletions are not expected here and
    would show up as a missing member of the exact-set assertion."""
    return {p for p, v in after.items() if before.get(p) != v} | (set(before) - set(after))


# ------------------------------------------------------------------- the end-to-end write test


def test_model_swap_in_a_workspace_session_writes_only_the_global_overlay(tmp_path, monkeypatch):
    """`localharness model <name>` from inside a workspace writes the GLOBAL overlay, and the
    workspace directory is not touched at all.

    Asserted as an exact before/after file diff over the whole fake `$HOME` (which contains BOTH
    the global layer and the project), so any stray write anywhere reddens this test.

    MEASURED FINDING — the requirement's own phrase is "exactly one file", and that is not what a
    swap does. It touches TWO files:
      1. `<global>/overrides.yaml` — the durable persist itself;
      2. `audit.jsonl`             — `org.audit_log_path` defaults to that bare relative name,
         which `resolve_runtime_path` resolves under whichever base dir it is given, and
         `persist_default_model` publishes one `ComponentMutated` per written path to it.

    AMENDED BY PHASE 41 (MEMS-04). When 40-03 measured this set, BOTH files were global. Phase 41
    moved the audit half to the WORKSPACE deliberately — the audit log follows the work — while the
    overlay half stays global forever, because there is one physical GPU daemon and a
    workspace-local `server.model` would fork it. So the expected set is now
    `{<global>/overrides.yaml, <workspace>/audit.jsonl}`.

    The exact-set DISCIPLINE is unchanged, and this amendment is not a weakening: the assertion is
    still an exact SET over an mtime snapshot of the whole fake `$HOME`, so a third file appearing
    anywhere, in either layer, still reddens it — and the blanket "nothing under the workspace may
    change" line below was REPLACED by a precise successor naming the one file that may, not
    deleted. Mutation-proven: pointing the overlay write at `audit_base_dir` (the C2 violation this
    test exists to catch) still fails it.
    """
    home, global_dir, ws_dir = _workspace_session(tmp_path, monkeypatch)

    before = _file_snapshot(home)
    result = runner.invoke(app, ["model", "other-model"])
    assert result.exit_code == 0, result.output
    after = _file_snapshot(home)

    overrides = global_dir / "overrides.yaml"
    assert overrides.exists(), "the global overlay was never written"
    persisted = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    assert persisted["provider"]["default_model"] == "other-model"

    # The workspace holds no config of its own, before or after.
    assert not (ws_dir / "overrides.yaml").exists()
    assert not (ws_dir / "config.yaml").exists()

    changed = _changed(before, after)
    assert changed == {overrides, ws_dir / "audit.jsonl"}, (
        "a model swap must touch the global overlay (+ the audit log, which follows the work) and "
        f"NOTHING else; changed={sorted(str(p) for p in changed)}"
    )
    ws_changed = {p for p in changed if p == ws_dir or ws_dir in p.parents}
    assert ws_changed == {ws_dir / "audit.jsonl"}, (
        "the audit record follows the work (MEMS-04) and is the ONLY thing a swap may write into a "
        "workspace; an overlay or config file here would fork the one physical GPU daemon's "
        f"server.model. changed under the workspace={sorted(str(p) for p in ws_changed)}"
    )


def test_model_swap_from_a_workspace_does_not_create_a_workspace_overlay(tmp_path, monkeypatch):
    """A workspace may STATE a provider preference; a swap still edits the global file.

    The workspace's own `config.yaml` declares `provider.default_model: ws-model`. That is a legal
    read-side override. The write side must not follow it: the workspace config is byte-identical
    afterwards and no `<workspace>/overrides.yaml` is minted.
    """
    home, global_dir, ws_dir = _workspace_session(tmp_path, monkeypatch)
    ws_config = ws_dir / "config.yaml"
    ws_config.write_text(
        yaml.dump({"provider": {"default_model": "ws-model"}}), encoding="utf-8"
    )
    before_bytes = ws_config.read_bytes()

    result = runner.invoke(app, ["model", "other-model"])
    assert result.exit_code == 0, result.output

    assert ws_config.read_bytes() == before_bytes, "the workspace config.yaml was rewritten"
    assert not (ws_dir / "overrides.yaml").exists(), "a workspace overlay was minted"
    persisted = yaml.safe_load((global_dir / "overrides.yaml").read_text(encoding="utf-8"))
    assert persisted["provider"]["default_model"] == "other-model"


# --------------------------------------------------------------------------- the argument pins


def test_persist_default_model_targets_the_global_overlay_given_a_workspace_dir(tmp_path):
    """`persist_default_model` writes under the dir it was GIVEN, with a workspace present on disk.

    The unit-level half of the same guarantee: the function resolves its overlay through
    `global_config_dir(config_dir)`, so the workspace sitting right there is not a candidate.

    Since phase 41 this is ALSO the default-preserving proof for `audit_base_dir`: no audit dir is
    passed here, so the audit log must still resolve under `config_dir` exactly as it did before
    MEMS-04 — every caller that does not opt in is byte-identical.
    """
    from localharness.config.models import HarnessConfig, ProviderConfig

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    ws_dir = tmp_path / "proj" / ".localharness"
    ws_dir.mkdir(parents=True)

    harness = HarnessConfig(
        provider=ProviderConfig(
            provider_type="vllm",
            base_url=_UNREACHABLE,
            default_model="m1",
            available_models=["m1", "m2"],
        )
    )
    warning = asyncio.run(model_ops.persist_default_model(harness, "m2", config_dir=global_dir))
    assert warning is None, warning

    assert (global_dir / "overrides.yaml").exists()
    assert not (ws_dir / "overrides.yaml").exists()
    assert harness.provider.default_model == "m2"


def test_model_cmd_passes_the_global_config_dir_to_persist():
    """STRUCTURAL pin on the single argument the whole MERG-03 write guarantee rests on.

    `loader._config_dir` and `loader._local_dir` are two DIFFERENT directories in a workspace
    session, and only the former is machine-wide truth. Pinned structurally because no behavioral
    test can tell the two apart at the callee: `persist_default_model` funnels whatever it is given
    through `global_config_dir()`, which is value-identical to `resolve_config_dir()` today — so
    the caller is where the choice is actually made, and where a regression would land.

    THE SUBSTRING TRAP (phase 41, recorded because it shaped the production naming). This guard is
    a plain string scan; it does not respect Python token boundaries. So a keyword argument whose
    name merely ENDS in `config_dir`, bound to the workspace attribute, tail-matches the forbidden
    literal and reddens this test even though the code is behaviorally correct. That is a FALSE
    trip, and it was demonstrated by mutation (d) of plan 41-04: rename the audit parameter to one
    ending in `config_dir` and pass the workspace inline, and this test fails while every
    behavioral test stays green. The production code therefore (a) names the parameter
    `audit_base_dir` and (b) binds the workspace to named locals `_ws` / `_audit_dir` in
    `model_cmd.py` instead of writing the expression inline at the call sites. Both halves are
    pinned positively below, so the naming cannot be "simplified" back into the trap.
    """
    source_path = _REPO_ROOT / "src" / "localharness" / "cli" / "model_cmd.py"
    source = source_path.read_text(encoding="utf-8")
    assert source.strip(), f"scanned nothing at {source_path}"

    assert "config_dir=loader._config_dir" in source, (
        "model_cmd must hand persist_default_model the GLOBAL config dir"
    )
    assert "config_dir=loader._local_dir" not in source, (
        "a workspace layer must never become the persist target — one GPU daemon, one server.model"
    )
    assert "audit_base_dir=_audit_dir" in source, (
        "model_cmd must thread the audit dir separately — one parameter cannot serve both the "
        "global overlay target and a workspace-following audit log"
    )
    assert "_audit_dir = _ws or loader._config_dir" in source, (
        "the audit fallback must be the session's config dir, not None: passing None would "
        "re-resolve through the env chain and leak `--config-dir D`'s audit log out of D"
    )


def test_init_never_reaches_a_workspace_layer():
    """`init` is the only command that scaffolds a `provider:` block, and it cannot name a workspace.

    A source scan, because the claim is about what the file CANNOT do: none of the four workspace
    -addressing symbols appear in it, so there is no path by which guided setup could write a
    provider block (or a `server:` block) into a project folder.
    """
    source_path = _REPO_ROOT / "src" / "localharness" / "cli" / "init_cmd.py"
    source = source_path.read_text(encoding="utf-8")
    assert source.strip(), f"scanned nothing at {source_path}"

    forbidden = [
        "resolve_workspace_layer",
        "discover_workspace_dir",
        "local_config_dir",
        "_local_dir",
    ]
    found = [sym for sym in forbidden if sym in source]
    assert not found, (
        f"init_cmd.py names workspace-addressing symbol(s) {found} — init writes the config that "
        "declares the provider and the managed server, and both are machine-wide truth"
    )
