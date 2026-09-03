"""MEMS-04 — the audit log follows the WORK, and only the audit log does.

`persist_default_model` has to write two things to two different layers at once, and until phase
41 it had one parameter to say where:

* the **overlay** (`overrides.yaml`) is MACHINE-WIDE truth. There is one physical GPU daemon and
  `server.model` is persisted alongside the default, so two workspaces must never each believe they
  own it (critique amendment #2, the C2 invariant). It resolves through `global_config_dir()`.
* the **audit record** is an observability artifact of the work being done. In a workspace session
  it belongs with that workspace (MEMS-04).

Widening the single `config_dir` to "the workspace when there is one" would have moved BOTH, because
`global_config_dir(x)` returns whatever `x` it is handed — its "global" guarantee is a calling
DISCIPLINE, not a property of the function (41-RESEARCH Pitfall 2). So the split is a second,
independent parameter, `audit_base_dir`, defaulting to `config_dir` so every caller that does not
opt in is byte-identical.

This file proves the split at all three places it has to hold: the function itself, the real CLI
(including the `--config-dir` full-replacement case, where the audit must NOT wander out of the
named dir), and a REPL session carrying a workspace. `tests/unit/test_provider_carveout_workspace.py`
holds the other half — that a swap driven from a real workspace session still writes its OVERLAY
globally, asserted as an exact file set.
"""
from __future__ import annotations

import asyncio

import yaml
from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from localharness.config.models import HarnessConfig, ProviderConfig
from localharness.config.paths import discover_workspace_dir, workspace_is_within_repo

runner = CliRunner()

# Unreachable on purpose (port 9 is `discard`): with no server to ask, `localharness model` takes
# its documented degrade path and persists anyway, which is what makes an offline end-to-end write
# test possible at all. Same reasoning — and same constant — as test_provider_carveout_workspace.py.
_UNREACHABLE = "http://127.0.0.1:9/v1"

_CONFIG_YAML = (
    "version: '1'\n"
    "provider:\n"
    "  provider_type: vllm\n"
    f"  base_url: {_UNREACHABLE}\n"
    "  default_model: global-model\n"
    "  available_models:\n"
    "    - global-model\n"
    "    - other-model\n"
)


def _harness() -> HarnessConfig:
    """A minimal valid harness whose `org.audit_log_path` is the shipped default `audit.jsonl` —
    a BARE relative name, which is the whole reason `resolve_runtime_path`'s base dir decides
    which layer the record lands in."""
    harness = HarnessConfig(
        provider=ProviderConfig(
            provider_type="vllm",
            base_url=_UNREACHABLE,
            default_model="m1",
            available_models=["m1", "m2"],
        )
    )
    assert harness.org.audit_log_path == "audit.jsonl", (
        "this file's whole premise is a bare relative audit path; an absolute default would be "
        "honored as-is and no base dir could move it"
    )
    return harness


def test_persist_writes_the_audit_under_the_workspace_and_the_overlay_global(tmp_path):
    """The two directory inputs diverge: overlay in G, audit in W, and NOTHING crosses over.

    Both negative assertions are load-bearing and are the ones that would catch the failure mode
    this design exists to prevent. `not (W / "overrides.yaml").exists()` catches the C2 violation
    (the overlay following the work); `not (G / "audit.jsonl").exists()` catches the audit
    silently ignoring its new parameter — which a positive-only test would pass through, because
    the audit bus creates its parent dir either way.
    """
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    ws_dir = tmp_path / "proj" / ".localharness"
    ws_dir.mkdir(parents=True)

    warning = asyncio.run(
        model_ops.persist_default_model(
            _harness(), "m2", config_dir=global_dir, audit_base_dir=ws_dir
        )
    )
    assert warning is None, warning

    assert (global_dir / "overrides.yaml").exists(), "the overlay must land in the GLOBAL dir"
    assert (ws_dir / "audit.jsonl").exists(), "the audit record must follow the work"
    assert not (ws_dir / "overrides.yaml").exists(), (
        "the overlay followed the audit dir into the workspace — one GPU daemon, one server.model"
    )
    assert not (global_dir / "audit.jsonl").exists(), (
        "the audit record ignored audit_base_dir and stayed global"
    )


def test_explicit_config_dir_keeps_the_audit_inside_it(tmp_path, monkeypatch):
    """LAYR-02's full-replacement contract meeting MEMS-04: `--config-dir D` from INSIDE a
    workspace writes D's audit log, not the workspace's.

    `--config-dir` is an explicit selection, so `resolve_workspace_layer` short-circuits before any
    walk and `_ws` is None; the audit dir must then be D. Passing None onward instead would
    re-resolve through the env chain and land the audit log in whatever `~/.localharness` the
    environment happens to name — outside the dir the user explicitly named.

    TWO fallbacks stand between here and that leak, and they are redundant with each other:
    `model_cmd`'s `_audit_dir = _ws or loader._config_dir` and `persist_default_model`'s own
    `audit_base_dir if audit_base_dir is not None else config_dir`. Removing either ALONE leaves
    this test green (measured — the plan predicted otherwise); removing BOTH reddens it right
    here. That is the mutation this test is graded by, and it is why the surviving belt-and-braces
    line in `model_cmd` is pinned structurally rather than claimed to be behaviorally load-bearing.

    Driven through the real Typer app from a real workspace CWD, because the claim is about what a
    command does, not about what a helper returns.
    """
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(_CONFIG_YAML, encoding="utf-8")
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")

    # A real in-project workspace, and the CWD is inside it — so discovery WOULD find it if the
    # explicit flag did not short-circuit first. Without this the test passes vacuously.
    proj = home / "proj"
    ws_dir = proj / ".localharness"
    ws_dir.mkdir(parents=True)
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    assert discover_workspace_dir() == ws_dir.resolve(), "no workspace to be ignored"
    assert workspace_is_within_repo(ws_dir.resolve(), proj)

    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "config.yaml").write_text(_CONFIG_YAML, encoding="utf-8")

    result = runner.invoke(app, ["model", "other-model", "--config-dir", str(explicit)])
    assert result.exit_code == 0, result.output

    assert (explicit / "audit.jsonl").exists(), (
        "`--config-dir D` is a full replacement: D's audit log is the one that must be written"
    )
    persisted = yaml.safe_load((explicit / "overrides.yaml").read_text(encoding="utf-8"))
    assert persisted["provider"]["default_model"] == "other-model"

    strays = sorted(str(p) for p in proj.rglob("audit.jsonl"))
    assert not strays, f"an audit log leaked into the workspace despite --config-dir: {strays}"
    assert not (global_dir / "audit.jsonl").exists(), "an audit log leaked into the global layer"


class _StubChannel:
    """Records what the REPL said. `_persist_default_model` speaks only through `send_message`
    (directly on failure, via `_send_info` for the audit warning and the pin notes), so this is
    the entire surface it touches — no session, no LLM, no terminal."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_message(self, text: str, metadata: dict | None = None) -> None:
        self.messages.append(text)


def test_repl_session_audit_follows_its_workspace(tmp_path):
    """A live session's `/model` swap: the overlay lands in the session's config dir, the audit
    record in the session's workspace.

    The REPL is constructed DIRECTLY with `config_dir=G, workspace=W` — that is the shape 41-05
    wires from `start_cmd`, and the property under test (`_audit_base_dir`) reads only those two
    stored values. `_server_config_dir` is asserted alongside it as the contrast that gives the
    split its meaning: the same session, the same stored dirs, and the GPU daemon still resolves
    global.
    """
    from localharness.cli.repl import OrchestratorREPL

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    ws_dir = tmp_path / "proj" / ".localharness"
    ws_dir.mkdir(parents=True)

    harness = _harness()
    repl = OrchestratorREPL(
        orchestrator=None,
        agent_loop=None,
        channel=_StubChannel(),
        bus=None,
        config_dir=global_dir,
        workspace=ws_dir,
        harness_config=harness,
    )
    assert repl._audit_base_dir == ws_dir, "the session's audit base must be its workspace"
    assert repl._server_config_dir == global_dir, (
        "the GPU daemon's dir must NOT follow the workspace — same session, opposite answer"
    )

    assert asyncio.run(repl._persist_default_model("m2")) is True

    assert (global_dir / "overrides.yaml").exists()
    assert (ws_dir / "audit.jsonl").exists()
    assert not (ws_dir / "overrides.yaml").exists(), "the session forked the one GPU daemon"
    assert not (global_dir / "audit.jsonl").exists(), "the session's audit did not follow its work"


def test_repl_without_a_workspace_audits_where_it_always_did(tmp_path):
    """The no-workspace session is byte-identical to pre-MEMS-04: both writes land in the config
    dir. `workspace` defaults to None at all 42 existing construction sites, so this is what every
    caller that has not been taught about workspaces still gets (LAYR-03)."""
    from localharness.cli.repl import OrchestratorREPL

    global_dir = tmp_path / "global"
    global_dir.mkdir()

    repl = OrchestratorREPL(
        orchestrator=None,
        agent_loop=None,
        channel=_StubChannel(),
        bus=None,
        config_dir=global_dir,
        harness_config=_harness(),
    )
    assert repl._audit_base_dir == global_dir, "no workspace → the audit base is the config dir"

    assert asyncio.run(repl._persist_default_model("m2")) is True
    assert (global_dir / "overrides.yaml").exists()
    assert (global_dir / "audit.jsonl").exists()


def test_pinned_agents_sees_workspace_agents(tmp_path):
    """39-05's carve-out, closed: the pin warning reads through the workspace layer.

    A workspace agent pinning a concrete `model:` traps a persisted switch exactly like a global
    one does, so warning about only half the roster is worse than not warning — the user is told
    the switch is complete when it is not. `pinned_agents(G)` (no layer named) must still report
    only the global roster, which is what keeps every non-workspace caller unchanged.
    """
    global_dir = tmp_path / "global"
    (global_dir / "agents").mkdir(parents=True)
    ws_dir = tmp_path / "proj" / ".localharness"
    (ws_dir / "agents").mkdir(parents=True)

    (ws_dir / "agents" / "ws-pinned.yaml").write_text(
        yaml.dump(
            {
                "name": "ws-pinned",
                "role": "a workspace agent that pins its own model",
                "model": "pinned-ws-model",
            }
        ),
        encoding="utf-8",
    )

    with_ws = model_ops.pinned_agents(global_dir, local_config_dir=ws_dir)
    assert ("ws-pinned", "pinned-ws-model") in with_ws, (
        f"a workspace agent's model pin was invisible to the switch warning: {with_ws}"
    )

    without_ws = model_ops.pinned_agents(global_dir)
    assert without_ws == [], (
        f"omitting local_config_dir must keep today's global-only roster, got {without_ws}"
    )
