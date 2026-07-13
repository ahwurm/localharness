"""Cross-feature seam 2 — `components set` and /model persistence share one user overlay.

Both write the SAME overrides.yaml through the SAME atomic path (issue #22): `components set`
routes agent.* through AgentConfig and everything else through HarnessConfig; /model persistence
writes provider.*/org.*. The seam no unit test crosses is that they must not clobber each other's
slice, in EITHER order. This module proves both directions:
  A. `components set agent.*`  →  /model persist  →  the agent value survives beside the new default.
  B. /model persist            →  `components set` (harness axis)  →  the model choice survives.
"""
from __future__ import annotations

import asyncio

import yaml
from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import load_overlay

runner = CliRunner()

_AGENT_AXIS = "agent.memory.consolidation.chapter_containment_guard_enabled"
_HARNESS_AXIS = "org.context.compaction_threshold_pct"


def _write_config(home):
    data = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "default_model": "model-a",
            "available_models": ["model-a", "model-b"],
        },
        "org": {"default_model": "model-a", "audit_log_path": str(home / "audit.jsonl")},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _persist_model(home, model):
    """The exact persistence /model performs, off a harness loaded from the current cascade."""
    harness = ConfigLoader(config_dir=home).load_harness()
    asyncio.run(model_ops.persist_default_model(harness, model))


def test_agent_set_then_model_persist_preserves_both(components_home):
    """Direction A: an agent-scoped kill-lever set, then a persisted /model switch — the agent
    value stays put (persist touches only provider.*/org.*) and the new default lands beside it."""
    _write_config(components_home)

    r = runner.invoke(app, ["components", "set", _AGENT_AXIS, "false"])
    assert r.exit_code == 0, r.output

    _persist_model(components_home, "model-b")

    overlay = load_overlay(components_home / "overrides.yaml")
    # The #22 agent slice survived the persist untouched.
    assert overlay["agent"]["memory"]["consolidation"]["chapter_containment_guard_enabled"] is False
    # The persisted default is present.
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"

    # Round-trip through a fresh loader: both resolve as written.
    fresh = ConfigLoader(config_dir=components_home)
    assert fresh.load_harness().provider.default_model == "model-b"
    # A plain inheriting agent reads back the overlaid agent default (per-agent yaml would win).
    (components_home / "agents").mkdir(exist_ok=True)
    (components_home / "agents" / "plain.yaml").write_text(
        "name: plain\nrole: plain role\n", encoding="utf-8"
    )
    assert (
        fresh.load_agent("plain").memory.consolidation.chapter_containment_guard_enabled is False
    )


def test_model_persist_then_harness_set_preserves_both(components_home):
    """Direction B: a persisted /model switch, then a `components set` on a harness axis — the
    model choice survives the second write, and the harness axis lands beside it."""
    _write_config(components_home)

    _persist_model(components_home, "model-b")
    r = runner.invoke(app, ["components", "set", _HARNESS_AXIS, "85.0"])
    assert r.exit_code == 0, r.output

    overlay = load_overlay(components_home / "overrides.yaml")
    # The persisted default survived the components-set write.
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"
    # The harness axis was written.
    assert overlay["org"]["context"]["compaction_threshold_pct"] == 85.0

    fresh = ConfigLoader(config_dir=components_home).load_harness()
    assert fresh.provider.default_model == "model-b"
    assert fresh.org.context.compaction_threshold_pct == 85.0
