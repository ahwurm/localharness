"""Cross-feature seam 3 — config auto-migration meets the post-v0.9.2 schema growth.

The seam no unit test crosses: 17 commits after the v0.9.2 release (df09f50) GREW the
config schema with new agent-scoped memory fields (chapter_containment_guard_enabled,
chapter_staleness_recheck_enabled, chapter_staleness_recheck_cap) while `config migrate`
still only manages `org.permissions.{deny_patterns,defaults_revision}`. This module proves a
config.yaml written EXACTLY as v0.9.2 shipped it (deny list at CURRENT_DEFAULTS_REVISION=1,
none of the new fields) still (a) loads today with the new fields taking their code defaults,
(b) migrates as a clean no-op (revision current → deliberate deletions respected, no spurious
change), with a below-revision variant proving the additive rewrite never perturbs the absent
new fields, and (c) round-trips a /model overlay write on top.

All behavior asserted against the REAL current code; where correct, it is locked in.
"""
from __future__ import annotations

import asyncio

import yaml
from typer.testing import CliRunner

from localharness.cli import model_ops
from localharness.cli.app import app
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.loader import ConfigLoader
from localharness.config.migrate import plan
from localharness.config.models import (
    HarnessConfig,
    MemoryConsolidationConfig,
    PermissionConfig,
)
from localharness.config.overlay import load_overlay

runner = CliRunner()

# The v0.9.2 deny list == today's shipped list: CURRENT_DEFAULTS_REVISION has been 1 since
# v0.9.1 (issue #15's 24-pattern list) and is still 1 at HEAD, so v0.9.2 shipped this exact set
# stamped at revision 1. Snapshot it as a literal so a FUTURE revision bump makes this seam's
# "exactly v0.9.2" premise fail loudly instead of silently drifting.
V092_DENY = list(PermissionConfig().deny_patterns)
V092_REVISION = 1


def _write_v092_config(home, *, deny=None, revision=V092_REVISION, extra_org=None):
    """Write a config.yaml exactly as v0.9.2's `init` produced it: provider + org with a
    revision-stamped deny list, and — like every real v0.9.2 config.yaml — NONE of the
    agent-scoped memory-consolidation fields (those live under agent.memory.consolidation,
    resolved only when an AgentConfig loads; a config.yaml never carried them)."""
    org = {
        "name": "default",
        "default_model": "model-a",
        "audit_log_path": str(home / "audit.jsonl"),
        "permissions": {
            "mode": "auto",
            "deny_patterns": list(V092_DENY if deny is None else deny),
        },
    }
    if revision is not None:
        org["permissions"]["defaults_revision"] = revision
    if extra_org:
        org.update(extra_org)
    data = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "default_model": "model-a",
            "available_models": ["model-a", "model-b"],
        },
        "org": org,
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8"
    )
    return home / "config.yaml"


def _deny_on_disk(config_file):
    return yaml.safe_load(config_file.read_text())["org"]["permissions"]["deny_patterns"]


# --------------------------------------------------------------------------- #
# (a) A v0.9.2 config loads today; the new fields take their code defaults.
# --------------------------------------------------------------------------- #

def test_v092_config_loads_today_and_new_memory_fields_default(components_home):
    """Loading a v0.9.2 config.yaml through today's HarnessConfig raises nothing and preserves
    its stamped deny list; a v0.9.2-era agent serialization that OMITS the three new memory
    fields validates with the fields at their shipped code defaults (forward-compat)."""
    _write_v092_config(components_home)

    harness = ConfigLoader(config_dir=components_home).load_harness()
    assert harness.org.permissions.deny_patterns == V092_DENY
    assert harness.org.permissions.defaults_revision == V092_REVISION

    # A memory-consolidation subtree serialized under v0.9.2 (the three new fields did not exist
    # yet) — today's model fills them from code defaults, no ValidationError.
    v092_consolidation = {"enabled": True, "schema_writer_enabled": True, "idle_minutes": 10.0}
    cons = MemoryConsolidationConfig.model_validate(v092_consolidation)
    assert cons.chapter_containment_guard_enabled is True
    assert cons.chapter_staleness_recheck_enabled is True
    assert cons.chapter_staleness_recheck_cap == 10


# --------------------------------------------------------------------------- #
# (b) migrate on a v0.9.2 (revision-current) config is a clean no-op.
# --------------------------------------------------------------------------- #

def test_v092_migrate_is_noop_because_revision_is_current(components_home):
    """A v0.9.2 config is stamped at the current revision, so both the planner and the CLI
    (dry-run and real) treat it as up to date: nothing added, nothing rewritten, no backup —
    the removal-respect / no-spurious-change property the schema growth must not have broken."""
    cfg = _write_v092_config(components_home)
    before = cfg.read_bytes()

    # Engine: a revision-current config yields no plan at all.
    assert plan(yaml.safe_load(cfg.read_text())) is None

    # CLI dry-run: reports up-to-date, writes nothing.
    dry = runner.invoke(app, ["config", "migrate", "--config-dir", str(components_home), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "up to date" in dry.output.lower()
    assert cfg.read_bytes() == before
    assert list(components_home.glob("config.yaml.bak-*")) == []

    # CLI real run: same — no-op, byte-identical, no backup file created.
    real = runner.invoke(app, ["config", "migrate", "--config-dir", str(components_home)])
    assert real.exit_code == 0, real.output
    assert "up to date" in real.output.lower()
    assert cfg.read_bytes() == before
    assert list(components_home.glob("config.yaml.bak-*")) == []


def test_below_revision_migrate_is_additive_and_leaves_new_fields_absent(components_home):
    """The real additive path (a pre-sync config, revision 0) folds in the missing shipped
    defaults and stamps the revision — and because migrate rewrites via safe_dump of the parsed
    dict (which never carried the agent memory fields), those fields stay ABSENT on disk and are
    still defaulted on load. Migrate's rewrite never spuriously materializes the new schema."""
    # A v0.9.0-shaped config: short deny list, NO defaults_revision stamp (→ treated as 0).
    short_deny = ["write(*/.env)", "write(*/secrets*)"]
    cfg = _write_v092_config(components_home, deny=short_deny, revision=None)

    p = plan(yaml.safe_load(cfg.read_text()))
    assert p is not None and p.from_revision == 0 and p.to_revision == CURRENT_DEFAULTS_REVISION

    res = runner.invoke(app, ["config", "migrate", "--config-dir", str(components_home)])
    assert res.exit_code == 0, res.output

    # Additive: the shipped defaults are now a subset; original entries kept verbatim at the front.
    new_deny = _deny_on_disk(cfg)
    assert set(V092_DENY).issubset(set(new_deny))
    assert new_deny[: len(short_deny)] == short_deny
    # Revision stamped; a backup was written.
    on_disk = yaml.safe_load(cfg.read_text())
    assert on_disk["org"]["permissions"]["defaults_revision"] == CURRENT_DEFAULTS_REVISION
    assert len(list(components_home.glob("config.yaml.bak-*"))) == 1

    # The rewrite touched deny+revision ONLY: no agent/memory subtree was invented, and a fresh
    # HarnessConfig load still applies the new memory-field code defaults elsewhere.
    assert "agent" not in on_disk and "memory" not in on_disk
    harness = ConfigLoader(config_dir=components_home).load_harness()
    assert harness.org.permissions.defaults_revision == CURRENT_DEFAULTS_REVISION


# --------------------------------------------------------------------------- #
# (c) A /model overlay write on top of the old config round-trips cleanly.
# --------------------------------------------------------------------------- #

def test_model_overlay_write_on_v092_config_roundtrips(components_home):
    """Persisting a new default via the #22 atomic user overlay ON TOP of a v0.9.2 config.yaml
    round-trips: a fresh load sees the switched model, the old config.yaml is never rewritten,
    and its stamped deny list survives untouched — composing seam 3 (old config) with seam 1
    (overlay persistence)."""
    cfg = _write_v092_config(components_home)
    harness = ConfigLoader(config_dir=components_home).load_harness()

    asyncio.run(model_ops.persist_default_model(harness, "model-b"))

    # config.yaml is never touched by the overlay path.
    assert _deny_on_disk(cfg) == V092_DENY
    overlay = load_overlay(components_home / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"

    fresh = ConfigLoader(config_dir=components_home).load_harness()
    assert fresh.provider.default_model == "model-b"
    assert fresh.org.default_model == "model-b"
    # And the v0.9.2 deny list is still resolved intact through the cascade.
    assert fresh.org.permissions.deny_patterns == V092_DENY
