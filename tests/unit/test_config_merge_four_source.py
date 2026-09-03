"""The four-source config merge, pinned in the owner's ruled order (2026-09-03, Option A).

Order: global `config.yaml` < global `overrides.yaml` < workspace `config.yaml` <
workspace `overrides.yaml`.

the SPECIFIC beats the GENERAL — the workspace's word wins wherever the two conflict, and the
global layer still governs everything the workspace is silent about.

One key is carved out of that rule: `org.permissions.deny_patterns` UNIONS across the layers
instead of being replaced, because safety accumulates and a workspace must never be able to
subtract a denial the global layer imposed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader, ConfigNotFoundError
from localharness.config.overlay import deep_merge


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    """The two config bases: a global dir (already holding a minimal valid `config.yaml`) and a
    workspace `.localharness/` under a project. Every workspace file is optional."""
    global_dir = tmp_path / "global"
    workspace_dir = tmp_path / "proj" / ".localharness"
    workspace_dir.mkdir(parents=True)
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    return global_dir, workspace_dir


# ---------------------------------------------------------------------------
# The ruled order
# ---------------------------------------------------------------------------

def test_ruled_example_workspace_config_beats_global_overrides(layers) -> None:
    """THE pinned acceptance test for the phase.

    The owner's ruling, verbatim: the global `overrides.yaml` sets
    `context.compaction_threshold: 0.75`, the workspace `config.yaml` says `0.85`, and the
    effective value is **0.85**. Source 3 beats source 2 — that is what makes this Option A
    rather than "the user's own overrides always win".

    Honest note on the literals: the ruling's own KEY cannot carry the ruling's own NUMBERS.
    `ContextConfig.compaction_threshold_pct` is a PERCENT (`ge=50.0, le=99.0`), so `0.75` fails
    schema validation outright. Both halves of the ruling are therefore asserted separately —
    the KEY in its schema-valid form (75.0 vs 85.0), and the literal NUMBERS on
    `OrgConfig.default_temperature`, whose range (`ge=0.0, le=2.0`) accepts them.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "overrides.yaml",
        {"org": {"context": {"compaction_threshold_pct": 75.0}, "default_temperature": 0.75}},
    )
    _write_yaml(
        ws / "config.yaml",
        {"org": {"context": {"compaction_threshold_pct": 85.0}, "default_temperature": 0.85}},
    )

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.org.context.compaction_threshold_pct == 85.0, (
        "Option A: the workspace config.yaml (source 3) must beat the global overrides.yaml "
        "(source 2) — expected the workspace's 85.0, got "
        f"{cfg.org.context.compaction_threshold_pct}"
    )
    assert cfg.org.default_temperature == 0.85, (
        "Option A, on the ruling's literal numbers: the workspace config.yaml (source 3) said "
        f"0.85 and the global overrides.yaml (source 2) said 0.75 — got "
        f"{cfg.org.default_temperature}"
    )


def test_workspace_overrides_beats_workspace_config(layers) -> None:
    """Source 4 over source 3: a workspace's own overrides are its most specific word."""
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"org": {"default_temperature": 0.30}})
    _write_yaml(ws / "overrides.yaml", {"org": {"default_temperature": 0.90}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.org.default_temperature == 0.90


def test_workspace_config_beats_global_config(layers) -> None:
    """Source 3 over source 1."""
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"provider": {"default_model": "workspace-model"}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.provider.default_model == "workspace-model"


def test_global_only_keys_still_apply_in_a_workspace_session(layers) -> None:
    """The global layer governs everything the workspace is silent about."""
    global_dir, ws = layers
    _write_yaml(global_dir / "config.yaml", {**_MINIMAL, "org": {"log_level": "debug"}})
    _write_yaml(ws / "config.yaml", {"org": {"default_temperature": 0.42}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.org.log_level == "debug"
    assert cfg.org.default_temperature == 0.42


def test_all_four_sources_contribute_distinct_keys(layers) -> None:
    """One key per source, all four present in the one result."""
    global_dir, ws = layers
    _write_yaml(global_dir / "config.yaml", {**_MINIMAL, "org": {"log_level": "warning"}})
    _write_yaml(global_dir / "overrides.yaml", {"org": {"default_max_tokens": 1234}})
    _write_yaml(ws / "config.yaml", {"org": {"name": "workspace-org"}})
    _write_yaml(ws / "overrides.yaml", {"org": {"default_temperature": 1.5}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.org.log_level == "warning"          # source 1
    assert cfg.org.default_max_tokens == 1234      # source 2
    assert cfg.org.name == "workspace-org"         # source 3
    assert cfg.org.default_temperature == 1.5      # source 4


# ---------------------------------------------------------------------------
# Which files are required
# ---------------------------------------------------------------------------

def test_workspace_files_are_optional(layers) -> None:
    """A workspace dir holding neither file is a no-op, not an error."""
    global_dir, ws = layers
    _write_yaml(global_dir / "overrides.yaml", {"org": {"log_level": "error"}})
    assert not (ws / "config.yaml").exists() and not (ws / "overrides.yaml").exists()

    layered = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()
    bare = ConfigLoader(config_dir=global_dir).load_harness()

    assert layered.model_dump() == bare.model_dump()


def test_missing_global_config_still_raises(layers) -> None:
    """A workspace never satisfies the one required file: the global `config.yaml`."""
    global_dir, ws = layers
    (global_dir / "config.yaml").unlink()
    _write_yaml(ws / "config.yaml", _MINIMAL)

    with pytest.raises(ConfigNotFoundError):
        ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()


def test_agent_section_is_excluded_from_all_four_sources(layers) -> None:
    """`agent:` is an agent-scope default layer (issue #22), not a HarnessConfig field.

    `HarnessConfig` is `extra="forbid"`, so a leaked `agent:` from EITHER overlay is a validation
    error. The sibling keys are asserted too, so a bug that skipped the overlays entirely could
    not pass this test by simply never leaking anything.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "overrides.yaml",
        {"agent": {"temperature": 0.1}, "org": {"log_level": "error"}},
    )
    _write_yaml(
        ws / "overrides.yaml",
        {"agent": {"temperature": 0.2}, "org": {"default_max_tokens": 999}},
    )

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.org.log_level == "error"
    assert cfg.org.default_max_tokens == 999


# ---------------------------------------------------------------------------
# The deny_patterns carve-out (MERG-02)
# ---------------------------------------------------------------------------

def test_workspace_deny_patterns_add_to_the_global_list(layers) -> None:
    """Neither pattern is one of the shipped defaults, so this cannot pass by accident."""
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"permissions": {"deny_patterns": ["bash_exec(*curl *)"]}}},
    )
    _write_yaml(
        ws / "config.yaml",
        {"org": {"permissions": {"deny_patterns": ["bash_exec(*wget *)"]}}},
    )

    deny = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()
    deny = deny.org.permissions.deny_patterns

    assert "bash_exec(*curl *)" in deny, "a workspace declaration must never subtract a global deny"
    assert "bash_exec(*wget *)" in deny, "the workspace's own deny must be enforced too"


def test_org_deny_union_is_order_preserving_and_deduplicating(layers) -> None:
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"permissions": {"deny_patterns": ["a_tool(x)", "b_tool(y)"]}}},
    )
    _write_yaml(
        ws / "config.yaml",
        {"org": {"permissions": {"deny_patterns": ["b_tool(y)", "c_tool(z)"]}}},
    )

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.org.permissions.deny_patterns == ["a_tool(x)", "b_tool(y)", "c_tool(z)"]


def test_workspace_declaring_an_empty_deny_list_removes_nothing(layers) -> None:
    """`deny_patterns: []` contributes nothing and removes nothing. That is the contract."""
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"permissions": {"deny_patterns": ["bash_exec(*curl *)"]}}},
    )
    _write_yaml(ws / "config.yaml", {"org": {"permissions": {"deny_patterns": []}}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert "bash_exec(*curl *)" in cfg.org.permissions.deny_patterns


def test_no_deny_declared_anywhere_keeps_the_shipped_defaults(layers) -> None:
    """THE fail-open guard.

    With nothing declared the union is empty, and splicing an empty list would REPLACE
    `PermissionConfig`'s shipped security defaults with nothing. Delete the `if union_deny:`
    guard in `load_harness` and this test goes red.
    """
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"org": {"default_temperature": 0.5}})

    deny = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()
    deny = deny.org.permissions.deny_patterns

    assert "bash_exec(*sudo *)" in deny
    assert len(deny) > 20, f"the shipped deny defaults were wiped: {deny}"


def test_raw_harness_dict_stays_global_only_and_unmutated(layers) -> None:
    """`raw_harness_dict()` is the GLOBAL `config.yaml` only, and the deny splice never writes
    back through it (the copy it returns is shallow, so an in-place splice WOULD show up here)."""
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"permissions": {"deny_patterns": ["bash_exec(*curl *)"]}}},
    )
    _write_yaml(
        ws / "config.yaml",
        {
            "org": {
                "log_level": "debug",
                "permissions": {"deny_patterns": ["bash_exec(*wget *)"]},
            }
        },
    )

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)
    loader.load_harness()
    raw = loader.raw_harness_dict()

    assert "log_level" not in raw["org"], f"workspace data leaked into raw_harness_dict: {raw}"
    assert raw["org"]["permissions"]["deny_patterns"] == ["bash_exec(*curl *)"]


# ---------------------------------------------------------------------------
# Provider = hardware truth (MERG-03)
# ---------------------------------------------------------------------------

def test_workspace_provider_partial_override_inherits_the_rest(layers) -> None:
    """`deep_merge` RECURSES on dict/dict collisions, so a partial `provider:` block needs no
    special-case code — this test is the proof, not a branch."""
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"provider": {"default_model": "ws-model"}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.provider.default_model == "ws-model"
    assert cfg.provider.provider_type == "vllm"
    assert cfg.provider.base_url == "http://localhost:8000/v1"


def test_workspace_with_no_provider_block_uses_the_global_provider(layers) -> None:
    """An absent key is the global layer's to answer: the provider is hardware truth."""
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"org": {"log_level": "debug"}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()

    assert cfg.provider.default_model == "global-model"
    assert cfg.provider.provider_type == "vllm"


# ---------------------------------------------------------------------------
# LAYR-03: no workspace layer, no behavior change
# ---------------------------------------------------------------------------

def test_no_workspace_layer_is_byte_identical(layers) -> None:
    """With no workspace named, the four-source fold must reproduce the old two-source result.

    Both halves matter: `local_config_dir=None` is the same as not passing it at all, AND the
    loaded config still equals `deep_merge(global config.yaml, global overrides.yaml)` computed
    independently here — the v0.12 behavior this milestone promised not to disturb.
    """
    global_dir, _ = layers
    global_cfg = {**_MINIMAL, "org": {"log_level": "warning", "default_temperature": 0.11}}
    # `log_level` is declared by BOTH so the overlay's win over the config is pinned here too:
    # with disjoint keys, swapping sources 1 and 2 in the fold would go unnoticed.
    global_overlay = {
        "agent": {"temperature": 0.9},
        "org": {"default_max_tokens": 2048, "log_level": "error"},
    }
    _write_yaml(global_dir / "config.yaml", global_cfg)
    _write_yaml(global_dir / "overrides.yaml", global_overlay)

    implicit = ConfigLoader(config_dir=global_dir).load_harness()
    explicit_none = ConfigLoader(config_dir=global_dir, local_config_dir=None).load_harness()

    assert implicit.model_dump() == explicit_none.model_dump()

    from localharness.config.models import HarnessConfig

    two_source = HarnessConfig.model_validate(
        deep_merge(global_cfg, {k: v for k, v in global_overlay.items() if k != "agent"})
    )
    assert implicit.model_dump() == two_source.model_dump()
