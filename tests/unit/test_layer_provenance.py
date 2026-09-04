"""Which file set this key — graded at the ONE module three commands will share.

`components get`, `doctor`'s overridden-key report (43-05) and `config show` (43-04) are three
views of a single question. This file grades the answer itself, unit-level against real temp
directories; the CLI proof that `components` actually asks it lives in
`test_components_workspace_layer.py`.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from localharness.config.loader import ConfigLoader

_GLOBAL_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "test-model",
    },
    "org": {"name": "GLOBAL-ORG-NAME", "log_level": "info"},
}


def _dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _layers(tmp_path: Path) -> tuple[Path, Path]:
    """(global config dir, workspace dir) — both real, the global one loadable."""
    global_dir = tmp_path / "home" / ".localharness"
    _dump(global_dir / "config.yaml", _GLOBAL_CONFIG)
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True, exist_ok=True)
    return global_dir, ws


def _loader(global_dir: Path, ws: Path | None) -> ConfigLoader:
    return ConfigLoader(config_dir=global_dir, local_config_dir=ws)


# ------------------------------------------------------------------ #
# build_layer_overlays — which dicts, keyed how
# ------------------------------------------------------------------ #


def test_no_workspace_gives_exactly_the_two_global_bands(tmp_path):
    """LAYR-03 at the provenance layer: with no workspace, attribution is what it always was.

    EXACTLY two keys, not "at least two" — an unpopulated band is inert in `_detect_layer`, but a
    band populated with `{}` when no workspace applies would still be a claim this module makes
    about a file that does not exist.
    """
    from localharness.registry.provenance import build_layer_overlays

    global_dir, _ = _layers(tmp_path)
    overlays = build_layer_overlays(_loader(global_dir, None), None)

    assert set(overlays) == {"global-config", "global-overrides"}
    assert overlays["global-config"]["org"]["name"] == "GLOBAL-ORG-NAME"


def test_a_workspace_adds_exactly_its_own_two_files(tmp_path):
    """Four bands, and the two workspace dicts are THAT workspace's files, read from disk."""
    from localharness.registry.provenance import build_layer_overlays

    global_dir, ws = _layers(tmp_path)
    _dump(ws / "config.yaml", {"org": {"name": "WORKSPACE-ORG-NAME"}})
    _dump(ws / "overrides.yaml", {"org": {"log_level": "debug"}})

    overlays = build_layer_overlays(_loader(global_dir, ws), ws)

    assert set(overlays) == {
        "global-config",
        "global-overrides",
        "workspace-config",
        "workspace-overrides",
    }
    assert overlays["workspace-config"] == {"org": {"name": "WORKSPACE-ORG-NAME"}}
    assert overlays["workspace-overrides"] == {"org": {"log_level": "debug"}}
    # The global band is still the global file — a workspace never rewrites what the global
    # layer SET, only what the merge RESOLVES to.
    assert overlays["global-config"]["org"]["name"] == "GLOBAL-ORG-NAME"


def test_absent_workspace_files_are_empty_dicts_not_errors(tmp_path):
    """A workspace with no config.yaml and no overrides.yaml is the shipped scaffold's own state
    the moment a user deletes the template — it must attribute nothing, not raise."""
    from localharness.registry.provenance import build_layer_overlays

    global_dir, ws = _layers(tmp_path)
    assert not (ws / "config.yaml").exists() and not (ws / "overrides.yaml").exists()

    overlays = build_layer_overlays(_loader(global_dir, ws), ws)

    assert overlays["workspace-config"] == {}
    assert overlays["workspace-overrides"] == {}


def test_the_agent_section_is_kept_globally_and_dropped_in_the_workspace(tmp_path):
    """The shipped asymmetry, asserted from BOTH sides in ONE body so neither can shadow the other.

    * workspace overrides `agent:` — DROPPED. Nothing reads it in v0.13 (loader._raw_config_sources
      strips it, 06-config.md documents it). Attributing a key to a file the harness never consults
      is the same lie F4 exists to fix, relocated.
    * global overrides `agent:` — KEPT. `load_agent` reads it; it is the per-agent default layer,
      and `components set agent.*` writes into it.

    Order matters: the workspace DROP is asserted first. A single-sided test passes for an
    implementation that strips both, and a future "fix" making the two symmetric would then ship
    green.
    """
    from localharness.registry.provenance import build_layer_overlays, layered_catalogue

    global_dir, ws = _layers(tmp_path)
    agent_block = {"agent": {"memory": {"recall_scope": "both"}}}

    # --- side 1: in the WORKSPACE overrides, the block is invisible ---
    _dump(ws / "overrides.yaml", agent_block)
    overlays = build_layer_overlays(_loader(global_dir, ws), ws)
    assert overlays["workspace-overrides"] == {}, (
        "the workspace overrides' agent: section must be dropped — nothing in v0.13 reads it"
    )
    cat, _ = layered_catalogue(global_dir, ws)
    assert cat["agent.memory.recall_scope"].winning_layer == "default"

    # --- side 2: the SAME block in the GLOBAL overrides is read ---
    _dump(global_dir / "overrides.yaml", agent_block)
    overlays = build_layer_overlays(_loader(global_dir, ws), ws)
    assert overlays["global-overrides"] == agent_block, (
        "the global overrides' agent: section must survive — load_agent reads it and "
        "`components set agent.*` writes into it"
    )
    cat, _ = layered_catalogue(global_dir, ws)
    assert cat["agent.memory.recall_scope"].winning_layer == "global-overrides", (
        "with the block present in BOTH files the global one must still win — the workspace "
        "copy is not merely outranked, it is not read at all"
    )


# ------------------------------------------------------------------ #
# layered_catalogue — the honest band, and the set->get round-trip
# ------------------------------------------------------------------ #


def test_each_band_names_the_file_that_actually_set_the_key(tmp_path):
    """Four files, four keys, four distinct bands — F2's whole claim in one catalogue."""
    from localharness.registry.provenance import layered_catalogue

    global_dir, ws = _layers(tmp_path)
    _dump(global_dir / "overrides.yaml", {"org": {"log_level": "warning"}})
    _dump(ws / "config.yaml", {"org": {"name": "WORKSPACE-ORG-NAME"}})
    _dump(ws / "overrides.yaml", {"org": {"context": {"compaction_threshold_pct": 71.0}}})

    cat, _ = layered_catalogue(global_dir, ws)

    assert cat["provider.default_model"].winning_layer == "global-config"
    assert cat["org.log_level"].winning_layer == "global-overrides"
    assert cat["org.name"].winning_layer == "workspace-config"
    assert cat["org.context.compaction_threshold_pct"].winning_layer == "workspace-overrides"
    # …and the VALUE follows the winning file, not just the label.
    assert cat["org.name"].current_value == "WORKSPACE-ORG-NAME"
    assert cat["org.context.compaction_threshold_pct"].current_value == 71.0


def test_a_workspace_config_key_outranks_the_same_key_globally(tmp_path):
    """The ruled order where it bites: the same path set in the global overrides AND the workspace
    config attributes to the workspace, because the SPECIFIC beats the GENERAL."""
    from localharness.registry.provenance import layered_catalogue

    global_dir, ws = _layers(tmp_path)
    _dump(global_dir / "overrides.yaml", {"org": {"log_level": "warning"}})
    _dump(ws / "config.yaml", {"org": {"log_level": "debug"}})

    cat, _ = layered_catalogue(global_dir, ws)

    assert cat["org.log_level"].winning_layer == "workspace-config"
    assert cat["org.log_level"].current_value == "debug"


def test_set_agent_star_round_trips_through_the_global_overlay(tmp_path):
    """`components set agent.*` writes the global overrides' agent: section; `get` must read it
    back. That is what apply_agent_overlay_values exists for — build_catalogue alone reports the
    compiled-in AgentConfig default for every agent.* path."""
    from localharness.registry.provenance import layered_catalogue

    global_dir, ws = _layers(tmp_path)
    _dump(global_dir / "overrides.yaml", {"agent": {"memory": {"recall_scope": "global"}}})

    cat, _ = layered_catalogue(global_dir, ws)

    assert cat["agent.memory.recall_scope"].current_value == "global"
    assert cat["agent.memory.recall_scope"].winning_layer == "global-overrides"
    # Untouched agent axes keep their compiled-in default — the overlay patches only what it
    # EXPLICITLY sets, so a placeholder agent's name-derived memory paths never leak in.
    assert cat["agent.memory.inject_into_context"].current_value is True


# ------------------------------------------------------------------ #
# overridden_paths — a VALUE diff, not a presence check
# ------------------------------------------------------------------ #


def test_restating_a_key_with_the_same_value_overrides_nothing(tmp_path):
    """The assertion a naive presence filter fails, and the reason this function exists.

    People copy a workspace config.yaml between projects; most of its keys end up restating what
    the global layer already said. Reporting those as "overridden" would make doctor's section
    noise on exactly the configs that are most common.

    The sibling assertion is load-bearing: a test that only checks the empty case passes for an
    implementation that always returns [].
    """
    from localharness.registry.provenance import layered_catalogue, overridden_paths

    global_dir, ws = _layers(tmp_path)

    # Same value as the global config's own org.log_level.
    _dump(ws / "config.yaml", {"org": {"log_level": "info"}})
    effective, _ = layered_catalogue(global_dir, ws)
    global_only, _ = layered_catalogue(global_dir, None)
    assert overridden_paths(effective, global_only) == []

    # One character different, and it IS an override.
    _dump(ws / "config.yaml", {"org": {"log_level": "debug"}})
    effective, _ = layered_catalogue(global_dir, ws)
    rows = overridden_paths(effective, global_only)
    assert len(rows) == 1
    path, entry, before = rows[0]
    assert path == "org.log_level"
    assert entry.current_value == "debug"
    assert before == "info", "the row must carry the value the global layer had, for the diff"


def test_overridden_paths_is_sorted_so_two_doctor_runs_print_the_same_order(tmp_path):
    """Sorted by path — a dict-order report reads as churn between two runs on one machine."""
    from localharness.registry.provenance import layered_catalogue, overridden_paths

    global_dir, ws = _layers(tmp_path)
    _dump(
        ws / "config.yaml",
        {"org": {"log_level": "debug", "name": "WORKSPACE-ORG-NAME"}},
    )
    effective, _ = layered_catalogue(global_dir, ws)
    global_only, _ = layered_catalogue(global_dir, None)

    rows = overridden_paths(effective, global_only)
    assert [r[0] for r in rows] == ["org.log_level", "org.name"]


def test_no_workspace_means_nothing_is_overridden(tmp_path):
    """The control: with no workspace, the two catalogues are the same catalogue."""
    from localharness.registry.provenance import layered_catalogue, overridden_paths

    global_dir, _ = _layers(tmp_path)
    a, _ = layered_catalogue(global_dir, None)
    b, _ = layered_catalogue(global_dir, None)
    assert overridden_paths(a, b) == []


def test_the_public_surface_is_the_four_names_three_commands_will_call():
    """43-04 and 43-05 both import from here; the module must not quietly drop one."""
    import localharness.registry.provenance as p

    for name in (
        "build_layer_overlays",
        "apply_agent_overlay_values",
        "layered_catalogue",
        "overridden_paths",
    ):
        assert callable(getattr(p, name)), name
