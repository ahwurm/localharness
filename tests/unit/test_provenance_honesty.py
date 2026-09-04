"""What `config show` / `components get` / `doctor` say about WHERE a value came from.

Three attributions were fabricated (A-H1/H3/M4):
  * the deny list is UNIONED across layers, and the display credited whichever layer happened to
    be highest — beside a list holding the other layer's patterns;
  * a `dict[K, V]` leaf is deep-merged key-wise, and got the same single-layer credit;
  * a workspace `server: null` clobbers the whole block, but carries no `server.runtime` key — so
    every leaf under it was credited to the global layer, which no longer supplies that value.
"""
from __future__ import annotations

import yaml

from localharness.registry.provenance import layered_catalogue

_GLOBAL = {
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://global-host:8000/v1",
        "default_model": "global-model",
    },
    "org": {
        "permissions": {"deny_patterns": ["global_pat"]},
        "context": {"model_context_overrides": {"modelA": 8192}},
    },
    "server": {"runtime": "llamacpp", "binary": "/usr/bin/llama-server", "model": "global-model"},
}


def _layers(tmp_path, ws_config: dict):
    global_dir, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    global_dir.mkdir(parents=True)
    ws.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(yaml.safe_dump(_GLOBAL), encoding="utf-8")
    (ws / "config.yaml").write_text(yaml.safe_dump(ws_config), encoding="utf-8")
    return global_dir, ws


def test_union_deny_list_is_labelled_accumulated(tmp_path):
    global_dir, ws = _layers(tmp_path, {"org": {"permissions": {"deny_patterns": ["ws_pat"]}}})

    effective, _ = layered_catalogue(global_dir, ws)
    entry = effective["org.permissions.deny_patterns"]

    assert entry.current_value == ["global_pat", "ws_pat"]  # premise: the value IS a union
    assert entry.winning_layer == "accumulated (global-config + workspace-config)"


def test_merged_dict_leaf_is_labelled_accumulated(tmp_path):
    global_dir, ws = _layers(
        tmp_path, {"org": {"context": {"model_context_overrides": {"modelB": 4096}}}}
    )

    effective, _ = layered_catalogue(global_dir, ws)
    entry = effective["org.context.model_context_overrides"]

    assert entry.current_value == {"modelA": 8192, "modelB": 4096}
    assert entry.winning_layer == "accumulated (global-config + workspace-config)"


def test_a_workspace_null_clobber_is_attributed_to_the_workspace(tmp_path):
    global_dir, ws = _layers(tmp_path, {"server": None})

    effective, _ = layered_catalogue(global_dir, ws)

    assert effective["server.runtime"].current_value is None
    assert effective["server.runtime"].winning_layer == "workspace-config"
    assert effective["server.binary"].winning_layer == "workspace-config"


def test_single_layer_attribution_is_untouched(tmp_path):
    """LAYR-03: with only the global layer contributing, the band names are exactly as before."""
    global_dir, ws = _layers(tmp_path, {})

    effective, _ = layered_catalogue(global_dir, ws)
    global_only, _ = layered_catalogue(global_dir, None)

    for path in ("org.permissions.deny_patterns", "org.context.model_context_overrides",
                 "server.runtime", "provider.base_url"):
        assert effective[path].winning_layer == "global-config", path
        assert global_only[path].winning_layer == "global-config", path


def test_a_workspace_that_only_overrides_wins_alone(tmp_path):
    """A scalar the workspace replaces is still a single-layer answer, not "accumulated"."""
    global_dir, ws = _layers(tmp_path, {"provider": {"base_url": "http://ws-host:8000/v1"}})

    effective, _ = layered_catalogue(global_dir, ws)

    assert effective["provider.base_url"].winning_layer == "workspace-config"


def test_layered_catalogue_reuses_a_supplied_loader(tmp_path):
    """doctor hands over the loader it already built rather than re-reading every file."""
    from localharness.config.loader import ConfigLoader

    global_dir, ws = _layers(tmp_path, {})
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)
    loader.load_harness()
    reads: list[int] = []
    original = ConfigLoader.load_harness

    def counting(self):
        reads.append(1)
        return original(self)

    ConfigLoader.load_harness = counting
    try:
        layered_catalogue(global_dir, ws, loader=loader)
    finally:
        ConfigLoader.load_harness = original

    assert len(reads) == 1  # the cached loader answered; no second parse of the same files


def test_an_empty_deny_list_says_the_shipped_defaults_still_enforce(tmp_path):
    """A-M3: `[]` on screen while two dozen patterns enforce is a display that lies by omission."""
    from localharness.config.models import PermissionConfig
    from localharness.registry.provenance import display_note

    shipped = len(PermissionConfig().deny_patterns)
    assert shipped > 0  # premise

    note = display_note("org.permissions.deny_patterns", [])

    assert f"+{shipped} shipped defaults always enforced" in note
    assert display_note("org.permissions.deny_patterns", ["rm -rf /"]) == ""
    assert display_note("provider.base_url", "") == ""
