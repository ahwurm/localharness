"""Phase 13 corpus sentinel: enforces category-level partition between train and holdout.
Load-bearing defense against thematic leakage. Runs as integration (touches real bench/scenarios/).
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import pytest
import yaml
from localharness.bench.schema import load_scenario

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "bench" / "scenarios"
CATEGORIES_FILE = Path(__file__).resolve().parents[2] / "bench" / "categories.yaml"

HOLDOUT_CATEGORIES = {
    "long_horizon_planning", "tool_ambiguity_resolution",
    "graceful_failure", "self_correction", "constraint_satisfaction",
}


def test_every_category_partitions_to_one_slice():
    fixtures = sorted(CORPUS_ROOT.rglob("*.yaml"))
    assert fixtures, f"No fixtures found under {CORPUS_ROOT}"
    cat_to_slices: dict[str, set[str]] = defaultdict(set)
    for path in fixtures:
        spec = load_scenario(path)
        cat_to_slices[spec.category].add(spec.slice)
    mixed = {c: s for c, s in cat_to_slices.items() if len(s) > 1}
    assert not mixed, f"Categories mixed across slices (forbidden): {mixed}"


def test_every_declared_train_category_has_at_least_one_fixture():
    """Holdout categories are populated in Wave 3 — only train-side coverage enforced here.
    Wave 3 PLAN adds a separate strict check for holdout coverage."""
    allowed = set(yaml.safe_load(CATEGORIES_FILE.read_text())["categories"].keys())
    train_allowed = allowed - HOLDOUT_CATEGORIES
    fixtures = sorted(CORPUS_ROOT.rglob("*.yaml"))
    used = {load_scenario(p).category for p in fixtures}
    missing_train = train_allowed - used
    assert not missing_train, f"Train-side categories missing fixtures: {missing_train}"


def test_every_fixture_has_valid_slice_and_category():
    fixtures = sorted(CORPUS_ROOT.rglob("*.yaml"))
    assert fixtures, f"No fixtures found under {CORPUS_ROOT}"
    for path in fixtures:
        spec = load_scenario(path)
        assert spec.slice in ("train", "holdout"), f"{path}: bad slice {spec.slice!r}"
        assert spec.category, f"{path}: empty category"
