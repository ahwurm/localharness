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


def test_every_declared_category_has_at_least_one_fixture():
    """Strict version (post-Wave-3): every category declared in bench/categories.yaml
    must have at least one fixture. Was relaxed in Wave 1 to allow holdout categories
    to be unpopulated; Wave 3 fills them, so we tighten the assertion."""
    allowed = set(yaml.safe_load(CATEGORIES_FILE.read_text())["categories"].keys())
    fixtures = sorted(CORPUS_ROOT.rglob("*.yaml"))
    used = {load_scenario(p).category for p in fixtures}
    missing = allowed - used
    assert not missing, f"Declared categories with no fixtures: {missing}"


def test_every_fixture_has_valid_slice_and_category():
    fixtures = sorted(CORPUS_ROOT.rglob("*.yaml"))
    assert fixtures, f"No fixtures found under {CORPUS_ROOT}"
    for path in fixtures:
        spec = load_scenario(path)
        assert spec.slice in ("train", "holdout"), f"{path}: bad slice {spec.slice!r}"
        assert spec.category, f"{path}: empty category"


def test_every_train_category_has_at_least_two_fixtures():
    """Within-class variance invariant: each train-side category needs >=2 fixtures so
    a mutation that helps the category must help BOTH variants (beats fixture-specific
    overfitting). Holdout categories are exempt — they intentionally have just 2 each,
    and that lower bar is enforced separately in Wave 3.
    """
    from collections import Counter
    fixtures = sorted(CORPUS_ROOT.rglob("*.yaml"))
    assert fixtures, f"No fixtures found under {CORPUS_ROOT}"
    counts: Counter[str] = Counter()
    for path in fixtures:
        spec = load_scenario(path)
        if spec.slice == "train":
            counts[spec.category] += 1
    thin = {cat: n for cat, n in counts.items() if n < 2}
    assert not thin, (
        f"Train categories with <2 fixtures (violates within-class variance invariant): {thin}. "
        f"Wave 2 must land >=2 fixtures per train category."
    )


def test_every_holdout_category_has_at_least_two_fixtures():
    """Strict holdout coverage: every holdout-side category must have exactly 2 fixtures.
    >=2 is required so a mutation must transfer across BOTH variants of an unseen class.
    More than 2 would amplify multi-trial exposure of the sealed slice (Karpathy failure
    mode #3, p-hacking via repeated holdout queries).
    """
    from collections import Counter
    fixtures = sorted(CORPUS_ROOT.rglob("*.yaml"))
    counts: Counter[str] = Counter()
    for path in fixtures:
        spec = load_scenario(path)
        if spec.slice == "holdout":
            counts[spec.category] += 1
    thin = {cat: counts.get(cat, 0) for cat in HOLDOUT_CATEGORIES if counts.get(cat, 0) < 2}
    assert not thin, (
        f"Holdout categories with <2 fixtures: {thin}. "
        f"Wave 3 must land exactly 2 fixtures per holdout category."
    )
