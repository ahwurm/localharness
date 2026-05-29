"""ARCH-02 — Pareto-front sampling, KNOWN-answer (Phase 15 Wave 0 RED stubs).

Each test is an xfail(strict=False) stub pinned to a hand-computed answer so the
two front algorithms go RED→GREEN as they land in 15-03. The sealed-slice
invariant (holdout never participates in front selection) is encoded by two
teeth: one exclusion test + one metrics-rejection test.

Module-level import is guarded so collection never breaks before the module ships.
"""
import pytest

try:
    from localharness.autoresearch.archive import ArchiveStore, ArchiveEntry, ArchiveQuery  # noqa: F401
except ImportError:
    pytest.skip("ArchiveStore not yet implemented (15-02)", allow_module_level=True)


async def test_per_fixture_known_front(archive_store, seeded_archive):
    """Per-fixture front = entries best on >=1 fixture (ties included), dominated-on-all excluded."""
    await seeded_archive(
        archive_store,
        [
            dict(id="A", status="in_flight", train_scores_per_fixture={"fx1": 0.9, "fx2": 0.5}),
            dict(id="B", status="in_flight", train_scores_per_fixture={"fx1": 0.4, "fx2": 0.9}),
            dict(id="C", status="in_flight", train_scores_per_fixture={"fx1": 0.3, "fx2": 0.4}),
            dict(id="D", status="in_flight", train_scores_per_fixture={"fx1": 0.9, "fx2": 0.5}),
        ],
    )
    front = {e.id for e in await archive_store.pareto_front_per_fixture()}
    assert front == {"A", "B", "D"}  # C dominated on every fixture; D ties A on fx1


async def test_2d_known_front(archive_store, seeded_archive):
    """2D (train_score, cost) front = non-dominated set among eligible (promoted, p<0.05) rows."""
    await seeded_archive(
        archive_store,
        [
            dict(id="P1", status="promoted", train_score=0.9, cost=1.0, p_value=0.01),
            dict(id="P2", status="promoted", train_score=0.8, cost=2.0, p_value=0.01),  # dominated by P1
            dict(id="P3", status="promoted", train_score=0.7, cost=0.5, p_value=0.01),
        ],
    )
    front = {e.id for e in await archive_store.pareto_front_2d(metrics=["train_score", "cost"])}
    assert front == {"P1", "P3"}  # P2 dominated (lower score AND higher cost than P1)


async def test_holdout_excluded_from_front(archive_store, seeded_archive):
    """An entry with the highest holdout_score but best on no train fixture is absent from the front."""
    await seeded_archive(
        archive_store,
        [
            dict(id="WIN", status="in_flight", train_scores_per_fixture={"fx1": 0.9}, holdout_score=0.1),
            dict(id="HOLD", status="in_flight", train_scores_per_fixture={"fx1": 0.2}, holdout_score=0.99),
        ],
    )
    front = {e.id for e in await archive_store.pareto_front_per_fixture()}
    assert "HOLD" not in front
    assert "WIN" in front


async def test_metrics_rejects_sealed_column(archive_store):
    """pareto_front_2d refuses holdout_score as a front axis (sealed-slice invariant)."""
    with pytest.raises(ValueError):
        await archive_store.pareto_front_2d(metrics=["holdout_score", "cost"])


async def test_per_fixture_status_filter(archive_store, seeded_archive):
    """A train_rejected entry that would win a fixture is excluded (only promoted|in_flight eligible)."""
    await seeded_archive(
        archive_store,
        [
            dict(id="REJ", status="train_rejected", train_scores_per_fixture={"fx1": 0.99}),
            dict(id="OK", status="in_flight", train_scores_per_fixture={"fx1": 0.5}),
        ],
    )
    front = {e.id for e in await archive_store.pareto_front_per_fixture()}
    assert "REJ" not in front
    assert "OK" in front


async def test_2d_status_and_pvalue_filter(archive_store, seeded_archive):
    """Only promoted rows with p_value < 0.05 are eligible for the 2D front."""
    await seeded_archive(
        archive_store,
        [
            dict(id="SIG", status="promoted", train_score=0.8, cost=1.0, p_value=0.01),
            dict(id="NOISE", status="promoted", train_score=0.9, cost=1.0, p_value=0.9),
        ],
    )
    front = {e.id for e in await archive_store.pareto_front_2d(metrics=["train_score", "cost"])}
    assert "SIG" in front
    assert "NOISE" not in front
