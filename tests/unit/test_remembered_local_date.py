"""The remembered date is the local calendar date of the ledger's UTC `at`."""
from __future__ import annotations

import time

import pytest

from localharness.provider.tuning import NOTHING_DERIVED, derived_clause

AT = "2026-10-17T03:00:00+00:00"


@pytest.fixture
def zone(monkeypatch):
    def set_zone(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield set_zone
    monkeypatch.undo()
    time.tzset()


def _clause(at: str) -> str:
    return derived_clause(NOTHING_DERIVED, shape="retry", cause="remembered", dropped=("draft-mtp",), remembered_at=at)


def test_a_launch_remembered_after_midnight_utc_reads_the_evening_before_in_los_angeles(zone):
    zone("America/Los_Angeles")

    assert _clause(AT) == "mtp off (remembered 2026-10-16)"


def test_the_same_instant_reads_its_utc_date_in_utc(zone):
    zone("UTC")

    assert _clause(AT) == "mtp off (remembered 2026-10-17)"


@pytest.mark.parametrize("at", ["not a date", "2026-10-17T03:00:00\x1b[2J"])
def test_an_at_that_does_not_parse_prints_no_date_and_no_escape(zone, at):
    zone("UTC")

    clause = _clause(at)

    assert clause == "mtp off (remembered)"
    assert "\x1b" not in clause
