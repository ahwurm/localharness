"""Phase 36 (SEMA-03/04): schema-fact store contract + schemas-first index render.

Task 1 proves the contract primitives (the tier:schema importance prior, the depth-tag
reader). Task 2 proves the index_mode=True render path renders a chapter in a "### Knowledge"
section ABOVE Persistent Facts, byte-stable across renders, zero bytes when there are no
chapters. Hand-written FIXTURE schema facts only — the chapter-writer (36-04) is not needed
to exercise this contract.
"""
from pathlib import Path

import pytest

from localharness.memory.sqlite import (
    MemoryStore,
    _importance_prior,
    _schema_depth,
)


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id="schema-agent", division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


# --- Task 1: contract primitives -------------------------------------------------

def test_schema_tier_importance_prior():
    """tier:schema carries a non-zero prior (0.5) so a chapter LEADS its section; with no
    explicit entry it would sink to the 0.0 floor (Pitfall 2) despite being a promotion."""
    assert _importance_prior(["tier:schema"], "") == 0.5


def test_schema_depth_parses_tag():
    assert _schema_depth(["schema", "depth:2"]) == 2


def test_schema_depth_defaults_zero_for_plain_lesson():
    assert _schema_depth(["schema"]) == 0
