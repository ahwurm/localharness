"""Rendering a config error must be cheap, whatever the rejected value is (H7).

YAML alias amplification: a file under 1KB whose anchors reference each other nine deep parses in
milliseconds and expands to millions of nodes. `repr()`-ing that into the error message measured at
6GB and 41 seconds — in `validate`, the command a user runs BECAUSE something is already wrong.
The value's SIZE is now reported from something cheap to know; the repr itself is never built.
"""
from __future__ import annotations

import time

import pytest

from localharness.config.loader import (
    VALUE_REPR_LIMIT,
    ConfigLoader,
    ConfigValidationError,
    _short_repr,
)

# 9^5 = 59,049 leaves from ~200 bytes. Enough to be pathological, small enough that CONSTRUCTING it
# (which the parser does either way) is not what this test measures.
AMPLIFIED = """name: bomb
role: r
temperature: &a ["x", "x", "x", "x", "x", "x", "x", "x", "x"]
max_tokens: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a]
timeout_seconds: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b]
tags: &d [*c, *c, *c, *c, *c, *c, *c, *c, *c]
capabilities: [*d, *d, *d, *d, *d, *d, *d, *d, *d]
"""


def test_amplified_value_renders_fast_and_small(tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "bomb.yaml").write_text(AMPLIFIED, encoding="utf-8")

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=tmp_path).load_agent("bomb")

    started = time.perf_counter()
    rendered = str(exc.value)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"rendering the error took {elapsed:.1f}s"
    assert "truncated" in rendered
    # The cap is PER VALUE — pydantic reports one error per offending element, so the message as a
    # whole grows with the number of errors, not with the size of the expansion behind each one.
    for field_err in exc.value.errors:
        assert len(str(field_err)) < VALUE_REPR_LIMIT + 400, str(field_err)[:200]
    assert len(rendered) < 100 * VALUE_REPR_LIMIT, f"error message is {len(rendered)} chars"


def test_a_long_string_says_how_long_it_really_was():
    rendered = _short_repr("y" * 10_000)

    assert len(rendered) < VALUE_REPR_LIMIT + 100
    assert "truncated, 10000 chars" in rendered


def test_a_big_container_says_how_many_items_it_really_had():
    rendered = _short_repr(list(range(5_000)))

    assert len(rendered) <= VALUE_REPR_LIMIT + 60
    assert "truncated, 5000 top-level items" in rendered


@pytest.mark.parametrize("value", [None, True, 42, 3.5, "short", [1, 2, 3], {"a": 1}])
def test_small_values_are_reported_verbatim(value):
    """Byte-identical for everything a user actually mistypes — the cap is for the pathological."""
    assert _short_repr(value) == repr(value)
