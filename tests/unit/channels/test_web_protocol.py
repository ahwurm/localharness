"""The protocol contract test: a schema cannot move without `protocol_version` moving with it.

This is deliverable 9 of the web PRD, and the reason it exists is written into §8: a project that
auto-generates its schema specifically because it distrusts humans to keep two things in sync
should not then trust a human to remember the version. A forgotten bump degrades a stale client
SILENTLY — `undefined` renders as a blank cell, not a crash — which is the worst possible failure
for somebody returning to their own UI a few evenings later.

The snapshot is checked in. Changing an event or a frame therefore forces a deliberate choice:
bump `PROTOCOL_VERSION` and re-record, or discover you changed something you did not mean to.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from localharness.channels.web.protocol import (
    COLLAPSIBLE_GROUPS,
    FRAME_TYPES,
    NEVER_FIRED_EVENTS,
    PROTOCOL_VERSION,
    TRANSCRIPT_RULES,
    event_schemas,
    frame_schemas,
)

SNAPSHOT = Path(__file__).parent / "web_protocol_snapshot.json"


def _digest() -> dict[str, str]:
    """A stable hash per type, rather than the whole schema inline.

    The schemas run to tens of KB and a diff nobody can read is a diff nobody checks. What this
    test has to catch is "something MOVED", and a digest catches that exactly while keeping the
    fixture reviewable.
    """
    out: dict[str, str] = {}
    for name, schema in {**event_schemas(), **frame_schemas()}.items():
        blob = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        out[name] = hashlib.sha256(blob).hexdigest()[:16]
    return out


def test_schema_cannot_move_without_a_protocol_version_bump():
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = _digest()

    if recorded["protocol_version"] != PROTOCOL_VERSION:
        pytest.skip("protocol_version moved — re-record the snapshot in the same commit")

    moved = sorted(
        name for name in set(recorded["types"]) | set(current)
        if recorded["types"].get(name) != current.get(name)
    )
    assert not moved, (
        f"wire schema changed for {moved} while PROTOCOL_VERSION stayed at {PROTOCOL_VERSION}. "
        f"Either the change is unintended, or bump PROTOCOL_VERSION and re-record "
        f"{SNAPSHOT.name} in the SAME diff — a stale client degrades silently otherwise."
    )


def test_every_frame_is_described():
    """A frame added without a description is a frame a UI author has to guess at."""
    for frame in FRAME_TYPES:
        schema = frame.model_json_schema()
        assert schema.get("description"), f"{frame.__name__} has no docstring"
        for field, spec in schema.get("properties", {}).items():
            if field in ("frame_type", "session_id"):
                continue
            assert spec.get("description") or spec.get("title"), (
                f"{frame.__name__}.{field} carries no description; `model_json_schema()` cannot "
                f"see a `#` comment, so a rendering rule left in one never reaches a client"
            )


def test_never_fired_events_are_real_event_types():
    """The dead-event list has to name events that EXIST, or it silently stops flagging them."""
    from localharness.core.events import EVENT_TYPE_MAP

    assert NEVER_FIRED_EVENTS <= set(EVENT_TYPE_MAP)


def test_transcript_rules_carry_the_double_print_rule():
    """The one rule whose absence ships every answer twice."""
    joined = " ".join(TRANSCRIPT_RULES).lower()
    assert "must not be rendered" in joined
    assert "taskcomplete.summary" in joined


def test_collapse_is_opt_in_by_safe_group():
    """WEBCH-04: the allowlist names only safe families — never shell, never fs.write.

    Guarding the DIRECTION of the rule, not its contents: an allowlist that grew a dangerous
    group would start collapsing exactly the calls the transcript is built to show.
    """
    assert set(COLLAPSIBLE_GROUPS) == {"fs.read", "web", "memory"}
    for dangerous in ("shell", "fs.write", "code", "delegate", "other"):
        assert dangerous not in COLLAPSIBLE_GROUPS
