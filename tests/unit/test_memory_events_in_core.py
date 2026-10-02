"""MEMP-04 — the memory event classes are core's vocabulary; only plugins produce them (Phase 47).

The seven classes live in `core/events.py` and are in `EVENT_TYPE_MAP` (so a replayed ledger line
of any of them still deserialises). No CORE-classified module (the classification is
`test_import_direction.py`'s, imported, not copied) constructs one. The composed half — a pass the
running plugin launches reaches the terminal channel's subscriptions on the session bus — is in
`tests/integration/test_memory_plugin_composed_e2e.py`.

FINDING (recorded in 47-07-SUMMARY): five of the seven have no producer anywhere in src/ —
MemoryGateFired, ExpectationAttached, OutcomeObserved, SurpriseScored, TurnEndMicroPassCompleted.
Only ConsolidationStarted/Finished are emitted (memory/consolidation.py). This test does not add
producers; it pins where they may NOT come from.
"""
from __future__ import annotations

import ast

from tests.unit.test_import_direction import _sources, classify

MEMORY_EVENTS = ("MemoryGateFired", "ExpectationAttached", "OutcomeObserved", "SurpriseScored",
                 "ConsolidationStarted", "ConsolidationFinished", "TurnEndMicroPassCompleted")


def test_memory_events_live_in_core():
    import localharness.core.events as ev
    for name in MEMORY_EVENTS:
        cls = getattr(ev, name)
        assert issubclass(cls, ev.BaseEvent), name
        assert ev.EVENT_TYPE_MAP[name] is cls, name


def _constructed(tree: ast.AST) -> set[str]:
    """Every memory event a Call constructs. The callee's whole subtree is searched, so
    `Cls(...)`, `ev.Cls(...)` and `(A if started else B)(...)` (consolidation.py's form) all count."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for n in ast.walk(node.func):
                name = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
                if name in MEMORY_EVENTS:
                    out.add(name)
    return out


def test_no_core_module_constructs_a_memory_event():
    core = [(rel, p) for rel, p in _sources() if classify(rel) == "core"]
    assert len(core) >= 80, f"only {len(core)} core files scanned — the classification moved"
    hits = {rel: sorted(names) for rel, p in core
            if (names := _constructed(ast.parse(p.read_text(encoding="utf-8"))))}
    assert hits == {}, f"core constructs memory events: {hits}"


def test_the_scan_sees_the_one_real_producer():
    """The walker is not vacuous: it finds the plugin side's one producer, in its real form."""
    producers = {rel: names for rel, p in _sources()
                 if (names := _constructed(ast.parse(p.read_text(encoding="utf-8"))))}
    assert producers == {"memory/consolidation.py": {"ConsolidationStarted", "ConsolidationFinished"}}, producers
    assert classify("memory/consolidation.py") == "plugin"
