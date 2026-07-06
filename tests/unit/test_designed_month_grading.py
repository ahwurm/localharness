"""MOVE 4 — the designed-month grading math (offline, fixture-based). Proves Stages A/B + the
verdict of `_grade_designed_month` against fabricated stores with a KNOWN expected grouping
(fabricated fixtures are allowed in unit tests only; a real provable run mines real content),
plus the inline Adjusted Rand Index and an end-to-end `--manifest --offline` reachability run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from localharness.memory.sqlite import MemoryStore

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import sema05_month_in_a_day as sema  # noqa: E402

AGENT = "orchestrator"


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id=AGENT, division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


# --- Adjusted Rand Index (inline, no deps) ----------------------------------------------

def test_ari_known_values():
    assert sema._ari([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0        # identical partitions
    assert sema._ari([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0        # same partition, relabeled
    assert sema._ari([0, 0, 0, 0], [0, 1, 2, 3]) == 0.0        # one cluster vs all-singletons
    # A partial-agreement case sits strictly between the random (0) and perfect (1) extremes.
    partial = sema._ari([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2])
    assert -0.5 <= partial < 1.0


# --- fixture builders (a controllable designed store) -----------------------------------

_TS = {"n": 1000}


async def _seed_atom(store, topic, claim, day, *, evidence=None, batch_prov=False):
    """A sem/ atom + a same-day history record it grounds against (A2). provenance is the source
    sitting `designed-{day}` unless batch_prov (the SEMA-05 defect) is forced for the A3 test."""
    _TS["n"] += 1
    await store.append_history({"v": 1, "agent_id": AGENT, "type": "user_message", "id": f"h{_TS['n']}",
                                "session_id": f"designed-{day}", "ts": _TS["n"], "content": evidence or claim})
    prov = "mined-from:batch-2026" if batch_prov else f"designed-{day}"
    return await store.store_fact(
        key=f"sem/{topic}/{_TS['n']}", value=claim, tags=["sem", "pending_consolidation"],
        confidence=0.65, provenance=prov, node_kind="fact",
    )


async def _seed_chapter(store, value, members):
    """A chapter node (node_kind='schema') with a member_of edge to each member atom."""
    _TS["n"] += 1
    sch = await store.store_fact(
        key=f"schema/cluster/{_TS['n']}", value=value, tags=["schema", "tier:schema", "depth:1"],
        confidence=0.8, node_kind="schema", provenance="cluster:designed",
    )
    for m in members:
        await store.add_edge(sch.id, m.id, "member_of")
    return sch


def _holds_manifest():
    return {
        "days": ["day1", "day2"],
        "topics": {
            "markets": {"expected_chapter": True, "days": ["day1", "day2"], "keywords": ["hbm", "foundry"]},
            "kyoto": {"expected_chapter": True, "days": ["day1", "day2"], "keywords": ["kyoto", "ryokan"]},
            "noise": {"expected_chapter": False},
        },
        "correction_arc": {"topic": "markets", "stale": "port 8000", "corrected": "port 8081",
                           "assert_query": "d1-m", "correct_query": "d2-m"},
        "queries": [
            {"id": "d1-m", "day": "day1", "topic": "markets", "text": "researching hbm foundry stocks"},
            {"id": "d1-k", "day": "day1", "topic": "kyoto", "text": "planning a kyoto ryokan trip"},
            {"id": "d1-n", "day": "day1", "topic": "noise", "text": "capital of mongolia today"},
            {"id": "d2-m", "day": "day2", "topic": "markets", "text": "hbm foundry earnings port 8081"},
            {"id": "d2-k", "day": "day2", "topic": "kyoto", "text": "kyoto ryokan onsen recommendation"},
        ],
    }


def _tqm(manifest):
    tqm: dict = {}
    for q in manifest["queries"]:
        tqm.setdefault(q["day"], []).append((q["id"], q["topic"], q["text"]))
    return tqm


async def _seed_holds(store):
    """Two clean chapters (markets, kyoto) with correct members; markets carries the corrected
    port 8081 (B4). Returns the manifest used."""
    m = _holds_manifest()
    mk1 = await _seed_atom(store, "markets", "researching hbm foundry stocks", "day1")
    mk2 = await _seed_atom(store, "markets", "hbm foundry earnings port 8081", "day2")
    ky1 = await _seed_atom(store, "kyoto", "planning a kyoto ryokan trip", "day1")
    ky2 = await _seed_atom(store, "kyoto", "kyoto ryokan onsen recommendation", "day2")
    await _seed_chapter(store, "hbm foundry earnings stocks", [mk1, mk2])
    await _seed_chapter(store, "kyoto ryokan onsen trip", [ky1, ky2])
    return m


# --- Stage A/B + verdict --------------------------------------------------------------

@pytest.mark.asyncio
async def test_grade_holds_on_clean_designed_store(store):
    m = await _seed_holds(store)
    g = await sema._grade_designed_month(store, m, _tqm(m))
    assert g["verdict"] == "HOLDS", g
    assert g["failing_stage"] is None
    assert g["stage_a"]["a1_recall"] == 1.0 and g["stage_a"]["a3_provenance_ok"] is True
    assert g["stage_b"]["b1_ok"] and g["stage_b"]["b2_ok"] and g["stage_b"]["b3_ok"] and g["stage_b"]["b4_ok"]
    assert g["stage_b"]["ari"] == 1.0
    assert g["stage_b"]["operational_rows_in_chapter"] == 0  # ruling c: no gate/predgate/learned in a chapter
    assert {pc["label"] for pc in g["stage_b"]["per_chapter"]} == {"markets", "kyoto"}


@pytest.mark.asyncio
async def test_grade_kill_on_ungrounded_atom(store):
    """A2 zero tolerance: an atom whose claim is not grounded in its day's transcript -> KILL."""
    m = await _seed_holds(store)
    # An atom whose value shares no >=6-char token with its cited day record -> ungrounded.
    await _seed_atom(store, "markets", "quarterly zeppelin fabrication numbers", "day1",
                     evidence="researching hbm foundry stocks")
    g = await sema._grade_designed_month(store, m, _tqm(m))
    assert g["verdict"] == "KILL" and g["stage_a"]["a2_kill"] is True


@pytest.mark.asyncio
async def test_grade_kill_on_unverified_figure_in_chapter(store):
    """B5 numeric net (carried verbatim): a chapter figure derivable from NO member atom -> KILL."""
    m = _holds_manifest()
    mk1 = await _seed_atom(store, "markets", "researching hbm foundry stocks", "day1")
    mk2 = await _seed_atom(store, "markets", "hbm foundry earnings port 8081", "day2")
    ky1 = await _seed_atom(store, "kyoto", "planning a kyoto ryokan trip", "day1")
    ky2 = await _seed_atom(store, "kyoto", "kyoto ryokan onsen recommendation", "day2")
    await _seed_chapter(store, "hbm foundry grew 4242 percent", [mk1, mk2])  # 4242 in no member
    await _seed_chapter(store, "kyoto ryokan onsen trip", [ky1, ky2])
    g = await sema._grade_designed_month(store, m, _tqm(m))
    assert g["verdict"] == "KILL" and g["stage_b"]["b5_kill"] is True


@pytest.mark.asyncio
async def test_grade_inconclusive_on_distractor_in_chapter(store):
    """B3: a noise-labelled atom folded into a chapter -> distractor exclusion fails -> INCONCLUSIVE."""
    m = _holds_manifest()
    # Three real markets atoms so precision stays >= 0.7 with the distractor (3/4) — isolating B3
    # (distractor exclusion) as the named failing stage rather than B2 (membership precision).
    mk1 = await _seed_atom(store, "markets", "researching hbm foundry stocks", "day1")
    mk2 = await _seed_atom(store, "markets", "hbm foundry earnings port 8081", "day2")
    mk3 = await _seed_atom(store, "markets", "hbm foundry capex is rising", "day1")
    ky1 = await _seed_atom(store, "kyoto", "planning a kyoto ryokan trip", "day1")
    ky2 = await _seed_atom(store, "kyoto", "kyoto ryokan onsen recommendation", "day2")
    noise = await _seed_atom(store, "markets", "capital of mongolia today", "day1")  # attributes to the noise query
    await _seed_chapter(store, "hbm foundry earnings stocks capex", [mk1, mk2, mk3, noise])  # distractor folded in
    await _seed_chapter(store, "kyoto ryokan onsen trip", [ky1, ky2])
    g = await sema._grade_designed_month(store, m, _tqm(m))
    assert g["verdict"] == "INCONCLUSIVE" and g["failing_stage"] == "B3"
    assert g["stage_b"]["noise_in_chapter"] == 1 and g["stage_b"]["b2_ok"] is True


@pytest.mark.asyncio
async def test_grade_invalid_on_batch_provenance(store):
    """A3: batch-level provenance (the SEMA-05 defect) -> INVALID (measurement, not subject)."""
    m = _holds_manifest()
    mk1 = await _seed_atom(store, "markets", "researching hbm foundry stocks", "day1", batch_prov=True)
    mk2 = await _seed_atom(store, "markets", "hbm foundry earnings port 8081", "day2")
    await _seed_chapter(store, "hbm foundry earnings stocks", [mk1, mk2])
    g = await sema._grade_designed_month(store, m, _tqm(m))
    assert g["verdict"] == "INVALID" and g["stage_a"]["a3_provenance_ok"] is False


@pytest.mark.asyncio
async def test_grade_inconclusive_on_correction_arc_stale_active(store):
    """B4: the stale value still asserted current in an active fact -> arc fails -> INCONCLUSIVE."""
    m = await _seed_holds(store)
    # A stale row survives active anywhere -> B4 must fail (the correction never truly superseded).
    await store.store_fact(key="remember/setup", value="our vLLM server listens on port 8000",
                           tags=["remember"], confidence=0.9, provenance="designed-day1")
    g = await sema._grade_designed_month(store, m, _tqm(m))
    assert g["verdict"] == "INCONCLUSIVE" and g["failing_stage"] == "B4"
    assert g["stage_b"]["stale_active"] is True


def test_shipped_manifest_is_well_formed():
    """The pre-committed designed-month manifest: 6 expected-chapter topics each spread across
    >= 2 days (the >=2-sitting stability bar), a correction arc, and query/topic coverage."""
    m = sema._load_manifest(_SCRIPTS / "sema05_designed_month_manifest.json")
    expected = {t: meta for t, meta in m["topics"].items() if meta.get("expected_chapter")}
    assert len(expected) == 6, expected.keys()
    for t, meta in expected.items():
        assert len(meta["days"]) >= 2, f"{t} must recur across >= 2 days"
        assert meta.get("keywords"), f"{t} needs Stage-C keywords"
    arc = m["correction_arc"]
    assert arc["stale"] and arc["corrected"] and arc["stale"] != arc["corrected"]
    topics_used = {q["topic"] for q in m["queries"]}
    assert set(expected) <= topics_used and "noise" in topics_used  # every topic + distractors driven


# --- end-to-end reachability (offline) --------------------------------------------------

@pytest.mark.asyncio
async def test_offline_manifest_mode_runs_and_grades(tmp_path: Path):
    """`--manifest --offline` drives designed-{day} sittings through the REAL loop, mines typed
    sem/ atoms with per-atom source provenance, clusters them into a chapter, and grades to a
    verdict with the full Stage A/B/C structure — proving the mode is wired end to end."""
    manifest = {
        "days": ["day1", "day2"],
        "topics": {"subagents": {"expected_chapter": True, "days": ["day1", "day2"],
                                 "keywords": ["subagent", "summarizer", "citation"]},
                   "noise": {"expected_chapter": False}},
        "queries": [
            {"id": "d1-q1", "day": "day1", "topic": "subagents",
             "text": "i am building a summarizer subagent for the harness"},
            {"id": "d1-q2", "day": "day1", "topic": "noise", "text": "whats the capital of mongolia"},
            {"id": "d2-q1", "day": "day2", "topic": "subagents",
             "text": "i am building a citation subagent for the harness"},
            {"id": "d2-q2", "day": "day2", "topic": "noise", "text": "convert 5 kilometers to miles"},
        ],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    args = sema._parse_args([
        "--offline", "--manifest", str(mpath), "--store", str(tmp_path / "store"),
        "--results", str(tmp_path / "results"), "--agent", AGENT,
    ])
    code = await sema.run(args)
    assert code == 0

    v = json.loads((tmp_path / "results" / "verdict.json").read_text(encoding="utf-8"))
    assert v["mode"] == "designed_month_manifest"
    assert v["verdict"] in {"HOLDS", "KILL", "INCONCLUSIVE", "INVALID"}
    # Full pre-committed structure present (Stages A/B/C + sensitivity).
    for k in ("a1_recall", "a2_kill", "a3_provenance_ok", "sem_atoms"):
        assert k in v["stage_a"]
    for k in ("b1_ok", "b2_ok", "b3_ok", "b4_ok", "ari", "per_chapter", "operational_rows_in_chapter"):
        assert k in v["stage_b"]
    assert isinstance(v["stage_c"], list) and "sensitivity" in v
    # NON-VACUOUS: the two day-distinct subagent atoms cluster into one chapter and grade clean.
    assert v["stage_a"]["sem_atoms"] == 2
    assert v["verdict"] == "HOLDS" and v["stage_b"]["ari"] == 1.0
    assert [pc["label"] for pc in v["stage_b"]["per_chapter"]] == ["subagents"]
    assert v["days"] == 2 and [s["session_id"] for s in v["sittings"]] == ["designed-day1", "designed-day2"]
