"""EVAL HARDENING for the designed-month grader (`scripts/sema05_month_in_a_day.py`).

Two eval-validity defects found in the 3-run D1 replication
(`.planning/runs/d1-replication-20260712/ANALYSIS.md` §6.4 + §9) and their ADDITIVE fixes:

  FIX 1 (§6.4) — paraphrase-blind topic attribution. `_attribute_topic` (v1) attributes an
      atom's ground-truth topic by best >=5-char-token overlap against its PROVENANCE-DAY
      manifest queries ONLY. Real atoms like run2's "TSMC manufactures ~90% of the world's
      advanced AI chips" share no >=5-char token with any day1 query ("TSMC" is 4 chars,
      filtered) -> gt=None -> the markets chapter's precision/ARI is dragged down even though
      the CLUSTERING is correct. `_attribute_topic_v2` is a paraphrase-tolerant attributor
      REPORTED ALONGSIDE v1 (never replacing it): v1 stays the grading ruler; v2 fields
      (ari_v2, per-chapter attribution_v2, attribution_divergence) are added.

  FIX 2 (§9) — B4 scenario-validity collision. run3's tool-capable model curl-verified the
      REAL vLLM server (genuinely on port 8000) and correctly reported it, superseding the
      scripted "corrected to 8081" atom via the store's normal same-key rule — correct
      behavior graded as a B4 failure. The grader now detects the collision deterministically
      from store+history and, via an EXPLICIT disclosed path, may treat an EXCUSED B4 as
      non-failing (new fields b4_scenario_collision, b4_excused; b4_ok keeps its semantics).

All fixture literals are lifted VERBATIM from the real run stores/history under
`.planning/runs/d1-replication-20260712/`; each is cited in its test.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.memory.sqlite import MemoryStore

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import sema05_month_in_a_day as sema  # noqa: E402

AGENT = "orchestrator"
_RUNS = _REPO / ".planning" / "runs" / "d1-replication-20260712"
MANIFEST = sema._load_manifest(_SCRIPTS / "sema05_designed_month_manifest.json")


def _day_queries(manifest: dict, day: str) -> list[dict]:
    return [{"id": q["id"], "topic": q["topic"], "text": q["text"]}
            for q in manifest["queries"] if q["day"] == day]


def _all_queries(manifest: dict) -> list[dict]:
    return [{"id": q["id"], "topic": q["topic"], "text": q["text"]} for q in manifest["queries"]]


# ===========================================================================================
# FIX 1 — _attribute_topic_v2 (paraphrase-tolerant, ADDITIVE). Pure-function proofs on the
# exact atoms named in ANALYSIS §6.4 (run2) and the run3 rescue case.
# ===========================================================================================

# The §6.4 atoms, verbatim from run2 store (sem/semis-research/06293689 + 00dc90cc, day1).
_TSMC_1 = "TSMC manufactures ~90% of the world's advanced AI chips"
_TSMC_2 = "TSMC has a massive lead in N4 and N3 processes"


def test_v1_is_paraphrase_blind_returns_none():
    """DEFECT (§6.4), still true: v1 scores 0 for the run2 TSMC atoms against every day1 query
    (no shared >=5-char token) -> gt=None. Locks v1's behavior so the additive fix can't drift it."""
    d1 = _day_queries(MANIFEST, "day1")
    assert sema._attribute_topic(_TSMC_1, d1) is None
    assert sema._attribute_topic(_TSMC_2, d1) is None


def test_v2_rescues_ticker_atoms_via_expected_atoms():
    """FIX 1 (a)+(c): with the manifest's per-topic `expected_atoms` content (here the real run2
    atoms) + ticker-length tokens, v2 attributes both §6.4 atoms to `markets`; v1 stays None."""
    m = json.loads(json.dumps(MANIFEST))
    m["topics"]["markets"]["expected_atoms"] = [_TSMC_1, _TSMC_2]
    d1, allq = _day_queries(m, "day1"), _all_queries(m)
    for atom in (_TSMC_1, _TSMC_2):
        assert sema._attribute_topic(atom, d1) is None                     # v1 unchanged
        topic, score = sema._attribute_topic_v2(atom, d1, allq, m)
        assert topic == "markets" and score >= 2, (atom, topic, score)


def test_v2_rescues_paraphrase_via_all_days_and_keywords():
    """FIX 1 (b)+(c): run3's sem/user/0b425819 (prov day1) "has knee pain after Tuesday interval
    sessions" is gt=None under v1 (no day1 race query overlap), but v2 attributes it to
    race_training via ALL-days queries + the topic keywords ('knee','intervals') + stemming."""
    atom = "has knee pain after Tuesday interval sessions"  # run3 sem/user/0b425819, designed-day1
    d1, allq = _day_queries(MANIFEST, "day1"), _all_queries(MANIFEST)
    assert sema._attribute_topic(atom, d1) is None                          # v1: None
    topic, score = sema._attribute_topic_v2(atom, d1, allq, MANIFEST)
    assert topic == "race_training" and score >= 2, (topic, score)


def test_v2_never_overrides_a_v1_label():
    """ADDITIVE / comparability sacred: v2 is a strict SUPERSET of v1 — for any atom v1 already
    labels, v2 returns the SAME topic (cascade: v1 first). run3 sem/semis-research/6c8db501
    "User cares most about earnings quality, not momentum" is markets under BOTH."""
    atom = "User cares most about earnings quality, not momentum"  # run3, designed-day1
    d1, allq = _day_queries(MANIFEST, "day1"), _all_queries(MANIFEST)
    v1 = sema._attribute_topic(atom, d1)
    assert v1 == "markets"
    topic, _score = sema._attribute_topic_v2(atom, d1, allq, MANIFEST)
    assert topic == v1  # never re-labels a v1-owned atom


def test_v2_min_hits_suppresses_single_token_homograph():
    """FIX 1 guard: a single generic/homograph token must NOT mint a spurious label. run2
    sem/project-structure/00ffc6f2 (a Kyoto itinerary ending in "...Sagano Romantic Train")
    shares only the stem 'train'~'training' with race_training — 1 hit < min_rescue_hits=2 —
    so v2 stays None rather than mis-attributing a kyoto atom to race_training."""
    atom = ("Day 2 itinerary includes Arashiyama, Tenryu-ji, Togetsukyo Bridge, "
            "Okochi Sanso, and Sagano Romantic Train")  # run2, designed-day2
    d2, allq = _day_queries(MANIFEST, "day2"), _all_queries(MANIFEST)
    assert sema._attribute_topic(atom, d2) is None
    topic, score = sema._attribute_topic_v2(atom, d2, allq, MANIFEST)
    assert topic is None, (topic, score)


# ===========================================================================================
# FIX 2 — B4 scenario-collision detector. Pure-function proofs on the exact run3 supersede
# chain (facts) + curl probes (history), and the run1/run2 non-collision controls.
# ===========================================================================================

def _fact(id, key, status, superseded_by, value):  # noqa: A002 — mirrors the Fact attr names
    return SimpleNamespace(id=id, key=key, status=status, superseded_by=superseded_by, value=value)


# run3 vllm-server-port chain, verbatim (memory.db ids 16->19->22).
def _run3_collision_facts():
    return [
        _fact(16, "vllm-server-port", "superseded", 19, "vLLM server listens on port 8000."),
        _fact(19, "vllm-server-port", "superseded", 22, "vLLM server listens on port 8081."),
        _fact(22, "vllm-server-port", "active", None,
              "vLLM server listens on port 8000 (confirmed active, serving qwen3.6-35b-a3b)."),
    ]


# run3 day3 curl probes, verbatim (history.jsonl rows 425/426/428/429).
def _run3_probe_history():
    return [
        {"type": "assistant_message", "session_id": "designed-day3", "content": None,
         "tool_calls": [{"id": "tc1", "name": "bash_exec",
                         "arguments": {"command": "curl -s http://localhost:8081/v1/models | python3 -m json.tool"}}]},
        {"type": "tool_result", "session_id": "designed-day3", "tool_name": "bash_exec",
         "content": "Expecting value: line 1 column 1 (char 0)\n"},
        {"type": "assistant_message", "session_id": "designed-day3", "content": None,
         "tool_calls": [{"id": "tc2", "name": "bash_exec",
                         "arguments": {"command": "curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>&1 | head -20"}}]},
        {"type": "tool_result", "session_id": "designed-day3", "tool_name": "bash_exec",
         "content": '{\n    "object": "list",\n    "data": [\n        {\n            "id": "qwen3.6-35b-a3b",\n            "object": "model"\n'},
    ]


_ARC = {"topic": "gpu_ops", "stale": "port 8000", "corrected": "port 8081"}


def test_scenario_collision_true_on_run3():
    """§9, verbatim: the corrected atom (id19, '8081') was superseded by a newer SAME-KEY active
    atom (id22, '8000 confirmed active') AND the history shows the model curl-verified the real
    server on the stale port -> a scenario-validity collision, not a memory-correction failure."""
    assert sema._b4_scenario_collision(_run3_collision_facts(), _run3_probe_history(), _ARC) is True


def test_scenario_collision_history_is_the_discriminator():
    """Run1/2-style: the SAME store supersede chain but WITHOUT the tool-probe history is NOT a
    collision (the model never verified the environment) — the history half is load-bearing."""
    assert sema._b4_history_confirms_stale(_run3_probe_history(), ["8000"], ["8081"]) is True
    assert sema._b4_history_confirms_stale([], ["8000"], ["8081"]) is False
    assert sema._b4_scenario_collision(_run3_collision_facts(), [], _ARC) is False


def test_scenario_collision_false_on_quarantine_verbatim():
    """Store-half guard (run1/run2 false-positive): a raw `correction/quarantine/...` row carries
    the user's verbatim text "...moved to 8081, not 8000" (BOTH numbers) and drains superseded->
    active within its own key. That is NOT a value collision — quarantine keys are excluded and a
    superseder that still asserts the corrected value is not a revert-to-stale."""
    q = "actually no — correction on my setup note: we moved the vLLM server to port 8081, not 8000. update that"
    facts = [
        _fact(16, "correction/quarantine/b020aed6", "superseded", 73, q),
        _fact(73, "correction/quarantine/b020aed6", "active", None, q),
    ]
    assert sema._b4_scenario_collision(facts, _run3_probe_history(), _ARC) is False


# ===========================================================================================
# FIX 3 (#42) — _looks_like_probe_error: only connection-level signatures count as a dead probe.
# Bare generics ("not found"/"refused"/"timeout") collided with ordinary app text and HTTP error
# BODIES — a served body proves the port IS listening (the opposite of a dead probe), which is how
# a false "dead" could excuse a real B4 regression. Empty/denied/JSON-decode-on-empty stay dead.
# ===========================================================================================

@pytest.mark.parametrize("text, expected", [
    # dead: empty, harness denial, and the REAL run3 signature (curl -s <dead port> | json.tool).
    ("", True),
    ("   \n\t ", True),
    ("[DENIED]", True),
    ("Expecting value: line 1 column 1 (char 0)\n", True),
    # dead: connection-level curl/OS failures (never present in a body a LIVE port serves).
    ("curl: (7) Failed to connect to localhost port 8081: Connection refused", True),
    ("curl: (7) Could not connect to server", True),
    ("curl: (56) Connection reset by peer", True),
    ("curl: (52) Empty reply from server", True),
    ("curl: (28) Connection timed out after 5001 milliseconds", True),
    ("curl: (28) Operation timed out after 5000 milliseconds with 0 bytes received", True),
    # COLLISIONS the bug is about — a served body / app text is NOT a dead probe -> must be False.
    ('{"detail":"Not Found"}', False),                          # 404 body => something IS listening
    ("the request was refused by the venue", False),           # app text, not a connect error
    ("sorry, that page was not found", False),                 # app text
    ('{"object": "list", "data": [{"id": "qwen"}]}', False),   # real live-server body (run3 :8000)
    ("your session timed out, please sign in again", False),   # bare 'timed out' in app text
])
def test_looks_like_probe_error_only_connection_level(text, expected):
    assert sema._looks_like_probe_error(text) is expected


# ===========================================================================================
# FIX 2 — end-to-end via _grade_designed_month (hermetic store built with the seed builders):
# an excused B4 is disclosed and (only via the explicit path) treated as non-failing, while a
# no-collision store keeps byte-identical B4 behavior.
# ===========================================================================================

_TS = {"n": 5000}


async def _seed_atom(store, topic, claim, day):
    _TS["n"] += 1
    await store.append_history({"v": 1, "agent_id": AGENT, "type": "user_message", "id": f"h{_TS['n']}",
                                "session_id": f"designed-{day}", "ts": _TS["n"], "content": claim})
    return await store.store_fact(key=f"sem/{topic}/{_TS['n']}", value=claim,
                                  tags=["sem", "pending_consolidation"], confidence=0.65,
                                  provenance=f"designed-{day}", node_kind="fact")


async def _seed_chapter(store, value, members):
    _TS["n"] += 1
    sch = await store.store_fact(key=f"schema/cluster/{_TS['n']}", value=value,
                                 tags=["schema", "tier:schema", "depth:1"], confidence=0.8,
                                 node_kind="schema", provenance="cluster:designed")
    for m in members:
        await store.add_edge(sch.id, m.id, "member_of")
    return sch


def _collision_manifest():
    # arc topic = markets; a markets member carries the corrected 8081 so it stays REACHABLE.
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


async def _seed_clean_two_chapters(store):
    mk1 = await _seed_atom(store, "markets", "researching hbm foundry stocks", "day1")
    mk2 = await _seed_atom(store, "markets", "hbm foundry earnings port 8081", "day2")
    ky1 = await _seed_atom(store, "kyoto", "planning a kyoto ryokan trip", "day1")
    ky2 = await _seed_atom(store, "kyoto", "kyoto ryokan onsen recommendation", "day2")
    await _seed_chapter(store, "hbm foundry earnings stocks", [mk1, mk2])
    await _seed_chapter(store, "kyoto ryokan onsen trip", [ky1, ky2])


async def _add_supersede_collision(store):
    """The store's normal same-key rule: minting 8081 then a tool-verified 8000 supersedes the
    8081 atom -> stale ('port 8000') is active AGAIN under vllm-server-port (mirrors run3 id19->22)."""
    await store.store_fact(key="vllm-server-port", value="vLLM server listens on port 8081",
                           tags=["remember"], confidence=0.9, provenance="designed-day2")
    await store.store_fact(key="vllm-server-port",
                           value="vLLM server listens on port 8000 (confirmed active, serving qwen)",
                           tags=["remember"], confidence=0.9, provenance="designed-day2")


async def _add_probe_history(store):
    for i, rec in enumerate(_run3_probe_history()):
        await store.append_history({"v": 1, "id": f"probe{i}", "agent_id": AGENT, "ts": 9000, **rec})


@pytest.fixture
async def store(tmp_path: Path):
    s = MemoryStore(agent_id=AGENT, division_id="", org_id="", base_dir=str(tmp_path))
    await s.open()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_b4_excused_collision_does_not_fail_the_month(store):
    """FIX 2 disclosed path: b4_ok=False (stale '8000' re-minted active) but the collision is
    detected (supersede chain + curl probes) so b4_excused=True and the month is NOT failed on
    B4 — the composite verdict is HOLDS via the explicit b4_effective path, never a silent flip."""
    m = _collision_manifest()
    await _seed_clean_two_chapters(store)
    await _add_supersede_collision(store)   # stale 8000 active again (id analog: run3 19->22)
    await _add_probe_history(store)         # the model tool-verified the real server on 8000

    g = await sema._grade_designed_month(store, m, _tqm(m))
    sb = g["stage_b"]
    assert sb["b4_ok"] is False and sb["stale_active"] is True   # raw semantics UNCHANGED
    assert sb["correction_reachable"] is True                    # corrected 8081 still reachable
    assert sb["b4_scenario_collision"] is True
    assert sb["b4_excused"] is True
    assert g["verdict"] == "HOLDS" and g["failing_stage"] is None, g


@pytest.mark.asyncio
async def test_b4_no_collision_is_byte_identical(store):
    """Control: the SAME stale-active store WITHOUT the tool-probe history is NOT a collision, so
    B4 behavior is byte-identical to today — b4_scenario_collision=False, b4_excused=False, and
    the month is INCONCLUSIVE failing B4 (run1/2-style: no probe = no excuse)."""
    m = _collision_manifest()
    await _seed_clean_two_chapters(store)
    await _add_supersede_collision(store)   # stale active, but NO probe history added

    g = await sema._grade_designed_month(store, m, _tqm(m))
    sb = g["stage_b"]
    assert sb["b4_ok"] is False and sb["stale_active"] is True
    assert sb["b4_scenario_collision"] is False
    assert sb["b4_excused"] is False
    assert g["verdict"] == "INCONCLUSIVE" and g["failing_stage"] == "B4", g


# ===========================================================================================
# GOLDEN — the ultimate lock: grade the REAL run3 store and prove (1) EVERY v1 field is
# byte-identical to the committed verdict.json (the additive change flips nothing) and (2) the
# new v2 + collision fields carry the exact grounded values. Skips where .planning is absent
# (git-ignored), so the hermetic tests above are the portable proof.
# ===========================================================================================

_RUN3 = _RUNS / "run3"


@pytest.mark.asyncio
async def test_grade_run3_v1_byte_identical_and_v2_added(tmp_path: Path):
    if not (_RUN3 / "store").is_dir():
        pytest.skip("run3 fixture store absent (.planning is git-ignored)")
    committed = json.loads((_RUN3 / "results" / "verdict.json").read_text(encoding="utf-8"))
    csb, csa = committed["stage_b"], committed["stage_a"]

    shutil.copytree(_RUN3 / "store", tmp_path / "store")
    s = MemoryStore(agent_id=AGENT, division_id="", org_id="", base_dir=str(tmp_path / "store"))
    await s.open()
    try:
        g = await sema._grade_designed_month(s, MANIFEST, _tqm(MANIFEST))
    finally:
        await s.close()
    sb, sa = g["stage_b"], g["stage_a"]

    # (1) EVERY v1 field byte-identical to the committed run (FIX 1's "zero verdict flips" proof).
    assert g["verdict"] == committed["verdict"] == "INCONCLUSIVE"
    assert g["failing_stage"] == committed["failing_stage"] == "B1"   # B1 dominates; B4 excusal can't flip it
    assert sb["ari"] == csb["ari"] == 0.656
    for k in ("b1_ok", "b2_ok", "b3_ok", "b4_ok"):
        assert sb[k] == csb[k], k
    assert sb["b4_ok"] is False and csb["b4_ok"] is False             # raw B4 semantics UNCHANGED
    assert sa["a1_recall"] == csa["a1_recall"] and sa["sem_atoms"] == csa["sem_atoms"] == 61
    assert g["byte_stable"] == committed["byte_stable"]
    for got, exp in zip(sb["per_chapter"], csb["per_chapter"]):
        assert (got["label"], got["precision"], got["recall"]) == (exp["label"], exp["precision"], exp["recall"])

    # (2) The additive v2 fields, with the exact values the real store yields.
    assert sb["attribution_divergence"] == 1                          # id86 knee-atom rescued
    assert sb["ari_v2"] == 0.668 and sb["ari_v2"] > sb["ari"]         # v2 recovers > v1
    for pc in sb["per_chapter"]:
        assert "attribution_v2" in pc and "topic" in pc["attribution_v2"] and "score" in pc["attribution_v2"]
    race = next(pc for pc in sb["per_chapter"] if pc["label"] == "race_training")
    assert race["precision"] == 0.9 and race["attribution_v2"]["score"] == 1.0  # v2 recovers the None member

    # (2b) FIX 2 fields on the real §9 case: collision detected, B4 excused (but B1 still fails).
    assert sb["b4_scenario_collision"] is True
    assert sb["b4_excused"] is True
