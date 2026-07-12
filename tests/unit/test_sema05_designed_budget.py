"""FIX 1 (run-10): the designed-month provable consolidates exactly ONCE, so the chapter-writer's
production default write_budget=3 structurally starves a month with >3 expected chapters — run-10's
markets cluster (4th biggest, after a size tiebreak) was never attempted, capping B1. The runner
must derive a NON-STARVING budget from the manifest and thread it to write_cluster_schemas; the
production default of 3 (tuned for recurring idle cycles) stays untouched.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.config.models import MemoryConsolidationConfig
from localharness.memory.consolidation import ConsolidationPass
from localharness.memory.embeddings import HashingEmbedder
from localharness.memory.sqlite import MemoryStore

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "scripts" / "sema05_designed_month_manifest.json"


def _load_script():
    """Import the standalone runner by path (scripts/ is not a package)."""
    path = _REPO / "scripts" / "sema05_month_in_a_day.py"
    spec = importlib.util.spec_from_file_location("sema05_month_in_a_day", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_schema_write_budget_from_manifest_beats_default_and_covers_chapters():
    """The runner derives its designed-month chapter budget straight from the real manifest: >3
    (the run-10 starvation is gone) AND >= every expected-chapter topic (each one can be attempted)."""
    sema = _load_script()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    budget = sema._schema_write_budget(manifest)
    expected = sum(1 for m in manifest["topics"].values() if m.get("expected_chapter"))
    assert budget > 3                          # the starving production default is overridden
    assert budget >= expected                  # every expected chapter can be attempted
    assert budget == len(manifest["topics"]) + 1  # at most len(topics) clusters can form (+1 headroom)


class _UngroundedLLM:
    """A chapter with NO member tokens: every attempted cluster trips the grounding KILL BEFORE any
    write, but the attempt is still logged — so len(attempts) == clusters actually attempted."""

    async def complete(self, prompt: str) -> str:
        return "zzz qqq xyzzy foobar" if "Write ONE" in prompt else ""


def _fake_cluster(i: int):
    m = SimpleNamespace(key=f"learned/tool{i}/k", value=f"topic {i} content words here",
                        id=1000 + i, source="test", provenance="", tags=[])
    return SimpleNamespace(members=[m], aux_members=[], sessions={f"s{i}a", f"s{i}b"}, depth=0)


@pytest.mark.asyncio
async def test_manifest_budget_reaches_writer_and_unstarves_attempts(tmp_path, monkeypatch):
    """The manifest-derived budget REACHES write_cluster_schemas through the real ConsolidationPass
    (cfg.schema_write_budget -> write_budget) and lets ALL >3 eligible clusters be attempted, where
    the production default of 3 starves the 4th/5th (the run-10 markets failure)."""
    from localharness.memory import chapter_writer

    async def _five_clusters(store, **kw):
        return [_fake_cluster(i) for i in range(5)]

    monkeypatch.setattr(chapter_writer, "find_stable_clusters", _five_clusters)

    sema = _load_script()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    budget = sema._schema_write_budget(manifest)

    def _cfg(wb):
        return MemoryConsolidationConfig(
            schema_writer_enabled=True, mining_enabled=False, reconcile_enabled=False,
            tag_discovery_enabled=False, mint_tagging_enabled=False, schema_write_budget=wb)

    async def _attempts(wb):
        s = MemoryStore(agent_id="budget-agent", division_id="", org_id="",
                        base_dir=str(tmp_path / str(wb)))
        await s.open()
        try:
            rep = await ConsolidationPass(s, _cfg(wb), llm=_UngroundedLLM(),
                                          embedder=HashingEmbedder()).run()
            return len(rep.schema_attempts)
        finally:
            await s.close()

    assert await _attempts(3) == 3          # the starving production default: only 3 of 5 attempted
    assert await _attempts(budget) == 5     # the manifest budget attempts every eligible cluster


# ---------------------------------------------------------------------------
# FIX 1 (extraction-yield): mining's write_budget is a GLOBAL per-pass cap — once it mints, the
# outer walk ABORTS every remaining chunk. The designed month consolidates exactly ONCE, so the
# production default (25) silently dropped run-6's transcript TAIL (41% never reached the LLM;
# sem_atoms landed at exactly 25). The runner must thread a non-starving single-pass-eval bound;
# the production default (25) stays untouched (owner decides that separately).
# ---------------------------------------------------------------------------


def test_mining_write_budget_beats_starving_default():
    """The runner's single-pass-eval mining bound is well above the production default (25) that
    starved run-6, with headroom over any legitimate 3-day month's atom count (~40 densest observed,
    ~200 records max) — so the global write cap can never bite mid-transcript in the one grading pass."""
    sema = _load_script()
    budget = sema._mining_write_budget()
    assert budget > 25          # clears the run-6 ceiling of exactly 25 with generous headroom
    assert budget == 500        # the documented single-pass-eval constant (NOT a production change)


@pytest.mark.asyncio
async def test_run_designed_month_threads_nonstarving_mining_budget(tmp_path, monkeypatch):
    """FIX 1 wiring: the grading-phase consolidation config the RUNNER actually builds (a real
    offline manifest run) carries mining_write_budget > 25 — proving the eval bound reaches the
    single pass, not just that a helper returns it (owner rule: no green test on unwired code)."""
    sema = _load_script()
    captured: dict = {}
    real_pass = sema.ConsolidationPass

    def _capturing(store, cfg, **kw):
        captured["mining_write_budget"] = cfg.mining_write_budget
        return real_pass(store, cfg, **kw)

    monkeypatch.setattr(sema, "ConsolidationPass", _capturing)

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
        "--results", str(tmp_path / "results"), "--agent", "orchestrator",
    ])
    code = await sema.run(args)

    assert code == 0
    assert captured.get("mining_write_budget", 25) > 25          # runner threaded a non-starving budget
    assert captured["mining_write_budget"] == sema._mining_write_budget()


# ---------------------------------------------------------------------------
# SWEEP UNLOCK (--cfg-overrides): the designed-month runner hardcoded its consolidation config, so a
# sweep could only vary --idle-passes. A JSON object of MemoryConsolidationConfig field overrides now
# threads into the grading-phase cfg (built by the pure seam _build_consolidation_cfg). Overrides win
# LAST over the three eval defaults; a typo'd/unknown knob must FAIL LOUDLY (extra=forbid) instead of
# silently no-op'ing and voiding a whole sweep run; default "{}" reproduces today's build exactly.
# ---------------------------------------------------------------------------

_TINY_MANIFEST = {
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


def test_cfg_overrides_reach_constructed_config():
    """(i) A --cfg-overrides knob lands on the cfg the runner builds — assert the FIELD VALUE on the
    real cfg, not just that the helper accepts the arg. The three eval defaults still hold alongside."""
    sema = _load_script()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cfg = sema._build_consolidation_cfg(manifest, '{"mining_novelty_fold_threshold": 0.35}')
    assert cfg.mining_novelty_fold_threshold == 0.35
    assert cfg.reconcile_enabled is True
    assert cfg.schema_write_budget == sema._schema_write_budget(manifest)
    assert cfg.mining_write_budget == sema._mining_write_budget()


def test_cfg_overrides_win_last_over_eval_defaults():
    """An override colliding with one of the three explicit eval kwargs (reconcile_enabled /
    schema_write_budget / mining_write_budget) WINS LAST via dict-merge — never a duplicate-keyword
    TypeError at construction."""
    sema = _load_script()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cfg = sema._build_consolidation_cfg(
        manifest, '{"reconcile_enabled": false, "mining_write_budget": 77}')
    assert cfg.reconcile_enabled is False        # override beat the explicit reconcile_enabled=True
    assert cfg.mining_write_budget == 77         # override beat _mining_write_budget()
    assert cfg.schema_write_budget == sema._schema_write_budget(manifest)  # untouched default remains


def test_cfg_overrides_default_reproduces_today_field_for_field():
    """(ii) Default "{}" must reproduce the original hardcoded construction EXACTLY — the unlock is
    additive, zero behaviour change when unused."""
    sema = _load_script()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    built = sema._build_consolidation_cfg(manifest, "{}")
    today = MemoryConsolidationConfig(
        reconcile_enabled=True,
        schema_write_budget=sema._schema_write_budget(manifest),
        mining_write_budget=sema._mining_write_budget())
    assert built.model_dump() == today.model_dump()


def test_cfg_overrides_unknown_field_raises_loudly():
    """(iii) A typo'd/unknown knob must FAIL LOUDLY (extra=forbid) — a silently ignored override
    would void a whole sweep run."""
    from pydantic import ValidationError
    sema = _load_script()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        sema._build_consolidation_cfg(manifest, '{"not_a_real_knob": 1}')


def test_cfg_overrides_invalid_json_raises_loudly():
    """Malformed JSON propagates at startup — no silent fallback to defaults."""
    sema = _load_script()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    with pytest.raises(json.JSONDecodeError):
        sema._build_consolidation_cfg(manifest, "{not json")


@pytest.mark.asyncio
async def test_run_designed_month_wires_cfg_overrides_arg(tmp_path, monkeypatch):
    """WIRING: --cfg-overrides on the real CLI reaches the cfg the runner hands ConsolidationPass in
    the grading pass (owner rule: no green test on unwired code)."""
    sema = _load_script()
    captured: dict = {}
    real_pass = sema.ConsolidationPass

    def _capturing(store, cfg, **kw):
        captured["fold"] = cfg.mining_novelty_fold_threshold
        return real_pass(store, cfg, **kw)

    monkeypatch.setattr(sema, "ConsolidationPass", _capturing)

    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(_TINY_MANIFEST), encoding="utf-8")
    args = sema._parse_args([
        "--offline", "--manifest", str(mpath), "--store", str(tmp_path / "store"),
        "--results", str(tmp_path / "results"), "--agent", "orchestrator",
        "--cfg-overrides", '{"mining_novelty_fold_threshold": 0.42}',
    ])
    code = await sema.run(args)

    assert code == 0
    assert captured.get("fold") == 0.42


def test_root_agent_config_denies_subject_ops_commands():
    """The 2026-07-11 sweep casualties: the subject model used bash_exec to docker-stop/rm its
    own vLLM server (4 strikes, logged in the casualty stores). The eval policy must deny the
    REAL strike commands — replayed verbatim here — while leaving ordinary commands alone."""
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.core.types import ToolCall

    sema = _load_script()
    a_cfg = sema._root_agent_config("orchestrator")
    ev = PermissionEvaluator()

    strikes = [
        "docker stop vllm-sweep-35b && docker rm vllm-sweep-35b",
        "cd /home/openclaw-user && PREFIX_CACHING=1 ./vllm-qwen-nvfp4-run.sh 2>&1",
        "docker stop vllm-qwen36-35b-official && sleep 3 && docker ps -a | grep vllm",
        'kill -9 2990349 2990862 2>/dev/null; sleep 2; echo "vLLM processes stopped"',
        "systemctl list-units --type=service | grep -i serve",
        "pkill -f vllm",
    ]
    for cmd in strikes:
        res = ev.evaluate(ToolCall(name="bash_exec", arguments={"command": cmd}), a_cfg.permissions)
        assert res.denied, f"strike command not denied: {cmd}"

    innocuous = [
        "echo hello",
        "grep skill notes.md",   # 'skill' must not trip the kill pattern
        "ls -la /tmp",
    ]
    for cmd in innocuous:
        res = ev.evaluate(ToolCall(name="bash_exec", arguments={"command": cmd}), a_cfg.permissions)
        assert not res.denied, f"innocuous command wrongly denied: {cmd} ({res.reason})"
