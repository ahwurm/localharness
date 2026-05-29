import shutil
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from localharness.config.models import AgentConfig, OrgConfig
from localharness.core.bus import EventBus
from localharness.core.events import ScenarioCompleted
from localharness.bench.runner import resolve_run_path
from localharness.memory.sqlite import MemoryStore


@pytest.fixture
def minimal_agent_config() -> AgentConfig:
    return AgentConfig(name="test-agent", role="Test agent for unit tests.")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "agents").mkdir()
    (tmp_path / "divisions").mkdir()
    return tmp_path


@pytest.fixture
def bus() -> EventBus:
    """In-memory event bus for tests. No JSONL persistence."""
    return EventBus()


@pytest.fixture
def bus_with_persistence(tmp_path: Path) -> EventBus:
    """Event bus with JSONL persistence to tmp dir."""
    return EventBus(persist_path=tmp_path / "events.jsonl")


@pytest.fixture
def components_home(tmp_path, monkeypatch):
    """Hermetic LocalHarness home dir.

    Creates tmp_path/.localharness/ with empty config.yaml and audit.jsonl;
    sets LOCALHARNESS_HOME so ConfigLoader, overlay writer, and EventBus
    all isolate to tmp_path. Yields the home Path.
    """
    home = tmp_path / ".localharness"
    home.mkdir(parents=True, exist_ok=True)
    # Minimal valid HarnessConfig YAML for loader smoke
    (home / "config.yaml").write_text(
        "version: '1'\n"
        "provider:\n"
        "  provider_type: ollama\n"
        "  base_url: http://localhost:11434/v1\n"
        "  default_model: test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home))
    return home


@dataclass
class FakeToolCall:
    """Minimal tool call object for mock LLM responses."""
    id: str
    name: str = ""
    arguments: dict = field(default_factory=dict)

    @property
    def function(self):
        import json
        class _Fn:
            pass
        fn = _Fn()
        fn.name = self.name
        fn.arguments = json.dumps(self.arguments)
        return fn


@dataclass
class FakeCompletionUsage:
    """Mirrors openai.types.completion_usage.CompletionUsage for tests."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class FakeLLMResponse:
    """Scripted LLM response for MockLLMClient."""
    content: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    usage: "FakeCompletionUsage | None" = None


class MockLLMClient:
    """Fake LLM client that returns scripted responses in sequence.

    return_tuple defaults to True post-10-01 (production stream_complete returns
    (message, usage) tuples). Set False only for legacy tests that intentionally
    test the bare-message shape.
    """

    def __init__(self, responses: list[FakeLLMResponse], return_tuple: bool = True) -> None:
        self._responses = list(responses)
        self._index = 0
        self._return_tuple = return_tuple

        class _Config:
            tool_call_mode = "native"
            context_window = 128_000

        self.config = _Config()

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        if self._index >= len(self._responses):
            resp = FakeLLMResponse(content="Done.")
        else:
            resp = self._responses[self._index]
            self._index += 1
        if self._return_tuple:
            return resp, resp.usage
        return resp


@pytest.fixture
def mock_llm_client():
    """Factory fixture: call with a list of FakeLLMResponse objects."""
    def _factory(responses: list[FakeLLMResponse], return_tuple: bool = True) -> MockLLMClient:
        return MockLLMClient(responses, return_tuple=return_tuple)
    _factory.Response = FakeLLMResponse
    _factory.ToolCall = FakeToolCall
    _factory.Usage = FakeCompletionUsage
    return _factory


@pytest.fixture
async def memory_store(tmp_path: Path) -> MemoryStore:
    """Fresh MemoryStore with temporary database."""
    store = MemoryStore(
        agent_id="test-agent",
        division_id="test-div",
        org_id="default",
        base_dir=str(tmp_path),
    )
    await store.open()
    yield store
    await store.close()


# -----------------------------------------------------------------------------
# Phase 11 bench harness fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def bench_results_dir(tmp_path: Path) -> Path:
    """tmp_path-scoped bench/results/{model}/{scenario}/ tree."""
    root = tmp_path / "bench" / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def fixture_scenario_path() -> Path:
    """Path to the static minimal_golden_scenario.yaml fixture."""
    return Path(__file__).parent / "fixtures" / "bench" / "minimal_golden_scenario.yaml"


@pytest.fixture
def fixture_rubric_scenario_path() -> Path:
    return Path(__file__).parent / "fixtures" / "bench" / "minimal_rubric_scenario.yaml"


@pytest.fixture
def fixture_invalid_scenario_path() -> Path:
    return Path(__file__).parent / "fixtures" / "bench" / "invalid_missing_prompt.yaml"


_BENCH_FIXTURE_SOURCE = Path(__file__).resolve().parent / "fixtures" / "bench"
_BENCH_FIXTURE_STAGED = Path("/tmp/bench_fixtures")


@pytest.fixture(scope="session", autouse=True)
def bench_fixtures_staged():
    """Stage non-YAML fixture data from tests/fixtures/bench/ to /tmp/bench_fixtures/.

    Used by bench/scenarios/02_single_read.yaml, 05_file_exploration.yaml (and
    future fixtures) which reference an absolute /tmp/bench_fixtures/<file>
    path so the agent loop's tools can find them at scenario-run time without
    dynamic-path injection.

    Recursively copies subdirectories (e.g. exploration_root/) so multi-file
    fixture trees stage cleanly. memory_seed.db and other binary data files
    are copied as-is by the non-yaml suffix filter.
    """
    _BENCH_FIXTURE_STAGED.mkdir(parents=True, exist_ok=True)
    if _BENCH_FIXTURE_SOURCE.exists():
        for src in _BENCH_FIXTURE_SOURCE.iterdir():
            if src.suffix == ".yaml":
                continue
            dst = _BENCH_FIXTURE_STAGED / src.name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy2(src, dst)
    yield _BENCH_FIXTURE_STAGED
    # No teardown — /tmp is volatile; leaving the file present lets manual
    # bench runs after the test suite still find it.


@pytest.fixture
def fake_completed_runs():
    """Factory: produce a list[ScenarioCompleted]-shaped dicts for aggregator tests.

    Returns dicts (not events) so this fixture works before ScenarioCompleted is added.
    Tests that need the real event can model_validate after Wave 1 lands.
    """
    def _factory(
        n: int,
        success: list[bool] | None = None,
        latency_total: list[float] | None = None,
        latency_ttft: list[float] | None = None,
        tokens_in: list[int] | None = None,
        tokens_out: list[int] | None = None,
        iterations: list[int] | None = None,
        parse_failures: list[int] | None = None,
        stuck_recoveries: list[int] | None = None,
        tool_call_count: list[int] | None = None,
        scenario_name: str = "test_scenario",
        model: str = "test-model",
    ) -> list[dict]:
        out = []
        for i in range(n):
            out.append({
                "scenario_name": scenario_name,
                "model": model,
                "success": (success[i] if success else True),
                "latency_ttft": (latency_ttft[i] if latency_ttft else 0.1 + 0.01 * i),
                "latency_total": (latency_total[i] if latency_total else 1.0 + 0.05 * i),
                "tokens_in": (tokens_in[i] if tokens_in else 100 + i),
                "tokens_out": (tokens_out[i] if tokens_out else 50 + i),
                "iterations": (iterations[i] if iterations else 3),
                "parse_failures": (parse_failures[i] if parse_failures else 0),
                "stuck_recoveries": (stuck_recoveries[i] if stuck_recoveries else 0),
                "tool_call_count": (tool_call_count[i] if tool_call_count else 2),
                "internal_latencies": {},
                "tokens_estimated": False,
            })
        return out
    return _factory


# -----------------------------------------------------------------------------
# Phase 15 mutation-archive fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
async def archive_store(tmp_path: Path):
    """Fresh ArchiveStore with a temporary .localharness/archive.db.

    Mirrors the memory_store fixture (conftest.py:133-144). The store module
    lands in 15-02; until then this fixture xfails any test that requests it.
    """
    try:
        from localharness.autoresearch.archive import ArchiveStore
    except ImportError:
        pytest.skip("ArchiveStore not yet implemented (15-02)")
    store = ArchiveStore(tmp_path / ".localharness" / "archive.db")
    await store.open()
    yield store
    await store.close()


@pytest.fixture
def seeded_archive():
    """Factory: async helper that writes ArchiveEntry rows into a store.

    Usage: `entries = await seeded_archive(store, [dict(...), dict(...)])`.
    Each dict overrides ArchiveEntry defaults (id auto-uuid if omitted).
    Mirrors fake_completed_runs (conftest.py:210). Import is deferred so this
    file collects before 15-02 ships the module.
    """
    import json
    import time
    import uuid

    async def _seed(store, specs: list[dict]):
        from localharness.autoresearch.archive import ArchiveEntry
        written = []
        for spec in specs:
            tspf = spec.get("train_scores_per_fixture")
            entry = ArchiveEntry(
                id=spec.get("id", str(uuid.uuid4())),
                parent_id=spec.get("parent_id"),
                component=spec.get("component", "agents.main.system_prompt"),
                diff=spec.get("diff", json.dumps({"before": "a", "after": "b"})),
                train_score=spec.get("train_score"),
                train_scores_per_fixture=tspf,
                holdout_score=spec.get("holdout_score"),
                p_value=spec.get("p_value"),
                cost=spec.get("cost"),
                ts=spec.get("ts", int(time.time())),
                approved_by=spec.get("approved_by"),
                status=spec.get("status", "in_flight"),
            )
            written.append(await store.write(entry))
        return written

    return _seed


# -----------------------------------------------------------------------------
# Phase 16 proposer fixtures
# -----------------------------------------------------------------------------


class FakeLLMClient:
    """Spy LLM client for proposer tests.

    Exposes complete() — the method the proposer calls (NOT MockLLMClient's
    stream_complete) — and a complete_calls counter so seal tests can assert the
    model was never reached. Returns (message, usage) like the real LLMClient.
    """

    def __init__(self, content: str):
        self._content = content
        self.complete_calls = 0

        class _Cfg:
            tool_call_mode = "native"
            context_window = 128_000

        self.config = _Cfg()

    async def complete(self, messages, tools=None, stream=False):
        self.complete_calls += 1

        class _Msg:
            pass

        msg = _Msg()
        msg.content = self._content
        return msg, FakeCompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)


def _scenario_yaml(name: str, slice_: str) -> str:
    """A complete, VALID ScenarioSpec YAML (mirrors minimal_golden_scenario.yaml)."""
    return (
        f"name: {name}\n"
        'prompt: "What is 2 + 2? Give just the number."\n'
        'expected_outcome: "Returns 4."\n'
        "success_criteria:\n"
        '  golden_output: "4"\n'
        "budget:\n"
        "  max_actions: 5\n"
        "  max_duration_minutes: 1.0\n"
        "  max_context_tokens: 32000\n"
        "limits:\n"
        "  max_latency_s: 30.0\n"
        "  max_tool_calls: 0\n"
        "tools_allowed: []\n"
        f"slice: {slice_}\n"
        "category: tool_basics\n"
    )


@pytest.fixture
def proposer_corpus(tmp_path: Path) -> Path:
    """tmp corpus holding ONE train fixture + ONE holdout fixture (both valid ScenarioSpecs).

    Scenario names mirror the proposer_results run_ids: prop_train_fx / prop_holdout_fx.
    The slice line is the single source of truth the PROP-03 seal resolves through.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "prop_train_fx.yaml").write_text(
        _scenario_yaml("prop_train_fx", "train"), encoding="utf-8"
    )
    (corpus / "prop_holdout_fx.yaml").write_text(
        _scenario_yaml("prop_holdout_fx", "holdout"), encoding="utf-8"
    )
    return corpus


@pytest.fixture
def proposer_results(tmp_path: Path) -> dict:
    """tmp results tree with canned JSONL traces for the seal/evidence tests.

    Writes one FAILED ScenarioCompleted trace per scenario (a failed train run is
    real mutation signal). run_id syntax is the resolved Phase 16 contract
    `{model}/{scenario}/{timestamp}`.
    """
    results = tmp_path / "results"
    model = "fakemodel"
    timestamp = "20260529T000000Z"

    def _write(scenario_name: str, success: bool) -> None:
        path = resolve_run_path(results, model, scenario_name, timestamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = ScenarioCompleted(
            scenario_name=scenario_name,
            model=model,
            success=success,
            latency_ttft=1.0,
            latency_total=1.0,
            tokens_in=5,
            tokens_out=5,
            iterations=1,
            parse_failures=0,
            stuck_recoveries=0,
            tool_call_count=0,
        ).model_dump_json()
        path.write_text(line + "\n", encoding="utf-8")

    _write("prop_train_fx", success=False)
    _write("prop_holdout_fx", success=False)

    return {
        "results": results,
        "train_run_id": f"{model}/prop_train_fx/{timestamp}",
        "holdout_run_id": f"{model}/prop_holdout_fx/{timestamp}",
    }
