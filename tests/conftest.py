import shutil
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from localharness.config.models import AgentConfig, OrgConfig
from localharness.core.bus import EventBus
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

    Used by bench/scenarios/02_single_read.yaml (and future fixtures) which
    reference an absolute /tmp/bench_fixtures/<file> path so the agent loop's
    `read` tool can find them at scenario-run time without dynamic-path injection.
    """
    _BENCH_FIXTURE_STAGED.mkdir(parents=True, exist_ok=True)
    if _BENCH_FIXTURE_SOURCE.exists():
        for src in _BENCH_FIXTURE_SOURCE.iterdir():
            if src.is_file() and src.suffix != ".yaml":
                shutil.copy2(src, _BENCH_FIXTURE_STAGED / src.name)
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
