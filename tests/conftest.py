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
    """Fake LLM client that returns scripted responses in sequence."""

    def __init__(self, responses: list[FakeLLMResponse], return_tuple: bool = False) -> None:
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
    def _factory(responses: list[FakeLLMResponse], return_tuple: bool = False) -> MockLLMClient:
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
