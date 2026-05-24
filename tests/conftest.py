import pytest
from pathlib import Path
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
