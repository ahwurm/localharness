"""The one place a ToolResult becomes an Observation carries a typed artifact (protocol v4).

A plugin tool returns `self.ok(..., artifact=ref.model_dump())`; the loop validates it into
Observation.artifact. Anything malformed is dropped (artifact None), never an error, and the
turn still completes."""
import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig, PermissionConfig
from localharness.core.artifacts import write_artifact
from localharness.core.events import ARTIFACT_ID_RE, ArtifactRef, Observation, TurnCompleted
from localharness.tools.base import Tool

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
VALID_ID = "art-20260930-120000-abcdef"


class _Registry:
    def __init__(self, result):
        self._result = result

    def get_tools_for_agent(self, agent_id, division_id, tool_config):
        return {}

    async def dispatch(self, name, arguments, agent_id, division_id, tool_config):
        return self._result


async def _observe(mock_llm_client, bus, result) -> Observation:
    R, TC = mock_llm_client.Response, mock_llm_client.ToolCall
    llm = mock_llm_client([R(content=None, tool_calls=[TC(id="tc-1", name="generate_image",
                                                          arguments={})]),
                           R(content="Done.")])
    loop = AgentLoop(
        config=AgentConfig(name="test-agent", role="Test agent.",
                           permissions=PermissionConfig(mode="unattended")),
        llm=llm, bus=bus, context_manager=ContextManager(), tool_registry=_Registry(result),
        permission_evaluator=PermissionEvaluator(),
    )
    await loop.run_turn("draw")
    assert len(bus.history(event_types=[TurnCompleted])) == 1
    [obs] = [e for e in bus.history(event_types=[Observation]) if e.tool_call_id == "tc-1"]
    return obs


# Tool.ok / Tool.err never touch self: called unbound, they are exactly the plugin contract.
def _ok(**metadata):
    return Tool.ok(None, "Image saved", **metadata)


@pytest.mark.asyncio
async def test_a_dumped_artifact_reaches_the_observation(mock_llm_client, bus, tmp_path):
    ref = write_artifact(tmp_path, "image", PNG, "image/png")
    obs = await _observe(mock_llm_client, bus, _ok(artifact=ref.model_dump()))
    assert obs.artifact == ArtifactRef(plugin="image", kind="image", id=ref.id, mime="image/png")
    assert ARTIFACT_ID_RE.fullmatch(obs.artifact.id)
    assert obs.error is None


@pytest.mark.asyncio
async def test_an_artifact_instance_reaches_the_observation(mock_llm_client, bus, tmp_path):
    ref = write_artifact(tmp_path, "image", PNG, "image/png")
    obs = await _observe(mock_llm_client, bus, _ok(artifact=ref))
    assert obs.artifact == ref


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"plugin": "../x", "kind": "image", "id": VALID_ID, "mime": "image/png"},
    {"plugin": "image", "kind": "image", "id": "img-20260101-000000-abcdef", "mime": "image/png"},
    {"plugin": "image", "kind": "image", "id": VALID_ID, "mime": "text/html"},
    "not-a-dict",
    42,
], ids=["traversal-plugin", "foreign-id", "html-mime", "string", "int"])
async def test_an_invalid_artifact_is_dropped_and_the_turn_continues(mock_llm_client, bus, bad):
    obs = await _observe(mock_llm_client, bus, _ok(artifact=bad))
    assert obs.artifact is None
    assert obs.error is None
    assert obs.output == "Image saved"


@pytest.mark.asyncio
async def test_a_failed_result_carries_no_artifact(mock_llm_client, bus, tmp_path):
    ref = write_artifact(tmp_path, "image", PNG, "image/png")
    obs = await _observe(mock_llm_client, bus,
                         Tool.err(None, "comfy down", artifact=ref.model_dump()))
    assert obs.artifact is None
    assert obs.error is not None


def test_an_old_logged_observation_with_the_v2_field_still_loads():
    old = {"agent_id": "a", "session_id": "s", "observation_type": "tool_result",
           "output": "Image saved", "image" + "_id": VALID_ID}
    obs = Observation.model_validate(old)
    assert obs.artifact is None
