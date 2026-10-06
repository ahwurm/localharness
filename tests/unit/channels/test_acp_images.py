"""Zed pastes a screenshot → an ACP image block → an image part the model sees."""
from __future__ import annotations

import base64

from tests.unit.channels.test_acp import FakeClient, FakeLLMResponse, _connect, _start, text_block
from tests.unit.channels.test_acp import keep_cwd  # noqa: F401, F811 — a fixture, used by name below
from tests.unit.test_image_content import png_bytes


async def test_initialize_advertises_image_prompts(tmp_path, keep_cwd):  # noqa: F811
    """Zed sends image blocks only to an agent whose promptCapabilities say image: true."""
    from acp.schema import ClientCapabilities

    from localharness.channels.acp import AcpChannel

    agent = AcpChannel(config_dir=str(tmp_path / "config"))
    conn, _tasks, _ = await _connect(agent, FakeClient())
    response = await conn.initialize(protocol_version=1, client_capabilities=ClientCapabilities())
    assert response.agent_capabilities.prompt_capabilities.image is True


async def test_a_pasted_image_reaches_the_model_as_a_part(tmp_path, monkeypatch, keep_cwd):  # noqa: F811
    from acp.schema import ImageContentBlock

    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="A chip.")])
    data = png_bytes(265, 51)
    await session.conn.prompt(
        session_id=session.session_id,
        prompt=[
            text_block("what UI element is this?"),
            ImageContentBlock(type="image", data=base64.b64encode(data).decode(), mime_type="image/png"),
        ],
    )
    user = [m for m in session.llm.seen_messages[0] if m.get("role") == "user"][0]
    body = user["content"]
    assert body[0] == {"type": "text", "text": "what UI element is this?"}
    assert body[1]["type"] == "image_url"
    assert body[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(data).decode()
    assert body[1]["_lh"]["name"] == "pasted image" and body[1]["_lh"]["width"] == 265
