"""Image parts through the shared spine: budget arithmetic, compaction, the XML downgrade, the
wire renderer. The loop-level test (run_turn(images=...) → provider) lives beside it."""
from __future__ import annotations

from localharness.agent.context import (
    TokenCounter,
    _shrink_content_to_budget,
    render_summarizer_input,
)
from localharness.core.content import DROPPED_IMAGE_PREFIX, image_part, text_part
from localharness.core.types import human_message, provider_messages
from localharness.provider.client import LLMClient
from tests.unit.test_image_content import png_bytes


def _turn(w: int = 640, h: int = 320, text: str = "what is wrong with this screen?"):
    return human_message([text_part(text), image_part(png_bytes(w, h), name="shot.png")])


def test_estimate_messages_charges_the_picture_by_patches():
    tc = TokenCounter()  # "off": offline estimator — the bench/test mode
    text_only = tc.estimate_messages([human_message("what is wrong with this screen?")])
    with_image = tc.estimate_messages([_turn(640, 320)])
    assert with_image == text_only + 20 * 10 + 2
    # count_messages (non-exact mode) falls back through the same arithmetic.
    assert tc.count_messages([_turn(640, 320)]) == with_image


def test_provider_messages_strip_the_private_part_record():
    wire = provider_messages([_turn()])[0]
    assert "_lh" not in wire
    assert [set(p) for p in wire["content"]] == [{"type", "text"}, {"type", "image_url"}]
    assert wire["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_summarizer_sees_the_picture_named_not_the_bytes():
    rendered = render_summarizer_input([_turn(), {"role": "assistant", "content": "The button overlaps."}])
    assert "[image: 640×320 png, shot.png]" in rendered
    assert "base64" not in rendered


def test_emergency_floor_drops_the_picture_with_an_explicit_line():
    tc = TokenCounter()
    msgs = [{"role": "system", "content": "sys"}, _turn(1920, 1080)]
    shrunk, changed = _shrink_content_to_budget(msgs, max_msg_tokens=200, token_counter=tc)
    assert changed is True
    body = shrunk[1]["content"]
    assert isinstance(body, str)
    assert body.startswith("what is wrong with this screen?\n" + DROPPED_IMAGE_PREFIX)
    assert tc.estimate_messages(shrunk) <= 200


def test_xml_downgrade_merges_a_parts_turn_without_losing_the_picture():
    msgs = [_turn(), {"role": "user", "content": "<tool_response>\nok\n</tool_response>"}]
    out = LLMClient._downgrade_history_for_xml(None, provider_messages(msgs))
    assert len(out) == 1
    kinds = [p["type"] for p in out[0]["content"]]
    assert kinds == ["text", "image_url", "text", "text"]
    assert out[0]["content"][-1]["text"].startswith("<tool_response>")
