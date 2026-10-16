"""Tests for the orchestrator router's static no-config welcome message."""
from __future__ import annotations

from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow


def test_no_config_message_points_at_live_site_root():
    """#51: the very first external pointer a new user sees must not be a dead link.
    localharness.dev/resources is a 404 — the message must point at the live site root."""
    msg = AgentCreationFlow.no_config_message()
    assert "/resources" not in msg  # the dead path is gone
    assert "https://localharness.dev" in msg  # points at a page that exists


def test_no_config_message_instance_method_matches_static():
    """The message is reachable both statically and via an instance (start_cmd uses the static)."""
    flow = AgentCreationFlow(AgentCardRegistry())
    assert "https://localharness.dev" in flow.no_config_message()


def test_no_config_message_teaches_start_first_and_needs_no_model_server():
    msg = AgentCreationFlow.no_config_message()
    assert "localharness start" in msg
    assert "localharness init" not in msg
    assert msg.index("localharness start") < msg.index("https://localharness.dev")
    assert "not required" in msg
