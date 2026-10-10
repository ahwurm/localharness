"""Shared doubles for the effort tests (Wave 0, 03-CONTEXT D-26).

An importable helper module, like tests/unit/launch_support.py: its name does not start with `test_`,
so pytest collects nothing from it. A test module imports what it needs by name:

    from tests.unit.effort_support import RecordingLLM, harness_with

- `RecordingLLM`: a `stream_complete` double that records every keyword dict it is called with and
  carries the client's effort surface (`rejected_effort_keys`, `accepted_effort_keys`,
  `effort_budget_key`).
- `harness_with(...)`: a validating `HarnessConfig` built from the resolver's inputs.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from localharness.config.models import (
    ActiveSelection,
    EndpointRef,
    HarnessConfig,
    ManagedServerConfig,
    OrgConfig,
    ProviderConfig,
)

PRIMARY_URL = "http://127.0.0.1:8080/v1"

_BUDGET_KEYS = {"llamacpp": "reasoning_budget_tokens"}


class RecordingLLM:
    """Records the keywords of every `stream_complete` call; answers from `responses` in order."""

    def __init__(
        self, *, provider_type: str = "llamacpp", base_url: str = PRIMARY_URL, model: str = "m",
        rejected: tuple[str, ...] | list[str] | frozenset[str] = (),
        accepted: tuple[str, ...] | list[str] | frozenset[str] = (),
    ) -> None:
        self.config = SimpleNamespace(
            provider_type=provider_type, base_url=base_url, model=model,
            max_tokens=None, tool_call_mode="native",
        )
        self.calls: list[dict[str, Any]] = []
        self.rejected_effort_keys: frozenset[str] = frozenset(rejected)
        self.accepted_effort_keys: frozenset[str] = frozenset(accepted)
        self.responses: list[tuple[Any, Any]] = [
            (SimpleNamespace(content="", tool_calls=None, finish_reason="stop"), None)
        ]

    @property
    def effort_budget_key(self) -> str | None:
        return _BUDGET_KEYS.get(self.config.provider_type)

    async def stream_complete(self, messages, tools=None, on_token=None, **kw):
        self.calls.append(dict(kw))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def _record(effort: str | None, *, model: str = "/tmp/m.gguf", port: int = 8081) -> ManagedServerConfig:
    return ManagedServerConfig(
        runtime="llamacpp", binary="/usr/bin/llama-server", model=model, port=port, effort=effort,
    )


def harness_with(
    *, server_effort: str | None = None, peer: str | None = None, peer_effort: str | None = None,
    active: str | None = None, active_effort: str | None = None, org_default: str | None = None,
    hand_run: bool = False,
) -> HarnessConfig:
    """A HarnessConfig whose effort-bearing records are the arguments.

    The primary serves PRIMARY_URL. `hand_run` leaves `server` unset (a server the user runs).
    `peer` adds one extra endpoint named "peer" whose lifecycle record carries `peer_effort`;
    `active` selects a peer by URL with `active_effort` persisted on the selection."""
    return HarnessConfig(
        provider=ProviderConfig(
            provider_type="llamacpp", base_url=PRIMARY_URL, default_model="m",
        ),
        server=None if hand_run else _record(server_effort),
        extra_endpoints=[
            EndpointRef(
                name="peer", base_url=peer, provider_type="llamacpp", gpu=True,
                lifecycle=_record(peer_effort, port=8082),
            )
        ] if peer else [],
        active_endpoint=ActiveSelection(
            name="peer", base_url=active, provider_type="llamacpp", model="m", effort=active_effort,
        ) if active else None,
        org=OrgConfig(default_effort=org_default),
    )
