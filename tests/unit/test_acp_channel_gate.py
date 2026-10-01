"""Issue #165: `localharness acp` could not bring up a session — the channel gate inside
_start_async refused channel_mode="acp", which channels/acp.py passes, and ACP's BaseException
panel turned it into "bring-up failed". This drives the REAL _start_async past the gate."""
from __future__ import annotations

import pytest

from tests.unit.test_start_cmd import _stub_start_boundaries, _write_agent

pytestmark = pytest.mark.asyncio


class _StubAcpChannel:
    """Everything _start_async needs from an ACP channel: an awaitable serve()."""

    def __init__(self) -> None:
        self.served = False

    async def serve(self, **kw) -> None:
        self.served = True

    def __getattr__(self, name):
        raise AttributeError(name)


async def test_acp_session_gets_past_the_channel_gate(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch)
    _write_agent(tmp_path / "agents", "solo")
    ch = _StubAcpChannel()
    await _start_async(None, False, False, str(tmp_path), channel_mode="acp", no_input=True,
                       acp_channel=ch)
    assert ch.served
