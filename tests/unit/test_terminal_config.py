"""Terminal-channel config: the type-anytime input box kill-switch (default ON) and the
tier-2 classifier off-switch (default ON), nested on HarnessConfig as `terminal`."""
from __future__ import annotations

import pytest

from localharness.config.models import HarnessConfig, ProviderConfig, TerminalConfig


def _harness(**terminal_kwargs):
    kw = {"provider": ProviderConfig(base_url="http://x/v1", default_model="m")}
    if terminal_kwargs:
        kw["terminal"] = TerminalConfig(**terminal_kwargs)
    return HarnessConfig(**kw)


def test_defaults_on():
    tc = TerminalConfig()
    assert tc.inputbox_enabled is True
    assert tc.input_router_tier2_enabled is True


def test_harness_has_terminal_by_default():
    hc = _harness()
    assert hc.terminal.inputbox_enabled is True
    assert hc.terminal.input_router_tier2_enabled is True


def test_kill_switches_off():
    hc = _harness(inputbox_enabled=False, input_router_tier2_enabled=False)
    assert hc.terminal.inputbox_enabled is False
    assert hc.terminal.input_router_tier2_enabled is False


def test_forbids_unknown_key():
    with pytest.raises(Exception):
        TerminalConfig(bogus=True)  # extra="forbid"
