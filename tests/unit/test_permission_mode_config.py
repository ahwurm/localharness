"""v0.14 permission spine — the config face of the gate (PRD §3.3, §3.4).

Three properties, one file:
  (a) `permissions.mode` is a SESSION MODE now, defaulting to the mode that asks; the retired
      `auto`/`manual` spellings still load, mapped and warned, so no existing config breaks.
  (b) `permissions.allow_patterns` is gone. A NON-EMPTY one fails loudly with the reason —
      config is a repo-travelling surface and must never be able to LOOSEN policy. An EMPTY one
      (what every pre-v0.14 `init` wrote) loads, is dropped, and warns: rejecting it stopped
      every existing install from starting (D2).
  (c) `permissions.ask` maps onto `GateSettings` without restating a single default.
"""
from __future__ import annotations

import warnings
from dataclasses import fields as dataclass_fields

import pytest
from pydantic import ValidationError

from localharness.agent.gate_types import DEFAULT_MODE, MODE_STRICTNESS, GateSettings
from localharness.config.models import AgentConfig, AskConfig, PermissionConfig


# --- (a) mode ---------------------------------------------------------------

def test_default_mode_is_the_mode_that_asks():
    """A fresh config gets `guarded`, not the never-ask behaviour v0.13 shipped."""
    assert PermissionConfig().mode == DEFAULT_MODE == "guarded"
    assert AgentConfig(name="x", role="y").permissions.mode == "guarded"


@pytest.mark.parametrize("legacy", ["auto", "manual"])
def test_legacy_mode_spellings_map_to_guarded_with_a_deprecation_warning(legacy):
    """Every config written before v0.14 says `auto` (or the unimplemented `manual` stub).

    Rejecting them would break those configs; mapping them to `unattended` would ship the gate
    switched off for everyone who already has one. They map to `guarded` and say so.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = PermissionConfig(mode=legacy)
    assert cfg.mode == "guarded"
    assert len(caught) == 1 and issubclass(caught[0].category, DeprecationWarning)
    message = str(caught[0].message)
    assert legacy in message and "unattended" in message, (
        "the warning must name both the retired spelling and the escape hatch for a "
        "human-less run"
    )


@pytest.mark.parametrize("mode", sorted(MODE_STRICTNESS))
def test_every_declared_mode_validates(mode):
    """The Literal and MODE_STRICTNESS are one vocabulary — a mode in the strictness table that
    config rejects would make the loader's narrow-only union undecidable."""
    assert PermissionConfig(mode=mode).mode == mode


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValidationError):
        PermissionConfig(mode="yolo")


# --- (b) allow_patterns -----------------------------------------------------

def test_allow_patterns_is_rejected_with_the_reason():
    """PRD §3.3: grants live in the global store, never in config. `extra="forbid"` would
    reject the key anyway; the point of the check is that the user is TOLD where grants went."""
    with pytest.raises(ValidationError) as exc:
        PermissionConfig(allow_patterns=["bash_exec(*)"])
    message = str(exc.value)
    assert "allow_patterns" in message and "grants.yaml" in message


def test_allow_patterns_is_rejected_inside_a_full_agent_config():
    """The real shape a user's yaml takes — the check must survive nesting, not just a direct
    PermissionConfig() call."""
    with pytest.raises(ValidationError):
        AgentConfig(name="x", role="y", permissions={"allow_patterns": ["bash_exec(*)"]})


@pytest.mark.parametrize("empty", [[], None], ids=["empty-list", "null"])
def test_an_empty_allow_patterns_loads_and_is_dropped_with_a_warning(empty):
    """Every pre-v0.14 `localharness init` wrote `allow_patterns: []` (D2).

    Rejecting the bare KEY meant no existing install could start, and `config migrate` — the
    documented repair — hit the same validator. An empty value carried no policy, so it loads,
    is dropped, and says so.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = PermissionConfig(allow_patterns=empty)
    assert not hasattr(cfg, "allow_patterns")
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "allow_patterns" in str(deprecations[0].message)


def test_an_empty_allow_patterns_loads_inside_a_full_agent_config():
    """Nesting again: the old init output is a whole config file, not a bare PermissionConfig."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = AgentConfig(name="x", role="y", permissions={"mode": "auto", "allow_patterns": []})
    assert cfg.permissions.mode == "guarded"


# --- (c) ask -> GateSettings ------------------------------------------------

def test_default_ask_config_reproduces_the_shipped_gate_settings():
    """An untouched config must hand the gate EXACTLY the defaults gate_types declares — if this
    drifts, config silently became a second, weaker copy of the policy."""
    assert AskConfig().to_gate_settings() == GateSettings()


def test_overrides_arrive_in_the_container_type_the_dataclass_declares():
    """YAML gives lists; GateSettings declares frozensets and tuples. The conversion is read off
    the dataclass field's own default, so it cannot drift from the type the gate expects."""
    settings = AskConfig(
        wrapper_commands=["env", "nohup"],
        protected_paths_home=["~/.ssh"],
        mcp_trusted_servers=["exa"],
    ).to_gate_settings()

    assert settings.wrapper_commands == frozenset({"env", "nohup"})
    assert settings.protected_paths_home == ("~/.ssh",)
    assert settings.mcp_trusted_servers == frozenset({"exa"})


def test_an_unset_rule_set_keeps_its_shipped_default():
    """None means "use the default", not "empty" — an override of one rule set must not blank
    the other twelve."""
    settings = AskConfig(wrapper_commands=["env"]).to_gate_settings()
    assert settings.read_only_signatures == GateSettings().read_only_signatures
    assert settings.destructive_signatures == GateSettings().destructive_signatures


def test_the_two_renamed_knobs_reach_their_gate_fields():
    """`permissions.ask.network_hosts` / `.timeout_s` feed `ask_network_hosts` / `ask_timeout_s`
    — the block already says "ask", the flat GateSettings namespace needs the prefix."""
    settings = AskConfig(network_hosts=True, timeout_s=45.0).to_gate_settings()
    assert settings.ask_network_hosts is True
    assert settings.ask_timeout_s == 45.0
    assert AskConfig().to_gate_settings().ask_timeout_s is None


def test_every_list_shaped_gate_rule_set_is_overridable_from_config():
    """Drift guard: a rule set added to GateSettings must gain its AskConfig field, or it
    becomes policy no user can tune while every neighbouring set is tunable."""
    list_shaped = {
        f.name for f in dataclass_fields(GateSettings)
        if isinstance(f.default, (frozenset, tuple))
    }
    from localharness.config.models import ASK_TO_GATE_FIELD

    reachable = set(AskConfig.model_fields) | {
        ASK_TO_GATE_FIELD[name] for name in AskConfig.model_fields if name in ASK_TO_GATE_FIELD
    }
    assert list_shaped <= reachable, f"no permissions.ask field for: {sorted(list_shaped - reachable)}"
