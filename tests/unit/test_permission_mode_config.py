"""v0.14 permission spine — the config face of the gate (PRD §3.3, §3.4).

Three properties, one file:
  (a) `permissions.mode` is a SESSION MODE now, defaulting to `auto` — the blacklist-only mode
      the owner ruled for on 2026-09-11; the retired `manual` spelling still loads, mapped and
      warned, so no existing config breaks.
  (b) `permissions.allow_patterns` is gone. A NON-EMPTY one fails loudly with the reason —
      config is a repo-travelling surface and must never be able to LOOSEN policy. An EMPTY one
      (what every pre-v0.14 `init` wrote) loads, is dropped, and warns: rejecting it stopped
      every existing install from starting (D2).
  (c) `permissions.ask` maps onto `GateSettings` without restating a single default.
"""
from __future__ import annotations

import logging
from dataclasses import fields as dataclass_fields

import pytest
from pydantic import ValidationError

from localharness.agent.gate_types import DEFAULT_MODE, MODE_STRICTNESS, GateSettings
from localharness.config.models import (
    LEGACY_MANUAL_MODE,
    AgentConfig,
    AskConfig,
    PermissionConfig,
)

# Where config's deprecation notices land now that they are log records rather than
# `warnings.warn` calls — named once so a module rename cannot leave the assertions passing
# vacuously against a logger nothing writes to.
MODELS_LOGGER = "localharness.config.models"


# --- (a) mode ---------------------------------------------------------------

def test_default_mode_is_auto_and_guarded_is_still_the_stricter_one():
    """A fresh config gets `auto` — the blacklist-only mode, not the one that asks.

    Owner ruling 2026-09-11, after a day on the shipped `guarded` default: "way too intrusive,
    it stopped me multiple times… the default should be an auto mode that almost never triggers
    unless genuinely risky / dangerous". `guarded` did not go anywhere — it is one `/mode
    guarded` away, and it is still the STRICTER of the two, which is what makes the loader's
    narrow-only union let a project layer move a session from the new default into it.
    """
    assert PermissionConfig().mode == DEFAULT_MODE == "auto"
    assert AgentConfig(name="x", role="y").permissions.mode == "auto"

    assert PermissionConfig(mode="guarded").mode == "guarded"
    assert MODE_STRICTNESS["guarded"] > MODE_STRICTNESS["auto"], (
        "auto is the looser default; a project layer must still be able to tighten into guarded"
    )


def test_auto_is_a_real_mode_now_and_loads_without_a_deprecation_notice(caplog):
    """`permissions.mode: auto` used to be the retired v0.13 spelling, mapped onto `guarded`
    with a warning. Since the owner ruling of 2026-09-11 it names the shipped default mode, so
    it has to load as itself and say nothing — a deprecation notice on the default would greet
    every new install with a warning about a spelling that is now the recommended one.
    """
    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        cfg = PermissionConfig(mode="auto")
    assert cfg.mode == "auto"
    assert not [r for r in caplog.records if "deprecated" in r.getMessage()]
    assert not [r for r in caplog.records if "permissions.mode" in r.getMessage()]


@pytest.mark.parametrize("legacy", ["manual"])
def test_legacy_mode_spellings_map_to_guarded_with_a_deprecation_warning(legacy, caplog):
    """`manual` is the last retired spelling — an unimplemented v0.13 stub.

    Rejecting it would break those configs; mapping it to `unattended` would ship the gate
    switched off for everyone who already has one. It maps to `guarded` and says so. (`auto`
    was the other member of this set until 2026-09-11, when it became a real mode; the test
    above covers it.)

    The notice is a LOG record, not `warnings.warn`: raised inside a validator (which is what
    `-W error::DeprecationWarning` does to it) it killed `localharness config migrate`, the
    repair path for these very configs.
    """
    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        cfg = PermissionConfig(mode=legacy)
    assert cfg.mode == LEGACY_MANUAL_MODE == "guarded"
    records = [r for r in caplog.records if "permissions.mode" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
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
def test_an_empty_allow_patterns_loads_and_is_dropped_with_a_warning(empty, caplog):
    """Every pre-v0.14 `localharness init` wrote `allow_patterns: []` (D2).

    Rejecting the bare KEY meant no existing install could start, and `config migrate` — the
    documented repair — hit the same validator. An empty value carried no policy, so it loads,
    is dropped, and says so.
    """
    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        cfg = PermissionConfig(allow_patterns=empty)
    assert not hasattr(cfg, "allow_patterns")
    notices = [r for r in caplog.records if "allow_patterns" in r.getMessage()]
    assert len(notices) == 1
    assert "grants.yaml" in notices[0].getMessage()


def test_an_empty_allow_patterns_loads_inside_a_full_agent_config():
    """Nesting again: the old init output is a whole config file, not a bare PermissionConfig.

    The mode here is the other half of that old init output. It said `manual` — the retired
    spelling that still maps onto `guarded` — because `auto`, the spelling those files more
    often carried, stopped being legacy on 2026-09-11 and would no longer prove the mapping ran
    through the nested model.
    """
    cfg = AgentConfig(name="x", role="y", permissions={"mode": "manual", "allow_patterns": []})
    assert cfg.permissions.mode == LEGACY_MANUAL_MODE


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


# GateSettings rule sets that are deliberately NOT reachable from `permissions.ask`. Every one
# of them LOOSENS the gate when a line is added to it, and config travels with a repo (PRD
# §3.3), so a cloned project could otherwise widen the gate of the machine that opened it:
#   - protected_paths_system_exempt: carves directories back OUT of the protected system set,
#     so an entry here removes protection rather than adding it.
#   - target_scoped_destructive_verbs: a verb added here is one `auto` stops asking about
#     wherever it can resolve the target.
#   - auto_blacklist: the whole structure that decides what the DEFAULT mode still asks about;
#     extending its signature sets is the one edit that makes the shipped default unsafe.
# Each carries the same reason in its own docstring in `agent/gate_types.py`.
GATE_ONLY_RULE_SETS = frozenset({
    "protected_paths_system_exempt",
    "target_scoped_destructive_verbs",
    "auto_blacklist",
})


def test_every_list_shaped_gate_rule_set_is_overridable_from_config():
    """Drift guard: a rule set added to GateSettings must gain its AskConfig field, or it
    becomes policy no user can tune while every neighbouring set is tunable.

    The three exceptions are named in :data:`GATE_ONLY_RULE_SETS` with the reason — and their
    exclusion is asserted rather than merely subtracted, so the guard bites in BOTH directions:
    a new tunable without an AskConfig field trips the first assertion, and a config surface
    quietly opened onto one of the gate-only sets trips the second.
    """
    from localharness.config.models import ASK_TO_GATE_FIELD

    reachable = set(AskConfig.model_fields) | {
        ASK_TO_GATE_FIELD[name] for name in AskConfig.model_fields if name in ASK_TO_GATE_FIELD
    }
    gate_fields = {f.name for f in dataclass_fields(GateSettings)}
    list_shaped = {
        f.name for f in dataclass_fields(GateSettings)
        if isinstance(f.default, (frozenset, tuple))
    } - GATE_ONLY_RULE_SETS

    assert GATE_ONLY_RULE_SETS <= gate_fields, (
        f"the exclusion set names fields GateSettings no longer has: "
        f"{sorted(GATE_ONLY_RULE_SETS - gate_fields)}"
    )
    assert list_shaped <= reachable, f"no permissions.ask field for: {sorted(list_shaped - reachable)}"
    assert not (GATE_ONLY_RULE_SETS & reachable), (
        f"permissions.ask now reaches a rule set that loosens the gate when extended: "
        f"{sorted(GATE_ONLY_RULE_SETS & reachable)}"
    )
