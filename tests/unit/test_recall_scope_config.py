"""MEMS-02's config knob: `agent.memory.recall_scope`.

Phase 42 plan 01, task 1. This file pins the CONTRACT only — nothing in `src/` reads
the field yet (42-02 builds the recall router, 42-03 wires it). The point of pinning it
first is that the plans which follow can be graded on routing alone.

The two tests that carry weight:
  * `test_out_of_set_value_is_rejected_at_config_time` uses a PLAUSIBLE wrong value
    ("nearest" — the word this project's own workspace walk uses), not a nonsense
    string, so it fails for the right reason.
  * `test_agent_yaml_recall_scope_survives_load_agent` goes through a real
    `ConfigLoader.load_agent()` off a real tmp tree, because a field that validates in
    isolation but is stripped by the agent/division/org merge would still be useless.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from localharness.config.loader import ConfigLoader
from localharness.config.models import AgentConfig, MemoryConfig
from localharness.registry.catalogue import build_catalogue
from localharness.registry.paths import walk_model_fields


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """The tmp-tree idiom from tests/unit/test_config_loader.py."""
    (tmp_path / "agents").mkdir()
    (tmp_path / "divisions").mkdir()
    return tmp_path


# ------------------------------------------------------------------ #
# 1. Unset means workspace — a session never reaches for the global
#    store unless the user asked for it.
# ------------------------------------------------------------------ #
def test_recall_scope_defaults_to_workspace() -> None:
    assert MemoryConfig().recall_scope == "workspace"
    # ...and through the agent object the loop actually reads (`_mem_cfg`).
    assert AgentConfig(name="a", role="r").memory.recall_scope == "workspace"


# ------------------------------------------------------------------ #
# 2. All three ruled literals validate.
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("value", ["workspace", "global", "both"])
def test_all_three_literals_validate(value: str) -> None:
    assert MemoryConfig(recall_scope=value).recall_scope == value


# ------------------------------------------------------------------ #
# 3. The knob is a CLOSED set, checked at config-validation time — not
#    at recall time, where a bad value would surface as a confusing
#    mid-session failure (or worse, a silent fallback to `both`).
# ------------------------------------------------------------------ #
def test_out_of_set_value_is_rejected_at_config_time() -> None:
    with pytest.raises(ValidationError) as exc_info:
        # "nearest" is the plausible wrong answer: it is the vocabulary this
        # project uses for the workspace WALK (nearest .localharness wins), so a
        # user who read the layering docs might reasonably try it here.
        MemoryConfig(recall_scope="nearest")
    errors = exc_info.value.errors()
    assert len(errors) == 1, errors
    assert errors[0]["loc"] == ("recall_scope",)
    # It must be rejected because the value is out of the CLOSED SET. Asserting
    # only on the message would let this test pass while the field does not exist
    # at all: MemoryConfig sets extra="forbid", so an unknown key raises an
    # `extra_forbidden` error that ALSO names "recall_scope".
    assert errors[0]["type"] == "literal_error", errors


# ------------------------------------------------------------------ #
# 4. It survives a REAL load_agent() — the merge does not strip it.
# ------------------------------------------------------------------ #
def test_agent_yaml_recall_scope_survives_load_agent(config_dir: Path) -> None:
    _write_yaml(
        config_dir / "agents" / "scoped.yaml",
        {"name": "scoped", "role": "Scoped agent", "memory": {"recall_scope": "both"}},
    )
    cfg = ConfigLoader(config_dir=config_dir).load_agent("scoped")
    assert cfg.memory.recall_scope == "both"


def test_agent_yaml_without_memory_block_recalls_workspace(config_dir: Path) -> None:
    _write_yaml(
        config_dir / "agents" / "plain.yaml",
        {"name": "plain", "role": "Plain agent"},
    )
    cfg = ConfigLoader(config_dir=config_dir).load_agent("plain")
    assert cfg.memory.recall_scope == "workspace"


# ------------------------------------------------------------------ #
# 5. Registry auto-enumeration: a bare Literal leaf is emitted by
#    walk_model_fields, so `components set` support is free. Asserting it
#    is cheaper than discovering later that the leaf was skipped.
# ------------------------------------------------------------------ #
def test_recall_scope_enumerates_as_a_components_axis() -> None:
    paths = {path for path, _ann in walk_model_fields(AgentConfig, prefix="agent")}
    assert "agent.memory.recall_scope" in paths

    # ...and it reaches the catalogue `components set` actually reads, with its
    # default carried (cfg=None still enumerates the agent surfaces — REG-04).
    entries = build_catalogue(None, overlays={})
    entry = entries["agent.memory.recall_scope"]
    assert entry.default_value == "workspace"


# ------------------------------------------------------------------ #
# 6. shared_read is a DIFFERENT axis and is untouched by this plan:
#    org-hierarchy (division/org context files) vs physical store.
# ------------------------------------------------------------------ #
def test_shared_read_is_a_different_axis_and_unchanged() -> None:
    cfg = MemoryConfig()
    assert cfg.shared_read == []
    assert MemoryConfig(shared_read=["division"]).shared_read == ["division"]
    # The two knobs are independent: setting one does not move the other.
    both = MemoryConfig(recall_scope="global", shared_read=["org"])
    assert both.recall_scope == "global"
    assert both.shared_read == ["org"]
