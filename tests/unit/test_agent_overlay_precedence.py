"""`components set agent.temperature|max_tokens|model` must reach the agent that runs.

Pre-existing since <=0.12.5: `load_agent` wrote `_resolve_scalar`'s SHIPPED DEFAULT into the merged
dict for those three fields before layering the `agent:` overlay UNDER it, so the overlay could
never win — the value was stored, confirmed by `components get`, and ignored at load. The shipped
precedence is agent yaml > division > org > `agent:` overlay > schema default: the overlay is the
"nothing else set it" slot, which is exactly `_resolve_scalar`'s `default` argument.
"""
from __future__ import annotations

import yaml

from localharness.config.loader import ConfigLoader


def _seed(base, subdir: str, name: str, **fields):
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(yaml.safe_dump({"name": name, **fields}), encoding="utf-8")


def _overlay(base, agent_section: dict):
    base.mkdir(parents=True, exist_ok=True)
    (base / "overrides.yaml").write_text(yaml.safe_dump({"agent": agent_section}), encoding="utf-8")


def test_overlay_agent_scalars_reach_load_agent(tmp_path):
    """The e2e the blocker is about: set it, load it, get it back."""
    _seed(tmp_path, "agents", "a1", role="r")
    _overlay(tmp_path, {"temperature": 0.2, "max_tokens": 111, "model": "overlay-model"})

    cfg = ConfigLoader(config_dir=tmp_path).load_agent("a1")

    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 111
    assert cfg.model == "overlay-model"


def test_explicit_agent_yaml_still_beats_the_overlay(tmp_path):
    """The overlay is a DEFAULT layer, not an override — the agent's own yaml wins."""
    _seed(tmp_path, "agents", "a1", role="r", temperature=0.9, max_tokens=222, model="yaml-model")
    _overlay(tmp_path, {"temperature": 0.2, "max_tokens": 111, "model": "overlay-model"})

    cfg = ConfigLoader(config_dir=tmp_path).load_agent("a1")

    assert cfg.temperature == 0.9
    assert cfg.max_tokens == 222
    assert cfg.model == "yaml-model"


def test_org_inheritance_still_beats_the_overlay(tmp_path):
    """Org/division inheritance sits above the overlay (loader step 5b's stated contract)."""
    _seed(tmp_path, "agents", "a1", role="r")
    (tmp_path / "org.yaml").write_text(
        yaml.safe_dump({"name": "o", "default_temperature": 0.45}), encoding="utf-8"
    )
    _overlay(tmp_path, {"temperature": 0.2})

    assert ConfigLoader(config_dir=tmp_path).load_agent("a1").temperature == 0.45


def test_no_overlay_keeps_the_shipped_defaults(tmp_path):
    """Byte-identical when nothing sets the overlay: the schema defaults still apply."""
    _seed(tmp_path, "agents", "a1", role="r")

    cfg = ConfigLoader(config_dir=tmp_path).load_agent("a1")

    assert (cfg.temperature, cfg.max_tokens, cfg.model) == (0.6, 4096, "inherit")
