"""Shared `/model` operations used by BOTH the REPL `/model` command and the
`localharness model` CLI.

The full REPL swap — live client mutation, TokenCounter rebind, managed-server restart,
channel I/O — stays in ``repl.py``; it is agent-loop-coupled and cannot run without a live
session. What is extracted here is the agent-loop-INDEPENDENT decision + persistence layer:
which models exist, and how a chosen default is durably + atomically recorded (mirroring the
`components set` overlay path, issue #22) and which loaded agents a persisted switch will NOT
reach because their per-agent yaml pins a concrete model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from localharness.config.models import HarnessConfig
from localharness.config.overlay import (
    _resolve_user_overlay_path,
    atomic_write_overlay,
    deep_merge,
    load_overlay,
)
from localharness.core.bus import EventBus
from localharness.core.events import ComponentMutated
from localharness.registry import set_value_in_dict

_AGENT_KEY = "agent"


def _merged_available_models(harness: Any, existing_overlay: dict, model: str) -> list[str]:
    """Union every known model name. The overlay's ``deep_merge`` REPLACES lists wholesale
    (config/overlay.py), so writing a bare ``[model]`` would DROP the rest — read-merge the
    in-memory harness list, the overlay's current list, and the new model, order-preserving."""
    out: list[str] = []
    overlay_avail = (existing_overlay.get("provider") or {}).get("available_models") or []
    for src in (list(harness.provider.available_models), overlay_avail, [model]):
        for m in src:
            if m and m not in out:
                out.append(m)
    return out


async def persist_default_model(harness: Any, model: str, *, actor: str = "cli") -> None:
    """Persist ``model`` as the org + provider default via the atomic USER OVERLAY — the same
    crash-safe, audited path ``components set`` uses (issue #22), replacing the prior full,
    non-atomic ``config.yaml`` rewrite.

    Flow: load overlay -> set ``provider.default_model`` + ``org.default_model`` +
    union(``available_models``) -> validate the merged HarnessConfig (raises on an invalid
    result, e.g. a collision with a configured ``proposer.model``) -> ``atomic_write_overlay``
    -> emit one ``ComponentMutated`` per path. Only ``provider.*`` / ``org.*`` are ever touched
    in the overlay, so an ``agent:`` slice (the kill-lever layer) is preserved untouched.
    Also mutates the in-memory ``harness`` so the live session's view stays consistent.
    """
    before_provider = harness.provider.default_model
    before_org = harness.org.default_model

    overlay_path = _resolve_user_overlay_path()
    existing = load_overlay(overlay_path)
    merged_avail = _merged_available_models(harness, existing, model)

    new_overlay = dict(existing)
    set_value_in_dict(new_overlay, "provider.default_model", model)
    set_value_in_dict(new_overlay, "org.default_model", model)
    set_value_in_dict(new_overlay, "provider.available_models", merged_avail)

    # Validate the SAME cascade the next `start` sees: current config ⊕ new overlay. Exclude the
    # agent-scope `agent:` section (not a HarnessConfig field — mirrors components_cmd and
    # load_harness's overlay handling). Raises ValidationError if the result is invalid.
    harness_overlay = {k: v for k, v in new_overlay.items() if k != _AGENT_KEY}
    HarnessConfig.model_validate(deep_merge(harness.model_dump(mode="python"), harness_overlay))

    atomic_write_overlay(overlay_path, new_overlay)

    # Keep the in-memory harness consistent with what was just persisted.
    harness.provider.default_model = model
    harness.org.default_model = model
    harness.provider.available_models = merged_avail

    # Audit trail (mirrors components_cmd): one ComponentMutated per path written.
    audit_path = harness.org.audit_log_path
    bus = EventBus(persist_path=Path(audit_path).expanduser() if audit_path else None)
    for path, before in (
        ("provider.default_model", before_provider),
        ("org.default_model", before_org),
    ):
        await bus.publish(
            ComponentMutated(
                path=path,
                before_value=before,
                after_value=model,
                layer="user",
                actor=actor,  # type: ignore[arg-type]
            )
        )
