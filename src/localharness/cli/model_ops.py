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

import yaml

from localharness.config.models import HarnessConfig
from localharness.config.overlay import (
    _resolve_user_overlay_path,
    atomic_write_overlay,
    deep_merge,
    load_overlay,
)
from localharness.config.paths import resolve_runtime_path
from localharness.core.bus import EventBus
from localharness.core.events import ComponentMutated
from localharness.registry import set_value_in_dict

_AGENT_KEY = "agent"


def list_live_models(base_url: str, timeout: float = 3.0) -> tuple[list[str], bool]:
    """Probe the OpenAI-compatible ``/models`` endpoint. Returns ``(model_ids, reachable)``.

    ``reachable`` is False ONLY when the endpoint could not be reached at all — kept distinct
    from reached-but-empty so callers can fail-loud on a bad target vs. degrade-with-disclosure
    on an unreachable runtime (the #16 lesson: a silently-wrong endpoint must not read as a
    legitimate empty result). Parses only the OpenAI ``data`` shape; Ollama/LM-Studio-aware
    listing (audit gap #2) is out of scope here — same shape the REPL /model list uses.
    """
    import httpx
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return [m["id"] for m in resp.json().get("data", [])], True
    except Exception:
        return [], False


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


async def persist_default_model(
    harness: Any, model: str, *, actor: str = "cli", config_dir: Any = None
) -> None:
    """Persist ``model`` as the org + provider default via the atomic USER OVERLAY — the same
    crash-safe, audited path ``components set`` uses (issue #22), replacing the prior full,
    non-atomic ``config.yaml`` rewrite.

    Flow: load overlay -> set ``provider.default_model`` + ``org.default_model`` +
    union(``available_models``) -> validate the merged HarnessConfig (raises on an invalid
    result, e.g. a collision with a configured ``proposer.model``) -> ``atomic_write_overlay``
    -> emit one ``ComponentMutated`` per path. Only ``provider.*`` / ``org.*`` are ever touched
    in the overlay, so an ``agent:`` slice (the kill-lever layer) is preserved untouched.
    Also mutates the in-memory ``harness`` so the live session's view stays consistent.

    ``config_dir`` (#35): the SAME resolved dir the harness was loaded from. The overlay write
    target and the audit-log path resolve against it — so a persist tracks ``--config-dir``
    instead of leaking to ``~/.localharness``. Callers thread their loader's ``_config_dir``.
    """
    before_provider = harness.provider.default_model
    before_org = harness.org.default_model

    overlay_path = _resolve_user_overlay_path(config_dir)
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

    # Audit trail (mirrors components_cmd): one ComponentMutated per path written. The audit
    # path resolves against the SAME config_dir (#35 — a bare default 'audit.jsonl' lands under it).
    audit_path = harness.org.audit_log_path
    bus = EventBus(
        persist_path=resolve_runtime_path(audit_path, config_dir) if audit_path else None
    )
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


def pinned_agents(config_dir: Path | None) -> list[tuple[str, str]]:
    """Agents whose per-agent yaml pins a concrete ``model:`` (not ``"inherit"``).

    A persisted org/provider ``default_model`` switch will NOT reach these on the next
    ``start`` — ``start_cmd`` resolves the per-agent pin first. That precedence is BY DESIGN
    (a deliberate override lever, e.g. the owner's orchestrator.yaml), so this only WARNS; it
    changes no behavior. Read the RAW yaml ``model`` field, NOT ``load_agent().model``: the
    loader RESOLVES ``inherit`` to the org default (a concrete string), which would make every
    inheriting agent look pinned. Returns ``[(agent_name, pinned_model), ...]``.
    """
    out: list[tuple[str, str]] = []
    if config_dir is None:
        return out
    agents_dir = Path(config_dir) / "agents"
    if not agents_dir.is_dir():
        return out
    for yml in sorted(agents_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        m = raw.get("model") if isinstance(raw, dict) else None
        if isinstance(m, str) and m and m != "inherit":
            out.append((raw.get("name") or yml.stem, m))
    return out
