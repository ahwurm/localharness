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
from localharness.config.paths import resolve_runtime_path
from localharness.core.bus import EventBus
from localharness.core.events import ComponentMutated
from localharness.registry import set_value_in_dict

_AGENT_KEY = "agent"


class MalformedModelListError(Exception):
    """The endpoint was REACHED but its ``/models`` reply wasn't an OpenAI-compatible model list
    (HTML error page, wrong API, unexpected JSON shape). Kept distinct from unreachable (#38) so
    callers stop rendering "is it running?" — the server IS running, base_url points at the
    wrong thing."""


def list_live_models(base_url: str, timeout: float = 3.0) -> tuple[list[str], bool]:
    """Probe the OpenAI-compatible ``/models`` endpoint. Returns ``(model_ids, reachable)``.

    Three outcomes, three signals (#38 — previously one blanket ``except`` collapsed the last two
    into a bogus ``reachable=False``, so a live-but-wrong endpoint read as "is it running?"):
      - transport error (connection refused/DNS/timeout) -> ``([], False)`` = unreachable;
      - reached, valid OpenAI ``data`` shape -> ``(ids, True)`` (empty list stays reachable — a
        legit empty result, the #16 lesson);
      - reached but the body isn't an OpenAI model list -> raise ``MalformedModelListError``.

    Parses only the OpenAI ``data`` shape; Ollama/LM-Studio-aware listing (audit gap #2) is out
    of scope here — same shape the REPL /model list uses (which now delegates here).
    """
    import httpx
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    except httpx.RequestError:
        return [], False  # transport failure → genuinely unreachable
    try:
        return [m["id"] for m in resp.json()["data"]], True
    except (ValueError, KeyError, TypeError) as exc:
        # Reached, but not an OpenAI /models list (bad JSON, missing `data`, wrong item shape).
        raise MalformedModelListError(
            f"{base_url} responded, but not with an OpenAI-compatible model list ({exc})"
        ) from exc


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
) -> str | None:
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

    Returns None on clean success, or a warning string (#37) when ONLY the post-write audit
    emit failed — the durable overlay write already succeeded, so callers surface this as a
    secondary note and still report the switch as done (never as a persist failure).
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
    # #34: with a managed server configured, ALSO persist server.model — else cold start rebuilds
    # `vllm serve <srv.model>` from the stale config value and relaunches the old model. Only when
    # a server block exists (never invent one: a bare server:{model} fails validation).
    if harness.server is not None:
        set_value_in_dict(new_overlay, "server.model", model)

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
    if harness.server is not None:
        harness.server.model = model

    # Audit trail (mirrors components_cmd): one ComponentMutated per path written. The audit
    # path resolves against the SAME config_dir (#35 — a bare default 'audit.jsonl' lands under it).
    # #37: scope the emit in its OWN try/except — it runs AFTER the durable overlay write, so a
    # failure here (unwritable audit log) must NOT surface as a persist failure. Return a warning
    # the callers show as a secondary note; the switch itself already succeeded.
    try:
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
    except Exception as exc:  # noqa: BLE001 — the durable overlay write already succeeded
        return f"persisted, but the audit log could not be written: {exc}"
    return None


def pinned_agents(config_dir: Path | None) -> list[tuple[str, str]]:
    """Agents a persisted org/provider ``default_model`` switch will NOT reach on the next
    ``start`` — because a concrete ``model:`` pins them ABOVE the org default in the resolution
    chain ``agent -> division -> org`` (``start_cmd`` resolves that chain).

    Two pin sources, both reported (#36 — the division source was previously missed, so a
    division-pinned agent silently ignored a switch with no warning):
      - the agent's own raw ``model:`` (an agent-level override lever) -> ``(name, model)``;
      - else, its division's ``model:`` (a division-wide pin) -> ``(name (via division X), model)``.
    An agent that inherits all the way to org is NOT listed — the switch DOES reach it.

    Reads RAW yaml via the loader's own reader (NOT ``load_agent().model``, which RESOLVES
    ``inherit`` to a concrete org default and would make every inheritor look pinned). Returns
    ``[(agent_label, pinned_model), ...]``.
    """
    out: list[tuple[str, str]] = []
    if config_dir is None:
        return out
    from localharness.config.loader import ConfigLoader, _load_yaml_file

    loader = ConfigLoader(config_dir=Path(config_dir))
    for stem in loader.list_agents():
        path = loader._find_file("agents", stem)
        if path is None:
            continue
        try:
            raw = _load_yaml_file(path)  # the loader's own yaml reader (shared parse/error path)
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        label = raw.get("name") or stem
        m = raw.get("model")
        if isinstance(m, str) and m and m != "inherit":
            out.append((label, m))  # agent-level pin
            continue
        # Agent inherits at its own level — a division-level pin still traps it.
        div_name = raw.get("division")
        if not div_name:
            continue
        try:
            division = loader.load_division(div_name)
        except Exception:
            continue
        if division.model and division.model != "inherit":
            out.append((f"{label} (via division {div_name})", division.model))
    return out
