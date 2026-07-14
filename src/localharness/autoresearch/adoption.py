"""AUTO-04 (AMENDED — auto-adopt, NOT a blocking human gate): commit a clean win into LIVE config.

A clean win is adopted by writing the after-value into the project-local
``{repo_root}/.localharness/overrides.yaml`` (the config cascade's highest-priority
layer, inside the git tree) using the EXACT ``components set`` overlay primitives, then
``git add`` + ``git commit`` in the MAIN repo. Reverting an adoption is literally
``git revert <sha>``. The human reviews ASYNCHRONOUSLY via the Phase 19 daily report;
nothing here blocks on human input.

Compound-live works for free: the next experiment's throwaway checkout opens at the new
HEAD, so the single-component overlay composes on prior adoptions automatically (no replay).

Defense-in-depth: ``adopt`` re-asserts the gate's anti-reward-hacking seal AND re-validates
the merged config BEFORE committing to live config. A sealed/off-registry/multi-component
row, or an after-value that produces an invalid config, raises ``AdoptionRefused`` with NO
overlay write and NO commit.

Reused primitives (the verified components-set path + the experiment-seal helpers):
  - 14: set_value_in_dict / coerce_value / build_catalogue   (registry)
        atomic_write_overlay / deep_merge / load_overlay     (config.overlay)
        HarnessConfig / AgentConfig                           (config.models)
  - 17: _OFFREGISTRY_PREFIXES / _is_multi_component idiom + the subprocess-git idiom
        (replicated here, defense-in-depth — NOT imported, so the seal cannot drift away
         from this boundary if experiment.py changes)
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from localharness.config.models import AgentConfig, HarnessConfig
from localharness.config.overlay import atomic_write_overlay, deep_merge, load_overlay, _resolve_user_overlay_path
from localharness.registry import build_catalogue, coerce_value, set_value_in_dict

# Defense-in-depth: re-assert the gate's anti-reward-hacking seal before committing to LIVE
# config (mirror experiment.py verbatim — the seal must hold at THIS boundary independently).
_OFFREGISTRY_PREFIXES = ("bench.", "scenario", "grader", "success_criteria", "holdout", "sentinel",
                         "org.enforce_capability_floor")
_MULTI_PATH_PATTERN = re.compile(r"[,\s;]")

# A registry-addressed agent component lives under the `agent.` namespace, which is NOT a key
# of HarnessConfig (extra="forbid"); it validates against AgentConfig instead. The placeholder
# name satisfies AgentConfig's name validator (lowercase-alnum-hyphen) without touching disk.
_AGENT_PREFIX = "agent."
_AGENT_VALIDATE_BASE = {"name": "adopt-validate", "role": "adopt-validate"}


class AdoptionRefused(Exception):
    """Raised when a row fails the seal/validation re-check at adoption time (status -> adoption_rejected)."""


def _is_multi_component(component: str, after: Any) -> bool:
    """True iff the proposal resolves to >1 component path (delimiter in path OR multi-key dot-path map)."""
    if _MULTI_PATH_PATTERN.search(component):
        return True
    if isinstance(after, dict) and len(after) > 1 and all(isinstance(k, str) and "." in k for k in after):
        return True
    return False


def _git(repo_root: Path, *args: str) -> str:
    """Run `git -C <repo_root> <args>` (check=True) and return trimmed stdout. Mirrors experiment.py."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _resolve_cfg(cfg):
    """Resolve a HarnessConfig when the caller passes cfg=None (LOCALHARNESS_HOME-rooted loader).

    Mirrors run_experiment's cfg=None path. The loop/CLI normally pass a live cfg; tests pass
    None alongside the components_home fixture, so the loader reads the hermetic home config.
    """
    if cfg is not None:
        return cfg
    from localharness.cli.components_cmd import _build_loader

    return _build_loader().load_harness()


def _validate_merged(cfg, component: str, new_overlay: dict) -> None:
    """Validate the new overlay BEFORE any disk write. Raises AdoptionRefused on an invalid config.

    The `agent.` namespace is a registry addressing convention, not a HarnessConfig field
    (HarnessConfig is extra="forbid"), so an agent.* overlay validates against AgentConfig and a
    harness-level overlay validates against the merged HarnessConfig dict. Either failure refuses
    the adoption with no write/commit.
    """
    try:
        if component.startswith(_AGENT_PREFIX):
            agent_overlay = new_overlay.get("agent", {})
            merged_agent = deep_merge(dict(_AGENT_VALIDATE_BASE), agent_overlay)
            AgentConfig.model_validate(merged_agent)
        else:
            project_dict = cfg.model_dump(mode="python") if hasattr(cfg, "model_dump") else {}
            merged = deep_merge(project_dict, new_overlay)
            HarnessConfig.model_validate(merged)
    except Exception as exc:  # pydantic ValidationError or any validate failure
        raise AdoptionRefused(
            f"adopting {component!r} produces an invalid config: {exc}"
        ) from exc


async def adopt(proposal_id: str, *, store, cfg, repo_root, bus=None) -> str:
    """Adopt a clean win into the LIVE project-local overlay + git commit. Returns the 40-char commit sha.

    Reuses the components-set overlay primitives + subprocess git. Re-asserts the seal
    (off-registry / multi-component / not-in-registry) and validates the merged config BEFORE
    writing — a failure marks the row ``adoption_rejected`` and raises ``AdoptionRefused`` with
    NO commit. Commits in the MAIN repo (``repo_root``), NOT the gate's throwaway checkout.

    NOTE: adopt() does NOT set status="adopted" on success — it returns the sha. The LOOP (18-05) /
    CLI (18-06) calls ``store.update_verdict(status="adopted")`` after a successful adopt(), mirroring
    the experiment runner's run-vs-verdict separation. The ``adoption_rejected`` status IS set here
    because it is an in-adopt refusal.
    """
    repo_root = Path(repo_root)
    cfg = _resolve_cfg(cfg)

    entry = await store.get(proposal_id)
    if entry is None:
        raise AdoptionRefused(f"no archive row for id {proposal_id!r}")
    # A row already declined at adoption is never re-offered or re-committed (the loop excludes
    # adoption_rejected from the held/re-offer list; this is the seam that locks it at adopt()).
    if entry.status == "adoption_rejected":
        raise AdoptionRefused(
            f"row {proposal_id!r} was already declined (adoption_rejected); not re-adopting"
        )
    decoded = entry.diff_decoded
    after_raw = decoded.get("after")
    component = entry.component

    # 1. Seal re-check (defense-in-depth — guards archive corruption / future schema slip).
    #    MUST run BEFORE any overlay write: adoption can NEVER widen the mutable surface to the
    #    grader/bench/holdout/multi-component surface.
    from localharness.autoresearch.experiment import _provenance_agent_cfg
    _user_overlay = load_overlay(_resolve_user_overlay_path())
    catalogue = build_catalogue(
        cfg,
        agent_cfg=_provenance_agent_cfg(),
        overlays={"user": _user_overlay},
    )
    cat_entry = catalogue.get(component)
    if (
        _is_multi_component(component, after_raw)
        or any(component.startswith(p) for p in _OFFREGISTRY_PREFIXES)
        or cat_entry is None
    ):
        await store.update_verdict(proposal_id, status="adoption_rejected")
        raise AdoptionRefused(f"component refused at adoption: {component!r}")

    # 2. Type-coerce the after value (mirror experiment.py write_experiment_overlay / components set).
    after = (
        after_raw
        if isinstance(after_raw, (dict, list))
        else coerce_value(str(after_raw), cat_entry.annotation)
    )

    # 3. Build the LIVE project-local overlay; validate the MERGED config BEFORE any write.
    overlay_path = repo_root / ".localharness" / "overrides.yaml"
    existing = load_overlay(overlay_path)
    new_overlay = set_value_in_dict(dict(existing), component, after)
    try:
        _validate_merged(cfg, component, new_overlay)
    except AdoptionRefused:
        await store.update_verdict(proposal_id, status="adoption_rejected")
        raise

    # 4. Atomic overlay write (does NOT touch disk until validation passed).
    atomic_write_overlay(overlay_path, new_overlay)

    # 5. Audit event (layer='user', actor='orchestrator', actor_detail=proposal_id).
    await _emit_component_mutated(bus, cfg, component, decoded.get("before"), after, proposal_id)

    # 6. git add + commit IN THE MAIN REPO (repo_root, NOT the gate's throwaway checkout) ->
    #    compound-live: the next experiment opens at this new HEAD, so overlays compose automatically.
    _git(repo_root, "add", str(overlay_path))
    tscore = f"{entry.train_score:.3f}" if entry.train_score is not None else "n/a"
    msg = f"autoresearch: adopt {component} ({proposal_id[:8]}) train={tscore}"
    _git(repo_root, "commit", "-m", msg)
    return _git(repo_root, "rev-parse", "HEAD")


async def _emit_component_mutated(bus, cfg, component, before, after, proposal_id) -> None:
    """Publish ComponentMutated(layer='user', actor='orchestrator', actor_detail=proposal_id).

    Uses the injected bus if provided (tests subscribe to it); otherwise builds one pointed at
    cfg.org.audit_log_path (mirrors components_cmd / experiment.py's audit path).
    """
    from localharness.core.events import ComponentMutated

    target_bus = bus
    if target_bus is None:
        from localharness.core.bus import EventBus
        from localharness.config.paths import resolve_runtime_path

        audit_path = getattr(getattr(cfg, "org", None), "audit_log_path", None)
        # #35: a bare default 'audit.jsonl' resolves under the config dir (env/~default), not CWD.
        target_bus = EventBus(
            persist_path=resolve_runtime_path(audit_path) if audit_path else None
        )
    await target_bus.publish(
        ComponentMutated(
            path=component,
            before_value=before,
            after_value=after,
            layer="user",
            actor="orchestrator",
            actor_detail=proposal_id,
        )
    )
