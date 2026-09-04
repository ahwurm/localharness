"""AUTO-04 (AMENDED — auto-adopt, NOT a blocking human gate): write a clean win into LIVE config.

A clean win is adopted by writing the after-value into the GLOBAL user overlay
(``<config_dir>/overrides.yaml``, resolved by ``_resolve_user_overlay_path``) using the EXACT
``components set`` overlay primitives. That is the file this module, ``experiment.py`` and
``proposer.py`` all READ for `before` provenance, so an adoption is visible to the next
proposal — the write and the reads are the same layer. The human reviews ASYNCHRONOUSLY via
the Phase 19 daily report; nothing here blocks on human input.

Adoption used to write ``{repo_root}/.localharness/overrides.yaml`` and ``git add`` + commit it.
Two things were wrong with that: the project dotdir is gitignored in most checkouts (this repo
included), so the add exited non-zero and killed the loop at its first success with the overlay
already written; and no reader ever looked at that file — the gate's arms are built from the
worktree checkout, which carries the committed tree only, and every provenance reader looks at
the global overlay. REVERSIBILITY: an adoption is no longer a git commit, so it is undone with
``localharness components set <path> <before>`` (or by editing the global overlay); the archive
row keeps the exact before/after and the audit log keeps the ComponentMutated event.

Git is now touched ONLY to stamp the HEAD the win was measured at, and never fatally: any git
failure logs and yields an empty sha, because the adoption (overlay + archive row + audit) has
already completed and the loop must not die on bookkeeping.

Defense-in-depth: ``adopt`` re-asserts the gate's anti-reward-hacking seal AND re-validates
the merged config BEFORE writing live config. A sealed/off-registry/multi-component row, or an
after-value that produces an invalid config, raises ``AdoptionRefused`` with NO overlay write.

Reused primitives (the verified components-set path + the experiment-seal helpers):
  - 14: set_value_in_dict / coerce_value / build_catalogue   (registry)
        atomic_write_overlay / deep_merge / load_overlay     (config.overlay)
        HarnessConfig / AgentConfig                           (config.models)
  - 17: _OFFREGISTRY_PREFIXES / _is_multi_component idiom + the subprocess-git idiom
        (replicated here, defense-in-depth — NOT imported, so the seal cannot drift away
         from this boundary if experiment.py changes)
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from localharness.config.models import AgentConfig, HarnessConfig
from localharness.config.overlay import atomic_write_overlay, deep_merge, load_overlay, _resolve_user_overlay_path
from localharness.registry import (
    LAYER_GLOBAL_OVERRIDES,
    build_catalogue,
    coerce_value,
    set_value_in_dict,
)

# Defense-in-depth: re-assert the gate's anti-reward-hacking seal before writing LIVE config
# (mirror experiment.py verbatim — the seal must hold at THIS boundary independently, and must not
# drift). See experiment.py's copy for which of these name real registry entries (sentinel.*,
# org.enforce_capability_floor, active_endpoint*, extra_endpoints — sealed ONLY by this tuple) and
# which are bench-side paths the catalogue lookup already refuses by omission.
_OFFREGISTRY_PREFIXES = ("bench.", "scenario", "grader", "success_criteria", "holdout", "sentinel",
                         "org.enforce_capability_floor",
                         "active_endpoint", "extra_endpoints")
_MULTI_PATH_PATTERN = re.compile(r"[,\s;]")

# A registry-addressed agent component lives under the `agent.` namespace, which is NOT a key
# of HarnessConfig (extra="forbid"); it validates against AgentConfig instead. The placeholder
# name satisfies AgentConfig's name validator (lowercase-alnum-hyphen) without touching disk.
_AGENT_PREFIX = "agent."
_AGENT_VALIDATE_BASE = {"name": "adopt-validate", "role": "adopt-validate"}

logger = logging.getLogger(__name__)


class AdoptionRefused(Exception):
    """Raised when a row fails the seal/validation re-check at adoption time (status -> adoption_rejected)."""


def _is_multi_component(component: str, after: Any) -> bool:
    """True iff the proposal resolves to >1 component path (delimiter in path OR multi-key dot-path map)."""
    if _MULTI_PATH_PATTERN.search(component):
        return True
    if isinstance(after, dict) and len(after) > 1 and all(isinstance(k, str) and "." in k for k in after):
        return True
    return False


def _head_sha(repo_root: Path) -> str:
    """The repo's HEAD sha — provenance for WHICH revision of the code the win was measured on.

    NEVER fatal. The adoption (overlay write + archive row + audit event) is already complete when
    this runs, so a missing repo / missing git / unborn HEAD logs a warning and yields "" rather
    than raising into the loop, which only catches AdoptionRefused (F1: a raise here abandoned a
    successful adoption mid-flight).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("adoption provenance sha unavailable for %s: %s", repo_root, exc)
        return ""


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
    """Adopt a clean win into the LIVE global user overlay. Returns ``repo_root``'s HEAD sha ("" if git fails).

    Reuses the components-set overlay primitives. Re-asserts the seal (off-registry /
    multi-component / not-in-registry) and validates the merged config BEFORE writing — a failure
    marks the row ``adoption_rejected`` and raises ``AdoptionRefused`` with NO write. The overlay
    written is the one every provenance reader reads (see the module docstring); ``repo_root`` is
    only read, for the returned sha — nothing is staged, committed, or written into the repo.

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
    # ONE path for the layer this module reads AND writes (the F3 coherence fix lives here).
    overlay_path = _resolve_user_overlay_path()
    _user_overlay = load_overlay(overlay_path)
    catalogue = build_catalogue(
        cfg,
        agent_cfg=_provenance_agent_cfg(),
        overlays={LAYER_GLOBAL_OVERRIDES: _user_overlay},
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

    # 3. Build the LIVE global overlay (same file read above); validate the MERGE BEFORE any write.
    new_overlay = set_value_in_dict(dict(_user_overlay), component, after)
    try:
        _validate_merged(cfg, component, new_overlay)
    except AdoptionRefused:
        await store.update_verdict(proposal_id, status="adoption_rejected")
        raise

    # 4. Atomic overlay write (does NOT touch disk until validation passed).
    atomic_write_overlay(overlay_path, new_overlay)

    # 5. Audit event (layer='user', actor='orchestrator', actor_detail=proposal_id).
    await _emit_component_mutated(bus, cfg, component, decoded.get("before"), after, proposal_id)

    # 6. Provenance ONLY: the HEAD the win was measured at. The adoption is complete at step 5 —
    #    nothing is staged or committed (the overlay lives outside the repo; in most checkouts the
    #    project dotdir is gitignored, so the old `git add` here exited 1 and killed the loop).
    #    Adopted values reach the NEXT proposal through this overlay, not through the git tree.
    return _head_sha(repo_root)


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
