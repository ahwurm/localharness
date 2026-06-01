"""Autoresearch proposer core (PROP-01/02/03).

Composes Phase 13/14/15 primitives + one model call into the mutation generator:
  - reads FAILED *train* traces (the reward-signal evidence),
  - asks a STRONGER, DISTINCT model (ProposerConfig — never the main provider model)
    for ONE change to ONE component,
  - returns an archive-shape Proposal {"before": current_value, "after": typed_after}.

Sealed-slice seal (PROP-03, second lock): `propose()` refuses ANY holdout run_id or
unknown-scenario run_id at ENTRY, BEFORE any model construction or call. The seal
resolves slice from the corpus fixture YAML (never the trace), so a renamed/removed
fixture or a smuggled holdout is refused. The CLI (16-03) maps ProposerError → exit 2.

return-only: propose() writes nothing. The opt-in `--archive` flag lives in 16-03;
Phase 17 is the primary archive writer. The pipeline stays pure/testable — the CLI
resolves bench paths via load_bench_config and passes corpus_path/results_path down.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from localharness.bench.orchestrator import _discover_scenarios
from localharness.bench.schema import load_scenario
from localharness.bench.runner import resolve_run_path
from localharness.core.events import ScenarioCompleted, deserialize_event
from localharness.provider.client import LLMClient, LLMConfig
from localharness.registry import build_catalogue, coerce_value


def _provenance_agent_cfg():
    """The live ADOPTED agent.* config (overrides.yaml, no experiment overlay) for catalogue
    `before` provenance. Returns None when nothing is adopted — behavior-identical to the old
    agent_cfg=None (build_catalogue's own model_construct default); only becomes a truthful
    live config once an agent.* mutation IS adopted (WARNING-2).
    """
    from localharness.config.models import AgentConfig
    from localharness.config.overlay import _resolve_user_overlay_path, deep_merge, load_overlay
    agent_overlay = load_overlay(_resolve_user_overlay_path()).get("agent", {})
    if not agent_overlay:
        return None
    return AgentConfig.model_validate(deep_merge({"name": "provenance", "role": "provenance"}, agent_overlay))


class ProposerError(Exception):
    """Raised on any proposer refusal (CLI maps to exit code 2)."""


class _RawProposal(BaseModel):
    """Schema the model's JSON reply must satisfy. `component` is NEVER taken from
    the model — it is pinned to the caller's input (SC4 atomicity)."""
    after: object
    rationale: str = Field(min_length=1)


def _parse(raw: str) -> _RawProposal:
    """Parse + schema-validate the model reply. Fails EXPLICITLY (never silently)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json  # optional local-model garble fallback

            data = json.loads(repair_json(raw))
        except Exception as exc:
            raise ProposerError(
                f"proposer returned unparseable JSON: {exc}\nraw={raw[:500]}"
            ) from exc
    if not isinstance(data, dict):
        raise ProposerError(f"proposer output is not a JSON object: {raw[:500]}")
    try:
        return _RawProposal.model_validate(data)
    except ValidationError as exc:
        raise ProposerError(
            f"proposer proposal failed schema validation: {exc}\nraw={raw[:500]}"
        ) from exc


def _render_pareto_evidence(front: list) -> str:
    """Render the per-fixture Pareto front into a compact reflection block (MODP-03).

    One line per train fixture: the best rate currently achieved and the mutation
    (component) that holds it — the GEPA "reflect across the front" signal so the
    proposer does not regress a fixture another mutation already wins. Sealed-slice-safe:
    reads ONLY each entry's train_scores_per_fixture (the sealed train slice; the sealed
    columns are never referenced). Returns "" for an empty/absent front (the back-compat
    sentinel — no Pareto block is then injected).
    """
    best: dict[str, tuple[float, str]] = {}   # fixture -> (best_rate, holder_component)
    for e in front:
        scores = getattr(e, "train_scores_per_fixture", None) or {}
        for fixture, rate in scores.items():
            if fixture not in best or rate > best[fixture][0]:
                best[fixture] = (rate, e.component)
    if not best:
        return ""
    return "\n".join(
        f"- {fixture}: best rate {rate:.2f} ({component})"
        for fixture, (rate, component) in sorted(best.items())
    )


async def _fetch_pareto_front(store) -> list:
    """Fetch the EXISTING per-fixture Pareto front for MODP-03 reflection.

    ``store=None`` ⇒ ``[]`` (the back-compat no-evidence path). When a store is given it
    is opened if not already open (a missing/empty archive DB is created empty by
    ``open()`` ⇒ the front is just ``[]``); a store this helper opened is closed again,
    while an already-open caller-owned store is left untouched. The front is the existing
    ``ArchiveStore.pareto_front_per_fixture()`` — never recomputed, sealed-slice-safe.
    """
    if store is None:
        return []
    opened_here = getattr(store, "_db", None) is None
    if opened_here:
        await store.open()
    try:
        return await store.pareto_front_per_fixture()
    finally:
        if opened_here:
            await store.close()


def _build_reflection_messages(
    component: str, before: object, type_name: str, traces: list[list],
    *, pareto_evidence: str = "",
) -> list[dict]:
    """Reflection prompt: failed-trace evidence + current value → one typed change.

    When ``pareto_evidence`` is non-empty (the per-fixture Pareto front rendered by
    ``_render_pareto_evidence``), it is injected as an ADDITIONAL evidence block and the
    rationale contract is augmented to require tying the change to BOTH the failures AND
    the front (MODP-03 — GEPA "reflect across the front"). With the default empty string
    the built messages are byte-identical to the pre-MODP-03 builder (back-compat).
    """
    system = (
        "You propose ONE change to a single config component. "
        'Return ONLY JSON {"after": <new value>, "rationale": <why>}. '
        "Do not change any other component. "
        f"The target component is {component} (type {type_name})."
    )
    summaries = []
    for events in traces:
        for e in events:
            if isinstance(e, ScenarioCompleted) and not e.success:
                summaries.append(f"- scenario {e.scenario_name!r} FAILED (model {e.model})")
    evidence = "\n".join(summaries) if summaries else "- (failed train run)"
    user = (
        f"Component: {component} (type {type_name})\n"
        f"Current value:\n{json.dumps(before)}\n\n"
        f"Failed TRAIN traces (the change must address these failures):\n{evidence}\n\n"
        + (
            "Pareto front — current best per fixture (reflect across these; do NOT regress "
            f"a fixture another mutation already wins):\n{pareto_evidence}\n\n"
            if pareto_evidence
            else ""
        )
        + 'Respond with JSON only: {"after": <new value for this component>, '
        + (
            '"rationale": <one sentence tying the change to the failures AND the per-fixture '
            'Pareto evidence>}.'
            if pareto_evidence
            else '"rationale": <one sentence tying the change to the failures>}.'
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True)
class Proposal:
    component: str
    before: object
    after: object
    rationale: str
    tokens_used: int | None = None  # proposer CompletionUsage.total_tokens (AUTO-03 self-meter); None if unknown

    @property
    def diff(self) -> str:
        """Archive-shape diff JSON (LOCKED): {"before": ..., "after": ...}."""
        return json.dumps({"before": self.before, "after": self.after})


def _split_run_id(run_id: str) -> tuple[str, str, str]:
    """Split a run_id into (model, scenario_name, timestamp). Refuse anything else."""
    parts = run_id.split("/")
    if len(parts) != 3 or not all(parts):
        raise ProposerError(
            f"run_id {run_id!r} must be '{{model}}/{{scenario}}/{{timestamp}}'"
        )
    return parts[0], parts[1], parts[2]


def _slice_by_scenario(corpus_path: Path) -> dict[str, str]:
    """Map scenario_name → slice from the corpus YAML (the seal's source of truth)."""
    out: dict[str, str] = {}
    for sp in _discover_scenarios(corpus_path):
        try:
            spec = load_scenario(sp)
        except Exception:
            continue
        out[spec.name] = spec.slice
    return out


def _enforce_seal(run_ids: list[str], corpus_path: Path) -> None:
    """PROP-03 second lock. Refuse if ANY run_id maps to holdout OR an unknown scenario."""
    slice_map = _slice_by_scenario(corpus_path)
    for rid in run_ids:
        _, scen_name, _ = _split_run_id(rid)
        sl = slice_map.get(scen_name)
        if sl is None:
            raise ProposerError(
                f"unknown scenario {scen_name!r} for run_id {rid!r} — refusing "
                "(could be a renamed/removed fixture or a smuggled holdout)"
            )
        if sl != "train":
            raise ProposerError(
                f"run_id {rid!r} is slice={sl!r}; proposer reads TRAIN only (PROP-03)"
            )


def _load_failed_traces(run_ids: list[str], results_path: Path) -> list[list]:
    """Load each run_id's JSONL trace; keep only runs with a failed ScenarioCompleted.

    Refuse if no failed train traces remain — the proposer never hallucinates a
    mutation against an all-passing/empty evidence set.
    """
    traces: list[list] = []
    for rid in run_ids:
        model, scen_name, ts = _split_run_id(rid)
        path = resolve_run_path(results_path, model, scen_name, ts)
        if not path.exists():
            raise ProposerError(f"no trace at {path} for run_id {rid!r}")
        events = [
            deserialize_event(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        failed = any(isinstance(e, ScenarioCompleted) and not e.success for e in events)
        if failed:
            traces.append(events)
    if not traces:
        raise ProposerError(
            "no failed TRAIN traces among supplied run_ids — nothing to propose"
        )
    return traces


async def propose(
    component: str,
    run_ids,
    *,
    cfg,
    corpus_path: Path,
    results_path: Path,
    llm=None,
    store=None,
    archive: bool = False,
) -> Proposal:
    """Generate ONE typed mutation Proposal for ONE component from failed train traces.

    The seal runs FIRST (before any model construction/call); failed-evidence is
    loaded next. The model call + parse + coerce + diff assembly land in Task 3.
    """
    _enforce_seal(list(run_ids), corpus_path)  # FIRST — before anything else
    failed = _load_failed_traces(list(run_ids), results_path)

    # (A) Resolve the component + current value via the registry (reuse Phase 14).
    catalogue = build_catalogue(cfg, agent_cfg=_provenance_agent_cfg())  # live adopted agent.* before-value
    entry = catalogue.get(component)
    if entry is None:
        raise ProposerError(
            f"unknown component path {component!r} — run `localharness components list`"
        )
    before = entry.current_value

    # (B) Build the proposer model client from ProposerConfig (PROP-02) — NEVER
    #     the main harness model. Tests inject `llm` to stay hermetic.
    if llm is None:
        if cfg.proposer is None:
            raise ProposerError(
                "no [proposer] config — set proposer.base_url/model (PROP-02)"
            )
        pc = cfg.proposer
        # Probe the proposer endpoint to determine tool_call_mode (FIDEL-03).
        # Reuses the same detect_capabilities surface as the matrix path — model-agnostic.
        _probe_client = LLMClient(LLMConfig(base_url=pc.base_url, model=pc.model, api_key=pc.api_key))
        _cap = await _probe_client.detect_capabilities()
        llm = LLMClient(
            LLMConfig(
                base_url=pc.base_url,
                model=pc.model,
                api_key=pc.api_key,
                timeout_seconds=pc.timeout_seconds,
                temperature=pc.temperature,
                max_tokens=pc.max_tokens,
                is_local=pc.is_local,
                tool_call_mode=_cap.tool_call_mode,
            )
        )

    # (C) Call the model (AFTER the seal + after `before` is read), parse, coerce.
    #     Reflect across the per-fixture Pareto front (MODP-03) when a store is given —
    #     the EXISTING ArchiveStore.pareto_front_per_fixture() (sealed-slice-safe), never
    #     recomputed. No store ⇒ [] ⇒ pareto_evidence="" ⇒ today's messages (back-compat).
    front = await _fetch_pareto_front(store)
    pareto_evidence = _render_pareto_evidence(front)
    messages = _build_reflection_messages(
        component, before, entry.type_name, failed, pareto_evidence=pareto_evidence
    )
    msg, usage = await llm.complete(messages)
    raw = msg.content or ""
    parsed = _parse(raw)
    try:
        typed_after = (
            parsed.after
            if isinstance(parsed.after, (dict, list))
            else coerce_value(str(parsed.after), entry.annotation)
        )
    except ValueError as exc:
        raise ProposerError(
            f"after value {parsed.after!r} invalid for {component} "
            f"({entry.type_name}): {exc}"
        ) from exc

    # (D) Proposal.component is pinned to the input — the model never widens scope (SC4).
    return Proposal(
        component=component,
        before=before,
        after=typed_after,
        rationale=parsed.rationale,
        tokens_used=(usage.total_tokens if usage is not None else None),
    )
