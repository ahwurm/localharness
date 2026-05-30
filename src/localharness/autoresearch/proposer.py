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


def _build_reflection_messages(
    component: str, before: object, type_name: str, traces: list[list]
) -> list[dict]:
    """Reflection prompt: failed-trace evidence + current value → one typed change."""
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
        'Respond with JSON only: {"after": <new value for this component>, '
        '"rationale": <one sentence tying the change to the failures>}.'
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
    messages = _build_reflection_messages(component, before, entry.type_name, failed)
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
