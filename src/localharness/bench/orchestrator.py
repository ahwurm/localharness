"""BENCH-01/02/06: top-level bench orchestrator — composes matrix x scenarios x runner.

Field naming (locked 11-02):
  BenchConfig.corpus_path
  BenchConfig.results_path
  MatrixEntry.provider     (e.g. "ollama")
  MatrixEntry.model_id     (passed verbatim to provider)
  MatrixEntry.base_url     (optional)
  MatrixEntry.num_ctx      (optional)

Matrix is opt-in (CONTEXT.md): `matrix=False` (default) iterates ONLY the first
entry (the current backend). `matrix=True` iterates the full matrix.

run_bench surface accepts test-driven kwargs (scenario name, corpus_path, results_path,
models, llm_client_factory) so the e2e + cli tests can drive it without a BenchConfig.
"""
from __future__ import annotations

import json as _json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from localharness.bench.aggregator import metrics_summary, should_stop
from localharness.bench.config import (
    BenchConfig,
    MatrixEntry,
    SamplingConfig,
    load_bench_config,
)
from localharness.bench.report import _sanitize_for_json, write_summary_json, write_summary_md
from localharness.bench.runner import accumulate_runs
from localharness.bench.schema import ScenarioSpec, load_scenario
from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Scenario discovery
# -------------------------------------------------------------------------

def _discover_scenarios(corpus_path: Path) -> list[Path]:
    """Find all *.yaml files under corpus_path RECURSIVELY (covers train/ and holdout/ subdirs)."""
    return sorted(p for p in Path(corpus_path).rglob("*.yaml") if p.is_file())


def _load_scenarios_from_paths(scenario_paths: list[Path]) -> list[ScenarioSpec]:
    out: list[ScenarioSpec] = []
    for sp in scenario_paths:
        try:
            out.append(load_scenario(sp))
        except Exception:
            log.exception("scenario_load_failed path=%s", sp)
            continue
    return out


def _filter_scenarios_by_name(scenarios: list[ScenarioSpec], name: str) -> list[ScenarioSpec]:
    return [s for s in scenarios if s.name == name]


def _filter_scenarios_by_slice(scenarios: list[ScenarioSpec], slice_: str) -> list[ScenarioSpec]:
    """Return scenarios whose .slice matches. slice_='all' passes everything through."""
    if slice_ == "all":
        return scenarios
    return [s for s in scenarios if s.slice == slice_]


# -------------------------------------------------------------------------
# LLM client factory builder (real-provider path)
# -------------------------------------------------------------------------

# Default OpenAI-compatible base URLs per known provider type. Any provider that
# exposes an OpenAI-compatible /v1 endpoint works; unknown providers must supply
# base_url on the matrix entry. This keeps the bench model-agnostic — swapping in
# a different model family is a config change, not a code change.
_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "llamacpp": "http://127.0.0.1:8080/v1",
}


def _build_bench_client(entry: MatrixEntry) -> Any:
    """Build an OpenAI-compatible LLMClient for a matrix entry (any provider).

    tool_call_mode is left at the LLMConfig default and is meant to be overwritten
    by a detect_capabilities() probe (see _run_one_model) so each model gets its
    natural mode (native vs xml) rather than a hardcoded guess.
    """
    from localharness.provider.client import LLMClient, LLMConfig

    base_url = entry.base_url or _DEFAULT_BASE_URLS.get(entry.provider)
    if not base_url:
        raise ValueError(
            f"Matrix entry {entry.name!r} (provider={entry.provider!r}) has no base_url "
            f"and no default is known. Add base_url to the matrix entry."
        )
    cfg = LLMConfig(
        base_url=base_url,
        model=entry.model_id,
        api_key="none",
        # #10: inherit the 600s LLMConfig default (was a hardcoded 300s that timed out slow
        # single-stream decode — the exact bench symptom in the issue).
        # #13: fall back to the canonical served-window constant, not a bare 128_000 literal.
        context_window=entry.num_ctx or DEFAULT_MAX_CONTEXT_TOKENS,
        is_local=True,
    )
    return LLMClient(cfg)


def build_llm_client_factory(entry: MatrixEntry) -> Callable[[ScenarioSpec], Any]:
    """Return a per-scenario factory building a client for this matrix entry.

    Provider-agnostic: any OpenAI-compatible endpoint works (vllm, ollama, …).
    This path does NOT probe capabilities — _run_one_model builds a probed, reused
    client for the default (non-injected) path. Kept for direct callers/tests.
    """
    def _factory(_scen: ScenarioSpec) -> Any:
        return _build_bench_client(entry)

    return _factory


# -------------------------------------------------------------------------
# Per-model execution (one matrix entry x all scenarios)
# -------------------------------------------------------------------------

async def _run_one_model(
    entry: MatrixEntry,
    scenarios: list[ScenarioSpec],
    results_path: Path,
    sampling: SamplingConfig,
    llm_client_factory: Optional[Callable[[ScenarioSpec], Any]] = None,
    min_runs_override: Optional[int] = None,
    max_runs_override: Optional[int] = None,
) -> int:
    """Run all scenarios for one matrix entry. Returns count of total runs executed."""
    if llm_client_factory is not None:
        factory = llm_client_factory
    else:
        # Build one client per matrix entry and probe it so tool_call_mode is
        # auto-detected (model-agnostic) instead of hardcoded. Reused across all
        # scenarios/runs for this entry — detect_capabilities() never raises.
        probed_client = _build_bench_client(entry)
        cap = await probed_client.detect_capabilities()
        log.info(
            "bench_probe model=%s mode=%s ctx=%d err=%s",
            entry.name, cap.tool_call_mode, cap.context_window, cap.probe_error,
        )

        def factory(_scen: ScenarioSpec) -> Any:
            return probed_client

    per_scenario: dict[str, dict[str, Any]] = {}
    runs_executed = 0

    for scen in scenarios:
        min_r = min_runs_override if min_runs_override is not None else (scen.min_runs or sampling.min_runs)
        max_r = max_runs_override if max_runs_override is not None else (scen.max_runs or sampling.max_runs)

        try:
            samples, stop_reason = await accumulate_runs(
                scen,
                entry.name,
                results_path,
                factory,
                min_runs_override=min_r,
                max_runs_override=max_r,
            )
        except Exception:
            log.exception("scenario_failed model=%s scenario=%s", entry.name, scen.name)
            continue

        if not samples:
            continue

        summary = metrics_summary(samples)
        per_scenario[scen.name] = {
            "summary": summary,
            "stop_reason": stop_reason,
            "n_runs": len(samples),
        }
        runs_executed += len(samples)

        scen_summary_path = results_path / entry.name / scen.name / "summary.json"
        write_summary_json(
            scen_summary_path,
            summary,
            scenario_name=scen.name,
            model=entry.name,
            stop_reason=stop_reason,
            n_runs=len(samples),
        )

    if per_scenario:
        model_summary_md = results_path / entry.name / "summary.md"
        write_summary_md(model_summary_md, per_scenario, entry.name)
        # Also write a model-level roll-up summary.json so bench compare can find it.
        model_summary_json = results_path / entry.name / "summary.json"
        rollup = {
            "model": entry.name,
            "scenarios": {
                scen_name: {
                    **info["summary"],
                    "n_runs": info["n_runs"],
                    "stop_reason": info["stop_reason"],
                    "stable": not str(info["stop_reason"]).startswith("max_runs_hit"),
                }
                for scen_name, info in per_scenario.items()
            },
        }
        model_summary_json.parent.mkdir(parents=True, exist_ok=True)
        model_summary_json.write_text(
            _json.dumps(_sanitize_for_json(rollup), indent=2, sort_keys=True, allow_nan=False)
        )

    return runs_executed


# -------------------------------------------------------------------------
# Default entry synthesis (no BenchConfig case)
# -------------------------------------------------------------------------

def _synthesize_default_entry() -> MatrixEntry:
    """Synthesize a MatrixEntry resolved from the running HarnessConfig (cfg.provider).

    Reads provider/model/base_url from the loaded HarnessConfig so the gate's bench
    client targets the REAL configured backend — no hardcoded ollama/bench-default.
    Raises RuntimeError if no config is available (CLAUDE.md: fail explicitly).
    """
    try:
        from localharness.config.loader import ConfigLoader
        cfg = ConfigLoader().load_harness()
    except Exception as exc:
        raise RuntimeError(
            "_synthesize_default_entry: no HarnessConfig available — "
            "inject a llm_client_factory or ensure ~/.localharness/config.yaml exists"
        ) from exc
    return MatrixEntry(
        name=cfg.provider.default_model,
        provider=cfg.provider.provider_type,
        model_id=cfg.provider.default_model,
        base_url=cfg.provider.base_url,
    )


def _resolve_matrix(
    config: Optional[BenchConfig],
    matrix: bool,
    models: Optional[list[str]],
) -> list[MatrixEntry]:
    """Resolve the matrix entries to run (locked opt-in semantics)."""
    if config is None or not config.matrix:
        return [_synthesize_default_entry()]
    entries = list(config.matrix)
    if models:
        entries = [m for m in entries if m.name in models]
        return entries
    if not matrix:
        # Default single-backend mode: entries[0] is SUPPOSED to be the current backend, but a
        # stale/hand-edited bench.yaml can drift from it (e.g. matrix[0] pins vllm while the
        # active config now points at ollama). Compare against the ACTUAL active provider — the
        # same HarnessConfig source _synthesize_default_entry() reads — and synthesize from it
        # instead of silently benching a model/provider the harness isn't even running.
        try:
            from localharness.config.loader import ConfigLoader
            active_provider = ConfigLoader().load_harness().provider.provider_type
        except Exception:
            active_provider = None  # active config unavailable — keep current behavior below
        if active_provider is not None and entries[0].provider != active_provider:
            log.info(
                "matrix_entry_provider_mismatch configured=%s active=%s — synthesizing from active config",
                entries[0].provider, active_provider,
            )
            return [_synthesize_default_entry()]
        return entries[:1]
    return entries


# -------------------------------------------------------------------------
# run_bench — async test-driven entry point (used by CLI + e2e tests)
# -------------------------------------------------------------------------

async def run_bench(
    scenario: Optional[str] = None,
    matrix: bool = False,
    models: Optional[list[str]] = None,
    threshold_overrides: Optional[list[str]] = None,
    corpus_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
    json_output: bool = False,
    llm_client_factory: Optional[Callable[[ScenarioSpec], Any]] = None,
    config_path: Optional[Path] = None,
    min_runs_override: Optional[int] = None,
    max_runs_override: Optional[int] = None,
    slice: str = "train",
) -> int:
    """Top-level bench entry point.

    Returns an exit code:
      0 = success (at least one scenario ran)
      2 = infra failure (no scenarios, empty corpus, no matrix entries match)

    Matrix is OPT-IN per CONTEXT.md: matrix=False (default) iterates ONLY the first
    configured backend. matrix=True iterates the full configured matrix. `models`
    is a name-subset filter (overrides the opt-in default).

    Either (corpus_path, results_path) OR (config_path) must be sufficient to resolve
    the corpus + results destination. If both are missing, attempts ./bench/bench.yaml.
    """
    # Resolve BenchConfig (best-effort)
    config: Optional[BenchConfig] = None
    if config_path is not None:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            try:
                config = load_bench_config(cfg_path)
            except Exception:
                log.exception("bench_config_load_failed path=%s", cfg_path)
                config = None
    elif corpus_path is None and results_path is None:
        default_cfg = Path("./bench/bench.yaml")
        if default_cfg.exists():
            try:
                config = load_bench_config(default_cfg)
            except Exception:
                log.exception("bench_config_load_failed path=%s", default_cfg)
                config = None

    # Resolve corpus + results paths (CLI args win over BenchConfig)
    resolved_corpus = Path(corpus_path) if corpus_path is not None else (config.corpus_path if config else None)
    resolved_results = Path(results_path) if results_path is not None else (config.results_path if config else None)

    if resolved_corpus is None or resolved_results is None:
        log.error("bench_paths_unresolved corpus=%s results=%s", resolved_corpus, resolved_results)
        if json_output:
            sys.stdout.write(_json.dumps({"status": "error", "reason": "corpus_path or results_path missing", "exit_code": 2}) + "\n")
        return 2

    # Discover + load scenarios
    scenario_paths = _discover_scenarios(resolved_corpus)
    if not scenario_paths:
        log.warning("no_scenarios corpus_path=%s", resolved_corpus)
        if json_output:
            sys.stdout.write(_json.dumps({"status": "error", "reason": "empty_corpus", "exit_code": 2, "total_runs": 0}) + "\n")
        return 2

    scenarios = _load_scenarios_from_paths(scenario_paths)
    if not scenarios:
        log.error("all_scenarios_failed_to_load")
        if json_output:
            sys.stdout.write(_json.dumps({"status": "error", "reason": "all_scenarios_failed_to_load", "exit_code": 2}) + "\n")
        return 2

    if scenario is not None:
        scenarios = _filter_scenarios_by_name(scenarios, scenario)
        if not scenarios:
            log.error("scenario_not_found name=%s", scenario)
            if json_output:
                sys.stdout.write(_json.dumps({"status": "error", "reason": f"scenario_not_found:{scenario}", "exit_code": 2}) + "\n")
            return 2
    else:
        # --scenario overrides --slice (single-fixture by-name invocation bypasses slice filter)
        scenarios = _filter_scenarios_by_slice(scenarios, slice)
        if not scenarios:
            log.warning("no_scenarios_for_slice slice=%s", slice)
            if json_output:
                sys.stdout.write(_json.dumps({
                    "status": "error",
                    "reason": f"no_scenarios_for_slice:{slice}",
                    "exit_code": 2,
                    "total_runs": 0,
                }) + "\n")
            return 2

    # Resolve matrix
    matrix_entries = _resolve_matrix(config, matrix=matrix, models=models)
    if not matrix_entries:
        log.error("no_matrix_entries_match models=%s", models)
        if json_output:
            sys.stdout.write(_json.dumps({"status": "error", "reason": "no_matrix_entries_match", "exit_code": 2}) + "\n")
        return 2

    sampling = config.sampling if config else SamplingConfig()

    total_runs = 0
    for entry in matrix_entries:
        log.info(
            "matrix_entry_start name=%s provider=%s model_id=%s",
            entry.name, entry.provider, entry.model_id,
        )
        try:
            runs = await _run_one_model(
                entry=entry,
                scenarios=scenarios,
                results_path=resolved_results,
                sampling=sampling,
                llm_client_factory=llm_client_factory,
                min_runs_override=min_runs_override,
                max_runs_override=max_runs_override,
            )
            total_runs += runs
        except Exception:
            log.exception("matrix_entry_failed name=%s", entry.name)
            continue

    if total_runs == 0:
        if json_output:
            sys.stdout.write(_json.dumps({"status": "error", "reason": "no_runs_executed", "exit_code": 2, "total_runs": 0}) + "\n")
        return 2

    if json_output:
        sys.stdout.write(_json.dumps({"status": "ok", "exit_code": 0, "total_runs": total_runs}) + "\n")
    return 0
