"""`localharness propose` command (PROP-01 / SC1).

Thin typer wrapper over the 16-02 ``autoresearch.proposer.propose()`` pipeline.
Mirrors the ``autoresearch_cmd`` / ``components_cmd`` idioms: default human output
(red/green diff + rationale) + ``--json``, ``_err`` to stderr with ``typer.Exit(2)``,
the worker-thread ``_run`` async bridge, and stdlib-``difflib`` diff rendering (no new
dependency). No new domain logic — the seal, evidence load, and parse all live in the
pipeline; this command resolves bench paths and surfaces ``ProposerError`` as exit 2.

return-only by default: the opt-in ``--archive`` flag writes a single ``in_flight``
``ArchiveStore`` row (uuid4 id, null scores) so the Phase 18 orchestrator has a
persistence seam. Phase 17 is the primary archive writer that fills scores.
"""
from __future__ import annotations

import asyncio
import difflib
import json as _json
import os
from pathlib import Path

import typer
from rich.console import Console

from localharness.autoresearch.proposer import ProposerError
from localharness.autoresearch.proposer import propose as propose_pipeline

console = Console()
err_console = Console(stderr=True)


# ------------------------------------------------------------------ #
# Helpers (mirror autoresearch_cmd.py — per-command duplication, no shared module)
# ------------------------------------------------------------------ #


def _archive_db_path() -> Path:
    """Resolve .localharness/archive.db, honoring LOCALHARNESS_HOME (mirrors _build_loader)."""
    home = os.environ.get("LOCALHARNESS_HOME")
    base = Path(home) if home else Path.cwd() / ".localharness"
    return base / "archive.db"


def _err(json_output: bool, message: str, exit_code: int = 2) -> None:
    """Print error to stderr (or JSON) and exit."""
    if json_output:
        typer.echo(_json.dumps({"error": message}), err=True)
    else:
        err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=exit_code)


def _run(coro):
    """Async-from-sync bridge.

    In production the CLI has no running loop, so ``asyncio.run`` succeeds directly.
    Under pytest-asyncio the calling thread already holds a live loop, so run the
    coroutine on a fresh loop in a worker thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no running loop (the normal CLI path)

    import threading

    result: dict = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagate to caller thread
            result["error"] = exc

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _to_lines(value) -> list[str]:
    """Render a diff side to text lines: strings split on newlines; else YAML/repr."""
    if value is None:
        return ["null"]
    if isinstance(value, str):
        return value.split("\n")
    try:
        import yaml

        return yaml.safe_dump(value, default_flow_style=False, sort_keys=False).splitlines()
    except Exception:
        return repr(value).splitlines()


def _render_diff(diff_json: str) -> None:
    """Inline red/green diff via stdlib difflib.ndiff + rich (no new dependency)."""
    try:
        data = _json.loads(diff_json)
    except (ValueError, TypeError):
        console.print(diff_json)
        return
    before_lines = _to_lines(data.get("before"))
    after_lines = _to_lines(data.get("after"))
    for line in difflib.ndiff(before_lines, after_lines):
        if line.startswith("- "):
            console.print(line, style="red")
        elif line.startswith("+ "):
            console.print(line, style="green")
        elif line.startswith("? "):
            continue
        else:
            console.print(line)


def _resolve_bench_paths() -> tuple[Path, Path]:
    """corpus_path + results_path from ./bench/bench.yaml (honors cwd)."""
    from localharness.bench.config import load_bench_config

    cfg_path = Path("bench/bench.yaml")
    if not cfg_path.exists():
        raise ProposerError(
            "bench/bench.yaml not found — cannot resolve corpus/results paths"
        )
    bc = load_bench_config(cfg_path)
    return Path(bc.corpus_path), Path(bc.results_path)


def _diff_blob(proposal, cfg) -> str:
    """Single-encoded archive diff blob incl. rationale + kind (GAP-1)."""
    import json

    from localharness.autoresearch.experiment import _provenance_agent_cfg
    from localharness.config.overlay import _resolve_user_overlay_path, load_overlay
    from localharness.registry.catalogue import build_catalogue

    try:
        _user_ov = load_overlay(_resolve_user_overlay_path())
        type_name = build_catalogue(
            cfg,
            agent_cfg=_provenance_agent_cfg(),
            overlays={"user": _user_ov},
        )[proposal.component].type_name
    except Exception:
        type_name = ""
    kind = "hyperparameter" if type_name in ("int", "float") else "prompt"
    return json.dumps({
        "before": proposal.before,
        "after": proposal.after,
        "rationale": proposal.rationale,
        "kind": kind,
    })


async def _write_in_flight(proposal, cfg) -> None:
    """Persist a single in_flight archive row (uuid4 id, null scores) under --archive."""
    import time
    import uuid

    from localharness.autoresearch.archive import ArchiveEntry, ArchiveStore

    store = ArchiveStore(_archive_db_path())
    await store.open()
    try:
        await store.write(
            ArchiveEntry(
                id=str(uuid.uuid4()),
                parent_id=None,
                component=proposal.component,
                diff=_diff_blob(proposal, cfg),
                train_score=None,
                train_scores_per_fixture=None,
                holdout_score=None,
                p_value=None,
                cost=None,
                ts=int(time.time()),
                approved_by=None,
                status="in_flight",
            )
        )
    finally:
        await store.close()


# ------------------------------------------------------------------ #
# propose
# ------------------------------------------------------------------ #


def propose(
    component: str = typer.Option(
        ..., "--component", help="Dot-path of the component to mutate (e.g. agent.role)"
    ),
    traces: list[str] = typer.Option(
        ..., "--traces", help="TRAIN run_ids {model}/{scenario}/{timestamp} (repeatable)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON {component, diff, rationale}"
    ),
    archive: bool = typer.Option(
        False, "--archive", help="Persist an in_flight archive row (default: return-only)"
    ),
) -> None:
    """Generate ONE typed mutation {diff, rationale} for ONE component from failed TRAIN traces.

    Seal refusals (holdout/unknown/no-evidence/malformed/off-target) surface as exit 2.
    """
    try:
        from localharness.cli.components_cmd import _build_loader

        from localharness.autoresearch.archive import ArchiveStore

        cfg = _build_loader().load_harness()
        corpus_path, results_path = _resolve_bench_paths()
        # Reflect across the per-fixture Pareto front (MODP-03): pass the existing
        # archive so a real run cites which mutation already wins each train fixture.
        # A missing/empty archive ⇒ empty front ⇒ pareto_evidence="" (no new flag, no crash).
        store = ArchiveStore(_archive_db_path())
        proposal = _run(
            propose_pipeline(
                component,
                traces,
                cfg=cfg,
                corpus_path=corpus_path,
                results_path=results_path,
                store=store,
            )
        )
    except ProposerError as exc:
        _err(json_output, str(exc), exit_code=2)
        return
    except Exception as exc:
        _err(json_output, f"propose failed: {exc}", exit_code=2)
        return

    if archive:
        try:
            _run(_write_in_flight(proposal, cfg))
        except Exception as exc:
            _err(json_output, f"archive write failed: {exc}", exit_code=2)
            return

    if json_output:
        typer.echo(
            _json.dumps(
                {
                    "component": proposal.component,
                    "diff": {"before": proposal.before, "after": proposal.after},
                    "rationale": proposal.rationale,
                }
            )
        )
    else:
        _render_diff(proposal.diff)
        console.print(f"[bold]rationale[/bold]\n{proposal.rationale}")
