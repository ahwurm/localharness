"""`localharness autoresearch report` (+ a thin `sentinel` alias) — the async human-review surface.

Phase 19 — REP-01..04. This is the ONLY human touchpoint in the Phase-18 fire-and-forget loop, so
it must be actionable at a glance. The overview renders the signed-off mockup (19-CONTEXT decision 2 +
19-RESEARCH § Terminal Rendering) as a window-card flow:

    1. Trajectory sparklines (train vs holdout + overfit gap)   — REP-01
    2. Pareto top-mutations table (id/component/train/hold/gap/p/cost/status)  — REP-02
    3. Adopted / Held / Rejected inbox (status enum → review buckets, git-revert one-liner)  — decision 2
    4. Sentinel alerts (overfit gap / near-duplicate collapse / saturation)  — REP-03/04

``report --show <id>`` drills into one mutation: change diff + hypothesis (rationale, or the exact
``hyperparameter (numeric tuning), no mechanism`` label for a numeric tuning) + GAP-2 proof
(only the ship-available stats: p-value + lift + per-fixture movers + holdout verdict; the richer
effect-size/CI block is intentionally NOT rendered — Phase 17 is sealed and the baseline vector
isn't stored) + the full child→root lineage.

Each run writes a durable, diffable markdown snapshot under ``<home>/autoresearch/reports/<ts>.md`` so
the remote/async reviewer has a copy.

~80% of this is wiring existing read APIs into a renderer (19-RESEARCH key insight): it consumes
``run_sentinel``/``sparkline``/``alerts_from_report`` (19-03) + ``pareto_front_*``/``lineage``/``query``
(15) verbatim, and reuses the ``autoresearch_cmd`` helpers (``_archive_db_path``/``_run``/``_err``/
``_render_diff``/``_resolve``/``_repo_root``/``_fmt_float``/``_fmt_ts``). The report is a terminal/markdown
sink with NO proposer import and NO write-back to the archive's train/holdout columns (Pitfall 6 — seal intact).
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import subprocess
from io import StringIO
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from localharness.autoresearch.archive import ArchiveEntry, ArchiveQuery, ArchiveStore
from localharness.autoresearch.sentinel import (
    alerts_from_report,
    run_sentinel,
    sparkline,
)
from localharness.cli.autoresearch_cmd import (
    _archive_db_path,
    _err,
    _fmt_float,
    _render_diff,
    _repo_root,
    _resolve,
    _run,
    autoresearch_app,
    console,
)

# Status enum → inbox grouping (Phases 15 + 18). promoted/in_flight/superseded are loop-internal
# (surfaced in the Pareto table, not the inbox).
_ADOPTED = {"adopted"}
_HELD = {"held"}
_REJECTED = {"train_rejected", "holdout_rejected", "adoption_rejected"}


# ------------------------------------------------------------------ #
# cfg — only cfg.sentinel is consumed; default-construct so report runs
# without a full provider config (the human's read-only knobs).
# ------------------------------------------------------------------ #


def _load_sentinel_cfg(
    overfit_gap: Optional[float],
    dup_similarity: Optional[float],
    dup_k: Optional[int],
    saturation_k: Optional[int],
):
    """Build a cfg object exposing ``.sentinel`` with flag overrides (flag > config-default).

    Only ``cfg.sentinel`` is read by ``run_sentinel``; a default ``SentinelConfig`` always exists
    (it is a ``default_factory`` field on ``HarnessConfig``). We default-construct it so the report
    never needs a provider/config-load just to render — then apply any per-invocation flag overrides
    (RESEARCH § Thresholds: flag > config > field-default).
    """
    from localharness.config.models import SentinelConfig

    sentinel = SentinelConfig()
    if overfit_gap is not None:
        sentinel.overfit_gap_threshold = overfit_gap
    if dup_similarity is not None:
        sentinel.duplicate_similarity = dup_similarity
    if dup_k is not None:
        sentinel.duplicate_consecutive_k = dup_k
    if saturation_k is not None:
        sentinel.saturation_k = saturation_k

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.sentinel = sentinel
    return cfg


def _reports_dir() -> Path:
    """`<home>/autoresearch/reports/` — mirrors `_archive_db_path()` (LOCALHARNESS_HOME-aware)."""
    return _archive_db_path().parent / "autoresearch" / "reports"


def _fmt_gap(e: ArchiveEntry) -> str:
    """train − holdout as a 3dp string (or '-' when either score is absent)."""
    if e.train_score is None or e.holdout_score is None:
        return "-"
    return f"{e.train_score - e.holdout_score:.3f}"


def _revert_oneliner(e: ArchiveEntry) -> str:
    """The `git revert <sha>` one-liner for an adopted mutation (Phase 18 adoptions are git commits).

    Best-effort: find the adoption commit by its conventional ``autoresearch: adopt <component>``
    subject in the main repo; if not recoverable, fall back to the documented template form so the
    reviewer always has the revert shape (the row id + component identify the change).
    """
    try:
        repo = _repo_root()
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "--format=%H",
             "--grep", f"autoresearch: adopt {e.component}", "-n", "1"],
            check=True, capture_output=True, text=True,
        )
        sha = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        if sha:
            return f"git revert {sha}"
    except Exception:
        pass
    return f"git revert <commit for 'autoresearch: adopt {e.component}'>"


# ------------------------------------------------------------------ #
# Overview data assembly (read-only over the archive)
# ------------------------------------------------------------------ #


async def _gather_overview(store: ArchiveStore, cfg) -> dict:
    """Read every section's source data in one async shell (Pattern 1)."""
    # Trajectory (REP-01): the promoted/adopted baseline curve over ts ASC (Open Q 4 — x-axis = ts).
    all_rows = await store.query(ArchiveQuery(limit=10_000))
    traj_rows = sorted(
        [r for r in all_rows if r.status in ("promoted", "adopted")],
        key=lambda r: r.ts,
    )

    # Pareto top-mutations (REP-02): per-fixture GEPA front ∪ the 2D cost×score front.
    front_pf = await store.pareto_front_per_fixture()
    try:
        front_2d = await store.pareto_front_2d(("train_score", "cost"))
    except ValueError:
        front_2d = []
    seen: set[str] = set()
    front: list[ArchiveEntry] = []
    for e in sorted(list(front_pf) + list(front_2d), key=lambda r: (r.train_score or 0.0), reverse=True):
        if e.id not in seen:
            seen.add(e.id)
            front.append(e)

    # Sentinel alerts (REP-03/04): the on-demand pass over the same archive.
    sreport = await run_sentinel(store, cfg)

    return {"all_rows": all_rows, "traj_rows": traj_rows, "front": front, "sreport": sreport}


def _render_overview(data: dict, *, to: Console) -> None:
    """Render the four window-cards to a (wide, non-tty-safe) Console."""
    traj_rows = data["traj_rows"]
    front = data["front"]
    all_rows = data["all_rows"]
    sreport = data["sreport"]

    # --- 1. Trajectory sparklines (train vs holdout + overfit gap) ---
    to.print("[bold]Trajectory[/bold] (promoted/adopted baseline over time)")
    if traj_rows:
        train = [r.train_score for r in traj_rows]
        hold = [r.holdout_score for r in traj_rows]
        gap = [
            (r.train_score - r.holdout_score)
            if (r.train_score is not None and r.holdout_score is not None)
            else None
            for r in traj_rows
        ]
        to.print(f"  train   [cyan]{sparkline(train)}[/cyan]")
        to.print(f"  holdout [magenta]{sparkline(hold)}[/magenta]")
        to.print(f"  gap     [red]{sparkline(gap)}[/red]")
    else:
        to.print("  [dim]no promoted/adopted mutations yet[/dim]")

    # --- 2. Pareto top-mutations table ---
    table = Table(title="Top mutations (Pareto front)", show_lines=False, pad_edge=False, expand=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("component", no_wrap=True)
    table.add_column("train")
    table.add_column("hold")
    table.add_column("gap")
    table.add_column("p")
    table.add_column("cost")
    table.add_column("status", style="green")
    for e in front:
        table.add_row(
            e.id[:8], e.component,
            _fmt_float(e.train_score), _fmt_float(e.holdout_score), _fmt_gap(e),
            _fmt_float(e.p_value), _fmt_float(e.cost), e.status,
        )
    to.print(table)

    # --- 3. Adopted / Held / Rejected inbox ---
    adopted = [e for e in all_rows if e.status in _ADOPTED]
    held = [e for e in all_rows if e.status in _HELD]
    rejected = [e for e in all_rows if e.status in _REJECTED]

    to.print("[bold green]Adopted[/bold green] (live changes — revert with the one-liner)")
    if adopted:
        for e in adopted:
            to.print(f"  {e.id[:8]} {e.component}  train={_fmt_float(e.train_score)}")
            to.print(f"      {_revert_oneliner(e)}")
    else:
        to.print("  [dim]none[/dim]")

    to.print("[bold yellow]Held[/bold yellow] (needs your async call)")
    if held:
        for e in held:
            to.print(
                f"  {e.id[:8]} {e.component}  train={_fmt_float(e.train_score)} "
                f"holdout={_fmt_float(e.holdout_score)} p={_fmt_float(e.p_value)}"
            )
    else:
        to.print("  [dim]none[/dim]")

    to.print("[bold red]Rejected[/bold red] (kept as parent material, not re-offered)")
    if rejected:
        for e in rejected:
            to.print(f"  {e.id[:8]} {e.component}  {e.status}  train={_fmt_float(e.train_score)}")
    else:
        to.print("  [dim]none[/dim]")

    # --- 4. Sentinel alerts ---
    to.print("[bold]Sentinel alerts[/bold]")
    alerts = alerts_from_report(sreport)
    if alerts:
        for a in alerts:
            to.print(f"  [red]![/red] {a.kind}: {a.detail}")
    else:
        to.print("  No sentinel alerts.")


def _overview_markdown(data: dict) -> str:
    """The SAME overview content as GitHub-flavored markdown (the durable snapshot)."""
    traj_rows = data["traj_rows"]
    front = data["front"]
    all_rows = data["all_rows"]
    sreport = data["sreport"]
    out: list[str] = []
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    out.append(f"# Autoresearch Report — {now}\n")

    out.append("## Trajectory\n")
    if traj_rows:
        train = [r.train_score for r in traj_rows]
        hold = [r.holdout_score for r in traj_rows]
        gap = [
            (r.train_score - r.holdout_score)
            if (r.train_score is not None and r.holdout_score is not None)
            else None
            for r in traj_rows
        ]
        out.append(f"- train   `{sparkline(train)}`")
        out.append(f"- holdout `{sparkline(hold)}`")
        out.append(f"- gap     `{sparkline(gap)}`\n")
    else:
        out.append("_no promoted/adopted mutations yet_\n")

    out.append("## Top mutations (Pareto front)\n")
    out.append("| id | component | train | hold | gap | p | cost | status |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for e in front:
        out.append(
            f"| {e.id[:8]} | {e.component} | {_fmt_float(e.train_score)} | "
            f"{_fmt_float(e.holdout_score)} | {_fmt_gap(e)} | {_fmt_float(e.p_value)} | "
            f"{_fmt_float(e.cost)} | {e.status} |"
        )
    out.append("")

    adopted = [e for e in all_rows if e.status in _ADOPTED]
    held = [e for e in all_rows if e.status in _HELD]
    rejected = [e for e in all_rows if e.status in _REJECTED]

    out.append("## Adopted\n")
    if adopted:
        for e in adopted:
            out.append(f"- `{e.id[:8]}` {e.component} — `{_revert_oneliner(e)}`")
    else:
        out.append("_none_")
    out.append("")

    out.append("## Held\n")
    if held:
        for e in held:
            out.append(
                f"- `{e.id[:8]}` {e.component} — train={_fmt_float(e.train_score)} "
                f"holdout={_fmt_float(e.holdout_score)} p={_fmt_float(e.p_value)}"
            )
    else:
        out.append("_none_")
    out.append("")

    out.append("## Rejected\n")
    if rejected:
        for e in rejected:
            out.append(f"- `{e.id[:8]}` {e.component} — {e.status}")
    else:
        out.append("_none_")
    out.append("")

    out.append("## Sentinel alerts\n")
    alerts = alerts_from_report(sreport)
    if alerts:
        for a in alerts:
            out.append(f"- **{a.kind}**: {a.detail}")
    else:
        out.append("_No sentinel alerts._")
    out.append("")
    return "\n".join(out)


def _write_snapshot(markdown: str, *, suffix: str = "") -> Path:
    """Write a timestamped markdown snapshot under `<home>/autoresearch/reports/` (Open Q 5).

    Append-only timestamped files are crash-safe — a plain write needs no atomic dance.
    """
    reports = _reports_dir()
    reports.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{stamp}{('-' + suffix) if suffix else ''}.md"
    path = reports / name
    path.write_text(markdown, encoding="utf-8")
    return path


# ------------------------------------------------------------------ #
# Drill-down (--show <id>): change + hypothesis + GAP-2 proof + lineage
# ------------------------------------------------------------------ #


async def _gather_drilldown(store: ArchiveStore, raw_id: str):
    """Resolve the id and fetch its lineage (read-only)."""
    entry, matches = await _resolve(store, raw_id)
    if entry is None:
        return (None, matches, [])
    lineage = await store.lineage(entry.id)
    return (entry, None, lineage)


def _render_drilldown(entry: ArchiveEntry, lineage: list[ArchiveEntry], *, to: Console) -> None:
    """Change panel + hypothesis panel + GAP-2 proof panel + child→root lineage."""
    try:
        decoded = entry.diff_decoded
    except (ValueError, TypeError):
        decoded = {}
    kind = decoded.get("kind")

    to.print(f"[bold]{entry.id}[/bold]  ({entry.component})")

    # 1. Change panel — before→after via the verbatim difflib red/green renderer.
    to.print("[bold]change[/bold]")
    _render_diff(entry.diff)

    # 2. Hypothesis panel.
    to.print("[bold]hypothesis[/bold]")
    if kind == "hyperparameter":
        # Hyperparameter mutations get NO manufactured mechanism (locked decision 2 / 5).
        to.print("  hyperparameter (numeric tuning), no mechanism")
        to.print(f"  {decoded.get('before')!r} -> {decoded.get('after')!r}")
    else:
        rationale = decoded.get("rationale") or "(no rationale recorded)"
        to.print(f"  {rationale}")

    # 3. Proof panel (GAP-2 option-a: p + lift + movers + holdout verdict ONLY; Phase 17 sealed).
    to.print("[bold]proof[/bold]")
    to.print(f"  p-value:        {_fmt_float(entry.p_value)}")
    to.print(f"  lift (train mean): {_fmt_float(entry.train_score)}")
    tspf = entry.train_scores_per_fixture
    if tspf:
        ordered = sorted(tspf.items(), key=lambda kv: kv[1], reverse=True)
        best = ordered[:3]
        worst = list(reversed(ordered[-3:]))
        to.print("  movers (top):    " + ", ".join(f"{k}={v:.3f}" for k, v in best))
        to.print("  movers (bottom): " + ", ".join(f"{k}={v:.3f}" for k, v in worst))
    if entry.status in ("promoted", "adopted"):
        to.print("  holdout: PASS (non-regression)")
    elif entry.status == "holdout_rejected":
        to.print("  holdout: FAIL")
    else:
        to.print(f"  holdout: {_fmt_float(entry.holdout_score)}")

    # 4. Lineage panel — child→root (git-log-oneline order; lineage() returns newest-first).
    to.print("[bold]lineage[/bold]")
    for e in lineage:
        to.print(f"  {e.id[:8]} {e.component} {e.status} {e.ts}")


def _drilldown_markdown(entry: ArchiveEntry, lineage: list[ArchiveEntry]) -> str:
    """The drill-down as markdown (a durable copy for the async/remote reviewer)."""
    try:
        decoded = entry.diff_decoded
    except (ValueError, TypeError):
        decoded = {}
    kind = decoded.get("kind")
    out: list[str] = [f"# Mutation {entry.id} ({entry.component})\n", "## change\n"]
    out.append(f"```\nbefore: {decoded.get('before')!r}\nafter:  {decoded.get('after')!r}\n```\n")
    out.append("## hypothesis\n")
    if kind == "hyperparameter":
        out.append("hyperparameter (numeric tuning), no mechanism\n")
        out.append(f"`{decoded.get('before')!r} -> {decoded.get('after')!r}`\n")
    else:
        out.append((decoded.get("rationale") or "(no rationale recorded)") + "\n")
    out.append("## proof\n")
    out.append(f"- p-value: {_fmt_float(entry.p_value)}")
    out.append(f"- lift (train mean): {_fmt_float(entry.train_score)}")
    tspf = entry.train_scores_per_fixture
    if tspf:
        ordered = sorted(tspf.items(), key=lambda kv: kv[1], reverse=True)
        out.append("- movers (top): " + ", ".join(f"{k}={v:.3f}" for k, v in ordered[:3]))
    if entry.status in ("promoted", "adopted"):
        out.append("- holdout: PASS (non-regression)")
    elif entry.status == "holdout_rejected":
        out.append("- holdout: FAIL")
    else:
        out.append(f"- holdout: {_fmt_float(entry.holdout_score)}")
    out.append("\n## lineage\n")
    for e in lineage:
        out.append(f"- `{e.id[:8]}` {e.component} {e.status} {e.ts}")
    out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ #
# `autoresearch report`
# ------------------------------------------------------------------ #


def report(
    show: Optional[str] = typer.Option(
        None, "--show", help="Drill into one mutation by id/prefix (change + hypothesis + proof + lineage)"
    ),
    json_output: bool = typer.Option(False, "--json", help="(reserved) emit structured output"),
    overfit_gap: Optional[float] = typer.Option(None, "--overfit-gap", help="Override sentinel.overfit_gap_threshold"),
    dup_similarity: Optional[float] = typer.Option(None, "--dup-similarity", help="Override sentinel.duplicate_similarity"),
    dup_k: Optional[int] = typer.Option(None, "--dup-k", help="Override sentinel.duplicate_consecutive_k"),
    saturation_k: Optional[int] = typer.Option(None, "--saturation-k", help="Override sentinel.saturation_k"),
) -> None:
    """Render the async human-review surface: trajectory + Pareto table + inbox + sentinel alerts.

    ``--show <id>`` drills into one mutation instead. A timestamped markdown snapshot is written
    under ``<home>/autoresearch/reports/`` every run. An empty/absent archive is a normal survey
    result (exit 0, mirrors ``archive list``).
    """
    cfg = _load_sentinel_cfg(overfit_gap, dup_similarity, dup_k, saturation_k)

    # ---- drill-down branch (Task 2) ----
    if show is not None:
        async def _go_show():
            db_path = _archive_db_path()
            if not db_path.exists():
                return (None, [], [])
            store = ArchiveStore(db_path)
            await store.open()
            try:
                return await _gather_drilldown(store, show)
            finally:
                await store.close()

        entry, matches, lineage = _run(_go_show())
        if entry is None:
            if matches:
                err = Console(stderr=True)
                err.print(f"[bold red]Error:[/bold red] ambiguous prefix {show!r} matches {len(matches)}:")
                for m in matches:
                    err.print(f"  {m.id} {m.component} {m.status}")
                raise typer.Exit(code=2)
            _err(json_output, f"no mutation matches id {show!r}", exit_code=2)
            return

        # Render to the live console AND capture for the durable snapshot.
        _render_drilldown(entry, lineage, to=console)
        _write_snapshot(_drilldown_markdown(entry, lineage), suffix=entry.id[:8])
        raise typer.Exit(code=0)

    # ---- overview branch (Task 1) ----
    async def _go():
        db_path = _archive_db_path()
        if not db_path.exists():
            return None
        store = ArchiveStore(db_path)
        await store.open()
        try:
            return await _gather_overview(store, cfg)
        finally:
            await store.close()

    data = _run(_go())
    if data is None:
        # Empty/absent archive — still a valid (empty) overview, exit 0.
        console.print("[bold]Autoresearch Report[/bold]")
        console.print("  [dim]no archive yet — nothing to report[/dim]")
        _write_snapshot("# Autoresearch Report\n\n_No archive yet._\n")
        raise typer.Exit(code=0)

    # Render at a fixed wide width so non-tty (CliRunner / pipes) don't crop columns (Pitfall 1).
    wide = Console(width=200)
    _render_overview(data, to=wide)
    _write_snapshot(_overview_markdown(data))
    raise typer.Exit(code=0)


# ------------------------------------------------------------------ #
# `autoresearch sentinel` — the thin standalone alias (Task 3)
# ------------------------------------------------------------------ #


def sentinel(
    json_output: bool = typer.Option(False, "--json", help="(reserved) emit structured output"),
    overfit_gap: Optional[float] = typer.Option(None, "--overfit-gap", help="Override sentinel.overfit_gap_threshold"),
    dup_similarity: Optional[float] = typer.Option(None, "--dup-similarity", help="Override sentinel.duplicate_similarity"),
    dup_k: Optional[int] = typer.Option(None, "--dup-k", help="Override sentinel.duplicate_consecutive_k"),
    saturation_k: Optional[int] = typer.Option(None, "--saturation-k", help="Override sentinel.saturation_k"),
) -> None:
    """Run ONLY the eval-sentinel pass (gaps + duplicates + the rotation suggestion); cron-friendly.

    Same pass as the inline section of ``report``, standalone. Writes a markdown snapshot too.
    The rotation suggestion NEVER names a sealed eval-slice fixture (the seal is preserved by
    ``saturated_fixtures``). An empty/absent archive is exit 0.
    """
    cfg = _load_sentinel_cfg(overfit_gap, dup_similarity, dup_k, saturation_k)

    async def _go():
        db_path = _archive_db_path()
        if not db_path.exists():
            return None
        store = ArchiveStore(db_path)
        await store.open()
        try:
            return await run_sentinel(store, cfg)
        finally:
            await store.close()

    sreport = _run(_go())

    buf = StringIO()
    out = Console(width=200, file=buf)
    out.print("[bold]Sentinel[/bold]")
    if sreport is None:
        out.print("  [dim]no archive yet — nothing to analyze[/dim]")
    else:
        alerts = alerts_from_report(sreport)
        out.print("[bold]alerts[/bold]")
        if alerts:
            for a in alerts:
                out.print(f"  [red]![/red] {a.kind}: {a.detail}")
        else:
            out.print("  No sentinel alerts.")
        rot = sreport.rotation
        out.print("[bold]rotation suggestion[/bold]")
        if rot.fixtures:
            out.print(f"  retire (saturated TRAIN): {', '.join(rot.fixtures)}")
            out.print(f"  {rot.detail}")
        else:
            out.print("  No saturated fixtures — corpus still discriminating.")

    rendered = buf.getvalue()
    console.print(rendered, end="")

    # Durable snapshot (markdown mirror of the standalone pass).
    md = ["# Sentinel\n"]
    if sreport is None:
        md.append("_No archive yet._\n")
    else:
        md.append("## alerts\n")
        alerts = alerts_from_report(sreport)
        if alerts:
            for a in alerts:
                md.append(f"- **{a.kind}**: {a.detail}")
        else:
            md.append("_No sentinel alerts._")
        md.append("\n## rotation suggestion\n")
        if sreport.rotation.fixtures:
            md.append(f"- retire (saturated TRAIN): {', '.join(sreport.rotation.fixtures)}")
            md.append(f"- {sreport.rotation.detail}")
        else:
            md.append("_No saturated fixtures._")
        md.append("")
    _write_snapshot("\n".join(md), suffix="sentinel")
    raise typer.Exit(code=0)


# Register both as siblings on the autoresearch app (run/review/adopt pattern; add_typer already
# forces command-group mode, so no extra @callback is needed — 18-06 note).
autoresearch_app.command("report")(report)
autoresearch_app.command("sentinel")(sentinel)
