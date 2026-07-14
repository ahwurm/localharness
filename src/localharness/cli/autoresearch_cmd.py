"""`localharness autoresearch archive` subapp: list, show, approve.

Phase 15 — ARCH-03. The user inspects "why did this mutation work" via
``show <id>`` (lineage is the highest-leverage feature per CONTEXT). Mirrors the
``components_cmd.py`` idioms verbatim: default human table + ``--json`` flag,
``_err`` helper, async-from-sync ``asyncio.run`` bridge, LOCALHARNESS_HOME-resolved
DB path. Diff rendering is stdlib ``difflib`` + ``rich`` (no new dependency). See:
  - 15-RESEARCH.md (CLI Wiring, Prefix resolution, Diff Rendering)
  - 15-CONTEXT.md (CLI surface — the four show display blocks)
"""
from __future__ import annotations

import asyncio
import difflib
import json as _json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from localharness.autoresearch.adoption import AdoptionRefused
from localharness.autoresearch.adoption import adopt as _adopt
from localharness.config.paths import config_dir_env_override
from localharness.autoresearch.archive import ArchiveEntry, ArchiveQuery, ArchiveStore
from localharness.autoresearch.loop import RunSummary, run_loop

autoresearch_app = typer.Typer(
    name="autoresearch",
    help="Autoresearch loop tools.",
    no_args_is_help=True,
)
archive_app = typer.Typer(
    name="archive",
    help="Inspect the mutation archive.",
    no_args_is_help=True,
)
autoresearch_app.add_typer(archive_app, name="archive")

console = Console()
err_console = Console(stderr=True)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _archive_db_path() -> Path:
    """Resolve .localharness/archive.db, honoring the config-dir env chain (#35).

    LOCALHARNESS_DIR (canonical) or LOCALHARNESS_HOME (legacy — the components_home fixture
    sets it to a ``.localharness/`` dir) puts the db at ``<dir>/archive.db``, exactly where the
    test's ArchiveStore writes. When neither is set, default to ``./.localharness/archive.db``
    (project-local per CONTEXT — note this differs from the config dir's ~/.localharness default).
    """
    override = config_dir_env_override()
    base = Path(override).expanduser() if override else Path.cwd() / ".localharness"
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

    In production the CLI is invoked with no running event loop, so ``asyncio.run``
    succeeds directly. Under pytest-asyncio the test body itself runs inside a live
    loop on this thread, so ``asyncio.run`` (and ``run_until_complete`` on any loop in
    this thread) raises "event loop is already running". Detect that case and execute
    the coroutine on a fresh loop in a worker thread, which has no running loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no running loop (the normal CLI path)

    # A loop is already running on this thread -> run the coroutine in a worker thread.
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


_SINCE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_since(s: str | None) -> int | None:
    """Parse a DURATION like ``24h``/``7d``/``30m`` into an epoch floor.

    None if s is None; else trailing unit (s/m/h/d). Raises ValueError on bad format.
    """
    if s is None:
        return None
    s = s.strip()
    if len(s) < 2 or s[-1] not in _SINCE_UNITS:
        raise ValueError(f"bad duration {s!r}: expected e.g. 24h, 7d, 30m")
    try:
        value = int(s[:-1])
    except ValueError as exc:
        raise ValueError(f"bad duration {s!r}: {exc}") from exc
    return int(time.time()) - value * _SINCE_UNITS[s[-1]]


def _parse_budget(s: str | None) -> float | None:
    """Parse a FORWARD duration like ``4h``/``30m``/``2d`` into SECONDS (None if s is None).

    Reuses the ``_SINCE_UNITS`` table (s/m/h/d) but, unlike ``_parse_since`` (an epoch floor
    for the archive window), this returns ``value * unit`` seconds — the loop's wallclock /
    per-proposal-timeout caps are forward durations. Raises ValueError on bad format; the run
    command maps that to exit 2 via ``_err`` (an abnormal start, distinct from a gate verdict).
    """
    if s is None:
        return None
    s = s.strip()
    # A bare number is interpreted as SECONDS (e.g. --budget 1 == 1s); a trailing unit scales.
    if s and s[-1] not in _SINCE_UNITS:
        try:
            return float(s)
        except ValueError as exc:
            raise ValueError(f"bad duration {s!r}: expected e.g. 4h, 30m, 2d, or seconds") from exc
    if len(s) < 2:
        raise ValueError(f"bad duration {s!r}: expected e.g. 4h, 30m, 2d")
    try:
        value = int(s[:-1])
    except ValueError as exc:
        raise ValueError(f"bad duration {s!r}: {exc}") from exc
    return float(value * _SINCE_UNITS[s[-1]])


def _entry_to_dict(e: ArchiveEntry) -> dict:
    """Full-row dict for --json: the 12 ArchiveEntry fields (tspf as a dict)."""
    return {
        "id": e.id,
        "parent_id": e.parent_id,
        "component": e.component,
        "diff": e.diff,
        "train_score": e.train_score,
        "train_scores_per_fixture": e.train_scores_per_fixture,
        "holdout_score": e.holdout_score,
        "p_value": e.p_value,
        "cost": e.cost,
        "ts": e.ts,
        "approved_by": e.approved_by,
        "status": e.status,
    }


def _fmt_float(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def _fmt_ts(ts: int) -> str:
    """Relative age string, e.g. '5m ago' / '3d ago'."""
    delta = int(time.time()) - ts
    if delta < 0:
        delta = 0
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


# ------------------------------------------------------------------ #
# list
# ------------------------------------------------------------------ #


@archive_app.command("list")
def archive_list(
    component: Optional[str] = typer.Option(None, "--component", help="Exact dot-path filter"),
    since: Optional[str] = typer.Option(None, "--since", help="Duration window, e.g. 24h, 7d"),
    status: Optional[str] = typer.Option(None, "--status", help="Lifecycle status filter"),
    limit: int = typer.Option(20, "--limit", help="Max rows (default 20)"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON array instead of a table"),
) -> None:
    """List recent mutations (ts DESC) with --component/--since/--status/--limit filters."""
    try:
        since_ts = _parse_since(since)
    except ValueError as exc:
        _err(json_output, str(exc), exit_code=2)
        return

    q = ArchiveQuery(component=component, status=status, since_ts=since_ts, limit=limit)

    async def _go() -> list[ArchiveEntry]:
        db_path = _archive_db_path()
        if not db_path.exists():
            return []  # no archive yet -> empty result, exit 0
        store = ArchiveStore(db_path)
        await store.open()
        try:
            return await store.query(q)
        finally:
            await store.close()

    entries = _run(_go())

    if json_output:
        typer.echo(_json.dumps([_entry_to_dict(e) for e in entries], indent=2))
        return

    table = Table(title="Mutation Archive", show_lines=False, pad_edge=False, expand=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("component", no_wrap=True)
    table.add_column("train")
    table.add_column("hold")
    table.add_column("p")
    table.add_column("cost")
    table.add_column("status", style="green")
    table.add_column("approved", style="dim")
    table.add_column("ts")
    for e in entries:
        table.add_row(
            e.id[:8],
            e.component,
            _fmt_float(e.train_score),
            _fmt_float(e.holdout_score),
            _fmt_float(e.p_value),
            _fmt_float(e.cost),
            e.status,
            e.approved_by or "-",
            _fmt_ts(e.ts),
        )
    # Render at a fixed wide width so non-tty (CliRunner / pipes) don't crop columns.
    Console(width=200).print(table)


# ------------------------------------------------------------------ #
# show / approve helpers
# ------------------------------------------------------------------ #


async def _resolve(store: ArchiveStore, raw_id: str):
    """Resolve a full UUID or 8-char hex prefix to a single entry.

    Returns (entry, matches): exactly one of these is meaningful.
      - unique hit  -> (ArchiveEntry, None)
      - not found   -> (None, [])
      - ambiguous   -> (None, [match, match, ...])   (>1 prefix match)
    """
    if len(raw_id) == 36:  # full UUID
        entry = await store.get(raw_id)
        return (entry, None) if entry is not None else (None, [])

    rows = await store.query(ArchiveQuery(limit=10_000))
    matches = [e for e in rows if e.id.startswith(raw_id)]
    if len(matches) == 0:
        return (None, [])
    if len(matches) == 1:
        return (matches[0], None)
    return (None, matches)


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


# ------------------------------------------------------------------ #
# show
# ------------------------------------------------------------------ #


@archive_app.command("show")
def archive_show(
    id: str = typer.Argument(..., help="8-char hex prefix or full UUID"),
    json_output: bool = typer.Option(False, "--json", help="Emit full row + lineage as JSON"),
) -> None:
    """Show one mutation: header + metrics + red/green diff + git-log-oneline lineage."""

    async def _go():
        db_path = _archive_db_path()
        if not db_path.exists():
            return (None, [], [])
        store = ArchiveStore(db_path)
        await store.open()
        try:
            entry, matches = await _resolve(store, id)
            if entry is None:
                return (None, matches, [])
            lineage = await store.lineage(entry.id)
            return (entry, None, lineage)
        finally:
            await store.close()

    entry, matches, lineage = _run(_go())

    if entry is None:
        if matches:  # ambiguous prefix
            if json_output:
                typer.echo(
                    _json.dumps({"error": "ambiguous prefix", "matches": [m.id for m in matches]}),
                    err=True,
                )
            else:
                err_console.print(f"[bold red]Error:[/bold red] ambiguous prefix {id!r} matches {len(matches)}:")
                for m in matches:
                    err_console.print(f"  {m.id} {m.component} {m.status}")
            raise typer.Exit(code=2)
        _err(json_output, f"no mutation matches id {id!r}", exit_code=2)
        return

    if json_output:
        payload = _entry_to_dict(entry)
        payload["lineage"] = [_entry_to_dict(e) for e in lineage]
        typer.echo(_json.dumps(payload, indent=2))
        return

    # Block 1: header
    console.print(f"[bold]{entry.id}[/bold]")
    console.print(f"  parent:    {entry.parent_id or '-'}")
    console.print(f"  component: {entry.component}")
    console.print(f"  status:    {entry.status}")
    console.print(f"  ts:        {entry.ts} ({_fmt_ts(entry.ts)})")
    console.print(f"  approved:  {entry.approved_by or '-'}")

    # Block 2: metrics
    console.print("[bold]metrics[/bold]")
    console.print(f"  train:   {_fmt_float(entry.train_score)}")
    console.print(f"  holdout: {_fmt_float(entry.holdout_score)}")
    console.print(f"  p_value: {_fmt_float(entry.p_value)}")
    console.print(f"  cost:    {_fmt_float(entry.cost)}")
    tspf = entry.train_scores_per_fixture
    if tspf:
        ordered = sorted(tspf.items(), key=lambda kv: kv[1], reverse=True)
        best = ordered[:3]
        worst = list(reversed(ordered[-3:]))
        console.print("  best:  " + ", ".join(f"{k}={v:.3f}" for k, v in best))
        console.print("  worst: " + ", ".join(f"{k}={v:.3f}" for k, v in worst))

    # Block 3: diff
    console.print("[bold]diff[/bold]")
    _render_diff(entry.diff)

    # Block 4: lineage (lineage() returns child->root = newest first = git-log-oneline order)
    console.print("[bold]lineage[/bold]")
    for e in lineage:
        console.print(f"{e.id[:8]} {e.component} {e.status} {e.ts}")


# ------------------------------------------------------------------ #
# approve
# ------------------------------------------------------------------ #


@archive_app.command("approve")
def archive_approve(
    id: str = typer.Argument(..., help="8-char hex prefix or full UUID"),
    approver: str = typer.Option(..., "--approver", help="Approver, e.g. human:alice"),
    comment: Optional[str] = typer.Option(None, "--comment", help="Optional note"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Append an approval to a mutation's history."""

    async def _go():
        db_path = _archive_db_path()
        if not db_path.exists():
            return (None, [])
        store = ArchiveStore(db_path)
        await store.open()
        try:
            entry, matches = await _resolve(store, id)
            if entry is None:
                return (None, matches)
            await store.add_approval(entry.id, approver, comment)
            refreshed = await store.get(entry.id)
            return (refreshed, None)
        finally:
            await store.close()

    entry, matches = _run(_go())

    if entry is None:
        if matches:
            if json_output:
                typer.echo(
                    _json.dumps({"error": "ambiguous prefix", "matches": [m.id for m in matches]}),
                    err=True,
                )
            else:
                err_console.print(f"[bold red]Error:[/bold red] ambiguous prefix {id!r} matches {len(matches)}:")
                for m in matches:
                    err_console.print(f"  {m.id} {m.component} {m.status}")
            raise typer.Exit(code=2)
        _err(json_output, f"no mutation matches id {id!r}", exit_code=2)
        return

    if json_output:
        typer.echo(_json.dumps({"id": entry.id, "approved_by": entry.approved_by}))
        return
    console.print(f"[green]approved[/green] {entry.id[:8]} by {approver}")


# ------------------------------------------------------------------ #
# run / review / adopt — AUTO-01 + AUTO-04 (the hands-off entrypoint)
#
# These are commands on ``autoresearch_app`` itself (siblings of the ``archive`` sub-app).
# ``add_typer(archive_app)`` above already forces Typer's command-group mode, so each command
# stays a NAMED subcommand (no extra @callback needed — confirmed: ``autoresearch run --help``
# resolves). The ``run`` command's exit code is the RUN code (0 = any clean halt), DISTINCT
# from the gate verdict band (a loop full of rejects is still exit 0); the ONLY non-zero exits
# are the early bad-flag / config-error paths via ``_err`` (Pitfall 5).
# ------------------------------------------------------------------ #


def _repo_root() -> Path:
    """Resolve the git toplevel of the cwd (the MAIN repo adoptions commit into)."""
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(Path.cwd()), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


@autoresearch_app.command("run")
def autoresearch_run(
    budget: Optional[str] = typer.Option(
        None, "--budget", help="wallclock cap, e.g. 4h, 30m (default: until the 5h window is ~spent)"
    ),
    max_cost: Optional[float] = typer.Option(
        None, "--max-cost", help="USD cap (sums archive per-row cost; ~$0 for a local proposer)"
    ),
    max_iterations: int = typer.Option(
        1000, "--max-iterations", help="hard iteration backstop so a metering bug can't loop forever"
    ),
    checkpoint_every: int = typer.Option(
        5,
        "--checkpoint-every",
        help="reserved for AUTO-01 literal compliance (held items surface via the Phase 19 report; this loop is fire-and-forget)",
    ),
    epsilon: float = typer.Option(
        0.2, "--epsilon", help="explore probability for the parent sampler"
    ),
    min_lift: Optional[float] = typer.Option(
        None,
        "--min-lift",
        help="effect-size floor for auto-adoption; default unset — calibrate from early-run data (see journal)",
    ),
    proposal_timeout: str = typer.Option(
        "30m", "--proposal-timeout", help="hard-kill a single hung experiment"
    ),
    claude_window_tokens: Optional[int] = typer.Option(
        None,
        "--claude-window-tokens",
        help="proposer 5h-window token budget; set to match your Max plan",
    ),
    json_output: bool = typer.Option(False, "--json", help="emit the RunSummary as JSON"),
) -> None:
    """Drive the autonomous self-improvement loop until a budget/breaker/interrupt halts.

    Exit code is the RUN code: 0 on ANY clean halt (budget / circuit_breaker / interrupt /
    complete). The ONLY non-zero exits are an early bad-flag or config-error (exit 2) — a gate
    verdict NEVER becomes the run's exit code (Pitfall 5).
    """
    # Bad flag = abnormal start = non-zero (exit 2), DISTINCT from the gate's 0-3 verdict band.
    try:
        budget_s = _parse_budget(budget)
        timeout_s = _parse_budget(proposal_timeout) or 1800.0
    except ValueError as exc:
        _err(json_output, str(exc), exit_code=2)
        return

    # checkpoint_every is accepted for AUTO-01 literal compliance; this loop is fire-and-forget
    # (held items surface async via the Phase 19 report — AUTO-04 amended to auto-adopt).
    _ = checkpoint_every

    try:
        from localharness.cli.components_cmd import _build_loader

        cfg = _build_loader().load_harness()
    except Exception as exc:  # config load failure = abnormal start, not a gate verdict
        _err(json_output, f"config error: {exc}", exit_code=2)
        return

    try:
        repo_root = _repo_root()
    except Exception as exc:
        _err(json_output, f"not inside a git repo: {exc}", exit_code=2)
        return

    async def _go() -> RunSummary:
        db_path = _archive_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = ArchiveStore(db_path)
        await store.open()
        try:
            return await run_loop(
                store=store,
                cfg=cfg,
                repo_root=repo_root,
                budget=budget_s,
                max_iterations=max_iterations,
                max_cost=max_cost,
                epsilon=epsilon,
                min_lift=min_lift,
                proposal_timeout=timeout_s,
                window_tokens=claude_window_tokens,
            )
        finally:
            await store.close()

    summary = _run(_go())

    # The RUN-COMPLETE summary (the run-complete ping content): counts + consumption + journal.
    if json_output:
        payload = {
            "run_id": getattr(summary, "run_id", None),
            "iterations": summary.iterations,
            "adopted": summary.adopted,
            "held": summary.held,
            "rejected": summary.rejected,
            "skipped": getattr(summary, "skipped", None),
            "seconds_elapsed": getattr(summary, "seconds_elapsed", None),
            "tokens_spent": getattr(summary, "tokens_spent", None),
            "top_wins": getattr(summary, "top_wins", None),
            "journal_path": getattr(summary, "journal_path", None),
            "halt_reason": getattr(summary, "halt_reason", None),
        }
        typer.echo(_json.dumps(payload))
    else:
        console.print("[bold]autoresearch run complete[/bold]")
        console.print(f"  halt:       {getattr(summary, 'halt_reason', '-')}")
        console.print(f"  iterations: {summary.iterations}")
        console.print(
            f"  adopted={summary.adopted}  held={summary.held}  "
            f"rejected={summary.rejected}  skipped={getattr(summary, 'skipped', 0)}"
        )
        elapsed = getattr(summary, "seconds_elapsed", None)
        if elapsed is not None:
            console.print(f"  time:       {elapsed:.1f}s")
        tokens = getattr(summary, "tokens_spent", None)
        if tokens is not None:
            console.print(f"  claude-window tokens: {tokens}")
        wins = getattr(summary, "top_wins", None) or []
        if wins:
            console.print("[bold]top wins[/bold]")
            for win in wins[:5]:
                wid, comp, score = (list(win) + [None, None, None])[:3]
                console.print(f"  {str(wid)[:8]} {comp} train={_fmt_float(score)}")
        console.print(f"  journal:    {getattr(summary, 'journal_path', '-')}")

    # Any clean halt is exit 0 — a gate reject inside the loop is NOT a non-zero run.
    raise typer.Exit(code=0)


def _review_card_dict(e: ArchiveEntry) -> dict:
    """A held-item review card for --json (the full row + a decoded before/after diff)."""
    payload = _entry_to_dict(e)
    try:
        payload["diff_decoded"] = e.diff_decoded
    except (ValueError, TypeError):
        payload["diff_decoded"] = None
    return payload


@autoresearch_app.command("review")
def autoresearch_review(
    limit: int = typer.Option(50, "--limit", help="Max held items to surface (default 50)"),
    json_output: bool = typer.Option(False, "--json", help="Emit held items as a JSON array"),
) -> None:
    """List HELD items as review cards (component, before->after diff, lift, p-value, id).

    Only ``held`` rows are reviewable — ``adoption_rejected`` rows are kept as parent material
    and never re-offered. An empty/absent archive is a normal survey result (exit 0).
    """

    async def _go() -> list[ArchiveEntry]:
        db_path = _archive_db_path()
        if not db_path.exists():
            return []  # no archive yet -> nothing held, exit 0
        store = ArchiveStore(db_path)
        await store.open()
        try:
            return await store.query(ArchiveQuery(status="held", limit=limit))
        finally:
            await store.close()

    held = _run(_go())

    if json_output:
        typer.echo(_json.dumps([_review_card_dict(e) for e in held], indent=2))
        return

    if not held:
        console.print("[dim]No held items to review.[/dim]")
        return

    console.print(f"[bold]{len(held)} held item(s) for review[/bold]")
    for e in held:
        console.print(f"\n[bold cyan]{e.id[:8]}[/bold cyan]  {e.component}")
        console.print(
            f"  train={_fmt_float(e.train_score)}  holdout={_fmt_float(e.holdout_score)}  "
            f"p={_fmt_float(e.p_value)}"
        )
        console.print("  diff:")
        _render_diff(e.diff)
        console.print(f"  adopt with: localharness autoresearch adopt {e.id[:8]}")


@autoresearch_app.command("adopt")
def autoresearch_adopt(
    id: str = typer.Argument(..., help="8-char hex prefix or full UUID of a held item"),
    json_output: bool = typer.Option(False, "--json", help="Emit the adoption result as JSON"),
) -> None:
    """Adopt a HELD item into LIVE config (overlay write + git commit in the MAIN repo).

    A ``adoption_rejected`` row is kept as parent material and never re-offered (refused, exit 2);
    an already-``adopted`` row is a no-op (exit 0). On a successful adopt the commit sha is printed
    and the row's status flips to ``adopted``.
    """
    try:
        from localharness.cli.components_cmd import _build_loader

        cfg = _build_loader().load_harness()
    except Exception as exc:
        _err(json_output, f"config error: {exc}", exit_code=2)
        return

    try:
        repo_root = _repo_root()
    except Exception as exc:
        _err(json_output, f"not inside a git repo: {exc}", exit_code=2)
        return

    async def _go():
        db_path = _archive_db_path()
        if not db_path.exists():
            return ("missing", None, [])
        store = ArchiveStore(db_path)
        await store.open()
        try:
            entry, matches = await _resolve(store, id)
            if entry is None:
                return ("resolve", None, matches)
            if entry.status == "adoption_rejected":
                return ("rejected", entry, [])
            if entry.status == "adopted":
                return ("already", entry, [])
            sha = await _adopt(entry.id, store=store, cfg=cfg, repo_root=repo_root)
            await store.update_verdict(entry.id, status="adopted")
            return ("adopted", entry, sha)
        finally:
            await store.close()

    try:
        outcome, entry, extra = _run(_go())
    except AdoptionRefused as exc:
        # The row is now adoption_rejected (set inside adopt); kept as parent material.
        _err(json_output, f"adoption refused: {exc}", exit_code=2)
        return

    if outcome == "missing":
        _err(json_output, f"no mutation matches id {id!r}", exit_code=2)
        return
    if outcome == "resolve":
        matches = extra
        if matches:  # ambiguous prefix
            if json_output:
                typer.echo(
                    _json.dumps({"error": "ambiguous prefix", "matches": [m.id for m in matches]}),
                    err=True,
                )
            else:
                err_console.print(
                    f"[bold red]Error:[/bold red] ambiguous prefix {id!r} matches {len(matches)}:"
                )
                for m in matches:
                    err_console.print(f"  {m.id} {m.component} {m.status}")
            raise typer.Exit(code=2)
        _err(json_output, f"no mutation matches id {id!r}", exit_code=2)
        return
    if outcome == "rejected":
        _err(
            json_output,
            f"{id!r} was already rejected; kept as parent material, never re-offered",
            exit_code=2,
        )
        return
    if outcome == "already":
        if json_output:
            typer.echo(_json.dumps({"id": entry.id, "status": "adopted", "sha": None}))
        else:
            console.print(f"[yellow]{entry.id[:8]} already adopted[/yellow] ({entry.component})")
        raise typer.Exit(code=0)

    # outcome == "adopted"
    sha = extra
    if json_output:
        typer.echo(_json.dumps({"id": entry.id, "status": "adopted", "sha": sha}))
    else:
        console.print(f"[green]adopted[/green] {entry.id[:8]} {entry.component} -> {sha[:8]}")
    raise typer.Exit(code=0)
