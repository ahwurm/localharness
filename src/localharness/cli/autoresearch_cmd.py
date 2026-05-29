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
import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from localharness.autoresearch.archive import ArchiveEntry, ArchiveQuery, ArchiveStore

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
    """Resolve .localharness/archive.db, honoring LOCALHARNESS_HOME (mirrors _build_loader).

    When LOCALHARNESS_HOME is set (the components_home fixture sets it to a
    ``.localharness/`` dir), the db sits at ``<home>/archive.db`` — exactly where the
    test's ArchiveStore writes. When unset, default to ``./.localharness/archive.db``
    (project-local per CONTEXT).
    """
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
