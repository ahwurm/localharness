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
