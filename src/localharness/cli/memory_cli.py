"""`localharness memory …` — browse and edit the agent's persistent memory, no session needed.

Owner ask (2026-09-16): memory must be easy to edit from the CLI as well as the app. Every verb
here speaks the SAME MemoryStore API the memory tools use — `store_fact` supersede (history
kept, read-back verified) and `forget_fact` retire — never raw sqlite, never a hard delete.
An edit is stamped `user_edit@<epoch>;cli`, the CLI sibling of the web page's `;web` stamp.

Store resolution mirrors `_start_async` (config dir → workspace layer → state dir → loader),
with `interactive=False`: this command must never pose the trust question — an untrusted cwd
simply resolves to the global store, and the header line SAYS which store answered, so an edit
never lands somewhere the eye didn't see named.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer

memory_app = typer.Typer(
    name="memory",
    help="Browse and edit the agent's persistent memory "
         "(list / show / edit / rm / archive / restore).",
    no_args_is_help=True,
)

_AGENT_OPT = typer.Option("orchestrator", "--agent", "-a", help="Agent whose store to open.")
_CONFIG_OPT = typer.Option(None, "--config-dir", help="Config dir override (advanced).")


async def _open_store(agent: str, config_dir: Optional[str]):
    """(store, db_path) for `agent` in the current workspace — the start_cmd resolution, lifted."""
    from localharness.config.loader import ConfigLoader
    from localharness.config.paths import resolve_config_dir
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.memory.sqlite import MemoryStore

    cfg_path = resolve_config_dir(config_dir)
    workspace = resolve_workspace_layer(config_dir, interactive=False)
    state_dir = workspace if workspace is not None else cfg_path
    loader = ConfigLoader(config_dir=cfg_path, local_config_dir=workspace)
    try:
        agent_config = loader.load_agent(agent)
        division = agent_config.division or "default"
    except Exception:
        division = "default"  # the store is keyed by agent_id; a missing yaml must not block reads
    store = MemoryStore(
        agent_id=agent, division_id=division, org_id="default",
        base_dir=str(state_dir), global_base_dir=str(cfg_path),
    )
    await store.open()
    return store, Path(state_dir) / "agents" / agent / "memory.db"


def _header(db_path: Path) -> None:
    typer.echo(f"store: {db_path}")


def _run(coro) -> None:
    asyncio.run(coro)


# Display-only: how many candidate names `archive --dry-run` prints before summarising the
# rest. NOT a behavioural bar — it changes nothing about what moves; sized to leave the
# per-source table on the same screen.
_PREVIEW_KEYS = 20


@memory_app.command("list")
def memory_list(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Full-text filter (FTS)."),
    limit: int = typer.Option(30, "--limit", "-n"),
    archived: bool = typer.Option(
        False, "--archived",
        help="List ARCHIVED facts instead (most recently archived first). The archive is "
             "deliberately outside the search index, so --query filters these by plain "
             "substring, not FTS.",
    ),
    agent: str = _AGENT_OPT,
    config_dir: Optional[str] = _CONFIG_OPT,
) -> None:
    """List facts, store-ranked (recency + activation), newest signal first."""
    async def go():
        from localharness.memory.sqlite import FactQuery

        store, db = await _open_store(agent, config_dir)
        try:
            _header(db)
            if archived:
                facts = await store.list_archived(limit=limit)
                if query:
                    needle = query.lower()
                    facts = [f for f in facts
                             if needle in f.key.lower() or needle in (f.value or "").lower()]
                if not facts:
                    typer.echo("no archived facts match." if query else "the archive is empty.")
                    return
                total = await store.count_archived()
                typer.echo(f"archived: {total} fact(s) — restore one with "
                           f"`localharness memory restore <id>`")
            else:
                facts = await store.query_facts(
                    FactQuery(text=query or None, min_confidence=0.0, limit=limit)
                )
                if not facts:
                    typer.echo("no facts match." if query else "memory is empty.")
                    return
            for f in facts:
                first = (f.value or "").strip().splitlines()[0] if (f.value or "").strip() else ""
                prefix = f"  [{f.id}] " if archived else "  "
                typer.echo(f"{prefix}{f.key}  —  {first[:90]}")
        finally:
            await store.close()
    _run(go())


@memory_app.command("archive")
def memory_archive(
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Report what WOULD be archived and move nothing.",
    ),
    agent: str = _AGENT_OPT,
    config_dir: Optional[str] = _CONFIG_OPT,
) -> None:
    """Archive dormant facts — off every hot path, restorable, never deleted.

    A fact archives when its salience (need + truth + stakes) falls below the store's own
    proven-useful floor: the lowest-scoring fact that was ever actually recalled or touched
    by you. That line is READ OUT OF THE STORE on every run, never configured, and a store
    with nothing recalled yet has no line and archives nothing.

    This verb is the explicit, owner-triggered run — it works whether or not the automatic
    step (`agent.memory.archival.enabled`) is on. Read a --dry-run first.
    """
    async def go():
        from localharness.memory.consolidation import archive_dormant_facts

        store, db = await _open_store(agent, config_dir)
        try:
            _header(db)
            run = await archive_dormant_facts(store, dry_run=dry_run)
            if run.line is None:
                typer.echo(
                    f"active facts: {run.active_before} — no line yet: nothing in this store "
                    f"has ever been recalled or touched by you, so there is no evidence to "
                    f"draw a floor from. Nothing archived."
                )
                return
            verb = "would archive" if dry_run else "archived"
            typer.echo(
                f"active facts: {run.active_before}  "
                f"(anchors: {run.anchors} recalled-or-yours)"
            )
            typer.echo(f"line S={run.line:.4f} — the lowest-scoring anchor")
            typer.echo(f"{verb}: {len(run.candidates) if dry_run else run.moved} fact(s)")
            if run.pinned_below_line:
                typer.echo(f"pinned below the line and KEPT: {run.pinned_below_line}")
            if not dry_run and run.refused:
                typer.echo(f"skipped (read since the last fold, or changed underfoot): "
                           f"{run.refused}")
            if not dry_run and run.vacuumed:
                typer.echo("vacuumed: more rows left than stayed, so the file was rewritten")
            by_source: dict[str, int] = {}
            for c in run.candidates:
                by_source[c.source or "(none)"] = by_source.get(c.source or "(none)", 0) + 1
            if by_source:
                typer.echo("by source:")
                for src, n in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
                    typer.echo(f"  {n:>5}  {src}")
                typer.echo(f"first {min(_PREVIEW_KEYS, len(run.candidates))}:")
                for c in sorted(run.candidates, key=lambda c: c.s)[:_PREVIEW_KEYS]:
                    typer.echo(f"  S={c.s:+.4f}  {c.key}")
                if len(run.candidates) > _PREVIEW_KEYS:
                    typer.echo(f"  … and {len(run.candidates) - _PREVIEW_KEYS} more")
            if dry_run:
                typer.echo("dry run — nothing moved.")
            else:
                typer.echo("restore any of them with `localharness memory restore <id>` "
                           "(`localharness memory list --archived` lists ids).")
        finally:
            await store.close()
    _run(go())


@memory_app.command("restore")
def memory_restore(
    fact_id: int = typer.Argument(..., help="Archived fact id (see `memory list --archived`)."),
    agent: str = _AGENT_OPT,
    config_dir: Optional[str] = _CONFIG_OPT,
) -> None:
    """Bring an archived fact back, byte-identical, onto every hot path."""
    async def go():
        store, db = await _open_store(agent, config_dir)
        try:
            _header(db)
            ok = await store.restore_fact(fact_id)
            if ok:
                fact = await store.get_fact_by_id(fact_id)
                name = fact.key if fact is not None else str(fact_id)
                typer.echo(f"restored {name} — searchable again, exactly as it was archived.")
                return
            typer.echo(
                f"nothing restored: id {fact_id} is not in the archive, or a newer active "
                f"fact already holds that name (the live one wins; the archived copy stays "
                f"safe). `localharness memory list --archived` lists what is there.",
                err=True,
            )
            raise typer.Exit(1)
        finally:
            await store.close()
    _run(go())


@memory_app.command("show")
def memory_show(
    name: str = typer.Argument(..., help="Exact fact name (names may carry slashes)."),
    history: bool = typer.Option(False, "--history", help="Every kept version, newest first."),
    agent: str = _AGENT_OPT,
    config_dir: Optional[str] = _CONFIG_OPT,
) -> None:
    """Print one fact in full."""
    async def go():
        store, db = await _open_store(agent, config_dir)
        try:
            _header(db)
            rows = await store.get_fact_history(name) if history else []
            fact = await store.get_fact(name)
            if fact is None and not rows:
                typer.echo(f"no fact named {name!r}", err=True)
                raise typer.Exit(1)
            for f in (rows or [fact]):
                typer.echo(f"— {f.key}  [{f.status}]  source={f.source or '-'}  "
                           f"provenance={f.provenance or '-'}")
                typer.echo(f.value or "")
                typer.echo("")
        finally:
            await store.close()
    _run(go())


@memory_app.command("edit")
def memory_edit(
    name: str = typer.Argument(..., help="Exact fact name to edit."),
    agent: str = _AGENT_OPT,
    config_dir: Optional[str] = _CONFIG_OPT,
) -> None:
    """Open the fact in $EDITOR; saving supersedes it — the old version stays in history."""
    import click

    async def read_current():
        store, db = await _open_store(agent, config_dir)
        try:
            return db, await store.get_fact(name)
        finally:
            await store.close()

    async def write_back(content: str, current) -> None:
        from localharness.memory.sqlite import USER_EDIT_PROVENANCE_PREFIX

        store, _ = await _open_store(agent, config_dir)
        try:
            await store.store_fact(
                key=name, value=content,
                tags=list(current.tags or []),          # carried, not dropped
                confidence=current.confidence,
                source="user_edit",
                provenance=f"{USER_EDIT_PROVENANCE_PREFIX}{int(time.time())};cli",
                node_kind=getattr(current, "node_kind", None) or "fact",
            )
        finally:
            await store.close()

    db, current = asyncio.run(read_current())
    _header(db)
    if current is None:
        typer.echo(f"no active fact named {name!r} — edit changes an existing fact", err=True)
        raise typer.Exit(1)
    # The store is CLOSED while $EDITOR is open — an editor session can be minutes long, and
    # the live agent must not share a write window with it any longer than the save itself.
    edited = click.edit(current.value or "", require_save=True)
    if edited is None or edited.strip() == (current.value or "").strip():
        typer.echo("unchanged.")
        return
    asyncio.run(write_back(edited.strip(), current))
    typer.echo(f"edited {name} — the previous version stays in history "
               f"(see `localharness memory show {name} --history`).")


@memory_app.command("rm")
def memory_rm(
    name: str = typer.Argument(..., help="Exact fact name to retire."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    agent: str = _AGENT_OPT,
    config_dir: Optional[str] = _CONFIG_OPT,
) -> None:
    """Retire a fact off every hot path (recoverable — never a hard delete)."""
    async def go():
        store, db = await _open_store(agent, config_dir)
        try:
            _header(db)
            fact = await store.get_fact(name)
            if fact is None:
                typer.echo(f"no active fact named {name!r}", err=True)
                raise typer.Exit(1)
            preview = (fact.value or "").strip().splitlines()[0][:90]
            if not yes and not typer.confirm(f"retire {name!r} ({preview})?"):
                typer.echo("kept.")
                return
            ok = await store.forget_fact(fact.id)
            typer.echo(f"retired {name} — recoverable in history." if ok
                       else "a live turn superseded it first; nothing changed.")
        finally:
            await store.close()
    _run(go())
