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
    help="Browse and edit the agent's persistent memory (list / show / edit / rm).",
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


@memory_app.command("list")
def memory_list(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Full-text filter (FTS)."),
    limit: int = typer.Option(30, "--limit", "-n"),
    agent: str = _AGENT_OPT,
    config_dir: Optional[str] = _CONFIG_OPT,
) -> None:
    """List facts, store-ranked (recency + activation), newest signal first."""
    async def go():
        from localharness.memory.sqlite import FactQuery

        store, db = await _open_store(agent, config_dir)
        try:
            _header(db)
            facts = await store.query_facts(
                FactQuery(text=query or None, min_confidence=0.0, limit=limit)
            )
            if not facts:
                typer.echo("no facts match." if query else "memory is empty.")
                return
            for f in facts:
                first = (f.value or "").strip().splitlines()[0] if (f.value or "").strip() else ""
                typer.echo(f"  {f.key}  —  {first[:90]}")
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
