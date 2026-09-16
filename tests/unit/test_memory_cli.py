"""`localharness memory list/show/edit/rm` — the CLI half of owner-editable memory.

Everything drives the real Typer app against a real store under an explicit --config-dir
(which pins the state dir and disables workspace discovery, so no trust question and no
dependence on the test runner's cwd)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import click
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.memory.sqlite import USER_EDIT_PROVENANCE_PREFIX, MemoryStore

runner = CliRunner()


def _seed(tmp_path: Path, key: str = "notes/searxng", value: str = "original") -> None:
    async def go():
        store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                            base_dir=str(tmp_path))
        await store.open()
        try:
            await store.store_fact(key=key, value=value, tags=["workaround"],
                                   source="remember")
        finally:
            await store.close()
    asyncio.run(go())


def _fact(tmp_path: Path, key: str):
    async def go():
        store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                            base_dir=str(tmp_path))
        await store.open()
        try:
            return await store.get_fact(key), await store.get_fact_history(key)
        finally:
            await store.close()
    return asyncio.run(go())


def test_list_and_show_print_the_fact(tmp_path):
    _seed(tmp_path)
    out = runner.invoke(app, ["memory", "list", "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "notes/searxng" in out.output and "original" in out.output
    assert str(tmp_path) in out.output               # the header names the store that answered

    out = runner.invoke(app, ["memory", "show", "notes/searxng",
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0 and "original" in out.output

    out = runner.invoke(app, ["memory", "show", "no/such", "--config-dir", str(tmp_path)])
    assert out.exit_code == 1


def test_edit_supersedes_with_cli_stamp_and_carries_tags(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setattr(click, "edit",
                        lambda text, require_save=True: text + "\nEDITED LINE")
    out = runner.invoke(app, ["memory", "edit", "notes/searxng",
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "previous version stays in history" in out.output

    fact, history = _fact(tmp_path, "notes/searxng")
    assert "EDITED LINE" in fact.value
    assert fact.provenance.startswith(USER_EDIT_PROVENANCE_PREFIX)
    assert fact.provenance.endswith(";cli")
    assert "workaround" in (fact.tags or [])         # tags carried, not dropped
    assert len(history) == 2                          # the original kept, superseded

    # Editor closed without changing anything → no phantom supersede.
    monkeypatch.setattr(click, "edit", lambda text, require_save=True: None)
    out = runner.invoke(app, ["memory", "edit", "notes/searxng",
                              "--config-dir", str(tmp_path)])
    assert "unchanged" in out.output
    _, history = _fact(tmp_path, "notes/searxng")
    assert len(history) == 2


def test_rm_retires_recoverably(tmp_path):
    _seed(tmp_path)
    out = runner.invoke(app, ["memory", "rm", "notes/searxng", "--yes",
                              "--config-dir", str(tmp_path)])
    assert out.exit_code == 0 and "retired" in out.output

    fact, history = _fact(tmp_path, "notes/searxng")
    assert fact is None                               # off the hot path…
    assert history and history[0].status == "superseded"   # …but never destroyed
