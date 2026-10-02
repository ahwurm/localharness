"""`localharness memory` pinned per verb from the unconverted tree (Phase 48 Wave 1, captured at
7757e09): exit code + normalised output for 19 cases, one JSON golden.

Invoked through the ROOT `app` only, so after the CLI moves into the memory plugin (48-04) the same
test exercises the lazy plugin mount. These goldens MUST hold unedited; a diff is a finding to
explain, never a golden to regenerate.

Normalisation (`_norm`): `str(cfg)` -> `<CFG>`, `str(tmp_path)` -> `<TMP>`,
`user_edit@\\d+` -> `user_edit@<EPOCH>`, `YYYY-MM-DD HH:MM` -> `<STAMP>`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.memory.consolidation import archive_listed_facts
from localharness.memory.sqlite import MemoryStore

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "memory_surfaces" / "cli_golden.json"
REGEN = os.environ.get("LOCALHARNESS_REGEN_GOLDEN") == "1"
ENV = {"COLUMNS": "100", "TERMINAL_WIDTH": "100", "NO_COLOR": "1"}
runner = CliRunner()


def _seed(cfg: Path) -> int:
    """The store `memory --config-dir cfg` opens; returns the archived fact's id."""
    async def go():
        store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                            base_dir=str(cfg))
        await store.open()
        try:
            await store.store_fact(key="user-editor", value="The user edits in Neovim with a tiling WM.",
                                   tags=["prefs"], source="remember")
            await store.store_fact(key="project-lang", value="The project is Python 3.12 on uv.",
                                   source="remember")
            await store.store_fact(key="deploy-target", value="staging box", source="remember")
            await store.store_fact(key="deploy-target", value="prod box", source="remember")
            old = await store.store_fact(key="old-note", value="An archived note.", source="consolidation")
            run = await archive_listed_facts(store, [old.id])
            assert run.moved == 1, run
            return old.id
        finally:
            await store.close()
    return asyncio.run(go())


# case id -> (argv, stdin, editor result: "append" | None | "unused")
CASES = {
    "help": None,
    "list": (["list"], None),
    "list_query": (["list", "-q", "neovim"], None),
    "list_query_nomatch": (["list", "-q", "zzzz"], None),
    "list_archived": (["list", "--archived"], None),
    "show": (["show", "user-editor"], None),
    "show_history": (["show", "deploy-target", "--history"], None),
    "show_missing": (["show", "nosuch"], None),
    "edit": (["edit", "user-editor"], None),
    "edit_unchanged": (["edit", "project-lang"], None),
    "edit_missing": (["edit", "nosuch"], None),
    "rm_yes": (["rm", "project-lang", "--yes"], None),
    "rm_declined": (["rm", "user-editor"], "n\n"),
    "rm_missing": (["rm", "nosuch"], None),
    "archive_dry_run": (["archive", "--dry-run"], None),
    "archive_from_list": (["archive", "--from-list", "<LIST>"], None),
    "archive_from_missing_list": (["archive", "--from-list", "<MISSING>"], None),
    "restore": (["restore", "<ARCHIVED>"], None),
    "restore_missing": (["restore", "99999"], None),
}


def _norm(text: str, cfg: Path, tmp_path: Path) -> str:
    text = text.replace(str(cfg), "<CFG>").replace(str(tmp_path), "<TMP>")
    text = re.sub(r"user_edit@\d+", "user_edit@<EPOCH>", text)
    return re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "<STAMP>", text)


@pytest.mark.parametrize("case", list(CASES))
def test_memory_cli_matches_the_golden(case, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    archived = _seed(cfg)
    listing = tmp_path / "list.txt"
    listing.write_text("# condemned\n1 | user-editor\nnot-an-id\n9999\n", encoding="utf-8")
    edits = {"edit": lambda text, require_save=True: text + "\nAlso uses tmux.",
             "edit_unchanged": lambda text, require_save=True: None}
    monkeypatch.setattr(click, "edit", edits.get(case, lambda *a, **k: pytest.fail("editor opened")))

    if CASES[case] is None:
        result = runner.invoke(app, ["memory"], env=ENV)
    else:
        argv, stdin = CASES[case]
        subs = {"<LIST>": str(listing), "<MISSING>": str(tmp_path / "missing.txt"),
                "<ARCHIVED>": str(archived)}
        argv = [subs.get(a, a) for a in argv]
        result = runner.invoke(app, ["memory", *argv, "--config-dir", str(cfg)], input=stdin, env=ENV)
    got = {"exit": result.exit_code, "output": _norm(result.output, cfg, tmp_path)}

    if REGEN:
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[case] = got
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert got == json.loads(GOLDEN.read_text())[case], f"`localharness memory` {case} drifted"
