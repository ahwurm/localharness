"""One-shot tag backfill migration (TAGG-04, Phase 36.2 Plan 03) — the attended, one-way
existing-store migration that files a validated CHILD tag onto every active untagged `sem/`
atom, so existing-store cross-references (fold candidates, supersede targets) key on the tag
axis the Plan-01/02 re-key introduced (RULING-D).

Migrate-safe by construction (the 531-row-priors discipline):
  - append-only        — only atom_tags rows are added (provenance='backfill'); facts rows are
                         byte-untouched (supersede-not-overwrite), so a pre-migration atom stays
                         readable/renderable throughout (its OWN fold still works via Plan 01's
                         slug fallback even before it is reached).
  - backup-then-write  — with backup=True an on-disk snapshot `<db>.backup-pre-tagbackfill-<ts>`
                         is written BEFORE the first write (the WAL is folded in first, so a
                         plain single-file copy is a faithful restore point).
  - idempotent         — an atom already carrying an edge-eligible child tag is excluded, so a
                         re-run tags 0 (add_atom_tag is ON CONFLICT DO NOTHING besides).
  - bounded revert     — revert_backfill is ONE provenance-scoped DELETE of this run's rows
                         (or restore the backup file).

This mirrors the shipped F4 idle step (_step_classify_untagged, consolidation.py:325-355) but
drains ALL untagged atoms in ONE attended pass instead of 10/cycle. The live path (a real
`--store`) is executed ATTENDED in Plan 04, behind the wave-3 KILL/verdict checkpoint; this
module is pure code + fake-LLM unit proof (no vLLM, no live store touched)."""
from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from typing import Any

from localharness.memory.tag_classify import file_atom_tags

# The migration target set: active semantic FACT atoms lacking an edge-eligible CHILD tag. A
# direct NOT EXISTS (rather than atoms_without_child_tag, which is per-bucket and requires an
# EXISTING bucket tag) so this catches BOTH legacy fully-untagged atoms and bucket-only atoms
# in one drain — a strict superset of the F4 idle pool.
_UNTAGGED_SEM_ATOMS = (
    "SELECT f.id, f.key, f.value FROM facts f "
    "WHERE f.agent_id = ? AND f.status = 'active' AND f.node_kind = 'fact' "
    "AND f.key LIKE 'sem/%' "
    "AND NOT EXISTS (SELECT 1 FROM atom_tags ac JOIN tags tc ON tc.id = ac.tag_id "
    "  WHERE ac.atom_id = f.id AND tc.parent_id IS NOT NULL "
    "  AND tc.status IN ('seeded','active'))"
)


async def _checkpoint(store: Any) -> None:
    """Fold the WAL into the main db file so a plain single-file copy is a faithful snapshot
    (SQLite runs in WAL mode; recent commits otherwise live in the -wal sidecar, not the .db)."""
    await store._db.commit()
    async with store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)") as cur:
        await cur.fetchall()


async def backfill_tags(store: Any, llm: Any, *, backup: bool = True,
                        cancel_event: Any = None) -> dict:
    """File a validated child (+bucket) tag onto every active untagged `sem/` atom.

    With backup=True an on-disk snapshot is written BEFORE the first atom_tags write. Returns
    `{"tagged", "processed", "backup", "start_ts"}` — `start_ts` is the revert boundary. Never
    overwrites facts; only appends atom_tags rows with provenance='backfill'."""
    assert store._db is not None
    cancel_event = cancel_event or asyncio.Event()
    db_path = str(store._db_path)

    backup_path = None
    if backup:
        await _checkpoint(store)
        backup_path = f"{db_path}.backup-pre-tagbackfill-{int(time.time())}"
        shutil.copy2(db_path, backup_path)

    start_ts = int(time.time())
    async with store._db.execute(_UNTAGGED_SEM_ATOMS, (store._agent_id,)) as cur:
        todo = [(r[0], r[1], r[2]) for r in await cur.fetchall()]

    tagged = 0
    for atom_id, key, value in todo:
        if cancel_event.is_set():
            break
        topic = key.split("/")[1] if key.startswith("sem/") else key
        _bucket, child = await file_atom_tags(
            store, llm, cancel_event,
            atom_id=atom_id, topic=topic, claim=value, provenance="backfill")
        if child is not None:
            tagged += 1
    return {"tagged": tagged, "processed": len(todo), "backup": backup_path, "start_ts": start_ts}


async def revert_backfill(store: Any, since_ts: int) -> int:
    """Bounded, one-way revert: DELETE exactly the provenance='backfill' atom_tags rows added
    at/after `since_ts` (the backfill's returned start_ts) — leaving 'mint'/'discovery' rows and
    every facts row intact. Single statement. (Full alternative: restore the on-disk
    `<db>.backup-pre-tagbackfill-<ts>` snapshot over the db file.) Returns rows deleted."""
    assert store._db is not None
    async with store._db.execute(
        "DELETE FROM atom_tags WHERE provenance='backfill' AND ts >= ?", (since_ts,)
    ) as cur:
        deleted = cur.rowcount
    await store._db.commit()
    return deleted


# ---------------------------------------------------------------------------
# CLI — the LIVE path (a real --store). Executed ATTENDED in Plan 04, behind the
# wave-3 KILL/verdict checkpoint; never run against a live store from wave 2.
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-shot tag backfill migration (TAGG-04): file a validated child tag onto "
                    "every active untagged sem/ atom. Backup-first, idempotent, bounded revert.")
    p.add_argument("--store", required=True,
                   help="Store base_dir (the agent's memory.db lives under agents/<agent>/). "
                        "LIVE path — executed ATTENDED in Plan 04.")
    p.add_argument("--agent", default="orchestrator", help="Agent id (default orchestrator).")
    p.add_argument("--model", default=None, help="Subject model (classify needs it).")
    p.add_argument("--base-url", default=None, help="vLLM base URL (classify needs it).")
    p.add_argument("--dry-run", action="store_true",
                   help="Count untagged sem/ atoms only — no LLM, no writes.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip the on-disk pre-migration backup (NOT recommended).")
    return p


async def _amain(args: argparse.Namespace) -> int:
    from localharness.memory.sqlite import MemoryStore
    store = MemoryStore(agent_id=args.agent, division_id="", org_id="", base_dir=args.store)
    await store.open()
    try:
        if args.dry_run:
            async with store._db.execute(_UNTAGGED_SEM_ATOMS, (store._agent_id,)) as cur:
                n = len(await cur.fetchall())
            print(f"dry-run: {n} active untagged sem/ atoms would be backfilled")
            return 0
        if not (args.model and args.base_url):
            # classify needs the model — refuse a silent no-op that "succeeds" tagging nothing.
            raise SystemExit(
                "refusing to backfill without --model and --base-url (classify needs the model); "
                "use --dry-run to count only")
        from localharness.memory.idle_llm import LLMTextAdapter
        from localharness.provider.client import LLMClient, LLMConfig
        llm = LLMTextAdapter(LLMClient(LLMConfig(base_url=args.base_url, model=args.model)))
        report = await backfill_tags(store, llm, backup=not args.no_backup)
        print(f"backfill: tagged={report['tagged']}/{report['processed']} "
              f"backup={report['backup']} start_ts={report['start_ts']}")
        print(f"revert (bounded): revert_backfill(store, {report['start_ts']})  "
              f"or restore {report['backup']}")
        return 0
    finally:
        await store.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain(_build_parser().parse_args())))
