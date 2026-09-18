"""The memory store is out of an agent's reach — owner order 2026-09-18.

"MAKE SURE it is inaccessible even to exec so a agent cant search a shitty old memory store."

Two halves, because either one alone is a half-truth:

1. **The exec surfaces refuse.** Shipped deny patterns put the store's artifacts out of reach
   of BOTH bash_exec and python_exec (python_exec's own docstring says it has "no
   sandbox/isolation — same trust posture as bash_exec", so a bash-only list is one
   `import sqlite3` from useless). Anchored on ARTIFACT NAMES, so ordinary work that merely
   says "memory" is untouched.
2. **The archive is structurally invisible.** No agent-facing read path queries
   `facts_archive` — asserted at AST level over the whole source tree, so a future read path
   that reaches into the archive fails this test rather than shipping. The CLI's
   `list --archived` / `restore` are OWNER verbs and stay.

Honest scope, stated here so nobody reads more into it: deny patterns match raw
pre-expansion argument text. They stop stumbling and casual access, not a determined
adversary — a shell glob (`sqlite3 *.db`), a renamed copy, or a path built at runtime all
walk past them. Real containment is `permissions.workspace_root` or a container.
"""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest

from localharness.agent.permissions import (
    MEMORY_STORE_DENY_GUIDANCE,
    PermissionEvaluator,
)
from localharness.config.models import PermissionConfig
from localharness.core.types import ToolCall
from localharness.memory.sqlite import FactQuery, MemoryStore, archive_stamp
from localharness.tools.builtin.memory_tools import MemoryGetTool, MemorySearchTool
from localharness.tools.builtin.read_tool import ReadTool

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"


def _denied(name: str, arguments: dict):
    return PermissionEvaluator().evaluate(
        ToolCall(name=name, arguments=arguments), PermissionConfig()
    )


# ---------------------------------------------------------------------------
# 1. The exec surfaces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    # The straight read the owner named — a leftover store, opened with a sqlite client.
    "sqlite3 ~/.localharness/agents/orchestrator/memory.db 'select * from facts'",
    "sqlite3 /home/u/.localharness/agents/orchestrator/memory.db .dump",
    # Every SQLite sibling and the backup shapes, in one pattern.
    "strings /old/backup/memory.db-wal | head -50",
    "cat ./.localharness/agents/x/memory.db-shm",
    "cp ~/.localharness/agents/orchestrator/memory.db /tmp/x.db",
    "grep -a Denver memory.db.bak",
    "tar xzf old.tgz && sqlite3 restored/memory.db 'select value from facts'",
    # The cold archive by TABLE name — catches a store someone copied to another filename.
    "sqlite3 /tmp/copied.db 'select * from facts_archive'",
    "sqlite3 ~/store/memory-archive.db .tables",
    # The owner-only CLI, which would otherwise be a way back into the archive.
    "localharness memory list --archived",
    "localharness memory restore 803",
    ".venv/bin/localharness memory show profile/home",
    "cd /tmp && uv run localharness memory list",
])
def test_exec_refuses_every_route_to_a_store(command: str):
    """MUTATION TARGET: drop `bash_exec(*memory.db*)` and the first seven of these redden."""
    result = _denied("bash_exec", {"command": command})
    assert result.denied, command


@pytest.mark.parametrize("code", [
    "import sqlite3; c = sqlite3.connect('~/.localharness/agents/x/memory.db')",
    "open('/tmp/old/memory.db-wal','rb').read()",
    "cur.execute('select * from facts_archive')",
])
def test_python_exec_is_covered_too(code: str):
    """MUTATION TARGET: drop the python_exec patterns and these redden. python_exec has no
    sandbox — a bash-only blacklist is one `import sqlite3` from useless."""
    assert _denied("python_exec", {"code": code}).denied, code


@pytest.mark.parametrize("command", [
    # Ordinary work that merely says "memory" — must stay allowed.
    "pytest tests/unit/test_memory_store.py -q",
    "git commit -m 'fix a memory leak in the cruncher cache'",
    "grep -rn 'memory_search' src/localharness/tools/",
    "python -c \"print('in-memory cache warmed')\"",
    "free -h && cat /proc/meminfo",
    "ls ~/.localharness/agents/orchestrator/",
    "localharness start --agent orchestrator",
    "localharness components set agent.memory.archival.enabled true",
])
def test_ordinary_work_is_untouched(command: str):
    """The patterns are anchored on artifact names and verb forms, not on the word 'memory'.
    A blacklist that stopped normal work would be turned off, and then it protects nothing."""
    assert not _denied("bash_exec", {"command": command}).denied, command


def test_the_refusal_tells_the_agent_what_to_use_instead():
    """A refusal without an alternative is a retry loop: the model knows it is blocked but not
    what the supported route is, so it tries the blocked one again."""
    reason = _denied("bash_exec", {"command": "sqlite3 memory.db .dump"}).reason
    assert "bash_exec(*memory.db*)" in reason            # which rule fired
    assert MEMORY_STORE_DENY_GUIDANCE in reason          # and what to do instead
    for tool in ("memory_search", "memory_get", "remember"):
        assert tool in reason
    # A denial with no better route keeps the bare reason — the guidance is not boilerplate.
    assert MEMORY_STORE_DENY_GUIDANCE not in _denied(
        "bash_exec", {"command": "sudo rm -rf /"}).reason


def test_the_patterns_ship_as_defaults_under_a_bumped_revision():
    """DEFAULT tier: on by default, and folded into an existing user config by the additive
    union migration (which keeps the user's own entries and never resurrects deletions)."""
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION

    shipped = PermissionConfig().deny_patterns
    for artifact in ("memory.db", "facts_archive", "memory-archive", "localharness memory "):
        assert f"bash_exec(*{artifact}*)" in shipped
        assert f"python_exec(*{artifact}*)" in shipped
    assert CURRENT_DEFAULTS_REVISION >= 3      # the bump that carries them to existing configs


def test_read_no_longer_teaches_the_bypass_for_a_store(tmp_path: Path):
    """`read` refused a .db already — but its hint said "use bash_exec with sqlite3", handing
    the model a recipe for the exact route the deny patterns exist to close."""
    async def go():
        store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                            base_dir=str(tmp_path))
        await store.open()
        try:
            await store.store_fact("profile/home", "the owner lives in Denver",
                                   source="remember")
        finally:
            await store.close()
        db = tmp_path / "agents" / "orchestrator" / "memory.db"
        store_err = (await ReadTool()._execute(path=str(db))).error
        other = tmp_path / "other.bin"
        other.write_bytes(b"\x00\x01binary but not a store\x00")
        other_err = (await ReadTool()._execute(path=str(other))).error
        return store_err, other_err

    store_err, other_err = asyncio.run(go())
    assert "sqlite3" not in store_err                  # the bypass recipe is gone
    assert "memory_search" in store_err and "memory_get" in store_err
    assert "sqlite3" in other_err                      # …only for stores; other binaries keep it


# ---------------------------------------------------------------------------
# 2. Structural invisibility of the archive
# ---------------------------------------------------------------------------

# The table as a SQL OBJECT — `FROM facts_archive`, `INTO facts_archive`, `ON facts_archive(`
# — not the word in prose. A bare keyword scan flagged this file's own deny-pattern
# documentation ("the facts_archive table") and missed nothing real; requiring the clause that
# actually references a table is what separates a query from a sentence about one.
_ARCHIVE_SQL_REF = re.compile(
    r"\b(?:FROM|INTO|UPDATE|JOIN|ON|TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?facts_archive\b",
    re.IGNORECASE,
)

# The ONLY places allowed to name facts_archive in SQL: the owner verbs and the migration that
# creates it. A read path added anywhere else shows up here as a new entry, and this test is
# the conversation about whether it should exist.
_ALLOWED_ARCHIVE_SQL_SITES = {
    ("memory/sqlite.py", "<module>"),       # MIGRATION_V8_TO_V9_SQL (creates it + its indexes)
    ("memory/sqlite.py", "archive_fact"),   # owner/consolidation verb: move in
    ("memory/sqlite.py", "restore_fact"),   # owner verb: move back
    ("memory/sqlite.py", "list_archived"),  # owner verb: `memory list --archived`
    ("memory/sqlite.py", "count_archived"), # owner verb: the count
}


def _archive_sql_sites() -> set[tuple[str, str]]:
    """Every (module, enclosing function) that puts `facts_archive` in a SQL string literal.

    AST, not grep, for two reasons: comments and prose are invisible to it (the deny patterns
    and design notes mention the table constantly), and a literal is attributable to the
    function that would execute it.
    """
    sites: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        docstrings = {
            node.body[0].value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        scope: list[str] = []

        def visit(node: ast.AST) -> None:
            is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_func:
                scope.append(node.name)
            for child in ast.iter_child_nodes(node):
                if (isinstance(child, ast.Constant) and isinstance(child.value, str)
                        and child not in docstrings
                        and _ARCHIVE_SQL_REF.search(child.value)):
                    rel = str(path.relative_to(SRC)).replace("\\", "/")
                    sites.add((rel, scope[-1] if scope else "<module>"))
                visit(child)
            if is_func:
                scope.pop()

        visit(tree)
    return sites


def test_no_agent_facing_path_queries_the_archive():
    """MUTATION TARGET: add a `SELECT … FROM facts_archive` to any agent-facing function and
    this reddens. The cold archive is reachable by owner verbs and nothing else."""
    sites = _archive_sql_sites()
    assert sites == _ALLOWED_ARCHIVE_SQL_SITES, (
        "a new code path queries facts_archive — if it is agent-facing, the archive just "
        f"stopped being invisible.\nunexpected: {sorted(sites - _ALLOWED_ARCHIVE_SQL_SITES)}\n"
        f"gone: {sorted(_ALLOWED_ARCHIVE_SQL_SITES - sites)}"
    )


def test_the_agent_facing_modules_never_name_the_archive_in_sql():
    """Said the other way round, against the modules the agent's own verbs run through."""
    agent_facing = {
        "tools/builtin/memory_tools.py",   # memory_search / memory_get / remember
        "memory/hierarchy.py", "memory/clustering.py", "memory/router.py",
        "memory/markdown.py", "memory/mining.py", "memory/discovery.py",
        "memory/consolidation.py", "memory/reconciliation.py", "memory/salience.py",
    }
    named = {module for module, _ in _archive_sql_sites()}
    assert not (named & agent_facing), named & agent_facing


# ---------------------------------------------------------------------------
# 3. …and the behaviour the structure is there to guarantee
# ---------------------------------------------------------------------------

def test_every_agent_facing_read_path_goes_blind_when_a_fact_is_archived(tmp_path: Path):
    """The structural test says no path NAMES the archive; this one says the paths behave
    that way end to end — search, get, the ambient shelf, and the tag graph."""
    async def go():
        store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                            base_dir=str(tmp_path))
        await store.open()
        try:
            fact = await store.store_fact(
                "ops/vllm-port", "the vllm server listens on port 8081",
                tags=["ops"], confidence=0.9, source="remember",
            )
            search, get = MemorySearchTool(store), MemoryGetTool(store)

            async def visible() -> dict[str, bool]:
                ctx = await store.load_context()
                tag = await store._get_tag_row("ops")
                atoms = await store.atoms_for_tag(tag.id) if tag else []
                return {
                    "memory_search": "ops/vllm-port" in (
                        await search._execute(query="vllm")).output,
                    "memory_get": (await get._execute(name="ops/vllm-port")).success,
                    "ambient": "ops/vllm-port" in (ctx.agent_memory_md or ""),
                    "tag_graph": any(a.key == "ops/vllm-port" for a in atoms),
                    "query_facts": bool(await store.query_facts(
                        FactQuery(text="vllm", min_confidence=0.0))),
                }

            before = await visible()
            # Reading it STAGED reads, and the store refuses to archive a row read since the
            # last fold — the rail firing exactly as designed. Fold first, as a pass would.
            assert await store.archive_fact(fact.id, surface="test", s_at_archive=-1.0,
                                            line_at_archive=None) is False
            await store.fold_staged_access()
            assert await store.archive_fact(fact.id, surface="test", s_at_archive=-1.0,
                                            line_at_archive=None) is True
            after = await visible()
            await store.restore_fact(fact.id)
            restored = await visible()
            return before, after, restored
        finally:
            await store.close()

    before, after, restored = asyncio.run(go())
    assert before["memory_search"] and before["memory_get"] and before["query_facts"]
    assert not any(after.values()), after        # EVERY agent-facing path goes blind
    assert restored["memory_search"] and restored["memory_get"] and restored["query_facts"]


def test_the_archive_stamp_says_which_surface_condemned_the_row():
    """Not access control — provenance. Whatever moved a row is on the row, forever."""
    assert archive_stamp("consensus-list", 1789769433) == "archived@1789769433;consensus-list"
