"""`/memory` — the user's window into the agent's persistent memory, navigated by the TAG HIERARCHY.

Pure and MODEL-FREE: every subcommand reads the store (the same reads the memory tools use, so it is
WAL-safe while a live turn writes) and returns plain text. The terminal channel renders system
messages as plain escaped text (TerminalChannel.send_message with agent_id=None), so this reads the
same in classic, non-TTY, and box mode — no rich objects, matching how `/model` renders.

Subcommands (dispatch routes on the first word):
  /memory                      overview: buckets -> children (+ proposed candidates) + recent feed
  /memory <bucket>[/<child>]   list memories under a tag path, newest first, paged (~20/page)
  /memory <child>              bare child name, resolved when unambiguous (tag names are unique)
  /memory show <id>            full detail + supersede chain + ambient-eligibility teaching line
  /memory forget <id>          preview; /memory forget <id> confirm retires it (supersede, never delete)
  /memory search <words>       the deterministic search path (query_facts), top hits with ids

Design notes:
- forget confirm is a two-step COMMAND form (`... <id> confirm`), not a y/n prompt: it is stateless
  and identical in classic + box mode (a mid-turn slash command queues and runs between turns, where
  there is no line to block on). The preview shows exactly what will be retired.
- browsing here deliberately does NOT touch_staged / record activation traces — this is user
  inspection, not the model's retrieval; polluting ranking with it would be wrong.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from rich.console import Group
from rich.text import Text
from rich.tree import Tree

from localharness.memory.sqlite import (
    AMBIENT_INJECTION_FLOOR,
    USER_FORGET_PROVENANCE_PREFIX,
    FactQuery,
)

_PAGE = 20        # listing rows per page
_SEARCH = 10      # search hits
_CLIP = 72        # value clip width in list/search rows
_LEAVES = 5       # overview: most-recent memories shown per child, as tree leaves
_CHAIN_CLIP = 56  # value clip width in supersede-chain nodes


# --------------------------------------------------------------------------- dispatch
async def dispatch(store: Any, arg: str) -> str:
    """Route a `/memory` argument string to a subcommand renderer. `store` is an opened MemoryStore
    (or None → unavailable). Returns plain text; never raises for user input."""
    if store is None:
        return "Memory isn't available in this session (running without a persistent store)."
    arg = (arg or "").strip()
    if not arg:
        return await render_overview(store)
    head, _, rest = arg.partition(" ")
    rest = rest.strip()
    sub = head.lower()
    if sub == "show":
        return await render_show(store, rest)
    if sub == "forget":
        return await render_forget(store, rest)
    if sub == "search":
        return await render_search(store, rest)
    # Anything else is a tag path (bucket, bucket/child, or bare child), optional trailing page.
    return await render_listing(store, arg)


# --------------------------------------------------------------------------- overview
async def render_overview(store: Any) -> Any:
    """The tag hierarchy as a rich Tree: store -> buckets(count) -> children(count) -> the ~5 most
    recent memories per child as leaves. A dim `proposed` branch carries discovery candidates.
    Empty branches are shown dim, not hidden. Returns a str only when the store is empty."""
    buckets = await store.buckets()
    if not buckets:
        return "No memory buckets yet — the store is empty."
    proposed_all = [t for t in await store.list_tags(status="proposed") if t.parent_id is not None]
    tree = Tree(Text("Memory", style="bold"), guide_style="dim")
    for b in buckets:
        b_atoms = await store.atoms_for_tag(b.id)
        b_node = tree.add(Text.assemble(
            (f"{b.name}  ", "bold" if b_atoms else "bold dim"), (f"({len(b_atoms)})", "dim")))
        for c in await store.active_children(b.id):
            c_atoms = await store.atoms_for_tag(c.id)
            c_node = b_node.add(Text.assemble(
                (f"{c.name}  ", "" if c_atoms else "dim"), (f"({len(c_atoms)})", "dim")))
            c_atoms.sort(key=lambda f: (f.updated_at, f.id), reverse=True)
            for f in c_atoms[:_LEAVES]:
                c_node.add(_leaf(f))
        proposed = [t for t in proposed_all if t.parent_id == b.id]
        if proposed:
            names = ", ".join(t.name for t in proposed)
            b_node.add(Text(f"proposed (discovery candidates): {names}  ({len(proposed)})",
                            style="dim italic"))
    tree.add(Text("browse: /memory <bucket>[/<child>]   detail: /memory show <id>   "
                  "find: /memory search <words>", style="dim"))
    return tree


def _leaf(f: Any) -> Text:
    """A recent-memory leaf: id (cyan) + clipped value + dim confidence."""
    return Text.assemble(
        (f"#{f.id}  ", "cyan"), _clip(f.value, _CLIP), (f"  conf {f.confidence:.2f}", "dim"))


# --------------------------------------------------------------------------- listing
async def render_listing(store: Any, arg: str) -> str:
    parts = arg.split()
    page = 1
    if len(parts) >= 2 and parts[-1].isdigit():
        page = max(1, int(parts[-1]))
        path = " ".join(parts[:-1])
    else:
        path = arg.strip()
    tag = await _resolve_tag(store, path)
    if tag is None:
        return (f"Unknown tag path: {path!r}. Run /memory to see the buckets and children, "
                "or /memory search <words>.")
    full = await _full_path(store, tag)
    facts = await store.atoms_for_tag(tag.id)
    facts.sort(key=lambda f: (f.updated_at, f.id), reverse=True)
    total = len(facts)
    if total == 0:
        return f"{full} — no memories filed here yet."
    start = (page - 1) * _PAGE
    if start >= total:
        return f"{full} has {total} memories; page {page} is past the end."
    chunk = facts[start:start + _PAGE]
    noun = "memory" if total == 1 else "memories"
    lines = [f"{full}  ({total} {noun}, page {page}):"]
    for f in chunk:
        lines.append(f"  #{f.id}  {_clip(f.value, _CLIP)}  conf {f.confidence:.2f} · {_age(f.updated_at)}")
    shown = start + len(chunk)
    if shown < total:
        lines.append(f"  … {total - shown} more — /memory {path} {page + 1}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- show
async def render_show(store: Any, arg: str) -> Any:
    """Full detail for one memory + its supersede chain as a mini-tree. Returns a str only for the
    usage / not-found errors; otherwise a Group(detail fields, chain mini-tree)."""
    fid = _parse_id(arg)
    if fid is None:
        return f"Usage: /memory show <id>  (an id number, e.g. /memory show 12). Got {arg!r}."
    fact = await store.get_fact_by_id(fid)
    if fact is None:
        return f"No memory with id {fid}. Browse with /memory or find one with /memory search <words>."
    tags = await store.tags_for_atom(fid)
    paths = _paths(tags)
    detail = Text()
    detail.append(f"Memory #{fact.id}\n", style="bold")
    detail.append(f"  value:       {fact.value}\n")
    detail.append(f"  key:         {fact.key}\n")
    detail.append(f"  tags:        {', '.join(paths) if paths else '(untagged)'}\n")
    detail.append(f"  confidence:  {fact.confidence:.2f}\n")
    detail.append(f"  status:      {fact.status}\n")
    detail.append(f"  source:      {fact.source or '(unknown)'}\n")
    detail.append(f"  created:     {_stamp(fact.created_at)}\n")
    detail.append(f"  updated:     {_stamp(fact.updated_at)}\n")
    detail.append(f"  provenance:  {fact.provenance or '(none)'}")
    if fact.status != "active" and fact.provenance.startswith(USER_FORGET_PROVENANCE_PREFIX):
        detail.append("\n  forgotten:   retired by you (user forget) — kept in history for audit",
                      style="dim")
    detail.append(f"\n  {_ambient_line(fact)}", style="dim")
    return Group(detail, Text(), await _chain_tree(store, fact))


def _ambient_line(fact: Any) -> str:
    """The teaching line: why this memory does/doesn't inject into the model's every-turn shelf."""
    floor = AMBIENT_INJECTION_FLOOR
    if fact.status != "active":
        return "ambient-eligible: no — retired (superseded rows never inject)"
    if fact.confidence >= floor:
        return (f"ambient-eligible: yes — confidence {fact.confidence:.2f} >= {floor:.2f} floor "
                "(injected into the model's memory shelf every turn)")
    return (f"ambient-eligible: no — confidence {fact.confidence:.2f} < {floor:.2f} floor "
            "(searchable, but not auto-injected into prompts)")


async def _chain_tree(store: Any, fact: Any) -> Tree:
    """The supersede chain as a nested mini-tree: oldest at the top, each newer version nested
    beneath the one it replaced (ancestors above, descendants below), the current one highlighted.
    History is per-key. A lone version reads as an untouched original."""
    history = await store.get_fact_history(fact.key)
    ordered = sorted(history, key=lambda h: (h.created_at, h.id))
    root = Tree(Text("supersede chain", style="bold"), guide_style="dim")
    node = root
    for h in ordered:
        if h.id == fact.id:
            label = Text.assemble(
                (f"#{h.id}  ", "bold green"),
                (_clip(h.value, _CHAIN_CLIP), "bold green"),
                ("   ← current", "bold green"))
        else:
            label = Text.assemble((f"#{h.id}  ", "cyan"), (_clip(h.value, _CHAIN_CLIP), "dim"))
        node = node.add(label)
    if len(ordered) <= 1:
        node.add(Text("original — nothing replaced it, nothing it replaced", style="dim italic"))
    return root


# --------------------------------------------------------------------------- forget
async def render_forget(store: Any, arg: str) -> str:
    parts = arg.split()
    confirmed = len(parts) >= 2 and parts[-1].lower() == "confirm"
    fid = _parse_id(parts[0]) if parts else None
    if fid is None:
        return ("Usage: /memory forget <id>  — shows a preview, then /memory forget <id> confirm "
                "retires it (kept in history, never hard-deleted).")
    fact = await store.get_fact_by_id(fid)
    if fact is None:
        return f"No memory with id {fid} — nothing to forget."
    if fact.status != "active":
        return (f"Memory #{fid} is already retired (status: {fact.status}) — nothing to forget. "
                f"/memory show {fid} shows its history.")
    if not confirmed:
        path = await _tag_path(store, fact)
        return (f"About to forget memory #{fid}:\n"
                f"  {_clip(fact.value, 100)}  [{path}] conf {fact.confidence:.2f}\n"
                "This retires it — removed from the model's memory shelf and from search, but kept "
                "in history (never hard-deleted).\n"
                f"Confirm with:  /memory forget {fid} confirm")
    if not await store.forget_fact(fid):
        return (f"Memory #{fid} changed under you (a live turn just superseded it) — nothing "
                f"retired. Re-check with /memory show {fid}.")
    return (f"Forgotten. Memory #{fid} retired — it no longer injects into prompts or shows in "
            f"listings/search. History kept: /memory show {fid}.")


# --------------------------------------------------------------------------- search
async def render_search(store: Any, arg: str) -> str:
    q = (arg or "").strip()
    if not q:
        return "Usage: /memory search <words>."
    facts = await store.query_facts(FactQuery(text=q, min_confidence=0.0, limit=_SEARCH))
    if not facts:
        return f"No memories matched {q!r}. (/memory browses by tag; /memory forget <id> retires one.)"
    lines = [f"Search {q!r} — top {len(facts)}:"]
    for f in facts:
        path = await _tag_path(store, f)
        lines.append(f"  #{f.id}  {_clip(f.value, _CLIP)}  [{path}] conf {f.confidence:.2f}")
    lines.append("Detail: /memory show <id>   retire: /memory forget <id>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- helpers
async def _resolve_tag(store: Any, path: str) -> Any:
    """Resolve a tag path to a Tag row. `bucket/child` validates the parent; a bare token resolves
    as a bucket or a child (tag names are unique per agent, so a bare child is unambiguous)."""
    path = (path or "").strip().strip("/")
    if not path:
        return None
    if "/" in path:
        bname, _, cname = path.partition("/")
        bucket = await store.get_tag(bname.strip())
        if bucket is None or bucket.parent_id is not None:
            return None
        child = await store.get_tag(cname.strip())
        if child is None or child.parent_id != bucket.id:
            return None
        return child
    return await store.get_tag(path)


async def _full_path(store: Any, tag: Any) -> str:
    if tag.parent_id is None:
        return tag.name
    parent = await store.get_tag_by_id(tag.parent_id)
    return f"{parent.name}/{tag.name}" if parent else tag.name


async def _tag_path(store: Any, fact: Any) -> str:
    """Compact single path for a fact (bucket/first-child), for list/search/overview rows."""
    tags = await store.tags_for_atom(fact.id)
    bucket = next((t.name for t in tags if t.parent_id is None), "")
    children = [t.name for t in tags if t.parent_id is not None]
    if bucket and children:
        return f"{bucket}/{children[0]}" + ("…" if len(children) > 1 else "")
    if bucket:
        return bucket
    if children:
        return children[0]
    return "untagged"


def _paths(tags: list) -> list[str]:
    """Full paths for the show view (bucket/child for each child; bucket alone if childless)."""
    bucket = next((t.name for t in tags if t.parent_id is None), "")
    children = [t.name for t in tags if t.parent_id is not None]
    if bucket and children:
        return [f"{bucket}/{c}" for c in children]
    if bucket:
        return [bucket]
    return children


def _parse_id(tok: str) -> int | None:
    tok = (tok or "").strip().lstrip("#")
    return int(tok) if tok.isdigit() else None


def _clip(value: str, n: int) -> str:
    s = " ".join((value or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _age(epoch: int) -> str:
    if not epoch:
        return "?"
    d = max(0, int(time.time()) - int(epoch))
    if d < 3600:
        return f"{d // 60}m"
    if d < 86400:
        return f"{d // 3600}h"
    return f"{d // 86400}d"


def _stamp(epoch: int) -> str:
    if not epoch:
        return "(unknown)"
    return datetime.fromtimestamp(int(epoch)).astimezone().strftime("%Y-%m-%d %H:%M")
