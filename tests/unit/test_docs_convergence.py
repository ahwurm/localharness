"""The public docs say what the code does — checked against the code, not promised.

A docs change cannot be "done" because prose was edited. Each check below is a pure function
`check_*(...) -> list[str]` (one string per disagreement) that reads a document AND the code object
it describes, and every source of truth is imported, never typed out: `BUILTIN_PLUGINS`,
`PLUGIN_API_VERSION`, `PluginManifest.model_fields`, `dataclasses.fields(PluginContext)`,
`inspect.signature(AgentLoop.__init__)`, `global_only_paths`, `ASK_GLOBAL_ONLY_FIELDS`, the
`ToolSchema` declaration defaults, `plugin_summary`. Each check has a live test on the real file and
a "bites" test that feeds it one deliberate break and asserts a non-empty violation list, so no
check is vacuous.

A live test marked `xfail(strict=True)` is a doc still known to be false; its reason names the work
that fixes it, and an early green is an XPASS failure, so the marker must be removed with the fix.

These tests never invoke the CLI (Typer caches the terminal width at first import) and read plugins
from `BUILTIN_PLUGINS` only, never discovery (a dev venv may have other plugins installed).
"""
from __future__ import annotations

import ast
import builtins
import dataclasses
import functools
import inspect
import keyword
import re
import typing
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from localharness.agent.loop import AgentLoop
from localharness.config.loader import ASK_GLOBAL_ONLY_FIELDS
from localharness.config.plugin_sections import global_only_paths
from localharness.dispatch.config import DiscordSettings, env_fallback
from localharness.plugins import api
from localharness.plugins.api import (
    Check, MemoryBrowse, MemorySlotPlugin, Plugin, PluginContext, PluginManifest, PluginPaths,
    SessionInfo, SetupField, plugin_summary,
)
from localharness.plugins.builtin import BUILTIN_PLUGINS
from localharness.plugins.plan import PlanState
from localharness.tools.base import ToolSchema
from tests.unit.test_release_docs import _BLOB, _REPO, _github_anchors

SRC = _REPO / "src" / "localharness"
_TREE = _BLOB.replace("/blob/", "/tree/")
SPEC00 = "docs/specs/00-architecture-overview.md"
SPEC09 = "docs/specs/09-hooks-plugins.md"


def xfail_until(plan: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"docs not yet converged — turned green by {plan}")


# ---------------------------------------------------------------- helpers: read docs as structure

def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def _fence(line: str) -> bool:
    return line.lstrip().startswith(("```", "~~~"))


def unfenced_lines(text: str) -> list[str]:
    """The text's lines with every fenced-code line (fences included) blanked; positions kept."""
    out, fenced = [], False
    for line in text.splitlines():
        if _fence(line):
            fenced = not fenced
            out.append("")
        else:
            out.append("" if fenced else line)
    return out


def _heading(line: str) -> tuple[int, str] | None:
    m = re.match(r"(#{1,6}) (.+)", line)
    return (len(m.group(1)), m.group(2).strip()) if m else None


def section(text: str, heading: str, *, prefix: bool = False) -> str | None:
    """The body under `heading` (a whole heading line such as "## Plugins", outside code fences)
    up to the next heading of the same or a higher level; None when the heading is absent. With
    `prefix`, the heading line need only start with `heading` (for "## 5." style numbering)."""
    raw, plain = text.splitlines(), unfenced_lines(text)
    level = len(heading) - len(heading.lstrip("#"))
    for i, line in enumerate(plain):
        if line.strip() == heading or (prefix and line.startswith(heading)):
            for j in range(i + 1, len(plain)):
                h = _heading(plain[j])
                if h and h[0] <= level:
                    return "\n".join(raw[i + 1:j])
            return "\n".join(raw[i + 1:])
    return None


def ticks(text: str) -> list[str]:
    """Every single-backtick code span outside fenced code, in order."""
    return [m.group(1) for line in unfenced_lines(text)
            for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", line)]


def tables(text: str) -> list[list[list[str]]]:
    """Each markdown table outside fenced code as rows of stripped cells (separator rows dropped)."""
    out: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*unfenced_lines(text), ""]:
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                current.append(cells)
        elif current:
            out.append(current)
            current = []
    return out


def table_rows(section_text: str) -> list[list[str]]:
    """The rows of the first table in `section_text` (header first), or []."""
    found = tables(section_text)
    return found[0] if found else []


def bullets(text: str) -> list[tuple[str, str]]:
    """(nearest heading, bullet text with its continuation lines) for every `- ` list item."""
    out: list[tuple[str, str]] = []
    heading, current = "", None
    for line in [*unfenced_lines(text), ""]:
        if (h := _heading(line)) is not None:
            heading = h[1]
        if line.startswith("- ") or (h is not None) or not line.strip() or not line.startswith(" "):
            if current is not None:
                out.append(current)
                current = None
        if line.startswith("- "):
            current = (heading, line)
        elif current is not None and line.startswith(" ") and line.strip():
            current = (current[0], current[1] + "\n" + line)
    return out


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", "\n".join(unfenced_lines(text))) if p.strip()]


# ---------------------------------------------------------------- the code's own index of names

@functools.cache
def _py_files() -> tuple[tuple[Path, ast.Module], ...]:
    return tuple((p, ast.parse(p.read_text(encoding="utf-8"))) for p in sorted(SRC.rglob("*.py")))


@functools.cache
def symbol_index() -> frozenset[str]:
    """Every name src/localharness defines: modules and packages, classes, functions, assigned
    names, arguments and attribute stores (`self.x = ...`)."""
    names: set[str] = set()
    for path, tree in _py_files():
        names.add(path.stem)
        names.update(part for part in path.relative_to(SRC).parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                names.add(node.attr)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
    return frozenset(names)


@functools.cache
def literal_index() -> frozenset[str]:
    """Every identifier-shaped string constant in src/localharness (Literal values such as `dev` and
    `warn`, command words such as `enable`), plus Python's builtins and keywords."""
    found = {node.value for _p, tree in _py_files() for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and re.fullmatch(r"[A-Za-z_][\w-]*", node.value)}
    return frozenset(found | set(dir(builtins)) | set(keyword.kwlist))


@functools.cache
def class_index() -> dict[str, list[tuple[set[str], dict[str, str], list[str]]]]:
    """Class name -> per definition: (member names, annotation text per annotated member, base
    class names). Read with `ast`, so a class is resolved without importing optional extras."""
    out: dict[str, list[tuple[set[str], dict[str, str], list[str]]]] = {}
    for _path, tree in _py_files():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            members: set[str] = set()
            annotations: dict[str, str] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
                    annotations[item.target.id] = ast.unparse(item.annotation)
                elif isinstance(item, ast.Assign):
                    members.update(t.id for t in item.targets if isinstance(t, ast.Name))
            for fn in node.body:  # instance attributes set in methods (self.x = ...)
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.update(n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
                                   and isinstance(n.ctx, ast.Store)
                                   and isinstance(n.value, ast.Name) and n.value.id == "self")
            bases = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute)
                     else "" for b in node.bases]
            out.setdefault(node.name, []).append((members, annotations, bases))
    return out


def _member_type(cls: str, attr: str, _seen: frozenset[str] = frozenset()) -> str | None | bool:
    """False if no definition of `cls` (or its bases) has `attr`; else the class name its
    annotation names (first CamelCase word), or None when that is unknown."""
    if cls in _seen or cls not in class_index():
        return False
    for members, anns, bases in class_index()[cls]:
        if attr in members:
            m = re.search(r"\b([A-Z]\w*)", anns.get(attr, ""))
            return m.group(1) if m and m.group(1) in class_index() else None
        for base in bases:
            found = _member_type(base, attr, _seen | {cls})
            if found is not False:
                return found
    return False


def _module_file(parts: list[str]) -> tuple[Path | None, list[str]]:
    """The longest `localharness.a.b` prefix that is a module or package, and the rest."""
    for n in range(len(parts), 1, -1):
        base = SRC.joinpath(*parts[1:n])
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate, parts[n:]
    return (SRC / "__init__.py", parts[1:]) if parts[0] == "localharness" else (None, parts)


# ---------------------------------------------------------------- check 1: spec 09 -> code

ALLOWLIST = frozenset({  # reviewed: spec 09 names these, and none is a LocalHarness symbol
    # Files a user edits or a 0.15 plugin shipped (dotted, but not Python names).
    "config.yaml", "overrides.yaml", "manifest.yaml",
    # Library names: pluggy and its `PluginManager`, Typer's app class, pydantic's `Field`.
    "pluggy", "PluginManager", "typer.Typer", "Field",
    # Prose word for the hook specification (the code class is `HarnesHookSpec`).
    "HookSpec",
    # The removed 0.15 module the "Plugins written for 0.15" section names (pinned gone below).
    "localharness.hooks",
})
LEGACY_HOOK_NAMES = ("on_agent_start", "on_agent_end", "on_event")
_MIME = re.compile(r"(?:image|text|application|audio|video)/[\w.+-]+")
_IDENT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_PATHISH = re.compile(r"[\w.\-]+(?:/[\w.\-]+)*/?")


def _resolve_path(token: str) -> bool:
    return any((root / token).exists() for root in (SRC, _REPO, _REPO / "examples", SRC.parent))


def _resolve_dotted(parts: list[str]) -> str | None:
    """None when `a.b.c` resolves, else why not."""
    if parts[0] == "localharness":
        file, rest = _module_file(parts)
        if file is None:
            return "no such module"
        if not rest:
            return None
        top = {n for _p, t in _py_files() if _p == file for n in _top_names(t)}
        if rest[0] not in top:
            return f"{file.relative_to(SRC).as_posix()} defines no {rest[0]}"
        parts = rest
    head, *tail = parts
    if head == "ctx":
        head = "PluginContext"
    if re.match(r"[A-Z]\w*[a-z]", head) and head in class_index():
        cls: str | None = head
        for attr in tail:
            if cls is None:
                return None
            found = _member_type(cls, attr)
            if found is False:
                return f"{cls} has no {attr}"
            cls = found  # type: ignore[assignment]
        return None
    if len(parts) > 1 and re.match(r"[A-Z]\w*[a-z]", head) and head not in symbol_index():
        return f"no class {head} in src/localharness"
    if parts[-1] in symbol_index() or (len(parts) == 1 and parts[0] in literal_index()):
        return None
    return f"no symbol {parts[-1]} in src/localharness"


def _top_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def check_spec09_names_resolve(text: str) -> list[str]:
    """Every identifier-shaped backtick span outside fenced code names something real: a `*.py`
    path exists, `Class.attr` is a member of THAT class (never matched by last segment alone),
    `ctx.x` is a PluginContext field, `localharness.x.y` is a module (and attribute), and any other
    name is defined somewhere in src/localharness. The 0.15 hook names must stay gone."""
    out: list[str] = []
    live = [n for n in LEGACY_HOOK_NAMES if n in symbol_index()]
    out += [f"`{n}`: the 0.15 hook name is defined in src again" for n in live]
    if (SRC / "hooks.py").exists() or (SRC / "hooks").is_dir():
        out.append("`localharness.hooks`: the 0.15 module exists again")
    for raw in dict.fromkeys(ticks(text)):
        token = re.sub(r"\(.*\)$", "", raw.strip())
        if token in ALLOWLIST or token in LEGACY_HOOK_NAMES or not token:
            continue
        if _MIME.fullmatch(token):
            continue
        if "/" in token or token.endswith(".py"):
            if _PATHISH.fullmatch(token) and not token.startswith(("/", ".", "~")) \
                    and (token.endswith(".py") or "/" in token.strip("/")) and not _resolve_path(token):
                out.append(f"`{raw}`: no such file under src/localharness, examples/ or the repo")
            continue
        if not _IDENT.fullmatch(token):
            continue
        if (why := _resolve_dotted(token.split("."))) is not None:
            out.append(f"`{raw}`: {why}")
    return out


def test_spec09_names_resolve_live():
    """Spec 09 names only identifiers, fields and files that exist."""
    assert check_spec09_names_resolve(_read(SPEC09)) == []


def test_spec09_names_resolve_bites():
    assert check_spec09_names_resolve("`PluginManifest.setup` and `plugins/plan.py`") == []
    found = check_spec09_names_resolve("`PluginManifest.stability` and `plugins/planner.py`")
    assert len(found) == 2, found
    assert check_spec09_names_resolve("`ctx.paths.artifact_dir` `ctx.nope`") == ["`ctx.nope`: PluginContext has no nope"]


# ---------------------------------------------------------------- check 2: code -> spec 09

def _public_methods(cls: type) -> list[str]:
    return [n for n, v in vars(cls).items() if callable(v) and not n.startswith("_")]


def coverage_names() -> dict[str, tuple[str, ...]]:
    """What spec 09 must name, read from the API module."""
    def fields(dc: type) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(dc))
    return {
        "PluginManifest": tuple(PluginManifest.model_fields),
        "PluginContext": fields(PluginContext),
        "PluginPaths": fields(PluginPaths),
        "SessionInfo": fields(SessionInfo),
        "SetupField": tuple(SetupField.model_fields),
        "Plugin": (*_public_methods(Plugin), "make_channel"),
        "MemorySlotPlugin": tuple(_public_methods(MemorySlotPlugin)),
        "MemoryBrowse": tuple(_public_methods(MemoryBrowse)),
        "PlanState": typing.get_args(PlanState),
        "Check.status": typing.get_args(Check.model_fields["status"].annotation),
        "types": ("MemoryWriteHandle", "ContextContribution", "ContextBudget", "Availability"),
    }


def check_spec09_covers_code(text: str, names: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Every name the plugin API defines appears in a backtick span of spec 09 — bare (`channels`,
    `channels()`, `"on"`) or qualified (`PluginManifest.channels`, `ctx.paths`)."""
    names = coverage_names() if names is None else names
    spans = {re.sub(r"\(.*\)$", "", t.strip()).strip("\"'") for t in ticks(text)}
    shown = spans | {s.rsplit(".", 1)[-1] for s in spans}
    return [f"{owner}: `{n}` is not named in spec 09"
            for owner, members in names.items() for n in members if n not in shown]


def test_spec09_covers_code_live():
    """Nothing the plugin API defines is missing from spec 09."""
    assert check_spec09_covers_code(_read(SPEC09)) == []


def test_spec09_covers_code_bites():
    names = coverage_names()
    good = " ".join(f"`{n}`" for members in names.values() for n in members)
    assert check_spec09_covers_code(good, names) == []
    found = check_spec09_covers_code(good.replace("`channels`", ""), names)
    assert any("`channels`" in v for v in found), found
    grown = {**names, "PluginManifest": (*names["PluginManifest"], "stability")}
    assert check_spec09_covers_code(good, grown) == ["PluginManifest: `stability` is not named in spec 09"]


# ---------------------------------------------------------------- check 2b: spec 09 status line

_STATUS = re.compile(r'^\*\*Stability:\*\* v1 — .*`PLUGIN_API_VERSION = "([^"]*)"`', re.M)


def check_spec09_status(text: str, version: str = api.PLUGIN_API_VERSION) -> list[str]:
    """Spec 09's status line is `**Stability:** v1 — `PLUGIN_API_VERSION = "<the code's value>"` ...`
    and the word UNSTABLE is gone."""
    out = []
    if (m := _STATUS.search(text)) is None:
        out.append('no line starting "**Stability:** v1 — " naming `PLUGIN_API_VERSION = "..."`')
    elif m.group(1) != version:
        out.append(f"status says PLUGIN_API_VERSION {m.group(1)!r}, the code says {version!r}")
    if "UNSTABLE" in text:
        out.append("spec 09 still says UNSTABLE")
    return out


def test_spec09_status_live():
    """Spec 09 states the API version the code carries, and no longer calls it unstable."""
    assert check_spec09_status(_read(SPEC09)) == []


def test_spec09_status_bites():
    good = f'**Stability:** v1 — `PLUGIN_API_VERSION = "{api.PLUGIN_API_VERSION}"`. Additive only.\n'
    assert check_spec09_status(good) == []
    assert check_spec09_status(good + "**Stability:** UNSTABLE (v1).\n")
    assert check_spec09_status(good, version="2")
    assert check_spec09_status("**Stability:** UNSTABLE (v1).\n")


# ---------------------------------------------------------------- check 3: machine-level-only list

MACHINE_ONLY_HEADING = "### Machine-level-only settings"
_DOTPATH = re.compile(r"(?:<name>|[a-z_][a-z0-9_]*)(?:\.[a-z_][a-z0-9_]*)+")
_FILE_SUFFIXES = (".yaml", ".yml", ".md", ".py", ".json", ".env", ".toml", ".txt")


def machine_only_code_set(plugins: Iterable[type[Plugin]] | None = None) -> frozenset[str]:
    """Every setting only the machine-level (global) config may set, computed from code: each
    bundled plugin's GLOBAL_ONLY fields, the `permissions.ask` keys a project may not set, and
    `<name>.enabled` for a plugin you install. (An AgentConfigModel may not mark GLOBAL_ONLY at all
    — the loader refuses such a plugin — so no `agent.` path can be machine-level only.)"""
    plugins = BUILTIN_PLUGINS if plugins is None else tuple(plugins)
    paths = {f"{P.manifest.name}.{p}" for P in plugins for p in global_only_paths(P.ConfigModel)}
    paths |= {f"permissions.ask.{f}" for f in ASK_GLOBAL_ONLY_FIELDS}
    return frozenset(paths | {"<name>.enabled"})


def check_machine_only_list(security_text: str, code_set: frozenset[str] | None = None) -> list[str]:
    """SECURITY.md's `### Machine-level-only settings` list equals the code's set, both ways. The
    list is the backticked dot-paths on the section's bullet lines (`- `); prose in the section —
    e.g. the tighten-only rules no code set enumerates — is not compared."""
    code = machine_only_code_set() if code_set is None else code_set
    body = section(security_text, MACHINE_ONLY_HEADING)
    if body is None:
        return [f"SECURITY.md has no '{MACHINE_ONLY_HEADING}' section"]
    doc = {t for _h, b in bullets(body) for t in ticks(b)
           if _DOTPATH.fullmatch(t) and not t.endswith(_FILE_SUFFIXES)}
    return ([f"machine-level only in code, not listed in SECURITY.md: {p}" for p in sorted(code - doc)]
            + [f"listed in SECURITY.md, not machine-level only in code: {p}" for p in sorted(doc - code)])


@xfail_until("51-04")
def test_machine_only_list_live():
    """SECURITY.md names every setting a project cannot set, exactly the code's set."""
    assert check_machine_only_list(_read("SECURITY.md")) == []


def test_machine_only_list_bites():
    code = machine_only_code_set()
    good = f"{MACHINE_ONLY_HEADING}\n\nA project cannot set these.\n\n" + "".join(
        f"- `{p}`\n" for p in sorted(code))
    assert check_machine_only_list(good, code) == []
    assert check_machine_only_list(good, code | {"image.timeout_s"}) == [
        "machine-level only in code, not listed in SECURITY.md: image.timeout_s"]
    dropped = good.replace("- `image.workflow`\n", "")
    assert check_machine_only_list(dropped, code) == [
        "machine-level only in code, not listed in SECURITY.md: image.workflow"]
    assert check_machine_only_list("## Something else\n", code)


# ---------------------------------------------------------------- check 4: three declaration readers

READERS = {"tools/capabilities.py": None, "agent/gate.py": "gate_family",
           "agent/context.py": "result_origin"}
DECLARATIONS = ("ingest", "host", "result_origin", "gate_family")


def declaration_defaults() -> dict[str, str]:
    """The fail-closed default of each tool declaration, from `ToolSchema` (None renders `none`)."""
    return {d: "none" if (v := ToolSchema.model_fields[d].default) is None else str(v)
            for d in DECLARATIONS}


def check_three_readers(security_text: str, defaults: dict[str, str] | None = None) -> list[str]:
    """SECURITY.md's paragraph that names `tools/capabilities.py` (the floor) also names
    `agent/gate.py` with `gate_family` and `agent/context.py` with `result_origin`; every file it
    names exists and each named identifier is read there; and it states each declaration's
    default as `<declaration>: <default>` equal to `ToolSchema`'s."""
    defaults = declaration_defaults() if defaults is None else defaults
    para = next((p for p in paragraphs(security_text) if "tools/capabilities.py" in ticks(p)), None)
    if para is None:
        return ["SECURITY.md has no paragraph naming `tools/capabilities.py`"]
    spans = ticks(para)
    out = [f"`{t}` does not exist under src/localharness" for t in spans
           if t.endswith(".py") and not (SRC / t).is_file()]
    for path, name in READERS.items():
        if path not in spans:
            out.append(f"the declarations paragraph does not name `{path}`")
        if name is not None:
            if name not in spans:
                out.append(f"the declarations paragraph does not name `{name}` (read by {path})")
            if (SRC / path).is_file() and name not in (SRC / path).read_text(encoding="utf-8"):
                out.append(f"{path} does not read `{name}`")
    stated = {m.group(1): m.group(2) for t in spans if (m := re.fullmatch(r"(\w+): (\S+)", t))}
    for d, want in defaults.items():
        if stated.get(d) != want:
            out.append(f"the declarations paragraph states `{d}: {stated.get(d)}`, "
                       f"ToolSchema's default is `{d}: {want}`")
    return out


@xfail_until("51-04")
def test_three_readers_live():
    """SECURITY.md names the floor, the gate and the context store by file, with the real defaults."""
    assert check_three_readers(_read("SECURITY.md")) == []


def _good_readers_paragraph(defaults: dict[str, str]) -> str:
    stated = ", ".join(f"`{d}: {v}`" for d, v in defaults.items())
    return ("The floor (`tools/capabilities.py`), the gate (`agent/gate.py`, `gate_family`) and the "
            f"context store (`agent/context.py`, `result_origin`) read one declaration: {stated}.\n")


def test_three_readers_bites():
    defaults = declaration_defaults()
    good = _good_readers_paragraph(defaults)
    assert check_three_readers(good) == []
    found = check_three_readers(good.replace("agent/gate.py", "agent/gates.py"))
    assert "`agent/gates.py` does not exist under src/localharness" in found, found
    assert check_three_readers(good, {**defaults, "ingest": "none"})
    assert check_three_readers("Nothing about the floor.\n")


# ---------------------------------------------------------------- check 5: the plugins table

TABLE_HEADER = ["Name", "What it does", "Default", "How to switch"]


def _norm(cell: str) -> str:
    return " ".join(cell.replace("`", "").split())


def check_plugins_table(md_text: str, heading: str = "## Plugins",
                        plugins: Iterable[type[Plugin]] | None = None) -> list[str]:
    """The table under `heading` (the one whose header is Name | What it does | Default | How to
    switch, else the first) has one row per bundled plugin in `BUILTIN_PLUGINS` order: What it
    does == `plugin_summary` (what `plugins list` prints); Default starts `on`/`off` and names
    `localharness[<extra>]` exactly when the plugin needs one; How to switch carries the command
    that changes the default state, and for a plugin off by default `--set <key>` for each of its
    setup keys."""
    plugins = BUILTIN_PLUGINS if plugins is None else tuple(plugins)
    body = section(md_text, heading)
    if body is None:
        return [f"no '{heading}' section"]
    found = tables(body)
    rows = next((t for t in found if t[0] == TABLE_HEADER), found[0] if found else [])
    if not rows:
        return [f"no table under '{heading}'"]
    out = [] if rows[0] == TABLE_HEADER else [f"header is {rows[0]}, want {TABLE_HEADER}"]
    want = [P.manifest.name for P in plugins]
    if (names := [_norm(r[0]) for r in rows[1:]]) != want:
        out.append(f"rows are {names}, want {want} (BUILTIN_PLUGINS order)")
    for P, row in zip(plugins, rows[1:]):
        m = P.manifest
        if len(row) != 4:
            out.append(f"{m.name}: {len(row)} cells, want 4")
            continue
        _name, what, default, switch = (_norm(c) for c in row)
        if what != _norm(plugin_summary(P)):
            out.append(f"{m.name}: What it does is {what!r}, plugins list says {_norm(plugin_summary(P))!r}")
        state = "on" if m.enabled_by_default else "off"
        if not default.startswith(state):
            out.append(f"{m.name}: Default is {default!r}, the plugin is {state} by default")
        extra = f"localharness[{m.requires_extra}]"
        if m.requires_extra and extra not in default:
            out.append(f"{m.name}: Default does not name {extra}")
        if not m.requires_extra and "localharness[" in default:
            out.append(f"{m.name}: Default names an extra the plugin does not need")
        verb = "disable" if m.enabled_by_default else "enable"
        if f"localharness plugins {verb} {m.name}" not in switch:
            out.append(f"{m.name}: How to switch lacks `localharness plugins {verb} {m.name}`")
        if not m.enabled_by_default:
            out += [f"{m.name}: How to switch lacks `--set {s.key}`" for s in m.setup
                    if f"--set {s.key}" not in switch]
    return out


@xfail_until("51-05")
def test_plugins_table_readme_live():
    """README's plugins table is the bundled plugin list, as `plugins list` shows it."""
    assert check_plugins_table(_read("README.md")) == []


@xfail_until("51-06")
def test_index_md_lists_the_bundled_plugins():
    """INDEX.md (internal, git-ignored) carries the same plugins table under its source section."""
    if not (_REPO / "INDEX.md").is_file():
        pytest.skip("INDEX.md is git-ignored; checked locally at close-out")
    assert check_plugins_table(_read("INDEX.md"), heading="## src/localharness/") == []


def _good_table() -> str:
    rows = []
    for P in BUILTIN_PLUGINS:
        m = P.manifest
        default = ("on" if m.enabled_by_default else "off") + (
            f"; needs `localharness[{m.requires_extra}]`" if m.requires_extra else "")
        verb = "disable" if m.enabled_by_default else "enable"
        sets = "".join(f" --set {s.key}=..." for s in m.setup) if not m.enabled_by_default else ""
        rows.append(f"| `{m.name}` | {plugin_summary(P)} | {default} | "
                    f"`localharness plugins {verb} {m.name}{sets}` |")
    return "## Plugins\n\n| " + " | ".join(TABLE_HEADER) + " |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"


def test_plugins_table_bites():
    good = _good_table()
    assert check_plugins_table(good) == []
    lines = good.splitlines()
    swapped = "\n".join([*lines[:4], lines[5], lines[4], *lines[6:]])
    assert check_plugins_table(swapped)
    image_row = next(line for line in lines if line.startswith("| `image`"))
    assert check_plugins_table(good.replace(image_row, image_row.replace("| off |", "| on |")))
    assert check_plugins_table(good.replace("Name | What it does", "Plugin | What it does"))


# ---------------------------------------------------------------- check 6: spec 00 section 5

WIRING_HEADING = "### Constructor wiring (the sanctioned exceptions)"


def wiring_code() -> dict[str, frozenset[str]]:
    params = inspect.signature(AgentLoop.__init__).parameters
    return {"AgentLoop.__init__": frozenset(p for p in params if p != "self"),
            "PluginContext": frozenset(f.name for f in dataclasses.fields(PluginContext))}


def check_spec00_wiring(text: str, code: dict[str, frozenset[str]] | None = None) -> list[str]:
    """Spec 00 section 5 opens on the event (data) plane, still names the Config Loader exception,
    and has `### Constructor wiring (the sanctioned exceptions)` with one bullet labelled
    `AgentLoop.__init__` and one labelled `PluginContext`: the other backticked names on each
    bullet equal the parameters / fields, both ways. AP1 no longer calls `registry.execute`."""
    code = wiring_code() if code is None else code
    s5 = section(text, "## 5.", prefix=True)
    if s5 is None:
        return ["spec 00 has no section 5"]
    out = []
    opening = next((p for p in paragraphs(s5) if not p.lstrip().startswith("#")), "").lower()
    if "event" not in opening or "data plane" not in opening:
        out.append("section 5's opening paragraph does not name the event bus as the data plane")
    if "config loader" not in s5.lower():
        out.append("section 5 no longer names the Config Loader exception")
    sub = section(s5, WIRING_HEADING)
    if sub is None:
        out.append(f"section 5 has no '{WIRING_HEADING}' subsection")
    else:
        found = {spans[0]: set(spans[1:]) for _h, b in bullets(sub) if (spans := ticks(b))}
        for label, want in code.items():
            if label not in found:
                out.append(f"no bullet labelled `{label}`")
                continue
            out += [f"`{label}`: `{n}` is not listed" for n in sorted(want - found[label])]
            out += [f"`{label}`: `{n}` is listed but is not in the code" for n in sorted(found[label] - want)]
    ap1 = section(text, "### AP1", prefix=True)
    if ap1 is None:
        out.append("spec 00 has no AP1")
    elif "registry.execute" in ap1:
        out.append("AP1 still presents `registry.execute` (the loop dispatches through its injected registry)")
    return out


def test_spec00_wiring_live():
    """Spec 00 section 5 names the constructor-wired collaborators exactly as the code has them."""
    assert check_spec00_wiring(_read(SPEC00)) == []


def _good_spec00(code: dict[str, frozenset[str]]) -> str:
    items = "".join(f"- `{label}`: " + ", ".join(f"`{n}`" for n in sorted(names)) + "\n"
                    for label, names in code.items())
    return ("## 5. Communication Rule\n\nFacts travel as events on the bus, the data plane.\n\n"
            "The Config Loader is injected at construction.\n\n"
            f"{WIRING_HEADING}\n\n{items}\n## 10. Anti-Patterns\n\n### AP1: Direct Component References\n\n"
            "The loop calls its injected registry.\n")


def test_spec00_wiring_bites():
    code = wiring_code()
    good = _good_spec00(code)
    assert check_spec00_wiring(good) == []
    assert check_spec00_wiring(good.replace(", `memory_slot`", "").replace("`memory_slot`, ", "")) == [
        "`AgentLoop.__init__`: `memory_slot` is not listed"]
    grown = {**code, "AgentLoop.__init__": code["AgentLoop.__init__"] | {"planner"}}
    assert check_spec00_wiring(good, grown) == ["`AgentLoop.__init__`: `planner` is not listed"]
    assert check_spec00_wiring(good.replace("The loop calls", "Calling registry.execute"))


# ---------------------------------------------------------------- check 7: CHANGELOG

def changelog_slice(text: str) -> str:
    """`## [Unreleased]` up to the next `## [` heading, plus `## [0.16.0]` if that heading exists."""
    lines = text.splitlines()

    def entry(title: str) -> list[str]:
        start = next((i for i, line in enumerate(lines) if line.startswith(title)), None)
        if start is None:
            return []
        end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## [")), len(lines))
        return lines[start:end]
    return "\n".join(entry("## [Unreleased]") + entry("## [0.16.0]"))


def discord_env_names() -> frozenset[str]:
    """The environment variables the Discord fallback reads, observed by running it."""
    seen: set[str] = set()

    class Spy(dict):
        def get(self, key, default=None):
            seen.add(key)
            return default

        def __contains__(self, key):
            seen.add(key)
            return False
    env_fallback(DiscordSettings(), Spy(), Path("/nonexistent"))
    return frozenset(seen)


IMAGE_FIRST_RELEASE = "- **Image generation, as a plugin"
GUARDRAILS_BULLET = "- **`GUARDRAILS.md` reaches the model with memory on or off.**"


def check_changelog_items(text: str) -> list[str]:
    """The release entry names the four user-visible changes: the `localharness plugins` command
    and its verbs, the settings keys (`memory.enabled` replacing deprecated `org.memory_enabled`,
    `image.comfyui_url`, `dispatch.discord.*`), every Discord variable the fallback reads with its
    0.17.0 removal, and the image plugin's first release under Added."""
    entry = changelog_slice(text)
    if not entry:
        return ["CHANGELOG has no ## [Unreleased] entry"]
    items = bullets(entry)
    out = []
    plugins_cmd = " ".join(b for _h, b in items if "localharness plugins" in b)
    out += [f"no bullet names `localharness plugins` with `{v}`" for v in ("list", "info", "enable", "disable")
            if not re.search(rf"\b{v}\b", plugins_cmd)]
    if not any("org.memory_enabled" in b and "memory.enabled" in b.replace("org.memory_enabled", "")
               and ("deprecat" in b.lower() or "deprecat" in h.lower()) for h, b in items):
        out.append("no bullet says `org.memory_enabled` is deprecated in favour of `memory.enabled`")
    out += [f"the entry does not name `{k}`" for k in ("image.comfyui_url", "dispatch.discord.") if k not in entry]
    out += [f"no bullet names `{n}` with its 0.17.0 removal" for n in sorted(discord_env_names())
            if not any(n in b and "0.17.0" in b for _h, b in items)]
    if not any(b.startswith(IMAGE_FIRST_RELEASE) and h == "Added" for h, b in items):
        out.append("no Added bullet for the image plugin's first release")
    return out


def check_changelog_structure(text: str) -> list[str]:
    """Every line of the entry is a heading, a list item or its indented continuation, a table row,
    fenced code, or the intro paragraph above the first `###` — so a clobbered bullet head cannot
    hide as a stray line; and the `GUARDRAILS.md` bullet keeps its head."""
    entry = changelog_slice(text)
    out = []
    in_intro = True
    for raw, line in zip(entry.splitlines(), unfenced_lines(entry)):
        if line.startswith("### "):
            in_intro = False
        if in_intro or not line.strip() or raw != line:
            continue
        if not (line.startswith(("#", "- ", "|")) or re.match(r" {2,}\S", line)):
            out.append(f"stray line in the release entry: {line!r}")
    if GUARDRAILS_BULLET not in entry:
        out.append(f"no bullet starting {GUARDRAILS_BULLET!r}")
    return out


def check_changelog(text: str) -> list[str]:
    return check_changelog_items(text) + check_changelog_structure(text)


def test_changelog_items_live():
    """The CHANGELOG names every user-visible change of the plugin release."""
    assert check_changelog_items(_read("CHANGELOG.md")) == []


@xfail_until("51-05")
def test_changelog_structure_live():
    """No CHANGELOG bullet has lost its head (the GUARDRAILS.md item renders as a bullet)."""
    assert check_changelog_structure(_read("CHANGELOG.md")) == []


def _good_changelog() -> str:
    names = ", ".join(f"`{n}`" for n in sorted(discord_env_names()))
    return (
        "# Changelog\n\n## [Unreleased]\n\nThis release adds plugins.\n\n### Added\n"
        "- **`localharness plugins`.** `list`, `info NAME`, `enable NAME`, `disable NAME`.\n"
        "- **Image generation, as a plugin (off by default).** Settings: `image.comfyui_url`.\n"
        "### Changed\n"
        f"{GUARDRAILS_BULLET} Core reads it\n  on every turn.\n"
        "- Discord reads `dispatch.discord.*`.\n"
        "### Deprecated\n- `org.memory_enabled` — use `memory.enabled`.\n"
        f"- The variables {names}\n  stop working in 0.17.0.\n"
        "\n## [0.15.1] — 2026-10-01\n\nstray text in an older entry is not checked\n")


def test_changelog_bites():
    good = _good_changelog()
    assert check_changelog(good) == []
    assert check_changelog_items(good.replace("0.17.0", "a later release"))
    broken = good.replace(GUARDRAILS_BULLET, " the model with memory on or off.**")
    assert len(check_changelog_structure(broken)) == 2, check_changelog_structure(broken)
    assert check_changelog_items(good.replace("`info NAME`, ", ""))


# ---------------------------------------------------------------- check 8: denylist

BANNED = (  # identifiers and sentences that described wiring the code no longer has
    "memory_loader", "MemoryLoader", "recall_router", "_seed_memory_store", "legacy_handles",
    "LEGACY_CHANNELS", "run_pre_hooks", "run_post_hooks", "PluginConflictError", "PluginVersionError",
    "discover_all", "config_schema", "still part of core", "Nothing calls it",
    "LOCALHARNESS_" + "COMFYUI_",  # built from parts: test_image_hygiene bans the literal in tests/
    "UNTRUSTED_INGEST", "HOST_DANGEROUS", "_MEMORY_TOOLS", "KIND_BY_GROUP",
)
INTERNAL_IDS = (  # internal phase numbers, plan ids and requirement ids
    re.compile(r"\b[Pp]hases?[ -]\(?\d"),
    re.compile(r"\b(?:4[4-9]|5[0-2])-\d{2}\b"),
    re.compile(r"\b(?:PAPI|PLUG|MEMP|ENAB|SAFE|DISP|AUTO|IMGP|WEBP|CORE|DOCS)-\d\d\b"),
)
_ENV_OK = ("deprecat", "fallback", "0.17", "Until", "until")


def public_docs() -> list[str]:
    docs = sorted(p.relative_to(_REPO).as_posix() for p in (_REPO / "docs").rglob("*.md"))
    return ["README.md", "SECURITY.md", "CHANGELOG.md",
            *[d for d in docs if not d.startswith("docs/competitive/")
              and d != "docs/task-intent-and-clarification-prd.md"]]


def _blocks(lines: list[str]) -> list[str]:
    """For each line, the block it sits in: its paragraph, list item (with continuation lines) or
    table row."""
    owner: list[str] = []
    start = 0
    for i in range(len(lines) + 1):
        line = lines[i] if i < len(lines) else ""
        if i == len(lines) or not line.strip() or line.lstrip().startswith(("- ", "* ", "|")):
            block = "\n".join(lines[start:i])
            owner += [block] * (i - start)
            start = i
            if i < len(lines) and not line.strip():
                owner.append("")
                start = i + 1
    return owner


def check_denylist(rel: str, text: str) -> list[str]:
    """No public doc names removed wiring, an internal phase / plan / requirement id, or a
    deprecated name outside its deprecation note: a Discord fallback variable only in a paragraph
    or list item that says deprecated / fallback / until 0.17, `org.memory_enabled` only within 200
    characters of "deprecated" / "older", `GuardrailTracker` only where the line or its section
    heading says design / not built. The CHANGELOG is history, so it may name all of these, but
    never an internal id. When the Discord fallback is deleted at 0.17.0, its doc lines go too."""
    out = []
    lines = text.splitlines()
    blocks, heading = _blocks(lines), ""
    for no, (line, plain) in enumerate(zip(lines, unfenced_lines(text)), 1):
        where = f"{rel}:{no}"
        heading = h[1] if (h := _heading(plain)) else heading
        for rx in INTERNAL_IDS:
            if rx is INTERNAL_IDS[1] and re.search(r"20\d\d-", line):  # a date, not a plan id
                continue
            out += [f"{where}: internal id {m.group(0)!r}" for m in rx.finditer(line)]
        if rel == "CHANGELOG.md":
            continue
        out += [f"{where}: {b!r}" for b in BANNED if b in line]
        if "GuardrailTracker" in line and not re.search(r"design|not built", line + heading, re.I):
            out.append(f"{where}: 'GuardrailTracker' presented as built")
        if rel == SPEC09 and "UNSTABLE" in line:
            out.append(f"{where}: 'UNSTABLE'")
        out += [f"{where}: {env!r} outside a deprecation/fallback note"
                for env in ("LOCALHARNESS_DISCORD_", "DISCORD_BOT_TOKEN")
                if env in line and not any(k in blocks[no - 1] for k in _ENV_OK)]
    if rel != "CHANGELOG.md":
        for m in re.finditer(r"org\.memory_enabled", text):
            near = text[max(0, m.start() - 200):m.end() + 200].lower()
            if "deprecat" not in near and "older" not in near:
                out.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: "
                           "'org.memory_enabled' not beside its deprecation")
    return out


_DENYLIST_RED = {"docs/specs/03-agent-loop.md": "51-06", "docs/specs/05-memory.md": "51-06",
                 "docs/specs/12-audit.md": "51-06"}


@pytest.mark.parametrize("rel", [
    pytest.param(rel, marks=xfail_until(_DENYLIST_RED[rel])) if rel in _DENYLIST_RED else rel
    for rel in public_docs()])
def test_denylist_live(rel):
    """No public doc carries stale wiring names or internal phase/plan/requirement ids."""
    assert check_denylist(rel, _read(rel)) == []


def test_denylist_bites():
    lines = ["para", "two", "", "- item", "  more", "| row |", "- next"]
    assert _blocks(lines) == ["para\ntwo", "para\ntwo", "", "- item\n  more", "- item\n  more",
                              "| row |", "- next"]
    assert check_denylist("README.md", "Dates like 2026-10-02 and 44-17 on a 2026-10-02 line are fine.\n") == []
    for text in ("the Phase-36 model look", "in Phase 4 we", "Phase 47 shipped it", "plan 51-03 did",
                 "memory is still part of core", "MEMP-02 says", "a future phase (36/37)"):
        assert check_denylist("README.md", text + "\n"), text
    assert check_denylist("CHANGELOG.md", "- Removed `run_pre_hooks`.\n") == []
    assert check_denylist("CHANGELOG.md", "- Shipped in Phase 49.\n")
    assert check_denylist("docs/specs/11-channels.md", "Set `LOCALHARNESS_DISCORD_TOKEN`.\n")
    assert check_denylist("docs/specs/11-channels.md", "`LOCALHARNESS_DISCORD_TOKEN` (deprecated).\n") == []
    assert check_denylist("README.md", "- Set\n  `LOCALHARNESS_DISCORD_TOKEN`.\n- It is deprecated.\n")
    assert check_denylist("docs/specs/12-audit.md", "## Tracker\n`GuardrailTracker` runs.\n")
    assert check_denylist("docs/specs/12-audit.md", "## Tracker (design, not built)\n`GuardrailTracker`.\n") == []
    assert check_denylist("SECURITY.md", "Set `org.memory_enabled` to turn memory off.\n")
    assert check_denylist(SPEC09, "**Stability:** UNSTABLE (v1).\n")


# ---------------------------------------------------------------- check 9: links

_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _heading_anchor_counts(markdown: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in unfenced_lines(markdown):
        for anchor in _github_anchors(line) if _heading(line) else ():
            counts[anchor] = counts.get(anchor, 0) + 1
    return counts


def check_links(rel: str, text: str, read: Callable[[str], str | None] | None = None) -> list[str]:
    """Every relative link and `#anchor` in `rel` resolves (repo blob/tree URLs count as relative):
    the file or folder exists and the heading exists in the target, exactly once (a duplicated
    heading's anchor is ambiguous on GitHub, so it fails loud)."""
    def default_read(target: str) -> str | None:
        path = _REPO / target
        return path.read_text(encoding="utf-8") if path.is_file() and path.suffix == ".md" else None
    read = read or default_read
    out = []
    body = "\n".join(re.sub(r"`[^`\n]*`", "", line) for line in unfenced_lines(text))
    for m in _LINK.finditer(body):
        url = m.group(1)
        for base in (_BLOB, _TREE):
            if url.startswith(base):
                url = url.removeprefix(base)
                target_base = _REPO
                break
        else:
            if re.match(r"[a-z]+:", url):
                continue
            target_base = (_REPO / rel).parent
        path, _, anchor = url.partition("#")
        if path:
            resolved = (target_base / path).resolve()
            if not resolved.exists():
                out.append(f"{rel}: link to missing {url!r}")
                continue
            target = resolved.relative_to(_REPO).as_posix() if resolved.is_relative_to(_REPO) else None
            content = read(target) if target and resolved.is_file() else None
        else:
            target, content = rel, text
        if anchor and content is not None and target and target.endswith(".md"):
            n = _heading_anchor_counts(content).get(anchor, 0)
            if n != 1:
                out.append(f"{rel}: #{anchor} names {'no heading' if n == 0 else f'{n} headings'} in {target}")
    return out


def test_links_live():
    """Every relative link and anchor in the public docs points at a real file and heading."""
    found = [v for rel in public_docs() for v in check_links(rel, _read(rel))]
    assert found == []


def test_links_bites():
    good = "## Here\n\n[a](#here) [b](SECURITY.md#plugins) [c](docs/specs/) [d](https://example.com)\n"
    assert check_links("README.md", good) == []
    assert check_links("README.md", good.replace("## Here", "## There")) == [
        "README.md: #here names no heading in README.md"]
    assert check_links("README.md", "[x](docs/nope.md)\n") == ["README.md: link to missing 'docs/nope.md'"]
    assert check_links("README.md", "## A\n## A\n[l](#a)\n") == ["README.md: #a names 2 headings in README.md"]
    assert check_links("README.md", f"[s]({_BLOB}docs/specs/09-hooks-plugins.md#no-such-heading)\n")
