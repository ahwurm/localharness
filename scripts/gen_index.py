#!/usr/bin/env python3
"""Generate INDEX.md, the file-structure index of this checkout.

Two views of one tree: what git exposes (the tracked files, which is the public repository) and
what stays on this machine (ignored and excluded paths), followed by a hygiene report: files that
are neither committed nor ignored, uncommitted edits, regenerable caches, stale build artifacts,
public files that name local-only directories, and version drift between pyproject, the package
and the changelog.

The output is local-only: `/INDEX.md` is git-ignored, because the local half names paths that are
not part of the public repository. `--public-only` writes the exposed half alone, for sharing.

Run it after every PyPI release:

    python scripts/gen_index.py                 # writes INDEX.md at the repository root
    python scripts/gen_index.py --clean         # also removes __pycache__, tool caches and dist/
    python scripts/gen_index.py --stdout --public-only

Standard library only. It never imports localharness, so it runs without the virtualenv, and every
fact in the output is read from the tree or asked of git: nothing is typed in by hand except the
one-line purposes of the root files and the layer each package belongs to (LAYERS below). A
package missing from LAYERS is listed as unclassified rather than dropped, so a new directory
shows up as a question instead of disappearing.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------- hand-kept tables

# Package directory (relative to src/localharness) -> architecture layer.
LAYERS: dict[str, str] = {
    "": "Package",
    "core": "Layer 1 - Foundation",
    "config": "Layer 1 (models.py) and Layer 2 - Infrastructure",
    "provider": "Layer 2 - Infrastructure",
    "tools": "Layer 2 - Infrastructure",
    "tools/builtin": "Layer 2 - Infrastructure (built-in tools; the image plugin's files live here too)",
    "tools/builtin/workflows": "Data - the image plugin's ComfyUI workflow",
    "plugins": "Layer 2 - Infrastructure (the plugin API and lifecycle)",
    "registry": "Layer 2 - Infrastructure (component catalogue)",
    "agent": "Layer 3 - Agent runtime",
    "orchestrator": "Layer 4 - Orchestration",
    "cli": "Layer 5 - User interface (plugin command modules belong to their plugins)",
    "channels": "Layer 5 - User interface",
    "bench": "Layer 5 - User interface (the benchmark runner)",
    "assets": "Data",
    "memory": "Plugin - memory",
    "dispatch": "Plugin - dispatch",
    "dispatch/adapters": "Plugin - dispatch (chat platform adapters)",
    "autoresearch": "Plugin - autoresearch",
    "channels/mobile": "Plugin - mobile",
    "channels/mobile/ui": "Plugin - mobile (the static phone shell)",
}
LAYER_ORDER = ("Package", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Plugin", "Data")

ROOT_FILES: dict[str, str] = {
    "pyproject.toml": "Package metadata, dependencies, extras, the `localharness` entry point, pytest and ruff settings",
    "uv.lock": "Resolved dependency versions (uv)",
    "README.md": "Public front page: what it is, quick start, runtimes, plugins",
    "SECURITY.md": "Trust boundaries, the approval gate, machine-level-only settings, threat model; checked against the code by tests",
    "CHANGELOG.md": "Release history (Keep a Changelog)",
    "CONTRIBUTING.md": "How to contribute: test conventions, hard rules, PR flow",
    "CODE_OF_CONDUCT.md": "Community standards",
    "LICENSE": "License",
    ".gitignore": "What never enters a commit (planning, local docs, caches, secrets)",
    ".nvmrc": "Node version for the mobile shell's screenshot script",
}
TOP_DIRS: dict[str, str] = {
    "src": "The package (`src/localharness`)",
    "tests": "Unit and integration suites, fixtures, the shared conftest",
    "docs": "Specs, plugin and runtime guides, reference architectures",
    "bench": "Benchmark corpus: `bench.yaml`, categories, fixtures, train and sealed holdout scenarios",
    "examples": "Example agents, the plugin template, a worked workflow",
    "scripts": "Maintainer scripts (this generator among them)",
    "subagents": "Specialist YAML definitions the `agent` tool can delegate to",
    "assets": "README media",
    ".github": "Issue and PR templates, CI and publish workflows",
}
LOCAL_NOTES: dict[str, str] = {
    ".planning": "GSD planning: roadmap, phases, state, research. Never committed; history was once purged of it.",
    ".gsd": "GSD sentinel files (ignored by its own .gitignore)",
    ".claude": "Claude Code local settings and session data",
    ".localharness": "Local runtime state: dogfood memories, the autoresearch archive",
    "dist": "Build artifacts from `uv build`",
    "reports": "Live dogfood ledgers (search-verifier writes here per UTC day)",
    "lh_test_artifacts": "REPL dogfood outputs written by the subject model",
    "CLAUDE.md": "Steering doc for Claude Code sessions in this repo",
    "INDEX.md": "This file",
    "ARCHITECT-REVIEW.md": "Architect trail",
    ".venv": "The virtualenv (`uv sync --extra dev`)",
    ".coverage": "Coverage data",
    "coverage.json": "Coverage report",
    "docs/competitive": "Competitive notes (internal)",
    ".env": "Secrets",
}
CACHE_DIRS = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")
INTERNAL_DIRS = (".planning/", "docs/competitive/")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".json", ".cfg", ".ini", ".sh", ".js",
                 ".html", ".css", ".svg", ".gitignore", ".nvmrc", ""}
# Never walked, never cleaned: git's own data, the virtualenv, and the local-only trees that belong to
# other tools (they are listed by size in the local-only section, from git's view, not walked).
SKIP_WALK = {".git", ".venv", "node_modules", ".claude", ".planning", ".localharness"}

# ----------------------------------------------------------------------------- git and files


def git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return out.stdout


def git_z(root: Path, *args: str) -> list[str]:
    return [p for p in git(root, *args, "-z").split("\0") if p]


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    try:
        return Path(git(here, "rev-parse", "--show-toplevel").strip())
    except subprocess.CalledProcessError:
        sys.exit("not inside a git checkout")


def size_kib(path: Path) -> int:
    try:
        return int(subprocess.run(["du", "-sk", "--", str(path)], check=True, capture_output=True,
                                  text=True).stdout.split()[0])
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        total = 0
        if path.is_file():
            return max(1, path.stat().st_size // 1024)
        for dirpath, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(dirpath) / f).stat().st_size
                except OSError:
                    pass
        return total // 1024


def human(kib: int) -> str:
    if kib < 1024:
        return f"{kib} KiB"
    if kib < 1024 * 1024:
        return f"{kib / 1024:.1f} MiB"
    return f"{kib / 1024 / 1024:.1f} GiB"


def ignore_source(root: Path, rel: str) -> str:
    """`<file>:<line> <pattern>` for the rule that ignores `rel`; a directory ignored only from inside
    (its own .gitignore) is named as such."""
    def ask(target: str) -> str:
        out = subprocess.run(["git", "-C", str(root), "check-ignore", "-v", "--", target],
                             capture_output=True, text=True).stdout.strip()
        if not out:
            return ""
        src, line, pattern, _path = re.split(r"[:\t]", out, maxsplit=3)
        return f"{src}:{line} `{pattern}`"
    found = ask(rel)
    if not found and (root / rel).is_dir():
        inner = next((p for p in (root / rel).iterdir()), None)
        if inner is not None:
            found = ask(inner.relative_to(root).as_posix())
            if found.startswith(rel.rstrip("/")):
                found = f"its own {found}"
    return found or "(no rule matched)"


def esc(cell: object) -> str:
    return str(cell).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_none_\n"
    head = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    return head + "".join("| " + " | ".join(esc(c) for c in r) + " |\n" for r in rows)


def first_line(doc: str, width: int = 120) -> str:
    line = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    line = line.rstrip(".")
    return line if len(line) <= width else line[: width - 1].rstrip() + "…"


def h1(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except (OSError, UnicodeDecodeError):
        pass
    return ""


def layer_rank(layer: str) -> int:
    for i, key in enumerate(LAYER_ORDER):
        if layer.startswith(key):
            return i
    return len(LAYER_ORDER)


# ----------------------------------------------------------------------------- the source tree


def module_rows(root: Path) -> dict[str, list[tuple[str, int, str]]]:
    """package dir -> [(module file, lines, first docstring line)], from the AST, never an import."""
    src = root / "src" / "localharness"
    by_pkg: dict[str, list[tuple[str, int, str]]] = {}
    for p in sorted(src.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        try:
            doc = ast.get_docstring(ast.parse(text)) or ""
        except SyntaxError as e:
            doc = f"(does not parse: {e.msg})"
        pkg = p.parent.relative_to(src).as_posix()
        pkg = "" if pkg == "." else pkg
        by_pkg.setdefault(pkg, []).append((p.name, text.count("\n") + (not text.endswith("\n")), first_line(doc)))
    for d in sorted(x for x in src.rglob("*") if x.is_dir() and "__pycache__" not in x.parts):
        rel = d.relative_to(src).as_posix()
        if rel not in by_pkg and any(f.is_file() for f in d.iterdir()):
            by_pkg[rel] = []  # a data-only directory (assets, workflows, the phone shell)
    return by_pkg


def data_files(root: Path, pkg: str) -> list[str]:
    d = root / "src" / "localharness" / pkg
    return sorted(f.name for f in d.iterdir() if f.is_file() and f.suffix != ".py" and f.name != "__pycache__")


def cli_surface(root: Path) -> tuple[list[tuple[str, str, bool]], list[tuple[str, str]]]:
    """(commands as (name, module, hidden), sub-apps as (name, variable)) read from cli/app.py."""
    text = (root / "src/localharness/cli/app.py").read_text(encoding="utf-8")
    imports = {m.group(2): m.group(1) for m in re.finditer(
        r"from localharness\.cli\.(\w+) import (\w+)", text)}
    for m in re.finditer(r"from localharness\.cli\.(\w+) import \(([^)]*)\)", text, re.S):
        for name in re.findall(r"\w+", m.group(2)):
            imports[name] = m.group(1)
    commands = [(m.group(1), imports.get(m.group(3), "?"), bool(m.group(2)))
                for m in re.finditer(r'app\.command\("([^"]+)"(,\s*hidden=True)?\)\((\w+)\)', text)]
    subapps = [(m.group(2), imports.get(m.group(1), m.group(1)))
               for m in re.finditer(r'app\.add_typer\((\w+),\s*name="([^"]+)"\)', text)]
    return commands, subapps


def bundled_plugins(root: Path) -> list[dict[str, str]]:
    """Each class in BUILTIN_PLUGINS with its manifest fields, read from the module text."""
    src = root / "src" / "localharness"
    text = (src / "plugins/builtin.py").read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"from localharness\.([\w.]+) import (\w+)", text):
        mod, cls = m.group(1), m.group(2)
        if mod == "plugins.api":
            continue
        path = src / (mod.replace(".", "/") + ".py")
        body = path.read_text(encoding="utf-8") if path.exists() else ""
        manifest = body[body.find("PluginManifest("):] if "PluginManifest(" in body else body

        def field(name: str, default: str = "") -> str:
            f = re.search(rf"\b{name}=(\"[^\"]*\"|True|False|None)", manifest)
            return f.group(1).strip('"') if f else default
        out.append({
            "class": cls, "module": mod.replace(".", "/") + ".py",
            "name": field("name", cls), "kind": field("kind", "?"),
            "on": field("enabled_by_default", "True"), "extra": field("requires_extra", "") or "-",
            "cli": ", ".join(f"`{c}`" for c in re.findall(r'CliDescriptor\(\s*name="([^"]+)"', body)) or "-",
            "slash": ", ".join(f"`/{s.lstrip('/')}`" for s in re.findall(r'SlashDescriptor\(\s*name="([^"]+)"', body)) or "-",
        })
    return out


def core_slash_commands(root: Path) -> list[str]:
    text = (root / "src/localharness/cli/slash_commands.py").read_text(encoding="utf-8")
    seen: list[str] = []
    for m in re.finditer(r'"(/[a-z][\w-]*)"', text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


# ----------------------------------------------------------------------------- hygiene


def version_facts(root: Path) -> dict[str, str]:
    facts = {}
    m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    facts["pyproject"] = m.group(1) if m else "?"
    init = root / "src/localharness/__init__.py"
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text(encoding="utf-8")) if init.exists() else None
    facts["package"] = m.group(1) if m else "(reads it from package metadata)"
    cl = root / "CHANGELOG.md"
    heads = re.findall(r"^## \[([^\]]+)\]", cl.read_text(encoding="utf-8"), re.M) if cl.exists() else []
    facts["changelog"] = next((h for h in heads if h.lower() != "unreleased"), "?")
    facts["changelog_unreleased"] = "yes" if any(h.lower() == "unreleased" for h in heads) else "no"
    return facts


def internal_mentions(root: Path, tracked: list[str]) -> list[tuple[str, int]]:
    out = []
    for rel in tracked:
        p = root / rel
        if rel in (".gitignore", "scripts/gen_index.py") or p.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        n = sum(text.count(needle) for needle in INTERNAL_DIRS)
        if n:
            out.append((rel, n))
    return out


def empty_dirs(root: Path) -> list[str]:
    found = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_WALK and d != "__pycache__"]
        if not dirs and not files and Path(dirpath) != root:
            found.append(Path(dirpath).relative_to(root).as_posix() + "/")
    return sorted(found)


def cache_dirs(root: Path) -> list[Path]:
    found = []
    for dirpath, dirs, _files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_WALK]
        for d in list(dirs):
            if d in CACHE_DIRS:
                found.append(Path(dirpath) / d)
                dirs.remove(d)
    return found


def md_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)", text) if not re.match(r"[a-z]+:", m.group(1))]


def clean(root: Path) -> list[str]:
    removed = []
    for d in cache_dirs(root):
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d.relative_to(root).as_posix() + "/")
    for name in ("dist", ".coverage", "coverage.json"):
        p = root / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(name + "/")
        elif p.exists():
            p.unlink()
            removed.append(name)
    return removed


# ----------------------------------------------------------------------------- render


def render(root: Path, *, public_only: bool, removed: list[str]) -> str:
    tracked = git_z(root, "ls-files")
    untracked = git_z(root, "ls-files", "--others", "--exclude-standard")
    modified = sorted({line[3:].split("\0")[0] for line in git(root, "status", "--porcelain", "-z").split("\0")
                       if line and line[:2].strip() and line[:2] != "??"})
    ignored_all = git_z(root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory")
    caches = [p for p in ignored_all if Path(p.rstrip("/")).name in CACHE_DIRS or p.endswith((".pyc", ".pyo"))]
    local = sorted(p for p in ignored_all if p not in caches and p != ".venv/")
    # A directory ignored from inside (its own .gitignore) is listed by git once as the directory and
    # once per file; keep the directory row only.
    dirs = {p for p in local if p.endswith("/")}
    local = [p for p in local if not any(p != d and p.startswith(d) for d in dirs)]
    head = git(root, "rev-parse", "--short", "HEAD").strip()
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    versions = version_facts(root)
    by_pkg = module_rows(root)
    n_modules = sum(len(v) for v in by_pkg.values())
    n_lines = sum(n for v in by_pkg.values() for _f, n, _d in v)
    def test_files(prefix: str) -> list[str]:
        return [t for t in tracked if t.startswith(prefix) and Path(t).name.startswith("test_") and t.endswith(".py")]
    tests_unit = test_files("tests/unit/")
    tests_integration = test_files("tests/integration/")
    docs = sorted(set(t for t in tracked + untracked if t.startswith("docs/") and t.endswith(".md")))
    untracked_set = set(untracked)

    o: list[str] = []
    w = o.append
    w("# LocalHarness file index\n")
    w(f"Generated {dt.date.today().isoformat()} by `python scripts/gen_index.py` at `{head}` on `{branch}`; "
      f"pyproject version {versions['pyproject']}. "
      + ("Public view only (`--public-only`).\n" if public_only else
         "Local-only: `/INDEX.md` is git-ignored, because the second half names paths that are not in the "
         "public repository. Regenerate after every PyPI release; `--clean` also drops regenerable caches.\n"))
    w("\n## At a glance\n")
    w(table(["What", "Count"], [
        ["Tracked files (what git exposes)", len(tracked)],
        ["Source modules under `src/localharness`", f"{n_modules} ({n_lines:,} lines)"],
        ["Unit test files", len(tests_unit)],
        ["Integration test files", len(tests_integration)],
        ["Public docs (`docs/**/*.md`)", len([d for d in docs if d not in untracked_set])],
        ["Bundled plugins", len(bundled_plugins(root))],
    ]))

    # ---- exposed tree
    w("\n## Git-exposed tree\n")
    w("Everything in this section is tracked and public unless a row says otherwise.\n")
    w("\n### Root\n")
    rows = []
    for rel in sorted(t for t in tracked if "/" not in t):
        rows.append([f"`{rel}`", ROOT_FILES.get(rel, "")])
    for d, purpose in TOP_DIRS.items():
        n = sum(1 for t in tracked if t.startswith(d + "/"))
        if n:
            rows.append([f"`{d}/`", f"{purpose} ({n} files)"])
    w(table(["Entry", "Purpose"], rows))

    w(f"\n### `src/localharness/` ({n_modules} modules, {n_lines:,} lines)\n")
    w("One row per module: its length and the first line of its docstring, read from the AST. The layer is "
      "the hand-kept `LAYERS` table in the generator; an unclassified package means that table needs a row.\n")
    for pkg in sorted(by_pkg, key=lambda k: (layer_rank(LAYERS.get(k, "unclassified")), k)):
        layer = LAYERS.get(pkg, "unclassified - add this directory to LAYERS in scripts/gen_index.py")
        title = "localharness/" if pkg == "" else f"{pkg}/"
        w(f"\n#### `{title}` - {layer}\n")
        rows = [[f"`{f}`", n, d] for f, n, d in by_pkg[pkg]]
        if rows:
            w(table(["Module", "Lines", "What it is"], rows))
        data = data_files(root, pkg)
        if data:
            w(("Data files: " if rows else "Data-only: ") + ", ".join(f"`{d}`" for d in data) + "\n")

    w("\n### `tests/`\n")
    fixtures = sorted(set(t.split("/")[2] for t in tracked if t.startswith("tests/fixtures/") and t.count("/") >= 2))
    w(table(["Where", "What"], [
        ["`tests/unit/`", f"{len(tests_unit)} test files; the structural pins (import direction, docs convergence, declared dependencies) live here"],
        ["`tests/integration/`", f"{len(tests_integration)} test files"],
        ["`tests/fixtures/`", ", ".join(f"`{f}`" for f in fixtures)],
        ["`tests/conftest.py`", "Shared fixtures and guards: the socket guard, the service-manager seam stub, `fake_home`"],
        ["`tests/dispatch_support.py`", "Support for the dispatch plugin's tests"],
    ]))

    w("\n### `docs/`\n")
    rows = []
    for rel in docs:
        status = "**untracked**" if rel in untracked_set else ""
        rows.append([f"`{rel}`", h1(root / rel), status])
    w(table(["Doc", "Title", ""], rows))

    w("\n### `bench/`, `examples/`, `scripts/`, `subagents/`, `assets/`, `.github/`\n")
    train = sum(1 for t in tracked if t.startswith("bench/scenarios/train/") and t.endswith(".yaml"))
    hold = sum(1 for t in tracked if t.startswith("bench/scenarios/holdout/") and t.endswith(".yaml"))
    rows = [
        ["`bench/`", f"`bench.yaml`, `categories.yaml`, `fixtures/`, `scenarios/train/` ({train} scenarios), "
                     f"`scenarios/holdout/` ({hold} scenarios, sealed: never run while proposing)"],
    ]
    for d in ("examples", "scripts", "subagents", "assets", ".github"):
        files = [t for t in tracked if t.startswith(d + "/")]
        if d == "subagents":
            excluded = [p for p in local if p.startswith("subagents/")]
            note = "; local-only (`.git/info/exclude`): " + ", ".join(f"`{Path(p).name}`" for p in excluded) if excluded and not public_only else ""
            rows.append([f"`{d}/`", ", ".join(f"`{Path(f).name}`" for f in files) + note])
        elif d == "scripts":
            rows.append([f"`{d}/`", "; ".join(f"`{Path(f).name}`: {first_line(script_doc(root / f), 90)}" if f.endswith('.py') else f"`{Path(f).name}`" for f in files)])
        else:
            rows.append([f"`{d}/`", ", ".join(f"`{f[len(d) + 1:]}`" for f in files)])
    w(table(["Where", "What"], rows))

    # ---- CLI surface
    commands, subapps = cli_surface(root)
    plugins = bundled_plugins(root)
    w("\n## CLI surface\n")
    w("Read from `cli/app.py` (core) and each bundled plugin's manifest (mounted lazily by `cli/plugin_mount.py`).\n")
    rows = [[f"`localharness {name}`" + (" (hidden)" if hidden else ""), f"`cli/{mod}.py`"] for name, mod, hidden in commands]
    rows += [[f"`localharness {name} …`", f"`cli/{mod}.py`"] for name, mod in subapps]
    w(table(["Command", "Module"], rows))
    rows = [[f"`localharness {c.strip('`')}`", f"plugin `{p['name']}`"] for p in plugins for c in p["cli"].split(", ") if c != "-"]
    w("\nPlugin-mounted commands (present only while the plugin is on):\n")
    w(table(["Command", "Owner"], rows))
    w("\nSlash commands: core table in `cli/slash_commands.py`: " + ", ".join(f"`{s}`" for s in core_slash_commands(root))
      + "; plugin rows: " + (", ".join(f"{p['slash']} ({p['name']})" for p in plugins if p["slash"] != "-") or "none") + ".\n")

    # ---- plugins
    w("\n## Bundled plugins\n")
    w("The one list is `BUILTIN_PLUGINS` in `plugins/builtin.py`; the rows below are read from each class's manifest.\n")
    w(table(["Plugin", "Kind", "On by default", "Needs extra", "Module", "Commands", "Slash"],
            [[f"`{p['name']}`", p["kind"], p["on"], p["extra"], f"`{p['module']}`", p["cli"], p["slash"]] for p in plugins]))

    # ---- local-only
    if not public_only:
        w("\n## Local-only (not git-exposed)\n")
        w("Paths that exist in this checkout and that git ignores or excludes. Sizes are `du -sk`.\n")
        rows = []
        for rel in local:
            key = rel.rstrip("/")
            note = LOCAL_NOTES.get(key, "")
            if key.startswith("subagents/"):
                note = "Specialist YAML kept out of the public repo"
            rows.append([f"`{rel}`", human(size_kib(root / key)), ignore_source(root, rel), note])
        w(table(["Path", "Size", "Ignored by", "What it is"], rows))
        exclude = root / ".git/info/exclude"
        if exclude.exists():
            entries = [line.strip() for line in exclude.read_text(encoding="utf-8").splitlines()
                       if line.strip() and not line.startswith("#")]
            w("\n`.git/info/exclude` (this clone only, not in any commit): " + (", ".join(f"`{e}`" for e in entries) or "empty") + "\n")

    # ---- hygiene
    w("\n## Hygiene\n")
    if removed:
        w("Removed by `--clean` on this run: " + ", ".join(f"`{r}`" for r in removed[:12])
          + (f" and {len(removed) - 12} more" if len(removed) > 12 else "") + ".\n\n")
    findings: list[str] = []
    if untracked:
        findings.append("**Untracked and not ignored** (commit, or add a rule): " + ", ".join(f"`{u}`" for u in untracked))
    if modified:
        findings.append("**Uncommitted edits to tracked files:** " + ", ".join(f"`{m}`" for m in modified))
    n_cache = len(caches)
    if n_cache:
        findings.append(f"**Regenerable caches:** {n_cache} ignored cache entries (`__pycache__`, tool caches) outside `.venv`; "
                        "`python scripts/gen_index.py --clean` removes them")
    dist = root / "dist"
    if dist.is_dir():
        arts = sorted(p.name for p in dist.iterdir() if not p.name.startswith("."))
        stale = [a for a in arts if versions["pyproject"] not in a]
        findings.append("**Build artifacts in `dist/`:** " + ", ".join(f"`{a}`" for a in arts)
                        + (f"; {len(stale)} do not match pyproject version {versions['pyproject']} (stale)" if stale else ""))
    mentions = internal_mentions(root, tracked)
    if mentions:
        findings.append("**Public files that name a local-only directory** (`.planning/`, `docs/competitive/`); a reader of the "
                        "public repo cannot follow these: " + ", ".join(f"`{p}` ({n})" for p, n in mentions))
    if versions["changelog"] != versions["pyproject"]:
        findings.append(f"**Version drift:** pyproject says {versions['pyproject']}, the newest CHANGELOG release heading is "
                        f"{versions['changelog']} (an Unreleased section is {'present' if versions['changelog_unreleased'] == 'yes' else 'absent'})")
    if not public_only:
        claude_md = root / "CLAUDE.md"
        if claude_md.exists():
            broken = [link for link in md_links(claude_md) if not (root / link).exists() and link != "INDEX.md"]
            if broken:
                findings.append("**CLAUDE.md links that do not resolve:** " + ", ".join(f"`{b}`" for b in broken))
    big = [(t, (root / t).stat().st_size) for t in tracked if (root / t).is_file() and (root / t).stat().st_size > 1024 * 1024]
    if big:
        findings.append("**Tracked files over 1 MiB:** " + ", ".join(f"`{t}` ({s / 1024 / 1024:.1f} MiB)" for t, s in big))
    empties = empty_dirs(root)
    if empties:
        findings.append("**Empty directories:** " + ", ".join(f"`{e}`" for e in empties))
    w("\n".join(f"- {f}" for f in findings) + "\n" if findings else "Nothing to report.\n")
    return "".join(o)


def script_doc(path: Path) -> str:
    try:
        return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="where to write (default: <repo>/INDEX.md)")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    ap.add_argument("--public-only", action="store_true", help="omit the local-only section and local hygiene rows")
    ap.add_argument("--clean", action="store_true", help="remove __pycache__, tool caches, dist/ and coverage files first")
    args = ap.parse_args(argv)
    root = repo_root()
    removed = clean(root) if args.clean else []
    text = render(root, public_only=args.public_only, removed=removed)
    if args.stdout:
        sys.stdout.write(text)
        return 0
    out = args.out or root / "INDEX.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({text.count(chr(10))} lines)")
    if removed:
        print(f"removed {len(removed)} regenerable paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
