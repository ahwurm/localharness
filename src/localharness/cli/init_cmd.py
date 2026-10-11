"""localharness init command — auto-detect LLM (or guided setup) and write config."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shlex
import shutil
import sys
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

import typer
from pydantic import SecretStr
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt, Prompt

from localharness.agent.context import response_reserve
from localharness.agent.gguf_header import (
    DiscoveredModel,
    discover,
    preset_name,
    rank_models,
    unique_preset_name,
    walk_ggufs,
)
from localharness.cli import init_import, workspace
from localharness.cli.errors import _HANDLED, report_filesystem_error
from localharness.cli.workspace import shown_path, visible_text
from localharness.config import trust
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.config.migrate import BACKUP_STAMP_FORMAT
from localharness.config.overlay import atomic_write_overlay, load_overlay
from localharness.config.paths import WORKSPACE_DIR_NAME, global_config_dir, resolve_config_dir
from localharness.config.redact import reveal, secret_values, validation_text
from localharness.core.private_files import ensure_private_dir, touch_private, write_private_bytes
from localharness.core.which import which_outside_cwd
from localharness.config.models import (
    AutosenseConfig,
    ContextConfig,
    EndpointRef,
    HarnessConfig,
    ManagedServerConfig,
    OrgConfig,
    PermissionConfig,
    ProviderConfig,
)
from localharness.provider import autosense, procfs, speed_stats, units
from localharness.provider import server as managed_server
from localharness.provider.server import RUNTIME_LABELS
from localharness.provider.client import CapabilityResult, LLMClient, LLMConfig
from localharness.provider.detector import (
    DEFAULT_PORTS,
    DetectorResult,
    ProviderType,
    detect_provider,
)
from localharness.provider.refarch import REF_ARCHS

console = Console()
err_console = Console(stderr=True)

# Port -> backend label. Single source for BOTH the --help probe-order line (derived below)
# and the "no server detected" printout — so the documented order can never drift from the
# detector's DEFAULT_PORTS (#52: :8000, vLLM's stock port, was silently missing from --help).
_PORT_LABELS: dict[int, str] = {
    8081: "vLLM", 8000: "vLLM", 11434: "Ollama", 1234: "LM Studio", 8080: "llama.cpp"
}
_PROBE_ORDER = ", ".join(f"{_PORT_LABELS.get(p, 'unknown')} (:{p})" for p in DEFAULT_PORTS)


def _build_base_url_for_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def _detect_max_model_len(base_url: str) -> int | None:
    """vLLM's /v1/models exposes max_model_len — fit the context budget to the live window.

    Returns None when the endpoint doesn't report it (Ollama, LM Studio, llama.cpp)."""
    try:
        import httpx
        data = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0).json()
        val = data["data"][0].get("max_model_len")
        return int(val) if val else None
    except Exception:
        return None


def _list_endpoint_models(base_url: str) -> list[str] | None:
    """Model ids advertised by an OpenAI-compatible /v1/models listing.

    Returns None when the listing can't be read at all (unreachable/silent endpoint) — that is a
    different answer from "reachable, serves nothing" ([]), and #118's caller must not conflate
    them: only the first is an honest "I couldn't ask the server"."""
    try:
        import httpx
        data = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0).json()
        return [m["id"] for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]
    except Exception:
        return None


def _detect_lmstudio_ctx(base_url: str) -> int | None:
    """LM Studio's /api/v0/models exposes the served window: the loaded model's
    loaded_context_length, falling back to the largest max_context_length.

    base_url is OpenAI-compat (…/v1); /api/v0 lives at the server root.
    Returns None on any error (→ safe context default)."""
    try:
        import httpx
        native = base_url.removesuffix("/v1")
        data = httpx.get(f"{native}/api/v0/models", timeout=2.0).json()
        entries = [e for e in (data.get("data") or []) if isinstance(e, dict)]
        loaded = next(
            (e for e in entries if e.get("state") == "loaded" and e.get("loaded_context_length")),
            None,
        )
        if loaded:
            return int(loaded["loaded_context_length"])
        caps = [e["max_context_length"] for e in entries if e.get("max_context_length")]
        return max(caps) if caps else None
    except Exception:
        return None


def _identify_endpoint_provider(base_url: str) -> ProviderType:
    """Identify the runtime behind an explicit --endpoint, reusing the detector's shape rules.

    provider_type is load-bearing, not a label: TokenCounter's exact GGUF counting and the
    served-window probe are keyed on it, so hardcoding "unknown" here downgraded a remote
    Ollama/LM Studio to approximate counting. Returns "unknown" only when identification
    genuinely fails (every probe is best-effort — a slow or private endpoint must not block init).
    """
    import httpx
    from localharness.provider.detector import _identify_provider

    native = base_url.removesuffix("/v1")
    try:  # Ollama self-identifies only on its native API — its /v1/models is a plain OpenAI list
        if isinstance(httpx.get(f"{native}/api/tags", timeout=2.0).json().get("models"), list):
            return "ollama"
    except Exception:
        pass
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0)
        ptype = _identify_provider(urlparse(base_url).port or 0, response.json(), response.headers)
    except Exception:
        return "unknown"
    # LM Studio 0.4.x is indistinguishable from vLLM on /v1/models; /api/v0 is unique to it.
    return "lmstudio" if ptype == "vllm" and _detect_lmstudio_ctx(base_url) is not None else ptype


_WORKSPACE_CONFIG_TEMPLATE = """\
# LocalHarness workspace config — this project's own layer.
#
# The specific beats the general: any key you set here wins over your machine-wide
# ~/.localharness/config.yaml, and the machine-wide layer still governs everything this
# file is silent about. Empty is a valid, useful state — the workspace already scopes
# this project's memory, sessions and logs without a single setting below.
#
# The full merge order, lowest priority first:
#   ~/.localharness/config.yaml  <  ~/.localharness/overrides.yaml
#     <  .localharness/config.yaml  <  .localharness/overrides.yaml
#
# Two rules that do NOT follow "workspace wins", on purpose:
#   * org.permissions.deny_patterns UNIONS across layers. Safety accumulates; a workspace
#     can add a denial but can never remove one your machine set.
#   * provider: is hardware truth and belongs to the machine. Leave it out of this file;
#     `localharness init` and `/model` always write the global layer.
#
# Uncomment what you need.
#
# org:
#   log_level: debug
#   permissions:
#     mode: read-only         # this project's sessions never write, edit, or run shell
#   context:
#     compaction_threshold_pct: 85.0
# memory:
#   enabled: false            # memory off for THIS project only: no store, no memory tools
#
# Per-agent settings live in .localharness/agents/<name>.yaml, not here. The one people
# look for first:
#   memory:
#     recall_scope: workspace   # workspace (default) | global | both
#   ...controls which memory store a session in this project RECALLS from. It moves reads
#   only — a session always writes to this project's own store whatever it says.
"""

# ENAB-05: the README `init` leaves in each plugins/ folder — where a user first learns how plugins
# arrive. The workspace copy says plainly that a project is never a plugin source (ENAB-06).
GLOBAL_PLUGINS_README = """\
# Plugins

A plugin adds tools, commands and checks to LocalHarness. There are two ways to add one, and
LocalHarness loads neither until you turn it on:

1. Install a package that declares a `localharness.plugins` entry point into the same Python
   environment as LocalHarness:
   - if you installed LocalHarness with `uv tool`, run
     `uv tool install --with <package> localharness`. Each run replaces the install's plugins and
     extras with the ones it names, so name your extras too (`'localharness[mobile]'`) and add a
     `--with` for every plugin you keep;
   - in a virtual environment, activate it and run `uv pip install <package>`.
2. Put a folder here, `plugins/<name>/`, whose `__init__.py` binds the name `plugin` to the
   plugin class.

`localharness plugins list` shows what was found and whether it is on.
`localharness plugins enable <name>` turns one on. Turning on a plugin you brought yourself is a
machine-level setting, and it is you vouching for what that plugin says its tools do. Plugin code
runs inside the harness with your privileges — treat it like any program you install.

To write one, copy the example plugin: `examples/plugin-template/` in the LocalHarness repository
(https://github.com/ahwurm/localharness/tree/main/examples/plugin-template).
"""

WORKSPACE_PLUGINS_README = """\
# Plugins in this project

LocalHarness never loads plugin code from a project folder — it does not look for plugins in this
folder, or anywhere else in this project. Plugins come from your machine: an installed package that
declares a `localharness.plugins` entry point, or a folder `plugins/<name>/` in your machine-level
config directory (`~/.localharness/` by default).

What a project CAN do is switch a plugin that ships with LocalHarness on or off for itself:

    localharness plugins enable <name> --workspace
    localharness plugins disable <name> --workspace

A plugin you brought yourself is turned on only in your machine-level settings
(`localharness plugins enable <name>`, without `--workspace`).

To write a plugin, copy the example plugin: `examples/plugin-template/` in the LocalHarness
repository (https://github.com/ahwurm/localharness/tree/main/examples/plugin-template).
"""

# State the harness writes under a project's .localharness/ stays out of git; the files a team
# shares (config.yaml, overrides.yaml, agents/*.yaml, plugins/README.md) stay committable.
WORKSPACE_GITIGNORE = """\
# LocalHarness state for this project: sessions, memory, logs and generated pictures. Git ignores it.
# Yours to commit: config.yaml, overrides.yaml, agents/*.yaml, plugins/README.md.
agents/*/
artifacts/
autoresearch/
audit.jsonl
.repl_history
*.db
*.db-shm
*.db-wal
*.log
"""


def _is_the_global_config_dir(target: Path) -> bool:
    """Would scaffolding here write the machine's own config directory?

    Two answers count, because two things can be "the global dir": whatever the env chain resolves
    to right now, and the literal `~/.localharness` that the chain falls back to. Compared by
    realpath so a symlinked `$HOME` (or a `~` that is itself a link) does not read as a different
    directory — the same identity rule the trust store uses.

    Never raises: a path that cannot be resolved is, by definition, not the config dir we are
    standing in, and this check must not become the thing that breaks the command.
    """
    def _real(path: Path) -> Optional[Path]:
        try:
            return path.resolve()
        except (OSError, RuntimeError):
            # RuntimeError is pathlib's OWN spelling of a symlink loop ("Symlink loop from ..."),
            # not an OSError — so catching only OSError turned `.localharness -> .localharness`
            # into a traceback here, in a check that exists to make a message clearer. Caught by
            # the full suite; the targeted run had been written before this function existed.
            return None

    here = _real(target)
    if here is None:
        return False
    return here in {_real(global_config_dir()), _real(Path.home() / WORKSPACE_DIR_NAME)}


def _refuse_existing(target: Path) -> None:
    """"There is already a workspace here" — from the pre-check and from losing the mkdir race.

    One function because the two callers must give the SAME answer: which one fires is a matter of
    microseconds, and a user watching ten parallel runs must not be able to tell them apart.
    """
    err_console.print(
        "[bold red]Error:[/bold red] "
        + escape(f"A workspace already exists at {target}. "
                 f"Edit {target / 'config.yaml'} directly")
        + " — this command never overwrites one.",
        soft_wrap=True,
    )
    raise typer.Exit(1)


def _remove_partial_workspace(target: Path) -> None:
    """Undo a scaffold that failed halfway, best effort.

    Only ever called on the failure path, and only on a tree THIS invocation just made — the
    refusal above guarantees nothing was there a moment ago. Failures here are ignored on purpose:
    the user is already being told the real error, and a second one about the cleanup helps nobody.
    """
    try:
        shutil.rmtree(target)
    except OSError:
        pass


def _scaffold_workspace(
    *,
    endpoint: str | None,
    model: str | None,
    config_dir: str | None,
    next_steps: bool = True,
) -> None:
    """`localharness init --workspace`: create ./.localharness/ for the project you are in.

    `next_steps=False` for the one other caller: `start` offering to create a workspace mid-
    startup (`cli/workspace.offer_workspace_creation`). Telling a user who is already starting the
    harness to "run `localharness start`" reads as a command that did not notice what it was doing;
    that session says what it does instead, which is carry on with the new layer already active.

    Deliberately NOT `discover_workspace_dir()`. Creating a workspace is an explicit act at an
    explicit place; discovery is for FINDING one. Walking up-tree here would silently scaffold
    into a parent project when you meant to start a new one.

    Prompt-free by design (dogfood F3: EOF-aborts break scripts) and never destructive: an
    existing workspace is refused with exit 1 even under --force, so this command cannot lose
    a config you wrote. Exit 1, not plain init's interactive exit 0 — this path is script-facing
    and "already there" must be distinguishable from "created" (orchestrator ruling, phase 43).
    """
    conflicts = [
        name
        for name, given in (("--endpoint", endpoint), ("--model", model), ("--config-dir", config_dir))
        if given is not None
    ]
    if conflicts:
        err_console.print(
            f"[bold red]Error:[/bold red] --workspace cannot be combined with "
            f"{', '.join(conflicts)}. A workspace layer never carries a provider block, and it is "
            f"always created in the current directory."
        )
        raise typer.Exit(2)

    try:
        target = Path.cwd() / WORKSPACE_DIR_NAME
    except OSError as exc:
        # H6: the working directory was deleted out from under this process. `Path.cwd()` is the
        # first thing this command touches, so there is nowhere to scaffold and nothing to say
        # except which call failed.
        report_filesystem_error(exc, "find the current directory", console=err_console)
        raise  # pragma: no cover - report_filesystem_error always exits

    # E-M4. Standing in $HOME, `./.localharness` IS the machine's global config dir — so the
    # refusal below fired with "a workspace already exists", which is a true sentence about the
    # wrong thing: that directory is not a workspace, and the user has not made a mistake this
    # command can undo by pointing them at their own config.yaml. Checked before the existence
    # test, because otherwise the more specific case never gets to speak.
    if _is_the_global_config_dir(target):
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"{target} is your machine-wide config directory, not a project workspace.")
            + " --workspace creates a layer for one project — run it from that project's "
            "directory. Use 'localharness init' (no --workspace) to set up this machine.",
            soft_wrap=True,
        )
        raise typer.Exit(2)

    # `is_symlink()` as well as `exists()` (H2): `exists()` FOLLOWS symlinks, so a `.localharness`
    # pointing at a deleted tree answered False, sailed past this refusal, and died in mkdir with
    # FileExistsError. A link that is there is there, whatever it points at.
    #
    # This is the friendly answer, not the guarantee — see the claim below.
    if target.is_symlink() or target.exists():
        _refuse_existing(target)

    # THE CLAIM, and the one operation this command's correctness rests on. A bare `mkdir()` is
    # atomic in the kernel: run ten of these at once in one directory and exactly one succeeds,
    # the other nine get EEXIST. Checking `exists()` first and then creating is a race with a
    # window between the two calls, and losing it used to mean a traceback (E cluster (d)) — nine
    # processes crashing over a workspace that was created perfectly well.
    #
    # `parents=False` on purpose: the parent is the current directory, which exists by definition
    # unless it was deleted underneath us — and that is a different message, not a silent mkdir -p.
    try:
        target.mkdir()
    except FileExistsError:
        # Lost the race, or the directory appeared between the check above and here. Same answer
        # either way, and the same answer the check gives: somebody else's workspace is there.
        _refuse_existing(target)
    except _HANDLED as exc:
        # Nothing was created, so nothing is cleaned up.
        report_filesystem_error(
            exc, f"create a workspace in {Path.cwd()}", console=err_console, paths=[target]
        )

    # Past the claim this process OWNS `target`, which is what makes the cleanup below safe: a
    # failure here removes a tree we just made, never one a concurrent winner is filling in. A
    # half-made workspace left behind an error message is the worst outcome available — the next
    # run would refuse to touch it.
    try:
        (target / "agents").mkdir()
        (target / "config.yaml").write_text(_WORKSPACE_CONFIG_TEMPLATE, encoding="utf-8")
        (target / "plugins").mkdir()
        (target / "plugins" / "README.md").write_text(WORKSPACE_PLUGINS_README, encoding="utf-8")
        (target / ".gitignore").write_text(WORKSPACE_GITIGNORE, encoding="utf-8")
    except _HANDLED as exc:
        _remove_partial_workspace(target)
        report_filesystem_error(
            exc, f"create a workspace in {Path.cwd()}", console=err_console, paths=[target]
        )

    # escape() around every path, markup outside it: `target` is `Path.cwd()/.localharness` and
    # a project folder named `[old] proj` is legal everywhere while `[old]` is rich markup.
    # Unescaped, the three lines that tell you WHERE your new workspace is would name a
    # directory that does not exist (39-05). soft_wrap so a deep path stays one copyable line.
    console.print("[green]✓[/green] " + escape(f"Workspace created at {target}"), soft_wrap=True)
    console.print(
        escape(f"  Config:  {target / 'config.yaml'}") + " (all comments — nothing is set yet)",
        soft_wrap=True,
    )
    console.print(escape(f"  Agents:  {target / 'agents'}"), soft_wrap=True)
    if next_steps:
        console.print(
            "  Next:    run `localharness start` from anywhere in this project — its memory, "
            "sessions and logs now stay here."
        )


def is_interactive(no_input: bool) -> bool:
    """Is there a person to ask? Not with --no-input, not without a terminal on stdin. Reads this
    module's `sys`, so a test that swaps init_cmd.sys for a fake terminal reaches every prompt."""
    return not no_input and sys.stdin is not None and sys.stdin.isatty()


@dataclass(frozen=True)
class SetupResult:
    """What core_setup did: the config on disk afterwards (None: nothing was written) and whether
    a model server answered — `start` goes on into a session only then."""
    config_file: Path | None
    server_ready: bool


@dataclass
class InitRecords:
    """A model server init writes as a record instead of finding it running: the primary launch
    record, its peers, the block shown before the write, and what the write needs besides. The
    guided setup returns one; discovery and the imports build the same carrier."""
    server: ManagedServerConfig
    served: str
    peers: list[EndpointRef] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # the shown block, printed before the write
    api_key: SecretStr | None = field(default=None, repr=False)  # an inline key bound for provider.api_key; unwrapped only where it is written
    deferred: list[Callable[[], None]] = field(default_factory=list)  # ledger writes, after the write
    base_url: str | None = None  # None: http://127.0.0.1:<server.port>/v1
    serving: bool = False  # a server already answers for it
    label: str = ""  # who it was imported from (the unit, or pid and command), for the audit detail

    def __post_init__(self) -> None:
        if isinstance(self.api_key, str):  # a caller that still hands the key over as text
            self.api_key = SecretStr(self.api_key)

    @property
    def launchable(self) -> bool:
        return bool(self.server.binary or self.server.docker_image)


@dataclass(frozen=True)
class Attached:
    """A model server init attaches to and writes no `server:` record for (D-25, INIT-10): the one
    line that says why, the model the provider names, and whether `start` goes on into a session
    (False when the server is keyed and its key is not ours to give)."""
    line: str
    default_model: str
    server_ready: bool = True
    result: DetectorResult | None = None  # the detection the provider is written from


def init_app(
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint", "-e",
            help="Override auto-detection. Full base URL: http://localhost:8081/v1",
            envvar="LOCALHARNESS_ENDPOINT",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m",
            help="Override model selection (use with --endpoint).",
            envvar="LOCALHARNESS_MODEL",
        ),
    ] = None,
    config_dir: Annotated[
        str | None,
        typer.Option(
            "--config-dir",
            help=(
                "Directory for LocalHarness config and agent data. "
                "Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness."
            ),
            envvar="LOCALHARNESS_DIR",
            show_default=False,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Start over without asking: rewrite the config and clear the model and server "
                 "choices saved in overrides.yaml.",
        ),
    ] = False,
    workspace: Annotated[
        bool,
        typer.Option(
            "--workspace",
            help="Scaffold ./.localharness/ for THIS project instead of configuring the machine. "
                 "Non-interactive; never writes a provider block; never overwrites an existing one.",
        ),
    ] = False,
    no_input: Annotated[bool, typer.Option(
        "--no-input",
        help="Ask nothing: use what is detected and the usual settings, write the config, and "
             "print what to do next.")] = False,
) -> None:
    """Auto-detect local LLM and write initial configuration.

    Writes config to <config-dir>/config.yaml on success. The --help probe-order
    line is derived from detector.DEFAULT_PORTS (see the __doc__ assignment below).
    """
    # Fork BEFORE the line below: `resolve_config_dir` answers with the machine-GLOBAL dir, and a
    # workspace branch placed after it (or reusing `config_path`) scaffolds a perfectly shaped tree
    # into ~/.localharness while passing every "the files exist" assertion.
    if workspace:
        _scaffold_workspace(endpoint=endpoint, model=model, config_dir=config_dir)
        return
    core_setup(config_dir, endpoint=endpoint, model=model, force=force,
               interactive=is_interactive(no_input))


def core_setup(config_dir: str | None, *, endpoint: str | None, model: str | None, force: bool,
               interactive: bool, starting: bool = False) -> SetupResult:
    """Set up the core: find or name the model server, then write the config. Asks nothing about
    plugins (owner ruling 2026-10-03: a plugin is set up when it is turned on); every outcome that
    leaves a config on disk ends with the bundled plugins and the command beside each.

    `interactive` gates every question; without it the run takes what is detected and the usual
    settings. An existing config is never rewritten without `force`: it is kept (asked on a
    terminal, said without one) or — after a "no", or for an explicit --endpoint/--model — the
    change goes to overrides.yaml through a validated overlay write. `force` starts over (R15): a
    fresh config.yaml AND the saved model and server choices cleared from overrides.yaml — both or
    neither; nothing else there is touched. `starting` is a first `start`
    calling in, which goes on into the session: its receipt leaves out "run start" and the star.
    Exits 1, writing nothing, on a request it cannot honour.
    """
    config_path = resolve_config_dir(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    ensure_private_dir(config_path)  # keys, the phone token and every session log land here (SEC-11)
    config_file = config_path / "config.yaml"

    if config_file.exists() and not force:
        changing = endpoint is not None or model is not None
        imported = None if changing else _imported_by_an_existing_install(config_path)
        if isinstance(imported, InitRecords):  # D-36: the change is the import — overrides.yaml, no question
            _show_records(imported)
            if _write_overlay_records(config_path, imported, adopt=_adopts(interactive)):
                overlay_file = ConfigLoader(config_dir=config_path).user_overlay_path
                console.print("[green]✓[/green] "
                              + escape(f"Updated {overlay_file} — your config.yaml is unchanged."),
                              soft_wrap=True)
            else:
                err_console.print(
                    "[bold red]Error:[/bold red] "
                    + escape(f"the settings above do not validate, or {config_path / 'overrides.yaml'} "
                             f"could not be written. Nothing was written; {config_file} is kept."),
                    soft_wrap=True)
                raise typer.Exit(1)  # a write that does not validate writes nothing and exits 1 (WR-06)
            outcome = SetupResult(config_file, True)
        elif isinstance(imported, Attached):  # attach-only: its one line, the config kept
            console.print(imported.line, markup=False, soft_wrap=True)
            console.print("[green]✓[/green] " + escape(f"Kept {config_file}."), soft_wrap=True)
            outcome = SetupResult(config_file, False)
        elif _existing_config(config_path, config_file, interactive, changing=changing):
            outcome = SetupResult(config_file, False)  # kept: no server was asked this run
        else:  # a change: overrides.yaml only, and the posture questions are not asked again
            result, selected_model = _pick_server(config_path, endpoint, model, interactive,
                                                     changing=True)
            _write_change(config_path, result, selected_model,
                          _probe_capabilities(result, selected_model))
            outcome = SetupResult(config_file, True)
    else:
        picked = _pick_server(config_path, endpoint, model, interactive, force=force)
        if isinstance(picked, SetupResult):  # skipped for now: saved unchecked, or nothing saved
            outcome = picked
        elif isinstance(picked, Attached):  # attached only: provider written, one line, no record
            console.print(picked.line, markup=False, soft_wrap=True)
            config_file = _write_attached(config_path, picked.result, picked, force=force)
            _receipt(config_file, starting)
            outcome = SetupResult(config_file, picked.server_ready)
        elif isinstance(picked, InitRecords):  # a record: shown, written, launched by the first start
            _show_records(picked)
            config_file = _write_records(config_path, picked, force=force,
                                        adopt=_adopts(interactive))
            _receipt(config_file, starting)
            outcome = SetupResult(config_file, picked.serving or picked.launchable)
        else:
            result, selected_model = picked
            cap = _probe_capabilities(result, selected_model)
            _write_config(config_path, result, selected_model, cap,
                          force=force)
            _receipt(config_file, starting)
            outcome = SetupResult(config_file, True)
    if outcome.config_file is not None:  # fresh, kept, changed or saved unchecked
        console.print()
        for line in _plugin_lines(config_path):
            console.print(escape(line), soft_wrap=True)
    return outcome


def _existing_config(config_path: Path, config_file: Path, interactive: bool, *,
                     changing: bool = False) -> bool:
    """A config is already there and --force was not given: True keeps it (the caller is done),
    False goes on to change it — in overrides.yaml, never config.yaml, so a re-run never wipes a
    config. `changing` (an explicit --endpoint/--model) is the request to change: nothing is asked.

    A config that cannot be read exits 1 first: neither keeping it nor validating a change on top
    of it would be honest, and `--force` is the way out the message names."""
    try:
        provider = ConfigLoader(config_dir=config_path).load_harness().provider
    except Exception as exc:  # noqa: BLE001 — whatever stops the read is reported, never a traceback
        # A ConfigValidationError's first line is only the file's name; its first error is the why.
        first = exc.errors[0] if isinstance(exc, ConfigValidationError) and exc.errors else exc
        reason = (str(first).splitlines() or [type(exc).__name__])[0]
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"{config_file} exists but cannot be read ({reason}). Fix it, or run "
                     "`localharness init --force` to start over."),
            soft_wrap=True,
        )
        raise typer.Exit(1)
    if changing:
        return False
    if interactive and not Confirm.ask(
        escape(f"Config exists: {provider.default_model} at {provider.base_url}. Keep it?"),
        default=True,
    ):
        return False
    console.print("[green]✓[/green] " + escape(f"Kept {config_file}."), soft_wrap=True)
    if not interactive:
        console.print(escape("  To start over: localharness init --force"))
    return True


def _imported_by_an_existing_install(config_path: Path) -> InitRecords | Attached | None:
    """D-36: an existing config.yaml that sets no `server:` may import the unit or process behind its
    provider. A config that cannot be read imports nothing — `_existing_config` reports it."""
    try:
        harness = ConfigLoader(config_dir=config_path).load_harness()
    except Exception:  # noqa: BLE001 — whatever stops the read is _existing_config's to report
        return None
    return _existing_install_import(config_path, harness)


def _pick_server(
    config_path: Path, endpoint: str | None, model: str | None, interactive: bool, *,
    changing: bool = False, force: bool = False,
) -> tuple[DetectorResult, str] | SetupResult | InitRecords | Attached:
    """The model server and the model: the explicit --endpoint, else detection — and, finding
    nothing on a terminal, the guided vLLM setup, then "skip for now" (whose SetupResult comes
    back as is). A change (`changing`) never sets a server up: finding nothing keeps the config
    and exits 1. Exits 1 on what it cannot honour."""
    if endpoint is not None:
        # Skip probe — build result manually
        base_url = _build_base_url_for_endpoint(endpoint)
        if model is None:
            # #118: ask the server before demanding --model. One model served = nothing to
            # disambiguate; otherwise keep the error but name the ids AND where they came from,
            # since the id is usually a long path-like string the user has to go dig out.
            listing = f"{base_url.rstrip('/')}/models"
            served = _list_endpoint_models(base_url)
            if served is None:
                err_console.print(
                    "[bold red]Error:[/bold red] --model is required when using --endpoint — "
                    f"and {listing} could not be read to pick one for you."
                )
                raise typer.Exit(1)
            if len(served) == 1:
                model = served[0]
                console.print(
                    f"  [green]✓[/green] Using [bold]{model}[/bold] — the only model served at {listing}"
                )
            elif not served:
                err_console.print(
                    "[bold red]Error:[/bold red] --model is required when using --endpoint — "
                    f"and {listing} lists no models. Load a model on the server, then run "
                    "`localharness init` again."
                )
                raise typer.Exit(1)
            else:
                err_console.print(
                    f"[bold red]Error:[/bold red] --model is required when using --endpoint. "
                    f"{listing} serves {len(served)} models:"
                )
                for served_id in served:
                    err_console.print(f"  - {served_id}")
                err_console.print("Re-run with --model <one of the ids above>.")
                raise typer.Exit(1)
        provider_type = _identify_endpoint_provider(base_url)
        if provider_type == "unknown":
            console.print(
                "  [yellow]⚠[/yellow]  Could not identify the runtime at this endpoint — recording "
                "provider_type: unknown (token counting falls back to approximate)."
            )
        else:
            console.print(f"  [green]✓[/green] Runtime: [bold]{provider_type}[/bold]")
        result = DetectorResult(
            found=True,
            provider_type=provider_type,
            base_url=base_url,
            models=[model],
            suggested_model=model,
            probe_duration_ms=0.0,
        )
        return result, model

    console.print("Probing for local LLM...")
    result = asyncio.run(detect_provider(timeout_seconds=1.0))
    if not result.found:
        if changing:  # guided setup would write a fresh config.yaml — a change never does
            _say_not_detected()
            console.print("\n" + escape(f"Nothing changed: {config_path / 'config.yaml'} is kept."),
                          soft_wrap=True)
            raise typer.Exit(1)
        presets = _discover_presets(config_path)  # D-13: GGUFs found are written, not asked about
        if presets is not None:
            return presets
        guided = _guided_setup(config_path) if interactive else None
        if guided is None:
            _say_not_detected()  # D-13: only an outcome that writes no record says nothing was found
            if interactive:
                return _skip_for_now(config_path, model, force=force)
            console.print(
                "\nStart your LLM server and run 'localharness init' again, or use:"
            )
            console.print(
                "  localharness init --endpoint http://your-host:port/v1 --model your-model-name"
            )
            raise typer.Exit(1)
        return guided

    console.print(f"  [green]✓[/green] {result.provider_type} found at {result.base_url}")
    if not changing:  # D-26: the listener decides before the model list is judged (a 401 has none)
        imported = _import_listener(config_path, result)
        if imported is not None:
            return imported
    if len(result.models) == 0:
        err_console.print("[bold red]Error:[/bold red] No models available at detected endpoint.")
        raise typer.Exit(1)
    if len(result.models) == 1:
        console.print(f"  Model: [bold]{result.models[0]}[/bold] (auto-selected)")
        return result, result.models[0]
    # Multiple models — the one --model named, Ollama's hot model, a question, or the first
    return result, _select_model(result, model, interactive)


def _say_not_detected() -> None:
    """The red line and the list of ports probed: said only when nothing was found AND no record
    is about to be written (a guided setup that writes one never says it)."""
    console.print("\n[bold red]✗ No local LLM detected.[/bold red]\n")
    console.print("Checked:")
    for port in DEFAULT_PORTS:
        name = _PORT_LABELS.get(port, "unknown")
        console.print(f"  http://localhost:{port}  ({name})  — connection refused")


# ------------------------------------------------------------------ presets from the GGUFs found
#
# Nothing is serving and the machine holds GGUFs: init writes them as presets (a primary `server:`
# and peers), shows them in one block and asks nothing and launches nothing (D-10, D-13, D-14, D-39).

PRIMARY_PORT = 8081


def _machine_autosense(config_path: Path) -> AutosenseConfig:
    """The `autosense:` section the machine's files set (its `model_roots` decide where init looks),
    else the defaults; a config that cannot be read never stops discovery (D-51)."""
    try:
        return ConfigLoader(config_dir=config_path).load_harness().autosense
    except Exception:  # noqa: BLE001 — no config yet, or an unreadable one: never a traceback
        pass
    try:  # a fresh machine has no config.yaml, but may already have an overrides.yaml saying where
        raw = load_overlay(config_path / "overrides.yaml").get("autosense")
        return AutosenseConfig.model_validate(raw) if isinstance(raw, dict) else AutosenseConfig()
    except Exception:  # noqa: BLE001 — whatever stops that read leaves the defaults
        return AutosenseConfig()


def _found_models(config_path: Path, *, unit_model: str | None = None) -> list[DiscoveredModel]:
    """The GGUF models under the machine's `model_roots`, best first (D-07): the unit's model, one the
    speed ledger recorded, one that fits the free memory, the newest. Free memory is probed once for
    all of them (D-09, D-49) and not at all when there is no model."""
    auto = _machine_autosense(config_path)
    found = discover(walk_ggufs(auto.model_roots)).models
    if not found:
        return []
    ledger = speed_stats.model_paths(speed_stats.default_speed_stats_path(config_path))
    memory = autosense.free_memory(auto)
    fits: dict[str, int | None] = {}
    for model in found:
        try:
            spec = ManagedServerConfig(runtime="llamacpp", model=str(model.path),
                                       mmproj=str(model.mmproj) if model.mmproj else None)
            fits[str(model.path)] = autosense.compute_fit(spec, auto, memory=memory).fits
        except Exception:  # noqa: BLE001 — one model that cannot be sized is ranked without a fit
            fits[str(model.path)] = None
    return rank_models(found, unit_model=unit_model, ledger_paths=ledger, fits=fits)


def _free_ports(start: int, count: int, *, taken: Collection[int]) -> list[int]:
    """The next `count` ports from `start` upward that no record uses (`taken`) and that nothing is
    listening on (D-11)."""
    ports: list[int] = []
    port = start
    while len(ports) < count and port <= 65535:
        if port not in taken and not procfs.listeners(port):
            ports.append(port)
        port += 1
    return ports


def _peer_records(models: Sequence[DiscoveredModel], *, start: int, taken_names: Collection[str],
                  taken_ports: Collection[int]) -> list[EndpointRef]:
    """One launchable peer per model (D-10): a GPU llama.cpp record on the next free port from
    `start`, named as the preset (`unique_preset_name` over every name already used), carrying
    only the model, its projector, its served name and its port — no binary, args, env or key file."""
    ports = _free_ports(start, len(models), taken=taken_ports)
    names = set(taken_names)
    peers: list[EndpointRef] = []
    for model, port in zip(models, ports):
        name = unique_preset_name(model.name, names)
        names.add(name)
        peers.append(EndpointRef(
            name=name,
            base_url=f"http://127.0.0.1:{port}/v1",
            provider_type="llamacpp",
            gpu=True,
            lifecycle=ManagedServerConfig(
                runtime="llamacpp", model=str(model.path),
                mmproj=str(model.mmproj) if model.mmproj else None,
                served_name=name, port=port, gpu=True),
        ))
    return peers


def _preset_clause(server: ManagedServerConfig) -> str:
    """`(llama.cpp)` or `(llama.cpp, +<projector basename>)`, every value through visible_text."""
    clause = RUNTIME_LABELS.get(server.runtime, server.runtime)
    if server.mmproj:
        clause += ", +" + visible_text(os.path.basename(server.mmproj))
    return f"({clause})"


def _preset_lines(server: ManagedServerConfig, served: str, peers: Sequence[EndpointRef], *,
                  binary_from: str, extras: Sequence[str] = ()) -> list[str]:
    """D-14: the primary's line, with its binary and where it came from when one was found, then
    the import's `extras` (already escaped lines), then one line per peer saying what `/model`
    switches to. Every value is escaped; printed `markup=False`."""
    primary = f"serving {visible_text(served)} {_preset_clause(server)} ← {shown_path(server.model)}"
    if server.binary:
        primary += f" · binary: {shown_path(server.binary)} ({binary_from})"
    lines = [primary, *extras]
    for peer in peers:
        life = peer.lifecycle
        if life is not None:
            lines.append(f"/model switches to {visible_text(peer.name)} {_preset_clause(life)} "
                         f"← {shown_path(life.model)}")
    return lines


def _no_binary_line(models: Sequence[DiscoveredModel], roots: Sequence[str]) -> str:
    """D-24: the models were found and there is no llama-server to launch them with — the roots
    that hold one are named, and the record is written declared (`server.binary` unset)."""
    holding: list[str] = []
    for root in roots:
        real = os.path.realpath(os.path.expanduser(root))
        if any(m.real == real or m.real.startswith(real.rstrip(os.sep) + os.sep) for m in models):
            holding.append(shown_path(os.path.expanduser(root)))
    n = len(models)
    return (f"found {n} GGUF{'s' if n != 1 else ''} under {', '.join(holding)}; "
            "no llama-server — set server.binary")


def _unit_skip_line(unit: str, reason: str) -> str:
    """D-24, first form: a unit that names a model could not be imported and no llama-server is on
    PATH either — the unit's own reason (a predicate from units.py) folded into the one line."""
    return f"{visible_text(unit)} {visible_text(reason)}; no llama-server found — set server.binary"


def _daemon_lines() -> list[str]:
    """D-41, D-54: an Ollama or LM Studio that answers holds models of its own; say how many and
    which (the first three), that they stay there and that no preset was written for them. Each
    daemon is asked on its own port: `detect_provider(timeout_seconds=1.0, ports=[11434])`, then
    `ports=[1234]`."""
    lines: list[str] = []
    for ports in ([11434], [1234]):
        try:
            found = asyncio.run(detect_provider(timeout_seconds=1.0, ports=ports))
        except Exception:  # noqa: BLE001, S112 — a probe that fails says nothing
            continue
        if not found.found or found.provider_type not in ("ollama", "lmstudio") or not found.models:
            continue
        label = RUNTIME_LABELS[found.provider_type]
        names = ", ".join(visible_text(m) for m in found.models[:3])
        if len(found.models) > 3:
            names += ", …"
        count = f"{len(found.models)} model{'s' if len(found.models) != 1 else ''}"
        lines.append(f"{label} holds {count} ({names}) — left in {label}, no preset written")
    return lines


def _unit_import(config_path: Path, *, only_port: int | None = None
                 ) -> tuple[InitRecords | None, tuple[str, str] | None]:
    """D-21, D-23, D-50: nothing answers, so the user's own units are sensed (units.user_units). The
    first unit whose journal, else unit file, yields a record is imported as `server:` (its model
    leading the ranking, discovery's other models its peers). Returns (records, None), or
    (None, the first skip that names a model as `(unit, reason)`, else None): an `AttachOnly` from a
    stopped unit has nothing to attach to, so it is that unit's skip. With `only_port` (an existing
    install's `provider.base_url` port, WR-05) a unit that serves another port is not the provider's
    server and is passed over without a word."""
    first_skip: tuple[str, str] | None = None
    ini = init_import.read_config_ini(init_import.user_config_ini())
    for state in units.user_units():
        res = units.unit_record(state)
        if isinstance(res, units.Skip):
            if res.names_model and first_skip is None:
                first_skip = (state.unit, res.reason)
            continue
        source, rec = res
        imp = init_import.import_record(rec, source=source, binary_resolver=which_outside_cwd, ini=ini)
        if isinstance(imp, init_import.AttachOnly):
            if first_skip is None:
                first_skip = (state.unit, imp.reason)
            continue
        if only_port is not None and imp.spec.port != only_port:
            continue  # an unrelated llama-server unit (an embedding server on :8090) is not this provider
        gap = units.journal_environment_gap(state, rec) if source == "journal" else None
        if gap is not None:  # WR-04: the unit's own Environment= could not be read: say so, import nothing silently
            imp = replace(imp, shown=(*imp.shown, f"environment of {visible_text(state.unit)} not read "
                                      f"({visible_text(gap)})"))
        holder = workspace.Holder(state.unit, None, None, os.getuid(), state)
        recs = _records_from_import(config_path, imp, holder, port=imp.spec.port, running=False,
                                    base_url=None, unit_model=os.path.realpath(imp.spec.model))
        return recs, None
    return None, first_skip


def _discover_presets(config_path: Path) -> InitRecords | None:
    """Nothing is serving: the person's own stopped llama-server unit becomes the primary `server:`
    when it can be read (D-21); else the GGUFs under the model roots become a primary (the best
    ranked, port 8081, the llama-server found on PATH when there is one) and peers, written as
    records by the first start's machinery — init asks nothing and launches nothing. None when no
    GGUF model is found (the caller goes on to the guided setup)."""
    from_unit, skipped = _unit_import(config_path)
    if from_unit is not None:
        return from_unit
    models = _found_models(config_path)
    if not models:
        return None
    first, rest = models[0], models[1:]
    binary = which_outside_cwd("llama-server")
    served = unique_preset_name(first.name, [])
    server = ManagedServerConfig(
        runtime="llamacpp", model=str(first.path),
        mmproj=str(first.mmproj) if first.mmproj else None,
        served_name=served, port=PRIMARY_PORT, binary=binary)
    peers = _peer_records(rest, start=PRIMARY_PORT + 1, taken_names={served},
                          taken_ports={PRIMARY_PORT})
    lines = _preset_lines(server, served, peers, binary_from="on PATH")
    if binary is None:  # D-24: one line, naming the unit that could not be used when there is one
        lines.append(_unit_skip_line(*skipped) if skipped is not None
                     else _no_binary_line(models, _machine_autosense(config_path).model_roots))
    return InitRecords(server=server, served=served, peers=peers, lines=lines)


# ------------------------------------------------------------------ the server already running
#
# A llama-server already answers on the detected port: its process becomes the primary `server:`
# record (D-26), shown with every value it took and one line saying who holds the port (D-29, D-32).

_autosense_mod = autosense  # `_second_line` takes a keyword named `autosense`, which shadows the module


def _import_listener(config_path: Path, result: DetectorResult) -> InitRecords | Attached | None:
    """D-25, D-26, D-53, D-54: the process listening on the detected port, before the model list is
    judged. An own-uid llama-server becomes records; every case the import cannot reproduce becomes an
    `Attached` (provider only, one line, exit 0). None means "today's attach path": nothing listens, or
    a listener this import has no business with (another uid answering 200, a user-owned process that
    is not llama-server). Only a listener that answered HTTP gets here, so a user-owned process that
    does not answer is never imported."""
    try:
        port = urlparse(result.base_url).port
    except ValueError:
        return None
    if port is None:
        return None
    found = procfs.listeners(port)
    if not found:
        return None
    # D-54: "answered 401" is the error code in the body the detector kept
    error = result.raw_response.get("error") if isinstance(result.raw_response, dict) else None
    answered_401 = isinstance(error, dict) and error.get("code") == 401
    me = os.getuid()
    mine = procfs.reachable_first([lst for lst in found if lst.uid == me])
    if not mine:  # another uid's listener is known by the uid column alone (T-5-59)
        if not answered_401:
            return None
        uid = found[0].uid
        try:
            import pwd  # function-local (D-71): the module imports where pwd is absent

            name = pwd.getpwuid(uid).pw_name
        except (ImportError, KeyError):
            name = str(uid)
        line = (f":{port} is keyed and owned by uid {uid} ({visible_text(name)}) — set provider.api_key; "
                "attached only, no server record written")
        return Attached(line, result.suggested_model, server_ready=False, result=result)
    pid = procfs.pid_for_inode(mine[0].inode)
    if pid is None:
        return None
    rec = procfs.read_process(pid)
    if isinstance(rec, procfs.ProcRefusal):
        state = units.unit_file_state(rec.unit) if rec.unit else None
        who = workspace.incumbent_subject(workspace.Holder(rec.unit, pid, None, me, state), port=port)
        return Attached(f"{who} {rec.reason}{init_import.ATTACH_ONLY_SUFFIX}", result.suggested_model,
                        result=result)
    runtime = init_import.runtime_of(rec)
    if runtime is None:
        return None
    holder = workspace.holder_of(rec)
    who = workspace.incumbent_subject(holder, port=port)
    if runtime != "llamacpp":  # a keyed Ollama or LM Studio has no place for a key; unkeyed attaches as before
        if not answered_401:
            return None
        return Attached(f"{who} runs {RUNTIME_LABELS[runtime]}, which cannot carry a key"
                        f"{init_import.ATTACH_ONLY_SUFFIX}", _attach_model(rec, result), result=result)
    imp = init_import.import_record(
        rec, source="live", binary_resolver=which_outside_cwd, port=port,
        ini=init_import.read_config_ini(init_import.user_config_ini()))
    if isinstance(imp, init_import.AttachOnly):
        return Attached(f"{who} {imp.reason}{init_import.ATTACH_ONLY_SUFFIX}", _attach_model(rec, result),
                        result=result)
    return _records_from_import(config_path, imp, holder, port=port, running=True,
                                base_url=result.base_url)


def _argv_value(argv: Sequence[str], short: str, long: str) -> str | None:
    """The value after `short` or `long` (space or `=` form, `_` read as `-`), else None."""
    for idx, tok in enumerate(argv):
        flag, eq, value = tok.partition("=")
        if flag.startswith("--"):
            flag = flag.replace("_", "-")
        if flag == long and eq:
            return value
        if flag in (short, long) and not eq and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _attach_model(rec, result: DetectorResult) -> str:
    """D-25: the model an attach-only provider names: the process's `-a`/`--alias`, else the stem of
    its `-m`/`--model`, else the detector's suggestion."""
    argv = rec.argv[1:] if rec.argv else ()
    environ = rec.environ or {}
    alias = _argv_value(argv, "-a", "--alias") or environ.get("LLAMA_ARG_ALIAS")
    if alias:
        return alias
    model = _argv_value(argv, "-m", "--model") or environ.get("LLAMA_ARG_MODEL")
    if model:
        return preset_name(model)
    return result.suggested_model


def _write_attached(config_path: Path, result: DetectorResult, attached: Attached, *,
                    force: bool = False) -> Path:
    """D-25: a provider-only config.yaml for an attach: the detected address, the model the attach names,
    no `server:`, no capability probe (so no claim about tool calling), nothing adopted. Memory and
    host tools are on, as in every fresh config."""
    model = attached.default_model
    harness = HarnessConfig(
        version="1",
        provider=ProviderConfig(
            provider_type=result.provider_type,
            base_url=result.base_url,
            api_key="none",
            default_model=model,
            available_models=[model] if model else [],
            supports_function_calling=None,
            timeout_seconds=600.0,
        ),
        org=OrgConfig(
            default_model=model,
            permissions=PermissionConfig(defaults_revision=CURRENT_DEFAULTS_REVISION),
        ),
        server=None,
    )
    return _write_harness(config_path, harness, True, force=force)


def _second_line(imp: init_import.Imported, holder: workspace.Holder, *, port: int, running: bool,
                 autosense: AutosenseConfig) -> tuple[str | None, int | None, int | None]:
    """D-29, D-32: the one line about one fact, from the one builder that also says the attach
    sentence. With a `-c` it is the window line (what the incumbent runs with, what the harness
    launches, and the setting that keeps the incumbent's number); without one, the attach form while
    the server runs and nothing once it is stopped. Returns (line, fit, trained): the triple's last
    two members, for the ledger. `context` is never set from the import."""
    if imp.ctx is None:
        if not running:
            return None, None, None
        return workspace.incumbent_sentence(holder, port=port, form="attach"), None, None
    fit: int | None = None
    trained: int | None = None
    try:
        if running:  # Phase 2 D-09: the window the incumbent serves is added back to the free memory
            fi = _autosense_mod.compute_fit(imp.spec, autosense, attached_served=imp.ctx)
        else:
            fi = _autosense_mod.compute_fit(imp.spec, autosense)
        fit = fi.fits
        trained = fi.header.get("context_length")
    except Exception:  # noqa: BLE001 — a fit that cannot be computed reads "with no -c", never a traceback
        pass
    window = workspace.window_clause(imp.ctx, fit, running=running)
    return workspace.incumbent_sentence(holder, port=port, form="attach", window=window), fit, trained


def _records_from_import(config_path: Path, imp: init_import.Imported, holder: workspace.Holder, *,
                         port: int, running: bool, base_url: str | None,
                         unit_model: str | None = None) -> InitRecords:
    """D-08, D-10, D-14, D-29: an Imported record is the primary; the other models discovery finds are
    its peers (next free ports above it, no binary or args of their own). The block is the primary
    line, the import's extras, the second line, then the peers; the weights path and the window triple
    are written to the ledger once the config is on disk. `unit_model` (D-64) is the stopped unit's
    model, which leads the ranking; None (a running process) ranks as discovery alone does."""
    spec = imp.spec
    real = os.path.realpath(os.path.expanduser(spec.model))
    rest = [m for m in _found_models(config_path, unit_model=unit_model) if m.real != real]
    peers = _peer_records(rest, start=spec.port + 1, taken_names={spec.served_name},
                          taken_ports={spec.port})
    if imp.binary_source == "process":
        binary_from = "from " + workspace.incumbent_subject(holder, port=port)
    else:
        binary_from = "on PATH"
    line, fit, trained = _second_line(imp, holder, port=port, running=running,
                                      autosense=_machine_autosense(config_path))
    extras = (*imp.shown, line) if line else imp.shown
    lines = _preset_lines(spec, spec.served_name, peers, binary_from=binary_from, extras=extras)
    ledger = speed_stats.default_speed_stats_path(config_path)
    served = spec.served_name
    deferred: list[Callable[[], None]] = []
    if imp.ctx is not None:  # the triple that was printed, so the first start does not print it again
        def _window() -> None:
            try:
                speed_stats.record_window_report(ledger, "llamacpp", served, imp.ctx, trained, fit)
            except OSError:
                pass  # a read-only config dir must not break init

        deferred.append(_window)

    def _weights() -> None:
        try:
            speed_stats.record_model_path(ledger, "llamacpp", served, spec.model)
        except OSError:
            pass  # a read-only config dir must not break init

    deferred.append(_weights)
    return InitRecords(
        server=spec, served=served, peers=peers, lines=lines,
        api_key=imp.api_key,
        deferred=deferred, base_url=base_url, serving=running,
        label=workspace.incumbent_subject(holder, port=port))


def _show_records(recs: InitRecords) -> None:
    """Print the block a record path shows before it writes (markup off: a name from a file is
    text). A llama.cpp record also says which models an Ollama or LM Studio holds (D-41)."""
    lines = list(recs.lines)
    if recs.server.runtime == "llamacpp":
        lines += _daemon_lines()
    for line in lines:
        console.print(line, markup=False, soft_wrap=True)


def _skip_for_now(config_path: Path, model: str | None, *, force: bool = False) -> SetupResult:
    """No server answered and the guided setup was declined, on a terminal: save the address and
    the model the user will use, unchecked — or save nothing and say the one next step. A config
    is written only when it names a model (never an incomplete one), and skipping is the user's
    choice, so it exits 0 either way. A bad address is refused before the model is asked."""
    console.print("Skip for now: give the address and model you will use, and init saves them without checking.")
    url = Prompt.ask("Model server address", default="http://localhost:8081/v1").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        console.print("Nothing saved: the address must start with http:// or https://.")
        return SetupResult(None, False)
    name = model or Prompt.ask("Model name (Enter to skip)", default="", show_default=False).strip()
    if not name:
        console.print(
            escape(f"Nothing saved. When a model server answers at {url}, run `localharness init` again."),
            soft_wrap=True,
        )
        return SetupResult(None, False)
    config_file = _write_harness(config_path, HarnessConfig(
        version="1",
        provider=ProviderConfig(
            provider_type="unknown",
            base_url=url,
            api_key="none",
            default_model=name,
            available_models=[name],
            timeout_seconds=600.0,
        ),
        org=OrgConfig(
            default_model=name,
            permissions=PermissionConfig(defaults_revision=CURRENT_DEFAULTS_REVISION),
        ),
        server=None,
    ), memory_on=True, force=force)
    console.print(
        "[green]✓[/green] "
        + escape(f"Saved {config_file} — not checked yet. Start your model server, then run `localharness start`."),
        soft_wrap=True,
    )
    return SetupResult(config_file, False)


def _probe_capabilities(result: DetectorResult, selected_model: str) -> CapabilityResult:
    """Ask the chosen server for its tool-calling mode; exit 1, writing nothing, when it never
    answered."""
    llm_cfg = LLMConfig(
        base_url=result.base_url,
        model=selected_model,
        timeout_seconds=300.0,
    )
    client = LLMClient(llm_cfg)

    async def _probe() -> CapabilityResult:
        # Closed in this loop, which owns its connections. Left open, the SDK finalizes it on
        # whatever loop is running when the GC reaches it — a first start's session loop, where
        # closing a transport of this closed loop raised "Event loop is closed".
        async with client:
            return await client.detect_capabilities()

    cap = asyncio.run(_probe())

    # Reachability is a SEPARATE axis from capability (CapabilityResult.server_reached): a probe
    # that never reached the server proves nothing about tool calling, so reporting "✓ configured"
    # and persisting supports_function_calling from it fabricates a verified setup.
    if not cap.server_reached:
        err_console.print(
            f"[bold red]Error:[/bold red] no answer from {result.base_url} ({cap.probe_error}). "
            "No config was written — start the model server (or fix --endpoint/--model) and run "
            "`localharness init` again."
        )
        raise typer.Exit(1)

    if cap.tool_call_mode == "native":
        console.print("  [green]✓[/green] Tool calling: native")
    else:
        console.print("  [yellow]⚠[/yellow]  Tool calling: XML fallback (less reliable than native)")
    return cap


def _served_window(result: DetectorResult, *, cfg_dir: Path | None = None, model: str | None = None,
                   deferred: "list[Callable[[], None]] | None" = None) -> int | None:
    """The served context window to write as the context budget, or None — saying which, and why.

    Fit the context budget to the served window when the provider reports it — a budget above
    the real window disables compaction and kills long turns. llama.cpp's served and trained
    windows come from `probe_window`; LM Studio keeps `_detect_lmstudio_ctx` (the model-free route
    its identification takes) and the rest keep `_detect_max_model_len` (D-33). `model` is the model
    the config will name (default: the detector's suggestion); the probe and the ledger key are for
    that model, the one `start` will resolve. The probed triple is recorded under `cfg_dir` so the
    first `start` compares against it; with `deferred` the record is appended there as a thunk for
    the caller to run once its write has succeeded, so a write that fails leaves no entry. The window
    line with the fit and its undo arrives with the import (Phase 5)."""
    model = model or result.suggested_model
    trained: int | None = None
    if result.provider_type == "llamacpp":
        from localharness.agent.context import probe_window
        # "-" when the endpoint listed nothing: the /props fallback needs no id, and probe_window
        # answers (None, None) for an empty model.
        probe = probe_window(result.base_url, model or "-", "llamacpp")
        max_len, trained = probe.served, probe.trained
    elif result.provider_type == "lmstudio":
        max_len = _detect_lmstudio_ctx(result.base_url)
    else:
        max_len = _detect_max_model_len(result.base_url)
        trained = max_len if result.provider_type == "vllm" else None
    if cfg_dir is not None and model and (max_len is not None or trained is not None):
        def _record(served: int | None = max_len, trained: int | None = trained) -> None:
            from localharness.provider.speed_stats import default_speed_stats_path, record_window_report
            try:
                record_window_report(default_speed_stats_path(cfg_dir), result.provider_type,
                                     model, served, trained, None)
            except OSError:
                pass  # a read-only config dir must not break init

        if deferred is not None:
            deferred.append(_record)
        else:
            _record()
    # max_context_tokens IS the served window: the harness subtracts the reply reserve at runtime
    # (agent.context.response_reserve), so subtracting it here too would reserve it twice.
    # Two independent floors: ContextConfig rejects anything under 1_000, and a window that
    # reserves nothing (<= 1_024) cannot run at all — `start` refuses it, so writing it would only
    # persist a budget the next command rejects.
    if max_len and max_len >= 1_000 and response_reserve(max_len) > 0:
        console.print(
            f"  [green]✓[/green] Context budget: {max_len:,} tokens "
            f"(the full served window — the harness reserves output room inside it)"
        )
        return max_len
    if max_len:
        console.print(
            f"  [yellow]⚠[/yellow]  The served window ({max_len:,} tokens) is too small to run "
            "the harness — at or below 1,024 tokens nothing is left to hold the model's reply "
            "once history fits. No context budget was written. Raise the window at the SERVER "
            "(llama.cpp [bold]-c[/bold], [bold]OLLAMA_CONTEXT_LENGTH[/bold], LM Studio's context "
            "length); the practical minimum is 1,025 tokens."
        )
    elif result.provider_type == "ollama":
        console.print(
            "  [yellow]⚠[/yellow]  Ollama's served window (num_ctx) is VRAM-tiered and not "
            "discoverable via the API — keeping the default budget. If your model runs a "
            "larger context, set [bold]OLLAMA_CONTEXT_LENGTH[/bold] and a matching "
            "[bold]org.context.max_context_tokens[/bold] in config.yaml (an oversized message "
            "hard-errors on recent Ollama)."
        )
    return None


# R15: what `init --force` clears from the machine's overrides.yaml — the model and server choices
# the core persists there: a re-run's change (_write_change), `/model` (model_ops.
# persist_default_model: the default model, its list, a managed server's model) and a peer-endpoint
# switch (model_ops.persist_active_endpoint: the whole active_endpoint record). Every other key and
# every plugin section is the user's, and stays.
_SAVED_CHOICE_KEYS = (
    "provider.provider_type", "provider.base_url", "provider.default_model",
    "provider.available_models", "provider.supports_function_calling", "org.default_model",
    "org.context.max_context_tokens", "server.model", "active_endpoint",
)


def _drop_saved_choices(overlay: dict) -> list[str]:
    """Remove the saved model and server choices from `overlay` IN PLACE — a parent the removal
    leaves empty goes too — and return the ones it held, in _SAVED_CHOICE_KEYS order."""
    dropped = []
    for path in _SAVED_CHOICE_KEYS:
        *parents, leaf = path.split(".")
        chain = [overlay]
        for key in parents:
            if not isinstance(chain[-1].get(key), dict):
                break
            chain.append(chain[-1][key])
        else:
            if leaf in chain[-1]:
                del chain[-1][leaf]
                dropped.append(path)
                for i in range(len(parents) - 1, -1, -1):
                    if chain[i + 1]:
                        break
                    del chain[i][parents[i]]
    return dropped


def _saved_choices_reset(config_path: Path, harness: HarnessConfig) -> tuple[Path, dict, list[str]] | None:
    """--force (R15): overrides.yaml without the saved model and server choices, checked BEFORE
    anything is written as the merged config the next start reads (persist_default_model's check).
    None: it holds none, and it is not rewritten. Exits 1, writing nothing, when overrides.yaml
    cannot be read or the reset does not validate — --force resets both files or neither."""
    from localharness.config.overlay import deep_merge
    from localharness.config.plugin_sections import core_harness_view

    overlay_path = ConfigLoader(config_dir=config_path).user_overlay_path
    try:
        overlay = load_overlay(overlay_path)
        if not isinstance(overlay, dict):
            raise ValueError("its top level is not a mapping")
    except Exception as exc:  # noqa: BLE001 — whatever stops the read is reported, never a traceback
        reason = (str(exc).splitlines() or [type(exc).__name__])[0]
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"{overlay_path} cannot be read ({reason}). Fix it, then run "
                     "`localharness init --force` again. Nothing was written."),
            soft_wrap=True,
        )
        raise typer.Exit(1)
    cleared = _drop_saved_choices(overlay)
    if not cleared:
        return None
    merged = deep_merge(harness.model_dump(mode="python"), {k: v for k, v in overlay.items() if k != "agent"})
    try:
        HarnessConfig.model_validate(core_harness_view(merged))
    except ValueError as exc:  # a pydantic ValidationError is one
        # location and message only, scrubbed: never the merged config pydantic carries (R16)
        why = validation_text(exc, secret_values(HarnessConfig, merged))
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"the new settings do not validate: {why}. Nothing was written."),
            soft_wrap=True,
        )
        raise typer.Exit(1)
    return overlay_path, overlay, cleared


def _overlay_shadows(config_path: Path) -> None:
    """The one case where a saved choice still wins over the config.yaml init just wrote: a fresh
    write WITHOUT --force — config.yaml was missing (deleted by hand, or a first `start`) while
    overrides.yaml survived, and nothing leaves overrides.yaml without --force. Say which choices
    win and how to clear them. After --force none is left to name: it clears them first."""
    overlay_path = ConfigLoader(config_dir=config_path).user_overlay_path
    try:
        saved = _drop_saved_choices(load_overlay(overlay_path))
    except Exception:  # noqa: BLE001 — a malformed overrides.yaml is start's to report, not this note's
        return
    if saved:
        console.print(
            "  [yellow]⚠[/yellow]  "
            + escape(f"{overlay_path} still sets {', '.join(saved)} — those win over config.yaml. "
                     "To clear them: localharness init --force"),
            soft_wrap=True,
        )


def _record_dicts_of(server: ManagedServerConfig | None,
                     peers: Sequence[EndpointRef]) -> tuple[dict | None, list[dict]]:
    """The server record and every peer as written to config.yaml: exclude_none dumps with the keys
    as typed, so no `null` preset field and no invented `effort` lands in a record init writes."""
    return (reveal(server.model_dump(mode="python", exclude_none=True)) if server is not None else None,
            [reveal(peer.model_dump(mode="python", exclude_none=True)) for peer in peers])


def _record_dicts(harness: HarnessConfig) -> tuple[dict | None, list[dict]]:
    """`_record_dicts_of` over a config's own server and peers."""
    return _record_dicts_of(harness.server, harness.extra_endpoints)


def _adopts(interactive: bool) -> bool:
    """Does this run adopt what it writes? Only a person at a terminal does (D-37): not `--no-input`,
    not a run without a tty, not a command the shell tool started (it carries LOCALHARNESS_COMMIT),
    so an agent running `init` cannot approve a launch for the next start."""
    return interactive and "LOCALHARNESS_COMMIT" not in os.environ


def _adopt_written(global_dir: Path, file: str, server: dict | None, peers: Sequence[dict]) -> None:
    """Record, as already approved, the launch entries init just wrote and showed (D-37, INIT-12):
    the primary's `server.` keys and each peer's `lifecycle.` keys, built from the dicts that went
    into ``file`` (never a re-read of the launch keys), and the posture entries (`bind_all`,
    `require_api_key`) the machine's files give under exactly those prefixes — those, only while
    they still say what was written (a value another file or a concurrent write changed is left for
    the next start to ask about). With no machine record the first one is made, of the launch kinds
    only; with one, these entries are added or replaced and nothing else changes (`adopt_entries`
    never re-snapshots). No key is ever an entry."""
    written: dict[str, dict] = {}  # entry-name prefix -> the dict written under it
    if server is not None:
        written["server."] = server
    for peer in peers:
        if isinstance(peer.get("lifecycle"), dict):
            written[f"extra_endpoints.{peer['name']}.lifecycle."] = peer["lifecycle"]
    entries = [e for prefix, mapping in written.items()
               for e in trust._launch_entries(file, prefix, mapping)]
    posture = {prefix + key: json.dumps(mapping.get(key))
               for prefix, mapping in written.items() for key in trust._POSTURE_KEYS}
    # `shown` is `<runtime>:<true|false>:<keyed|open>`: the middle part is the value in effect.
    entries += [e for e in trust.machine_snapshot(global_dir)
                if e["kind"] == trust.LAUNCH_KIND and e["name"] in posture
                and str(e.get("shown", "")).split(":")[1:2] == [posture[e["name"]]]]
    if trust.recorded_machine(global_dir) is not None:
        trust.adopt_entries(global_dir, entries)
    else:
        trust.record_machine(global_dir, sorted(entries, key=trust._entry_key),
                             kinds=frozenset({trust.LAUNCH_KIND, trust.LAUNCH_MODEL_KIND}))


def _result_from_record(recs: InitRecords) -> DetectorResult:
    """The detection-shaped answer for a record: nothing was probed, so the address is the one the
    record will serve on and the one model is the name it will be served under."""
    return DetectorResult(
        found=True,
        provider_type=recs.server.runtime,
        base_url=recs.base_url or f"http://127.0.0.1:{recs.server.port}/v1",
        models=[recs.served],
        suggested_model=recs.served,
        probe_duration_ms=0.0,
    )


def _write_records(config_path: Path, recs: InitRecords, *, force: bool = False,
                   adopt: bool = False) -> Path:
    """A fresh config.yaml from a record: the provider points at the address the record will serve
    on and names the served model; no capability probe and no context budget (a server that is not
    up cannot be asked — the first `start` probes it and fits the window). Memory and host tools
    are on. The record's deferred writes run only once the file is on disk."""
    result = _result_from_record(recs)
    # Dumped BEFORE the config is built: HarnessConfig lends the primary's binary to a peer that has
    # none, in place, at load. The file says what the record says (a peer carries no binary), and
    # that same dump is what an adoption records (D-37, D-59).
    server_dict, peer_dicts = _record_dicts_of(recs.server, recs.peers)
    harness = HarnessConfig(
        version="1",
        provider=ProviderConfig(
            provider_type=result.provider_type,
            base_url=result.base_url,
            api_key=recs.api_key.get_secret_value() if recs.api_key else "none",
            default_model=recs.served,
            available_models=[recs.served],
            supports_function_calling=None,
            timeout_seconds=600.0,
        ),
        org=OrgConfig(
            default_model=recs.served,
            permissions=PermissionConfig(defaults_revision=CURRENT_DEFAULTS_REVISION),
        ),
        server=recs.server.model_copy(deep=True),
        extra_endpoints=[peer.model_copy(deep=True) for peer in recs.peers],
    )
    written = _write_harness(config_path, harness, True, force=force,
                             records=(server_dict, peer_dicts))
    for thunk in recs.deferred:  # only now that the config is on disk
        thunk()
    if adopt:  # D-59: exactly the dicts that were written, not a fresh look at the disk
        _adopt_written(global_config_dir(config_path), "config.yaml", server_dict, peer_dicts)
    return written


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _existing_install_import(config_path: Path, harness: HarnessConfig) -> InitRecords | Attached | None:
    """D-36, D-54: an install whose config.yaml sets no `server:` imports what runs behind its provider
    address. An own-uid process on that port that answers detection (`found` required, so a process
    that answers no HTTP is never imported) goes through the live import; with nothing listening there,
    a stopped unit that names a model is the record. None keeps today's flow: a `server:` already set,
    a provider on another machine, a port another uid holds, a listener the import has no business
    with, no unit."""
    if harness.server is not None:
        return None
    try:
        parsed = urlparse(harness.provider.base_url)
        port, host = parsed.port, parsed.hostname
    except ValueError:
        return None
    if port is None or host not in _LOOPBACK_HOSTS:  # a provider elsewhere is not this machine's process
        return None
    found = procfs.listeners(port)
    if not found:
        if harness.provider.provider_type != "llamacpp":  # an Ollama / vLLM provider is not a llama-server unit
            return None
        return _unit_import(config_path, only_port=port)[0]
    if not any(lst.uid == os.getuid() for lst in found):
        return None
    try:
        result = asyncio.run(detect_provider(timeout_seconds=1.0, ports=[port]))
    except Exception:  # noqa: BLE001 — a probe that fails imports nothing
        return None
    if not result.found:
        return None
    return _import_listener(config_path, result)


def _write_overlay_records(config_path: Path, recs: InitRecords, *, adopt: bool) -> bool:
    """D-36, D-37, D-60: an existing install's import goes to overrides.yaml through the one validated
    overlay write: `server:` (exclude_none), discovery's peers only while the merged `extra_endpoints`
    is still empty when the write happens (checked inside `mutate`, T-5-69), and an inline key as
    `provider.api_key`. config.yaml is not touched. The audit triples name entries, never the key's
    value. True: written — the ledger writes run and, when `adopt`, exactly the dicts written are
    adopted (file `overrides.yaml`). False: nothing written."""
    global_dir = global_config_dir(config_path)
    server_dict, peer_dicts = _record_dicts_of(recs.server, recs.peers)
    wrote: dict[str, list[dict]] = {"peers": []}
    audit: list[tuple[str, object, object]] = []  # filled by mutate: what was written, not what was meant

    def entries(prefix: str, mapping: dict) -> list[tuple[str, object, object]]:
        found = [(e["name"], None, e["shown"]) for e in trust._launch_entries("overrides.yaml", prefix, mapping)]
        found += [(prefix + key, None, json.dumps(mapping[key])) for key in trust._POSTURE_KEYS
                  if key in mapping]
        return found

    def mutate(overlay: dict) -> None:
        audit.clear()
        wrote["peers"] = []
        overlay["server"] = copy.deepcopy(server_dict)
        audit.extend(entries("server.", server_dict or {}))
        merged = (overlay["extra_endpoints"] if "extra_endpoints" in overlay
                  else workspace.raw_settings(global_dir)[0].get("extra_endpoints"))
        if peer_dicts and not merged:
            overlay["extra_endpoints"] = copy.deepcopy(peer_dicts)
            wrote["peers"] = peer_dicts
            for peer in peer_dicts:
                audit.extend(entries(f"extra_endpoints.{peer['name']}.lifecycle.", peer.get("lifecycle")
                                     or {}))
        if recs.api_key:
            provider = overlay.get("provider")
            overlay["provider"] = {**(provider if isinstance(provider, dict) else {}),
                                   "api_key": recs.api_key.get_secret_value()}
            audit.append(("provider.api_key", None, "(set)"))

    ok = workspace.write_overlay_validated(
        global_dir, mutate, audit,
        actor_detail=f"localharness init: imported {recs.label or 'the model server'} into overrides.yaml")
    if not ok:
        return False
    for thunk in recs.deferred:  # only now that overrides.yaml is on disk
        thunk()
    if adopt:  # D-37: exactly the dicts that were written, not a fresh look at the disk
        _adopt_written(global_dir, "overrides.yaml", server_dict, wrote["peers"])
    return True


def _write_harness(config_path: Path, harness: HarnessConfig, memory_on: bool, *,
                   force: bool = False,
                   records: tuple[dict | None, list[dict]] | None = None) -> Path:
    """Write config.yaml (owner-only), the agents/ folder and plugins/README.md beside it. With
    `force` (R15) the saved model and server choices leave overrides.yaml in the same step, checked
    before either file is written; without it, say which saved choices still win."""
    import yaml

    reset = _saved_choices_reset(config_path, harness) if force else None
    config_file = config_path / "config.yaml"
    if force and config_file.exists():
        # --force starts over; the config it replaces stays beside it, owner-only, so it can be
        # undone. A distinct infix: never counted or pruned as one of migrate's backups.
        stamp = datetime.now().strftime(BACKUP_STAMP_FORMAT)
        backup = config_path / f"config.yaml.before-init-{stamp}"
        write_private_bytes(backup, config_file.read_bytes())
        console.print("[green]✓[/green] " + escape(f"Saved your previous config to {backup}"),
                      soft_wrap=True)
    # memory.enabled is the memory plugin's key (MEMP-06); the deprecated org.memory_enabled is
    # never written, so a fresh install has nothing to warn about. On is the default: no key.
    # Keys are written as typed (`reveal`): a JSON-mode dump would write the mask in their place.
    data = reveal(harness.model_dump(mode="python", exclude={"org": {"memory_enabled"}}))
    # D-12: no null preset field in a record (`records` are dumps the caller made first)
    server_dict, peer_dicts = records if records is not None else _record_dicts(harness)
    if server_dict is not None:
        data["server"] = server_dict
    data["extra_endpoints"] = peer_dicts
    text = yaml.safe_dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    # Owner-only from its first byte: this file carries `provider.api_key` and the deny/ask policy
    # every session on this machine is gated by (a re-init over an older, wider file tightens it).
    write_private_bytes(config_file,
                        (text + ("" if memory_on else "memory:\n  enabled: false\n")).encode("utf-8"))
    # #53: create the agents directory alongside the config. doctor names `init` as the remedy
    # for a missing agents dir, so init must actually create it (previously only `start` and
    # `doctor --fix` did, which left that remedy non-functional).
    (config_path / "agents").mkdir(parents=True, exist_ok=True)
    # ENAB-05. Written only when absent: a `--force` re-init replaces config.yaml, never notes the
    # user keeps in this README.
    plugins_readme = config_path / "plugins" / "README.md"
    plugins_readme.parent.mkdir(exist_ok=True)
    if not plugins_readme.exists():
        touch_private(plugins_readme)  # nothing in the config folder is left for others to read
        plugins_readme.write_text(GLOBAL_PLUGINS_README, encoding="utf-8")
    if reset:
        overlay_path, overlay, cleared = reset
        atomic_write_overlay(overlay_path, overlay)  # an emptied overrides.yaml stays, as {}
        console.print(
            "[green]✓[/green] "
            + escape(f"Cleared the saved model and server choices in {overlay_path} "
                     f"({', '.join(cleared)})."),
            soft_wrap=True,
        )
    elif not force:
        _overlay_shadows(config_path)
    return config_file


def _write_config(
    config_path: Path, result: DetectorResult, selected_model: str, cap: CapabilityResult,
    *, force: bool = False,
) -> Path:
    """A fresh config.yaml from the server found and the model chosen. Both defaults (the agent may
    edit files and run commands, memory is on) apply, with a terminal or without (D-38)."""
    # Stamp fresh configs at the current shipped-defaults revision so the first `start` never
    # spuriously migrates and any later deliberate removal of a default is respected (the
    # revision gate in config/migrate.py only protects configs stamped current at birth).
    org_kwargs: dict = {
        "default_model": selected_model,
        "permissions": PermissionConfig(defaults_revision=CURRENT_DEFAULTS_REVISION),
    }
    records: list[Callable[[], None]] = []
    window = _served_window(result, cfg_dir=config_path, model=selected_model, deferred=records)
    if window:
        org_kwargs["context"] = ContextConfig(max_context_tokens=window)
    harness = HarnessConfig(
        version="1",
        provider=ProviderConfig(
            provider_type=result.provider_type,
            base_url=result.base_url,
            api_key="none",
            default_model=selected_model,
            available_models=result.models,
            supports_function_calling=(cap.tool_call_mode == "native"),
            timeout_seconds=600.0,
        ),
        org=OrgConfig(**org_kwargs),
    )
    written = _write_harness(config_path, harness, True, force=force)
    for record in records:  # only now that the config is on disk (WR-05)
        record()
    return written


def _write_change(config_path: Path, result: DetectorResult, selected_model: str,
                  cap: CapabilityResult) -> Path:
    """A re-run's change: the new server and model go to the machine's overrides.yaml through the
    validated overlay write `/model` uses (persist_default_model's precedent) — validated as the
    merged config the next `start` reads, so config.yaml stays byte-identical and a change that
    does not validate writes nothing."""
    from localharness.cli.components_cmd import _core_secret_values, _validate_overlay
    from localharness.registry import set_value_in_dict

    loader = ConfigLoader(config_dir=config_path)
    overlay = load_overlay(loader.user_overlay_path)
    for path, value in (
        ("provider.provider_type", result.provider_type),
        ("provider.base_url", result.base_url),
        ("provider.default_model", selected_model),
        ("provider.available_models", list(result.models)),
        ("provider.supports_function_calling", cap.tool_call_mode == "native"),
        ("org.default_model", selected_model),
    ):
        set_value_in_dict(overlay, path, value)
    records: list[Callable[[], None]] = []
    window = _served_window(result, cfg_dir=config_path, model=selected_model, deferred=records)
    if window:
        set_value_in_dict(overlay, "org.context.max_context_tokens", window)
    try:
        _validate_overlay(loader, "provider.base_url", overlay)
    except ValueError as exc:  # a pydantic ValidationError is one
        # location and message only, scrubbed: never the merged config pydantic carries (R16)
        why = validation_text(exc, _core_secret_values(loader, overlay))
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"the new settings do not validate: {why}. Nothing was written."),
            soft_wrap=True,
        )
        raise typer.Exit(1)
    atomic_write_overlay(loader.user_overlay_path, overlay)
    for record in records:  # a change that did not validate wrote nothing, and records nothing (WR-05)
        record()
    console.print(
        "[green]✓[/green] "
        + escape(f"Updated {loader.user_overlay_path} — your config.yaml is unchanged."),
        soft_wrap=True,
    )
    return loader.user_overlay_path


RECEIPT_ROW = "The agent can edit files, run commands and remember across sessions — org.permissions.mode: read-only or memory.enabled: false turns either off."  # noqa: E501


def _receipt(config_file: Path, starting: bool) -> None:
    """The last lines of a fresh setup. A first `start` goes straight on into the session, so it
    is not told to run `start`."""
    # escape(): the receipt runs AFTER config.yaml is on disk, so an unescaped markup-named config
    # dir crashed here and exited 1 — init reporting failure for a setup that had just succeeded.
    console.print("\n[green]✓[/green] " + escape(f"LocalHarness configured at {config_file}."), soft_wrap=True)
    console.print(RECEIPT_ROW, markup=False, soft_wrap=True)  # D-38: the defaults, and the undo
    if not starting:
        console.print("  Run 'localharness start' to begin.")
        console.print(
            "\n[dim]★ If this saves you an API bill, a star helps others find it →[/dim] "
            "[cyan]https://github.com/ahwurm/localharness[/cyan]"
        )


PLUGINS_HEADER = "Plugins — the command beside each one turns it on or sets it up:"
PLUGINS_IN_SESSION = "In a session, /plugins enable <name> does this too and turns it on right away."


def _plugin_lines(config_path: Path) -> list[str]:
    """The end of init: every bundled plugin with the plan's own state text (PlanEntry.display) and,
    for one that is on, the command that runs its setup step (PlanEntry.step_command) — the words
    `plugins list` and doctor print, never restated here. Asks nothing."""
    from localharness.plugins.resolve import resolve  # lazy: only an init that wrote a config pays
    try:
        entries = [e for e in resolve(ConfigLoader(config_dir=config_path)).plan.entries if e.bundled]
    except Exception:  # noqa: BLE001 — a plugin problem never fails the core setup
        return ["Plugins: run `localharness plugins list` to see them."]
    width = max((len(e.name) for e in entries), default=0)
    return [PLUGINS_HEADER] + [
        f"  {e.name.ljust(width)}  {e.display}"
        + ("" if e.enable_command or not e.step_command else f" — set up: {e.step_command}")
        for e in entries] + [PLUGINS_IN_SESSION]


def _guided_setup(config_path: Path) -> InitRecords | None:
    """No server detected: install vLLM and fetch a reference model — and write the record. The
    first `start` launches it, after its own trust decision; init launches and waits for nothing.

    Returns the record (the shown preset and flags lines inside) or None if declined or
    non-interactive (caller keeps the manual-instructions exit)."""
    # The managed vLLM install/pidfile/log/venv are machine-wide (C2 single-pidfile invariant):
    # always the global layer, never a workspace one. `init` never discovers a workspace, so this
    # is the same value today — the point is that the site says which layer it means.
    server_config_path = global_config_dir(config_path)
    if not sys.stdin.isatty():
        return None
    if not Confirm.ask("\nSet up vLLM and a model now?", default=True):
        return None

    # --- Hardware ---------------------------------------------------------- #
    console.print("\nPick your hardware (reference architectures):")
    for i, ra in enumerate(REF_ARCHS, start=1):
        console.print(f"  {i}. {ra.name}  [{ra.status}] — {ra.default_model}")
    console.print(f"  {len(REF_ARCHS) + 1}. Other / set up manually")
    choice = IntPrompt.ask("Select", default=1)
    if not 1 <= choice <= len(REF_ARCHS):
        return None
    ra = REF_ARCHS[choice - 1]
    console.print(f"  Reference doc: [cyan]{ra.doc}[/cyan]")
    if not sys.platform.startswith(ra.platform):
        console.print(
            f"  [yellow]⚠[/yellow]  {ra.name} targets {ra.platform}; this machine is {sys.platform}. "
            "Continuing, but the reference numbers won't apply."
        )

    # --- Runtime: existing binary > profile's install route ----------------- #
    binary = managed_server.find_vllm(server_config_path)
    launch, image = "binary", None
    if binary:
        console.print("  [green]✓[/green] " + escape(f"vLLM found: {binary}"), soft_wrap=True)
    elif ra.launch == "docker":
        if which_outside_cwd("docker") is None:
            err_console.print(
                "[bold red]Error:[/bold red] No vllm binary and no docker. "
                f"This hardware's supported route is the NVIDIA container — see {ra.doc}."
            )
            raise typer.Exit(1)
        launch, image = "docker", ra.docker_image
        console.print(
            f"  vLLM will run via Docker image [bold]{image}[/bold] (pulled on first launch; needs the NVIDIA container toolkit)."
        )
    else:
        venv = managed_server.server_dir(server_config_path) / "venv"
        # Confirm.ask renders through the same markup parser: an unescaped venv path crashed the
        # PROMPT, so guided setup died before the user could answer it.
        if not Confirm.ask(
            f"  Install [bold]{ra.pip_package}[/bold] into " + escape(str(venv)) + "?", default=True
        ):
            console.print(f"  Install it yourself, then re-run init — see {ra.doc}.")
            raise typer.Exit(1)
        try:
            binary = managed_server.install_vllm_venv(server_config_path, str(ra.pip_package))
        except RuntimeError as exc:
            err_console.print(
                "[bold red]Error:[/bold red] " + escape(f"{exc}\nSee {ra.doc} for the manual route."),
                soft_wrap=True,
            )
            raise typer.Exit(1)
        console.print("  [green]✓[/green] " + escape(f"Installed: {binary}"), soft_wrap=True)

    # --- Model -------------------------------------------------------------- #
    console.print(f"\n  Reference model: [bold]{ra.default_model}[/bold]")
    console.print(f"  [dim]{ra.model_note}[/dim]")
    model = Prompt.ask("  Model (HF repo id, or local checkpoint path)", default=ra.default_model)
    if not Path(model).expanduser().exists():  # repo id → ensure it's in the HF cache
        if managed_server.is_model_cached(model):
            console.print("  [green]✓[/green] Already downloaded (Hugging Face cache).")
        else:
            if not Confirm.ask("  Download [bold]" + escape(model) + "[/bold] now?", default=True):
                raise typer.Exit(1)
            try:
                managed_server.download_model(model)
            except Exception as exc:
                err_console.print(
                    "[bold red]Error:[/bold red] " + escape(f"download failed: {exc}"), soft_wrap=True
                )
                raise typer.Exit(1)
            console.print("  [green]✓[/green] Download complete.")

    # --- The record ----------------------------------------------------------#
    srv = ManagedServerConfig(
        launch=launch,
        binary=binary,
        docker_image=image,
        model=model,
        port=8081,
        extra_args=list(ra.serve_extra_args),
        refarch=ra.key,
    )
    # The preset block (D-14), every value through visible_text: the binary was just found or
    # installed in front of the person, so its clause carries no provenance.
    how = f"binary: {shown_path(binary)}" if binary else f"docker_image: {visible_text(image)}"
    lines = [f"serving {visible_text(model)} ({RUNTIME_LABELS['vllm']}) "
             f"\u2190 {shown_path(model)} \u00b7 {how}"]
    if srv.extra_args:  # D-68: the flags the record launches with are shown before they are adopted
        lines.append("flags: " + visible_text(shlex.join(srv.extra_args)))
    return InitRecords(server=srv, served=model, lines=lines)


def _select_model(result: DetectorResult, model: str | None, interactive: bool) -> str:
    """Select one of several served models: the one --model names, Ollama's hot model when
    unambiguous, the user's pick on a terminal — else the first, saying how to pick another."""
    if model in result.models:
        return model
    if result.provider_type == "ollama":
        hot = _get_ollama_hot_model(result.base_url)
        if hot and hot in result.models:
            console.print(f"  Model: [bold]{hot}[/bold] (active — auto-selected)")
            return hot

    if interactive:
        console.print("\nAvailable models:")
        for i, m in enumerate(result.models, start=1):
            console.print(f"  {i}. {m}")
        choice = IntPrompt.ask("Select model", default=1)
        idx = max(1, min(choice, len(result.models))) - 1
        return result.models[idx]
    first = result.models[0]
    console.print(
        "  Model: [bold]" + escape(first) + "[/bold] "
        + escape(f"(the first of {len(result.models)} served — `localharness init --endpoint "
                 f"{result.base_url} --model <name>` picks another)"),
        soft_wrap=True,
    )
    return first


def _get_ollama_hot_model(base_url: str) -> str | None:
    """Query Ollama /api/ps to get currently loaded model. Returns None on any error."""
    try:
        import httpx
        # base_url is OpenAI-compat (e.g. http://localhost:11434/v1); strip /v1 for native API
        native_url = base_url.removesuffix("/v1")
        response = httpx.get(f"{native_url}/api/ps", timeout=1.0)
        data = response.json()
        models = data.get("models", [])
        if len(models) == 1:
            return models[0].get("name")
    except Exception:
        pass
    return None


# #52: derive the documented probe order from the detector's port list so `init --help` can
# never drift from what detect_provider() actually probes (it silently dropped :8000, vLLM's
# stock port). Typer resolves a command's help from its callback __doc__ at CLI-build time, so
# assigning it here (module load) is picked up.
init_app.__doc__ = (
    "Auto-detect local LLM and write initial configuration.\n\n"
    f"Probes known ports in order: {_PROBE_ORDER}. "
    "Writes config to <config-dir>/config.yaml on success."
    " Ends with every bundled plugin and the command that turns it on or sets it up."
)
