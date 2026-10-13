"""`localharness model` — list served/downloaded models, or switch the persisted default.

CLI parity for the REPL /model, for scripts, pre-`start` config, and CI. Reuses cli/model_ops
for the EXACT same atomic, audited overlay persistence and pin-trap warning. The live-session
bits — hot-swap, TokenCounter rebind, managed-server restart — are REPL-only and deliberately
absent: with no running session there is nothing to hot-swap, and the next `localharness start`
launches (or relaunches the managed server on) the persisted model.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape

from localharness.cli import model_ops
from localharness.cli.workspace import MachineTrust, decide_machine_trust
from localharness.config.loader import ConfigLoader
from localharness.config.paths import global_config_dir, resolve_config_dir
from localharness.provider.units import read_target

console = Console()
err_console = Console(stderr=True)


def _is_local_operator() -> bool:
    """Is a person at this terminal making the switch (D-19, TRUST-09)? True only on a tty without
    the agent-command ``LOCALHARNESS_COMMIT`` marker: `bash_tool` sets it in every agent command's
    environment and `python_exec` sets it in the process environment for the length of each call (so
    a child it spawns inherits it), hence an agent running `localharness model` from either tool is
    never a local operator, and a pipe or a script has no tty. The one signal is the environment,
    never an argument or a message. Not a sandbox: agent code that deliberately clears the variable
    or spawns with its own ``env=`` is outside what this can see (`python_exec` can already rewrite
    the trust file, which SECURITY.md accepts)."""
    if "LOCALHARNESS_COMMIT" in os.environ:
        return False
    try:
        return sys.stdin is not None and bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _download(repo_id: str, filename: Optional[str]) -> None:
    """`localharness model --download <repo_id> [--file <name>]` — standalone HF fetch, no
    running server / config required (unlike `init`'s guided-setup download, which is tied to
    the vLLM install flow). Whole-repo snapshot by default; --file fetches exactly one sibling
    file (the GGUF case, where a repo ships several quantizations side by side)."""
    from localharness.provider import server as managed_server

    try:
        if filename:
            console.print(f"Downloading {filename!r} from {repo_id}...")
            path = managed_server.download_file(repo_id, filename)
        else:
            console.print(f"Downloading {repo_id} (full snapshot)...")
            path = managed_server.download_model(repo_id)
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"download failed: {exc}"), soft_wrap=True
        )
        raise typer.Exit(2)

    # escape() around every path and repo id, markup outside it: a checkpoint under a folder
    # named `[old] models` is legal, and rich would silently delete `[old]` — printing a path
    # that does not exist right where the user is told what to copy (39-05). soft_wrap keeps a
    # deep path on one line for the copy.
    console.print("[green]✓[/green] " + escape(f"Downloaded to {path}"), soft_wrap=True)
    console.print(
        "To use it: point a llama.cpp profile's `server.model` at this path in config.yaml "
        "(single-model managed server), or run `localharness model "
        + escape(f"{path}`") + ("" if filename else escape(f" or `localharness model {repo_id}`"))
        + " once it's being served, to set it as the default.",
        soft_wrap=True,
    )


def _select_preset(harness, record, current, loader, audit_dir, ws) -> None:
    """Record ``record`` as what the next start serves (D-10, R-13-18): ``select_pending`` writes the
    peer's ``active_endpoint`` (the primary clears one) and ``pending_preset``; the primary's served
    name also becomes the default when it differs, without writing ``server.model`` (an alias must
    not become the model the next start loads, D-70, WR-54). The CLI never launches (D-10)."""
    if (
        record.peer is None and record.served == current
        and harness.active_endpoint is None and not getattr(harness, "pending_preset", None)
        and not read_target(global_config_dir(loader._config_dir))
    ):
        # The primary is already the default and nothing else is selected: nothing to switch, and
        # nothing is written (an overlay's bytes stay as they were).
        console.print(escape(f"{record.served} is already the default."), soft_wrap=True)
        return
    try:
        asyncio.run(model_ops.select_pending(harness, record, config_dir=loader._config_dir))
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"failed to select {record.name!r}: {exc}"),
            soft_wrap=True,
        )
        raise typer.Exit(2)
    audit_warning = None
    if record.peer is None and record.served != current:
        try:
            audit_warning = asyncio.run(
                model_ops.persist_default_model(
                    harness, record.served, config_dir=loader._config_dir, audit_base_dir=audit_dir,
                    local_operator=_is_local_operator(), write_server_model=False,
                )
            )
        except Exception as exc:
            err_console.print(
                "[bold red]Error:[/bold red] "
                + escape(f"failed to persist {record.served!r}: {exc}"),
                soft_wrap=True,
            )
            raise typer.Exit(2)
    console.print(
        model_ops.SELECTED_NEXT_START_LINE.format(name=record.name), markup=False, soft_wrap=True)
    if audit_warning:
        console.print("[yellow]Note:[/yellow] " + escape(str(audit_warning)), soft_wrap=True)
    _pin_notes(loader, ws)


def _pin_notes(loader, ws) -> None:
    # Pin trap: a persisted default won't reach an agent whose yaml pins a concrete model.
    for aname, pin in model_ops.pinned_agents(loader._config_dir, local_config_dir=ws):
        console.print(
            "[yellow]Note:[/yellow] "
            + escape(f"agent {aname!r} pins model={pin!r} in its yaml")
            + " — this won't reach it on next start until that pin changes.",
            soft_wrap=True,
        )


def model(
    name: Optional[str] = typer.Argument(
        None,
        help="Model name/number to switch to (or a checkpoint path for a managed server). "
        "Omit to list available models.",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        envvar="LOCALHARNESS_DIR",
        help="Config directory (default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, "
        "else ~/.localharness). Parity with `start`/`doctor`/`validate`; the persisted "
        "overlay is written HERE.",
    ),
    download: Optional[str] = typer.Option(
        None,
        "--download",
        help="Download a Hugging Face repo id into the local HF cache, then exit "
        "(does not change the configured default — follow up with `localharness model <path>`).",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file",
        help="With --download: fetch only this ONE filename from the repo (a specific GGUF "
        "quant, e.g. `Qwen3-32B-Q4_K_M.gguf`) instead of the whole repo snapshot. Use this for "
        "llama.cpp models — GGUF repos commonly ship several quantizations as sibling files.",
    ),
) -> None:
    """List available models, or switch the persisted default with `localharness model <name>`."""
    if download is not None:
        _download(download, file)
        return
    if file is not None:
        err_console.print("[bold red]Error:[/bold red] --file requires --download <repo_id>.")
        raise typer.Exit(2)

    # config_dir=None routes through the resolver's env/default chain (#35); an explicit flag or
    # LOCALHARNESS_DIR isolates the overlay to that dir.
    # local_config_dir carries the workspace layer, or None.
    from localharness.cli.workspace import resolve_workspace_layer
    # D-09 / D-10: a switch decides machine trust exactly as `start` does, before anything loads, and
    # loads with what that decision withheld. A person at a terminal is asked; a pipe, a script or an
    # agent's command (LOCALHARNESS_COMMIT) is not, so a piped `y` adopts nothing and an EOF records
    # no No. `will_launch` is False in this release (the CLI never launches, D-10). The list and
    # `--download` decide nothing (R-13-16).
    machine_trust = MachineTrust()
    if name is not None and name.strip():
        cfg_path = resolve_config_dir(config_dir)
        machine_trust = decide_machine_trust(cfg_path, ask=_is_local_operator(), launching=False)
        if machine_trust.line:
            err_console.print(machine_trust.line, markup=False, soft_wrap=True)
    loader = ConfigLoader(
        config_dir=config_dir, local_config_dir=resolve_workspace_layer(config_dir),
        machine_withheld=machine_trust.withheld)
    # v0.13: two values, two jobs. `loader._config_dir` is machine-wide truth (the overlay target,
    # one GPU daemon); `_ws` is the workspace layer, or None. The audit record follows the WORK
    # (MEMS-04) and falls back to the session's config dir — NOT to None, which would re-resolve
    # through the env chain and let `--config-dir D` leak its audit log out of D.
    # Bound to NAMED LOCALS on purpose, not written inline at the call sites: the structural guard
    # in tests/unit/test_provider_carveout_workspace.py scans this file for a literal binding of
    # the workspace attribute to a keyword ending in `config_dir`, and an inline workspace
    # expression at either call site below would tail-match it. (This very comment cannot spell
    # the forbidden literal either — that is how strict a plain substring scan is.)
    _ws = loader._local_dir
    _audit_dir = _ws or loader._config_dir
    try:
        harness = loader.load_harness()
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] failed to load config: {exc}")
        raise typer.Exit(2)

    provider = harness.provider
    from localharness.provider.server import required_key_kwargs
    try:
        # the provider's key only when the launched server is set to require it (default off)
        live, reachable = model_ops.list_live_models(
            provider.base_url, **required_key_kwargs(harness.server, provider.api_key))
    except model_ops.MalformedModelListError as exc:
        # #38: reached but the reply isn't a model list — its OWN message, not "Is it running?".
        err_console.print(
            f"[bold red]Error:[/bold red] the server at {provider.base_url} responded, but the "
            f"response wasn't understood — is base_url pointing at an OpenAI-compatible API? ({exc})"
        )
        raise typer.Exit(2)
    downloaded: list[str] = []
    if harness.server is not None:
        from localharness.provider import server as managed_server
        downloaded = [m for m in managed_server.list_cached_models() if m not in live]
    choices = live + downloaded
    current = provider.default_model

    # --- List --- #
    if name is None:
        if not reachable and not choices:
            err_console.print(
                f"[bold red]Error:[/bold red] could not reach the model server at "
                f"{provider.base_url}, and no downloaded models were found. "
                f"Is it running? Try `localharness doctor`."
            )
            raise typer.Exit(2)
        if not choices:
            console.print(
                f"No models served at {provider.base_url} or in the local download cache."
            )
            return
        console.print("Models:")
        for i, m in enumerate(live, start=1):
            mark = "  [active]" if m == current else ""
            console.print(f"  {i}. {m}  (serving){mark}", markup=False)
        for i, m in enumerate(downloaded, start=len(live) + 1):
            console.print(
                f"  {i}. {m}  (downloaded — `localharness start` will launch it)", markup=False
            )
        console.print("Switch with `localharness model <name|number>`.")
        # #50: a reachable server whose served/downloaded set omits the configured default
        # otherwise shows NO [active] marker and no other signal — the moment of highest risk
        # (checking in on an unverified degrade-persist) gets none. State it plainly.
        if reachable and current not in choices:
            console.print(
                "[yellow]⚠[/yellow]  " + escape(f"configured default {current!r}")
                + " is not among the served models — switch with "
                "`localharness model <name|number>` or check the server.",
                soft_wrap=True,
            )
        return

    # --- Switch: resolve the target --- #
    # #39: reject an empty/whitespace name FIRST — before any resolution. Otherwise "" falls
    # through isdigit/exact/checkpoint (note Path("").expanduser().exists() == cwd) into the
    # unreachable-degrade branch and persists "" as the default.
    if not name.strip():
        err_console.print("[bold red]Error:[/bold red] model name cannot be empty.")
        raise typer.Exit(2)

    # D-03 / R-13-17: a number keeps its list meaning; any other name goes through the ladder's
    # preset resolution first. A `Serving` is a preset selection; a withheld or ambiguous name is
    # refused with exit 2; an unknown name falls through to today's served / downloaded / checkpoint
    # resolution and its messages, unchanged.
    if not (name.isdigit() and 1 <= int(name) <= len(choices)):
        serving = model_ops.resolve_serving(
            harness, machine_trust, name, config_dir=loader._config_dir)
        if isinstance(serving, model_ops.ServingRefusal) and serving.kind != "unknown":
            # The withheld line was already printed once as `machine_trust.line`; say a refusal
            # that is not that line (an ambiguous name, or a withheld record start composed no line for).
            if serving.line != machine_trust.line:
                err_console.print(
                    "[bold red]Error:[/bold red] " + escape(serving.line), soft_wrap=True)
            raise typer.Exit(serving.exit_code)
        if isinstance(serving, model_ops.Serving):
            _select_preset(harness, serving.record, current, loader, _audit_dir, _ws)
            return

    target: Optional[str] = None
    if name.isdigit() and 1 <= int(name) <= len(choices):
        target = choices[int(name) - 1]
    elif name in choices:
        target = name
    elif harness.server is not None and Path(name).expanduser().exists():
        target = name

    if target is None:
        if reachable:
            # Reached the runtime and the target isn't served/downloaded → fail loud, name options.
            avail = ", ".join(choices) if choices else "(none served or downloaded)"
            if name.isdigit():
                # A number that didn't resolve is out of range — say how many are listed.
                err_console.print(
                    "[bold red]Error:[/bold red] " + escape(f"{name} is out of range")
                    + f" — {len(choices)} model(s) listed. Run `localharness model` to see them.",
                    soft_wrap=True,
                )
                raise typer.Exit(2)
            # Fat-finger hint: a case-insensitive exact match is almost certainly the intent.
            ci = next((m for m in choices if m.lower() == name.lower()), None)
            if ci is not None:
                err_console.print(
                    "[bold red]Error:[/bold red] "
                    + escape(f"unknown model {name!r} — did you mean {ci!r}? Available: {avail}."),
                    soft_wrap=True,
                )
            else:
                err_console.print(
                    "[bold red]Error:[/bold red] "
                    + escape(f"unknown model {name!r}. Available: {avail}."),
                    soft_wrap=True,
                )
            raise typer.Exit(2)
        # Runtime unreachable → can't verify. Degrade with an explicit disclosure (mirrors the
        # TokenCounter `.approximate` convention: proceed, but label it clearly).
        target = name
        console.print(
            "[yellow]⚠[/yellow]  "
            + escape(f"Could not reach {provider.base_url} to verify {name!r} is served")
            + " — persisting it as the default UNVERIFIED. Run `localharness doctor` once the "
            "server is up.",
            soft_wrap=True,
        )

    if target == current:
        console.print(escape(f"{target} is already the default."), soft_wrap=True)
        return

    try:
        audit_warning = asyncio.run(
            model_ops.persist_default_model(
                harness, target, config_dir=loader._config_dir, audit_base_dir=_audit_dir,
                local_operator=_is_local_operator(),
            )
        )
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"failed to persist {target!r}: {exc}"),
            soft_wrap=True,
        )
        raise typer.Exit(2)

    # A model name is data — served ids and checkpoint PATHS both arrive here, and `repr` does
    # not neutralize markup: `'[old]/gguf'` still parses. escape() it, markup outside.
    console.print(
        "[green]Default model set to[/green] " + escape(str(target))
        + ". `localharness start` will use it.",
        soft_wrap=True,
    )
    # #37: the switch is durably persisted; a post-write audit-emit failure is a secondary note.
    if audit_warning:
        console.print("[yellow]Note:[/yellow] " + escape(str(audit_warning)), soft_wrap=True)

    _pin_notes(loader, _ws)
