"""`localharness model` — list served/downloaded models, or switch the persisted default.

CLI parity for the REPL /model, for scripts, pre-`start` config, and CI. Reuses cli/model_ops
for the EXACT same atomic, audited overlay persistence and pin-trap warning. The live-session
bits — hot-swap, TokenCounter rebind, managed-server restart — are REPL-only and deliberately
absent: with no running session there is nothing to hot-swap, and the next `localharness start`
launches (or relaunches the managed server on) the persisted model.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from localharness.cli import model_ops
from localharness.config.loader import ConfigLoader

console = Console()
err_console = Console(stderr=True)


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
) -> None:
    """List available models, or switch the persisted default with `localharness model <name>`."""
    # config_dir=None routes through the resolver's env/default chain (#35); an explicit flag or
    # LOCALHARNESS_DIR isolates the overlay to that dir.
    loader = ConfigLoader(config_dir=config_dir)
    try:
        harness = loader.load_harness()
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] failed to load config: {exc}")
        raise typer.Exit(2)

    provider = harness.provider
    try:
        live, reachable = model_ops.list_live_models(provider.base_url)
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
        return

    # --- Switch: resolve the target --- #
    # #39: reject an empty/whitespace name FIRST — before any resolution. Otherwise "" falls
    # through isdigit/exact/checkpoint (note Path("").expanduser().exists() == cwd) into the
    # unreachable-degrade branch and persists "" as the default.
    if not name.strip():
        err_console.print("[bold red]Error:[/bold red] model name cannot be empty.")
        raise typer.Exit(2)

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
            err_console.print(
                f"[bold red]Error:[/bold red] unknown model {name!r}. Available: {avail}."
            )
            raise typer.Exit(2)
        # Runtime unreachable → can't verify. Degrade with an explicit disclosure (mirrors the
        # TokenCounter `.approximate` convention: proceed, but label it clearly).
        target = name
        console.print(
            f"[yellow]⚠[/yellow]  Could not reach {provider.base_url} to verify {name!r} is "
            f"served — persisting it as the default UNVERIFIED. Run `localharness doctor` once the "
            f"server is up."
        )

    if target == current:
        console.print(f"{target} is already the default.")
        return

    try:
        audit_warning = asyncio.run(
            model_ops.persist_default_model(harness, target, config_dir=loader._config_dir)
        )
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] failed to persist {target!r}: {exc}")
        raise typer.Exit(2)

    console.print(f"[green]Default model set to[/green] {target}. `localharness start` will use it.")
    # #37: the switch is durably persisted; a post-write audit-emit failure is a secondary note.
    if audit_warning:
        console.print(f"[yellow]Note:[/yellow] {audit_warning}")

    # Pin trap: a persisted default won't reach an agent whose yaml pins a concrete model.
    for aname, pin in model_ops.pinned_agents(loader._config_dir):
        console.print(
            f"[yellow]Note:[/yellow] agent {aname!r} pins model={pin!r} in its yaml — "
            f"this won't reach it on next start until that pin changes."
        )
