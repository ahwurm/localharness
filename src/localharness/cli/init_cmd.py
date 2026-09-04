"""localharness init command — auto-detect LLM (or guided setup) and write config."""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt, Prompt

from localharness.agent.context import response_reserve
from localharness.cli.errors import _HANDLED, report_filesystem_error
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.loader import ConfigLoader
from localharness.config.paths import WORKSPACE_DIR_NAME, global_config_dir, resolve_config_dir
from localharness.config.models import (
    ContextConfig,
    HarnessConfig,
    ManagedServerConfig,
    OrgConfig,
    PermissionConfig,
    ProviderConfig,
)
from localharness.provider import server as managed_server
from localharness.provider.client import LLMClient, LLMConfig
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


def _detect_llamacpp_nctx(base_url: str) -> int | None:
    """llama.cpp's /props exposes the served context window (n_ctx).

    base_url is OpenAI-compat (…/v1); /props lives at the server root.
    Returns None on any error. (Ollama is handled by the safe default — its
    /v1/models reports no window and /api/show gives the model's trained max,
    not the served num_ctx, so clamping to it would overshoot.)"""
    try:
        import httpx
        native = base_url.removesuffix("/v1")
        props = httpx.get(f"{native}/props", timeout=2.0).json()
        val = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
        return int(val) if val else None
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
#   context:
#     compaction_threshold_pct: 85.0
#
# Per-agent settings live in .localharness/agents/<name>.yaml, not here. The one people
# look for first:
#   memory:
#     recall_scope: workspace   # workspace (default) | global | both
#   ...controls which memory store a session in this project RECALLS from. It moves reads
#   only — a session always writes to this project's own store whatever it says.
"""


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
    *, endpoint: str | None, model: str | None, config_dir: str | None
) -> None:
    """`localharness init --workspace`: create ./.localharness/ for the project you are in.

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

    # `lexists`, not `exists` (H2): `exists()` FOLLOWS symlinks, so a `.localharness` pointing at
    # a deleted tree answered False, sailed past this refusal, and died in mkdir with
    # FileExistsError. A link that is there is there, whatever it points at.
    if target.is_symlink() or target.exists():
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(f"A workspace already exists at {target}. "
                     f"Edit {target / 'config.yaml'} directly")
            + " — this command never overwrites one.",
            soft_wrap=True,
        )
        raise typer.Exit(1)

    # Validated before anything is written, and every remaining failure caught: a symlink loop
    # (ELOOP), an unwritable project directory, a full disk. A half-made workspace left behind an
    # error message is the worst outcome here — the next run would refuse to touch it (E cluster).
    try:
        (target / "agents").mkdir(parents=True)
        (target / "config.yaml").write_text(_WORKSPACE_CONFIG_TEMPLATE, encoding="utf-8")
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
    console.print(
        "  Next:    run `localharness start` from anywhere in this project — its memory, "
        "sessions and logs now stay here."
    )


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
            help="Overwrite existing config without prompting.",
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

    config_path = resolve_config_dir(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    config_file = config_path / "config.yaml"

    # Prompt before overwrite
    if config_file.exists() and not force:
        overwrite = Confirm.ask(
            f"Config exists at {config_file}. Overwrite?", default=False
        )
        if not overwrite:
            raise typer.Exit(0)

    # ------------------------------------------------------------------ #
    # Provider detection / endpoint override
    # ------------------------------------------------------------------ #
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
        selected_model = model
        server_config = None
    else:
        console.print("Probing for local LLM...")
        result = asyncio.run(detect_provider(timeout_seconds=1.0))
        server_config = None

        if not result.found:
            console.print("\n[bold red]✗ No local LLM detected.[/bold red]\n")
            console.print("Checked:")
            for port in DEFAULT_PORTS:
                name = _PORT_LABELS.get(port, "unknown")
                console.print(f"  http://localhost:{port}  ({name})  — connection refused")
            guided = _guided_setup(config_path)
            if guided is None:
                console.print(
                    "\nStart your LLM server and run 'localharness init' again, or use:"
                )
                console.print(
                    "  localharness init --endpoint http://your-host:port/v1 --model your-model-name"
                )
                raise typer.Exit(1)
            result, selected_model, server_config = guided
        else:
            console.print(f"  [green]✓[/green] {result.provider_type} found at {result.base_url}")

            if len(result.models) == 0:
                err_console.print("[bold red]Error:[/bold red] No models available at detected endpoint.")
                raise typer.Exit(1)
            elif len(result.models) == 1:
                selected_model = result.models[0]
                console.print(f"  Model: [bold]{selected_model}[/bold] (auto-selected)")
            else:
                # Multiple models — check for hot model on Ollama, otherwise prompt
                selected_model = _select_model(result)

    # ------------------------------------------------------------------ #
    # Capability probe
    # ------------------------------------------------------------------ #
    llm_cfg = LLMConfig(
        base_url=result.base_url,
        model=selected_model,
        timeout_seconds=300.0,
    )
    client = LLMClient(llm_cfg)
    cap = asyncio.run(client.detect_capabilities())

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

    # ------------------------------------------------------------------ #
    # Write config
    # ------------------------------------------------------------------ #
    from pydantic_yaml import to_yaml_str

    # Fit the context budget to the served window when the provider reports it —
    # a budget above the real window disables compaction and kills long turns.
    # Stamp fresh configs at the current shipped-defaults revision so the first `start` never
    # spuriously migrates and any later deliberate removal of a default is respected (the
    # revision gate in config/migrate.py only protects configs stamped current at birth).
    org_kwargs: dict = {
        "default_model": selected_model,
        "permissions": PermissionConfig(defaults_revision=CURRENT_DEFAULTS_REVISION),
    }
    if result.provider_type == "llamacpp":
        max_len = _detect_llamacpp_nctx(result.base_url)
    elif result.provider_type == "lmstudio":
        max_len = _detect_lmstudio_ctx(result.base_url)
    else:
        max_len = _detect_max_model_len(result.base_url)
    # max_context_tokens IS the served window: the harness subtracts the reply reserve at runtime
    # (agent.context.response_reserve), so subtracting it here too would reserve it twice.
    # Two independent floors: ContextConfig rejects anything under 1_000, and a window that
    # reserves nothing (<= 1_024) cannot run at all — `start` refuses it, so writing it would only
    # persist a budget the next command rejects.
    if max_len and max_len >= 1_000 and response_reserve(max_len) > 0:
        org_kwargs["context"] = ContextConfig(max_context_tokens=max_len)
        console.print(
            f"  [green]✓[/green] Context budget: {max_len:,} tokens "
            f"(the full served window — the harness reserves output room inside it)"
        )
    elif max_len:
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
        server=server_config,
    )
    config_file.write_text(to_yaml_str(harness), encoding="utf-8")
    # #53: create the agents directory alongside the config. doctor names `init` as the remedy
    # for a missing agents dir, so init must actually create it (previously only `start` and
    # `doctor --fix` did, which left that remedy non-functional).
    (config_path / "agents").mkdir(parents=True, exist_ok=True)
    console.print(f"\n[green]✓[/green] LocalHarness configured at {config_file}.")
    console.print("  Run 'localharness start' to begin.")
    console.print(
        "\n[dim]★ If this saves you an API bill, a star helps others find it →[/dim] "
        "[cyan]https://github.com/ahwurm/localharness[/cyan]"
    )


def _guided_setup(
    config_path: Path,
) -> tuple[DetectorResult, str, ManagedServerConfig] | None:
    """No server detected: install vLLM, download a reference model, launch, wait ready.

    Returns (detection-equivalent result, served model, server config) so the caller
    falls into the normal capability-probe/config-write path, or None if declined or
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
        console.print(f"  [green]✓[/green] vLLM found: {binary}")
    elif ra.launch == "docker":
        if shutil.which("docker") is None:
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
        if not Confirm.ask(f"  Install [bold]{ra.pip_package}[/bold] into {venv}?", default=True):
            console.print(f"  Install it yourself, then re-run init — see {ra.doc}.")
            raise typer.Exit(1)
        try:
            binary = managed_server.install_vllm_venv(server_config_path, str(ra.pip_package))
        except RuntimeError as exc:
            err_console.print(f"[bold red]Error:[/bold red] {exc}\nSee {ra.doc} for the manual route.")
            raise typer.Exit(1)
        console.print(f"  [green]✓[/green] Installed: {binary}")

    # --- Model -------------------------------------------------------------- #
    console.print(f"\n  Reference model: [bold]{ra.default_model}[/bold]")
    console.print(f"  [dim]{ra.model_note}[/dim]")
    model = Prompt.ask("  Model (HF repo id, or local checkpoint path)", default=ra.default_model)
    if not Path(model).expanduser().exists():  # repo id → ensure it's in the HF cache
        if managed_server.is_model_cached(model):
            console.print("  [green]✓[/green] Already downloaded (Hugging Face cache).")
        else:
            if not Confirm.ask(f"  Download [bold]{model}[/bold] now?", default=True):
                raise typer.Exit(1)
            try:
                managed_server.download_model(model)
            except Exception as exc:
                err_console.print(f"[bold red]Error:[/bold red] download failed: {exc}")
                raise typer.Exit(1)
            console.print("  [green]✓[/green] Download complete.")

    # --- Launch + readiness --------------------------------------------------#
    srv = ManagedServerConfig(
        launch=launch,
        binary=binary,
        docker_image=image,
        model=model,
        port=8081,
        extra_args=list(ra.serve_extra_args),
        refarch=ra.key,
    )
    cmd = managed_server.serve_command(srv)
    base_url = f"http://localhost:{srv.port}/v1"
    console.print(f"\n  Launching: [dim]{' '.join(cmd)}[/dim]")
    console.print(f"  Log: {managed_server.log_path(server_config_path)}")
    managed_server.start_server(server_config_path, cmd)
    console.print("  Waiting for the server — model load can take several minutes...")
    try:
        models = asyncio.run(managed_server.wait_ready(base_url, config_dir=server_config_path))
    except (RuntimeError, TimeoutError) as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1)
    served = models[0] if models else model
    console.print(f"  [green]✓[/green] vLLM serving [bold]{served}[/bold] on :{srv.port} (managed — `localharness start` restarts it after reboots)")
    result = DetectorResult(
        found=True,
        provider_type="vllm",
        base_url=base_url,
        models=models or [model],
        suggested_model=served,
        probe_duration_ms=0.0,
    )
    return result, served, srv


def _select_model(result: DetectorResult) -> str:
    """Select model from multiple available options. Auto-selects hot Ollama model if unambiguous."""
    if result.provider_type == "ollama":
        hot = _get_ollama_hot_model(result.base_url)
        if hot and hot in result.models:
            console.print(f"  Model: [bold]{hot}[/bold] (active — auto-selected)")
            return hot

    console.print("\nAvailable models:")
    for i, m in enumerate(result.models, start=1):
        console.print(f"  {i}. {m}")
    choice = IntPrompt.ask("Select model", default=1)
    idx = max(1, min(choice, len(result.models))) - 1
    return result.models[idx]


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
)
