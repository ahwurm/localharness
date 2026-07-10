"""localharness init command — auto-detect LLM (or guided setup) and write config."""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from localharness.config.loader import ConfigLoader
from localharness.config.models import (
    ContextConfig,
    HarnessConfig,
    ManagedServerConfig,
    OrgConfig,
    ProviderConfig,
)
from localharness.provider import server as managed_server
from localharness.provider.client import LLMClient, LLMConfig
from localharness.provider.detector import DEFAULT_PORTS, DetectorResult, detect_provider
from localharness.provider.refarch import REF_ARCHS

console = Console()
err_console = Console(stderr=True)


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


def _fit_context_tokens(max_model_len: int, output_reserve: int = 4_096) -> int:
    """Context budget that compacts BEFORE the served window's input cap."""
    return max(8_192, max_model_len - output_reserve)


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
        str,
        typer.Option(
            "--config-dir",
            help="Directory for LocalHarness config and agent data.",
            envvar="LOCALHARNESS_DIR",
        ),
    ] = "~/.localharness",
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Overwrite existing config without prompting.",
        ),
    ] = False,
) -> None:
    """Auto-detect local LLM and write initial configuration.

    Probes known ports in order: vLLM (:8081), Ollama (:11434),
    LM Studio (:1234), llama.cpp (:8080). Writes config to
    <config-dir>/config.yaml on success.
    """
    config_path = Path(config_dir).expanduser()
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
            err_console.print("[bold red]Error:[/bold red] --model is required when using --endpoint")
            raise typer.Exit(1)
        result = DetectorResult(
            found=True,
            provider_type="unknown",
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
            port_names = {8081: "vLLM", 8000: "vLLM", 11434: "Ollama", 1234: "LM Studio", 8080: "llama.cpp"}
            console.print("\n[bold red]✗ No local LLM detected.[/bold red]\n")
            console.print("Checked:")
            for port in DEFAULT_PORTS:
                name = port_names.get(port, "unknown")
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
    org_kwargs: dict = {"default_model": selected_model}
    if result.provider_type == "llamacpp":
        max_len = _detect_llamacpp_nctx(result.base_url)
    else:
        max_len = _detect_max_model_len(result.base_url)
    if max_len:
        fitted = _fit_context_tokens(max_len)
        org_kwargs["context"] = ContextConfig(max_context_tokens=fitted)
        console.print(
            f"  [green]✓[/green] Context budget: {fitted:,} tokens "
            f"(served window {max_len:,} − 4,096 output reservation)"
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
    binary = managed_server.find_vllm(config_path)
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
        venv = managed_server.server_dir(config_path) / "venv"
        if not Confirm.ask(f"  Install [bold]{ra.pip_package}[/bold] into {venv}?", default=True):
            console.print(f"  Install it yourself, then re-run init — see {ra.doc}.")
            raise typer.Exit(1)
        try:
            binary = managed_server.install_vllm_venv(config_path, str(ra.pip_package))
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
    console.print(f"  Log: {managed_server.log_path(config_path)}")
    managed_server.start_server(config_path, cmd)
    console.print("  Waiting for the server — model load can take several minutes...")
    try:
        models = asyncio.run(managed_server.wait_ready(base_url, config_dir=config_path))
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
