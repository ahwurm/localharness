"""localharness init command — auto-detect LLM and write config."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt

from localharness.config.loader import ConfigLoader
from localharness.config.models import HarnessConfig, OrgConfig, ProviderConfig
from localharness.provider.client import LLMClient, LLMConfig
from localharness.provider.detector import DEFAULT_PORTS, DetectorResult, detect_provider

console = Console()
err_console = Console(stderr=True)


def _build_base_url_for_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def init_app(
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint", "-e",
            help="Override auto-detection. Full base URL: http://localhost:8000/v1",
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

    Probes known ports in order: vLLM (:8000), Ollama (:11434),
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
    else:
        console.print("Probing for local LLM...")
        result = asyncio.run(detect_provider(timeout_seconds=1.0))

        if not result.found:
            port_names = {8000: "vLLM", 11434: "Ollama", 1234: "LM Studio", 8080: "llama.cpp"}
            console.print("\n[bold red]✗ No local LLM detected.[/bold red]\n")
            console.print("Checked:")
            for port in DEFAULT_PORTS:
                name = port_names.get(port, "unknown")
                console.print(f"  http://localhost:{port}  ({name})  — connection refused")
            console.print(
                "\nStart your LLM server and run 'localharness init' again, or use:"
            )
            console.print(
                "  localharness init --endpoint http://your-host:port/v1 --model your-model-name"
            )
            raise typer.Exit(1)

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

    harness = HarnessConfig(
        version="1",
        provider=ProviderConfig(
            provider_type=result.provider_type,
            base_url=result.base_url,
            api_key="none",
            default_model=selected_model,
            available_models=result.models,
            supports_function_calling=(cap.tool_call_mode == "native"),
            timeout_seconds=300.0,
        ),
        org=OrgConfig(default_model=selected_model),
    )
    config_file.write_text(to_yaml_str(harness), encoding="utf-8")
    console.print(f"\n[green]✓[/green] LocalHarness configured at {config_file}.")
    console.print("  Run 'localharness start' to begin.")


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
        response = httpx.get(f"{base_url}/api/ps", timeout=1.0)
        data = response.json()
        models = data.get("models", [])
        if len(models) == 1:
            return models[0].get("name")
    except Exception:
        pass
    return None
