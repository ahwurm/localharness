"""localharness doctor command — prerequisite checks."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.rule import Rule

from localharness.config.loader import ConfigLoader
from localharness.config.models import HarnessConfig

console = Console()

_PASS = "[green]✓[/green]"
_FAIL = "[bold red]✗[/bold red]"
_INFO = "[cyan]i[/cyan]"


def doctor(
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR"),
    ] = "~/.localharness",
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Attempt to auto-fix detected issues."),
    ] = False,
) -> None:
    """Run prerequisite checks and report system health.

    Checks Python version, config, LLM endpoint reachability,
    model availability, and directory structure.

    Exit code 0 if all pass, 1 if any fail.
    """
    cfg_path = Path(config_dir).expanduser()
    failures: list[str] = []

    console.print()
    console.print(Rule("LocalHarness Doctor"))

    # 1. Python version
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 12):
        console.print(f"{_PASS} Python {py_str} (required: >=3.12)")
    else:
        console.print(f"{_FAIL} Python {py_str} (required: >=3.12)")
        failures.append("python-version")

    # 2. Config file exists
    config_file = cfg_path / "config.yaml"
    if config_file.exists():
        console.print(f"{_PASS} Config file: {config_file}")
    else:
        console.print(f"{_FAIL} Config file not found: {config_file}")
        console.print(f"       Run 'localharness init' to create it.")
        failures.append("config-missing")
        # Can't continue without config
        _summarize_and_exit(failures)

    # 3. Config file valid
    harness: HarnessConfig | None = None
    loader = ConfigLoader(config_dir=cfg_path)
    try:
        harness = loader.load_harness()
        console.print(f"{_PASS} Config valid")
    except Exception as exc:
        console.print(f"{_FAIL} Config invalid: {exc}")
        failures.append("config-invalid")

    # 4. LLM endpoint reachable
    if harness is not None:
        base_url = harness.provider.base_url
        # Determine models endpoint
        if "11434" in base_url:
            models_url = base_url + "/api/tags"
        else:
            models_url = base_url + "/v1/models"
        try:
            resp = httpx.get(models_url, timeout=5.0)
            console.print(f"{_PASS} LLM endpoint reachable: {base_url}")

            # 5. Model available
            default_model = harness.provider.default_model
            model_ids: list[str] = []
            try:
                data = resp.json()
                if "data" in data:
                    model_ids = [m["id"] for m in data.get("data", [])]
                elif "models" in data:
                    model_ids = [m["name"] for m in data.get("models", [])]
            except Exception:
                pass

            if not model_ids or default_model in model_ids:
                console.print(f"{_PASS} Model available: {default_model}")
            else:
                console.print(f"{_FAIL} Model not found: {default_model}")
                console.print(f"       Available: {', '.join(model_ids[:5])}")
                failures.append("model-not-found")

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            console.print(f"{_FAIL} LLM endpoint unreachable: {base_url}")
            console.print(f"       {exc}")
            failures.append("llm-unreachable")
    else:
        console.print(f"  (skipped: no valid config)")

    # 6. Config directory writable
    if os.access(cfg_path, os.W_OK):
        console.print(f"{_PASS} Config directory writable")
    else:
        console.print(f"{_FAIL} Config directory not writable: {cfg_path}")
        failures.append("config-dir-not-writable")

    # 7. Agents directory
    agents_dir = cfg_path / "agents"
    if agents_dir.exists():
        console.print(f"{_PASS} Agents directory exists")
        if fix and not os.access(agents_dir, os.W_OK):
            agents_dir.mkdir(parents=True, exist_ok=True)
    else:
        if fix:
            agents_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"{_PASS} Agents directory created")
        else:
            console.print(f"{_INFO}  Agents directory not found (run --fix or 'localharness init')")

    # 8. Tool call mode info
    if harness is not None:
        sfn = harness.provider.supports_function_calling
        if sfn is True:
            console.print(f"{_INFO}  Tool calling: native")
        elif sfn is False:
            console.print(f"{_INFO}  Tool calling: XML fallback")
        else:
            console.print(f"{_INFO}  Tool calling: unknown (run 'localharness init' to probe)")

    _summarize_and_exit(failures)


def _summarize_and_exit(failures: list[str]) -> None:
    console.print()
    console.print(Rule())
    if failures:
        console.print(f"[bold red]{len(failures)} issue(s) found.[/bold red]")
        raise typer.Exit(code=1)
    else:
        console.print("[bold green]All checks passed.[/bold green]")
