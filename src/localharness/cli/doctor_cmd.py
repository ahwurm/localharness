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

            # 5b. Window reconciliation: configured budget vs served max_model_len.
            served = None
            try:
                entries = data.get("data", []) if isinstance(data, dict) else []
                for m in entries:
                    if m.get("id") == default_model or len(entries) == 1:
                        # #9: llama.cpp reports its served window as meta.n_ctx (no max_model_len).
                        served = (
                            m.get("max_model_len")
                            or m.get("context_length")
                            or (m.get("meta") or {}).get("n_ctx")
                        )
                        break
            except Exception:
                served = None

            # #13: LM Studio serves its window at /api/v0/models (not /v1/models). Query it the
            # same way `init` fits the budget: the loaded model's loaded_context_length, else the
            # largest max_context_length. Keeps doctor's reconciliation from false-reporting
            # 'max_model_len not reported' on a healthy LM Studio.
            if served is None and harness.provider.provider_type == "lmstudio":
                native = base_url.rstrip("/")
                if native.endswith("/v1"):
                    native = native[: -len("/v1")]
                try:
                    lm = httpx.get(f"{native}/api/v0/models", timeout=5.0).json()
                    lm_entries = [e for e in (lm.get("data") or []) if isinstance(e, dict)]
                    loaded = next(
                        (e for e in lm_entries
                         if e.get("state") == "loaded" and e.get("loaded_context_length")),
                        None,
                    )
                    if loaded:
                        served = int(loaded["loaded_context_length"])
                    else:
                        caps = [e["max_context_length"] for e in lm_entries if e.get("max_context_length")]
                        served = max(caps) if caps else None
                except Exception:
                    served = None

            cfg_ctx = None
            for _root_name in ("orchestrator", "default"):  # Phase 33.1: new root name, then pre-migration fallback
                try:
                    cfg_ctx = loader.load_agent(_root_name).context.max_context_tokens
                    break
                except Exception:
                    continue

            if served and cfg_ctx:
                reserve = 4_096
                if cfg_ctx > served:
                    console.print(
                        f"{_FAIL} Context budget {cfg_ctx:,} EXCEEDS served window "
                        f"{served:,} — compaction can't fire, long turns will 400 at the "
                        f"provider input cap. `start` clamps to {served - reserve:,}."
                    )
                    failures.append("context-budget-too-high")
                elif cfg_ctx < (served - reserve) * 0.75:
                    console.print(
                        f"{_FAIL} Context budget {cfg_ctx:,} is far BELOW served window "
                        f"{served:,} — wasting >25% of the window. Run 'localharness init' "
                        f"to refit (e.g. {served - reserve:,})."
                    )
                    failures.append("context-budget-too-low")
                else:
                    console.print(
                        f"{_PASS} Context budget {cfg_ctx:,} fits served window {served:,}"
                    )
            elif served is None:
                console.print(
                    f"{_INFO}  Served max_model_len not reported — can't reconcile context budget"
                )

            # 5c. Token-counting capability — aligned with #8's runtime-aware detection.
            # vLLM serves POST /tokenize -> {count}; llama.cpp serves POST /tokenize {content}
            # -> {tokens:[...]}; Ollama and LM Studio serve NO tokenize endpoint, so approximate
            # (tiktoken cl100k) counting is expected there — an INFO line, NOT a failure. Branch
            # on the detected provider_type so a healthy non-vLLM runtime does not false-fail.
            root = base_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[: -len("/v1")]
            provider_type = harness.provider.provider_type
            if provider_type in ("ollama", "lmstudio"):
                console.print(
                    f"{_INFO}  Token counting: approximate (tiktoken cl100k) — {provider_type} "
                    f"serves no /tokenize endpoint; budget gates fire conservatively. Use vLLM "
                    f"or llama.cpp for exact counts."
                )
            elif provider_type == "llamacpp":
                try:
                    tk = httpx.post(f"{root}/tokenize", json={"content": "token"}, timeout=5.0)
                    if tk.status_code == 200 and isinstance(tk.json().get("tokens"), list):
                        console.print(
                            f"{_PASS} Tokenizer endpoint reachable (/tokenize) — exact counts (llama.cpp)"
                        )
                    else:
                        console.print(
                            f"{_FAIL} llama.cpp /tokenize returned {tk.status_code} — token "
                            f"accounting falls back to tiktoken cl100k (approximate)."
                        )
                        failures.append("tokenize-unreachable")
                except Exception:
                    console.print(
                        f"{_FAIL} llama.cpp /tokenize unreachable at {root}/tokenize — token "
                        f"accounting falls back to tiktoken cl100k (approximate)."
                    )
                    failures.append("tokenize-unreachable")
            else:  # vllm (and unknown): the exact /tokenize {count} contract is expected here
                try:
                    tk = httpx.post(
                        f"{root}/tokenize",
                        json={"model": default_model, "prompt": "token"},
                        timeout=5.0,
                    )
                    if tk.status_code == 200 and "count" in tk.json():
                        console.print(f"{_PASS} Tokenizer endpoint reachable (/tokenize) — exact counts")
                    else:
                        console.print(
                            f"{_FAIL} /tokenize returned {tk.status_code} — token accounting "
                            f"falls back to tiktoken cl100k (inaccurate for non-cl100k models)."
                        )
                        failures.append("tokenize-unreachable")
                except Exception:
                    console.print(
                        f"{_FAIL} /tokenize unreachable at {root}/tokenize — token accounting "
                        f"falls back to tiktoken cl100k (inaccurate for Qwen et al.)."
                    )
                    failures.append("tokenize-unreachable")

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

    # 9. Web search dependency (builtin web_search tool needs ddgs)
    try:
        import ddgs  # noqa: F401
        console.print(f"{_PASS} Web search ready (ddgs installed)")
    except ImportError:
        console.print(f"{_FAIL} Web search unavailable: 'ddgs' not installed")
        console.print(f"       Run 'uv sync' to install it.")
        failures.append("ddgs-missing")

    _summarize_and_exit(failures)


def _summarize_and_exit(failures: list[str]) -> None:
    console.print()
    console.print(Rule())
    if failures:
        console.print(f"[bold red]{len(failures)} issue(s) found.[/bold red]")
        raise typer.Exit(code=1)
    else:
        console.print("[bold green]All checks passed.[/bold green]")
