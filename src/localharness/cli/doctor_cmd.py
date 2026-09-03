"""localharness doctor command — prerequisite checks."""
from __future__ import annotations

import os
import subprocess
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
_WARN = "[yellow]⚠[/yellow]"

_AMD_PCI_VENDOR_ID = "0x1002"


def _has_amd_gpu() -> bool:
    """An AMD/ATI GPU is present, via sysfs PCI vendor IDs — no subprocess, no extra binary
    required (rocm-smi/lspci may not be installed even when an AMD GPU is). Linux-only
    (issue #144's plan): returns False, never raises, on any other platform, and degrades the
    same way in a container/sandbox where /sys/class/drm isn't readable."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        for vendor_file in Path("/sys/class/drm").glob("card[0-9]*/device/vendor"):
            try:
                if vendor_file.read_text().strip().lower() == _AMD_PCI_VENDOR_ID:
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def _llama_server_linked_backends(binary: str) -> "set[str] | None":
    """Which GPU backend(s) `binary` is linked against — a subset of {'hip', 'vulkan'} (both
    empty means CPU-only/unknown backend libs). None means UNDETERMINABLE (not Linux, `ldd`
    missing, binary missing/unreadable) — callers must treat that as "can't tell", never as
    "CPU-only", or a missing `ldd` would falsely read as a healthy CPU build.

    Recognizes both the current llama.cpp/ggml library names (`libggml-hip`/`libggml-vulkan`)
    and the underlying vendor libs they link against (`libamdhip64`/`libvulkan`) so this works
    across the `LLAMA_HIPBLAS`/`GGML_HIPBLAS` -> `GGML_HIP` build-flag rename too (issue #144).
    """
    if not sys.platform.startswith("linux"):
        return None
    if not Path(binary).is_file():
        return None
    try:
        proc = subprocess.run(
            ["ldd", binary], capture_output=True, text=True, timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.lower()
    backends: set[str] = set()
    if "libamdhip64" in text or "libggml-hip" in text:
        backends.add("hip")
    if "libvulkan" in text or "libggml-vulkan" in text:
        backends.add("vulkan")
    return backends


def _strip_v1(base_url: str) -> str:
    """Return the server root: base_url minus a trailing slash and /v1 suffix.

    ProviderConfig.base_url is always written WITH a /v1 suffix (config/models.py;
    detector._build_base_url — "Always includes /v1"). Native endpoints — Ollama's
    /api/tags, llama.cpp/vLLM's /tokenize, LM Studio's /api/v0 — live at the server root,
    so probes must strip /v1 first (mirrors init_cmd's removesuffix("/v1")).
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def doctor(
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR"),
    ] = "~/.localharness",
    fix: Annotated[
        bool,
        typer.Option(
            "--fix",
            help="Create a missing agents directory (doctor's only auto-fix today).",
        ),
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
        # base_url always carries a /v1 suffix — build probes from the stripped root so we
        # hit <root>/v1/models (not /v1/v1/models) and Ollama's native <root>/api/tags.
        root = _strip_v1(base_url)
        if "11434" in base_url:
            models_url = root + "/api/tags"
        else:
            models_url = root + "/v1/models"
        try:
            resp = httpx.get(models_url, timeout=5.0)
            console.print(f"{_PASS} LLM endpoint reachable: {base_url}")

            # 5. Model available
            default_model = harness.provider.default_model
            model_ids: list[str] = []
            data: dict = {}
            try:
                data = resp.json()
                if "data" in data:
                    model_ids = [m["id"] for m in data.get("data", [])]
                elif "models" in data:
                    model_ids = [m["name"] for m in data.get("models", [])]
            except Exception:
                pass

            status = getattr(resp, "status_code", None)
            if isinstance(status, int) and not (200 <= status < 300):
                # A real error status means the model list can't be trusted — an empty/404
                # body must NOT sail through the benefit-of-doubt pass below. (A live
                # httpx.Response always exposes an int status; the isinstance guard keeps
                # under-specified test mocks on the benefit-of-doubt path.)
                console.print(f"{_FAIL} Model check failed: {models_url} returned HTTP {status}")
                failures.append("model-not-found")
            elif not model_ids or default_model in model_ids:
                console.print(f"{_PASS} Model available: {default_model}")
            else:
                console.print(f"{_FAIL} Model not found: {default_model}")
                console.print(f"       Available: {', '.join(model_ids[:5])}")
                failures.append("model-not-found")

            # 5b. Window reconciliation: configured budget vs served max_model_len.
            # #137 (critic fix): the budget below is resolved for the ROOT AGENT's model, so
            # the served window must be matched against that SAME model. Keying it to
            # provider.default_model paired one model's pin with another model's window on
            # multi-model endpoints — a specifically-attributed, backwards verdict.
            cfg_ctx = None
            pinned_for: str | None = None
            budget_model = default_model
            for _root_name in ("orchestrator", "default"):  # Phase 33.1: new root name, then pre-migration fallback
                try:
                    _root = loader.load_agent(_root_name)
                    budget_model = _root.model if _root.model != "inherit" else default_model
                    cfg_ctx, _is_pinned = _root.context.resolve_budget(budget_model)
                    pinned_for = budget_model if _is_pinned else None
                    break
                except Exception:
                    continue

            served = None
            try:
                entries = data.get("data", []) if isinstance(data, dict) else []
                for m in entries:
                    if m.get("id") == budget_model or len(entries) == 1:
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
                try:
                    lm = httpx.get(f"{root}/api/v0/models", timeout=5.0).json()
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

            # The effective budget (#137) was resolved above, before the served-window lookup,
            # so both halves of this check speak about the SAME model.
            if served and cfg_ctx:
                reserve = 4_096
                # Name the value actually checked — a pinned number and the scalar are different
                # settings, and the fix for each lives in a different place.
                attr = f" (pinned for {pinned_for})" if pinned_for else ""
                if cfg_ctx > served:
                    console.print(
                        f"{_FAIL} Context budget {cfg_ctx:,}{attr} EXCEEDS served window "
                        f"{served:,} — compaction can't fire, long turns will 400 at the "
                        f"provider input cap. `start` clamps to {served - reserve:,}."
                    )
                    failures.append("context-budget-too-high")
                elif cfg_ctx < (served - reserve) * 0.75:
                    fix = (
                        f"Raise context.model_context_overrides['{pinned_for}']"
                        if pinned_for
                        else "Run 'localharness init' to refit"
                    )
                    console.print(
                        f"{_FAIL} Context budget {cfg_ctx:,}{attr} is far BELOW served window "
                        f"{served:,} — wasting >25% of the window. {fix} "
                        f"(e.g. {served - reserve:,})."
                    )
                    failures.append("context-budget-too-low")
                else:
                    console.print(
                        f"{_PASS} Context budget {cfg_ctx:,}{attr} fits served window {served:,}"
                    )
            elif served is None:
                console.print(
                    f"{_INFO}  Served max_model_len not reported — can't reconcile context budget"
                )

            # 5c. Token-counting capability — ASK THE COUNTER, don't re-derive it.
            #
            # This block used to branch on `harness.provider.provider_type` and run its own
            # probe per runtime. That forked the truth: `TokenCounter` treats provider_type as a
            # HINT and probes both exact shapes, so it self-heals when the config has drifted
            # from the running server — doctor did not, and on a box whose config said
            # "llamacpp" while the server spoke vLLM it sent a llama.cpp-shaped body, got a
            # rejection, and reported "token accounting falls back to tiktoken cl100k" while the
            # counts were in fact EXACT. The tool people run for reassurance was the wrong one.
            #
            # So doctor now constructs the SAME counter the runtime uses and reports its
            # RESOLVED mode. One resolver, one answer, and the two can no longer disagree.
            if "model-not-found" in failures:
                # Cascade guard: the counter needs a served model name. With a stale one, every
                # downstream probe fails FOR THAT REASON and doctor invents a second, phantom
                # problem — a user then chases two issues that are one.
                console.print(
                    f"{_INFO}  Token counting: not checked — fix the model above first "
                    f"(the tokenizer probe needs a served model name)."
                )
            else:
                from localharness.agent.context import TokenCounter

                try:
                    _tc = TokenCounter(
                        base_url=harness.provider.base_url,
                        model=default_model,
                        provider_type=harness.provider.provider_type,
                    )
                    _mode = _tc.mode
                except Exception as exc:
                    _mode = "off"
                    console.print(f"{_INFO}  Token counting: probe failed ({exc})")

                if _mode in ("vllm", "llamacpp"):
                    console.print(
                        f"{_PASS} Token counting: exact — server-side /tokenize "
                        f"({_mode} contract)."
                    )
                    if _mode != harness.provider.provider_type:
                        # Not a failure: the counter already adapted. But the config is stale and
                        # anything else reading it (including a future doctor check) will be wrong.
                        console.print(
                            f"{_INFO}  config says provider_type={harness.provider.provider_type!r} "
                            f"but the server speaks {_mode!r} — counting adapted automatically; "
                            f"run `localharness init` to persist the real one."
                        )
                elif _mode == "exact_local":
                    console.print(
                        f"{_PASS} Token counting: exact — from the served model's own GGUF vocab "
                        f"(this runtime serves no /tokenize)."
                    )
                elif _mode == "approximate":
                    # Approximate is EXPECTED on runtimes that serve no tokenizer (Ollama, LM
                    # Studio) and a real FAULT on ones that should (vLLM, llama.cpp). That
                    # expectation is a property of the RUNNING server, so ask the endpoint
                    # resolver — never `harness.provider.provider_type`, which is the stale
                    # value this whole rewrite exists to stop trusting.
                    from localharness.cli.init_cmd import _identify_endpoint_provider

                    try:
                        _actual = _identify_endpoint_provider(harness.provider.base_url)
                    except Exception:
                        _actual = None
                    # #6: quantify it. "approximate" alone is not actionable; measured against a
                    # real Qwen tokenizer the cl100k estimator is ~0% off on English and JSON,
                    # -3.5% on code, and +141% on CJK. The error is content-shaped, not uniform.
                    _caveat = (
                        "Error is content-dependent: ~0% on English/JSON, ~4% on code, "
                        "but >100% on CJK, so context budgets can be badly wrong on non-Latin text."
                    )
                    if _actual in ("vllm", "llamacpp"):
                        console.print(
                            f"{_FAIL} Token counting: approximate (inflated cl100k) — this "
                            f"server should serve /tokenize but none answered. {_caveat}"
                        )
                        failures.append("tokenize-unreachable")
                    else:
                        console.print(
                            f"{_INFO}  Token counting: approximate (inflated cl100k) — this "
                            f"runtime serves no /tokenize and no local GGUF was usable. {_caveat}"
                        )
                else:
                    console.print(
                        f"{_FAIL} Token counting: no tokenizer resolved (mode={_mode!r})."
                    )
                    failures.append("tokenize-unreachable")

                # MESSAGE-LEVEL exactness. Content-level exact (above) is not the same as
                # whole-message exact: only a server that applies its own chat template makes
                # count_messages == usage.prompt_tokens. Each runtime exposes that differently —
                # vLLM via /tokenize messages-mode, llama.cpp via /apply-template — so the probe
                # stays runtime-specific. It is gated on the RESOLVED mode, never the configured
                # provider_type, which is the whole point of this rewrite.
                if _mode == "vllm":
                    try:
                        mt = httpx.post(
                            f"{root}/tokenize",
                            json={"model": default_model,
                                  "messages": [{"role": "user", "content": "x"}],
                                  "add_generation_prompt": True},
                            timeout=5.0,
                        )
                        msg_exact = mt.status_code == 200 and "count" in mt.json()
                    except Exception:
                        msg_exact = False
                    if msg_exact:
                        console.print(
                            f"{_PASS} Tokenizer endpoint reachable (/tokenize) — exact counts, "
                            f"message-level (chat template applied server-side)"
                        )
                    else:
                        console.print(
                            f"{_INFO}  /tokenize has no messages mode — whole-message counts use "
                            f"a summed estimate; upgrade vLLM for template-exact message counts."
                        )
                elif _mode == "llamacpp":
                    try:
                        at = httpx.post(
                            f"{root}/apply-template",
                            json={"messages": [{"role": "user", "content": "x"}],
                                  "model": default_model},
                            timeout=5.0,
                        )
                        msg_exact = at.status_code == 200 and isinstance(at.json().get("prompt"), str)
                    except Exception:
                        msg_exact = False
                    if msg_exact:
                        console.print(
                            f"{_PASS} Tokenizer endpoint reachable (/tokenize) — exact counts, "
                            f"message-level via /apply-template (llama.cpp)"
                        )
                    else:
                        console.print(
                            f"{_INFO}  /apply-template not served — whole-message counts use a "
                            f"summed estimate; upgrade llama.cpp for template-exact message counts."
                        )

            if harness.provider.provider_type == "llamacpp":
                # Slot-window check. llama-server may run several slots, but slot COUNT alone
                # proves nothing: with a unified KV cache (modern llama.cpp default) the slots
                # SHARE one cache and --ctx-size is not divided, so every request still sees the
                # full window. Keying on `total_slots > 1` therefore false-fired on healthy
                # servers — and contradicted the very number it printed ("~1/4 ... (131072
                # tokens/slot)" for a 131072 launch). /props reports the true per-request window,
                # so compare THAT to the budget and speak up only when a request genuinely
                # cannot hold it. Warn, never fail — a shared server may be deliberate.
                try:
                    props = httpx.get(f"{root}/props", timeout=5.0).json()
                    n_slots = int(props.get("total_slots") or 1)
                    slot_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
                    if n_slots > 1 and slot_ctx and cfg_ctx and int(slot_ctx) < cfg_ctx:
                        console.print(
                            f"{_INFO}  llama.cpp gives each request {int(slot_ctx):,} tokens "
                            f"across {n_slots} parallel slots — below the {cfg_ctx:,} context "
                            f"budget, so this server is dividing --ctx-size among its slots. "
                            f"LocalHarness is single-stream: relaunch llama-server with "
                            f"--parallel 1 so one slot owns the full window, then re-run "
                            f"'localharness init --force'."
                        )
                except Exception:
                    pass  # /props absent on older builds — the check is advisory only

                # AMD backend check (#144): llama.cpp's cmake quietly PREFERS Vulkan over
                # HIP/ROCm unless the build explicitly disables it, and -ngl offloads onto the
                # GPU either way — so a Vulkan build looks perfectly healthy while running the
                # slower, less-tested backend. Only checkable for a harness-MANAGED llama.cpp
                # server: we need the actual binary path, which an attach-only endpoint never
                # gives us. Advisory only (never fails) — a Vulkan build still works.
                if (
                    harness.server is not None
                    and harness.server.runtime == "llamacpp"
                    and harness.server.binary
                ):
                    backends = _llama_server_linked_backends(harness.server.binary)
                    if (
                        backends is not None
                        and "vulkan" in backends
                        and "hip" not in backends
                        and _has_amd_gpu()
                    ):
                        console.print(
                            f"{_WARN} AMD GPU detected, but {harness.server.binary} is linked "
                            f"against Vulkan, not HIP/ROCm — it will run, but on the slower, "
                            f"less-tested backend. Rebuild with -DGGML_HIP=ON -DGGML_VULKAN=OFF "
                            f"-DAMDGPU_TARGETS=<your gfx> (see docs/runtimes/llamacpp.md)."
                        )
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
