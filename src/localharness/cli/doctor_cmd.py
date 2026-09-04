"""localharness doctor command — prerequisite checks."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from localharness.config.loader import ConfigLoader
from localharness.config.migrate import BACKUP_PREFIX, BACKUP_STAMP_FORMAT
from localharness.config.models import HarnessConfig
from localharness.config.paths import resolve_config_dir

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


def _print_overridden_keys(cfg_path: Path, workspace: Path) -> None:
    """Name the winning layer for every key this workspace actually CHANGES.

    Two catalogue builds — this session's layering, and the same machine with the workspace
    switched off — diffed by resolved VALUE. A workspace that restates a key with the value the
    global layer already had has overridden nothing, and listing it would make this section noise
    on exactly the configs people copy between projects.

    Value-diffed rather than presence-filtered on purpose; `registry.provenance.overridden_paths`
    owns that definition and `config show` reads the same function, so the two commands can never
    disagree about what "overridden" means.

    Kept out of doctor's command body: that function is already ~420 lines, and a section which
    prints nothing at all outside a workspace should be readable as one unit.
    """
    from localharness.registry.provenance import layered_catalogue, overridden_paths

    try:
        effective, _ = layered_catalogue(cfg_path, workspace)
        global_only, _ = layered_catalogue(cfg_path, None)
    except Exception as exc:  # noqa: BLE001
        # A broken config already fails loudly two lines below; this section must never be the
        # thing that crashes doctor, the one command people run when things are already wrong.
        console.print(f"       {escape(f'(layer report unavailable: {exc})')}")
        return

    rows = overridden_paths(effective, global_only)
    if not rows:
        console.print("       No overrides — the global config governs every key.")
        return
    console.print(f"       {len(rows)} key(s) overridden by this workspace:")
    for path, entry, before in rows:
        # The whole row goes through escape(): a band renders as `[workspace-config]`, which rich
        # would otherwise parse as a style tag and fail on, and a config VALUE can be any string
        # a user typed. (41-06's `[old] proj` lesson, applied to values as well as paths.)
        console.print(
            escape(f"         {path} = {entry.current_value!r}  [{entry.winning_layer}]"
                   f"  (global: {before!r})"),
            # soft_wrap so a long value is handed to the TERMINAL whole instead of arriving with
            # a newline folded into it — it still looks wrapped on screen, and it is one line in
            # the data. 43-04's measured lesson from the real binary, same wave, same class.
            soft_wrap=True,
        )


def _print_migration_state(cfg_path: Path, harness: HarnessConfig) -> None:
    """Say which shipped-defaults revision this config carries, and when it was last migrated.

    `localharness start` folds new shipped deny-defaults into config.yaml on the first start after
    an upgrade — announced, additive, lossless, with a timestamped backup — but that announcement
    prints once and scrolls away in a long or failing session, so the only durable trace was a
    .bak file nobody was told to look for (v0.13 dogfood F6). No new state is written to support
    this: the backup FILE is the record, and its own filename carries the timestamp.

    This is the one v0.13 output change that is NOT gated on a workspace, and that is deliberate —
    it is a product decision, not layering behavior. If the owner vetoes F6 before release, delete
    this function and its single call site; nothing else in doctor depends on it.
    """
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION

    stamped = harness.org.permissions.defaults_revision
    if stamped >= CURRENT_DEFAULTS_REVISION:
        console.print(f"{_INFO} Security defaults: revision {stamped} (current)")
    else:
        console.print(
            f"{_INFO} Security defaults: revision {stamped}, shipped revision is "
            f"{CURRENT_DEFAULTS_REVISION}. `localharness start` will fold the missing defaults in "
            f"on its next run, or run `localharness config migrate` now."
        )

    # Sorted lexicographically, which for BACKUP_STAMP_FORMAT is also CHRONOLOGICAL — do not
    # "fix" this into an mtime sort: a hand-copied backup keeps its name and loses its mtime.
    backups = sorted(cfg_path.glob(f"{BACKUP_PREFIX}*"))
    if not backups:
        return
    latest = backups[-1]
    stamp = latest.name.split(BACKUP_PREFIX, 1)[-1]
    try:
        when = datetime.strptime(stamp, BACKUP_STAMP_FORMAT).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        when = stamp
    # soft_wrap: a backup path folded across two lines is a path the user cannot copy (43-04).
    console.print(escape(f"       Last migrated {when}; backup at {latest}"), soft_wrap=True)


def doctor(
    config_dir: Annotated[
        str | None,
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
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
    cfg_path = resolve_config_dir(config_dir)
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
        # Glyph outside escape(), path inside — the same split 39-05's layer lines use, and for
        # the same measured reason: rich silently DELETES `[old]` from a folder named `[old] proj`,
        # so an unescaped path here names a file that does not exist.
        console.print(_PASS + " " + escape(f"Config file: {config_file}"), soft_wrap=True)
    else:
        console.print(
            _FAIL + " " + escape(f"Config file not found: {config_file}"), soft_wrap=True
        )
        console.print(f"       Run 'localharness init' to create it.")
        failures.append("config-missing")
        # Can't continue without config
        _summarize_and_exit(failures)

    # 3. Config file valid
    harness: HarnessConfig | None = None
    from localharness.cli.workspace import resolve_workspace_layer
    workspace = resolve_workspace_layer(config_dir)
    loader = ConfigLoader(config_dir=cfg_path, local_config_dir=workspace)
    # Success criterion 2: the chosen layer is stated, never silently applied. Printed only when
    # a layer applies — with none, doctor's output stays byte-identical to v0.12 (LAYR-03).
    # Phase 43 (CLI-02) is where doctor grows the full both-layers/per-key provenance report.
    if workspace is not None:
        # escape() around the path halves, glyph outside it: `_PASS` needs markup, a filesystem
        # path must not have it. Measured, not theoretical — a folder named `[old] proj` printed
        # as `/tmp/ proj/...` here, i.e. doctor reporting a path that does not exist, in the one
        # command people run to find out where their config comes from. (`[/]` raises outright.)
        console.print(_PASS + " " + escape(f"Workspace layer: {workspace}"), soft_wrap=True)
        console.print(escape(f"       Global layer:    {cfg_path}"), soft_wrap=True)
        _print_overridden_keys(cfg_path, workspace)
    try:
        harness = loader.load_harness()
        console.print(f"{_PASS} Config valid")
    except Exception as exc:
        # The message is 43-01's ConfigValidationError and CARRIES THE OWNING FILE'S PATH — the
        # whole point of that plan. Passing it through unescaped is how the right attribution
        # still reaches the user as a path they cannot open. escape() only, no string surgery:
        # if the attribution itself is ever wrong, the bug is in config/loader.py.
        console.print(_FAIL + " " + escape(f"Config invalid: {exc}"))
        failures.append("config-invalid")

    # F6, and the one v0.13 output change that is NOT workspace-gated (see the function's
    # docstring). ONE call site on purpose: an owner veto before release is a two-line revert.
    if harness is not None:
        _print_migration_state(cfg_path, harness)

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
        console.print(_FAIL + " " + escape(f"Config directory not writable: {cfg_path}"))
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
