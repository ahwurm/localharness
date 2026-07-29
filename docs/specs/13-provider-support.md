# Spec 13: Provider Support & Lifecycle

**Component:** `src/localharness/provider/lifecycle.py`, `src/localharness/provider/server.py`, `config/models.py` (`EndpointRef`, `ManagedServerConfig`, `LocalModelEntry`, `ActiveSelection`), `cli/repl.py` (`/model`)
**Status:** v1
**Dependencies:** [Spec 02: LLM Provider Layer](02-provider.md) (the request path + capability probe), [Spec 06: Configuration System](06-config.md) (the config models), [Spec 10: CLI](10-cli.md) (the REPL and `/model`)

---

## Purpose

"Support Ollama, LM Studio, and llama.cpp to a public-1.0 bar" sounds like a wire rewrite. It is not, and this spec exists to say why and to document the two things it actually is.

The **request path is already provider-agnostic.** Every backend is reached through one `LLMClient` (spec 02) wrapping `openai.AsyncOpenAI(base_url=…)`. There is no `VLLMProvider` / `OllamaProvider` class hierarchy — `ProviderType` (`detector.py`) is a bare `Literal["ollama", "vllm", "llamacpp", "lmstudio", "unknown"]` string tag, carried for labeling, not dispatch. The one thing that genuinely changes the code path is not provider identity but the per-client **capability probe** (`LLMClient.detect_capabilities`, spec 02): it sets `tool_call_mode ∈ {native, xml, text}`, and *that* selects native tool-calls vs XML injection via `FnCallConverter`. A model that dispatches tools natively takes the native path whether it is served by vLLM or Ollama.

So broadening provider support decomposed into exactly two subsystems, both new in the 0.11 milestone and neither touching the request path:

1. **A lifecycle layer** — the harness bringing a backend *up*, checking it, and tearing it *down* (`provider/lifecycle.py`).
2. **A cross-endpoint model tree** — `/model` switching a live session across endpoints and frameworks, launching a cold one on demand (`config/models.py` + `cli/repl.py`).

---

## Two orthogonal axes

`provider_type` answers *which wire protocol and tokenizer a backend speaks* — it drives the request path (spec 02) and the token counter (§Introspection). **Lifecycle** is the orthogonal axis: *how the harness launches, checks, and stops that backend.* The two are independent — the same OpenAI-compat protocol can be brought up as a docker container, a spawned binary, a warm-loaded daemon, or a CLI-driven server.

The concrete field carrying the lifecycle axis is `ManagedServerConfig.runtime` (`Literal["vllm", "llamacpp", "ollama", "lmstudio"]`, default `"vllm"`). It shares the same value-set names as `provider_type` but selects a *strategy*, not a protocol — `serve_command()` and `strategy_for()` both dispatch on `runtime`. A config never sets `runtime` unless it is a harness-managed launch; attach-only endpoints carry only `provider_type`.

---

## The lifecycle-strategy layer (`provider/lifecycle.py`)

A `LifecycleStrategy` is the launch-strategy abstraction. It unifies the heavy-to-heavy cross-framework `/model` swap — *stop the incumbent → confirm the accelerator is free → launch the target* — into something the REPL invokes uniformly instead of a manual box-dance.

```python
@dataclass
class LiveEndpoint:
    base_url: str          # where the brought-up backend serves
    served_models: list[str]  # ids it actually serves (wait_ready's /models list, or the intended key)
    handle: Any            # opaque stop target the SAME strategy understands: container name | pid | "lms-daemon"

@dataclass
class Liveness:
    alive: bool
    detail: str | None = None  # human-readable reason (container name / pid / why down)

@runtime_checkable
class LifecycleStrategy(Protocol):
    async def activate(self, spec: ManagedServerConfig, config_dir: Path, base_url: str, *,
                       timeout_seconds: float = 1800.0,
                       on_poll: Callable[[float], None] | None = None) -> LiveEndpoint: ...
    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None: ...   # verified + idempotent
    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness: ... # endpoint/name-based
```

The load-bearing contract on `liveness` (and on the `activate` fail-fast probes): **it is endpoint- or name-based, NEVER the client pid.** This is the `#99`/`#100` orphan-client-pid bug class — a `docker run --rm` client's pid is the sig-proxy client, not the server, so a live client pid does not mean the server is up. `activate` raises **only** `RuntimeError` / `TimeoutError` on failure; the `/model` swap callers catch exactly those to tear down and restore, so every strategy maps its native errors (`httpx.*`, `OSError`, non-zero exit) into that vocabulary — a leaked exception would crash the session with the just-spawned server orphaned.

All four strategies reuse the same `provider/server.py` launch primitives (`serve_command` → `start_server` → `wait_ready`; `stop_server`; the pidfile at `<config-dir>/vllm/server.pid`; `serve.log`) so the proven detached-launch / readiness / verified-stop recipe is shared, not reinvented. `stop` runs `server.stop_server` **off the event loop** via `asyncio.to_thread` — a verified stop can block ~90 s worst-case, and a synchronous call inside `async def` would freeze heartbeats, the channel adapter, and idle consolidation.

### `ManagedVllmStrategy` (`runtime="vllm"`, the default)

The harness-managed vLLM server (init guided setup / reboot autostart / `/model` swap). `launch="docker"` runs a foreground `docker run --rm --name localharness-vllm` client whose SIGTERM propagates to the container; `launch="binary"` execs a `vllm` executable. Both flow through the same `server.py` functions.

- **activate** = `serve_command` → `start_server` → `wait_ready` (returns served ids); `handle` = the container name (docker) or the pid (binary).
- **stop** = `stop_server(config_dir, launch=spec.launch)` — the `#100` verified stop.
- **liveness** = **name-based** for docker (`docker_container_running` via `docker inspect`, the fix); pid-based for binary (`server_pid`, correct there — the pidfile pid *is* the vLLM process).

### `SpawnedProcessStrategy` (`runtime="llamacpp"`)

A harness-spawned single-process OpenAI server — llama.cpp's `llama-server`. `serve_command` builds the simpler `llama-server -m <gguf> --host 127.0.0.1 --port <port>` shape (a GGUF file *is* the model, so there is no docker mount / `--served-model-name` indirection; the served alias `-a`, context `-c`, and GPU offload `-ngl` ride in `extra_args`).

- **activate** = the same `serve_command` → `start_server` → `wait_ready`; `handle` = the launched pid.
- **stop** = `stop_server(config_dir, launch="binary")` — SIGTERM the process group, SIGKILL on timeout. `launch` is pinned `"binary"`: a spawned server is a single tracked process, never a `docker run` client.
- **liveness** = `server_pid(config_dir) is not None` — **correct here** (unlike docker): the pidfile pid is the real `llama-server` process. This is exactly the case the docker orphan-pid bug was not.

### `DaemonStrategy` (`runtime="ollama"`)

The harness spawns the `ollama serve` daemon itself and loads the target model into it. Two Ollama facts shape it:

- `ollama serve` takes **no** model argument (models load lazily per request), so `activate` must *warm-load* `spec.model` — a `POST /api/generate` with an empty prompt (`keep_alive:-1`), Ollama's canonical "load into memory without generating" call — after the daemon is ready, plus a `_pull` if the tag is not on disk. Otherwise `served_models` would mean only "daemon up" (Ollama's `/v1/models` lists pulled-to-disk models, resident or not).
- **stop kills the whole daemon** (owner ruling): stopping the daemon process is a hard, verified accelerator-free guarantee (same class as docker-stop / process-kill), whereas a per-model `keep_alive:0` unload is a soft signal that may not promptly free memory. A spawned daemon leads its own process group (`start_new_session=True`), so `stop_server`'s `killpg` reaps the daemon *and* its runner children (which hold the weights).

Mechanically the daemon is a harness-spawned process, so it reuses `SpawnedProcessStrategy`'s launch / pidfile / verified-stop / pid-liveness primitives; the Ollama-specific parts are only the env-built `serve_command` (`env OLLAMA_HOST=… OLLAMA_KEEP_ALIVE=-1 [CUDA_VISIBLE_DEVICES=] ollama serve`), the warm-load, and the pull. `activate` **fail-fasts** if a daemon is already listening — the harness owns only daemons it pidfile-tracks; a second `ollama serve` could not bind and would silently mis-track, so attach-and-own is refused loudly (deferred to a later phase).

### `LmsStrategy` (`runtime="lmstudio"`)

The harness drives LM Studio's headless `lms` CLI: `lms daemon up` (start/attach the persistent `llmster` daemon) → `lms load <model> --gpu {max|off} -y` → `lms server start --port <p> --bind 127.0.0.1`, and tears it down with `lms daemon down`. Unlike `DaemonStrategy`'s foreground `ollama serve`, **every `lms` subcommand returns immediately** (the `llmster` daemon backgrounds itself), so there is no foreground process to pidfile-track. Two consequences:

- **liveness and the pre-existing-server fail-fast are endpoint-based** (`GET {v1}/models`), never a pidfile. `LmsStrategy` touches none of `server.py`'s pidfile primitives, so it **sidesteps the single-pidfile invariant entirely** (§GPU-lock) — an lmstudio launch never writes the shared pidfile a vLLM/llama.cpp/ollama launch does.
- **stop = `lms daemon down`** (the whole-daemon analog of the Ollama ruling), settled empirically 2026-07-29: a bare `lms server stop` tears down only the HTTP listener while `llmster` keeps the model resident (`lms ps` still idle, the engine still in memory) — a soft signal. `lms daemon down` reaps the whole daemon + engine. It returns rc 0 when it stopped something and rc 1 when already down; both are the goal state, so `stop` treats `{0, 1}` as success and propagates any other exit / missing binary / timeout as the activate contract's `(RuntimeError|TimeoutError)`.

The stop then **polls `pgrep -x llmster` until the process is actually gone** (≤ 50 attempts, ~5 s), because both the HTTP endpoint and `lms daemon status` flip to "down" ~0.4 s *before* the process holding the weights exits — so only process-gone is a sound accelerator-free signal on unified memory. If it never dies, `stop` raises rather than report a false free. Serving is local-only (`--bind 127.0.0.1`) on the caller's port; `--gpu off` when `spec.gpu` is False (so the lifecycle validates with no GPU window), `--gpu max` when heavy. A manual `--gpu` in `extra_args` is **rejected at config load** — it would (CLI last-wins) override the GPU-lock-derived flag and desync the bookkeeping into a two-heavy freeze.

### `strategy_for(spec)`

The single construction point. Dispatches on `spec.runtime`: `"llamacpp"` → `SpawnedProcessStrategy`, `"ollama"` → `DaemonStrategy`, `"lmstudio"` → `LmsStrategy`, anything else (`"vllm"`, the default) → `ManagedVllmStrategy`. The REPL `/model` swap and `start` autostart route through it, so the *strategy*, not the caller, owns how a backend is launched and stopped.

---

## The verified-stop doctrine (`#100`)

A cross-cutting invariant: **every `stop` confirms the resource is actually freed before returning.** A false free is the worst outcome — the caller would then launch a second heavy into a still-occupied accelerator (on unified memory, an instant OOM freeze). The verb-specific proofs:

| Strategy | Free proof |
|---|---|
| docker (vLLM) | `docker stop -t 60` → poll `docker inspect` until the name no longer resolves (`_wait_name_free`); `docker rm -f` fallback; **raise** if the name still resolves. A vLLM drain can outlast the grace and the `--rm` removal races the next `docker run` into a name conflict — the container must be *gone*, not merely signalled. |
| binary / spawned | SIGTERM the process group, then poll `_alive(pid)` (a zombie counts as dead, `#99`); SIGKILL on timeout. |
| daemon (Ollama) | same pid-group teardown — `killpg` reaps the daemon and its weight-holding runner children. |
| lms (LM Studio) | `lms daemon down`, then poll `pgrep -x llmster` until gone (the endpoint flips down ~0.4 s early). |

The deny-list backstop belongs here too: destructive `bash_exec` verbs (`docker stop/kill/rm`, `systemctl stop`, `pkill`/`kill`, `shutdown`) are blocked for the *subject model* (`PermissionConfig.deny_patterns`, issue `#15` — the run where the subject stopped its own vLLM). Lifecycle teardown is the harness's job, never the agent's.

After a verified stop, `free_accelerator` waits `GPU_FREE_SETTLE_SECONDS` (3.0). On the DGX Spark's **unified memory** (119 GiB shared CPU+GPU; no discrete VRAM), the verified stop — process/container *gone* — **is** the accelerator-free guarantee, because the kernel reclaims the pages on process exit. The settle is cheap insurance against a reclaim-vs-mmap race under memory pressure, not the gate itself, and is read from the module global at call time so it stays tunable from live swap data.

---

## The model tree & cross-framework switch

Two config shapes describe the tree:

- **`LocalModelEntry`** (`config/models.py`) — a locally-downloaded checkpoint the managed server can serve by name: `name` (picker id + `--served-model-name`), `path` (host checkpoint dir), per-model `extra_args` (appended after the server's shared args — e.g. a MoE-only flag that must never reach a dense sibling), and display-only `quant` / `tps` (`tps` set only from a real measurement, never estimated). A swap changes checkpoint, served id, and model-specific flags in one place (`serve_command`'s `entry_for` branch).
- **`EndpointRef`** (`config/models.py`, listed on `HarnessConfig.extra_endpoints`) — a *peer* endpoint the tree can switch to: `base_url` + `provider_type` + `api_key`/`extra_headers` + `gpu` + optional `lifecycle` (a nested `ManagedServerConfig` for a peer the harness can launch itself). `lifecycle=None` (the default) is an **attach-only** peer that must already be up. The primary provider is the implicit `endpoint[0]` — never duplicated into the list.

**The insight that makes cross-framework cheap:** the same-endpoint hot-swap already re-points `llm.config.model`, re-probes capabilities (`detect_capabilities`), and rebinds the token counter — with *no server call* (a running server that serves many models, e.g. Ollama, just answers a different id). Generalizing to a different framework only additionally re-points `base_url` + `provider_type` (`llm.rebind_endpoint`) and relabels the counter's tokenize contract. `_handle_model_cmd` resolves a target (number, name, or checkpoint path) against `choices = live + downloaded + peer + cold` and branches into one of the switch actions:

| Action | Condition | Mechanics |
|---|---|---|
| Already serving here | `target in live` | Re-point `llm.config.model`, re-probe, rebind counter. Instant, no server call. |
| Peer already running | `target in peer_target` | `rebind_endpoint(peer)` → re-probe → rebind counter with the **peer's** `provider_type` (so an Ollama target is labeled approximate, not the old vLLM's exact contract). |
| Cold, launch it | `target in cold_target` | Free the accelerator (§GPU-lock), `strategy_for(ep.lifecycle).activate(...)`, then rebind as above. Client binds to the real `live_ep.served_models[0]` the peer reports. |
| Downloaded, not served | else, managed | Re-point client back to the primary first, `strategy.stop` → mutate `managed.model` → `strategy.activate` (the vLLM restart-on-swap). |

Peer discovery (`_discover_peer_models`) probes the primary + each configured peer **concurrently** (each `_live_models` off-loop), mapping every not-locally-served model to its `EndpointRef`; the first-configured endpoint wins a name collision (the hidden duplicate is *noted*, not dropped), and an unreachable / malformed peer is skipped with a note, never raised. Peers are only probed when needed — a routine `/model <local-name>` pays zero peer-probe latency. Cold launchable peers (`_cold_lifecycle_targets`) surface even while down (config-derived served-names via `_cold_served_name`), so a heavy-swap target is pickable before it exists. A bare `/model` renders the tree **grouped by endpoint** (`_grouped_model_lines`) — each peer's models indented under a `▸ <framework> · <host:port>` header, numbered continuously so `/model <number>` matches what is shown.

A non-primary switch is persisted **additively** into `HarnessConfig.active_endpoint` (an `ActiveSelection`: name/base_url/provider_type/model/api_key). This never mutates the primary `provider`/`server` identity — reusing the default-model persistence would rewrite `server.model` and make the next `start` build a `vllm serve <peer>` command against a checkpoint that isn't there.

---

## The GPU-lock (single heavy at a time)

On a unified-memory box one heavy runtime pins most of the 119 GiB, so **at most one GPU-heavy launched server may serve at a time.** `free_accelerator` encodes the rule:

```python
async def free_accelerator(
    incumbent: tuple[ManagedServerConfig, str] | None,   # (spec, base_url) of the current GPU occupant
    target: ManagedServerConfig, config_dir: Path, *, settle_seconds: float | None = None,
) -> tuple[ManagedServerConfig, str] | None:             # returns the stopped incumbent, for restore
```

If a *different* heavy incumbent holds the accelerator, it verified-stops it through *its own* strategy, settles, and returns the stopped `(spec, base_url)` so a failed launch can restore it. It returns `None` when nothing heavy is up, or when the incumbent *is* the target (a same-server model-reload, whose stop the caller performs). It only ever **frees** — it never starts or probes anything.

The REPL tracks the occupant in `self._active_heavy` (a `(spec, base_url)` tuple), **independent of the bound endpoint** — the client can sit on a CPU-light peer while a heavy vLLM still holds the GPU. It is seeded at session start from the managed server and updated on every landed swap.

**The single-pidfile invariant.** The harness tracks one launched server via one per-config pidfile (`<config-dir>/vllm/server.pid`). A second launch would overwrite the first's handle → an orphaned, un-stoppable process. `EndpointRef._lifecycle_requires_gpu` enforces the consequence at config-load time: **a launchable peer (a `lifecycle` block) must be `gpu=True` on both the endpoint and its lifecycle spec** — a launchable heavy is then the sole live launched server, so the shared pidfile always tracks it. A CPU-light *launchable* peer would coexist with the heavy primary and break the invariant; attaching to an already-running CPU peer (`lifecycle=None`) is fully supported, *launching* one needs per-server tracking (a future phase). `LmsStrategy` writes no pidfile, so it sidesteps the invariant while still counting as gpu-heavy for the lock.

**Restore-on-failure.** A launch that fails after the incumbent was stopped must not leave the box with two heavies or nothing serving. The half-started target is torn down **first** with `_safe_stop` (a best-effort verified stop that swallows errors) *before* `_restore_incumbent` re-launches the old server — because the shared pidfile means the restore's own `start_server` would otherwise clobber the orphan's handle and strand a second untracked heavy. If restore also fails, the REPL reports honestly that the box needs manual attention rather than claiming a clean state.

---

## Introspection

Introspection is where "just OpenAI-compat" leaks, and the rule is **never a silent guess.**

**Token counting** (`TokenCounter`, `agent/context.py`). Prefers the served model's exact tokenizer, probed once at construction and re-probed on every `/model` swap (`rebind`, which also clears the content-hash cache — counts under a different tokenizer are wrong). Selected by `provider_type`:

- **vLLM** — `POST {root}/tokenize {model, prompt}` → `{"count": N}`.
- **llama.cpp** — `POST {root}/tokenize {content}` → `{"tokens": [...]}` (llama-server has no `count` field).
- **Ollama / LM Studio** — serve **no** `/tokenize` endpoint, so the harness counts EXACTLY from the served model's OWN GGUF (`agent/gguf_tokenizer.py`, mode `exact_local`): `llama-cpp-python` in `vocab_only` mode loads just the vocab + embedded chat template (no weights, no GPU) from the same GGUF the server runs — located on disk from the LM Studio models dir (`lms ls --json`) or the Ollama manifest→blob — and `count_messages` renders the model's real chat template then tokenizes, so the count equals the server's own to the token (verified live vs LM Studio `usage.prompt_tokens` and Ollama `prompt_eval_count`). Only when no local GGUF / `llama-cpp-python` is reachable (e.g. a remote server) does it fall back to the labeled-approximate cl100k×`APPROX_TOKENIZE_SAFETY_FACTOR` estimate — which *undercounts* Qwen by ~1.85× on digit/code text, so the factor biases to over-count (compaction fires early instead of a real overflow 400, issue `#8`).
- **Unknown** `provider_type` probes both exact shapes and locks onto whichever answers. A runtime *known* to serve `/tokenize` (vllm/llamacpp) that is unreachable is a **hard error** — it refuses to substitute an approximate count for a runtime that should count exactly.

**Context window** (`probe_served_window`, `agent/context.py` — the provider-aware source of truth; blocking `httpx`, run off-loop by callers). Returns an int ("refit the budget to this") or `None` ("unknowable — disclose, don't guess", `#31`):

- **vLLM / OpenAI-compat** — the `/v1/models` entry whose `id == model`, field `max_model_len` | `context_length` (id-matched, so a multi-model endpoint returns the *target* model's window). `init` fits the budget as `max_model_len − output_reserve`.
- **llama.cpp** — `GET /props` `n_ctx` (at the server root, not `/v1`).
- **LM Studio** — `GET /api/v0/models` `loaded_context_length` — the window the model was *loaded* with, never `max_context_length` (over-reporting would let the `start` guard pass a config that then 400s mid-session).
- **Ollama** — `GET /api/ps` `context_length` of the *loaded* model (the served `num_ctx`, **not** `/api/show`'s model ceiling, which over-reports). Because Ollama lazy-loads on first request, this is absent until the model is resident, so `init` cannot reliably auto-probe it and falls back to the config value with an explicit approximate flag; the in-session refit reads `/api/ps` once loaded.

`doctor` (spec 10) reconciles all of this live per runtime — tokenize contract reachable, served window vs configured budget — and labels every approximate path.

---

## Support status

The [provider support matrix](../../README.md#supported-runtimes) is the honest scoreboard; per-runtime setup lives in the `docs/runtimes/` pages (`llamacpp.md`, `ollama.md`, `lmstudio.md`; vLLM in the [reference architectures](../reference-architectures/README.md)).

One honest line: **all four runtimes have a harness-managed lifecycle and a live end-to-end round-trip** (detect → serve → tool-call → verified stop) on the DGX Spark reference machine, including a live cross-framework heavy-swap. **Recorded bench runs currently exist for vLLM only**; the other runtimes ship opt-in `bench.yaml` entries you run against your own model, gated by `LOCALHARNESS_LIVE_{OLLAMA,LLAMACPP,LMSTUDIO}=1`.
