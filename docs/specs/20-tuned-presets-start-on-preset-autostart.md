# Spec 20 — Tuned Presets: Out-of-the-Box Launch, `start --preset`, Autostart

**Status:** Revision 10, 2026-10-16 — §11 widened to the first-run job (§1 item 5, G7) and the interview (§11.5); four rounds of four critics: round four parsimony SHIP, design / security / friction REVISE with every item folded here (§15) and no fifth pass, by the owner's instruction to plan on this text (the plan checker verifies the folds); revision 9 was vetted (eight design-critic, eight security-critic and eight friction-critic passes; round 8: **SHIP from all three** on §§5–16, with eight minors folded in here — see §15 Iteration 8; §17 is an optional, out-of-scope extension at the owner's request, advisory only)
**Version target:** 0.19.0
**API version:** `PLUGIN_API_VERSION` unchanged ("1")
**Numbering:** 18 is the task record (reserved by spec 17 §1), 19 the delegation surface; this is 20.

---

## 1. Problem

Spec 16 (0.18.0) made every GGUF on the box a **preset**: `init` discovers the files, writes a `server:`
record and `extra_endpoints[]` peers, and `start` / `/model` launch them with a fitted `-c`. What a
discovered preset still lacks is everything a person puts into a hand-written profile script. The owner's
own box (RTX 5090, llama.cpp `43fe9c642`, vLLM 0.31.0) has six such scripts under `~/setup/llm/profiles/`
and a switcher (`llm start <profile>` → `ln -sfn` + `systemctl --user restart llm.service`), and the
harness cannot reproduce any of the four things they do:

1. **The flags that make a model fast and fit.** The profiles pin `--cache-type-k q8_0 --cache-type-v q8_0`,
   `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.6`, `-ngl 99`, `--n-cpu-moe 48`,
   `-fa on`, `--jinja`, `-np 1`. A preset `init` writes from discovery carries `model`, `mmproj`,
   `served_name` and `port` and nothing else (`cli/init_cmd.py:841 _peer_records`, by design: "no binary,
   args, env or key file"), so its launch is `llama-server -m … --mmproj … -a … -c <fit> --port … --host
   127.0.0.1`: f16 KV (half the window the same memory holds at q8_0), no MTP drafting (the profile's
   notes measure ~116 t/s decode with it), nothing else. Every one of those flags is **derivable**: the
   GGUF header says whether the model has MTP layers (`nextn_predict_layers`, present on both 27B files),
   whether it is MoE (`expert_count`), and llama-server's own `--fit` (on by default in this build,
   `common/common.h:481`) already places layers and experts.
2. **The model's own sampling is overwritten on every request.** The GGUFs carry
   `general.sampling.temp = 1.0`, `top_k = 20`, `top_p = 0.95` (read on the box with the harness's own
   header reader), llama-server loads them as its defaults (`common/common.cpp:1088
   common_init_sampler_from_model`, called at `:1325`), and the harness then sends `temperature: 0.6` with
   every request: `LLMConfig.temperature: float = 0.6` (`provider/client.py:72`), written into the body at
   `:1251`, `:1761`, `:1841` unconditionally. The value is the dataclass default, nothing else: neither
   `LLMConfig(...)` call in `start` (`start_cmd.py:1829`, `:2102`) passes a temperature, so the loader's
   resolved `AgentConfig.temperature` (agent > division > `org.default_temperature`, `loader.py:1662–1667`)
   never reaches a request — the owner's critics' `temperature: 0.2` is not sent today, and
   `loader.py:1682` records that switching the rung on "needs its own decision". The owner's profiles send
   `--temp 1.0` for exactly this model; the harness undoes it with a constant.
3. **Switching models does not survive a restart.** `/model <peer>` and `localharness model <peer>` persist
   the choice as `active_endpoint` (`cli/model_ops.py:310`), and **nothing reads it back**
   (`model_ops.py:327`: "the `start` path does not yet READ `active_endpoint` back"; `start_cmd.py:2032`).
   Worse: the pidfile is one per config dir, shared by the primary and every peer (`cli/repl.py`
   `_safe_stop`), and holds `<pid> <start>` with no port (`provider/server.py:873`). After `/model
   flash-next` and quit, the next `start` probes the primary's port (nothing answers), takes the launch
   branch (`start_cmd.py:1849`), asks `_managed_server_running` (`:278`), which reads the live pidfile — the
   peer's — and answers *yes*, so `start` prints "still loading — waiting" and polls the primary's port for
   `wait_ready`'s 1800 s bound (`:783–785`). The person's last choice is not resumed; it is a 30-minute
   hang (by reading; the live run in §14 confirms it).
4. **No way to start on a chosen model, and nothing comes up at login.** `start --model` is a hot pick
   among already-served ids and refuses a managed single-model server (`start_cmd.py:3103`). The owner's
   box starts its server from a user unit; the harness has no equivalent, and a person who wants the
   server up before the harness, or for other clients, writes the unit by hand.

5. **A fresh box is not one command.** `pip install localharness` then `localharness start` does run
   `init`'s setup on a terminal (owner ruling 2026-10-03), and what it finds it writes without a question:
   every GGUF under `~/models` and the HF cache becomes a preset, the ranked best becomes `server:`, and
   the session starts on it. But a box with **no** model gets a vLLM-only menu of two reference
   architectures (`_guided_setup`): nothing for llama.cpp, the runtime the harness is developed against
   and the one the 8 GB consumer-GPU architecture uses, and no link to a file the box could run. A model
   that arrives **after** the first run never becomes a preset: an existing install with a `server:`
   block imports nothing (`_existing_install_import`), and discovery's peers are written only while
   `extra_endpoints` is empty. The person is never asked which model should serve first, and the receipt
   names `/model` but not `start --preset`, `model <name>` or `--autostart` — the commands this spec adds
   are the ones nobody is told about.

The jobs, in the owner's words: *"a user goes `localharness init`, it scans for model files, … makes
[start/swap scripts with optimized configs], then `localharness start --scripts` allows you to start
localharness with a specific start script … and `/model` begins to work correctly out of the box."* And the whole of it, 2026-10-16 (lightly edited):
*"fresh computer, download localharness, run `localharness start`, and it runs you through it: if you don't
have the model it gives you download links; if it autosenses a model is downloaded it asks whether to make a
start/swap script for the models it found (any that has no script and was not designated a no), sets them
up, asks which to launch, and tells you the start command for later — training people that they can be
lazy, run `localharness start`, and have things work out."*

## 2. Ground truth (what exists)

### 2.1 Reused from spec 16, unchanged

This spec is a layer on spec 16 (0.18.0), not a sibling. Every mechanism below is used as spec 16 built
it; nothing here redefines a preset, a launch path, a trust kind or a line.

| Spec 16 mechanism | Where it is defined | How spec 20 uses it |
|---|---|---|
| A preset = `server:` or an `extra_endpoints[]` peer with a `lifecycle` (`ManagedServerConfig`), `None` = "the harness computes it", no `"auto"` literal | §3 G1, §5 | the three new record fields follow the same spelling; the three knobs join `autosense:`, which already rides every `activate` |
| One launch decision on every launch path, `activate` the one enforcement point, `serve_command` pure, derived values as keywords that fill an unset record slot (`context=`, `cache_ram=`) | §5, §6.2 | derivation produces three more such keywords; one sensed-then-planned launch is what `activate` executes and what `serve`, `--script` and `doctor` read (§6) |
| The fit, the window line on change, the halve-once retry, "no log text is parsed"; `wait_ready` | §6.1–6.2 | the fit gains the draft-context share; the retry widens to the derived flags, is chosen in one place (`plan_launch`) and confirmed in one place (`confirm_launch`); the unit's wait is the same `wait_ready` with a liveness callback |
| `init` discovers GGUFs, writes presets, shows one line per preset, launches nothing; an interactive `init` adopts exactly what it showed | §8, §9 | discovery, the lines and the write move into the model library (§11); the preset line gains `mtp`; the vLLM peers show their binary; "asks nothing" becomes one prompt on a terminal (§11.1) |
| Import from `/proc`, the journal and a unit file; `provider/units.py` (`UnitState`); `provider/procfs.py` (`read_process`, `_unit_from_cgroup`); the incumbent-sentence builder | §8, §10 | autostart (§9) reuses `user_units`/`read_unit`/`read_journal`, `unit_file_state` for the identity test, `read_process` to verify a pidfile's process, and the sentence builder |
| Machine trust: `launch` / `launch_model` kinds, one key table, adoption per kind, withholding (`MachineTrust.withheld`, the loader's `machine_withheld=`), `decide_machine_trust(ask=can_ask, launching=)` (`start_cmd.py:1564–1565`; `can_ask` is tty + no `LOCALHARNESS_COMMIT`), the headless rule | §9 | the three new fields join `LAUNCH_MODEL_KEYS`; `serve` never adopts; a withheld record is withheld, never "gone"; **every surface that launches runs `decide_machine_trust` with `ask=<local operator>` and loads with `machine_withheld`** — `start` as today, `serve` with `ask=False`, the CLI with `ask=_is_local_operator()` (§8.1); a `decide_machine_trust(ask=True)` literal never appears under `cli/` (structural test) |
| `active_endpoint`, `persist_active_endpoint` / `clear_active_endpoint`, the D-35 preset resolution | §5, §7.2, §9 | one ladder (`resolve_serving`) reads every selection store; one `persist_landed` writes the outcome after success on every surface |
| `local_operator` evidence for a CLI command | §9 | gates `--autostart on\|off` and the CLI's immediate swap |
| The speed ledger as the per-model on-change store | §6.1, §7.4 | the sampling line, `last_served` and the remembered retry live there |
| `open_regular`, `touch_private`, `overlay.atomic_write_text` (spec 17), the bounded GGUF header reader, `_visible`/`shown_path`, `strip_url_credentials`, `config/redact.py` (`scrub`, `secret_values`), `log_tail`, `child_environment` (a blocklist over a base, `server.py:736–770`) | §6.2, §8 | every small file the harness owns is written through `atomic_write_text` (moved into `core/private_files.py`, re-exported by `overlay`) and read through `open_regular`; every shown line is escaped and redacted at the channel boundary; `serve`'s environment is `child_environment` over an allowlisted base, so spec 16's blocklist contract stands |
| `resolve_effort` from the live `base_url` | §7.2 | unchanged; §8.4 |
| Zero `init` asks; lines on change only; `/reasoning` untouched; the one machine-trust question at a terminal `start` | §12 | carried as rulings, except the zero-asks ruling, which spec 16 recorded as reversible by the owner and §11 revises to one prompt |

Spec 16 items this spec **closes**: `start` never reading `active_endpoint` back (`model_ops.py:327`);
the per-record pidfile follow-up ("IN-15"); the swap-back/peer resumption gap; a planted pidfile making
the harness signal a process (§8.3); `log_tail` reaching a channel unredacted (§9.4); the pidfile write
following a planted symlink (§8.3). Spec 16 §12 items that stay **open**: launching a peer beside a
foreign unit when it fits; Ollama's VRAM-tiered window; records for models Ollama/LM Studio hold.

### 2.2 The tree and the box

Line numbers against the tree at `ffdf2b4` (v0.18.0) and the owner's llama.cpp build `43fe9c642`
(2026-10-06), vLLM 0.31.0, systemd as installed (`systemd.exec(5)`).

| Piece | Where | State |
|---|---|---|
| llama.cpp launch argv | `provider/server.py:604–672 serve_command` | `[binary, -m, model, -a, --mmproj, --cache-type-k/-v, -c, --cache-ram, --api-key-file, --port, (--host 0.0.0.0), *extra_args, (--host 127.0.0.1)]`; pure; the record's `context`/`cache_ram` win over the fit keywords |
| Launch sizing and the retry | `provider/lifecycle.py:218–253`, `:280–385 SpawnedProcessStrategy.activate` | decision, sizing and `prepare_config_ini_home` run once (`:309–331`); the retry rebuilds argv at `ctx_kw // 2`; stops inside `activate` at `:361, 373, 395` |
| `activate` call sites, `strategy_for`, `free_accelerator`, the lock | `start_cmd.py:818`, `repl.py:2130, 2281, 2555, 2689`; `lifecycle.py:730` (pure, twelve call sites); `:755–780 free_accelerator`; `launch_lock` (`:831`, an asyncio context manager over the cross-process `flock` on `server.lock_path`, with `_CONTENDED`) | every call passes `autosense=` |
| Discovery | `agent/gguf_header.py:177, 186, 424`; `cli/init_cmd.py:967, 841, 873` | header read per candidate; a peer record carries `model, mmproj, served_name, port, gpu, runtime` only |
| What the header holds on the box | `read_header` | the two 27B files: `qwen35`, 65 layers, `head_count_kv 4`, `key_length 256`, `context_length 262144`, **`nextn_predict_layers 1`**, `general.sampling.temp 1.0 / top_k 20 / top_p 0.95`; Flash-Next: `qwen4exp`, 48 layers, **`expert_count 512`**, no `nextn_predict_layers` |
| llama-server defaults in this build | `common/arg.cpp`, `common/common.h`, `tools/server/server.cpp` | `--fit on`; `-ngl auto`; `-fa auto`; `--jinja` on; `-np -1` → 4 slots, `kv_unified = true`; `--spec-type none` (`n_max 3`, `p_min 0.0`); `--reasoning-format auto` |
| What `--fit` does | `common/fit.h`, `common/fit.cpp:463, 488, 523, 618` | modifies only unset params (ctx only when 0); aborts when `n_gpu_layers` is set; MoE expert tensors to the CPU first |
| MTP in llama.cpp | `src/llama-arch.cpp:227`; `include/llama.h:601`; `common/speculative.cpp:2599–2648`; `src/llama-context.cpp:3884–3888` | `draft-mtp` builds an MTP draft context against the target; refused without MTP layers; the draft context holds its own KV for the `nextn_predict_layers` layers |
| Quantized KV and FA | `src/llama-context.cpp:3843–3849` | quantized V needs FA; `-fa auto` enables it; only `-fa off` errors |
| Sampling defaults from the model | `common/common.cpp:1088–1146`, `:1325`; `tools/server/server-task.cpp:45–81` | `general.sampling.*` become the server's defaults; `/props` reports them under `default_generation_settings.params`, the response `probe_window` already fetches (`agent/context.py:1199`) |
| The harness's temperature | `provider/client.py:72, 1251, 1761, 1841`; `config/defaults.py:12`; `config/models.py:1317, 1525, 1683`; `config/loader.py:1662–1667, 1682`; `start_cmd.py:1829, 2102` | the dataclass default is sent; the resolved rung is inert; root and children share one `LLMClient`; the two `temperature=0.0` probes are deliberate |
| Persisted selection; `model_cmd`'s loader | `cli/model_ops.py:310, 373`; `cli/model_cmd.py:119–135` (`ConfigLoader(config_dir=, local_config_dir=)` **without `machine_withheld=`**, no `decide_machine_trust`), `:200–222, 316–318`; `start_cmd.py:1581–1583` (passes `machine_withheld` after `decide_machine_trust`); `cli/repl.py:2773`; `_launch_key`/`_peer_session_key` at `cli/repl.py:183, 202` | `model_cmd` only persists today, so an unwithheld load did not matter |
| Liveness and the pidfile | `cli/start_cmd.py:278`; `server.py:915, 873, 900`, `:817–819` (`touch_private` + `write_text`, follows a planted symlink); `stop_server` (`:944`); `agent/gate_types.py:934` (no `vllm/`, no `speed_stats.json`) | liveness answers "is *a* harness launch alive"; the pidfile, the ledger and `vllm/` are agent-writable; a planted `<pid> <start>` makes the harness signal that group (pre-existing); Ollama shares the pidfile, port in `OLLAMA_HOST`; docker's pid is the client |
| `/proc` readers | `provider/procfs.py:91 ProcRecord` (no `sid`), `:269 _unit_from_cgroup` (a claim), `:353` imports `server.start_time_from_stat`; `units.py:32` imports `procfs` | `server` ↔ `procfs` and `procfs` ↔ `units` would cycle |
| `start`'s launch branch | `cli/start_cmd.py:1849–1870`, `:770–815` | probes `provider.base_url`; pidfile alive → wait on the primary's port (1800 s; `wait_ready` has no port, `server.py:1060`) |
| `start --model` | `start_cmd.py:3103` | session-only hot pick |
| REPL incumbent and swap | `cli/repl.py:430–457`, `:1849`, the cold branch `:2080–2160` (`_active_heavy` at `:2114, 2156`), `:2478`, `:2533`, `:2502`; `_active_launch` | nothing at REPL exit restores the primary |
| User units | `provider/units.py:67, 77 (_run: PATH systemctl), 111, 127, 173, 312`; `lifecycle.py:541–556 LmsStrategy` | the box: `llm.service` disabled and inactive; lingering on |
| What decides what runs at the next login | `agent/gate_types.py:882–887`; `config/models.py:513–515`; `config/paths.py:8–10, 32–42`; `cli/workspace.py:844–860`; `config/loader.py:2196–2207` | the user manager reads `~/.config/systemd/user` **and** `~/.local/share/systemd/user`, `~/.config/systemd/user.conf`, `~/.config/environment.d/`, `systemctl --user set-environment` **and D-Bus** (`SetEnvironment`, `EnableUnitFiles`, `LinkUnitFiles`); `systemctl --user link\|enable /abs\|edit\|set-property\|add-wants` write into those directories; the desktop runs `~/.config/autostart/*.desktop`; all silent in `auto` |
| systemd facts this spec relies on | `systemd.exec(5)`, `systemd.service(5)` as installed | **`UnsetEnvironment=` is applied as the final step and undoes `Environment=` too** ("may undo assignments from any configuration source, including … `Environment=`"); `$INVOCATION_ID` is a new 128-bit id each time the unit goes from inactive to activating (a `Restart=` is a new invocation; an `exec` inside one is not); `NRestarts` counts **automatic** restarts and resets on a manual start; `StartLimitBurst` counts **every** start and `systemctl reset-failed` resets the counter; a comment line (`#`) in a unit file is ignored silently, an unknown key is warned about |
| The installed harness | `~/.local/bin/localharness` → the uv tool venv's console script; `pyproject.toml:131`; no `__main__.py` | `python -I -m localharness` does not run today; the owner's llama.cpp finds its libraries by rpath; `child_environment` (`server.py:749`) says `CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, PATH, HF_HOME` must survive; the owner's shell `PATH` begins `~/.local/bin` (user-writable), which the uv tool tree the unit runs from shares |
| CLI tree | `cli/app.py:34–49` | `model` has two modes already; `ask-rate` is hidden; `model_cmd` runs coroutines with `asyncio.run` |
| Local-operator evidence | `cli/model_cmd.py:28` | tty + no `LOCALHARNESS_COMMIT`; a forged tty passes (spec 16 §8 residual) |
| HF cache | `provider/server.py:558` (`scan_cache_dir`, honours `HF_HOME`; the owner exports `HF_HOME=~/models/hf` in the shell), `:514 find_vllm` | the owner's `vllm` is on no search path |
| vLLM 0.31.0 flags | `vllm serve --help=all` | `--reasoning-parser`, `--tool-call-parser {…}`, `--enable-auto-tool-choice`, `--generation-config auto`, `--max-model-len` derived when unset; vLLM/Triton look up `cc`/`ninja` on `PATH` at JIT time |
| Machine trust, ledger, `notify_ratio` | `config/trust.py:189, 201, 603`; `provider/speed_stats.py:105 _load` (plain `read_text`), `:130, 194, 204`; `config/models.py:2426` | a `None` field contributes no entry; the ledger has no read cap |
| Writers and redactors | `config/overlay.py:137 atomic_write_text`; `config/redact.py:55 scrub`, `:154 secret_values`; `server.py:36` imports from `config` | the one config writer; a key suffix and a key held only in an `api_key_file` are not masked today |

## 3. Goals

- G1. **A discovered preset launches tuned**: KV cache type, MTP drafting and the fit derived from the GGUF
  header and the machine at launch, overridable by a record field or an `extra_args` flag, named on the
  launch line; a vLLM launch gets its parsers. Nothing is pinned.
- G2. **The model's own sampling applies** unless a person set a temperature, and the harness says so only
  when the two differ.
- G3. **The model that is serving is the session's model; an explicit choice is honoured now, at the next
  start and at the next boot; the last choice is launched when nothing is.** One ladder decides on every
  surface, every launching surface runs the same trust decision, and a peer's launch is never mistaken for
  the primary's.
- G4. **Autostart is one command**: keeps a server the harness already runs, never runs a unit it cannot
  prove is its own, retries once, bounds its own loop without refusing the person's switches, and tells
  the person once what happened at boot.
- G5. **The launch is readable** (`localharness model --script`), generated from the record, never read back.
- G6. Everything is additive; a 0.18 config loads; a hand-run server keeps working.
- G7. **A fresh box is one command.** `localharness start` finds a model or offers one to fetch, confirms
  the presets it will make and which serves first, and names the command for later; a model that arrives
  later is found at the next terminal `start`. One question on the common path, answered by Enter; none
  off a terminal. Two components own it: the model library and the interview (§11).

## 4. Non-goals

- No new runtime; no benchmark-driven tuning; no `-ngl`, `--n-cpu-moe`, `-t`, `-np` or `-fa` from the
  harness. No change to the one-heavy lock (spec 13). A per-child temperature (§16). The task record stays
  spec 18.

## 5. Config shape

```yaml
server:
  runtime: llamacpp
  model: ~/models/gguf/Swift-1.5-Qwen3.8-27B-NVFP4-Q8mix.gguf
  mmproj: ~/models/gguf/mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf
  served_name: qwen-27b-swift-1.5
  # cache_type_k / cache_type_v: unset = autosense.kv_cache_type (§6.1)       (existing fields)
  # speculative: off | draft-mtp     # new, llamacpp only; unset = derived from nextn_predict_layers (§6.2)
  port: 8080
extra_endpoints:
  - name: deckard-31b
    base_url: http://127.0.0.1:8000/v1
    provider_type: vllm
    gpu: true
    lifecycle:
      gpu: true
      runtime: vllm
      binary: /home/awurm/.venvs/vllm/bin/vllm
      model: DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking
      # reasoning_parser: gemma4      # new, vllm only; unset = derived from config.json model_type (§6.3); off = none
      # tool_call_parser: gemma4      # new, vllm only; off = none
autosense:                            # existing machine-only section; three knobs join the ratios
  kv_cache_type: q8_0                 # the derived K and V type for llama.cpp; f16 restores 0.18's launch
  draft_mtp: true                     # whether a model with MTP layers drafts with them
  draft_p_min: 0.6                    # --spec-draft-p-min for a derived draft-mtp (llama.cpp's own default is 0.0)
# pending_preset: flash-next          # written by `localharness model <name>` from the CLI; read by start and serve (§8.1)
org:
  default_temperature: null           # now Optional; null = send none, the server's defaults apply (§7)
```

Every new record field is `Optional`, `None` = derived, runtime-gated in `_launch_target_present`; the
pin-off word is `off`. Precedence for every derived flag: **record field > `extra_args`/`env:`/`config_ini`
flag (any spelling) > derivation** (the `has_ctx_arg` rule). `speculative`, `reasoning_parser`, `tool_call_parser`
join `LAUNCH_MODEL_KEYS` (a `None` yields no entry); `pending_preset` joins `HARNESS_GLOBAL_ONLY_FIELDS`;
`kv_cache_type` is validated against llama.cpp's type list; `draft_p_min` in `[0, 1]`.

## 6. One sensed, planned, confirmed launch

### 6.0 `sense_launch` → `plan_launch` → execute → `confirm_launch`

```
sense_launch(spec, config_dir, autosense, *, unit) -> Sensed      # the I/O, once; writes nothing
plan_launch(spec, sensed, decision, *, autosense, api_key_held) -> LaunchPlan   # pure; owns the retry choice
LaunchPlan.retry() -> LaunchPlan                                   # pure, from the same Sensed
confirm_launch(config_dir, record_key, *, plan: LaunchPlan | None, marker: Marker | None,
               live: LiveEndpoint, process: ProcRecord, unit: UnitRuntimeState | None,
               context_length: int) -> None                        # the ONE ledger writer
```

`live.window.served` is the served window (`probe_window(...).served`, the one reading spec 16's window
line uses); `process` is the verified `ProcRecord` whose argv decides `dropped`; `unit` carries the live
`InvocationID` and `NRestarts`; `context_length` comes from `Sensed` when called from a plan and from the
bounded header reader when called from a marker.

`sense_launch` reads the header (`FIT_KEYS` + `nextn_predict_layers`, `expert_count`), free memory,
`/proc/meminfo`, the weights size, `config.json` (vLLM), the ledger's remembered retry and the boot marker
(§9.3) — all through `open_regular` under caps. `plan_launch` fits, derives, builds the `serve_command`
keywords, the environment **names**, the `config_ini_home` path, the report and `tuned`, and **alone
decides** whether the retry shape applies (§6.4). `activate` senses, plans, **prepares** the `config.ini`
home (the one write, by an executor only) and executes (`start_server` → `wait_ready`); on a death with
derived keywords it executes `plan.retry()` from the first plan. `serve` (§9.3) senses, plans, prepares
and `execve`s; `--script` and `doctor` render.

**`confirm_launch` is the only writer of the ledger's `retry` record and the only remover of the boot
marker.** It never senses: after a load the free memory is the loaded machine's and would say nothing
true about the fit. Two forms. **From a plan** (`activate`, which still holds the `Sensed` taken before the
launch): a successful retry writes `retry` from that plan's values; a full-shape success clears a
remembered record. **From a marker** (`start`'s attach and `UnitStrategy`'s wait, §9.3–9.4), **only when a
marker exists**: the marker must be bound to the run that is serving — `marker.invocation` equal to the
unit's live `InvocationID` (`show -p InvocationID`; `serve` writes the marker and `exec`s inside the same
invocation, so equality means "the process serving now is the one this marker describes"), the verified
live process's argv `-c` equal to the marker's planned `-c` (not the `/props` window: llama.cpp rounds `-c`
up, 37001 → 37120), **and the unit's `NRestarts ≥ 1`** (a marker alone,
with no automatic restart behind it, never writes the ledger); `dropped` is `["draft-mtp"]` **only when the
record's own `derive` would have emitted `spec_type` for this header and knobs** and the verified live
process's argv carries no `--spec-type` — a record with `speculative: off` or a header without MTP layers
can never be recorded as having dropped drafting, whatever a marker claims; `full_fit` and `free_bytes` come
from the marker **clamped** (`full_fit ≤` the header's `context_length`, `free_bytes ≤` total memory), since
a planted value only moves the threshold at which the full shape is tried again. Any marker whose
invocation matches is removed on that attach, whatever the window said; a marker whose invocation does not
match is left for the run it belongs to (it expires on age, §9.3). `LaunchPlan` holds no secret.
`LiveEndpoint` gains `tuned`.

`derive(spec, header, autosense) -> Tuned` is pure (`provider/tuning.py`): the derived keyword values
(`cache_type`, `spec_type`, `draft_p_min`, `reasoning_parser`, `tool_call_parser`; `None` = nothing derived)
and one reason each. `serve_command` emits each **at the record field's own slot, only when the field is
unset**. Header ints are validated (`block_count > 0`, `0 ≤ nextn_predict_layers ≤ block_count`, int type).

### 6.1 KV cache type (llama.cpp)

Neither `cache_type_k/v` set, no cache type in `extra_args`/`env:`/`config_ini`, no `-fa off` (off, 0, false or disabled, any case, in any of
the three) → `cache_type =
autosense.kv_cache_type` (default `q8_0`) for K and V; the fit prices it. `f16` (with `draft_mtp: false`)
reproduces 0.18's argv keywords; the fit of a record whose `env:`/`config_ini` sets a cache type, or that
drafts with MTP by hand, is now priced at those types (intended), so its `-c` can differ from 0.18's.

### 6.2 MTP drafting (llama.cpp)

`autosense.draft_mtp` true, `speculative` unset, no `--spec-type`/`-md`/`--spec-draft-*`/`-hfd`/
`LLAMA_ARG_SPEC_*` (in `extra_args`, `env:` or `config_ini`), header `nextn_predict_layers ≥ 1` → `--spec-type draft-mtp --spec-draft-p-min
<autosense.draft_p_min>`. `speculative: off` pins off; `speculative: draft-mtp` is a record value the retry
never drops. **The fit counts the draft context:** `kv_bytes_per_token × (1 + nextn_predict_layers /
block_count)` with drafting on (`66/65` on the 27B). Verified live (§14).

### 6.3 Parsers (vLLM)

Neither parser set, none of `--reasoning-parser`/`--tool-call-parser`/`--enable-auto-tool-choice` in
`extra_args` → `config.json` (local path or the HF snapshot path from `huggingface_hub`'s cache API;
`open_regular`, 1 MiB cap, `json.loads` with `RecursionError`/`ValueError` caught) → `model_type` → one
table in `provider/tuning.py`:

| `model_type` | `--reasoning-parser` | `--tool-call-parser` |
|---|---|---|
| `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_5`, `qwen3_5_moe` | `qwen3` | `qwen3_coder` |
| `gemma4` | `gemma4` | `gemma4` |
| `glm4_moe`, `glm4v_moe` | `glm45` | `glm45` |
| `deepseek_v3` | `deepseek_r1` | `deepseek_v3` |
| `gpt_oss` | `openai_gptoss` | `openai` |
| `llama`, `llama4` | — | `llama3_json` / `llama4_json` |
| `mistral`, `mistral3` | — | `mistral` |
| `kimi_k2` | — | `kimi_k2` |

A hit emits the pair plus `--enable-auto-tool-choice`; no hit derives nothing; `off` pins a side off.
`--max-model-len`, `--kv-cache-dtype`, `--gpu-memory-utilization`, `--max-num-seqs` are not derived.

### 6.4 The retry: chosen in one place, confirmed in one place, remembered

A launch that dies before readiness with **any** derived keyword is stopped (verified) and relaunched
**once** as `plan.retry()` (derived `spec_type` dropped, derived `-c` halved from the same sensed fit),
then given up with the log tail. The launch line's derived clause: `mtp off, -c 57344 (retried: the full
shape died) · server.speculative: off keeps the full window`. KV-only launches are not retried; record and
`extra_args` values are never touched.

**`plan_launch` is the only place the retry shape is chosen**, from two sensed facts:

- `Sensed.remembered`: the ledger entry `retry = {full_fit, free_bytes, dropped, binary, kv_cache_type,
  draft_p_min, at}` that `confirm_launch` wrote. When the fresh fit is not larger than `full_fit`, the plan
  is the retry shape — the `-c` is **the fresh fit halved, never the recorded number**, `dropped` a closed
  set (`⊆ {"draft-mtp"}`). A fresh fit above `full_fit` tries the full shape; `confirm_launch` clears the
  record on its success; a changed `binary`, `kv_cache_type`, `draft_p_min`, or a record
  `context`/`speculative` set since, makes it moot. The derived clause reads `mtp off (remembered <date>)`
  on change.
- `Sensed.prior_death` (§9.3): a boot marker **for this record and binary**, `cause: death` or written with
  `shape: full`, younger than `TimeoutStartSec`, with the unit's `NRestarts ≥ 1`.

The ledger is read through `open_regular` under a cap (`speed_stats._load`, `:105`).

### 6.5 What the launch says

The launch line gains the derived clause in its existing parenthesis, never the fitted `-c` (the window
line prints it): `Managed llama.cpp is not running — starting it (kv q8_0 · mtp (1 layer); model load can
take several minutes)...`. `doctor` prints a `Tuning:` row per launchable record (sense + plan, nothing
prepared). `init`'s preset lines gain `mtp`.

## 7. Sampling: the model's own

Two decisions, kept apart (`loader.py:1682`): (1) **the constant goes** — `LLMConfig.temperature: float |
None = None`, the three builders include `"temperature"` only when not None (`_sampling_kwargs`), the two
`0.0` probes stay, the bench stops sending `0.6` (README names the knob); (2) **the rung is switched on for
the root** (owner's decision, §15) — `start` passes the root agent's resolved `temperature` into both
`LLMConfig` calls; `OrgConfig.default_temperature`, `DivisionConfig.temperature` (today `float = 0.6`) and
`AgentConfig.temperature` become `Optional[float] = None`; `_resolve_scalar` drops `None`;
`DEFAULT_TEMPERATURE` and the `model_fields_set` guard go; a child file's `temperature` stays inert (§16).
**Migration:** `init_nulls` revision 2 nulls exactly `org.default_temperature: 0.6` in `config.yaml` and
`overrides.yaml` (same backup, receipt, read-only retry; agent and division files untouched); receipt
clause `org.default_temperature nulled (it was never applied; the model's own sampling now governs — set
org.default_temperature: 0.6 to restore)`; `init` writes `null`. **What is said:** `WindowProbe.sampling`
from the `/props` response already fetched; one line on change (ledger `sampling`), only when a rung sets a
temperature **and** the server's default differs: `sampling: temperature 0.6 from org.default_temperature
(the model's own is 1.0 — unset org.default_temperature to use it)`; nothing beside the receipt; `doctor`
always prints the `Sampling:` row.

## 8. `start` serves what serves, honours an explicit choice, launches the last one

### 8.1 One ladder, every surface

`model_ops.resolve_serving(harness, trust: MachineTrust, preset: str | None) -> Serving` is the one ladder
`start`, `serve`, `localharness model` and `--script` read. `_launch_key` and `_peer_session_key` move from
`cli/repl.py:183, 202` into `cli/model_ops.py`.

```
--preset <name>  >  pending_preset  >  unit-target  >  active_endpoint  >  server: (the primary)
    explicit          explicit          in flight       resumed             default
```

- `<name>`: a lifecycle peer by `name` or served name, or the primary by its served name — **exact match
  first**, then case-folded, then a **unique** prefix (ambiguous → refused naming both); otherwise `unknown
  preset '<name>' — presets: …`, exit 2.
- Names resolve against **every** record, then `trust.withheld` is checked: a **withheld** record →
  `start` and `localharness model` print one refusal line (spec 16's withheld line), `serve` exits 78,
  **no selection is cleared**. Only a name with **no record** is gone: an `active_endpoint` is matched by
  `base_url`, then by a unique `model`; a `pending_preset` or `unit-target` naming nothing is removed with
  one line.
- Three stores, three lifetimes: `pending_preset` (unconsumed intent, `model_cmd`), `unit-target` (in-flight
  target, written **only** by `UnitStrategy.activate` through `units.write_target`, removed or rolled back
  by `swap` on a non-binary target, a failed swap or a cancellation), `active_endpoint` (last success).
- **One `model_ops.persist_landed(record)`** writes the outcome after a successful launch or swap on every
  surface — `persist_active_endpoint` or `clear_active_endpoint`, `last_served` (§8.2), and the removal of
  `pending_preset` — called by `start`, the REPL wrapper (replacing its `_persist_landed`) and `model_cmd`.
- `Serving` carries the **record** (never a selection row's `base_url`), the rung, and `explicit: bool`.

**`localharness model <name>` from the CLI.** It first runs **`decide_machine_trust(ask=local,
launching=will_launch)`** with `local = _is_local_operator()` (tty + no `LOCALHARNESS_COMMIT`,
`model_cmd.py:28`) — exactly `start`'s `ask=can_ask` (`start_cmd.py:1564`): the agent runs this command
too, and `_confirm_on_a_tty` is `rich.Confirm.ask`, which reads whatever stdin holds, so `ask=True` would let
`echo y | localharness model x` adopt an agent's own machine-file edit, and an EOF would record a No nobody
chose. Off a terminal only the **uncovered** entries are withheld (on the common path there are none), and
a withheld target is refused with spec 16's line, `… — run localharness start on a terminal`, exit 2. It
loads the harness **with `machine_withheld`** as `start` does (`start_cmd.py:1581–1583`); today's loader
call (`model_cmd.py:119`) passes neither, which did not matter while the command only persisted.
`will_launch` is true when the swap below will launch, so a llama.cpp `bind_all: true` target is asked
about at its moment of use rather than quiet-withheld (`workspace.py:810–826`). **The question is gated on
the terminal; the action is not.** The CLI performs the `UnitStrategy` swap **now** when the harness's own
unit is **active** (§9.2's identity test), **a machine record exists** (`recorded_machine(dir) is not
None` — the same guard `serve` has, so a swap never rides the silent first-start adoption at
`workspace.py:854–856`), **`LOCALHARNESS_COMMIT` is absent** (the agent's shell and python tools set it on
every command; a script, a cron job or an `ssh -T` run does not) and the target is **not withheld**. This
is **not a boundary against the agent** — it can clear the marker — and it is not meant as one: the swap
grants nothing the agent does not already have through `systemctl --user restart localharness-llm.service`
(allowed) and a `pending_preset`/`unit-target` write; it selects among fingerprinted, confirmed records,
the way `serve` does at boot (a docker, Ollama or LM Studio target launches outside the unit, as `start`
launches it). What the marker buys is that an agent command, by default, takes the deferred path. This
keeps the owner's `llm start <profile>` working from a script. The swap runs through `lifecycle.swap` under
`asyncio.run` (the launch lock is the cross-process `flock`, so a REPL launching at the same time waits on
it as it does today), with a console `SwapReporter` that prints the launch line **before** the wait
(`stopping qwen…, starting flash-next (kv q8_0 · mtp (1 layer); model load can take several minutes)...`)
and the receipt after: `Serving flash-next — autostart restarted; a REPL session on this machine now talks
to it` — the second clause when `session_presence.live_sessions(config_dir)` (spec 16's registry of live
sessions, pruned by pid) is non-empty. `persist_landed` follows. Otherwise (no active unit, or the agent)
it persists `active_endpoint` as today **and** writes `pending_preset: <name>` (the primary swap-back too);
the receipt reads `Selected <name> — the next start serves it`, or `… — the next start or boot serves it`
**only when the harness's unit is enabled**. `start` consumes the key through `persist_landed` — after a
launch or swap, **and after an attach that resolved on the `pending_preset` rung** (a boot served it; the
attach is the outcome); `serve` reads it and **never removes it** (it `execve`s). Two structural tests: the immediate-swap path cannot run
without a `MachineTrust` in hand, and no `decide_machine_trust(ask=True)` literal appears under `cli/`
beyond the gated `on`.

### 8.2 Attach, swap or launch

`start` senses what is serving with one sensor shared with `/model` (`model_ops.sense_serving`):
`read_launch` names the port a harness launch holds; the sensor connects to the selected record's
`base_url` and to every other launchable record whose `base_url` is loopback, in parallel. Then:

- **The selected record answers** → attach on that record. **`last_served`** (ledger top-level: the record
  name; written by `persist_landed` and by every successful attach in `start` and the REPL, never by
  `serve`) gates two clauses on separate branches, so one event triggers at most one: the resume clause
  `(your last /model choice)` prints iff the rung is `active_endpoint` and `last_served ≠ record`; the
  attach-to-another clause below prints iff `last_served ≠ the answering record`. When a boot marker
  exists, `confirm_launch` runs from it (§6.0) and the attach line carries **once** `(retried at boot: mtp
  off, -c halved · server.speculative: off keeps the full window)` — only for a `cause: death` marker when
  the ledger held no `retry` record before.
- **Another preset answers, selection not explicit** → attach to **it**, never launch beside it; one clause
  (`serving flash-next; your last choice was qwen-27b-swift-1.5 — /model qwen-27b-swift-1.5 swaps`); **no
  selection is persisted** (`last_served` is written); `unit-target` stays on the older choice (stated).
- **Another preset answers, selection explicit** → the swap.
- **Nothing answers** → launch the selected record after the liveness test; `persist_landed` after success.

**The swap**: `lifecycle.swap(incumbent: Heavy | None, target: Heavy, *, config_dir, autosense, unit,
reporter: SwapReporter) -> SwapOutcome`, lifted from `repl.py:2080–2160` with `_restore_incumbent` and
`_safe_stop`. `Heavy(spec, base_url, peer, launch_kw)` is built by the caller (provider imports no `cli`)
with the spec object itself. `SwapReporter`: `freed()`, `launched(live)`, `restored(spec)`, `line(text)`.
**Cancellation**: after the incumbent was freed, `swap` stops the half-started target (verified), calls
`freed()`, rolls `unit-target` back to the incumbent, runs no restore. Under the harness's unit, a
docker/Ollama/LM Studio target launches through its own strategy outside the unit, `swap` **removes**
`unit-target` (the ladder then falls to `active_endpoint`), and its launch line carries `(runs outside
autostart — the next boot serves nothing)`. The REPL wrapper keeps its own `_launch_lock`, `_active_launch`,
`rebind_endpoint`, capability re-probe, token-counter refit and `_announce_window`; persistence is
`persist_landed`. The "holds the GPU" fact is folded into the launch line.

**The stopper is the strategy's, chosen by the launch's owner.** `read_launch(config_dir, *, expect,
main_pid)` reports `Launch.owner`: `"unit"` iff the verified process sits in the `localharness-llm.service`
cgroup **and** `pid == main_pid` — the unit's `MainPID`, which the **caller** supplies from
`unit_runtime_state` (`procfs` never calls `systemctl`) — with the `MainPID`'s start time compared to rule
out reuse; else `"spawned"`. `UnitStrategy.stop(spec)`: **own-unit re-check first**; `"unit"` →
`reset-failed`, `systemctl --user stop` (a **foreign** unit's process is named, never stopped); `"spawned"`
→ the wrapped strategy's `stop`. `free_accelerator`, `_safe_stop` and the restore are unchanged beyond
`unit=`. So the first `/model` after `--autostart on` frees the server the harness kept running.

The liveness test behind "nothing answers": `read_launch(expect=strategy.launch_identity(spec), main_pid=)`.
A verified live launch of **this record** → "still loading — waiting" (`wait_ready(expect=)`). Of
**another** record answering nowhere → the swap stops it. A process whose argv is `-I -m localharness
serve`, in the unit's cgroup **and** the unit's `MainPID` → "starting": waited up to `TimeoutStartSec`,
then named, never signalled. A foreign process → the incumbent sentence. `--model` keeps its meaning.
`start --scripts` is a hidden alias of `--preset`.

### 8.3 The pidfile knows its port, and its process is proven before it is signalled

`<pid> <start> <port>`; `_parse_pidfile` accepts one, two or three tokens. **The `/proc` side lives in
`provider/procfs.py`** (`start_time_from_stat` moves in; `ProcRecord` gains `sid`; `server` imports `procfs`
one way). `LaunchIdentity(exe_names, port, argv_contains, port_env)` is plain data from
`strategy.launch_identity(spec)`: llama.cpp `({binary basename, its target's, "llama-server"}, port, (), "LLAMA_ARG_PORT")`; vLLM binary
`({"vllm", "python…"}, port, ("serve",), None)`; docker `({"docker"}, None, ("--name", container), None)`;
Ollama `({"ollama"}, port, (), "OLLAMA_HOST")`; LM Studio `None`. One pure `procfs.identity_mismatch(record,
expect) -> str | None`: uid, `exe` basename in `exe_names`, every `argv_contains` token, the **record's**
port where the identity says, `sid == pid`, and for `owner == "unit"` the cgroup and `MainPID` tests.

`procfs.read_launch(config_dir, *, expect=None, main_pid=None) -> Launch(pid, start, port, owner, verified)
| None`: reads the file, applies `_same_process`, **never unlinks on a mismatch**, treats a port outside
`1–65535` as absent, verifies when `expect` is given, takes a legacy file's port from the verified process.
**`expect` is required on every signalling path and supplied inside the strategies**: every strategy `stop`
and the stops inside `activate` (`lifecycle.py:361, 373, 395`) call `stop_server(config_dir, launch,
expect=self.launch_identity(spec))`; a structural test pins that `stop_server(` is never called without
`expect`. A pid that fails is `pidfile names pid 4242 (<exe>), which is not a harness launch — not
signalled; stop it, or remove <pidfile>`. **Residual (`SECURITY.md`):** a `setsid` server of the person's
own on the record's port with the record's binary, named by a planted pidfile, passes every test and would
be stopped by an explicit swap; a wrapper-script binary whose process execs `llama-server` passes the `exe`
test, one that execs a differently named server fails it and is named.

**The harness's own small files** — the pidfile, `unit-target`, the boot marker, the unit — are written
through **`atomic_write_text`** (moved from `config/overlay.py:137` into `core/private_files.py`, re-exported
by `overlay`), which gains an optional `dir_fd` so the server directory is opened `O_DIRECTORY|O_NOFOLLOW`
once. Reads go through `open_regular` under a cap and a name charset.

### 8.4 Effort

Unchanged: the rung resolves from `active_endpoint.effort` for a resumed peer; the `start_cmd.py:2032`
comment is rewritten.

## 9. Autostart

### 9.1 Surface and scope

`localharness model --autostart on | off | status`. `on`/`off` need a local operator; otherwise `autostart
is set from a terminal: localharness model --autostart on`, exit 2. **Scope:** `llamacpp` and vLLM `launch:
binary`; another kind: `autostart serves llama.cpp and a vLLM binary; <name> is docker — localharness model
<a llama.cpp preset> first`. Under the unit, a docker/Ollama/LM Studio `/model` runs outside it (§8.2); at
the next boot `serve` finds no `unit-target` and an `active_endpoint` the unit cannot serve, exits **0** with
`<name> is docker — not served by the unit`; `status` translates it. After its local-operator gate, `on`
runs `decide_machine_trust(ask=True)` — true here because the gate has just proven a terminal, the one
place under `cli/` where the literal is allowed by the structural test's single exemption; the one ruled
question; on the common path already answered.

### 9.2 The unit

```ini
# written by localharness 0.19.0 — `localharness model --autostart off` removes it
# env PATH=/usr/local/cuda/bin:/home/awurm/.local/bin:/usr/local/bin:/usr/bin:/bin
# env HF_HOME=/home/awurm/models/hf
[Unit]
Description=LocalHarness model server (serves the preset `localharness model` selected)
After=network.target
StartLimitIntervalSec=5400
StartLimitBurst=3

[Service]
WorkingDirectory=/home/awurm
ExecStart=/home/awurm/.local/share/uv/tools/localharness/bin/python -I -m localharness serve --config-dir /home/awurm/.localharness
UnsetEnvironment=LOCALHARNESS_DIR LOCALHARNESS_HOME LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH GCONV_PATH PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
Restart=on-failure
RestartSec=5
RestartPreventExitStatus=78
TimeoutStartSec=1800
KillSignal=SIGINT

[Install]
WantedBy=default.target
```

`ExecStart` is the interpreter the harness runs under (`-I`) and the module (`src/localharness/__main__.py`,
new), the global config dir pinned; `WorkingDirectory=` the absolute home. **The shell's `PATH` (always)
and, when set in that shell, `HF_HOME` and `LD_LIBRARY_PATH` are captured at `on` as `# env NAME=value`
comment lines** in the unit body — comments because `UnsetEnvironment=` is applied last and would undo an
`Environment=` line for the same name (§2.2), and because the interpreter itself must start under the
scrubbed environment; systemd ignores comments silently; the lines are part of the **byte-equal body**, so
`serve` reads them from the unit file it has just verified (`units.captured_env()`, the inverse of
`render_harness_unit`), never from `os.environ`. **The captured values are anchored in the machine
record, and the snapshot reads them from the unit file.** Byte-equality alone is circular for them (`body
== render(argv, dir, captured_env(body))` holds for any well-formed line), and an entry kept only in the
record would not survive: `decide_machine_trust` always works from `machine_snapshot`, which it re-records
on the no-question path and on every Yes (`workspace.py:870–899`), so anything the snapshot does not
produce reads as a removal and is dropped — by the next `start`, and by `serve`'s own decision before it
could compare. So **`machine_snapshot` emits the entry**: when `harness_unit_path()` (honouring
`XDG_CONFIG_HOME`, so tests can redirect it) exists **and its `ExecStart` names this `global_dir` as
`--config-dir`** (one unit per user, possibly several config dirs; a dir the unit does not serve is never
asked about it), it reads the file's `# env` block through `captured_env` and emits one **`launch`**-kind
entry — `{file: "systemd/localharness-llm.service", kind: "launch", name: "autostart.env", shown:
<compact JSON of the triple>}` — through the existing `(prefix, mapping)` helper (`_launch_entries(file,
"autostart.", {"env": captured})`; `env` is already in `LAUNCH_KEYS`), **never a new kind**, which the
"record predates the kind" rule (`workspace.py:857`) would adopt silently once and so accept a unit planted
before the first 0.19 start. The pseudo-`file` is a pairing key only — no loader consumer matches it
(`_apply_launch_withholds` walks `config.yaml` and `overrides.yaml`, `_peer_withholds` matches
`extra_endpoints.` names), so nothing is dropped from the config on its account — and the machine question
and the withheld line show the unit's **real path** through `machine_diff`'s `annotations`, not a path that
reads as living under the config dir. From there everything is spec 16's: `on` adopts the entry it just wrote (as a terminal `/model`
adopts its own write); a changed `# env` line is an ordinary uncovered entry — asked about at the next
terminal `start`, withheld under `serve`; `serve` **exits 78 when `autostart.env` is in `trust.withheld`,
and also when the unit file exists but the entry is missing or unparsable** — never "nothing to check";
`off` removes the file and the entry is pruned as an ordinary removal. `captured_env` parses only `# env`
lines between the header line and `[Unit]`, accepts the closed name set `PATH`, `HF_HOME`,
`LD_LIBRARY_PATH` with no duplicates, and both `on` and `serve` apply the charset rule and the
**absolute-entry rule to every element of `PATH` and `LD_LIBRARY_PATH` and to `HF_HOME`** (a relative or
empty element would resolve against `WorkingDirectory=`, the user's own home): a failing `PATH` or
`LD_LIBRARY_PATH` element is **dropped at `on` and named with its variable in the receipt** (`…; PATH
captured from this shell, dropped relative: PATH ./bin, LD_LIBRARY_PATH ./lib`); a relative `HF_HOME` is
one whole value the shell meant relative to its working directory, so `on` **makes it absolute against the
shell's cwd** and names that in the receipt instead of dropping it; anything that reaches `serve` non-absolute
is a 78. `StartLimitBurst=3` over `5400 s` bounds an unattended
load loop; **every `restart`/`stop` the harness issues is preceded by `reset-failed`** (after the own-unit
re-check), so a person's switches never count toward it. Every path and value on the `ExecStart`/
`WorkingDirectory`/`# env` lines matches `[A-Za-z0-9/._+@:=-]+` or `on` refuses. The file is written through
`atomic_write_text`; the receipt is one line and names the path and the captured `PATH`:

```
autostart on: ~/.config/systemd/user/localharness-llm.service serves qwen-27b-swift-1.5 at login and boot (linger on; PATH captured from this shell)
```

or `… at login (at boot too after: loginctl enable-linger awurm; …)`. Then `daemon-reload` and `enable`
(`--now` only when nothing of the harness's serves), every `systemctl`/`journalctl` through
`units.systemctl_path()` (`/usr/bin`, `/bin` only). **A server the harness already runs is kept** (`enable`
without `--now`, `— takes over at the next login or /model`). **A second launcher** (a unit on a preset's
port, `enabled` or `active`) is named once at `on` and in `status`. `off`: `disable --now`, remove the file
only when it is the harness's, `daemon-reload`. `status`: `UnitFileState`, `ActiveState`, `NRestarts`,
`MainPID`, the ladder's target, linger, the captured `PATH`, the second-launcher line, the marker (`serving X
at the retry shape since boot — mtp off`), and the translated exit (`78` → `stopped: the machine record
withholds <record> — run localharness start in a terminal`; start limit → `stopped after 3 failed loads —
journalctl --user -u localharness-llm.service`; non-binary target → `<name> is docker — not served by the
unit`). `doctor`'s `Autostart:` row says the same.

**"The harness's own unit"**: `FragmentPath == harness_unit_path()`, **empty `DropInPaths`**, and the file
byte-equal to `render_harness_unit` for this interpreter, config dir and captured values. Anything else is
foreign: `off` leaves it, `UnitStrategy` is not chosen, `status` names the failed test. **Re-run before
every `restart` and `stop`.**

### 9.3 `serve`, the unit's body

`localharness serve` (hidden top-level command). In order: `os.environ["LOCALHARNESS_DIR"] = <pinned>`;
the loader with `local_config_dir=None`; **`recorded_machine(dir) is None → 78`**;
`decide_machine_trust(ask=False)` and the harness loaded with `machine_withheld`; `resolve_serving(harness,
trust, None)` — withheld → **78**, a target the unit cannot serve → **0** with the line; the §8.2 liveness
test — a verified live launch of this record → `already serving (pid 4242) — not started twice`, **exit
0**; the own-unit identity test on its own unit file and `captured_env()` from it; `sense_launch(unit=)` →
`plan_launch`; `prepare_config_ini_home`; under the launch lock — **the same `flock` on `server.lock_path`
with the same `_CONTENDED` marker, taken synchronously** (one lock, two entry points) — the pidfile and the
**boot marker**; lock released, fd `CLOEXEC`; **`os.execve`** with an environment built as
**`child_environment(spec.runtime, base, …)` over an allowlisted base** — `HOME` set from the pinned home,
`PATH`/`HF_HOME`/`LD_LIBRARY_PATH` **from `captured_env()` only** (checked against the recorded
`autostart.env` entry, §9.2), and from the inherited environment exactly the names `USER LOGNAME LANG
LC_ALL LC_CTYPE LC_MESSAGES XDG_RUNTIME_DIR CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER NVIDIA_VISIBLE_DEVICES
NVIDIA_DRIVER_CAPABILITIES` — named one by one, **no wildcard** (`CUDA_*` would admit
`CUDA_INJECTION64_PATH`) — so spec 16's blocklist contract (`server.py:748`) is kept over a base the manager
cannot widen, and the record's `env:`, `XDG_CONFIG_HOME` and the key are added as today. `child_environment`
returns `None` when it changed nothing (`server.py:755–759`, an unkeyed vLLM binary with no `env:`): `serve`
execs **`env if env is not None else dict(base)`**, never `os.environ` (a test pins it).

**The boot marker** `<server dir>/boot-attempt` = `{record: <record key>, binary, invocation:
$INVOCATION_ID, cause: death | remembered | none, shape: full | retry, planned_context, full_fit,
free_bytes, at}`. `Sensed.prior_death` is true iff a marker exists **for this record and binary**, its
`cause` is `death` or it was written with `shape: full`, it is **younger than `TimeoutStartSec`**, and the
unit's `NRestarts ≥ 1`; `plan_launch` then returns the retry shape and `serve` writes the new marker with
`cause: death`. A marker for another record or binary, or older than the load window, is removed and
ignored — so a marker left behind when nobody runs `start` expires, and a server that died hours after a
good load does not come back degraded. `invocation` binds the marker to the run for `confirm_launch`
(§6.0); it plays no part in `prior_death`, since a `Restart=` is a new invocation.

**The marker closes the loop through `confirm_launch`** (§6.0): the next `start` that attaches to the
unit's server, or `UnitStrategy`'s wait, confirms a marker whose `invocation` is the live one — `cause:
death` with the live argv's `-c` equal to `planned_context` → the ledger `retry` record (clamped marker fit,
`dropped` from the live argv) and the one-time clause; `cause: remembered` → removed silently (the §6.4
on-change clause covers it); any other outcome → removed, nothing written. A person who never runs `start`
sees the marker in `status`/`doctor`.

### 9.4 While the harness's unit exists, the harness launches through it

`start` senses the unit once per process; `strategy_for(spec, unit=state)` returns a `UnitStrategy` wrapping
the spawned strategy when `state` is the harness's own unit **and** `spec` is llamacpp or a vLLM binary,
else today's strategy; a **structural test pins every `strategy_for(` call under `src/` to carry `unit=`**.
`UnitStrategy.activate(spec)`: `units.write_target` (record names only; a typed `.gguf` restart refused:
`under autostart /model takes preset names — put <path> under autosense.model_roots and run localharness
init to add it as a peer`); own-unit re-check; `reset-failed`; `restart`; then **the same `wait_ready(expect=…)`**
with a `liveness` callback reading `unit_runtime_state` each poll (`ActiveState, NRestarts, ExecMainStatus,
MainPID, InvocationID`) — **tolerating one rise of `NRestarts` when the plan carries derived keywords** (the
unit's own retry) and stopping at the second, or at `ActiveState=failed`, with the journal's last lines
**raw in the `RuntimeError`**. On success it calls `confirm_launch` from the marker. Tails are redacted **at
the channel boundary** — the REPL's `SwapReporter.line`, the CLI's, `start`'s console — with
`strip_url_credentials` and `redact.scrub(text, secrets, *, patterns=)`: the secret set is
`secret_values(HarnessConfig, merged)` **plus the key read from any `api_key_file` the session holds**, and
`patterns` masks the two forms llama-server logs (`Bearer \S+`, its `--verbose` key-suffix line). On a
failed swap the restore rewrites `unit-target` to the incumbent. `stop`: §8.2's owner rule. `liveness`: the
pidfile. `/model <peer>` under autostart is: write the target, restart, wait — the owner's `swap-llm.sh`,
harness-owned.

## 10. The script

`localharness model --script [NAME]` renders a `LaunchPlan` as POSIX `sh`:

```sh
# qwen-27b-swift-1.5 — the launch `localharness start` / `/model` runs (0.19.0); derived: kv q8_0 (autosense.kv_cache_type) · draft-mtp (nextn_predict_layers=1)
# -c left to llama-server --fit (the harness fitted 114688 today); environment: LLAMA_ARG_*/LLAMA_API_KEY/MTMD_BACKEND_DEVICE dropped, XDG_CONFIG_HOME → ~/.localharness/config-ini/8080
exec /home/awurm/src/llama.cpp/build/bin/llama-server \
  -m /home/awurm/models/gguf/Swift-1.5-Qwen3.8-27B-NVFP4-Q8mix.gguf \
  --mmproj /home/awurm/models/gguf/mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf \
  -a qwen-27b-swift-1.5 \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 19660 \
  --spec-type draft-mtp --spec-draft-p-min 0.6 \
  --port 8080 --host 127.0.0.1
```

Every token `visible_text(shlex.quote(tok))`, comments through `visible_text`; no key (inline →
`# LLAMA_API_KEY=<provider.api_key>`, `extra_args --api-key` → `<key>`, a key file → its path); `env:` as
`NAME=value`; `--all` emits one shell function per preset; a vLLM record prints `vllm serve …`. Never read
back; nothing written.

## 11. `init` and the first run

The job (§1 item 5, G7): `pip install localharness`, then `localharness start`, and the box serves a model.
A first `start` with no config runs `init`'s core setup on a terminal and goes on into the session
(`start_cmd.py`, the first-start branch; owner ruling 2026-10-03). Spec 16 §8's rule stands — **every record
`init` writes was shown, and an interactive `init` adopts exactly what it showed** — and one ruling of spec
16 §12, recorded there as reversible by the owner, is revised: on a terminal the setup asks **one** question,
through the interview (§11.5), whose default is the lazy answer. Without a person — no tty, `--no-input`,
`LOCALHARNESS_COMMIT` — zero asks stays the rule: a first run writes its records unadopted as today (spec 16
§8), an existing-config run appends nothing, and `start` senses no files at all (§11.3). The workspace-trust
question (`settle_startup_trust`, spec 16) is outside this count.

**One predicate for "a person is here":** `workspace.local_operator(no_input) -> bool` — a tty on stdin **and
on stdout** (a question rendered into a pipe has no visible default), no `LOCALHARNESS_COMMIT`, not
`--no-input`; `model_cmd._is_local_operator`, `init_cmd.is_interactive` and `init_cmd._adopts` fold into it.
`init` and the first-start branch compute it once at the CLI boundary and hand it down as
`core_setup(interactive=)` and `sync(person=)`; a structural test pins that every `sync(` call under `src/`
passes a value that came from `local_operator(`. The interview (§11.5) calls the same predicate itself and
fails closed without it — the component is the last line, the callers the first.

### 11.0 One owner, two modules: the model library

- **`provider/model_library.py`** is pure and imports nothing from `cli/` (no `provider/` module does today;
  it imports `agent/gguf_header`, which itself imports only `core/`). `scan(roots, recorded, library_path,
  *, scope, rank) -> Found`: `walk_ggufs` over the given roots (bounded by `DISCOVERY_MAX_*`,
  symlinked directories never descended, a missing root skipped), deduplicated by realpath, set-differenced
  against `recorded` (the realpath of every `model` and `mmproj` of `server:` and every peer of the merged
  config, passed in by the caller) and against the library file; only a file in neither is header-typed
  (`discover`, whose return type changes to one `FileKind` per path — `model`, `projector`, `shard`, `other`,
  `unreadable` — the enum defined beside it in `agent/gguf_header.py`; not `Verdict`, which `gate_types`
  already owns) and ranked by `rank`, a callable `cli/library` builds from `rank_models` with `unit_model`,
  `ledger_paths` (`speed_stats.model_paths`) and `fits` (`compute_fit` over `_machine_autosense`, memory
  probed once and only when some file is new) — spec 16 §8's order. `Found` carries the new models, the seen-unrecorded ones and the declined ones with their facts;
  it prints nothing. `peers_for(models, *, start, taken_names, taken_ports)` and `free_ports` are
  `_peer_records` and `_free_ports` moved here with public names. The library file (`Library.load` /
  `Library.save`) lives here too.
- **The library file**, `<global>/model-library.json`: private; read through `open_regular` under a cap (as
  `speed_stats` reads the ledger), every field validated (an ISO date, an int size, the `FileKind` enum, at most
  `DISCOVERY_MAX_FILES` entries) and every printed value through `visible_text`; written only by
  `provider/model_library` through `atomic_write_text` under the overlay lock (§11.4); a
  `PROTECTED_CONFIG_DIR_ENTRIES` member. Per realpath: `first_seen`, `size`, the verdict, `declined`. Its one
  meaning: **new is first sighting** — a file absent from the file and from `recorded` is new whatever its
  mtime (a `cp -p`, `mv` or `rsync` keeps an old one; mtime is also the agent's to set), and a file whose
  verdict **becomes** `model` (re-typed after a size change, or `unreadable` the last time) is a first
  sighting as a model, its `first_seen` reset. `other`, `projector` and `shard` are never re-read;
  `unreadable` is re-tried; a resized file is re-typed. **Baseline:** on a run that is not a first run
  (§11.1), when the library file is missing or corrupt, the first walk records every file `seen` and asks
  nothing (an upgrade, an attach-only install, a deleted file); a removed single entry makes that file new
  again, and so does a removed peer (the person removed its record; one question settles it). The first-run
  `sync` writes the file (the shown files and the declines) before it hands back its records. The declined-
  offer store of spec 16 is not used for files. The record of a shown file is always built from a fresh
  header read at show time, never from the stored verdict.
- **`cli/library.py`** is the side that talks and writes. `InitRecords`, `preset_lines`, `show_records`,
  `_machine_autosense`, `_adopt_written`, `_record_dicts_of` and `_result_from_record` move here from
  `init_cmd` (which imports them back; `library` never imports `init_cmd`). It owns `sync(config_path, *,
  scope, person, preset=None) -> Sync(records, lines, appended, declined)`: it gathers roots and `recorded`,
  calls `scan`, procs the interview (§11.1), writes, adopts, prints the receipt line. On a first run
  `records` is what `core_setup` hands to `_write_records`, whose order changes: `_write_harness` writes
  `config.yaml` with `extra_endpoints: []`, then `append_peers` writes the peers into `overrides.yaml`
  (after `--force`'s reset of the saved choices, which now also clears `extra_endpoints` —
  `_SAVED_CHOICE_KEYS` — so "both files or neither" holds and a first run's `recorded` is honestly empty),
  then two adoptions: `server.` under `config.yaml`, the peers under `overrides.yaml` (trust entries are
  keyed by file). If the peer write returns False, `config.yaml` stands with its `server:`, one line says the
  peers were not written, and the files stay in the library as `seen`, so `localharness model --setup`
  (§11.1) lists them. On every other run with a person `sync` appended its peers itself and adopted exactly
  the lines it showed. **Peers always live in `overrides.yaml`.**
  `append_peers(config_path, peers)` appends to that file's `extra_endpoints` list through
  `write_overlay_validated` (the one validated overlay writer, now under the overlay lock, the
  set-difference re-checked inside `mutate`), names unique over every record (`unique_preset_name`), ports
  over every recorded port; `server:` is never touched here. A 0.18 `config.yaml` that still holds the
  list is the one-time **copy** the posture flow already performs (`cli/workspace.py`: the list is copied
  whole into `overrides.yaml` — one key, and an `overrides.yaml` list replaces — `config.yaml`'s own list
  left in place and shadowed, `PEER_LIST_COPIED` printed): the first append reuses that copy, and
  `_copied_peer_entries` re-adopts a copied peer **only where its `config.yaml` twin was already covered**
  — a peer edited since its adoption is copied unadopted and the next start asks about it; with no machine
  record nothing is adopted; the line carries both counts. There is no second copier. `append_peers` is
  built as a `mutate` fragment over the overlay, so the import's `_write_overlay_records` composes its
  `server:` write and the fragment in its one atomic write with one audit (its "only while empty" rule
  goes; this is the one peer writer); `record_path(path)` is `scan` for one file plus `append_peers`;
  `counts(config_path)` serves `doctor`. `_discover_presets` is deleted; `_records_from_import` takes
  `peers_for`; `_record_dicts` stays a thin caller of the moved `_record_dicts_of`; `_guided_setup` is
  reduced to the vLLM install step, which runs without its own `Confirm` when the interview's option carried
  the consent (§11.2). Nothing else
  walks a root, types a header for a record, asks about a file or writes a peer.
- **Callers.** `init`: `sync(scope="all")` in `_pick_server` where `_discover_presets` ran (a **first run**
  — no `config.yaml`, or `--force`, under which `recorded` is empty and the baseline is off — after
  `_unit_import`, which wins silently as spec 16 §8 says: the unit import is authoritative, the one
  intentional silent path), **and** inside `_existing_config` after the config has loaded (an unreadable
  config exits 1 first, as today), not when `changing` (`--endpoint` / `--model` ask nothing); when `sync`
  appended, `Keep it?` is not asked and the line reads `Added N; kept config.yaml.`. `start`:
  `sync(scope="new")` once per process (§11.3). `model --download`: `record_path` (§11.2). `doctor`:
  `counts`. A first run that **attached** to a running server (`Attached`, spec 16 §8) baselines and asks
  nothing.

### 11.1 Models found: one question

The shown block is a bulleted list (the interview's options are the numbered things), the ranked best
first, one line per unrecorded model: `• <name> (llama.cpp, mtp, +<projector>) ← <shown_path> · <size> ·
first seen <date>`, a declined one tagged
`(declined earlier)`, a vLLM checkpoint as a peer line **only when a `vllm` binary is found** (`· binary:
<shown_path> (on PATH | harness venv)`), otherwise nothing; `scope="all"` lists every unrecorded model,
`scope="new"` only the first sightings. There is no second block: after the choice the primary's line
gains its binary clause and the list is the record that is written and adopted. Then **one interview
question** (§11.5), its default the lazy answer:

- **First run, N ≥ 2** — `Set up these N models?` with the options `All, serving <best> first` (default; its
  description: `every file above becomes a preset; <best> serves on this start`), `Choose which, and which
  first` (`pick the files, then the one to serve`), `Skip for now` (`nothing is set up; run localharness
  start again later`). The second option opens two interview steps, a multi-select `Set up which?` (all
  pre-selected) and, only when two or more stay selected, a single-select `Serve which first?` (the best
  pre-selected). A file is **declined** only by being unselected there; Enter with none selected declines
  them all (the everything-declined line); the default declines nothing. **N = 1** — `Set up <name>?` with
  `Set it up` (default) and `Not now`. With `start --preset <name>` or `--scripts` naming a found file, that
  file serves first and no order step runs. **Cancel** (Esc, Ctrl-C) at any step of a first run writes
  nothing, declines nothing and prints the `Skip for now` line; it is never silent. `server:` is written only on a first run; every other run
  appends peers. Everything declined on a first run → one line naming the library and `run localharness
  start again`, exit 0, nothing written (`_skip_for_now` stays the path of `init --endpoint`).
- **A later run** — on `start` (first sightings only): `New model <name> — set it up?` with `Set it up`
  (default), `Not now`, and for N ≥ 2 `Choose which`; `Not now` and a cancel leave them `seen` (`start`
  stops asking). On `init` (`scope="all"`) the question comes **only when there is a first sighting**, with
  the same three options; its default sets up the first sightings alone — never a file the baseline or a
  `Not now` kept quiet, and never a declined one. **The way back is `localharness model --setup`**: the
  multi-select `Set up which?` over every unrecorded file on the box — first sightings pre-selected,
  `seen` ones not, declined ones tagged and not — and selecting a declined file clears its decline. Every
  answer prints one line saying what happened and the way back: after `Not now` or an unselected file,
  `… — localharness model --setup lists them`; after a set-up peer, `/model <name> switches in a session;
  localharness start --preset <name> starts on it`. `doctor`'s row (§11.4) names the same command.
- **The receipt**, first run only: the list above as the preset block, then **one line**: `Later: just run
  localharness start — it brings back your last choice.`, with ` /model <name> switches in a session.` only
  when there is a peer; from `init` it replaces `Run 'localharness start' to begin.`. No other verb is
  taught here. The README's Quick start and `no_config_message` teach `localharness start` as the first
  command and say a model server is not required (§13).

### 11.2 Nothing found: the download offer

When no server answers and no model file is found, on a terminal, the old `Set up vLLM and a model now?`
menu is replaced by one interview question:

- **The table is `provider/refarch.py`**, one row per reference architecture of
  `docs/reference-architectures/`: `runtime`, `platforms` (replaces `platform`), `needs` (`nvidia`,
  `apple-silicon`, or none), `repo` (replaces `default_model`), the **one file** (`None` for a vLLM
  checkpoint, fetched whole as today), `bytes`, `revision` (a 40-hex commit), `sha256` (both from the hub's
  LFS metadata at that commit; a test pins every row's shapes), `page`, status, and the route fields the row
  needs (`docker_image`, `pip_package`, optional). New: the 8 GB consumer-GPU row (llama.cpp, `Q4_K_M`, one
  file, linux and darwin, no `needs`). The Spark's llama.cpp route is **not** offered — its doc needs a
  second file, the MTP head, which `discover` does not pair — so the Spark keeps its vLLM row (`needs:
  nvidia`); the Mac mini row keeps vLLM-metal (`needs: apple-silicon`). Kept in sync with the docs as the
  module's docstring already requires.
- **The default is sensed, by one ordered rule**: this platform's rows whose `needs` the box meets (an
  answering `nvidia-smi`; darwin on arm64); among them, a row whose runtime is on `PATH` (`llama-server`,
  `vllm`, `which_outside_cwd`); then, with no runtime present, the llama.cpp row (its install is one line,
  §11.5's install line); then the largest whose `bytes` fit the sensed memory less the `autosense`
  reserve — the GPU's free bytes from `free_memory`, else the box's RAM (`_read_meminfo` on Linux,
  `os.sysconf` elsewhere; no new subprocess); memory unknown → the smallest, and the line says `needs about
  <size>` instead of `fits in your <N> GB`. The question: `Download <file> (<size>, fits in your <N> GB)?` —
  for the small row `a small model that runs anywhere; larger ones under Show the other rows` — with the
  options `Download it` (default; on a vLLM row its description also carries the one consent that row
  needs, `and install vllm into <venv>` or `and pull <image>`, so the old model prompt, download confirm and
  venv `Confirm` are not asked again), `Show the other rows` (the rows with their pages, then the same
  question), `Not now`. **No runtime** is not a dead end: the file is runtime-independent, so the question
  still comes, with the platform's install line printed above it, again after the download ending in `then run
  localharness start`, and on the next `start`'s `no llama-server — set server.binary` line (spec 16 §8).
- `Download it` fetches the one file (a vLLM row: the whole repo through `download_model`, as today, with no
  further prompt) through `download_file(repo, file, *, revision)` into the HF cache,
  then verifies the returned file **through one descriptor** opened with `open_regular`: the bytes counted
  and hashed against `bytes` and `sha256`, the header read from the same descriptor — a mismatch is named
  and nothing is recorded (the HF cache is a model root the agent can write under, and `hf_hub_download`
  returns a cached path for a pinned commit without a network round trip, so the hash is what makes the
  offer's claim true). Then `record_path` builds the record from **that path alone**, prints its record
  line (name, runtime, binary, port, path), writes and adopts exactly that line, with no second question.
  `Not now` prints the page and `localharness model --download <repo> --file <file>` and §11.1's run-again
  line, exit 0. The vLLM rows keep their install prompt and their whole-repo, unpinned `download_model`
  (as today; §12 says what that leaves open). Nothing is launched by `init`; the first `start` launches
  after its own trust decision (spec 16 §8).
- **`model --download`** ends by calling `record_path` on its returned path when that is one `.gguf` file
  (never a snapshot directory), a `config.yaml` exists, a person is present, **and the file verifies** —
  resolved once: `HfApi.model_info(repo, revision=<ref>).sha` gives the 40-hex commit, which is passed to
  both the download and `get_paths_info`, whose `lfs.sha256` is hashed against the returned file through one
  descriptor as above (never the sha in the returned path, which sits in the agent-writable cache); the
  commit and hash are printed with the record line. Otherwise —
  no config yet, no person, no hub metadata, a mismatch — it writes nothing and says `start finds it`; the
  file then surfaces as a first sighting at the next terminal `start`, shown as any other.

### 11.3 A model that arrives later

A terminal `start` calls `sync(scope="new")` **once per process**, in the sync body beside the first-start
branch (not inside `_start_async`, which runs under `asyncio.run`; not on the `/plugins` restart pass; not
after a first-start setup that just ran `sync`; not with `--list-models`), for the terminal channel only,
with a person, **before** `decide_machine_trust` so an adopted append appears there as covered, and never
when the config dir cannot be written. The walk is a bounded directory listing with no header read for a
known path; silent when there is no first sighting; one question (§11.1) when there is, a cancel counting as `Not
now`. Without a person `start` walks nothing and writes nothing. `localharness model --setup` is the same
`sync` with the multi-select forced (§11.1), for a person only.

### 11.4 `doctor`, and the overlay lock

One `Models:` row from `counts`: `N set up · M found, not set up · K declined — localharness model --setup
lists them`, the zero clauses dropped — the one place a person sees what the box holds that the harness is not
serving. **The overlay lock** is new: `<global>/overrides.yaml.lock`, a sidecar `flock` acquired as
`project_records.yaml.lock` is (`O_NOFOLLOW | O_NONBLOCK`, mode 0600, `LOCK_NB` polled for a bounded budget,
then fail closed: `write_overlay_validated` returns False and `sync` prints `overrides.yaml is busy — nothing
written`), held across the raw read, `mutate` and the write and released before the audit publish, taken
around the library file's write as well, never nested (the two writes in `sync` are sequential, pinned), a
`PROTECTED_CONFIG_DIR_ENTRIES` member beside `project_records.yaml.lock` (SECURITY.md's count and the gate
test follow); until now the overlay write was an unlocked read-modify-write.

### 11.5 The interview

Every question of §11 — and only these, for now — is asked through **one component**, `cli/interview.py`,
the harness's second prompt_toolkit app beside `cli/picker.py` (a different widget: `picker` is one
horizontal line for `/effort`; this is vertical, with descriptions and `multi`; it shares `PICKER_STYLE` and
the picker's `input` / `output` test seams; prompt_toolkit is already a dependency, no new one).
`ask(question, options, *, default, cancel, multi=False) -> Answer`, with `Option(key, label, hint)` frozen
and `Answer(keys, cancelled)`: the question on one line, each option on its own line with its label and
hint, the default highlighted (`default` an index, or a set of keys when `multi`); the arrow keys move the
highlight and a typed number jumps to that option (toggles it when `multi`; a two-digit buffer, the list
scrolling past the terminal's height); Enter takes the highlight (when `multi`, the selected set, which may
be empty); Space toggles when `multi`; Esc and Ctrl-C return `cancelled` with `cancel` as the key, and the
caller treats a cancel as its no-change answer and prints its line, never silently. A dim hint line is part
of the widget (`↑↓ move · Space select · Enter confirm` when `multi`). The app is inline, keeps the chosen
line in scrollback (`erase_when_done=False`) and lets hints wrap on a narrow terminal. It calls
`local_operator` itself and fails closed (`cancelled`, nothing read) without a tty on both ends or under the
marker — a prompt_toolkit input reads a pipe if asked, so the gate is inside the component as well as in
its callers; and when the terminal cannot draw it (`TERM=dumb`, too small), the caller takes the non-person
path (records unadopted, the next-`start` line) rather than raising. The caller states only the question,
the options with what each does, the default and the cancel; the keys and the look are the component's,
written once and tested once, so no question in this spec carries a key grammar. The `rich` prompts elsewhere are untouched by this spec and named as debt for a later decision: the
`Confirm`, `IntPrompt` and `Prompt` sites in `init_cmd` (`Keep it?`, `_skip_for_now`, `_select_model`, the
vLLM install step's remaining asks) and `_confirm_on_a_tty` — so one `init` can still show two prompt
grammars. The install line §11.2 prints is one constant per
platform in `cli/library.py`: darwin `brew install llama.cpp`; linux `a llama.cpp release build for your
GPU from github.com/ggml-org/llama.cpp/releases, llama-server on PATH — docs/runtimes/llamacpp.md`.

## 12. Security

- **Machine-only.** Knobs are `autosense` fields; `pending_preset` joins `HARNESS_GLOBAL_ONLY_FIELDS`; the
  three record fields join `LAUNCH_MODEL_KEYS`. A derived flag, the header, `autosense.*`, the ledger,
  `unit-target` and the boot marker are **not** fingerprinted: each selects among closed values or among
  fingerprinted, non-withheld records; the marker is bound to a record, a binary and a run, its `dropped`
  never reaches the ledger (the live argv does), and its fit values are clamped before they move a threshold.
- **Every launching surface runs the same trust decision, and puts its question only to a local
  operator**: `start` (`ask=can_ask`, as today), `serve` (`ask=False`, never adopting, exit 78), the CLI
  (`ask=_is_local_operator()`; `--autostart on` after its own gate), each loading with `machine_withheld`;
  structural tests pin that no swap runs without a `MachineTrust` and that no `decide_machine_trust(ask=True)`
  literal exists under `cli/` beyond the gated `on`. The CLI's immediate swap runs without a question when
  a machine record exists, `LOCALHARNESS_COMMIT` is absent and the target is not withheld — a selection
  among fingerprinted, confirmed records, as at boot; the marker is a default, not a boundary (the agent
  can clear it, and gains nothing it does not have through the allowed `systemctl --user restart` and a
  `pending_preset` write). The
  captured unit environment is a fingerprinted `launch` entry (`autostart.env`) **emitted by
  `machine_snapshot` from the unit file**, so it survives every re-record and a tampered `# env` line is
  withheld like any other machine-file edit (missing or unparsable → 78). What the anchor protects against
  is a writer that can change the unit file but not the trust store (a write the path gate misses, a
  shell-side edit, a restored copy); it does not protect against `python_exec`, which runs in-process and
  can rewrite the store — spec 16's own residual, no wider here.
- **The unit's environment and config dir are pinned**: `-I`, `--config-dir`, `LOCALHARNESS_DIR` set first,
  `WorkingDirectory=`, `UnsetEnvironment=` (applied last, so the interpreter starts scrubbed), the captured
  values read from the verified unit body, and `child_environment` over an allowlisted base. Named
  residuals: the D-Bus route to the user manager (why the unit relies on its byte-equal body and the
  allowlisted base, not on a closed directory); the harness's own install tree and the captured `PATH`'s
  user-writable entries run at boot exactly as a spawned launch runs them from the shell.
- **A unit is run or stopped only when it is provably the harness's**, re-checked before every
  `restart`/`stop` through a system-directory `systemctl`; a foreign unit's process is named, never stopped.
- **Protected paths and deny entries.** `~/.config/systemd`, `~/.local/share/systemd/user`,
  `~/.config/environment.d`, `~/.config/autostart` join `PROTECTED_PATHS_HOME_DEFAULT`. The deny bundle
  (owner to confirm, §15): `systemctl --user set-environment|import-environment|unset-environment`, `link`,
  `edit`, `set-property`, `add-wants|add-requires`, `enable` with a `/`-bearing argument, `busctl --user` /
  `gdbus call --session` / `dbus-send --session` aimed at `org.freedesktop.systemd1`, and `localharness
  model --autostart`. The gate applies to the agent's tools only; a test pins that the harness's own writes
  never route through a verdict.
- **A pidfile is a claim, not proof**: strategy-supplied identity, the record's port, uid, `exe`, argv
  tokens, session leader, and cgroup + `MainPID` (start time compared) for a unit launch, before any
  `killpg`; `expect=None` never authorises a signal; the harness's small files are written through
  `atomic_write_text` under a no-follow directory fd.
- **Secrets.** `--script` prints placeholders; `serve` passes the key in the environment; tails are redacted
  at the channel boundary with the config's secrets, the key-file key and the two logged forms. HF-cache
  content integrity is a pre-existing residual.
- **No code from files**; every reader is bounded; `tuning.py` runs nothing.
- **The library file is deny-only** and machine-global, a protected config-dir entry: an entry an agent
  plants hides a file it could have deleted; a removed entry makes that file new again; a removed or
  corrupt file re-baselines (nothing asked at `start` until `init`, a functional loss the owner can see in
  `doctor`'s row, not a security one); its fields are validated and capped on read. Nothing is adopted or
  launched through it: the record of a shown file comes from a fresh header read at show time. Every
  question, every library write and `start`'s walk run only for a person (`local_operator`: a tty on stdin
  and stdout, no `LOCALHARNESS_COMMIT`, no `--no-input`; the interview enforces it again inside itself) — a gate against the agent's own tools, not a sandbox: a spawn
  that clears the marker under a pty answers its own questions, the residual `_is_local_operator`'s
  docstring already names; without a person a first run writes its records unadopted as today, no library file, and an
  existing-config run appends nothing. **The record is shown and adopted, never the file's content**: a
  file planted under a model root is shown with its path, size, first-seen date and header facts and asked
  about on the next terminal run, never adopted unshown (spec 16's projector rule, extended to every
  discovery); `model --download` records only for a person and only a file it verified against the hub at a commit
  resolved from the hub, never from the cache path. The 0.18 list copy re-adopts only a peer whose
  `config.yaml` twin was covered.
  Residuals, named: a file whose bytes are swapped after its record was adopted is not re-shown (the
  fingerprint covers the record; the record's path is the HF snapshot link the agent can retarget, which
  also covers the gap between verify and launch); planted files sorting early in a root can crowd real ones
  past `DISCOVERY_MAX_FILES`, and a flood of planted valid models is listed in full within that cap; and the
  later-run question's default is `Set it up` — a file the person did not bring is accepted by Enter on a
  highlighted default they can see, the owner's job statement deciding it (§15).
- **The download offer fetches from a fixed table** (`refarch.py`): for the llama.cpp row one named file
  at a pinned commit through `download_file(revision=)`, then its `bytes` and `sha256` verified on the
  returned file through one `open_regular` descriptor before any record — the HF cache is a model root the
  agent can write under and `hf_hub_download` returns a cached path for a pinned commit without a network
  round trip, so without the hash a same-size planted file would pass as the table's; the record is built
  from that path and a fresh header read alone. What the hash does not cover: the row's author, and a
  compromised repository at that commit. The vLLM rows keep today's whole-repo, unpinned `download_model`,
  and `is_model_cached` accepts a cached directory by name — a planted blob under an etag name is taken as
  cached and the record's `model` is a repo id resolved at launch: that residual stays theirs.

## 13. Files touched

| File | Change |
|---|---|
| `config/models.py` | `ManagedServerConfig.speculative`, `reasoning_parser`, `tool_call_parser`; `AutosenseConfig.kv_cache_type`, `draft_mtp`, `draft_p_min`; `HarnessConfig.pending_preset`; `OrgConfig.default_temperature`, `DivisionConfig.temperature`, `AgentConfig.temperature` → `Optional[float] = None`; the deny-list bundle |
| `config/loader.py`, `config/defaults.py`, `config/migrate.py`, `config/trust.py` | `pending_preset` machine-only; temperature `None` pass-through; `DEFAULT_TEMPERATURE` removed; `init_nulls` revision 2; the three fields in `LAUNCH_MODEL_KEYS`; **`machine_snapshot` emits the `autostart.env` `launch` entry from the harness unit file** (when its `--config-dir` is this `global_dir`) through `_launch_entries`, with the unit's real path in `machine_diff`'s annotations |
| `config/redact.py` | `scrub(text, secrets, *, patterns=())` |
| `core/private_files.py`, `config/overlay.py` | `atomic_write_text` moves in (optional `dir_fd`); `overlay` re-exports it |
| `provider/tuning.py` (new) | `derive`, the `model_type` table, `read_model_config`, `render_script` |
| `provider/autosense.py`, `agent/gguf_header.py`, `agent/context.py` | `FIT_KEYS` + MTP/expert keys, `fit(draft_share=)`; `DISCOVERY_KEYS` + `nextn_predict_layers`, `DiscoveredModel.mtp`; `WindowProbe.sampling` |
| `provider/procfs.py` | `start_time_from_stat` moved in; `ProcRecord.sid`; `LaunchIdentity`; `identity_mismatch`; `read_launch(config_dir, *, expect=None, main_pid=None) -> Launch(pid, start, port, owner, verified)`; the "starting" rule |
| `provider/server.py` | `serve_command(..., cache_type=, spec_type=, draft_p_min=, reasoning_parser=, tool_call_parser=)`; pidfile 1/2/3 tokens through `atomic_write_text`; `server_pid` over `procfs.read_launch`; `stop_server(..., *, expect)` required; `wait_ready(..., expect=, liveness=)`; `child_environment(runtime, base, …)` unchanged in contract, called by `serve` over the allowlisted base; the sync twin of the launch lock on `lock_path` with `_CONTENDED` |
| `provider/lifecycle.py` | `sense_launch`, pure `plan_launch` (owns the retry choice), `LaunchPlan.retry()`, `confirm_launch(plan= \| marker=)` (the one ledger writer; from a marker only when `invocation` matches, `dropped` from the live argv, fit values clamped); `activate` = sense → plan → prepare → execute → confirm; `LifecycleStrategy.launch_identity(spec)`; every `stop` passes `expect=`; `swap(…, reporter)` with `Heavy`, `SwapReporter`, cancellation, `unit-target` removal/rollback, `_restore_incumbent`/`_safe_stop` moved in; `strategy_for(spec, unit=None)`; `UnitStrategy` (`activate` = `write_target` → own-unit re-check → `reset-failed` → `restart` → `wait_ready` with the unit-state callback → `confirm_launch(marker=)`; `stop` = own-unit re-check → by owner; `liveness` = pidfile) |
| `provider/units.py` | `systemctl_path()` (system dirs only; its subprocess environment sets `XDG_RUNTIME_DIR=/run/user/<uid>` when unset and that directory exists and is owned by the uid, so a cron or `ssh -T` run reaches the user manager); `harness_unit_path` (honours `XDG_CONFIG_HOME`), `render_harness_unit(exec_argv, config_dir, captured)`, `captured_env(body)`, `unit_serves(config_dir) -> bool` (the `ExecStart --config-dir` test), `write_harness_unit`, `remove_harness_unit`, `is_harness_unit`, `systemctl_user` (incl. `reset-failed`), `unit_runtime_state` (+ `InvocationID`), `journal_tail`, `linger_on`, `other_launchers`, `translate_exit`, `write_target`/`read_target`/`remove_target`, `write_marker`/`read_marker` |
| `provider/speed_stats.py` | `_load` through `open_regular` with a cap; `record_sampling`/`last_sampling`, `record_retry`/`last_retry`/`clear_retry`, `record_last_served`/`last_served` |
| `provider/client.py` | `LLMConfig.temperature: float \| None = None`; `_sampling_kwargs` |
| `cli/model_ops.py` | `_launch_key`, `_peer_session_key` moved here; `resolve_serving(harness, trust, preset)`; `sense_serving`; `persist_landed(record)`; `redact_for_channel(text, harness, key_file_key)` |
| `cli/start_cmd.py` | `--preset/-p` (+ hidden `--scripts`); `sense_serving`; attach / explicit swap / launch; `read_launch(expect=, main_pid=)`; `persist_landed`; `last_served`; the derived, resume, other-preset, boot-retry clauses; the sampling override line; the root temperature into both `LLMConfig` calls; the unit sensed once; redaction at the console; `confirm_launch(marker=)` on attach when a marker exists; `library.sync(scope="new", person=)` once per process in the sync body (terminal channel, before `decide_machine_trust`, not on the `/plugins` restart pass, not after a first-start setup, not with `--list-models`); `local_operator(no_input)` computed once at the boundary for `core_setup(interactive=)` and `sync(person=)` |
| `cli/repl.py` | `_active_heavy` from the serving record; the cold branch through `lifecycle.swap` with the REPL `SwapReporter`; `persist_landed`; wrappers; typed-`.gguf` refusal under the unit; `strategy_for(..., unit=)` |
| `cli/model_cmd.py` | `decide_machine_trust(ask=local_operator(no_input), launching=will_launch)` + `machine_withheld` before any swap; the immediate `UnitStrategy` swap (console reporter: launch line before the wait, receipt after with the `live_sessions` clause; `persist_landed`), else `pending_preset` with the unit-gated receipt; `--script [--all]`; `--autostart on\|off\|status` (local-operator gate, then `ask=True`; scope refusal with its fix; captured `# env` lines recorded as the `autostart.env` launch entry, relative `PATH` entries dropped and named; `enable` without `--now` over a live launch; one-line receipt; `status` incl. captured `PATH`, second launcher, marker, translated exit) |
| `cli/serve_cmd.py` (new), `cli/app.py`, `src/localharness/__main__.py` (new) | hidden `serve` per §9.3 |
| `provider/model_library.py` (new) | pure, imports no `cli/`: `scan(roots, recorded, library_path, *, scope, rank) -> Found`, `peers_for`, `free_ports` (moved from `init_cmd` with public names), `Library.load` / `Library.save` (`<global>/model-library.json`: `first_seen`, `size`, verdict, `declined`; `open_regular` under a cap, validated and capped fields, `atomic_write_text` under the overlay lock, the baseline, the verdict transition) |
| `cli/library.py` (new) | `InitRecords`, `preset_lines`, `show_records`, `_machine_autosense`, `_adopt_written`, `_record_dicts_of`, `_result_from_record` moved from `init_cmd`; `sync(config_path, *, scope, person, preset=None, choose=False) -> Sync`; `append_peers` as a `mutate` fragment (peers always in `overrides.yaml`; the 0.18 list reaches it through the posture flow's existing copy, `_copied_peer_entries` and `PEER_LIST_COPIED`), `record_path`, `counts`; the `rank` callable; the §11.1 questions and their after-lines; the `Later:` line; the download offer with the ordered default rule, the one-descriptor `bytes` + `sha256` verification and the record line; the per-platform install line; the `doctor` row |
| `cli/interview.py` (new) | `Option`, `Answer`, `ask(question, options, *, default, cancel, multi=False)`: the one prompt_toolkit question app of §11.5 (inline, `PICKER_STYLE` and the picker's test seams, the hint line, scrolling, cancel); `local_operator` enforced inside; no typed fallback |
| `cli/init_cmd.py`, `cli/doctor_cmd.py` | `init` calls `sync(scope="all")` in `_pick_server` (a first run: no `config.yaml` or `--force`; after `_unit_import`) and inside `_existing_config` after the load, not when `changing`, `Keep it?` skipped when `sync` appended; `_found_models`, `_peer_records`, `_free_ports`, `_discover_presets` deleted; `_records_from_import` takes `peers_for`; `_write_overlay_records` calls `append_peers` (its empty-only rule gone); `_write_records` writes `config.yaml` with `extra_endpoints: []`, then the peers into `overrides.yaml`, then two adoptions, the peer-write False path said; `_SAVED_CHOICE_KEYS` gains `extra_endpoints`; `_guided_setup` reduced to the vLLM install step; `is_interactive` and `_adopts` folded into `local_operator`; a first run that attached baselines; the receipt's `Later:` line replaces `Run 'localharness start' to begin.`; `doctor`'s `Models:` row beside `Tuning:`, `Sampling:`, `Autostart:` |
| `cli/workspace.py`, `cli/model_cmd.py` | `local_operator(no_input)` (the one predicate; `_is_local_operator` folded in), `write_overlay_validated` under the new `overrides.yaml.lock`; `--download` ends in `record_path` for one `.gguf` verified at a hub-resolved commit under a config with a person, else says `start finds it`; `model --setup` = `sync(choose=True)` (§11.1) |
| `provider/refarch.py`, `provider/server.py`, `agent/gguf_header.py`, `agent/gate_types.py`, `SECURITY.md` | `RefArch`: `runtime`, `platforms`, `needs`, `repo`, `file`, `bytes`, `revision`, `sha256`, `page`, route fields optional, the 8 GB llama.cpp row, a test on every row's shapes; `download_file(repo, file, *, revision)`; `discover`'s return type becomes one `FileKind` per path (`unreadable` included), `rank_models` unchanged and exported; `model-library.json` and `overrides.yaml.lock` in `PROTECTED_CONFIG_DIR_ENTRIES` (SECURITY.md's count, now ten, and the gate test beside `project_records.yaml.lock`); the library file in SECURITY.md's written-files list; `init_cmd`'s and spec 16 §8's "config.yaml is not touched" text amended for the one-time list move |
| `README.md`, `orchestrator/router.py` | Quick start and `no_config_message` teach `localharness start` first; a model server is not required |
| `agent/gate_types.py` | the four login-time directories in `PROTECTED_PATHS_HOME_DEFAULT` |
| `bench/`, `SECURITY.md`, `docs/specs/10-cli.md`, `13-provider-support.md` | the bench temperature paragraph; the autostart unit (environment order, captured comment lines, allowlisted base, identity test, owner rule, exit codes, the marker, the deny bundle, the four directories, the residuals), the trust decision on every launching surface, the pidfile identity check, the redaction boundary, the three new `launch_model` keys; the CLI additions; the strategy-layer additions |

## 14. Tests

- Unit, on fakes: `derive` on the three box headers and every precedence case; `serve_command` keywords at
  the field's slot only when unset; the draft share; `sense_launch` once / `plan_launch` and `retry()` pure
  and deterministic; identical argv for `activate`, `serve`, `--script`, `doctor`, the home prepared only by
  executors; the vLLM table and bad `config.json`; **the retry chosen only in `plan_launch`** (from a death;
  from `remembered` with `-c` from the fresh fit; from `prior_death` only with a marker for this record and
  binary, `cause: death` or `shape: full`, younger than `TimeoutStartSec`, `NRestarts ≥ 1`); **`confirm_launch`
  the only ledger writer** (structural over `record_retry(` callers), never sensing, from a plan with the
  pre-launch values, from a marker only when `invocation` matches the live one and the live argv's `-c` equals
  `planned_context`, `dropped` from the live argv (a marker claiming `dropped: []` beside an argv without
  `--spec-type` records `draft-mtp`), fit values clamped, any matching marker removed, a non-matching one
  left; `start` calling it only when a marker exists; the ledger read under a cap; the pidfile and
  `identity_mismatch` cases as before, `Launch.owner` from cgroup + caller-supplied `main_pid`,
  `UnitStrategy.stop` re-checking the unit **before** `reset-failed`, picking `systemctl stop` for a unit
  launch, the wrapped `stop` for a spawned one, naming a foreign unit's process; `atomic_write_text` with
  `dir_fd`; `resolve_serving`: the full ladder, withheld → refusal/78 with nothing cleared, gone names by
  `base_url` then unique `model`; `sense_serving` loopback-only; `start` attach / clauses gated by
  `last_served` / explicit swap / launch / `--preset` naming what serves / `--scripts`; **`persist_landed`
  the one writer** of `active_endpoint`, `last_served` and the `pending_preset` removal, called by `start`,
  the REPL wrapper and `model_cmd` (structural); the CLI: `decide_machine_trust(ask=<local operator>,
  launching=)` and `machine_withheld` before any swap — `ask=False` with a piped stdin (a piped `y` adopts
  nothing; an EOF records no No), `launching=True` when the swap launches (a llama.cpp `bind_all: true`
  target asked on a terminal, not quiet-withheld) — (structural: the swap path cannot run without a
  `MachineTrust`; no `decide_machine_trust(ask=True)` literal under `cli/` except `on` after its gate), a
  withheld target refused with the line and exit 2, the immediate swap only with the unit active, no
  `LOCALHARNESS_COMMIT` **and** the target not withheld (a no-tty script run swaps; a run under
  `LOCALHARNESS_COMMIT` writes `pending_preset` only), its launch line before the wait and its receipt
  after, the REPL clause only when `live_sessions` is non-empty, `pending_preset` otherwise with `… the
  next start serves it` (no unit) / `… or boot serves it` (unit enabled); `confirm_launch` from a marker only with `NRestarts ≥ 1`, `dropped`
  only when `derive` would have emitted `spec_type` (a `speculative: off` record with a planted marker
  records nothing); `/model` and `start --preset` identical fake-lifecycle sequences; `SwapReporter`
  callbacks, cancellation (target stopped, `freed()`, `unit-target` rolled back, no restore); `unit-target`
  written only by `UnitStrategy.activate` (structural over `write_target(` callers), removed by `swap` for a
  docker target with the clause, rolled back by the restore; `--model` with `--preset`; temperature (`None`
  on all three builders, the root value sent, `None` division, `0.0` probes, migration rev 2, `init` dumps
  `null`); the sampling override line with its undo on change; `--script` escaping, placeholders, `--all`
  functions, vLLM; autostart: gates and refusals with their fixes, the rendered unit (interpreter `-I -m`,
  `--config-dir`, `WorkingDirectory`, the `# env` comment lines for `PATH` always and `HF_HOME`/
  `LD_LIBRARY_PATH` only when set in the shell, relative `PATH` entries dropped and named, `UnsetEnvironment`
  without the captured names' values ever depending on it, start limit), `captured_env(render(...)) ==
  captured` round trip, **`machine_snapshot` emitting `autostart.env` (kind `launch`, file
  `systemd/localharness-llm.service`) from the unit file only when its `--config-dir` is this dir** (a second
  config dir's snapshot emits nothing), `on` adopting it, the entry **still recorded after a `start`'s
  `decide_machine_trust` and after `serve`'s own**, a changed `# env` line uncovered (asked on a terminal
  with the unit's real path in the line, withheld under `serve` → 78), a unit file present with the entry
  missing or unparsable → 78, `off` pruning it as a removal, a relative element in `PATH` or
  `LD_LIBRARY_PATH` dropped and named with its variable at `on`, a relative `HF_HOME` made absolute against
  the shell's cwd and named, anything non-absolute at `serve` → 78, `captured_env` ignoring `# env` lines
  outside the header block and refusing an unknown or duplicate name, `systemctl_path()`'s environment
  carrying `XDG_RUNTIME_DIR` under a cron-like empty environment, the immediate swap refused into the
  `pending_preset` path when no machine record exists, an attach on the `pending_preset` rung consuming the
  key, charset refusal, `daemon-reload` then `enable --now` / `enable` alone over a live
  launch, `reset-failed` before every `restart`/`stop` and after the identity re-check, system-dir
  `systemctl`, body shown only on a differing rewrite, `off` only on all three identity tests, the identity
  test failing on a drop-in / `FragmentPath` / one byte and re-run before each `restart`/`stop`, the
  second-launcher line only for an enabled or active unit, the one-line receipt with the captured `PATH`,
  linger stat, exit translation, the marker in `status`; `UnitStrategy` only from a harness-own `unit=` and
  a binary runtime, the structural `strategy_for(` test, the wait tolerating exactly one `NRestarts` rise
  with derived keywords and none without, stopping at the second with the raw tail, redaction (`scrub` with
  `patterns`, the key-file key, URL credentials) applied in the REPL, CLI and console reporters and absent
  from the `RuntimeError`; `serve`: pinned dir before any resolver, 78 (no record, withheld), 0 (already
  serving, non-binary target), `captured_env` read from the verified unit file and not from `os.environ` (a
  manager `PATH` decoy absent), the allowlisted base through `child_environment` (decoys incl.
  `CUDA_INJECTION64_PATH` absent, `CUDA_VISIBLE_DEVICES` and the key present, nothing on argv, and
  `dict(base)` used when `child_environment` returns `None` — `os.environ` never passed), the marker with
  record, binary, invocation,
  cause, shape, `planned_context`, pidfile + marker under the sync lock (the same `lock_path`) before
  `execve`, the module entry importable; the new fields fingerprinted with `None` → no entry;
  `pending_preset` dropped from a workspace layer; the four protected directories and the deny bundle; the
  harness's own writes never through a verdict; `doctor`'s rows; `init`'s clause and vLLM peers; **the model library and the interview** (§11): `scan` takes roots and
  `recorded` in, prints nothing, reads a header only for a path in neither (new is first sighting: a
  backdated or `cp -p` file is new, a known path with a future mtime is not, a verdict becoming `model` is
  new), realpath dedupe, `other`/`projector`/`shard` never re-read, `unreadable` re-tried, a resized file
  re-typed; the library file written only by the provider module (structural), the baseline off on a first
  run and on when the file is missing or corrupt (an attach-only install included), a removed entry and a
  removed peer new again, the first-run `sync` writing the file, an oversized, non-regular or malformed file
  refused; the interview app: the default highlighted and taken by Enter, arrows and numbers, Space when
  multi, Esc and Ctrl-C the no-change option, never reached without a person (structural: every `ask(`
  caller is behind `person`); the first-run question with its default writing every shown file and the best
  as `server:` (peers into `overrides.yaml`), `Choose which` declining only an unselected file and picking the
  first, N = 1 and `--preset` folding the order step, `_unit_import` winning silently, `--force` a first run
  with `recorded` empty, a first run that attached asking nothing; `scope="all"` listing every unrecorded
  model with the declined ones tagged and excluded from the default, `scope="new"` only first sightings, `Not
  now` marking seen so `start` stops and `init` lists, selecting a declined file clearing its decline, every
  after-line; `append_peers` into `overrides.yaml` under the lock with names and ports unique over every
  record and `server:` untouched, the 0.18 one-time list move with its re-adoption and line, two writers
  racing under the lock losing nothing; `init`'s `_existing_config` call site reached with a server answering
  and not, skipped when `changing`, an unreadable config exiting first, `Keep it?` skipped after an append;
  nothing asked, walked or written without a person (a faked tty under `LOCALHARNESS_COMMIT`, no tty,
  `--no-input`; a first run then writes unadopted records and the next terminal `start` asks once; an
  existing-config run appends nothing); the structural `sync(person=local_operator(...))` pin; the `Later:`
  line on the first run only, its `/model` clause only with a peer, replacing `init`'s run-start line; the
  `doctor` row; the download default by the ordered rule from a faked platform, `needs`, `PATH` and memory
  (a CPU-only box never offered the Spark row, a Mac with `llama-server` the GGUF row, unknown memory → the
  smallest with `needs about`), no runtime → the question still asked with the install line printed before
  and after, `Show the other rows`, `Not now` → the page, the command and the run-again line with exit 0,
  `Download it` through a stubbed `download_file` with the revision passed and `bytes` and `sha256` checked
  through one descriptor (a mismatch, a FIFO or a device named, nothing recorded) and the record from that
  path alone while a planted sibling is not swept up, its record line printed before the write;
  `model --download` recording only one `.gguf` under a config for a person that verifies against a stubbed
  hub hash, otherwise saying `start finds it`; a planted file shown with path, size, first-seen date and
  header facts and never adopted unshown; every single-file `refarch` row's `revision` 40-hex and `sha256` 64-hex (the vLLM rows carry none, §11.2); the interview driven with a pipe on stdin, a
  non-tty stdout and under the marker reading nothing and returning `cancelled`; the first-run write order
  with the peer-write False path; `--force` clearing `extra_endpoints`; the 0.18 copy re-adopting only a
  covered twin; `model --setup` over seen and declined files; `init` asking only on a first sighting and its
  default never taking a baseline or `Not now` file; the vLLM row's one consent; `Serve which first?` skipped
  with one selected; a held lock failing closed with the line; `model --download` with a planted
  `snapshots/<fakesha>` recording nothing; the §2 carry rows and SECURITY.md's protected-entry count; no test touches the network. **No test
  runs `systemctl`, `loginctl`, `journalctl`, `busctl` or a unit.**
- Live, on the reference box (a runbook, as spec 16's live check was): (1) `init` after the 0.18 import — `mtp` on
  both 27B lines; (2) `start` on the Swift preset launches with `q8_0` KV, `draft-mtp`, a fitted `-c`;
  `/props` shows `speculative.types: draft-mtp` and the asked window; decode in the profile's band; (3) a
  `--debug` body carries no `temperature`; `/props` `temperature 1.0`; (4) `/model flash-next`, `quit`,
  `start` attaches within seconds with the resume clause once, `start --preset qwen-27b-swift-1.5` swaps
  back; (5) `llm.service` hand-started on :8080 with `active_endpoint` = flash-next: `start` attaches to
  qwen with the clause, launches nothing; (6) `localharness model flash-next` from the shell (no unit) →
  `Selected … — the next start serves it`, then `start` swaps; (7) `--autostart on` while the spawned qwen
  serves: `enable` only, the server untouched, the receipt names the captured `PATH`; `/model flash-next`
  frees it (the spawned stopper) and the unit serves flash-next (`MainPID` = `llama-server`); five `/model`
  switches inside an hour all succeed; `localharness model qwen-27b-swift-1.5` from the shell prints the
  launch line, then `Serving … — autostart restarted`; reboot: the unit serves qwen; `start` attaches
  silently; `readelf -d` on `llama-server` confirms rpath and the unit's environment (`systemctl --user
  show-environment` is not used; `cat /proc/<MainPID>/environ` through the owner's shell) holds the captured
  `PATH` and no manager `LD_LIBRARY_PATH`; `--autostart off` removes the unit, `llm.service` untouched; (8)
  `--script` pasted into a shell serves on a spare port; `--script --all > f; . ./f; serve_flash_next` serves;
  (9) a fresh config dir on the box: `localharness start` → the interview with `All, serving qwen-27b-swift-1.5
  first` highlighted, Enter, the receipt's `Later:` line; `model --download --file` a small GGUF → its commit,
  hash and record line at once, the next `start` asks nothing; a GGUF copied in with `cp -p` → the next `start`
  asks once, `Not now` and no later `start` asks, `localharness model --setup` lists it, unselecting it
  declines it and the next `--setup` shows it tagged; `doctor`'s `Models:` row counts both; the interview
  on a narrow terminal and under tmux and `script`.

## 15. Open questions and recorded decisions

- **Revision 10, final text (2026-10-16) — the fourth round folded; no fifth pass, by the owner's instruction
  to plan.** Parsimony SHIP (notes folded: `Option` / `Answer`, the picker's style and seams shared, the
  `rich` sites named as debt, `_record_dicts` a thin caller). Design REVISE, folded: the first-run write
  order with split adoption and the False path; `--force` clearing `extra_endpoints`; the 0.18 list reaching
  the overlay through the posture flow's existing copy with `_copied_peer_entries` (no second copier); the
  `ask` contract (`cancel`, a set default, cancel per step); `scan` taking a `rank` callable because
  `rank_models` needs the ledger and the fits; `FileKind`, not a second `Verdict`; the lock's acquisition,
  scope, failure and non-nesting; `append_peers` as a `mutate` fragment so the import stays one write; a
  cancel on `start` counting as `Not now`. Security REVISE, folded: the interview enforcing the gate inside
  itself with stdout a tty too; the copy re-adopting only a covered twin; `model --download` resolving the
  commit from the hub, never the cache path; the lock fail-closed; "never adopted unshown" and no library
  file without a person. Friction REVISE, folded: the vLLM row's one consent inside `Download it`; `init`
  asking only on a first sighting with a default that never takes a baseline, `Not now` or declined file,
  and `localharness model --setup` as the one way back (named by every after-line and `doctor`); bullets
  instead of list numbers; the hint line and number-toggles in `multi`; `Serve which first?` skipped with one
  selected; `Set it up` / `Not now` / `Skip for now` everywhere with descriptions; cancel never silent; the
  app inline with scrollback kept and a draw failure taking the non-person path; the install block ending
  in `then run localharness start`; the small row's text. The plan checker verifies these folds against the
  source before execution.
- **Revision 10, fourth draft (2026-10-16) — the interview, and the third round folded.** The owner's call,
  2026-10-16: every question of §11 is asked through one interview component (§11.5, `cli/interview.py`,
  prompt_toolkit, beside `cli/picker.py`) — a question, options with a highlighted default, arrows or a
  number, Enter — so no question in this spec carries a key grammar and the drafts' `[Y/n · d · numbers]`
  forms are gone; the later-run default is `Set it up`, the lazy answer, with security's `[y/N]` note
  retired into a §12 residual (the default is visible before Enter). Third-round folds: parsimony (SHIP,
  notes) — `_adopt_written` and the record helpers re-homed to `cli/library`, one predicate
  `local_operator`, one peer-write rule (peers always in `overrides.yaml`), `scan(unit_model=)`, public
  names `peers_for` / `free_ports`, `_machine_autosense` in `cli/library`, the unit import named as the one
  intentional silent path, the stale `_is_local_operator` mentions fixed; design — `provider/model_library`
  takes roots and `recorded` in and imports no `cli/`, the `config.yaml` append writer that did not exist
  replaced by "peers always in `overrides.yaml`" with the 0.18 one-time move, the overlay lock defined
  (`overrides.yaml.lock`, a protected entry), `--force` a first run, `sync` inside `_existing_config` after
  the load and not when `changing` with `Keep it?` skipped after an append, `person` threaded from one
  boundary computation, the verdict-becomes-`model` first sighting, the first-run `sync` writing the file,
  the removed-peer rule, `platforms` and `repo` replacing `platform` and `default_model`, one shown block;
  security — `model --download` verifying against the hub's LFS hash before it records (a cache hit on a
  pinned commit is otherwise a planted path), the one-descriptor hash-and-header read, the vLLM residual
  stated exactly, a non-person existing-config run appending nothing, the library file's fields validated
  and capped, the flood named; friction — `needs` on the rows so a CPU-only box is never offered the
  Spark checkpoint and a Mac with `llama-server` gets the GGUF row, the install line written per platform
  and printed before the question, the way back from `Not now` and from a decline as after-lines and as
  `init`'s `Choose which`, the declined files excluded from `init`'s default, `start --preset <name>` named
  after a shell download, `Show the other rows` instead of a key. Fourth round pending.
- **Revision 10, third draft (2026-10-16) — the second round of the four critics folded.** All four REVISE
  on the second draft; folded: parsimony — the component split into `provider/model_library.py` (pure:
  `scan`, `append_peers`, `record_path`, `counts`, the file) and `cli/library.py` (the prompt, `sync`, the
  offer, the row), `RefArch` and `download_file(revision=)` named as schema and signature changes, the
  `init_cmd.py` deletions listed so no second path survives, the `Verdict` enum in one place, the operator
  test read inside `sync` rather than passed; design — `Sync`'s contract and the `preset` argument, "new"
  as first sighting (not mtime), the baseline on `config.yaml` present rather than records present, the
  append into the file that wins the merge (no copied peers, so no re-asked trust), `init`'s second call
  site in the existing-config branch, `server:` only when no `config.yaml` existed, `--no-input` in the
  gate, `unreadable` re-tried, the §2 carry rows, the library write under the overlay lock, `sync` skipped
  after a first-start setup and with `--list-models`; security — `sha256` on the row and verified on the
  returned path (a pinned commit alone lets a planted cache file pass), `model --download` recording only
  for a local operator and one `.gguf`, the record line printed before the offer's write, the vLLM rows'
  unpinned whole-repo fetch said plainly, the gate called a gate and not a sandbox, the fresh header read
  at show time, the roots from the machine config; friction — `init` reachable as the way back (the
  existing-config call site; declined files listed tagged so the undo is a typed number, not a file edit),
  a first run that attached asking nothing, `d = not now` named and its after-line specified, the N = 1
  wording, the `Later:` line on the first run only with the `/model` clause only with a peer, the `l` key
  dropped, the install line printed again after the download and on the next `start`'s refusal, No at the
  offer and everything-declined exiting with one line rather than `_skip_for_now`'s two prompts, the
  numbered lines, memory sensed from RAM where there is no GPU. **Still open for the owner:** the
  later-run default — `[Y/n]` as drafted (the lazy path; friction) or `[y/N]` (security: a file the person
  did not bring should take a typed `y`; the first-run prompt keeps Enter either way). Third round pending.
- **Revision 10, second draft (2026-10-16) — the first-run job; four critics' first round folded.**
The
  owner's job statement (§1 item 5) widened §11; spec 16 §12's zero-asks ruling, recorded there as
  reversible by the owner, is revised. First-round verdicts, all REVISE, folded as follows. Parsimony
  (the fourth critic, added this day: plugin or core; which component; readable and extensible in one
  place): core, and shape (b) — one new module, `cli/model_library.py`, owns the inventory, the file, the
  prompt, the write and the offer, with `init`, `start`, `doctor` and `model --download` as callers,
  instead of the first draft's five-file spread. Design: the append inside `mutate` (an `overrides.yaml`
  list replaces, so the old empty-only gate was the only thing making it safe), names and ports seeded
  from every record, `server:` untouched after the first run; the Spark's two-file llama.cpp route not
  offered; `free_memory` None defined; the call site in `start`'s sync body before `decide_machine_trust`;
  everything-declined → skip-for-now; `--preset`, N = 1 and the unit import fold the prompt; the record
  after a download from the returned path alone; realpath dedupe; `other`-typed and growing files
  remembered. Security: local-operator gate on every prompt, write and walk; what is shown (path, size,
  mtime, header facts, `new since`); the pinned revision and size check; the §12 claims narrowed with the
  residuals named; the one library file replaces the shared declined-offer store for files. Friction:
  one prompt whose Enter never declines (a decline is a typed number), `n` on a later arrival meaning
  "not now" rather than a permanent No; the receipt cut to today's block plus one `Later:` line teaching
  `localharness start`; the download menu skipped when one row is sensed and the offer kept when no
  runtime is installed; the upgrade baseline so an old install is never asked about dozens of files;
  README and `no_config_message` teaching `start` first. **Open for the owner:** the friction and security
  critics split on the later-arrival default — `[Enter = yes]` as drafted (the owner's lazy path) or
  `[y/N]` (security's: a planted file should take a typed `y`); §12 names it as a residual under the
  owner's job statement. Second round of all four critics pending on this draft.
- **Iteration 8 (applied 2026-10-13) — SHIP from all three critics on §§5–16.** Eight minors folded:
  design — the `autostart.env` entry is emitted only for the config dir the unit's `ExecStart` names,
  `harness_unit_path()` honours `XDG_CONFIG_HOME`, the unit's real path rides `machine_diff`'s annotations,
  the §13 trust row names the emission; security — the marker is described as a default, not a boundary,
  and the immediate swap also requires a machine record to exist (`serve`'s own guard); friction —
  `systemctl_path()`'s environment supplies `XDG_RUNTIME_DIR` for cron, an attach on the `pending_preset`
  rung consumes the key, the drop receipt names the variable and a relative `HF_HOME` is made absolute
  rather than dropped. Residual the design pass noted: the `bash_tool.py` line that sets
  `LOCALHARNESS_COMMIT` has moved since 0.18 (cosmetic).
- **Iteration 7 (applied 2026-10-13).** Design REVISE (1 major) and security REVISE (1 major, 1 minor) on
  the same point, folded: `machine_snapshot` emits `autostart.env` from the unit file as a `launch`-kind
  entry through the existing `(prefix, mapping)` helper, so the anchor survives every re-record; `serve`
  exits 78 when the entry is withheld, missing or unparsable; a new kind was refused (silent adoption of a
  planted unit); the absolute-element rule covers `LD_LIBRARY_PATH` and `HF_HOME`; what the anchor does and
  does not protect against stated. Friction REVISE (1 major, 1 minor), folded: the CLI's immediate swap is
  gated on "`LOCALHARNESS_COMMIT` absent and the target not withheld" — the conditions `serve` launches
  under — not on a tty, so a script, cron job or `ssh -T` run switches as the owner's `llm start <profile>`
  does, while the question stays on the terminal; "off a terminal" wording narrowed to uncovered entries
  with the exit code. §17 shape taken from the advisory: `remote_control` on the `EndpointRef`, not a
  lifecycle.
- **Iteration 6 (applied 2026-10-13).** Design SHIP with two minors, folded: the CLI receipt's REPL clause
  reads `session_presence.live_sessions`; `confirm_launch`'s inputs typed, with `context_length`'s source
  named. Security REVISE (1 major, 2 minor) and friction REVISE (1 major, 2 minor) on the same gap, folded:
  the CLI runs `decide_machine_trust(ask=_is_local_operator(), launching=will_launch)` — `start`'s own
  `ask=can_ask` — so a piped `y` adopts nothing and an EOF records no No; `--autostart on` asks only after
  its gate; a structural test forbids the `ask=True` literal elsewhere under `cli/`; `serve` execs
  `dict(base)` when `child_environment` returns `None`; the marker writes the ledger only with `NRestarts ≥
  1` and `dropped` only when `derive` would have emitted `spec_type`; the captured unit environment is
  anchored as the fingerprinted `autostart.env` launch entry, `serve` exiting 78 on a mismatch, with the
  parse limited to the header block and a closed name set; the exec allowlist names every variable (no
  wildcard); the dropped-`PATH` wording fixed. §17 added at the owner's request (optional, not in 0.19.0).
- **Iteration 5 (applied 2026-10-13).** Design REVISE (2 major, 3 minor): `confirm_launch` never senses —
  from a plan with pre-launch values, or from a marker bound to the live `InvocationID` with the served
  window equal to `planned_context`, `dropped` from the live argv and the marker's fit values clamped; the
  systemd ground truth corrected (`UnsetEnvironment=` is applied last and undoes `Environment=`), so the
  captured `PATH`/`HF_HOME`/`LD_LIBRARY_PATH` ride as `# env` comment lines in the byte-equal body and
  `serve` reads them with `captured_env()` from the verified file, never `os.environ`; `invocation`'s role
  stated; `persist_landed` as the one outcome writer on every surface; `unit-target` written only by
  `UnitStrategy.activate`; the sync launch lock stated as the same `flock` and `_CONTENDED`; `serve`'s
  environment as `child_environment` over an allowlisted base. Security REVISE (1 major, 3 minor): the
  CLI's immediate swap runs `decide_machine_trust` (then `ask=True`; iteration 6 gated it on the local
  operator) and loads with `machine_withheld` (structural
  test); relative `PATH` entries dropped and named; marker `dropped` never reaches the ledger; `stop`
  re-checks the unit before `reset-failed`; the captured `PATH` shown in the receipt and `status`, the
  install-tree residual named. Friction SHIP with five minors, folded: the CLI swap prints the launch line
  before the wait and names a live REPL session in its receipt; it persists through `persist_landed`; the
  `Selected` receipt says `or boot` only when the unit is enabled; a matching marker is always removed on
  attach; the peer refusal names `autosense.model_roots` and `localharness init`.
- **Iteration 4 (applied 2026-10-13).** Marker bound to record, binary, run and age; `atomic_write_text`
  reused with `dir_fd`; `MainPID` passed into `procfs`; `units.write_target`; one `confirm_launch`; the two
  logged redaction patterns; `wait_ready` with a callback; the owner rule in `UnitStrategy.stop`; the D-Bus
  twins in the deny bundle; `HOME`/`PATH`/`HF_HOME`/`LD_LIBRARY_PATH` from the unit's own captured values;
  the server directory opened no-follow; the key-file key in the scrub set; `MainPID` start time compared;
  a foreign unit named, never stopped; `unit-target` rolled back on cancellation; `reset-failed` before
  every harness-issued `restart`/`stop`; the boot clause only for a `death` marker with no prior record;
  `last_served` defined; the docker-under-unit clause; the pidfile refusal's fix; the marker in
  `status`/`doctor`; the CLI's immediate unit swap.
- **Iteration 3 (applied 2026-10-13).** One ladder incl. `unit-target`; `plan_launch` owns the retry;
  procfs owns the `/proc` side; `expect` inside the strategies; `launch_identity` plain data; redaction at
  the boundary; cancellation contract; withheld ≠ gone; no-follow writes; the first deny bundle; allowlisted
  exec environment; "starting" bounded; `Launch.owner`; the unit wait tolerating one rise; `pending_preset`
  at boot; the boot marker; scope refusal fix; two-line script comment; receipt path; `status` second
  launcher.
- **Iteration 2 (applied 2026-10-13).** Strategy-owned identity; `unit-target`; sense/plan/retry split;
  `swap` with `Heavy` and a reporter; remembered retry vs `full_fit`; pinned `LOCALHARNESS_DIR`; structural
  `strategy_for` test; `systemctl_path()`; four login-time directories protected;
  `set|import|unset-environment` denied; `LD_LIBRARY_PATH GCONV_PATH` unset; identity test re-run; explicit
  no-record check; the pidfile port never trusted alone; session-leader and cgroup tests; ledger bounded;
  tails redacted; `pending_preset`; `enable` without `--now` over a live launch; the unit's retry and start
  limit; on-change clauses; the retry undo; one-line receipt; exit translation.
- **Iteration 1 (applied 2026-10-13).** §7 rebuilt on the inert rung; knobs into `autosense`; per-field
  `serve_command` keywords; one launch plan; `swap` in `lifecycle`; `_launch_key` into `model_ops`; the unit
  sensed once; autostart scoped; `off`; sampling from `probe_window`; one `read_launch`; hidden `serve`;
  pinned `ExecStart`; `serve` never adopting; unit identity test; procfs verification; vLLM peer binary
  shown; `--script` escaping; header ints validated; plain `start` attaches to what answers; retry
  remembered; unit wait bounded; sampling line reduced; GPU-holder folded; second-launcher line gated;
  `--script` leaves `-c` to `--fit`; unique-prefix names; `--scripts` alias.
- **Zero asks (carried from spec 16).** `init` asks nothing; autostart is an explicit command; the one
  machine-trust question is the same one on every surface that puts it (`start`, `--autostart on`, the
  CLI's immediate swap).
- **`--preset`, with `--scripts` as a hidden alias.**
- **Switching the temperature rung on for the root — confirmed by the owner 2026-10-13.**
- **Derive at launch, pin nothing.** **`kv_cache_type` `q8_0`, `draft_p_min` 0.6 — confirmed by the owner 2026-10-13.**
- **The four protected login-time directories and the deny bundle — confirmed by the owner 2026-10-13.** Each entry writes
  what runs at the next login — the criterion the existing `stop|disable|kill|mask` entries share — against
  the owner's 2026-09-11 ruling that hard blacklists stay minimal.
- **Residuals named in `SECURITY.md`:** the D-Bus route to the user manager; a planted pidfile naming the
  person's own `setsid` server on the record's port; HF-cache content integrity (pre-existing); the
  harness's install tree and the captured `PATH`'s user-writable entries running at boot as they do from
  the shell; a plain `start` attaching to a non-selected preset leaves `unit-target` and so the boot on the
  older choice; a death after readiness with no `start` in between is confirmed by nothing until the next
  attach.
- **Unchanged from revision 1:** a llama-server without `--fit`; vLLM peers needing a binary on `PATH` or in
  the harness venv; `--max-model-len`/`--kv-cache-dtype` for vLLM; `-np`; autostart beside a foreign enabled
  unit; the MTP draft context's KV share.

## 16. Out of scope

- Multi-GPU, tensor split, `--fit-target` per device.
- Draft models from a second file (`-md`).
- Ollama and LM Studio tuning; the unit serving a docker, Ollama or LM Studio record (they run outside it).
- A per-child temperature and a session `/temperature` command.
- A typed `.gguf` restart under autostart (add a peer instead).
- Downloading a model by preset name.
- Remote presets over a tailnet (§17) — recorded, not scheduled.

## 17. Optional: presets on another authorized device over the tailnet

**Status: an owner-requested sketch (2026-10-13), not part of 0.19.0 unless the owner schedules it.** The
job: *"my laptop uses the Spark"* — the harness on one machine (the laptop) works against a preset served by
another machine on the same tailnet (the Spark), and can switch what the Spark serves without a shell on it.
Everything below reuses spec 16 and this spec; nothing new runs on the remote beyond its own harness CLI.

- **What already works.** An `extra_endpoints[]` peer whose `base_url` is a tailnet address
  (`http://spark.tail1234.ts.net:8080/v1`) is an attach-only peer today: `/model <peer>` binds to it when it
  answers, its `api_key` rides as the bearer, `probe_window` reads its window. The Spark's side is spec 16
  as it stands: a preset with `bind_all: true` and `require_api_key: true` (or an `api_key_file`), the
  posture line and `doctor` rows naming the exposure, machine trust covering the record. The tailnet is
  the network boundary; the key is the per-request one.
- **The extension.** The peer stays **attach-only** (`lifecycle: None`, `gpu: false` — a lifecycle would
  make it a heavy of the laptop's and the one-heavy rule would stop the laptop's own server to "free the
  GPU" for a model on the Spark) and gains a sibling `remote_control: {host: <ssh host alias>, via: ssh}`
  on the `EndpointRef`, fingerprinted (`remote_control.host` runs `ssh` on the laptop). `/model <peer>`
  attaches as today and first calls a small `RemoteControl.select(served name)` — `ssh -T -o BatchMode=yes
  <host> -- localharness model <served name>` — then `wait_ready` on the peer's `base_url`, failing fast
  when the served name the peer reports is not the one asked for, and naming its bound while it waits.
  Under `ssh -T` there is no tty, so on the remote the question is never put; with §8.1's rule the remote
  **swaps anyway** when its own unit is active, `LOCALHARNESS_COMMIT` is absent and the record is not
  withheld — the Spark's own records and machine record decide, the laptop can only pick among them — and
  otherwise the remote writes `pending_preset` and prints `Selected …`, which the laptop relays prefixed
  with the host. A withheld record is the remote's one-line refusal, relayed with the host prefix. The
  laptop never runs `systemctl` on the Spark, never reads its files, never holds its keys beyond the peer's
  `api_key`, which never rides the `ssh` argv or environment.
- **Authorization** is SSH's: the host alias, the key in `~/.ssh` (a protected path), the remote's
  `authorized_keys`. The harness adds no credential store. `ssh` runs through `which_outside_cwd` with list
  argv, `-T`, `-o BatchMode=yes`, `-o ConnectTimeout=`, `--` before the remote command; the host alias must
  match `[A-Za-z0-9._@-]+` and may not begin with `-`; remote stdout/stderr is hostile — capped, through
  `_visible`, redacted at the channel boundary with the laptop's secrets.
- **Discovery.** None. `init` writes no remote peers; the person writes the peer (`name`, `base_url`,
  `provider_type`, `api_key`, `remote_control`) once. `--script` for a remote peer prints the `ssh` line.
- **What would need the critics' passes before scheduling:** which channels may trigger a remote swap
  (`/model` from Discord exists; proposed: terminal and ACP only); what the laptop's `/model` line says
  during a remote load; whether a remote `pending_preset` should persist across the Spark's reboots (today
  it is read, never removed by `serve`); redaction of ssh output; the owner's device list (proposed: any
  host the laptop's `ssh` can reach, since the remote's own trust gates every launch).
