# Spec 22 — The Fit Planner, the Harness Window Floor, and a Clean First Switch

**Status:** Draft, revision 2, 2026-10-17 — revision 1's four critics (design, security, ux-friction, parsimony) all REVISE; every major and every parsimony finding folded here, by the owner's instruction to revise once
**Version target:** 0.19.1
**API version:** `PLUGIN_API_VERSION` unchanged ("1")
**Kind:** a delta spec. It changes specs 16, 20, 08, 10, 13 and 19 in the places §12 lists, and nothing else in them.

---

## 1. Problem

The owner's job, in his words: *"it just needs to represent what someone who does uv install for localharness
and has some models then runs init or start or whatever."* And: *"auto sense for model configs needs way better
optimization."* And: *"also validate the gemma swap too."*

Three episodes on the reference box this morning (RTX 5090 32 GB, 61 GiB RAM, llama.cpp `43fe9c642`, vLLM
0.31.0) show four defects, each a harness gap.

1. **The fit prices one case: every weight on the GPU.** `/model Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`
   launched with no `-c`, llama-server's own `--fit` shrank the window to its 4,096 floor, the swap refit the
   budget to 4,096, and the first turn failed with `EMERGENCY context floor: utilization 187.0%` and `Active step
   and the complete prompt cannot fit with reply reserve`. The owner: *"the actual issue is that it doesnt
   understand hey this is large but its an MOE and the system has VRAM and DRAM for the weight offload right."*
   The same fit also prices KV on every layer of a hybrid model that keeps KV on one layer in four (§2.2), so
   the dense 27B's fit of 66,437 is a quarter of what its free memory holds.
2. **Nothing stops a window the harness cannot run in.** No number says how small a window the harness needs,
   so a 4,096-token launch, a 4,096-token refit and a 4,096-token child all pass silently until a turn fails.
3. **A child keeps the window its runner was built with.** After the swap back to the 27B (served 66,560), a
   delegated `web-researcher` ran at 4,096 and fired the emergency floor twice: the runner holds an int taken at
   `start`, and the swap changes only the orchestrator's live budget.
4. **`/model` lists names, not launches, and promises what the state forbids.** With `llm.service` (the owner's
   own unit) serving, the list offered four HF repos "(downloaded — switching restarts the managed server)": a
   bf16 gemma that cannot fit the GPU, an embedding model, and two NVFP4 checkpoints that fit; `/model 2`
   answered `is not a GGUF file`; two down peers were listed twice ("skipped … unreachable", then "cold … /model
   launches it on the GPU"); and the cold line's launch was refused by the foreign unit. The owner: *"so
   basically i had one model option and it didnt even work … did it even auto gen?"*

## 2. Ground truth (what exists)

Line numbers against the tree at `4776013d` and llama.cpp `43fe9c642` (2026-10-06).

### 2.1 The code each episode ran

| Piece | Where | State |
|---|---|---|
| The fit | `provider/autosense.py:212–240 fit` | `usable = free − weights − compute_margin_ratio × weights`, None when not positive; None when `offload != "gpu"` or the runtime is not llama.cpp |
| What `fit` reads | `autosense.py:32–35 FIT_KEYS` | `expert_count` is read and never priced |
| KV per token | `autosense.py:142–151 kv_bytes_per_token` | `block_count × kv_heads × head_dim × (K + V bytes)`: every layer, attention or not |
| The draft share | `provider/tuning.py:107–112 draft_share` | `1 + nextn / block_count` |
| Residency | `autosense.py:159–180 offload_kind` | from `extra_args` flags only (`-ot`, `--n-cpu-moe`, `-ngl`), never from the model's shape |
| Weights bytes | `autosense.py:193–209 weights_bytes` | file sizes of every shard plus `mmproj`: input-layer tables, which llama.cpp keeps on the CPU, are counted as GPU bytes |
| Prompt cache | `autosense.py:243–253 cache_ram_mib` | host weights are the whole GGUF when `offload != "gpu"`, else 0 |
| Launch sizing | `provider/lifecycle.py:532–565 _size_llamacpp` | no fit → mode `fit-by-server`, no `-c`, no `--fit-ctx` |
| The header reader | `agent/gguf_header.py:123–175` | KV section only; `n_tensors` is read and skipped (`:141`); tensor infos are never read |
| The swap's refit | `cli/repl.py:2859–2932 _refresh_token_counter` | `ctx.max_context_tokens = window` for the probed served window, any size |
| The start-time window guard | `cli/start_cmd.py:2484–2512` | refuses only a window whose `response_reserve` is ≤ 0 (≤ 1,024 tokens) |
| The reply reserve | `agent/context.py:35–74 response_reserve`, `:32` | 4,096 on a window ≥ 12,288, else `max(256, window // 8)` |
| The fixed prompt | `agent/context.py:2190–2193` | tool schemas estimated by `TokenCounter.estimate_messages`; the system prompt is message 0 of history |
| Compaction's target | `agent/context.py:215, 230` | a fire lands usage at 0.60 of the effective limit; it triggers at 0.80 |
| The child's window | `cli/start_cmd.py:3031, 3055` → `agent/subagent.py:1770, 1783, 1880–1898 _make_child_ctx` | `make_explore_agent_runner(max_context_tokens=agent_config.context.max_context_tokens)`: an int captured once; every child's `ContextManager` takes it; the nested runner (`:1911–1921`) and the cruncher (`:1625`) inherit it |
| The model tree | `cli/repl.py:1898–1905` (`downloaded = registry + hf_cached`), `:1996–2010` (the downloaded and cold suffixes), `:2462–2474` (the "skipped" notes) | HF repo ids from `server.list_cached_models` (`provider/server.py:571–582`), unresolved; a down lifecycle peer gets a "skipped" note and a cold entry |
| The library's checkpoints | `provider/model_library.py:288–310` | a snapshot is text generation when an architecture ends in `ForCausalLM` / `ForConditionalGeneration`; `Qwen/Qwen3-Embedding-0.6B` declares `Qwen3ForCausalLM` and passes |
| The GGUF refusal | `cli/repl.py:145 NOT_A_GGUF_LINE` | any non-`.gguf` target on a llama.cpp primary |
| The foreign-unit refusal | `cli/workspace.py:686–701 incumbent_sentence`; `cli/start_serving.py:214, 251` | names the unit and a `systemctl --user stop`; `init`/`start` print only the attach form |
| The swap | `provider/lifecycle.py:1915 swap` | senses and plans the target inside `activate`, after `free_accelerator`; a failed launch stops the target and restores the incumbent (`_restore_incumbent`) |
| Runtime if-chains | `lifecycle.py:1260–1300`; `server.py:243, 246, 337, 343, 698, 781–809, 853–884`; `cli/library.py:188–1005` (eleven sites); `cli/serve_cmd.py:267`; `provider/detector.py:29`; `provider/refarch.py:21` | each compares `runtime` to a string |
| The capability probe | `cli/start_cmd.py:1243–1255`; `provider/client.py:970` | retries a transport failure, but an answer from a server still loading reads as "falling back to xml" |

### 2.2 The models on the box, measured

Read with a 60-line, read-only GGUF parser (KV section skipped, tensor infos read, each tensor's bytes the
distance to the next tensor's offset, the last one's to the end of the data section). "Input" is a tensor
llama.cpp tags `LLM_TENSOR_LAYER_INPUT` (`src/llama-arch.cpp:750–752, 957`: `token_embd`,
`per_layer_token_embd`), which it always places on the CPU (`src/llama-model.cpp:1607`, `dev_input =
cpu_dev`) and reads with `GET_ROWS`, one row per token. "Experts" is a tensor whose name holds `exps`, the set
llama.cpp's own fit moves to the CPU (`common/fit.cpp:618`). "GPU" is everything else.

| | Flash-Next IQ3_XXS (2 shards) | Swift-1.5 27B NVFP4-Q8mix | Qwen3.8-27B NVIDIA NVFP4 |
|---|---|---|---|
| Files | 75,966,073,120 B (70.75 GiB) + projector 907,543,520 B = **71.59 GiB** | 19,924,343,072 B (18.56 GiB) + projector 927,606,912 B | 21,467,224,576 B (19.99 GiB) + projector 927,607,488 B |
| Experts | **42,912,972,800 B (39.97 GiB)** | 0 | 0 |
| Input tables | **29,073,290,240 B (27.08 GiB)**, of which `per_layer_token_embd` 26.82 GiB | 1,350,860,800 B (1.26 GiB) | 2,542,796,800 B (2.37 GiB) |
| GPU (dense) | **3,968,785,920 B (3.70 GiB)** | 18,562,465,856 B (17.29 GiB) | 18,913,411,136 B (17.61 GiB) |
| Header | `qwen4exp`, `block_count 48`, `context_length 262144`, `expert_count 512`, `expert_used_count 10`, `full_attention_interval 4`, `head_count_kv 2`, `key_length 256`, indexer `head_count 4`, `key_length 128` | `qwen35`, `block_count 65`, `nextn_predict_layers 1`, `context_length 262144`, `full_attention_interval 4`, `head_count_kv 4`, `key_length 256` | same as Swift |
| Layers that hold KV | 12 (`attn_k` on 3, 7, …, 47); 36 linear-attention layers (`ssm_out`) hold a fixed recurrent state | 16 trunk layers (3, 7, …, 63) plus the MTP layer 64, which only the draft context holds | same as Swift |
| KV bytes per token, f16 | 12 × 2 × 2 × 256 × 2 = 24,576, plus the indexer's 12 × 256 × 2 = 6,144: **30,720** | 16 × 2 × 4 × 256 × 2 = **65,536**; draft +4,096 | same |
| KV bytes per token, q8_0 | 13,056 + 3,264 = **16,320** | **34,816**; draft +2,176 (share 1 + 1/16) | same |
| What the harness prices today, q8_0 | 52,224 (48 layers): 3.2× high | 141,440 × (1 + 1/65): 4.06× high | same |

The two hybrid rules come from llama.cpp itself: `src/models/qwen35.cpp:17–22` and `src/models/qwen4exp.cpp:148–156`
mark layer `i` recurrent when `(i + 1) % full_attention_interval != 0`, and an MTP layer is never recurrent.
The indexer cache is one key head of `indexer.key_length × 2` per attention layer and per token, with no V
(`src/llama-memory-hybrid-idx.cpp:53–66`).

What 71.6 GiB means on 31.8 GiB of VRAM and 61.2 GiB of RAM (`MemTotal 64,139,128 kB`; `MemAvailable` 52.9 GiB
when measured): 3.70 GiB must sit on the GPU, 39.97 GiB of experts must sit in RAM, and 27.08 GiB of per-token
tables are read row by row through `mmap` and need not be resident. The owner's own profile ran exactly that
split (`~/setup/llm/profiles/flashnext-llama.sh`: `-ngl 99 --n-cpu-moe 48 -c 32768`), measured in
`flashnext-BENCH.md`: 10.6 GiB of GPU used, llama-server RSS 44.8 GiB, 6.9 of 8 GiB swap in use, decode
23.8 t/s against 116 t/s for the 27B. So the box can honestly serve Flash-Next, with its experts in RAM, at about
a fifth of the 27B's decode speed, and with thin RAM headroom.

At the free figure of episode 3 (30.2 GiB), the 27B's GPU bytes and projector (18.15 GiB), today's 10% margin
(1.82 GiB) and the corrected q8_0 + draft KV (36,992 B) give 297k tokens, capped at the trained 262,144 (9.03 GiB of
KV); today's formula said 66,437.

### 2.3 llama.cpp's fit, as built

| Fact | Where |
|---|---|
| `--fit on` is the default; `--fit-ctx` (`fit_params_min_ctx`) defaults to 4,096; `--fit-target` defaults to 1,024 MiB per device | `common/common.h:481, 483, 486`; `common/arg.cpp:2871–2929` |
| `-c 0` sets the minimum to `UINT32_MAX` (no reduction) | `common/arg.cpp:1642–1650` |
| With `-c` unset, the fit first projects the whole model on the device and, on a deficit, **reduces the context** by linear interpolation between the trained window and `--fit-ctx`, rounded to 256 | `common/fit.cpp:393–445` |
| With `-c` set, the context is left alone ("context size set by user") | `common/fit.cpp:454–455` |
| Only then does it move every `ffn_*_exps` tensor to the CPU, then fill dense layers back to front | `common/fit.cpp:616–645, 671–` |
| Host RAM is checked only on a CPU-only machine (`nd == 0`); with a GPU, experts go to the CPU unpriced | `common/fit.cpp:317–327` |
| A hand-set `-ngl`, `--tensor-split` or `-ot` aborts the fit with a warning, and the server loads with the hand-set values | `common/fit.cpp:463–485, 894–896` |

So episode 2's 4,096 is the order of llama.cpp's two steps: it shrank the window for a 70.75 GiB projection before
it moved 39.97 GiB of experts off the GPU. A launch that carries its own `-c` skips the first step and keeps the second.

### 2.4 The HF cache on the box

| Snapshot | What it holds | Size | What it is |
|---|---|---|---|
| `DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking` | 2 safetensors shards, `model_type gemma4`, bf16, no quantization | 62,546,338,264 B (58.25 GiB) | a vLLM target whose weights exceed the GPU |
| `nvidia/Qwen3.8-27B-NVFP4` | 3 shards, `qwen3_5`, `hf_quant_config.json`, 16 of 64 layers `full_attention` | 21,921,697,280 B (20.42 GiB) | a vLLM target that fits |
| `ukisai/Swift-1.5-Qwen3.8-27b-NVFP4` | 5 shards, same shape | 21,921,695,568 B (20.42 GiB) | same |
| `Qwen/Qwen3-Embedding-0.6B` | `modules.json`, `config_sentence_transformers.json`, `1_Pooling/` | 1.11 GiB | an embedding model |

No snapshot holds a GGUF; a snapshot file is a symlink into the repo's `blobs/`, which today's walk never follows
(`agent/gguf_header.py:249–300`, `lstat`-filtered). vLLM keeps every weight on the GPU unless told otherwise, and
the parser table `provider/tuning.py:42–48 MODEL_TYPE_PARSERS` holds `qwen3_5 → (qwen3, qwen3_coder)` and
`gemma4 → (gemma4, gemma4)`.

### 2.5 Strata, as read

A read-only clone of `github.com/Niko1221/Strata` at `296b59d` (2026-10-14), never executed; paths are the clone's.

| Fact | Where |
|---|---|
| A C++/CUDA engine (`engine/strata`, a line protocol on stdout, no HTTP) under a Python HTTP server (`serve/server.py`), MIT; its `third_party/ggml` is llama.cpp's | `AGENTS.md:3–5`, `LICENSE:1`, `README.md:235`, `server.py:740–850` |
| Launched as `python serve/server.py --engine strata --config strata-<tag>.json --port P` from the clone root; the server spawns `engine/strata --serve …` and restarts a dead engine | `setup.py:4912–4913`; `server.py:~815–818` |
| Experts copied to pinned RAM, a VRAM expert cache (`--expert-cache auto`, 700 MiB left free), the per-token table read from SSD (`--ple-gguf`), MTP + suffix drafting (`--spec 4 --spec-min-p 0.5 --mtp`) | `setup.py:6560–6583, 6674` |
| Binds 127.0.0.1 by default; key from `--api-key` or `STRATA_API_KEY`; `/health` is unauthenticated | Strata's `SECURITY.md:46–51`; `server.py:5337–5341` |
| The served window is the engine's `--max-context`, reported as `/v1/models` `meta.n_ctx`, `/health` `max_context`, `/props` `default_generation_settings.n_ctx` | `server.py:843–845, 3203–3207, 5369–5386, 5607–5630` |
| SIGTERM or SIGINT on the server sends `QUIT` to the engine and exits; no pidfile | `server.py:6944–6965` |
| The installer `setup.sh` makes `.venv`, may `apt-get` under privilege escalation, downloads the engine from `releases/latest` with a size and SHA-256 check published by the same origin, then the MTP draft tensors | `setup.sh:9–69`; `setup.py:175, 181, 3317–3335, 6519–6538` |
| `--gguf-dir` uses existing shards in place; Swift's template is the owner's file names; Swift has no hash table | `setup.py:393, 416, 1716–1733, 5767, 6395–6401, 6487–6489` |

## 3. Goals

- G1. **A launch is priced from the model's tensor table**: dense weights, a compute margin and the KV of the
  layers that hold KV on the GPU; experts in RAM when the model has them and RAM holds them; `-c` from that, and
  llama-server's `--fit` left to place tensors. A model that fits nowhere is refused in one sentence, before
  anything stops.
- G2. **One number says how small a window the harness can run in**, computed from the live root, and no launch,
  swap, attach or child runs below it.
- G3. **A child's window is the parent's window now**, read at spawn.
- G4. **`/model` numbers only what this box can switch to now**, each entry with its runtime, where its weights
  live and its window; it never lists a peer twice, and says once what a foreign unit means.
- G5. **The runtimes are one table**, so a runtime is one row; Strata (`runtime: strata`) is the first new row, an
  opt-in second preset for the Flash-Next family until its live rows pass.
- G6. Every open question spec 20's live run left is decided (§16).

## 4. Non-goals

- No `-ngl`, `--n-cpu-moe`, `-ot` or `-t` from the harness (spec 20 §4 stands): placement stays the server's.
- No vLLM CPU offload (`--cpu-offload-gb`): a vLLM checkpoint either fits the GPU or is not offered.
- No SSD tier of the harness's own; Strata's is Strata's.
- The harness never runs Strata's installer (§11.5).
- No change to the one-heavy lock, the ladder, autostart, the interview or the trust kinds of spec 20.

## 5. Config shape

```yaml
autosense:                        # existing machine-only section
  compute_margin_ratio: 0.10      # now of the GPU-resident bytes, not of the file
  cache_ram_ratio: 0.40           # now of MemAvailable less the experts priced into RAM
  strata_dir: null                # new: the Strata install; unset = sensed (§11.5)
```

`strata_dir` is the one new key, and the `strata` runtime brings two record fields (§11.1).
The floor (§8) is derived, never configured, because a configured floor would be a second owner beside
`response_reserve`.
The lifecycle keeps one small state file, `<config_dir>/server-floor.json` (§8.2), a cache of the last full floor.

## 6. The runtime table

`provider/runtimes.py` (new) holds `RUNTIMES: Mapping[str, Runtime]`, one row per runtime, and lands before any
other change in this spec.

| Column | What it answers | Today's if-chains it replaces |
|---|---|---|
| `strategy(spec)` | the `LifecycleStrategy` that launches and stops it | `lifecycle.py:1260–1268 strategy_for` |
| `unit_servable(spec)` | whether the autostart unit can serve it | `lifecycle.py:1278–1286` |
| `label(spec)` | the kind word in lines (`llama.cpp`, `vLLM binary`, `docker`, `Ollama`, `LM Studio`, `Strata`) | `lifecycle.py:1289–1300 unit_kind`; `server.py:343` |
| `key_env` | the environment name that carries the key | `server.py:337 KEY_ENV` |
| `probe` | the served and trained window from the live server | the runtime branches of `agent/context.py:1175 probe_window` |
| `price(spec, sensed, autosense)` | the plan's window and placement, or a refusal | `lifecycle.py:532–565 _size_llamacpp` and the vLLM "not computed" |
| `facts` | reasoning and tool parsers, where sampling comes from, what `/effort` maps onto | `tuning.py:42–48 MODEL_TYPE_PARSERS` and the runtime tests in `tuning.derive` |
| `record_fields` | the record fields valid for it (`_launch_target_present`'s gating) | `config/models.py` runtime validators |

The other runtime comparisons read the row too: `server.py:243, 246, 698, 781–809, 853–884`,
`cli/library.py:188, 300, 305, 311–312, 354, 582–586, 890, 930, 1005`, `cli/serve_cmd.py:267`,
`provider/detector.py:29`, `provider/refarch.py:21`, `config/trust.py`'s launch-key gating and `tuning.py`.
`LifecycleStrategy` methods stay as they are (`activate`, `stop`, `liveness`, `launch_identity`); `serve_command`,
`wait_ready` and `probe_window` stay shared free functions that dispatch on the row, not on a string.
Each runtime's price is one function in one module: llama.cpp's and vLLM's in `provider/autosense.py`, Strata's
in `provider/strata.py`; `MODEL_TYPE_PARSERS` stays in `tuning.py` as vLLM's `facts`.
The table is a refactor with no behaviour change, landed and tested on its own before §7.

## 7. The fit planner

### 7.1 The tensor table

The bounded reader moves from `agent/gguf_header.py` to `provider/gguf_header.py`, since the fit, discovery and the
library that use it are the provider's; `agent/gguf_tokenizer.py` and `agent/context.py`'s trained-window probe
import it from there.
It gains `read_tensor_table(path) -> TensorTable(entries, data_bytes)`: the KV section is walked with the existing
bounded cursor (values skipped), then `n_tensors` tensor infos are read under three new caps (`MAX_TENSORS =
65,536`, a name ≤ 256 bytes, `n_dims ≤ 4`).
A tensor's bytes are the distance from its offset to the next offset in sorted order, the last one's to the end of
the file, after the data section start aligned to `general.alignment` (default 32).
Offsets that are not strictly increasing, not aligned, or past the end of the file make the table unreadable.
A split model's table is the union of its shards', each shard opened through `open_regular`.
`classify(name) -> "input" | "experts" | "gpu"` is one pure function: the four `LLM_TENSOR_LAYER_INPUT` names
(`token_embd`, `per_layer_token_embd`, `pos_embd`, `token_types`) are input, a name holding `_exps` is experts,
everything else is GPU.
`kv_layers(entries, header) -> (main, draft)` counts the blocks that hold `attn_k`, or hold `attn_qkv` and no
`ssm_` tensor, split at `block_count − nextn_predict_layers`.
An indexer (`blk.N.indexer.k_proj`) adds `indexer.key_length × 2` K-only elements per token on its layer.
Sliding-window layers are priced at the full window, which errs high.

The table is cross-checked against the header before it prices anything, and a check that fails makes the model
"not split" (priced as today, the whole file on the GPU, which errs high):
- `1 ≤ main ≤ block_count`, and with `full_attention_interval` set, `main` equals the trunk's block count divided
  by the interval (the count llama.cpp's own rule gives, `qwen35.cpp:17–22`);
- experts are counted only when the header sets `expert_count`;
- input tables are at most 50% of the file (Flash-Next's are 38%, the 27B's 7–12%).

`Sensed` (spec 20 §6.0) gains `split: TensorSplit(gpu, experts, input, kv_main, kv_draft, indexer) | None`, read
once by `sense_launch`.

### 7.2 The price

```
gpu_bytes   = split.gpu + mmproj
margin      = compute_margin_ratio × gpu_bytes
kv_per_tok  = kv_main × 2 × kv_heads × head_dim × bytes(K, V) + indexer bytes, × draft_share when drafting
draft_share = 1 + kv_draft / kv_main
fits        = min(trained, (free_vram − gpu_bytes − margin − experts_on_gpu) // kv_per_tok), rounded down to 256
ram_room    = MemAvailable − unified_reserve_ratio × MemTotal
cache_ram   = min(cache_ram_ratio × (MemAvailable − experts_in_ram), ram_room − experts_in_ram), clamped at 0
```

Residency comes from the model's shape, not from flags.
A model without `expert_count` keeps everything on the GPU, as today.
A model with `expert_count` is priced experts on the GPU first, and experts in RAM when that does not reach the
floor (§8); the second placement is the one llama-server's own fit reaches by moving `ffn_*_exps`
(`common/fit.cpp:616–645`), so the harness's `-c` and llama-server's placement agree.
Experts in RAM need `experts ≤ ram_room`; an unreadable `/proc/meminfo` makes `ram_room` unknown, which refuses.
Input tables are priced against neither memory: llama.cpp reads them through `mmap` one row per token.
`offload_kind` keeps its flag reading for a record that places tensors by hand (`-ot`, `--n-cpu-moe`), which stays
"not computed".
A record with a hand-set `-ngl` on a model whose GPU bytes plus experts exceed the free VRAM is refused, because
`-ngl` turns llama-server's fit off (`common/fit.cpp:463–464, 894–896`) and every expert would load on the GPU:
`<name> sets -ngl in server.extra_args, which turns llama-server's --fit off, and its experts do not fit the GPU —
drop the -ngl, or add --n-cpu-moe to extra_args.`

The plan emits `-c fits` whenever it priced one, so llama-server never reduces the context and only places tensors
(`common/fit.cpp:454–455`).
`fit-by-server` survives only for a model whose table and header cannot be read, and that launch carries
`--fit-ctx <floor>` so llama-server's reduction stops at the floor, not at 4,096.
The halve-once retry (spec 16 §6.2) halves to `max(floor, fits // 2)` and is skipped when `fits // 2 < floor`.
With `-c` set, llama-server cannot correct an optimistic `fits`: a margin too small for the compute buffers spills
a dense layer to the CPU (slower, not failed). The margin is the open risk of this plan (§16), and the retry and
the live rows are its only guards today.

On this box: the 27B at 30.2 GiB free fits 262,144 (297k before the cap; today's formula said 66,437).
Flash-Next at 30.2 GiB free and 52.9 GiB available fits 262,144 with experts in RAM: `ram_room` is 52.9 − 9.2 =
43.7 GiB, the experts take 39.97, and `cache_ram` gets the 3.7 GiB left.

### 7.3 The refusal

When neither placement fits, the plan is a refusal, raised before anything is stopped:

```
Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS cannot run here: its experts need 40.0 GiB of RAM and only 22.2 GiB is free beside the system's reserve — close something that holds RAM, or pick a smaller quant.
Qwen3.8-27B-Nvidia-NVFP4 cannot run here: it needs 17.6 GiB of GPU memory plus a 15,360-token window and 12.0 GiB is free — stop what holds the GPU, or pick a smaller quant.
```

### 7.4 vLLM

A vLLM record is priced from its checkpoint (spec 16 §6.2's "not computed" is replaced), in `provider/autosense.py`.
The weights are the snapshot's `*.safetensors` bytes; the KV layers are `config.json`'s `layer_types` entries that
read `full_attention` (else `num_hidden_layers`), with `num_key_value_heads` and `head_dim` from `text_config` when
present; the KV type is bf16 unless `--kv-cache-dtype` is in `extra_args`; the budget is vLLM's own
`gpu_memory_utilization` (0.90 unless set) of the GPU total.
Weights above that budget make the checkpoint "not offered" (§10.2); otherwise the plan emits `--max-model-len
fits` through the same `context` keyword, vLLM's memory profile at load is the verifier, and a load that dies
takes the halve-once retry.
The two NVFP4 checkpoints price at 0.90 × 31.8 − 20.42 − 2.04 = 6.16 GiB of KV at 65,536 B a token: 100,864 tokens.

### 7.5 The window line

The launch line says the window and where the weights live:

```
window: 262,144 · experts in RAM (40.0 GiB) · dense, projector and KV on the GPU (8.5 of 30.2 GiB free)
window: 262,144 · all on the GPU (27.2 of 30.2 GiB free)
window: 100,864 · vLLM · all on the GPU (26.6 of vLLM's 28.6 GiB)
```

The per-tensor figures, the token tables read through `mmap`, the prompt cache and the RAM reserve are
`doctor`'s `Tuning:` row, not the launch line.
`FIT_BY_SERVER_LINE` gains the floor: `window: served by llama-server's --fit (at least 15,360) · served N`.

## 8. The harness floor

### 8.1 One number

```
harness_floor(fixed_tokens, max_response_tokens) =
    the smallest W in 256, 512, 768, … such that W and every larger window satisfy
    fixed_tokens ≤ COMPACTION_TARGET_USAGE_FRACTION × (W − response_reserve(W, max_response_tokens))
fixed_tokens(role) = estimate(system prompt) + estimate(tool schemas)
```

`agent/context.py` gains `harness_floor`, pure, beside `response_reserve`.
`response_reserve` is not monotone (it jumps from `W // 8` to 4,096 at 12,288, `context.py:56–74`), so a window
above the smallest one that fits can fail the test; the floor is the least window above which none does, found by
scanning up from 12,288 and stepping back down, so no closed form holds.
The fixed part is what compaction can never remove, and compaction lands usage at 0.60 of the effective limit
(`context.py:215`), so a fixed part above that fraction leaves no turn compaction can rescue.
`estimate` is `TokenCounter.estimate_messages`, the estimate `build_messages` already uses for the schemas
(`context.py:2190–2193`), so the floor needs no server.
It is built on `response_reserve`, so it never lowers the reserve and never subtracts it twice.
The root's floor gates launches, swaps and attaches; each role's own floor gates its spawn.
From the episodes, the root's fixed part plus a short first message was about 6,700 tokens (187% of 3,584) and the
`web-researcher`'s about 4,370 (122% of 3,584): floors of 15,360 and 8,448; the live run records the real figures (§15).

### 8.2 Where it is computed

- **In a session** (the REPL's `/model`, `localharness model` beside a live session): from the live root's system
  prompt and registry, at plan time.
- **At `start`, before its launch** (the registry does not exist yet, `start_cmd.py:2266` against `:2622`): from the
  root agent's rendered system prompt and the core tool schemas, which are code; the MCP tools' share comes from the
  cache when it holds one for this config, and is otherwise missing.
- **At `start`, after the registry exists**: the full floor, written to the cache and checked against the served
  window (§8.3).

The cache, `<config_dir>/server-floor.json`, holds `{root, roles: {name: floor}, at}`, is written by
`lifecycle.record_floor` through `atomic_write_text`, joins `PROTECTED_CONFIG_DIR_ENTRIES`, and is clamped on read
to `[1,024, trained]`; it only ever raises the live figure (`max(live, cached)`), so a planted or stale value can
refuse a launch the person sees, never let a smaller window through.

### 8.3 Where it applies

| Site | Behaviour | What the person sees |
|---|---|---|
| Plan (`plan_launch`, every surface) | a priced `fits`, a record `context` or a Strata config window below the root floor is a refusal; a `fit-by-server` launch carries `--fit-ctx <floor>` | `<name> fits only a 9,216-token window here and the harness needs at least 15,360 — free GPU memory, or pick a smaller model.` |
| Swap, before it commits (`provider/lifecycle.py:1915 swap`) | the target is sensed and planned **before** `free_accelerator`, against the memory the swap will have: free now plus what the verified incumbent's pid holds (`nvidia-smi --query-compute-apps=pid,used_memory`), else the card's total less 1 GiB; a refusal stops nothing | the plan's refusal sentence, then `qwen-27b-swift-1.5 keeps serving.` |
| Swap, after the launch (the backstop, same function) | `swap` takes `floor=` and treats a served window below it as a failed launch: its existing failure branch stops the target through its verified stop and restores the incumbent (`_restore_incumbent`); `_refresh_token_counter` (`repl.py:2859`) still never aborts a swap, and runs only after `swap` succeeded | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS served a 4,096-token window, below the 15,360 the harness needs — qwen-27b-swift-1.5 serves again.` |
| `start` and attach (`start_cmd.py:2484–2512`) | the full floor is compared with the served window before the first turn; below it, `start` exits 1 (the ≤ 1,024 branch becomes this one) | `qwen-27b-swift-1.5 on llm.service serves a 4,096-token window and the harness needs at least 15,360 — raise its -c, or stop the unit and let the harness launch it.` |
| Spawn (§9) | a child whose own floor exceeds the live window is not run; the `agent` tool returns an error result | `web-researcher needs an 8,448-token window and the server serves 4,096; not delegated.` |

## 9. Children's budget

The handoff site is the `max_context_tokens=` argument at `cli/start_cmd.py:3055`, in the
`make_explore_agent_runner(...)` call that opens at `:3031`.
It becomes `window=lambda: ctx_mgr.max_context_tokens`, the root's live `ContextManager` (built at `start_cmd.py:2843`).
`make_explore_agent_runner` (`agent/subagent.py:1770`) takes `window: Callable[[], int | None]` in place of the int.
`_make_child_ctx` (`subagent.py:1880–1898`) calls it at each spawn; the nested runner (`:1911–1921`) receives the
same callable, and the cruncher reads its window from the child context it is handed (`:1625`).
The child's window is read once, at spawn: a child lives inside one turn and a swap runs between turns.
A structural test pins that no call to `make_explore_agent_runner` passes an int window.

## 10. `/model`: entries are launch plans

### 10.1 One builder

`cli/model_ops.py` gains `plan_entries(harness, *, live, library, memory, floor) -> Entries`, the one place the
list is built and priced (off the event loop, one `free_memory` reading); the REPL's `/model`
(`repl.py:1898–2010, 2462–2474`) and `localharness model` both call it and only render.
An entry is a file or a checkpoint, never a name: a file with two records (a llama.cpp and a Strata preset) is one
entry, shown with the record the person landed on last (`last_served`), else the preferred runtime (§11.3); the
other record stays addressable by name (`/model <name>-strata`).
The library resolves an HF snapshot by its contents (`provider/model_library.py`):

| Snapshot holds | Entry | Plan |
|---|---|---|
| a `.gguf` | llama.cpp, typed by `discover` | §7.2 |
| `*.safetensors` and a `config.json` | vLLM, parsers from its `facts` row | §7.4 |
| `modules.json`, `config_sentence_transformers.json`, a `*_Pooling/` directory, or no text-generation architecture | none | — |

An entry that is not yet a record shows `(not set up)`; choosing it runs spec 20 §11's one question for a person
(`Set it up and switch?`), and off a terminal says `localharness model --setup sets it up`.

### 10.2 The lines

With the owner's unit serving:

```
Models:
▸ llama.cpp · 127.0.0.1:8080 · llm.service (yours)
  1. qwen-27b-swift-1.5  (serving)  ● [active]  [129.5 t/s measured]
▸ available once llm.service is stopped
     Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS  llama.cpp · experts in RAM · 262,144
     Qwen3.8-27B-Nvidia-NVFP4  llama.cpp · 262,144
     nvidia/Qwen3.8-27B-NVFP4  vLLM · 100,864 (not set up)
     ukisai/Swift-1.5-Qwen3.8-27b-NVFP4  vLLM · 100,864 (not set up)
  not offered: DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking — 58.3 GiB of bf16 weights, and vLLM keeps them all on the 31.8 GiB GPU
```

With it stopped, the second group is `▸ this box can launch`, its entries are numbered 2–5 and switch, and the
`not offered` line stays.

- The group suffix says who owns the server: `· the harness`, `· autostart`, `· <unit> (yours)`.
- Only what can be switched to now is numbered; nothing numbered is refused when picked.
- "(downloaded — switching restarts the managed server)" and "(cold on … — /model launches it on the GPU)" are gone.
- A down peer with a `lifecycle` is one entry; a "skipped … unreachable" note stays only for a peer the harness
  cannot launch.
- The window figures are the plan's, against the memory free beside what serves when the harness owns it (spec 16's
  `attached_served` rule), against the card's total otherwise.
- `/model <name>` typed for an unnumbered entry prints its reason (the foreign-unit sentence, or the `not offered`
  reason as a sentence) and launches nothing.

### 10.3 The foreign-unit sentence

One builder in `cli/workspace.py`, beside `incumbent_sentence` and with its escaping (`visible_text` on every name,
`shlex.quote` on the unit in the command, `workspace.py:682–702`), makes the sentence:

```
llm.service serves qwen-27b-swift-1.5 and is yours, so the harness lists models but never stops it; systemctl --user disable --now llm.service hands the GPU to the harness (localharness model --autostart on brings your choice up at login).
```

The verb is `disable --now` for an enabled unit and `stop` for one that is not.
`/model`, `init` and `start` print it on change only, keyed on `(unit, served)` in the speed ledger's on-change
record (spec 16 §6.1's mechanism): the first time, and again when the unit or what it serves changes; otherwise the
group header's `· llm.service (yours)` and `available once llm.service is stopped` say it.
For an owner whose stopped unit `init` imported, `init` alone prints, once:

```
llm.service is yours and was imported as the server: record; from now on the harness launches qwen-27b-swift-1.5 itself and never touches the unit.
```

with ` — systemctl --user disable llm.service keeps it from starting at login beside the harness` appended only
when `unit_file_state` reads `enabled`.

### 10.4 The gemma swap

The gemma snapshot resolves to a vLLM plan whose weights exceed the budget, so it is the `not offered` line.
`/model DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking` prints, and stops nothing:

```
DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking cannot run here: vLLM keeps its 58.3 GiB of bf16 weights on the GPU and this one has 31.8 GiB — a GGUF quant (llama.cpp keeps part of it in RAM) or an NVFP4 build of it would fit.
```

The session stays on the model that served, and the next turn runs on it.
On a box whose GPU holds it, the same entry is numbered, set up on the way, launched with `--reasoning-parser gemma4
--tool-call-parser gemma4 --enable-auto-tool-choice --max-model-len <fit>`, and a turn runs after the swap.
This is a named live row (§15).

## 11. Strata: the MoE server (`runtime: strata`)

Strata keeps every expert pinned in RAM, caches hot experts in VRAM, reads the per-token tables from SSD and
drafts with MTP plus suffix matching (§2.5), the placement this spec's llama.cpp plan only approximates.
It is one row of `RUNTIMES` and one module, `provider/strata.py`, with one entry point, `StrataStrategy`.
Nothing in this section is measured on the box until the §15 live rows run, so Strata is opt-in (§11.3).
Strata is third-party and serves one family, which makes it a plugin candidate; the owner ruled it a core runtime.

### 11.1 The record

```yaml
extra_endpoints:
  - name: Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-strata
    base_url: http://127.0.0.1:8083/v1
    provider_type: strata                       # new provider type
    gpu: true
    lifecycle:
      runtime: strata                           # new runtime
      gpu: true
      engine_dir: ~/src/Strata                  # new, strata only: the install root
      engine_config: strata-swift-IQ3_XXS.json  # new, strata only: the config its setup wrote, relative to engine_dir
      model: ~/models/gguf/flashnext/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
      mmproj: ~/models/gguf/flashnext/mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf
      served_name: swift-1.5-iq3_xxs
      port: 8083
      # context: unset = the --max-context Strata's setup wrote, checked against the floor (§8)
```

- The interpreter is derived, `<engine_dir>/.venv/bin/python`, never a field: Strata's venv is its own.
- The host is 127.0.0.1 always: `bind_all` and `extra_args` are refused for `runtime: strata` by the row's
  `record_fields`, since the engine's arguments live in the config file and Strata's API has had no outside audit
  (Strata's `SECURITY.md:67`).
- `engine_dir` and `engine_config` join the `launch` keys (thirteen with them); the config file's bytes are one
  `launch` entry (`lifecycle.engine_config`), as spec 16 §8 records `server.config_ini`; and the code that runs is
  fingerprinted too: the SHA-256 of `serve/server.py` and of `engine/strata` are `launch` entries
  (`lifecycle.strata_code`), re-hashed at launch only when a file's `(size, mtime_ns, inode)` differs from the
  recorded triple. A changed file is withheld like any machine-file edit and asked about at the next terminal
  `start`. The rest of the tree (`serve.*` modules, the venv's site-packages, `~/.config/strata/settings.json`)
  is a named residual (§13).
- The record maps onto the owner's shards through the setup's `--gguf-dir ~/models/gguf/flashnext --family swift
  --model IQ3_XXS`: Swift's template `Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf`
  (`setup.py:393`) is exactly the owner's file names, and with `--gguf-dir` the shards are used in place.
- A record whose `model` is not the config's `--native` path is refused:
  `<name>'s Strata config serves another file than its model: path — re-run Strata's setup with --gguf-dir, or fix the record.`
- **Trust consequence.** Swift has no pinned SHA-256 in Strata's installer (only Unsloth does,
  `setup.py:416, 6487–6489`), so Strata checks only that the shards are present and the tensor directory complete
  (`setup.py:6395–6401`). The model bytes Strata serves are the bytes in that folder, as for the llama.cpp record:
  both records fingerprint the path, not the bytes (spec 20 §12's residual, unchanged).

### 11.2 The strategy

| Concern | Behaviour |
|---|---|
| Launch | argv `[<engine_dir>/.venv/bin/python, <engine_dir>/serve/server.py, --engine, strata, --config, <copy>, --port, <port>, --host, 127.0.0.1]`, cwd `engine_dir` (`server.py` imports `serve.*` from it); never `--open`, never `--api-key` on argv |
| The config copy | `<config_dir>/strata/<port>.json`, written by an executor only, from the recorded `engine_config` bytes (`json.loads` under a cap), through an **allowlist**: top-level `model_name` (set to `served_name`), `sampling` (filled from the GGUF's `general.sampling.*` when absent), `reasoning_effort`, `parallel`, `gpu` (an int), and the engine `args`, whose flags must be among `--pack --native --ple-gguf --expert-profile --expert-cache --prefill --spec --spec-min-p --mtp --max-context --kv --kv-resident --vram-reserve-mib --mmap-experts --resident-experts --resident-budget-gib`, every path argument resolving under `engine_dir`, Strata's data directory or the record's model folder; the harness sets `host` 127.0.0.1, `port`, `cwd` = `engine_dir` and `log` = `<config_dir>/strata/<port>.log`; any other key (`env`, `api_key`, `cors_origins`, `mcp`, an engine path) or flag refuses the launch, naming it. The key names follow `setup.py:6574–6583, 6674, 6683` and are confirmed against a real config at implementation. `<config_dir>/strata/` joins `PROTECTED_CONFIG_DIR_ENTRIES` |
| Environment | `child_environment` over the allowlisted base; the key as `STRATA_API_KEY` when keyed; `STRATA_SKIP_SHA256`, `STRATA_PREBUILT_URL`, `STRATA_SOURCE`, `STRATA_EXECV` and `HF_ENDPOINT` join the blocklist |
| Identity | `LaunchIdentity(exe_names={"python*"}, port=<port>, argv_contains=("<engine_dir>/serve/server.py", "--engine", "strata", "--config", "<copy>"))`; the copy path is per port; the engine child (`engine/strata --serve`) is never an identity |
| Readiness | `wait_ready` treats a refused connection as loading (the socket opens only after the engine's `READY`, `server.py:843, 6918`) and polls `GET /health` until 200 with `"loaded": true` and `"service": "strata"` (`server.py:5337–5341`); liveness is the verified parent; the bound is Strata's own 900 s (`STRATA_ENGINE_READY_S`, `server.py:268`) plus a minute; a parent that exits is a failure with the log tail |
| Served window | the row's `probe`: `/v1/models` `data[0].meta.n_ctx` with the key (`server.py:5369–5386`), else `/health` `max_context`; trained from the GGUF header through `resolve_gguf_path` |
| Retry | one, on a load death: the copy's `--max-context` halved, never below the floor; remembered through `confirm_launch` as for llama.cpp (spec 20 §6.4), with `context` the only dimension; a second death ends with the log tail and `Strata could not load <name> twice; its low-RAM setup (setup.sh --low-ram) or the llama.cpp preset may.` |
| Stop | SIGTERM to the parent, a wait for its exit (the engine's `QUIT` frees the pinned RAM, `server.py:6944–6965`), then SIGKILL to the group after the grace; before every signal the pid is re-proven from `/proc` (exe and argv against the identity, start time against the pidfile), never taken from the pidfile alone; the engine child is never signalled first, since the server restarts a dead engine |
| Logs | the parent's stdout and stderr into the harness's server log, the engine's stderr into the copy's `log`; `log_tail` reads both, redacted at the channel boundary |
| `--script` | a here-document that writes the copy, then `cd <engine_dir> && exec .venv/bin/python serve/server.py --engine strata --config <copy> --port <port> --host 127.0.0.1`, the key as a `STRATA_API_KEY=<key>` placeholder |
| Autostart | the row's `unit_servable` is true; `serve` (spec 20 §9.3) `chdir`s to `engine_dir` before it `execve`s the venv's python |

The `facts` column, one row per runtime:

| Runtime | Reasoning and tools | Sampling | `/effort` |
|---|---|---|---|
| llama.cpp | server-side (`--jinja`, `--reasoning-format auto`) | the GGUF's, applied by llama-server (spec 20 §7) | spec 16 §7.4 |
| vLLM | `MODEL_TYPE_PARSERS` | `--generation-config auto` | spec 16 §7.4 |
| Strata | server-side parser and chat-template tools; field names unverified (§11.6) | written into the copy's `sampling` from the GGUF header, because Strata is greedy when a request carries none (`DETAILS.md:1269–1271`) and spec 20 §7 sends none | `reasoning_effort` none / low / medium / high (`DETAILS.md:755–760`), a prompt instruction, not a token cap (`DETAILS.md:760–762`); the harness's own cap bounds the reply |

Strata serves one request at a time unless its config sets `parallel` ≥ 2 (`README.md:169`), so a delegated child
queues behind its parent on the same server.

### 11.3 Routing and the price

A file of the Flash-Next family (`general.architecture qwen4exp` and a name matching Strata's Swift template) gets
a Strata record only when an install is found (§11.5) and the person sets it up; the record is a second preset
beside the llama.cpp one, and never replaces it.
The preferred runtime for the file's single `/model` entry is llama.cpp until live rows 8–11 (§15) pass on the
reference box, and the person's last landing between the two wins after that.
`provider/strata.py`'s `price` checks what Strata does not size for itself:
- RAM: `experts + max(10 GB, unified_reserve_ratio × MemTotal) ≤ MemAvailable`, the 10 GB being Strata's own
  headroom rule (Strata's `MODELS.md`, "Will it fit?") and the larger of the two counted once, since both describe
  room for the OS; an unreadable `/proc/meminfo` refuses. On the box: 39.97 + 9.31 = 49.3 GiB of 52.9 available.
- VRAM: the dense layers, the projector and the window's KV at the config's `--kv` type must fit; the rest is
  Strata's `--expert-cache auto` beside its `--vram-reserve-mib` (700 MiB by default).
- The window: the config's `--max-context`, or the record's `context`, checked against the floor (§8).

The launch line: `window: 65,536 · Strata · experts pinned in RAM (40.0 GiB) · one request at a time`.

### 11.4 What Strata needs from the box

| Need | Strata's statement | The box |
|---|---|---|
| GPU | NVIDIA RTX 20–50, 12 GB+ (`README.md:58–71`); sm_120 needs a CUDA 13 build (`CMakeLists.txt:188–191`), and the prebuilt engine is CUDA 13.0 (`setup.py:3728`) | RTX 5090, 31.8 GiB |
| Driver, CPU | NVIDIA driver ≥ 580, x86-64 AVX2 (`docs/INSTALL.md:18, 21`) | driver 595.99 |
| RAM | ~43 GB of IQ3_XXS experts plus ~10 GB headroom (`MODELS.md`) | experts measured at 42.9 GB; 52.9 GiB (56.8 GB) available: about 3.6 GiB to spare |
| Disk | the model (in place with `--gguf-dir`), the ~5–6 GB MTP draft tensors from `Qwen/Qwen3.8-Flash-Next` converted to `mtp/rt` (`setup.py:6519–6538`), a tokenizer pack, the engine zip; `Strata-data/` beside the clone and `~/.config/strata/settings.json` (`setup.py:4351–4357`) | 1.2 TB free |
| The n-gram table | `per_layer_token_embd`, read from SSD through `--ple-gguf` (`setup.py:6560–6574`) | 26.82 GiB, in shard 1 (shard 2 holds none, measured) |
| Python | 3.10+, its own `.venv`, pinned requirements (`requirements.txt:4–15`), CUDA libraries from pip (~0.4 GB) | — |

### 11.5 Install and sensing

The harness never runs `setup.sh`: it can `apt-get` under privilege escalation (`setup.sh:46–53`), downloads its
engine from `releases/latest` (`setup.py:175, 181`), and on a later run may fetch or compile another
(`setup.py:5323–5341`).
The install is offered only to a person who picks the Flash-Next entry when its llama.cpp plan is refused or places
the experts in RAM (the measured slow placement, 23.8 t/s against 116), as one line after the launch line or the
refusal, on change only:
`Strata is built for this model; localharness doctor shows how to install it beside your shards.`
`doctor`'s `Strata:` row holds the command:

```
git clone https://github.com/Niko1221/Strata ~/src/Strata && cd ~/src/Strata && ./setup.sh --yes --no-start --family swift --model IQ3_XXS --gguf-dir ~/models/gguf/flashnext --context 65536 --vision no --no-browser
```

An install is sensed by `plan_entries` at every listing, at `autosense.strata_dir` when set, else at `~/Strata` or
`~/src/Strata`; it is an install only when `serve/server.py` and `.venv/bin/python` are regular files, `engine/strata`
is a regular executable, the directory and those files are owned by the user and not group- or world-writable,
and a `strata-*.json` there names in `--native` a file the library knows; each check is an `lstat` or
`open_regular`, and nothing in the directory runs to decide it.
A sensed install adds `· Strata available` to the Flash-Next entry; picking `/model <entry>-strata` then runs spec
20 §11's one question (`Set it up and switch?`), which shows the record and adopts it with its `launch` entries.

### 11.6 What is not known

- The reasoning and tool-call field names in responses and streamed deltas (`reasoning_content` or `reasoning`).
- How the Swift chat template behaves under `reasoning_effort`.
- The IQ3_XXS decode speed on a 5090: Strata reports 62 t/s on a 12 GB RTX 5070 (`README.md:24–49`) and IQ2_XS
  165–179 t/s on a community 5090; none is measured here.
- The first-start time and the OOM margin on 61 GiB with 43 GB pinned.
- The window this box can afford at IQ3_XXS (262,144 on 64 GB is "not measured" by Strata, `DETAILS.md:72`).
- Which release tag matches `296b59d`, so that the engine can be pinned (§13).

## 12. What changes in specs 16, 20, 08, 10, 13 and 19

| Spec, section | Was | Becomes | Why |
|---|---|---|---|
| 16 §6.2 | `fits` from the whole file, KV on every layer; "not computed" for vLLM and any offload | §7.2 and §7.4 | the fit was a quarter of the truth on the hybrid 27B and None on the MoE |
| 16 §6.1 | `fit-by-server` is a normal outcome, "never a refusal" | a fallback for an unreadable table only, with `--fit-ctx <floor>`; a model that fits nowhere is a refusal | a 4,096 window served silently is worse than a sentence |
| 16 §6.3 | host weights are the whole GGUF under any offload | the experts priced into RAM, with the reserve | the prompt cache shares RAM with the experts |
| 16 §5, 13 | the runtime and provider-type lists | the `RUNTIMES` table, with `strata` | §6 |
| 16 §7 (`/effort`) | levels enforced per runtime | Strata's map onto `reasoning_effort` | §11.2 |
| 16 §8 `/model` | four branches over names; a foreign incumbent is one imperative sentence | §10 | episode 1 |
| 20 §6.0, §6.4 | `Sensed` holds the header; the retry halves `-c` | `Sensed.split`; the retry halves to the floor; Strata halves `--max-context` | the planner needs the table; a window under the floor is the failure the retry exists to avoid |
| 20 §8.2 swap | the floor is not checked | §8.3: before the commit, then the backstop | episode 2 |
| 20 §8.2 resume clause, §9.3 receipt, §11.1 ranking, §14 (2) | as written | §16's decisions | the live run |
| 20 §13 | `rank_models` unchanged | its row names the two tie-breaks (amended in spec 20) | §16 |
| 08 "Reserve for model output" | `response_reserve` is the one reserve | unchanged; `harness_floor` sits beside it, built on it | the floor never lowers the reserve |
| 08 `ContextManager` | `max_context_tokens` is a construction argument | unchanged for the root; a child's comes from the root's live value at spawn | episode 3 |
| 10 "Slash commands" and `localharness model` | the listing's suffixes | §10's lines, built by `plan_entries` | episode 1 |
| 19 §3.3 call sites | the runner is built with the resolved window | with a window callable | episode 3 |

## 13. Security

- **No foreign process is signalled.** The foreign-unit sentence and every refusal are read-only, the unit's
  identity comes from `workspace.port_holder` and `unit_file_state` as today, and the commands in the sentence are
  the person's to run. The new stop path, Strata's SIGTERM-then-SIGKILL, signals only a process proven from `/proc`
  before every signal (exe and argv against the identity, start time against the pidfile); the pidfile and the
  config copy under `<config_dir>/strata/` are agent-writable claims, never proof.
- **The sentence builder escapes what it interpolates**: `visible_text` on the unit and served names, `shlex.quote`
  on the unit in the command, as `incumbent_sentence` does.
- **HF snapshot contents decide a launch, so the whole snapshot is validated.** The library walks every entry of the
  snapshot directory with `lstat`; a symlink whose realpath leaves that repo's `blobs/`, or any entry that is neither
  a regular file, a directory, nor such a symlink, makes the snapshot unlisted. That covers every file vLLM loads
  (`config.json`, `tokenizer*.json`, `chat_template*`, `generation_config.json`, `preprocessor_config.json`).
  `config.json` and `tokenizer_config.json` are read under `MODEL_CONFIG_CAP` with `json.loads` only; an `auto_map`
  in either lists the snapshot as `not offered: needs trust_remote_code`, and the harness never passes
  `--trust-remote-code`. A vLLM launch passes the verified snapshot directory, not the repo id, and carries
  `HF_HUB_OFFLINE=1`, so a cache entry never reaches the network.
- **Nothing unrecorded is launched.** An entry that is not a record is set up through spec 20 §11's shown record and
  its adoption, for a person only; the plan of a shown entry is recomputed from a fresh read at launch.
- **vLLM arguments come from the table.** `model_type` selects a `facts` row or nothing; no snapshot value reaches
  argv except the snapshot path and the computed `--max-model-len`.
- **A hostile tensor table can mis-price, within bounds.** The reader caps counts, names and dimensions and refuses
  non-monotonic or out-of-file offsets; names chosen to shrink the GPU price are bounded by the §7.1 cross-checks
  (attention layers against `block_count` and the interval, experts only with `expert_count`, input at most half the
  file). What remains is one optimistic `-c` for the person's own file: an OOM or a spilled layer at load, which the
  halve-once retry and the floor-bounded backstop end. It is a denial of service against the person's own launch,
  not an escalation.
- **RAM pricing fails closed.** An unreadable `/proc/meminfo` refuses experts in RAM and a Strata launch; both apply
  the reserve (§7.2, §11.3); unknown free VRAM is "not computed" as today.
- **The floor never lowers the reply reserve**, and its cache only raises the live figure (§8.2).
- **Strata runs only what was recorded.** `server.py` and `engine/strata` are fingerprinted `launch` entries
  (§11.1); the config copy is built from recorded bytes through an allowlist (§11.2); `engine_dir` passes the
  ownership and mode checks at every launch; nothing of Strata is put on the harness's `PATH`; the engine's escape
  hatches are removed from the child environment. Named residuals: the rest of Strata's tree (`serve.*` modules, its
  venv's site-packages, `~/.config/strata/settings.json`) runs unfingerprinted, so a writer into a user-owned
  `engine_dir` that leaves the two hashed files alone is not asked about; the engine release is `latest`, checked
  only against a size and SHA-256 published by the same origin (`setup.py:3317–3335`), integrity against transport,
  not origin; Swift's shards are not hash-checked by Strata.
- **SECURITY.md is edited with this spec**: `autosense` (lines 230–232) gains that `strata_dir` selects a program the
  harness runs; "eleven launch keys" (585–589) becomes thirteen with `engine_dir` and `engine_config`, beside the
  `lifecycle.strata_code` hashes and `lifecycle.engine_config`; the sentence that derived flags "each pick among
  closed values" (589) names the vLLM `--max-model-len` and Strata's halved `--max-context` as computed numbers; the
  register gains entries for the third-party engine and its installer, `provider_type: strata`, an HF snapshot as a
  launch input, and `server-floor.json` and `strata/` as protected config-dir entries; `HARNESS_GLOBAL_ONLY_FIELDS`
  (`config/loader.py:506`) already covers `autosense` whole, and a test pins that `strata_dir` from a workspace layer
  is dropped; `tests/unit/test_docs_convergence.py`'s counts move with the text.

## 14. Files touched

| File | Change |
|---|---|
| `provider/runtimes.py` (new) | `Runtime`, `RUNTIMES`; landed first, no behaviour change |
| `provider/lifecycle.py`, `provider/server.py`, `cli/library.py`, `cli/serve_cmd.py`, `provider/detector.py`, `provider/refarch.py`, `config/models.py`, `config/trust.py`, `provider/tuning.py` | each runtime if-chain of §6 reads the row |
| `provider/gguf_header.py` (moved from `agent/`) | `read_tensor_table`, `TensorTable`, `classify`, `kv_layers`, the cross-checks, the new caps; importers updated |
| `provider/autosense.py` | `TensorSplit`; the llama.cpp and vLLM prices; `cache_ram_mib` with experts and the reserve; the trimmed window line; the refusal sentences |
| `provider/tuning.py` | `draft_share` as `1 + kv_draft / kv_main`; the `facts` rows; the remembered date in local time |
| `provider/lifecycle.py` | `Sensed.split`; `plan_launch` through the row's `price`; the refusal before any stop; `--fit-ctx` in `fit-by-server`; the floored retry; `swap(…, floor=)` planning before `free_accelerator` and the backstop in its failure branch; `record_floor` and `server-floor.json` |
| `provider/strata.py` (new) | `StrataStrategy`, the config copy and its allowlist, identity, readiness, retry, stop, `price`, `--script` |
| `provider/server.py` | `serve_command(..., fit_ctx=)`; vLLM's `context` as `--max-model-len`; `HF_HUB_OFFLINE=1` for a snapshot launch; the child-environment blocklist additions |
| `provider/model_library.py` | the snapshot kinds and the whole-snapshot `lstat` walk; the `auto_map` test; Strata install sensing |
| `agent/context.py` | `harness_floor`; `probe_window` dispatching on the row |
| `agent/subagent.py` | `make_explore_agent_runner(window=)`; `_make_child_ctx` reads it at spawn; the child floor check |
| `cli/model_ops.py` | `plan_entries` |
| `cli/start_cmd.py` | the floor before the launch, the full floor after the registry (replacing the ≤ 1,024 branch); `window=` at the runner; the capability probe after the launch's `wait_ready` |
| `cli/repl.py`, `cli/model_cmd.py` | render `plan_entries`; the swap passes the live floor; the receipt's retry clause |
| `cli/workspace.py`, `cli/init_cmd.py`, `cli/start_serving.py` | the foreign-unit and imported-unit sentences, one builder, on change |
| `cli/session_trust.py`, `cli/workspace.py` | the guarded-mode notice printed once |
| `cli/doctor_cmd.py` | the `Tuning:` row's per-tensor and RAM figures; the `Strata:` row |
| `agent/gate_types.py` | `server-floor.json` and `strata/` in `PROTECTED_CONFIG_DIR_ENTRIES` |
| `docs/specs/16, 20, 08, 10, 13, 19`, `SECURITY.md`, `tests/unit/test_docs_convergence.py` | §12's amendments; §13's SECURITY.md edits |

## 15. Tests

- The table: every runtime's row answers each column; a structural test that no `runtime ==` comparison remains
  outside `provider/runtimes.py`; `strategy_for`, `unit_servable` and `unit_kind` give today's answers for the four
  existing runtimes.
- The tensor table, against the real numbers: sparse fixture GGUFs (`os.truncate` to the real size, so they cost no
  disk) whose tables reproduce §2.2's offsets; `classify` and `kv_layers` give 3,968,785,920 / 42,912,972,800 /
  29,073,290,240 bytes and 12 attention layers for Flash-Next, and 16 + 1 for both 27B files; KV per token 16,320 B
  (q8_0) and 30,720 B (f16) for Flash-Next, 34,816 + 2,176 B for the 27B. A hostile fixture (every tensor named
  `token_embd` or `_exps`, no `attn_k`) fails the cross-checks and prices as the whole file.
- `fit` at the episodes' figures: the 27B at 30.2 GiB free fits 262,144; Flash-Next at 30.2 free and 52.9 available
  fits 262,144 with experts in RAM and `cache_ram` 3.7 GiB; Flash-Next at 31.4 available is the RAM refusal; the 27B
  at 12.0 free is the VRAM refusal; an unreadable `/proc/meminfo` refuses.
- The plan carries `-c` whenever priced; `--fit-ctx <floor>` only in `fit-by-server`; the retry floored; a hand
  `-ngl` on a model that needs experts in RAM refused.
- vLLM on the four snapshots' `config.json` (fixtures, sizes stubbed): 100,864 for both NVFP4 27B; the gemma
  `not offered` line; the embedding snapshot absent; an `auto_map` in `config.json` or `tokenizer_config.json` not
  offered; a `tokenizer.json` symlink leaving `blobs/` unlists the snapshot; the argv holding the snapshot path and
  the table's parsers only, `HF_HUB_OFFLINE=1` in its environment.
- The floor: the ascending scan across the 12,288 jump; 6,700 → 15,360 and 4,370 → 8,448; the cache only raising the
  live figure and clamped; the swap refusing before `free_accelerator` (a fake incumbent never stopped), the backstop
  through `swap`'s failure branch (target stopped, incumbent restored, `_refresh_token_counter` never called);
  `start` and attach exiting 1 under the floor before any turn.
- Children: a runner built at 4,096, the root refit to 66,560, the next spawn at 66,560 (nested and cruncher too);
  the structural test over `make_explore_agent_runner(` callers; a child floor above the window returns the error.
- The listing, through `plan_entries` from both renderers: nothing numbered is refused when picked; the
  `available once llm.service is stopped` group; one entry per file with two records; a down lifecycle peer once; the
  foreign-unit sentence on change only, escaped (a unit name with a control character and a space).
- Strata: the row's columns; identity matching the venv python with the record's tokens and refusing the engine
  child, another port's copy and a planted pidfile; the re-proof before each signal; the argv and cwd; the copy's
  allowlist (an `env` key, an unknown engine flag and a path outside the three roots each refuse, naming it); the
  environment blocklist; readiness (refused and `loaded: false` are loading, a dead parent is a failure); the `probe`
  on a fixture `/v1/models`; the halved `--max-context` retry and the second-death sentence; stop order; `--script`
  round-tripping the copy; the fingerprint asking after `server.py` changes and not after a touch with an equal hash;
  the install sensing refusing a symlinked `server.py`, a group-writable directory, a missing engine and a config
  naming an unknown file; a `--native` mismatch refused; the RAM check with the reserve and an unreadable
  `/proc/meminfo`; no test runs anything under an engine directory.
- Live, on the reference box, a runbook (`--results` isolated, nothing run against the holdout):
  1. With `llm.service` stopped, `/model Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`: the launch line shows experts in
     RAM and the plan's window; a turn and a delegated turn complete with no emergency floor; the `CUDA0 compute
     buffer size` read by eye against the margin; or, if RAM is short that day, the refusal and nothing stopped.
  2. `/model nvidia/Qwen3.8-27B-NVFP4` (set up on the way): vLLM with `reasoning qwen3 · tools qwen3_coder` and
     `--max-model-len` from the plan; a turn runs.
  3. The gemma swap: the `not offered` line; `/model` on its name prints the sentence, stops nothing, and a turn runs.
  4. The round trip 27B → Flash-Next → 27B, then a delegated turn: the child's window equals the served window, no
     emergency floor; the 27B's decode against its 116 t/s band at the new window.
  5. `/model` with `llm.service` serving: §10.2's lines, the sentence once; `init` and `start` silent on the second run.
  6. `/model` with `llm.service` stopped: numbered entries, every one launches.
  7. The real `fixed_tokens` of the root and the `web-researcher`, and their floors, written into §8.1.
  8. Strata install, run by the owner from `doctor`'s command; the next `/model` shows `· Strata available`.
  9. `/model Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-strata`: set up, launched, `/health` `loaded`, the window line
     names Strata, the served window at least the floor, a turn and a delegated turn run; decode recorded beside
     llama.cpp's 23.8 t/s.
  10. `localharness start` on the Strata record: attaches when it serves, launches and waits when it does not.
  11. Strata → llama.cpp 27B → Strata: each stop verified, `MemAvailable` back within 2 GiB after Strata stops, a
     delegated turn after each.
  12. The response field names for reasoning and tool calls from one `--debug` body, written into §11.6.

## 16. Open questions and recorded decisions

- **Decided — the resume clause.** `(your last /model choice)` prints when `start` **launches** a record on the
  `active_endpoint` rung, and never on an attach to the server the person left. Spec 20 §14 (4) expected it after
  `/model` + quit + `start`, while §8.2 gated it on `last_served ≠ record` and `/model` writes `last_served`, so it
  could not print. A clause reports news, and an attach to the model the person left running is none; a launch after
  a reboot without autostart is. Spec 20 §14 (4) becomes a silent attach, and a row is added: reboot without
  autostart, then `start` launches the last choice with the clause.
- **Decided — the first-run ranking.** `rank_models` gains two keys after "fits": the larger window first, then fewer
  GPU-resident bytes, then the newest. No preset name is special in an OSS ranking. Decode is memory-bandwidth bound,
  so at an equal window the model that reads fewer bytes a token is the faster one, and on the box that is Swift
  (18,562,465,856 B against 18,913,411,136 B); both cap at 262,144, which is why the mtime tie-break chose the NVIDIA
  file. Spec 20 §14 (9) stands as written, and spec 20 §13's file row says so.
- **Decided — the `model` receipt after a tolerated retry.** `Serving <name> — autostart restarted` gains `(retried
  with a smaller window, speculative decoding off)` when the wait tolerated a restart and `confirm_launch` recorded
  the retry; how to keep the full shape is `doctor`'s. The marker belongs to a boot, but the person who ran the
  command is the one who needs to know the shape changed.
- **Decided — MTP is judged from the launch, not from `/props`.** On this build `/props` reports
  `speculative.types: none` with MTP on, so the probe and `doctor` read `--spec-type draft-mtp` from the verified
  live process's argv, the evidence `confirm_launch` already holds. Spec 20's live row (2) is amended.
- **Decided — the remembered date is local.** `tuning.py:303` prints `remembered_at[:10]`, the UTC day; it prints
  `datetime.fromisoformat(at).astimezone().date()`. The ledger keeps UTC; the person reads their own calendar.
- **Decided — the guarded-mode notice prints once.** `session_trust.DECLINED_NOTICE` (`cli/session_trust.py:88`) and
  the offer-decline notice (`cli/workspace.py:379, 400`) both fire on a first start's "no"; the offer path records
  the decline and prints nothing, and `session_trust` alone says it.
- **Decided — the capability probe waits for its own launch.** A launching surface runs `detect_capabilities` only
  after that launch's `wait_ready` returned, against the `LiveEndpoint` it produced; an attach probes at once. A probe
  against a server still loading cannot tell "no tools" from "not ready".
- **Decided — the imported old unit.** §10.3's sentence, printed once by `init`, its disable clause only for an
  enabled unit.
- **Decided — where the floor lives.** Computed live at plan time; cached in the lifecycle's `server-floor.json` only
  for `start`'s pre-registry launch, where the MCP share is not yet known; never in the speed ledger, which holds
  measurements.
- **Decided — Strata is opt-in.** A second preset, set up by the person, preferred only by the person's own landing
  until rows 8–11 pass; the owner ruled it core, though it is a plugin candidate.
- **Open — the compute margin of a split launch.** The Flash-Next profile used 10.6 GiB of GPU at `-c 32768` against
  4.6 GiB of dense weights, projector and q8_0 KV, so about 5–6 GiB went to compute buffers, recurrent state and the
  CUDA context, against a 10% margin of 0.45 GiB. On this box the window caps at the trained 262,144 with room to
  spare, so the margin does not move the plan; on a smaller card it would, and `-c` set keeps llama-server from
  correcting it. Live row 1 records the buffer sizes; a margin keyed to the model's shape waits for that number.
- **Open — a dense launch at the trained window.** The corrected 27B fit leaves 1.2 GiB above the 10% margin at
  262,144; a spilled layer is slower, not failed, and nothing reports it (no log text is parsed). Live row 4 reads the
  decode speed against the 116 t/s band.
- **Open — Strata's unknowns (§11.6)**, each with a live row.

## 17. Out of scope

- An SSD tier of the harness's own, vLLM CPU offload, and any harness-chosen `-ngl`, `--n-cpu-moe` or `-ot`.
- Sliding-window KV pricing (priced at the full window, which errs high).
- Launching beside a foreign unit when it fits (spec 16 §12, still open).
- Strata's other families (Coder, the original Flash-Next), its multi-GPU flags, `--replicas`, `--lazy` and idle
  unload, its MCP and image endpoints, and its browser app.
