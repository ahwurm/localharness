# Image Generation — Qwen-Image-2.1 on ComfyUI

Image generation is a plugin that ships with LocalHarness, off until you turn it on. It talks to a
ComfyUI server you run on your own machine.

The harness ships the client, a doctor check and a workflow-template contract; the model, its
weights and the ComfyUI server are yours to provide.

## How it works

| Piece | What it does |
|---|---|
| **Tool** | `generate_image` — offered to the model only when image is on and `image.comfyui_url` is set. Declared: reads no outside content, writes only into the folder LocalHarness gives it, result trusted. |
| **Command** | `localharness generate-image "a prompt" [--out picture.png]` — listed only while image is on. |
| **App** | A tool result carries a typed `artifact`; the phone page shows it inline from `GET /api/artifacts/image/{id}`. |
| **Health** | `localharness doctor` runs the plugin's check: ComfyUI answers, the template loads, and ComfyUI has every model file the template needs. |

## Settings

`image.comfyui_url` and `image.workflow` are machine-level only; the other two may be set per project.

| Setting | Meaning |
|---|---|
| `image.enabled` | Default off; a project may switch it. |
| `image.comfyui_url` | Machine-level only. The address of your ComfyUI server, e.g. `http://127.0.0.1:8188`. |
| `image.workflow` | Machine-level only. Empty = the shipped `qwen-image-2.1` template; a path = your own template. |
| `image.timeout_s` | Seconds to wait for one picture (the first one also loads the model). Default 570; a project may change it. |

Machine-level only means a project folder cannot set it: ComfyUI runs whatever graph it is sent, so
a cloned project must never choose the server or the graph. A project can turn image on, but only
against the server your machine already points at.

## Setup

Turn it on with `localharness plugins enable image` on a terminal: it asks for the ComfyUI address,
checks it, and if ComfyUI is not ready prints the short version below. `/plugins enable image` in a
running session does the same while image is not set up, then restarts the session with image on
and your conversation kept. In a script:
`localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188` (this form does not run
the check; run `localharness doctor` afterwards).

```
Image needs ComfyUI running on this machine, with three model files:
  models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors   (about 7.3 GB)
  models/text_encoders/qwen3vl_8b_int8_convrot.safetensors          (about 9.4 GB)
  models/vae/qwen_image_2.1_vae_bf16.safetensors                    (about 0.7 GB)
Start it from your ComfyUI folder: venv/bin/python main.py --listen 127.0.0.1 --port 8188
Then run `localharness doctor` to check.
```

### The coding-agent prompt

After those six lines the CLI prints "Or paste this into your coding agent to set it up for your
hardware:" and the prompt below, with the address you gave filled in and, when this machine reports
its GPU, a sentence naming it. `localharness plugins info image` prints the prompt for your machine
any time.

```
Set up ComfyUI on this machine for LocalHarness image generation. Install ComfyUI and run it so
it answers at http://127.0.0.1:8188; keep it off the open internet. Put these Qwen-Image-2.1 INT8 files
in its models folder: diffusion_models/qwen_image_2.1_int8_convrot.safetensors (about 7.3 GB),
text_encoders/qwen3vl_8b_int8_convrot.safetensors (about 9.4 GB) and
vae/qwen_image_2.1_vae_bf16.safetensors (about 0.7 GB). The weights are under the Qwen Research
License: personal, non-commercial use. Keep the UNETLoader weight_dtype at "default" (the fp8
fast mode spoils the pictures). On an NVIDIA GB10 (DGX Spark) the INT8 kernels compile
on first use and need the Python headers: start ComfyUI with C_INCLUDE_PATH pointing at them.
You are done when http://127.0.0.1:8188/system_stats answers and `localharness doctor` shows image
reachable.
```

In more detail: the three files go under ComfyUI's own `models/` folder —
`models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors` (~7.3 GB),
`models/text_encoders/qwen3vl_8b_int8_convrot.safetensors` (~9.4 GB) and
`models/vae/qwen_image_2.1_vae_bf16.safetensors` (~0.7 GB). On a GB10, start ComfyUI with the
command under [Running ComfyUI on the GB10](#running-comfyui-on-the-gb10). The weights are under
the Qwen Research License (see [Reference model](#reference-model-qwen-image-21-int8-convrot)); they
are not distributed with this repo, and where to download them is left to you or your agent.
LocalHarness does not install ComfyUI or download weights for you yet.

## Swapping image models is a template, not a code change

A template is a ComfyUI API-format graph (JSON) with placeholder values `__LH_PROMPT__`,
`__LH_WIDTH__`, `__LH_HEIGHT__`, `__LH_STEPS__`, `__LH_SEED__` (ints are written as quoted
placeholders and substituted post-parse), optional `__LH_PREFIX__`. Top-level keys without a
`class_type` are stripped before POST, so templates can carry `_comment` keys. A template must
contain `__LH_PROMPT__`; the rest are optional. Point `image.workflow` at your template. The shipped
reference template: `src/localharness/tools/builtin/workflows/qwen-image-2.1.json`.

## Where pictures are saved

Pictures are saved under `<state dir>/artifacts/image/` — the project's `.localharness/` in a
project, `~/.localharness/` otherwise — named by an id LocalHarness makes
(`art-YYYYMMDD-HHMMSS-xxxxxx`). They are kept until you delete them. The web app serves only that
folder and only ids of that shape. A new project's `.localharness/.gitignore` keeps them out of git.

`localharness generate-image` is different: it writes the picture where `--out` says, or to
`./image-<YYYYmmdd-HHMMSS>-<hex6>.png` in the current folder.

## Reference model: Qwen-Image-2.1 INT8 (convrot)

Strongest open local text-to-image model at time of writing (notably in-image
text/typography). **License: Qwen Research License — non-commercial/research-eval, no
output carve-out.** Fine for local/personal use; do not ship its outputs commercially,
and the weights are not distributed with this repo.

Model files (ComfyUI directories):

```
models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors   (~7.3 GB)
models/text_encoders/qwen3vl_8b_int8_convrot.safetensors          (~9.4 GB)
models/vae/qwen_image_2.1_vae_bf16.safetensors                    (~0.7 GB)
```

**Measured on the GB10 (2026-09-21):** ~24 s per 1024×1024 image at 30 steps; ~4 s per
512×512 at 8 steps once weights are warm; ~15–20 GB peak — **coexists with the serving
LLM** (verified live beside vLLM on this box; no GPU swap, no serving pause).

**Known gotcha:** keep `UNETLoader.weight_dtype` at `default` (or plain `fp8_e4m3fn`).
`fp8_e4m3fn_fast` corrupts Qwen-Image output — the shipped template pins `default` and a
unit test pins the template.

## Running ComfyUI on the GB10

The INT8/Triton kernel compile needs the Python headers on the include path:

```
cd ~/ComfyUI
C_INCLUDE_PATH="$HOME/ComfyUI/pyheaders/python3.12:$HOME/ComfyUI/pyheaders" \
  venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

The lifecycle is deliberately external: you run ComfyUI (tmux/systemd), the harness talks
to it and says exactly what to start when it is down. Supervised start/stop is future
work, not a shipped claim.
