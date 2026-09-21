# Image Generation Module — Qwen-Image-2.1 on ComfyUI

**Status: TESTED** (maintainer hardware, DGX Spark GB10, 2026-09-21). The image module is
**opt-in and model-agnostic**: the harness ships the client and a workflow-template
contract; the model, its weights and the ComfyUI server are operator-provided. Base
installs never register the tool.

## How the module works

| Piece | Where |
|---|---|
| Tool | `generate_image` — registered only when `LOCALHARNESS_COMFYUI_URL` is set |
| CLI | `localharness generate-image "<prompt>" [--width --height --steps --seed --out]` |
| App | tool results carry a typed image id; the web UI renders it via `GET /api/images/{id}` |
| Health | `localharness doctor` prints an Image module line when the env var is set |

Environment contract (operator config — none of this is visible to the subject model):

```
LOCALHARNESS_COMFYUI_URL        e.g. http://127.0.0.1:8188  (unset = module off)
LOCALHARNESS_COMFYUI_WORKFLOW   path to a workflow template; default = shipped qwen-image-2.1
LOCALHARNESS_COMFYUI_TIMEOUT_S  generation deadline, default 570 (first call loads weights)
```

**Swapping image models is a template, not a code change.** A template is a ComfyUI
API-format graph (JSON) with placeholder values `__LH_PROMPT__`, `__LH_WIDTH__`,
`__LH_HEIGHT__`, `__LH_STEPS__`, `__LH_SEED__` (ints are written as quoted placeholders
and substituted post-parse), optional `__LH_PREFIX__`. Top-level keys without a
`class_type` are stripped before POST, so templates can carry `_comment` keys. A template
must contain `__LH_PROMPT__`; the rest are optional. The shipped reference template:
`src/localharness/tools/builtin/workflows/qwen-image-2.1.json`.

Generated PNGs land in `<workspace>/.localharness/artifacts/images/` under a
harness-minted id; the web endpoint serves only that directory, only by exact id shape.

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

v1 lifecycle is deliberately external: you run ComfyUI (tmux/systemd), the harness talks
to it and says exactly what to start when it is down. Supervised start/stop is future
work, not a shipped claim.
