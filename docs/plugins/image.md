# The image plugin

The `image` plugin makes pictures with a ComfyUI server that you run on your own machine. It ships with LocalHarness, is off until you turn it on, and its code is [`image_plugin.py`](../../src/localharness/tools/builtin/image_plugin.py).

## Turn it on and off

```bash
localharness plugins enable image                                         # on a terminal: asks for the ComfyUI address, then checks it
localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188  # in a script
localharness plugins disable image
```

Either takes effect at the next `localharness start`. Until `image.comfyui_url` is set, the plugin reports itself unconfigured and adds nothing.

## What it adds

- The `generate_image` tool, for the agent.
- `localharness generate-image "<prompt>"` with `--width`, `--height`, `--steps`, `--seed` and `--out`.
- A `doctor` check that asks ComfyUI whether it answers and has the template's model files.

Pictures are saved under `<state dir>/artifacts/image/`. The phone app shows them inline and in its Pictures gallery, and the Discord bot posts them as files.

## Settings

| Setting | Meaning |
|---|---|
| `image.comfyui_url` | **Machine-level only.** Your ComfyUI server's address, e.g. `http://127.0.0.1:8188`. |
| `image.workflow` | **Machine-level only.** Path to your own ComfyUI workflow template; empty uses the shipped `qwen-image-2.1` one. |
| `image.timeout_s` | Seconds to wait for one picture; the first one also loads the model. Default 570. |

A project cannot set the two machine-level ones: ComfyUI runs whatever graph it is sent, so a repository you clone must never choose the server or the graph. A project can turn image on, but only against the server your machine already points at.

## Not there yet

- LocalHarness does not install or start ComfyUI and does not download its model files (about 17 GB for the shipped template, under the Qwen Research License: personal, non-commercial use). The [setup guide](../reference-architectures/image-generation.md) says what to run.
- The terminal does not display pictures.

## More

- [Setup guide and settings](../reference-architectures/image-generation.md)
- [Spec 09, the bundled plugins](../specs/09-hooks-plugins.md#the-bundled-plugins)
- [localharness.dev/plugins/image](https://localharness.dev/plugins/image/)
- [Write your own plugin](../../examples/plugin-template/README.md)
