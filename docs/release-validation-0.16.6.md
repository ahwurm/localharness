# LocalHarness 0.16.6 validation

Candidate: the 0.16.6 code at `94a2555` (the release commit, tag `v0.16.6`). This is a
post-release validation run on a single developer box — an NVIDIA RTX 5090 serving `swift 1.5`
via vLLM on `:8080` (16,384-token window) — not the CI publication gate: the 0.16.5
document's CI runs on `3538fe8`, `aeb65e1`, and `b4719c2` remain the record for that release, and
the 0.16.6 publication gate is the release commit's own CI run. Persistent memory was not involved
in any run.

## Deterministic checks

Full suite on the candidate code, with the `dev` and `mobile` dependency extras installed
(`uv sync --extra dev --extra mobile`): **7,482 passed, 6 failed, 24 skipped, 4 xfailed**
(≈1:48). The 0.16.5 baseline for the same invocation was 7,454 passed; the net gain is the new
image-input coverage — `test_image_content.py` (117), `test_repl_image.py` (110),
`test_mobile_images.py` (99), `test_acp_images.py` (40), `test_image_input_loop.py` (66),
`test_image_input_spine.py` (59), and the `test_dispatch_attachments.py` additions — which
exercise the picture-as-content-part path, the model's own 32×32-patch token charge, the
`max_image_tokens` downscale/refuse rule, the `/image` and `/image clear` staging, the
compaction naming of a dropped picture, and the HTTP-400 byte redaction.

The 6 failures are all in `tests/unit/test_start_cmd.py` (the per-model context-pin and
window-guard cases) and are a **test-environment coupling, not a code regression**. Those tests
write a config that hardcodes `base_url: http://localhost:8080/v1` and stub `_probe_llm` and
`probe_served_window` but not `model_ops.list_live_models`. On a machine with a live server on
`:8080` — this one runs vLLM serving `swift-1.5` with a 16,384-token window — the un-stubbed
model-list call reaches the real server, finds a single model, and switches the session to it
(`"Configured model 'pinned-model' is not served at http://localhost:8080/v1; using
'swift-1.5'"`), so the pin no longer governs and the window guard trips on the 120,000 scalar.
The same six pass on a machine with nothing on `:8080`; they are not affected by the 0.16.6
change, which touches no start-guard code.

## Distribution checks

`uv sync` built and installed the candidate wheel (`localharness==0.16.6`) into the project venv
and the suite ran against it. No wheel/sdist build or clean-install upgrade was performed for this
post-release run; that is the release commit's CI gate.

## Limitations carried forward

- This is a single-machine, single-run validation on a developer box, not a CI gate and not a
  preregistered live comparison; it establishes that the candidate's deterministic suite is green
  apart from the documented environment coupling, not general reliability.
- The 0.16.5 document's preregistered live comparison (thirteen synthetic cases, baseline vs
  candidate) was not repeated for 0.16.6; the image-input feature is covered deterministically and
  was verified live through the terminal only, per the CHANGELOG's known limitations.
- The six `test_start_cmd.py` failures are reproducible on any host serving a model on
  `:8080`; a hermetic fix (stub `list_live_models`, or point the fixture at a dead port) is a
  separate change and is out of scope for this docs-only commit.
