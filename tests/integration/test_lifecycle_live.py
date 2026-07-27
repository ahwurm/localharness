"""0.11 Phase B — live proof that the HARNESS SPAWNS llama.cpp itself.

Unlike tests/integration/test_live_providers.py (which ATTACHES to an already-running server and
certifies the OpenAI-compat round trip), this test exercises the lifecycle MECHANISM end to end:
`SpawnedProcessStrategy.activate()` actually launches `llama-server` on the box through the
harness's own `serve_command` -> `start_server` -> `wait_ready` chain, `.liveness()` confirms it
is up, `.stop()` tears it down (verified), and `.liveness()` confirms the process is gone.

Gated OFF by default behind its OWN marker + env var (live_llamacpp_lifecycle /
LOCALHARNESS_LIVE_LLAMACPP_LIFECYCLE) — DISTINCT from live_llamacpp so opting into the attach
round trip never also spawns a second GPU process here. The CPU-only suite launches NOTHING: the
autouse _skip_live_providers gate (conftest.py) skips this whenever the env var is unset. Run it
ONLY in an attended GPU window with the GPU free (owner's box rule: one serving framework on the
GPU at a time):

    LOCALHARNESS_LIVE_LLAMACPP_LIFECYCLE=1 .venv/bin/python -m pytest \
        tests/integration/test_lifecycle_live.py -v -s

Assumes the box layout: llama-server at ~/llama.cpp/build/bin/llama-server and the Q4_K_M GGUF at
~/models/Qwen3.6-35B-A3B-GGUF/. Under an opted-in run a missing binary/model HARD-FAILS (never
skips) — a real failure of an explicit request, per the 'fail explicitly' rule.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider.client import LLMClient, LLMConfig
from localharness.provider.lifecycle import SpawnedProcessStrategy

_LLAMA_SERVER = Path("~/llama.cpp/build/bin/llama-server").expanduser()
_GGUF = Path("~/models/Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf").expanduser()
_PORT = 8080
_ALIAS = "qwen3.6-35b-a3b"


@pytest.mark.live_llamacpp_lifecycle
async def test_llamacpp_spawn_lifecycle(tmp_path):
    """activate (harness spawns llama-server) -> served + one real completion -> liveness True ->
    stop -> liveness False. The end-to-end box proof for SpawnedProcessStrategy."""
    for p in (_LLAMA_SERVER, _GGUF):
        if not p.exists():
            pytest.fail(f"live_llamacpp_lifecycle is opted in but {p} is missing")

    # The exact box recipe (task spec): the alias/context/offload flags ride in extra_args; the
    # command builder turns this into `llama-server -m <gguf> --host 127.0.0.1 --port 8080 ...`.
    spec = ManagedServerConfig(
        runtime="llamacpp",
        launch="binary",
        binary=str(_LLAMA_SERVER),
        model=str(_GGUF),
        port=_PORT,
        extra_args=["-c", "32768", "--parallel", "1", "-ngl", "99", "--jinja", "-a", _ALIAS],
        gpu=True,
    )
    base_url = f"http://127.0.0.1:{_PORT}/v1"
    strat = SpawnedProcessStrategy()

    ep = await strat.activate(
        spec, tmp_path, base_url,
        timeout_seconds=900.0,  # cold 35B Q4_K_M GGUF load; wait_ready fails FAST if it crashes
        on_poll=lambda s: print(f"  loading llama-server... {s:.0f}s", flush=True),
    )
    try:
        # The HARNESS launched it, and it serves the aliased model id.
        assert _ALIAS in ep.served_models, f"served {ep.served_models}, expected {_ALIAS!r}"
        assert isinstance(ep.handle, int)  # spawned stop-handle = the launched pid
        assert strat.liveness(spec, tmp_path).alive is True

        # One real completion through the harness client — proves the spawned server generates.
        client = LLMClient(LLMConfig(base_url=base_url, model=_ALIAS, is_local=True))
        await client.detect_capabilities()
        client.config.max_tokens = 64
        message, _usage = await client.complete([{"role": "user", "content": "Reply with one short word."}])
        produced = (message.content or "").strip() or (getattr(message, "reasoning_content", "") or "").strip()
        assert produced != "", "spawned llama-server produced no output"
    finally:
        # Always tear the GPU process down, even if an assertion above failed.
        await strat.stop(spec, tmp_path)

    # Verified stop: the pidfile is gone, so pid-based liveness reads down.
    assert strat.liveness(spec, tmp_path).alive is False
