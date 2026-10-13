"""The vLLM side of derive (spec 20 §6.3, D-12): parsers from the checkpoint's config.json model_type."""
from __future__ import annotations

import json

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider.server import serve_command
from localharness.provider.tuning import derive, read_model_config


def _checkpoint(tmp_path, config: object = None, *, name: str = "ckpt"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"} if config is None else config))
    return d


def _spec(model, **over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "vllm", "launch": "binary", "binary": "/x/vllm",
                                  "model": str(model), "port": 8000, **over})


def _argv(spec: ManagedServerConfig) -> list[str]:
    tuned = derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec))
    return serve_command(spec, reasoning_parser=tuned.reasoning_parser,
                         tool_call_parser=tuned.tool_call_parser)


def test_a_local_checkpoint_selects_its_parsers_and_they_reach_the_argv(tmp_path):
    spec = _spec(_checkpoint(tmp_path))
    cfg = read_model_config(spec)
    assert cfg == {"model_type": "qwen3_moe"}
    tuned = derive(spec, {}, AutosenseConfig(), model_config=cfg)
    assert (tuned.reasoning_parser, tuned.tool_call_parser) == ("qwen3", "qwen3_coder")
    assert dict(tuned.reasons) == {"reasoning_parser": "model_type=qwen3_moe",
                                   "tool_call_parser": "model_type=qwen3_moe"}
    assert serve_command(spec, reasoning_parser="qwen3", tool_call_parser="qwen3_coder") == [
        "/x/vllm", "serve", str(tmp_path / "ckpt"), "--port", "8000",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        "--enable-auto-tool-choice", "--host", "127.0.0.1"]


def test_the_derived_flags_precede_the_persons_extra_args(tmp_path):
    spec = _spec(_checkpoint(tmp_path), extra_args=["--max-num-seqs", "4"])
    assert _argv(spec) == [
        "/x/vllm", "serve", str(tmp_path / "ckpt"), "--port", "8000",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        "--enable-auto-tool-choice", "--max-num-seqs", "4", "--host", "127.0.0.1"]
