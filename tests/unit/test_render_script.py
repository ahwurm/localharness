"""`tuning.render_script`: a `LaunchPlan` as POSIX sh (spec 20 §10).

Plans come from the real `sense_launch` and `plan_launch` over the faked box (`Box`); nothing is started or read
from the host.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from localharness.cli.workspace import visible_text
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import lifecycle, server, tuning
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import write_gguf
from tests.unit.test_launch_fit_retry import Box
from tests.unit.test_tuning_derive import SWIFT

GIB = 2**30
KNOBS = AutosenseConfig()


@pytest.fixture
def box(monkeypatch):
    world = Box(monkeypatch)
    world.mem = FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi")
    world.weights = 18 * GIB
    return world


def _spec(tmp_path: Path, filename: str = "model.gguf", **over) -> ManagedServerConfig:
    model = write_gguf(tmp_path / filename, SWIFT)
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server", "model": str(model),
                                  "port": 8080, **over})


def _vllm(tmp_path: Path, **over) -> ManagedServerConfig:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    return ManagedServerConfig(**{"runtime": "vllm", "launch": "binary", "binary": "/x/vllm",
                                  "model": str(ckpt), **over})


def _plan(tmp_path, spec, *, keyed: bool = False):
    sensed = lifecycle.sense_launch(spec, tmp_path, KNOBS)
    decision = server.launch_decision(spec, keyed)
    return lifecycle.plan_launch(spec, sensed, decision, autosense=KNOBS, api_key_held=keyed)


def _render(plan, spec, **over) -> str:
    kw = {"name": "qwen", "version": "0.19.0", "key_placeholder": "provider.api_key",
          "home_shown": "~/.localharness/xdg/8080", "visible": visible_text}
    return tuning.render_script(plan, spec, **{**kw, **over})


def _exec_tokens(script: str) -> list[str]:
    body = [ln for ln in script.splitlines() if not ln.startswith(("#", "export "))]
    return shlex.split("\n".join(body).replace("\\\n", " "))[1:]


def _without_pair(argv, flag: str, value: str) -> list[str]:
    at = next(i for i in range(len(argv) - 1) if argv[i] == flag and argv[i + 1] == value)
    return [*argv[:at], *argv[at + 2:]]


def test_a_fitted_launch_is_two_comments_then_an_exec_with_the_fit_left_to_llama_server(tmp_path, box):
    spec = _spec(tmp_path, served_name="qwen")
    plan = _plan(tmp_path, spec)

    lines = _render(plan, spec).splitlines()

    assert lines[0] == (
        "# qwen — the launch `localharness start` / `/model` runs (0.19.0); "
        "derived: kv q8_0 (autosense.kv_cache_type) · draft-mtp (nextn_predict_layers=1)")
    assert lines[1] == (
        f"# -c left to llama-server --fit (the harness fitted {plan.full_fit} today); environment: "
        "LLAMA_ARG_*/LLAMA_API_KEY/MTMD_BACKEND_DEVICE dropped, XDG_CONFIG_HOME → ~/.localharness/xdg/8080")
    assert lines[2] == "exec /x/llama-server \\"
    assert lines[3].startswith("  -m ") and not lines[-1].endswith("\\")
    assert all(ln.endswith(" \\") for ln in lines[2:-1])


def test_the_exec_tokens_are_the_plan_argv_in_order_without_the_fitted_window_pair(tmp_path, box):
    spec = _spec(tmp_path)
    plan = _plan(tmp_path, spec)

    tokens = _exec_tokens(_render(plan, spec))

    assert tokens == _without_pair(plan.argv, "-c", str(plan.full_fit))
    assert plan.report.mode == "fit" and "-c" in plan.argv


def test_a_record_context_keeps_its_window_and_says_so(tmp_path, box):
    spec = _spec(tmp_path, context=32768)
    plan = _plan(tmp_path, spec)

    script = _render(plan, spec)

    assert "-c 32768 from the record" in script.splitlines()[1]
    tokens = _exec_tokens(script)
    assert tokens == list(plan.argv) and tokens[tokens.index("-c") + 1] == "32768"


def test_extra_args_carrying_c_keep_the_pair_and_the_comment_says_where_it_came_from(tmp_path, box):
    spec = _spec(tmp_path, extra_args=["-c", "8192"])
    plan = _plan(tmp_path, spec)

    script = _render(plan, spec)

    assert plan.report.mode == "extra_args"
    assert "-c from extra_args" in script.splitlines()[1]
    assert _exec_tokens(script) == list(plan.argv) and "8192" in _exec_tokens(script)


def test_an_unknown_free_memory_leaves_c_to_llama_server_with_no_fitted_value(tmp_path, box):
    box.mem = None
    spec = _spec(tmp_path)
    plan = _plan(tmp_path, spec)

    script = _render(plan, spec)

    assert plan.report.mode == "fit-by-server"
    assert script.splitlines()[1].startswith("# -c left to llama-server --fit; environment:")
    assert _exec_tokens(script) == list(plan.argv)


def test_every_token_is_quoted_and_a_control_character_is_shown_not_sent(tmp_path, box):
    spec = _spec(tmp_path, "my model$x.gguf", served_name="\x1b[2Kq")
    plan = _plan(tmp_path, spec)

    script = _render(plan, spec, name="a\x1b[31mb")

    assert "\x1b" not in script
    assert f"'{spec.model}'" in script
    assert "'\\x1b[2Kq'" in script
    assert script.splitlines()[0].startswith("# a\\x1b[31mb — the launch")
    assert shlex.split(script.splitlines()[-1])[-1] == "127.0.0.1"


def test_an_inline_key_is_a_placeholder_and_never_its_value(tmp_path, box):
    spec = _spec(tmp_path, require_api_key=True)
    plan = _plan(tmp_path, spec, keyed=True)

    script = _render(plan, spec, key_placeholder="extra_endpoints.flash.api_key")

    assert "# LLAMA_API_KEY=<extra_endpoints.flash.api_key>" in script.splitlines()
    assert "sk-secret" not in script


@pytest.mark.parametrize("extra, shown", [
    (["--api-key", "sk-x"], "--api-key '<key>'"),
    (["--api-key=sk-y"], "'--api-key=<key>'"),
])
def test_an_extra_args_api_key_value_is_replaced_by_key(tmp_path, box, extra, shown):
    spec = _spec(tmp_path, extra_args=extra)
    plan = _plan(tmp_path, spec)

    script = _render(plan, spec)

    assert shown in script
    assert "sk-x" not in script and "sk-y" not in script


def test_a_key_file_stays_its_path(tmp_path, box):
    key = tmp_path / "key"
    key.write_text("sk-file")
    spec = _spec(tmp_path, require_api_key=True, api_key_file=str(key))
    plan = _plan(tmp_path, spec, keyed=True)

    script = _render(plan, spec)

    assert f"--api-key-file {key}" in script
    assert "sk-file" not in script


def test_a_record_env_is_one_export_per_name_in_sorted_order_before_the_exec(tmp_path, box):
    spec = _spec(tmp_path, env={"LLAMA_ARG_THREADS": "a b", "LLAMA_ARG_N_GPU_LAYERS": "99"})
    plan = _plan(tmp_path, spec)

    lines = _render(plan, spec).splitlines()

    exports = [ln for ln in lines if ln.startswith("export ")]
    assert exports == ["export LLAMA_ARG_N_GPU_LAYERS=99", "export LLAMA_ARG_THREADS='a b'"]
    assert lines.index(exports[-1]) < next(i for i, ln in enumerate(lines) if ln.startswith("exec "))


def test_a_vllm_binary_record_renders_serve_with_its_parsers_and_no_config_home(tmp_path, box):
    spec = _vllm(tmp_path)
    plan = _plan(tmp_path, spec)

    lines = _render(plan, spec, name="vl").splitlines()

    assert "derived: reasoning qwen3 (model_type=qwen3) · tools qwen3_coder (model_type=qwen3)" in lines[0]
    assert lines[1] == "# environment: VLLM_API_KEY dropped"
    assert lines[2] == "exec /x/vllm \\"
    assert lines[3] == f"  serve {shlex.quote(spec.model)} \\"
    assert "  --reasoning-parser qwen3 \\" in lines and "  --tool-call-parser qwen3_coder \\" in lines
    assert "XDG_CONFIG_HOME" not in "\n".join(lines)


def test_a_function_wraps_the_same_body_in_a_subshell_sh_accepts(tmp_path, box):
    spec = _spec(tmp_path, env={"LLAMA_ARG_N_GPU_LAYERS": "1"})
    plan = _plan(tmp_path, spec)

    script = _render(plan, spec, function="serve_qwen")

    lines = script.splitlines()
    assert lines[:2] == ["serve_qwen() {", "  ("] and lines[-2:] == ["  )", "}"]
    assert any(ln.startswith("    exec ") for ln in lines)
    path = tmp_path / "f.sh"
    path.write_text(script + "\n")
    assert subprocess.run(["sh", "-n", str(path)], capture_output=True).returncode == 0


def test_a_retry_shape_adds_its_clause_to_the_derived_words(tmp_path, box):
    spec = _spec(tmp_path)
    retry = _plan(tmp_path, spec).retry()

    first = _render(retry, spec).splitlines()[0]

    assert retry.shape == "retry"
    assert "kv q8_0 (autosense.kv_cache_type) · " in first and "(retried: the full shape died)" in first


def test_the_scrub_names_shown_are_what_the_launch_scrubs():
    assert set(tuning.SCRUBBED_SHOWN) == {"llamacpp", "vllm"}
    for token in tuning.SCRUBBED_SHOWN["llamacpp"].split("/"):
        assert server._scrubbed("llamacpp", token.replace("*", "X"))
    assert server._scrubbed("llamacpp", "LLAMA_ARG_X") and not server._scrubbed("llamacpp", "VLLM_API_KEY")
    assert server._scrubbed("vllm", tuning.SCRUBBED_SHOWN["vllm"])
