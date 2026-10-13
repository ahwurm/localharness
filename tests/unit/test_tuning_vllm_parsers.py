"""The vLLM side of derive (spec 20 §6.3, D-12): parsers from the checkpoint's config.json model_type."""
from __future__ import annotations

import json
import os

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import tuning
from localharness.provider.server import serve_command
from localharness.provider.tuning import MODEL_CONFIG_CAP, NOTHING_DERIVED, derive, read_model_config


_REAL_SEAM = tuning._hf_cached_config  # captured at import, before the autouse fixture replaces it


def _hf_cache(monkeypatch, lookup) -> None:
    """Exercise the real snapshot lookup, with the library function replaced by `lookup`."""
    monkeypatch.setattr(tuning, "_hf_cached_config", _REAL_SEAM)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lookup)


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


# ---- the whole table ---------------------------------------------------------------------------

TABLE = {
    "qwen3": ("qwen3", "qwen3_coder"), "qwen3_moe": ("qwen3", "qwen3_coder"),
    "qwen3_next": ("qwen3", "qwen3_coder"), "qwen3_5": ("qwen3", "qwen3_coder"),
    "qwen3_5_moe": ("qwen3", "qwen3_coder"), "gemma4": ("gemma4", "gemma4"),
    "glm4_moe": ("glm45", "glm45"), "glm4v_moe": ("glm45", "glm45"),
    "deepseek_v3": ("deepseek_r1", "deepseek_v3"), "gpt_oss": ("openai_gptoss", "openai"),
    "llama": (None, "llama3_json"), "llama4": (None, "llama4_json"),
    "mistral": (None, "mistral"), "mistral3": (None, "mistral"), "kimi_k2": (None, "kimi_k2"),
}


def test_the_table_is_exactly_spec_20_section_6_3():
    from localharness.provider.tuning import MODEL_TYPE_PARSERS
    assert dict(MODEL_TYPE_PARSERS) == TABLE


@pytest.mark.parametrize("model_type", sorted(TABLE))
def test_table_row_derives_and_emits_per_model_type(tmp_path, model_type):
    reasoning, tool = TABLE[model_type]
    spec = _spec(_checkpoint(tmp_path, {"model_type": model_type}))
    tuned = derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec))
    assert (tuned.reasoning_parser, tuned.tool_call_parser) == (reasoning, tool)
    expected = {k: f"model_type={model_type}" for k, v in
                (("reasoning_parser", reasoning), ("tool_call_parser", tool)) if v is not None}
    assert dict(tuned.reasons) == expected
    argv = _argv(spec)
    assert ("--reasoning-parser" in argv) == (reasoning is not None)
    if reasoning is not None:
        assert argv[argv.index("--reasoning-parser") + 1] == reasoning
    assert argv[argv.index("--tool-call-parser") + 1] == tool
    assert argv.count("--enable-auto-tool-choice") == 1


def test_a_family_with_no_reasoning_parser_gets_the_tool_side_alone(tmp_path):
    argv = _argv(_spec(_checkpoint(tmp_path, {"model_type": "llama"})))
    assert "--reasoning-parser" not in argv
    assert argv[-5:] == ["--tool-call-parser", "llama3_json", "--enable-auto-tool-choice",
                         "--host", "127.0.0.1"]


# ---- precedence --------------------------------------------------------------------------------

def test_off_on_the_reasoning_field_pins_that_side_off_and_derives_the_other(tmp_path):
    spec = _spec(_checkpoint(tmp_path), reasoning_parser="off")
    tuned = derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec))
    assert tuned.reasoning_parser is None and tuned.tool_call_parser == "qwen3_coder"
    argv = _argv(spec)
    assert "--reasoning-parser" not in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder"


def test_off_on_the_tool_field_emits_no_tool_flag_and_no_auto_tool_choice(tmp_path):
    spec = _spec(_checkpoint(tmp_path), tool_call_parser="off")
    argv = _argv(spec)
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert "--tool-call-parser" not in argv and "--enable-auto-tool-choice" not in argv


def test_a_record_value_is_emitted_as_is_and_the_other_side_still_derives(tmp_path):
    spec = _spec(_checkpoint(tmp_path), tool_call_parser="hermes")
    argv = _argv(spec)
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert argv.count("--enable-auto-tool-choice") == 1


def test_a_record_value_is_emitted_with_nothing_derived(tmp_path):
    spec = _spec(tmp_path, tool_call_parser="hermes")  # a directory with no config.json
    assert read_model_config(spec) is None
    argv = _argv(spec)
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    assert "--reasoning-parser" not in argv


@pytest.mark.parametrize("extra", [
    ["--reasoning-parser", "qwen3"], ["--reasoning-parser=qwen3"],
    ["--tool-call-parser", "hermes"], ["--tool-call-parser=hermes"],
    ["--enable-auto-tool-choice"], ["--max-num-seqs", "4", "--enable-auto-tool-choice"],
])
def test_a_parser_flag_in_extra_args_derives_nothing(tmp_path, extra):
    spec = _spec(_checkpoint(tmp_path), extra_args=extra)
    tuned = derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec))
    assert tuned == NOTHING_DERIVED
    argv = _argv(spec)
    assert argv.count("--enable-auto-tool-choice") == extra.count("--enable-auto-tool-choice")
    def parser_flags(tokens):
        return sum(t.split("=")[0] in ("--reasoning-parser", "--tool-call-parser") for t in tokens)

    assert parser_flags(argv) == parser_flags(extra)


@pytest.mark.parametrize("entry_extra", [
    ["--reasoning-parser", "qwen3"], ["--tool-call-parser=hermes"], ["--enable-auto-tool-choice"]])
def test_a_parser_flag_in_the_local_model_entrys_extra_args_derives_nothing(tmp_path, entry_extra):
    ckpt = _checkpoint(tmp_path)
    spec = _spec("served", local_models=[{"name": "served", "path": str(ckpt), "extra_args": entry_extra}])
    assert read_model_config(spec) == {"model_type": "qwen3_moe"}
    assert derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec)) == NOTHING_DERIVED


def test_a_serving_flag_already_in_extra_args_is_not_duplicated(tmp_path):
    spec = _spec(tmp_path, tool_call_parser="hermes", extra_args=["--enable-auto-tool-choice"])
    assert _argv(spec).count("--enable-auto-tool-choice") == 1


def test_the_local_models_entry_directory_is_where_config_json_is_read(tmp_path):
    ckpt = _checkpoint(tmp_path, {"model_type": "gemma4"})
    spec = _spec("served", local_models=[{"name": "served", "path": str(ckpt)}])
    tuned = derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec))
    assert (tuned.reasoning_parser, tuned.tool_call_parser) == ("gemma4", "gemma4")


# ---- the Hugging Face snapshot -----------------------------------------------------------------

def test_a_repo_id_reads_config_json_from_the_cache_snapshot(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path, {"model_type": "deepseek_v3"}) / "config.json"
    seen = {}

    def lookup(**kw):
        seen.update(kw)
        return str(path)

    _hf_cache(monkeypatch, lookup)
    spec = _spec("org/repo")
    tuned = derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec))
    assert seen == {"repo_id": "org/repo", "filename": "config.json"}
    assert (tuned.reasoning_parser, tuned.tool_call_parser) == ("deepseek_r1", "deepseek_v3")


def _raises(**kw):
    raise RuntimeError("hub trouble")


@pytest.mark.parametrize("lookup", [
    lambda **kw: None, lambda **kw: object(), lambda **kw: 7, _raises,
], ids=["none", "sentinel-object", "int", "raises"])
def test_a_cache_lookup_with_no_usable_path_derives_nothing(monkeypatch, lookup):
    _hf_cache(monkeypatch, lookup)
    spec = _spec("org/repo")
    assert read_model_config(spec) is None
    assert derive(spec, {}, AutosenseConfig(), model_config=read_model_config(spec)) == NOTHING_DERIVED


def test_the_default_cache_lookup_in_the_suite_finds_nothing():
    assert tuning._hf_cached_config("org/repo") is None
    assert read_model_config(_spec("org/repo")) is None


# ---- every bad config.json ---------------------------------------------------------------------

def _bad_file(tmp_path, kind: str):
    d = tmp_path / "bad"
    d.mkdir()
    target = d / "config.json"
    if kind == "missing":
        pass
    elif kind == "cap":
        target.write_bytes(b" " * (MODEL_CONFIG_CAP + 1))
    elif kind == "malformed":
        target.write_text("{bad json")
    elif kind == "recursion":
        target.write_text("[" * 100000)
    elif kind == "list":
        target.write_text("[1, 2]")
    elif kind == "int-model-type":
        target.write_text('{"model_type": 7}')
    elif kind == "list-model-type":
        target.write_text('{"model_type": ["qwen3"]}')
    elif kind == "unknown-model-type":
        target.write_text('{"model_type": "unknown_arch"}')
    elif kind == "fifo":
        os.mkfifo(target)
    elif kind == "directory":
        target.mkdir()
    return d


@pytest.mark.parametrize("kind", ["missing", "cap", "malformed", "recursion", "list", "int-model-type",
                                  "list-model-type", "unknown-model-type", "fifo", "directory"])
def test_a_bad_config_json_derives_nothing_and_raises_nothing(tmp_path, kind):
    spec = _spec(_bad_file(tmp_path, kind))
    cfg = read_model_config(spec)
    assert derive(spec, {}, AutosenseConfig(), model_config=cfg) == NOTHING_DERIVED
    if kind in ("missing", "cap", "malformed", "recursion", "list", "fifo", "directory"):
        assert cfg is None
    assert _argv(spec) == ["/x/vllm", "serve", str(tmp_path / "bad"), "--port", "8000",
                           "--host", "127.0.0.1"]


def test_a_file_at_exactly_the_cap_is_read(tmp_path):
    d = tmp_path / "edge"
    d.mkdir()
    body = b'{"model_type": "qwen3"}'
    (d / "config.json").write_bytes(body + b" " * (MODEL_CONFIG_CAP - len(body)))
    assert read_model_config(_spec(d)) == {"model_type": "qwen3"}


def test_the_read_goes_through_open_regular(tmp_path, monkeypatch):
    calls = []
    real = tuning.open_regular
    monkeypatch.setattr(tuning, "open_regular", lambda p: (calls.append(str(p)), real(p))[1])
    read_model_config(_spec(_checkpoint(tmp_path)))
    assert calls == [str(tmp_path / "ckpt" / "config.json")]


# ---- launch modes, other runtimes, the flags never derived -------------------------------------

def test_docker_emits_the_derived_flags_after_model_and_served_name_before_extra(tmp_path):
    ckpt = _checkpoint(tmp_path)
    spec = ManagedServerConfig(
        runtime="vllm", launch="docker", docker_image="vllm/vllm-openai", model="served",
        local_models=[{"name": "served", "path": str(ckpt), "extra_args": ["--max-num-seqs", "2"]}],
        extra_args=["--dtype", "auto"])
    argv = _argv(spec)
    tail = argv[argv.index("--model"):]
    assert tail == ["--model", "/models/serving", "--served-model-name", "served",
                    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
                    "--enable-auto-tool-choice", "--dtype", "auto", "--max-num-seqs", "2"]


def test_docker_with_a_repo_id_derives_from_the_cache_snapshot(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path, {"model_type": "mistral3"}) / "config.json"
    _hf_cache(monkeypatch, lambda **kw: str(path))
    spec = ManagedServerConfig(runtime="vllm", launch="docker", docker_image="img", model="org/repo")
    argv = _argv(spec)
    assert argv[argv.index("--model"):] == ["--model", "org/repo", "--tool-call-parser", "mistral",
                                            "--enable-auto-tool-choice"]


def test_a_llamacpp_record_never_gets_a_parser_flag(tmp_path):
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf")
    assert read_model_config(spec) is None
    tuned = derive(spec, {}, AutosenseConfig(), model_config={"model_type": "qwen3"})
    assert tuned.reasoning_parser is None and tuned.tool_call_parser is None
    argv = serve_command(spec, reasoning_parser="qwen3", tool_call_parser="qwen3_coder")
    assert not {"--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice"} & set(argv)
    assert argv == serve_command(spec)


def test_nothing_derived_leaves_todays_vllm_argv_untouched(tmp_path):
    spec = _spec("org/repo", extra_args=["--max-model-len", "65536"])
    assert serve_command(spec) == ["/x/vllm", "serve", "org/repo", "--port", "8000",
                                   "--max-model-len", "65536", "--host", "127.0.0.1"]


@pytest.mark.parametrize("model_type", sorted(TABLE))
def test_table_derivation_never_touches_sizing_flags(tmp_path, model_type):
    argv = _argv(_spec(_checkpoint(tmp_path, {"model_type": model_type})))
    assert not {"--max-model-len", "--kv-cache-dtype", "--gpu-memory-utilization",
                "--max-num-seqs"} & set(argv)


def test_a_model_type_from_the_file_is_never_emitted_as_a_value(tmp_path):
    spec = _spec(_checkpoint(tmp_path, {"model_type": "--evil", "architectures": ["--x"]}))
    assert "--evil" not in _argv(spec) and "--x" not in _argv(spec)


def test_a_relative_model_never_reads_a_config_json_from_the_working_directory(tmp_path, monkeypatch):
    """IN-04: `Qwen/Qwen3-8B` is a repo id; a same-named directory under the cwd is never read."""
    (tmp_path / "Qwen").mkdir()
    _checkpoint(tmp_path / "Qwen", name="Qwen3-8B")
    monkeypatch.chdir(tmp_path)
    looked_up: list[str] = []
    monkeypatch.setattr(tuning, "_hf_cached_config", lambda repo_id: looked_up.append(repo_id))
    assert read_model_config(_spec("Qwen/Qwen3-8B")) is None
    assert looked_up == ["Qwen/Qwen3-8B"]
