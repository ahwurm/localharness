"""INIT-05 / INIT-06 / INIT-07 / INIT-10 (translation half): one pure function turns a llama-server process
record (live, journal or unit file) into a validated record, or into the one reason it is attached to only.

The records here are built by hand; nothing reads a real /proc, runs systemctl, opens a key file or
touches ~/.localharness (D-18, D-43). Every quoted line is asserted verbatim (D-45).
"""
from __future__ import annotations

import os

import pytest

from localharness.cli import init_import
from localharness.cli.init_import import (
    ATTACH_ONLY_SUFFIX,
    AttachOnly,
    Imported,
    import_record,
    runtime_of,
)
from localharness.provider.procfs import ProcRecord

NO_BINARY = lambda name: None  # noqa: E731 - the resolver of a box with nothing on PATH

BOX_ARGV = ("llama-server", "-m", "models/a.gguf", "--mmproj", "/m/mmproj-a.gguf", "--alias", "qwen-27b",
            "--host", "127.0.0.1", "--port", "8080", "-ngl", "99", "-c", "65536", "-np", "1", "-fa", "on",
            "--jinja", "--cache-type-k", "q8_0", "--cache_type_v", "q8_0")


def proc(argv=BOX_ARGV, *, exe="/opt/llama/bin/llama-server", cwd="/home/u", unit="llama-server.service",
         environ=None, mapped=("/home/u/models/a.gguf",), start_time=1) -> ProcRecord:
    return ProcRecord(pid=1234, uid=os.getuid(), argv=tuple(argv), exe=exe, cwd=cwd, start_time=start_time,
                      unit=unit, environ=environ, mapped_ggufs=tuple(mapped))


def box(**kw) -> ProcRecord:
    kw.setdefault("environ", {"LLAMA_ARG_N_PARALLEL": "1", "HF_TOKEN": "hf_decoy"})
    return proc(**kw)


def imported(rec, **kw) -> Imported:
    out = import_record(rec, source=kw.pop("source", "live"), binary_resolver=kw.pop("binary_resolver", NO_BINARY),
                        **kw)
    assert isinstance(out, Imported), out
    return out


def attach(rec, **kw) -> str:
    out = import_record(rec, source=kw.pop("source", "live"), binary_resolver=kw.pop("binary_resolver", NO_BINARY),
                        **kw)
    assert isinstance(out, AttachOnly), out
    return out.reason


# ----------------------------------------------------------------------------- the tracer slice


def test_a_box_shaped_llama_server_translates_into_a_validated_record():
    out = imported(box())
    spec = out.spec
    assert spec.runtime == "llamacpp"
    assert spec.binary == "/opt/llama/bin/llama-server"
    assert spec.model == "/home/u/models/a.gguf"
    assert spec.mmproj == "/m/mmproj-a.gguf"
    assert spec.served_name == "qwen-27b"
    assert spec.port == 8080
    assert spec.cache_type_k == spec.cache_type_v == "q8_0"
    assert spec.context is None
    assert spec.bind_all is False
    assert spec.extra_args == ["-ngl", "99", "-np", "1", "-fa", "on", "--jinja"]
    assert spec.env == {"LLAMA_ARG_N_PARALLEL": "1"}
    assert out.ctx == 65536
    assert out.api_key is None
    assert out.binary_source == "process"


def test_the_extras_are_shown_and_nothing_unrelated_from_environ_is():
    out = imported(box())
    assert "flags: -ngl 99 -np 1 -fa on --jinja" in out.shown
    assert "env: LLAMA_ARG_N_PARALLEL=1" in out.shown
    for text in (*out.shown, repr(out), repr(out.spec), str(out.spec.model_dump())):
        assert "hf_decoy" not in text


@pytest.mark.parametrize("exe, argv0, expected", [
    ("/opt/llama/bin/llama-server", "llama-server", "llamacpp"),
    (None, "/usr/local/bin/llama-server", "llamacpp"),
    ("/usr/bin/ollama", "ollama", "ollama"),
    ("/usr/bin/lms", "lms", "lmstudio"),
    (None, "llmster", "lmstudio"),
    ("/usr/bin/python3", "python3", None),
    ("/opt/vllm/bin/vllm", "vllm", None),
    ("/opt/llama/bin/llama-server (deleted)", "llama-server", "llamacpp"),
])
def test_runtime_of_names_the_server_from_exe_or_argv0(exe, argv0, expected):
    assert runtime_of(proc((argv0, "x"), exe=exe)) == expected


# ------------------------------------------------------------------------ the argv / environ table


def test_a_token_nothing_claims_stays_exactly_as_written():
    argv = ("llama-server", "-m", "/m/a.gguf", "--flash_attn", "on", "--no-mmap", "--jinja")
    assert imported(proc(argv, mapped=())).spec.extra_args == ["--flash_attn", "on", "--no-mmap", "--jinja"]


def test_the_equals_form_and_underscores_are_read_like_the_space_form():
    argv = ("llama-server", "--model=/m/a.gguf", "--ctx_size=4096", "--cache_type_k=q4_0", "--cache-ram=512",
            "--mmproj=/m/p.gguf", "--alias=x", "-ngl", "9")
    out = imported(proc(argv, mapped=()))
    assert (out.spec.model, out.ctx, out.spec.cache_type_k, out.spec.cache_ram, out.spec.mmproj,
            out.spec.served_name) == ("/m/a.gguf", 4096, "q4_0", 512, "/m/p.gguf", "x")
    assert out.spec.extra_args == ["-ngl", "9"]


def test_mm_is_the_projector_flag():
    out = imported(proc(("llama-server", "-m", "/m/a.gguf", "-mm", "/m/p.gguf"), mapped=()))
    assert out.spec.mmproj == "/m/p.gguf"
    assert out.spec.extra_args == []


def test_c_is_the_window_input_only_never_the_record_and_never_left_in_extra_args():
    out = imported(proc(("llama-server", "-m", "/m/a.gguf", "-c", "32768"), mapped=()))
    assert out.ctx == 32768
    assert out.spec.context is None
    assert "-c" not in out.spec.extra_args and "32768" not in out.spec.extra_args


def test_a_zero_window_is_the_models_own_and_carries_no_input():
    assert imported(proc(("llama-server", "-m", "/m/a.gguf", "-c", "0"), mapped=())).ctx is None


def test_the_named_environment_fills_a_field_only_when_argv_did_not():
    environ = {"LLAMA_ARG_ALIAS": "from-env", "LLAMA_ARG_CTX_SIZE": "8192", "LLAMA_ARG_CACHE_TYPE_K": "f16",
               "LLAMA_ARG_CACHE_RAM": "256", "LLAMA_ARG_MMPROJ": "/m/env.gguf", "LLAMA_ARG_MODEL": "/m/env-model.gguf"}
    out = imported(proc(("llama-server", "--alias", "from-argv"), environ=environ, mapped=()))
    assert out.spec.served_name == "from-argv"
    assert out.ctx == 8192
    assert out.spec.cache_type_k == "f16"
    assert out.spec.cache_ram == 256
    assert out.spec.mmproj == "/m/env.gguf"
    assert out.spec.model == "/m/env-model.gguf"
    assert out.spec.env is None


def test_every_other_llama_arg_goes_to_env_and_nothing_else_from_environ_is_read():
    environ = {"LLAMA_ARG_N_GPU_LAYERS": "99", "LLAMA_ARG_THREADS": "8", "HF_TOKEN": "hf_decoy",
               "CUDA_VISIBLE_DEVICES": "0", "PATH": "/bin", "LLAMA_CACHE": "/cache"}
    out = imported(proc(("llama-server", "-m", "/m/a.gguf"), environ=environ, mapped=()))
    assert out.spec.env == {"LLAMA_ARG_N_GPU_LAYERS": "99", "LLAMA_ARG_THREADS": "8"}
    assert "CUDA" not in repr(out) and "hf_decoy" not in repr(out)


def test_served_name_falls_back_to_the_model_stem():
    out = imported(proc(("llama-server", "-m", "/m/My Model.Q4.gguf"), mapped=()))
    assert out.spec.served_name == "My-Model.Q4"


def test_the_model_a_relative_path_names_is_joined_to_the_process_cwd():
    out = imported(proc(("llama-server", "-m", "./models/a.gguf", "--mmproj", "p/b.gguf"), cwd="/srv", mapped=()))
    assert out.spec.model == "/srv/models/a.gguf"
    assert out.spec.mmproj == "/srv/p/b.gguf"


def test_a_relative_path_with_no_known_cwd_cannot_be_carried():
    rec = proc(("llama-server", "-m", "models/a.gguf"), cwd=None, mapped=())
    assert attach(rec) == "holds a value a record cannot carry (model)"


# ------------------------------------------------------------------------------- port and binary


def test_the_port_is_argv_then_environment_then_the_listener_then_llama_servers_default():
    base = ("llama-server", "-m", "/m/a.gguf")
    assert imported(proc(base + ("--port", "9001"), environ={"LLAMA_ARG_PORT": "9002"}, mapped=())).spec.port == 9001
    assert imported(proc(base, environ={"LLAMA_ARG_PORT": "9002"}, mapped=()), port=9003).spec.port == 9002
    assert imported(proc(base, mapped=()), port=9003).spec.port == 9003
    assert imported(proc(base, mapped=())).spec.port == 8080
    assert imported(proc(base + ("--port=9004",), mapped=())).spec.port == 9004


@pytest.mark.parametrize("bad", ["abc", "0", "70000", "-1"])
def test_a_port_that_is_not_a_port_cannot_be_carried(bad):
    assert attach(proc(("llama-server", "-m", "/m/a.gguf", "--port", bad), mapped=())) \
        == "holds a value a record cannot carry (port)"


def test_every_port_form_is_gone_from_extra_args():
    argv = ("llama-server", "-m", "/m/a.gguf", "--port", "8081", "--port=8082", "--host", "127.0.0.1")
    assert imported(proc(argv, mapped=())).spec.extra_args == []


def test_the_binary_is_the_exe_when_its_basename_is_argv0s():
    out = imported(proc(("llama-server", "-m", "/m/a.gguf"), exe="/opt/llama/bin/llama-server", mapped=()))
    assert (out.spec.binary, out.binary_source) == ("/opt/llama/bin/llama-server", "process")


def test_a_bare_argv0_goes_through_the_resolver_never_the_process_cwd():
    seen = []

    def resolver(name):
        seen.append(name)
        return "/usr/bin/llama-server"

    rec = proc(("llama-server", "-m", "/m/a.gguf"), exe="/usr/bin/env", cwd="/work", mapped=())
    out = imported(rec, binary_resolver=resolver)
    assert (out.spec.binary, out.binary_source) == ("/usr/bin/llama-server", "path")
    assert seen == ["llama-server"]


def test_an_unresolvable_bare_argv0_leaves_the_binary_unset():
    rec = proc(("llama-server", "-m", "/m/a.gguf"), exe="/usr/bin/env", mapped=())
    out = imported(rec)
    assert out.spec.binary is None and out.binary_source is None


def test_an_argv0_with_a_directory_part_is_joined_to_the_process_cwd():
    rec = proc(("./bin/llama-server", "-m", "/m/a.gguf"), exe=None, cwd="/opt/llama", mapped=())
    out = imported(rec)
    assert (out.spec.binary, out.binary_source) == ("/opt/llama/bin/llama-server", "process")
    absolute = proc(("/abs/llama-server", "-m", "/m/a.gguf"), exe=None, mapped=())
    assert imported(absolute).spec.binary == "/abs/llama-server"
    no_cwd = proc(("bin/llama-server", "-m", "/m/a.gguf"), exe=None, cwd=None, mapped=())
    assert imported(no_cwd).spec.binary is None


def test_a_deleted_exe_is_not_a_binary():
    rec = proc(("llama-server", "-m", "/m/a.gguf"), exe="/opt/llama/bin/llama-server (deleted)", mapped=())
    assert imported(rec, binary_resolver=lambda n: "/usr/bin/llama-server").spec.binary == "/usr/bin/llama-server"


# ------------------------------------------------------------------------------- a record cannot carry


def test_a_value_the_record_refuses_is_one_named_reason_not_a_pydantic_error():
    rec = proc(("llama-server", "-m", "/m/a.gguf", "-a", "two\nlines"), mapped=())
    assert attach(rec) == "holds a value a record cannot carry (served_name)"
    rec = proc(("llama-server", "-m", "/m/a.gguf"), environ={"LLAMA_ARG_thing": "x"}, mapped=())
    assert attach(rec) == "holds a value a record cannot carry (env)"  # llama.cpp names are upper case


def test_a_cache_ram_the_field_cannot_hold_stays_an_argument():
    out = imported(proc(("llama-server", "-m", "/m/a.gguf", "-cram", "-1"), mapped=()))
    assert out.spec.cache_ram is None
    assert out.spec.extra_args == ["-cram", "-1"]


def test_a_dangling_value_flag_cannot_be_carried():
    assert attach(proc(("llama-server", "-m", "/m/a.gguf", "--port"), mapped=())) \
        == "holds a value a record cannot carry (--port)"


def test_a_process_that_is_not_llama_server_is_not_translated():
    assert "does not translate" in attach(proc(("python3", "serve.py"), exe="/usr/bin/python3"))


def test_the_attach_only_tail_is_the_documented_one():
    assert ATTACH_ONLY_SUFFIX == " — attached only, no server record written"
    assert init_import.HF_CACHE_SHOWN == "~/.cache/huggingface/hub"
