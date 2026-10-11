"""INIT-05 / INIT-06 / INIT-07 / INIT-10 (translation half): one pure function turns a llama-server process
record (live, journal or unit file) into a validated record, or into the one reason it is attached to only.

The records here are built by hand; nothing reads a real /proc, runs systemctl, opens a key file or
touches ~/.localharness (D-18, D-43). Every quoted line is asserted verbatim (D-45).
"""
from __future__ import annotations

import os
import socket

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
from tests.unit.gguf_support import write_gguf
from tests.unit.launch_support import record_opens

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


@pytest.mark.parametrize("value", ["abc", "-1", "9" * 5000, "١٢٣", "1.5", "1" * 10])
def test_a_cache_ram_in_the_environment_the_field_cannot_hold_stays_a_variable_and_never_raises(value):
    out = imported(llama(environ={"LLAMA_ARG_CACHE_RAM": value}))
    assert out.spec.cache_ram is None
    assert out.spec.env == {"LLAMA_ARG_CACHE_RAM": value}


def test_a_cache_ram_figure_in_the_environment_fills_the_field_and_argv_wins():
    assert imported(llama(environ={"LLAMA_ARG_CACHE_RAM": "512"})).spec.cache_ram == 512
    out = imported(llama("-cram", "64", environ={"LLAMA_ARG_CACHE_RAM": "-1"}))
    assert out.spec.cache_ram == 64 and out.spec.env is None


def test_an_over_long_cache_ram_on_argv_stays_an_argument():
    out = imported(llama("--cache-ram", "9" * 5000))
    assert out.spec.cache_ram is None and out.spec.extra_args == ["--cache-ram", "9" * 5000]


def test_a_value_no_step_could_place_is_attach_only_not_a_traceback(monkeypatch):
    def boom(_text):
        raise ValueError("Exceeds the limit")

    monkeypatch.setattr(init_import, "_plain_ram", lambda text: True)
    monkeypatch.setattr(init_import, "_int", boom)
    assert attach(llama("--port", "9", "-c", "9")) == "holds a value a record cannot carry (record)"


def test_a_dangling_value_flag_cannot_be_carried():
    assert attach(proc(("llama-server", "-m", "/m/a.gguf", "--port"), mapped=())) \
        == "holds a value a record cannot carry (--port)"


def test_a_process_that_is_not_llama_server_is_not_translated():
    assert "does not translate" in attach(proc(("python3", "serve.py"), exe="/usr/bin/python3"))


def test_the_attach_only_tail_is_the_documented_one():
    assert ATTACH_ONLY_SUFFIX == " — attached only, no server record written"
    assert init_import.HF_CACHE_SHOWN == "~/.cache/huggingface/hub"


# ================================================================================ Task 2
#
# Keys, binding, TLS, download identity, --verbose, the unit-file grade and every attach-only reason.

KEY = "sk-verysecretvalue-123"


def llama(*args, **kw) -> ProcRecord:
    """A llama-server record with `-m /m/a.gguf` and the given arguments."""
    kw.setdefault("mapped", ())
    return proc(("llama-server", "-m", "/m/a.gguf", *args), **kw)


def everywhere(out: Imported) -> str:
    """Every place a key could leak to: shown lines, the record (dump, JSON, repr), the import's repr."""
    return "\n".join([*out.shown, repr(out), repr(out.spec), str(out.spec.model_dump()),
                      out.spec.model_dump_json()])


# ------------------------------------------------------------------------------------------ keys


@pytest.mark.parametrize("flag", ["--api-key", "--api_key"])
def test_an_inline_argv_key_goes_to_the_key_path_and_nowhere_else(flag):
    out = imported(llama(flag, KEY, "-ngl", "9"))
    assert out.api_key is not None and out.api_key.get_secret_value() == KEY
    assert out.spec.require_api_key is True
    assert out.spec.api_key_file is None
    assert out.spec.extra_args == ["-ngl", "9"]
    assert KEY not in everywhere(out)


def test_the_equals_form_of_the_key_is_stripped_too():
    out = imported(llama(f"--api-key={KEY}"))
    assert out.api_key.get_secret_value() == KEY
    assert KEY not in everywhere(out) and out.spec.extra_args == []


def test_the_environment_key_is_the_same_path():
    out = imported(llama(environ={"LLAMA_API_KEY": KEY}))
    assert out.api_key.get_secret_value() == KEY and out.spec.require_api_key is True
    assert out.spec.env is None
    assert KEY not in everywhere(out)


def test_the_same_key_twice_is_one_key():
    out = imported(llama("--api-key", KEY, environ={"LLAMA_API_KEY": KEY}))
    assert out.api_key.get_secret_value() == KEY


@pytest.mark.parametrize("rec", [
    llama("--api-key", "A,B"),
    llama("--api-key", "A", environ={"LLAMA_API_KEY": "B"}),
    llama("--api-key", "A", "--api-key", "B"),
])
def test_more_than_one_key_is_attach_only(rec):
    assert attach(rec) == "holds more than one --api-key"


def test_a_key_file_is_recorded_joined_to_the_process_cwd_and_never_opened(tmp_path):
    target = tmp_path / "keys" / "k"
    target.parent.mkdir()
    target.write_text("topsecret\n")
    with record_opens() as opened:
        out = imported(llama("--api-key-file", "keys/k", cwd=str(tmp_path)))
        env_out = imported(llama(environ={"LLAMA_ARG_API_KEY_FILE": "keys/k"}, cwd=str(tmp_path)))
    assert str(target) not in opened
    for one in (out, env_out):
        assert one.spec.api_key_file == str(target)
        assert one.spec.require_api_key is True
        assert one.api_key is None
        assert f"key file: {target} (read at start, never by init)" in one.shown
        assert one.spec.extra_args == [] and one.spec.env is None
        assert "topsecret" not in everywhere(one)


def test_the_key_file_line_is_verbatim_for_an_absolute_path():
    out = imported(llama("--api-key-file", "/srv/keys/k"))
    assert "key file: /srv/keys/k (read at start, never by init)" in out.shown


def test_llama_arg_api_key_and_vllm_api_key_are_stripped_and_dropped():
    out = imported(llama(environ={"LLAMA_ARG_API_KEY": "secret-x", "VLLM_API_KEY": "secret-y"}))
    assert out.api_key is None and out.spec.require_api_key is False and out.spec.env is None
    assert "secret-x" not in everywhere(out) and "secret-y" not in everywhere(out)


def test_an_inline_key_beside_a_key_file_cannot_be_placed():
    assert attach(llama("--api-key", KEY, "--api-key-file", "/k")) \
        == "holds a key the import cannot place (--api-key and --api-key-file)"


def test_two_different_key_files_are_more_than_one_key():
    rec = llama("--api-key-file", "/a", environ={"LLAMA_ARG_API_KEY_FILE": "/b"})
    assert attach(rec) == "holds more than one --api-key"
    assert imported(llama("--api-key-file", "/a", environ={"LLAMA_ARG_API_KEY_FILE": "/a"})).spec.api_key_file == "/a"


def test_a_relative_key_file_with_no_known_cwd_cannot_be_placed():
    assert attach(llama("--api-key-file", "keys/k", cwd=None)) \
        == "holds a key the import cannot place (--api-key-file)"
    assert attach(llama(environ={"LLAMA_ARG_API_KEY_FILE": "k"}, cwd=None)) \
        == "holds a key the import cannot place (LLAMA_ARG_API_KEY_FILE)"


def test_anything_else_key_shaped_is_attach_only():
    assert attach(llama("--api-key-extra", "x")) == "holds a key the import cannot place (--api-key-extra)"
    assert attach(llama("--api_key_extra=x")) == "holds a key the import cannot place (--api_key_extra)"
    assert attach(llama(environ={"LLAMA_ARG_API_KEY_BACKUP": "x"})) \
        == "holds a key the import cannot place (LLAMA_ARG_API_KEY_BACKUP)"


def test_a_value_that_merely_mentions_api_key_is_not_key_shaped():
    out = imported(llama("--chat-template-file", "/etc/api-key.jinja"))
    assert out.spec.extra_args == ["--chat-template-file", "/etc/api-key.jinja"]


def test_a_hugging_face_token_on_argv_is_dropped_never_shown():
    out = imported(llama("-hft", "hf_argvsecret", "--hf_token=hf_other", "-ngl", "9"))
    assert out.spec.extra_args == ["-ngl", "9"]
    assert "hf_argvsecret" not in everywhere(out) and "hf_other" not in everywhere(out)


MCP_JSON = '{"mcpServers":{"x":{"url":"https://h","headers":{"Authorization":"Bearer SECRET123"}}}}'


def test_an_mcp_servers_document_is_a_credential_never_shown_or_recorded():
    # D-27 only annotates the flag; the value can embed headers/env tokens, so D-19(2) refuses it
    reason = attach(llama("--mcp-servers-json", MCP_JSON))
    assert reason == "holds a key the import cannot place (--mcp-servers-json)"
    assert "SECRET123" not in reason
    assert attach(llama(f"--mcp_servers_json={MCP_JSON}")) \
        == "holds a key the import cannot place (--mcp_servers_json)"
    assert attach(llama(environ={"LLAMA_ARG_MCP_SERVERS_JSON": MCP_JSON})) \
        == "holds a key the import cannot place (LLAMA_ARG_MCP_SERVERS_JSON)"


def test_the_path_form_of_the_mcp_option_is_still_imported_and_annotated():
    out = imported(llama("--mcp-servers-config", "/etc/mcp.json"))
    assert out.spec.extra_args == ["--mcp-servers-config", "/etc/mcp.json"]
    assert "(serves a shell-exec tool on the port)" in " ".join(out.shown)


@pytest.mark.parametrize("name", [
    "LLAMA_ARG_HF_TOKEN", "LLAMA_ARG_SOME_SECRET", "LLAMA_ARG_PASSWORD", "LLAMA_ARG_PASSWD",
    "LLAMA_ARG_AUTH_HEADER", "LLAMA_ARG_CREDENTIALS", "LLAMA_ARG_ADMIN_KEY",
])
def test_a_llama_arg_name_that_reads_as_a_credential_is_attach_only_without_its_value(name):
    rec = llama(environ={name: "hf_SECRET"})
    assert attach(rec) == f"holds a key the import cannot place ({name})"


def test_a_single_dash_equals_token_is_split_like_the_double_dash_form():
    out = imported(llama("-hft=hf_SECRET", "-ngl", "9"))
    assert out.spec.extra_args == ["-ngl", "9"]
    assert "hf_SECRET" not in everywhere(out)
    out = imported(llama("-ctk=q8_0"))
    assert out.spec.cache_type_k == "q8_0" and not out.spec.extra_args
    assert attach(llama("-mu=https://user:tok@host/m.gguf")) \
        == "a record needs a local path; none mapped under ~/.cache/huggingface/hub"
    assert attach(llama("-secret-thing=abc")) == "holds a key the import cannot place (-secret-thing)"


# ------------------------------------------------------------------------------------ the bind


def test_every_interface_without_a_key_shows_the_exposure_phrase():
    out = imported(llama("--host", "0.0.0.0"))
    assert out.spec.bind_all is True
    assert "every interface, no key" in out.shown


def test_a_keyed_wide_bind_shows_no_exposure_phrase_but_names_the_widening():
    out = imported(llama("--host", "0.0.0.0", "--api-key", KEY))
    assert out.spec.bind_all is True
    assert "every interface, no key" not in out.shown
    assert "server.bind_all: true (--host 0.0.0.0)" in out.shown


def test_a_keyed_by_file_wide_bind_names_the_widening_too():
    out = imported(llama("--host", "0.0.0.0", "--api-key-file", "/k"))
    assert out.spec.bind_all is True and "server.bind_all: true (--host 0.0.0.0)" in out.shown


def test_the_exposure_phrase_is_judged_for_a_declared_only_record_too():
    rec = llama("--host", "0.0.0.0", exe="/usr/bin/env")
    out = imported(rec)
    assert out.spec.binary is None and out.spec.bind_all is True
    assert "every interface, no key" in out.shown


@pytest.mark.parametrize("host", ["192.168.1.5", "::", "10.0.0.1,127.0.0.1"])
def test_an_address_beyond_loopback_widens(host):
    assert imported(llama("--host", host)).spec.bind_all is True


@pytest.mark.parametrize("host", ["127.0.0.1,::1", "127.0.0.2", "::ffff:127.0.0.1", "localhost"])
def test_loopback_spellings_import_nothing(host):
    out = imported(llama("--host", host))
    assert out.spec.bind_all is False
    assert not any("server.bind_all" in line or "every interface" in line for line in out.shown)


def test_a_host_name_is_attach_only_and_never_resolved(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a name must never be resolved")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "gethostbyname", boom)
    assert attach(llama("--host", "ip6-localhost")) == "binds ip6-localhost (a host name)"
    assert attach(llama("--host", "127.0.0.1,my.box.lan")) == "binds my.box.lan (a host name)"


def test_a_unix_socket_is_attach_only():
    assert attach(llama("--host", "/run/llama.sock")) == "binds /run/llama.sock (a UNIX socket)"


def test_the_bind_can_come_from_the_environment_alone():
    assert imported(llama(environ={"LLAMA_ARG_HOST": "0.0.0.0"})).spec.bind_all is True
    assert imported(llama("--host", "127.0.0.1", environ={"LLAMA_ARG_HOST": "0.0.0.0"})).spec.bind_all is False


def test_the_last_host_flag_wins():
    assert imported(llama("--host", "0.0.0.0", "--host", "127.0.0.1")).spec.bind_all is False
    assert imported(llama("--host", "127.0.0.1", "--host=0.0.0.0")).spec.bind_all is True


@pytest.mark.parametrize("args", [("--host", "0.0.0.0", "--port", "9"), ("--host=0.0.0.0", "--port=9")])
def test_every_host_and_port_form_leaves_extra_args(args):
    assert imported(llama(*args, "-ngl", "9")).spec.extra_args == ["-ngl", "9"]


def test_an_empty_host_cannot_be_carried():
    assert attach(llama("--host", " , ")) == "holds a value a record cannot carry (host)"


# --------------------------------------------------------------------- unit-file grade (D-23)

UNIT_WIDE = ("llm.service binds 0.0.0.0 with no key file — imported on 127.0.0.1 · "
             "server.bind_all: true beside server.api_key_file restores it (asks once)")


@pytest.mark.parametrize("source", ["journal", "unit"])
def test_a_wide_bind_in_a_unit_with_no_key_file_imports_on_loopback_with_the_undo_line(source):
    out = imported(llama("--host", "0.0.0.0", unit="llm.service", cwd=None), source=source)
    assert out.spec.bind_all is False
    assert UNIT_WIDE in out.shown
    assert "every interface, no key" not in out.shown


@pytest.mark.parametrize("source", ["journal", "unit"])
def test_a_key_file_on_argv_restores_the_wide_bind(source):
    out = imported(llama("--host", "0.0.0.0", "--api-key-file", "/k", unit="llm.service"), source=source)
    assert out.spec.bind_all is True
    assert UNIT_WIDE not in out.shown


def test_a_unit_that_names_a_key_in_its_environment_keeps_the_wide_bind():
    out = imported(llama("--host", "0.0.0.0", unit="llm.service", environ={"LLAMA_API_KEY": KEY}), source="unit")
    assert out.spec.bind_all is True and out.api_key.get_secret_value() == KEY
    assert not any("with no key file" in line for line in out.shown)


def test_a_journal_record_with_an_inline_key_still_imports_on_loopback():
    out = imported(llama("--host", "0.0.0.0", "--api-key", KEY, unit="llm.service", cwd=None), source="journal")
    assert out.spec.bind_all is False and UNIT_WIDE in out.shown


def test_a_live_process_has_no_unit_grade():
    assert imported(llama("--host", "0.0.0.0", unit="llm.service"), source="live").spec.bind_all is True


def test_the_unit_line_shows_escapes_not_control_bytes():
    out = imported(llama("--host", "0.0.0.0", unit="a\x1b[2Kb.service"), source="unit")
    assert any(line.startswith("a\\x1b[2Kb.service binds 0.0.0.0") for line in out.shown)


# ------------------------------------------------------------------------------------------- TLS


def test_tls_on_argv_is_attach_only_naming_the_flag_as_given():
    assert attach(llama("--ssl-key-file", "/k.pem")) == "serves TLS (--ssl-key-file)"
    assert attach(llama("--ssl_cert_file=/c.pem")) == "serves TLS (--ssl_cert_file)"


def test_tls_in_the_environment_is_attach_only_naming_the_variable():
    assert attach(llama(environ={"LLAMA_ARG_SSL_CERT_FILE": "/c.pem"})) == "serves TLS (LLAMA_ARG_SSL_CERT_FILE)"


def test_tls_is_judged_before_everything_else():
    rec = llama("--ssl-key-file", "/k.pem", "--api-key", "A,B", "--host", "my.lan")
    assert attach(rec) == "serves TLS (--ssl-key-file)"


# ---------------------------------------------------------------------------- download identity


def mapped_model(tmp_path, name="abc", arch="llama"):
    blob = tmp_path / "models--org--repo" / "blobs" / name
    blob.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(blob, {"general.architecture": arch, "general.type": "model", f"{arch}.embedding_length": 64})
    return str(blob)


def test_a_download_identity_resolves_to_the_mapped_gguf(tmp_path):
    blob = mapped_model(tmp_path)
    rec = proc(("llama-server", "-hf", "org/repo", "-hff", "Q4.gguf"), mapped=(blob,))
    out = imported(rec)
    assert out.spec.model == blob
    assert out.spec.served_name == "Q4"
    assert out.spec.extra_args == []
    assert not any("org/repo" in line for line in out.shown)


def test_the_alias_beats_the_hugging_face_file_stem(tmp_path):
    blob = mapped_model(tmp_path)
    out = imported(proc(("llama-server", "-hf", "org/repo", "-hff", "Q4.gguf", "-a", "mine"), mapped=(blob,)))
    assert out.spec.served_name == "mine"


def test_with_no_hugging_face_file_the_name_comes_from_the_mapped_model(tmp_path):
    blob = mapped_model(tmp_path, name="deadbeef")
    out = imported(proc(("llama-server", "--hf-repo", "org/repo"), mapped=(blob,)))
    assert out.spec.served_name == "deadbeef"


def test_the_download_environment_names_are_the_same_identity(tmp_path):
    blob = mapped_model(tmp_path)
    out = imported(proc(("llama-server",), environ={"LLAMA_ARG_HF_REPO": "org/repo", "LLAMA_ARG_HF_FILE": "Z.gguf"},
                        mapped=(blob,)))
    assert (out.spec.model, out.spec.served_name, out.spec.env) == (blob, "Z", None)


def test_a_download_beside_a_local_model_still_serves_what_is_mapped(tmp_path):
    blob = mapped_model(tmp_path)
    out = imported(proc(("llama-server", "-m", "/elsewhere/x.gguf", "-mu", "https://h/x.gguf"), mapped=(blob,)))
    assert out.spec.model == blob and out.spec.extra_args == []


def test_a_projector_is_not_the_served_model(tmp_path):
    projector = tmp_path / "blobs" / "p"
    projector.parent.mkdir()
    write_gguf(projector, {"general.architecture": "clip", "general.type": "mmproj"})
    rec = proc(("llama-server", "-hf", "org/repo"), mapped=(str(projector),))
    assert attach(rec) == "a record needs a local path; none mapped under ~/.cache/huggingface/hub"


def test_no_mapped_path_is_attach_only_and_no_mmap_adds_its_remark():
    plain = proc(("llama-server", "-hf", "org/repo"), mapped=())
    assert attach(plain) == "a record needs a local path; none mapped under ~/.cache/huggingface/hub"
    nommap = proc(("llama-server", "-hf", "org/repo", "--no-mmap"), mapped=())
    assert attach(nommap) == ("a record needs a local path; none mapped under ~/.cache/huggingface/hub "
                              "(--no-mmap maps nothing)")
    assert "(--no-mmap" not in attach(proc(("llama-server", "-hf", "org/repo", "--no-mmap-x"), mapped=()))


def test_a_process_with_no_model_at_all_uses_the_mapped_one_too(tmp_path):
    blob = mapped_model(tmp_path)
    assert imported(proc(("llama-server", "--jinja"), mapped=(blob,))).spec.model == blob
    assert attach(proc(("llama-server", "--jinja"), mapped=())).startswith("a record needs a local path")


# --------------------------------------------------------------------------------------- --verbose


@pytest.mark.parametrize("flag", ["-v", "--verbose", "--log-verbose", "--log_verbose"])
def test_verbose_is_dropped_from_a_keyed_record_with_its_line(flag):
    out = imported(llama(flag, "--api-key", KEY, "-ngl", "9"))
    assert out.spec.extra_args == ["-ngl", "9"]
    assert ("--verbose dropped: llama-server logs the key's last four characters; "
            "add it under server.extra_args to keep the log") in out.shown


def test_verbose_is_dropped_when_the_key_is_a_key_file_too():
    out = imported(llama("--verbose", "--api-key-file", "/k"))
    assert out.spec.extra_args == [] and any(line.startswith("--verbose dropped") for line in out.shown)


@pytest.mark.parametrize("flag", ["-v", "--verbose", "--log-verbose"])
def test_verbose_stays_on_an_unkeyed_record(flag):
    out = imported(llama(flag, "-ngl", "9"))
    assert out.spec.extra_args == [flag, "-ngl", "9"]
    assert not any("--verbose dropped" in line for line in out.shown)


def test_a_verbosity_level_is_not_the_verbose_flag():
    out = imported(llama("--api-key", KEY, "-lv", "3"))
    assert out.spec.extra_args == ["-lv", "3"]


# --------------------------------------------------------------------------------- annotations


def test_an_annotated_environment_name_is_suffixed_and_nothing_is_refused():
    out = imported(llama(environ={"LLAMA_ARG_TOOLS": "all"}))
    assert out.spec.env == {"LLAMA_ARG_TOOLS": "all"}
    assert "env: LLAMA_ARG_TOOLS=all (serves a shell-exec tool on the port)" in out.shown


def test_an_annotated_flag_is_suffixed_after_the_copyable_flags():
    out = imported(llama("--tools", "all"))
    assert out.spec.extra_args == ["--tools", "all"]
    assert "flags: --tools all · --tools (serves a shell-exec tool on the port)" in out.shown
    out = imported(llama("--webui-mcp-proxy", "-ngl", "9"))
    assert "flags: --webui-mcp-proxy -ngl 9 · --webui-mcp-proxy (serves a shell-exec tool on the port)" in out.shown


def test_the_served_directory_suffix_is_for_an_unkeyed_record_only():
    unkeyed = imported(llama(environ={"LLAMA_ARG_STATIC_PATH": "/w"}))
    assert "env: LLAMA_ARG_STATIC_PATH=/w (served to anyone who reaches the port)" in unkeyed.shown
    keyed = imported(llama("--api-key", KEY, environ={"LLAMA_ARG_STATIC_PATH": "/w"}))
    assert "env: LLAMA_ARG_STATIC_PATH=/w" in keyed.shown


# -------------------------------------------------------------------------------------- D-45


def test_control_bytes_in_an_argument_print_as_escapes():
    out = imported(llama("-ngl", "\x1b[2K", "--note", "a\nb", environ={"LLAMA_ARG_X": "v\x1b[0m"}))
    flags = next(line for line in out.shown if line.startswith("flags:"))
    assert "\\x1b[2K" in flags and "a\\x0ab" in flags
    assert "LLAMA_ARG_X=v\\x1b[0m" in next(line for line in out.shown if line.startswith("env:"))
    for line in out.shown:
        assert all(ord(ch) >= 0x20 for ch in line), line


def test_a_reason_naming_an_argument_escapes_it():
    assert attach(llama("--host", "bad\x1b[2Kname")) == "binds bad\\x1b[2Kname (a host name)"


# -------------------------------------------------------------------------- other runtimes


def test_ollama_and_lm_studio_cannot_carry_a_key():
    assert attach(proc(("ollama", "serve"), exe="/usr/bin/ollama")) == "runs Ollama, which cannot carry a key"
    assert attach(proc(("lms", "server", "start"), exe="/usr/bin/lms")) \
        == "runs LM Studio, which cannot carry a key"


def test_the_import_source_table_is_the_servers_not_a_second_list():
    import inspect

    text = inspect.getsource(init_import)
    assert "excluded_option(" in text and "classify_hosts(" in text and "effective_posture(" in text
    assert "getaddrinfo" not in text and "gethostbyname" not in text
