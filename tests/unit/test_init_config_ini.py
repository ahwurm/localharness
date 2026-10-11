"""INIT-07 (config.ini half): llama-server's own config file is carried masked, shown and recorded, never
with a key, and a TLS or download key in it makes the import attach-only (D-28, D-56).

Every file lives under `tmp_path`; the owner's real ~/.config is never read (`XDG_CONFIG_HOME` and `HOME`
are pointed at the tmp tree), and no key file is ever opened.
"""
from __future__ import annotations

import os
import signal

import pytest

from localharness.cli.init_import import (
    CONFIG_INI_CAP,
    AttachOnly,
    Imported,
    IniImport,
    IniRead,
    import_record,
    mask_config_ini,
    read_config_ini,
    user_config_ini,
)
from localharness.provider.procfs import ProcRecord
from tests.unit.launch_support import record_opens

KEY = "sk-inikeysecret-1"

FULL = (b"[*]\nctx-size = 8192\napi-key = K\nLLAMA_API_KEY = K2\napi-key-file = /k\nhf-token = T\n"
        b"HF_TOKEN = T2\n; comment\nhost = 0.0.0.0\nport = 9000\n")


def proc(argv=("llama-server", "-m", "/m/a.gguf"), *, environ=None, unit="llm.service", cwd="/home/u") -> ProcRecord:
    return ProcRecord(pid=1, uid=os.getuid(), argv=tuple(argv), exe="/opt/llama/bin/llama-server", cwd=cwd,
                      start_time=1, unit=unit, environ=environ, mapped_ggufs=())


def ini_read(text: str | bytes, directory: str = "/cfg/llama.cpp") -> IniRead:
    data = text if isinstance(text, bytes) else text.encode()
    return IniRead(directory, mask_config_ini(data), None)


def run(rec, ini=None, **kw):
    return import_record(rec, source=kw.pop("source", "live"), binary_resolver=lambda n: None, ini=ini, **kw)


def imported(rec, ini=None, **kw) -> Imported:
    out = run(rec, ini, **kw)
    assert isinstance(out, Imported), out
    return out


def attach(rec, ini=None, **kw) -> str:
    out = run(rec, ini, **kw)
    assert isinstance(out, AttachOnly), out
    return out.reason


# ------------------------------------------------------------------------------------ the mask


def test_the_mask_strips_keys_tokens_and_the_bind_and_routes_what_the_import_needs():
    masked = mask_config_ini(FULL)
    assert masked.kept == "[*]\nctx-size = 8192\n; comment\n"
    assert masked.keys == {"api-key": "K", "LLAMA_API_KEY": "K2", "api-key-file": "/k",
                           "host": "0.0.0.0", "port": "9000"}
    assert masked.refused is None
    for secret in ("K2", "T2", "hf-token", "HF_TOKEN", "api-key", "host", "port"):
        assert secret not in masked.kept


@pytest.mark.parametrize("line, key", [("hf-repo = org/x", "hf-repo"), ("ssl-key-file = /k", "ssl-key-file"),
                                       ("LLAMA_ARG_SSL_CERT_FILE=/c", "LLAMA_ARG_SSL_CERT_FILE"),
                                       ("HF-FILE = a.gguf", "HF-FILE")])
def test_a_download_or_tls_key_is_refused_by_name_and_not_kept(line, key):
    masked = mask_config_ini(f"ctx-size = 1\n{line}\nthreads = 4\n".encode())
    assert masked.refused == key
    assert key not in masked.kept and masked.kept == "ctx-size = 1\nthreads = 4\n"


def test_the_first_refused_key_is_the_one_named():
    assert mask_config_ini(b"hf-repo = a/b\nssl-key-file = /k\n").refused == "hf-repo"


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_every_newline_parses_like_lf(newline):
    text = newline.join(["[*]", "ctx-size = 4096", "api-key = K", "port = 9", ""])
    masked = mask_config_ini(text.encode())
    assert masked.kept == "[*]\nctx-size = 4096\n"
    assert masked.keys == {"api-key": "K", "port": "9"}


def test_a_comment_character_ends_the_value_as_llama_cpps_grammar_has_it():
    masked = mask_config_ini(b"api-key = ab#cd ; note\nhost = 0.0.0.0 # all\nport = 80;x\n")
    assert masked.keys == {"api-key": "ab", "host": "0.0.0.0", "port": "80"}


def test_a_key_in_an_inactive_section_is_stripped_but_not_routed():
    masked = mask_config_ini(b"[*]\napi-key = A\n[some-model]\napi-key = B\nhost = 10.0.0.1\nctx-size = 1\n")
    assert masked.keys == {"api-key": "A"}
    assert masked.kept == "[*]\n[some-model]\nctx-size = 1\n"


def test_a_credential_spelled_out_of_llamas_grammar_is_still_never_kept():
    masked = mask_config_ini(b"api_key = S3\nhf_token = S4\nHft = S5\nLLAMA_ARG_API_KEY = S6\nctx-size = 3\n")
    assert masked.kept == "ctx-size = 3\n"
    assert not masked.keys and masked.refused is None


def test_a_credential_valued_key_is_never_kept_and_makes_the_import_attach_only():
    doc = '{"mcpServers":{"x":{"headers":{"Authorization":"Bearer SECRET123"}}}}'
    for key in ("mcp-servers-json", "LLAMA_ARG_MCP_SERVERS_JSON", "some-token", "LLAMA_ARG_X_SECRET"):
        masked = mask_config_ini(f"ctx-size = 3\n{key} = {doc}\n".encode())
        assert masked.kept == "ctx-size = 3\n"
        assert masked.keyshaped == key
        assert attach(proc(), ini_read(f"ctx-size = 3\n{key} = {doc}\n")) \
            == f"holds a key the import cannot place ({key})"
    assert mask_config_ini(b"ctx-size = 3\n").keyshaped is None


def test_text_that_is_not_utf8_masks_to_nothing():
    assert mask_config_ini(b"ctx-size = \xff\xfe\n") == IniImport("", {}, None)


def test_an_empty_or_all_stripped_file_keeps_nothing():
    assert mask_config_ini(b"").kept == ""
    assert mask_config_ini(b"api-key = K\n\n\n").kept == ""


# -------------------------------------------------------------------------------- the capped read


def test_an_absent_file_is_nothing_shown_nothing_recorded(tmp_path):
    result = read_config_ini(tmp_path / "llama.cpp" / "config.ini")
    assert result == IniRead(str(tmp_path / "llama.cpp"), None, None)


def test_the_cap_is_inclusive(tmp_path):
    path = tmp_path / "config.ini"
    path.write_bytes(b"; " + b"x" * (CONFIG_INI_CAP - 3) + b"\n")
    assert path.stat().st_size == CONFIG_INI_CAP
    result = read_config_ini(path)
    assert result.ini is not None and result.line is None


def test_one_byte_over_the_cap_is_not_carried_with_the_one_line(tmp_path):
    path = tmp_path / "config.ini"
    path.write_bytes(b"; " + b"x" * (CONFIG_INI_CAP - 2) + b"\n")
    assert path.stat().st_size == CONFIG_INI_CAP + 1
    result = read_config_ini(path)
    assert result.ini is None
    assert result.line == f"config.ini ({tmp_path}) is over 65536 bytes — not carried"


def test_the_directory_in_the_line_is_shown_with_the_home_prefix_and_escapes(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".config" / "llama\x1bcpp" / "config.ini"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (CONFIG_INI_CAP + 1))
    assert read_config_ini(path).line == "config.ini (~/.config/llama\\x1bcpp) is over 65536 bytes — not carried"


def test_a_file_that_is_not_text_is_not_carried(tmp_path):
    path = tmp_path / "config.ini"
    path.write_bytes(b"ctx-size = \xff\n")
    result = read_config_ini(path)
    assert result.ini is None and result.line == f"config.ini ({tmp_path}) is not UTF-8 text — not carried"


def test_a_fifo_is_treated_as_absent_without_blocking(tmp_path):
    fifo = tmp_path / "config.ini"
    os.mkfifo(fifo)
    signal.alarm(5)
    try:
        result = read_config_ini(fifo)
    finally:
        signal.alarm(0)
    assert result == IniRead(str(tmp_path), None, None)


def test_a_directory_at_the_path_is_treated_as_absent(tmp_path):
    (tmp_path / "config.ini").mkdir()
    assert read_config_ini(tmp_path / "config.ini").ini is None


def test_the_file_is_masked_as_it_is_read(tmp_path):
    path = tmp_path / "config.ini"
    path.write_bytes(FULL)
    assert read_config_ini(path).ini == mask_config_ini(FULL)


def test_the_user_file_follows_xdg_config_home_else_dot_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert user_config_ini() == tmp_path / "xdg" / "llama.cpp" / "config.ini"
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    assert user_config_ini() == tmp_path / ".config" / "llama.cpp" / "config.ini"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert user_config_ini() == tmp_path / ".config" / "llama.cpp" / "config.ini"


# ---------------------------------------------------------------------------- the merge into the import


def test_the_kept_text_is_recorded_and_shown_as_key_value_pairs():
    out = imported(proc(), ini_read("[*]\nctx-size = 8192 ; the window\nthreads = 4\n"))
    assert out.spec.config_ini == "[*]\nctx-size = 8192 ; the window\nthreads = 4\n"
    assert "config.ini (/cfg/llama.cpp): ctx-size=8192, threads=4" in out.shown


def test_an_annotated_ini_key_is_suffixed():
    out = imported(proc(), ini_read("tools = all\npath = /w\n"))
    assert ("config.ini (/cfg/llama.cpp): tools=all (serves a shell-exec tool on the port), "
            "path=/w (served to anyone who reaches the port)") in out.shown
    keyed = imported(proc(("llama-server", "-m", "/m/a.gguf", "--api-key", KEY)), ini_read("path = /w\n"))
    assert "config.ini (/cfg/llama.cpp): path=/w" in keyed.shown


def test_shown_ini_values_escape_control_bytes():
    out = imported(proc(), ini_read("note = a\x1b[2Kb\n"))
    line = next(line for line in out.shown if line.startswith("config.ini"))
    assert "note=a\\x1b[2Kb" in line and all(ord(ch) >= 0x20 for ch in line)


def test_no_ini_key_or_token_reaches_the_record_or_a_shown_line():
    text = "[*]\nctx-size = 8192\napi-key = SECRETK\nhf-token = SECRETT\nHF_TOKEN = SECRETT2\n; comment\n"
    out = imported(proc(), ini_read(text))
    assert out.api_key.get_secret_value() == "SECRETK"
    assert out.spec.config_ini == "[*]\nctx-size = 8192\n; comment\n"
    blob = "\n".join([*out.shown, repr(out), repr(out.spec), out.spec.model_dump_json()])
    for secret in ("SECRETK", "SECRETT", "SECRETT2"):
        assert secret not in blob


def test_an_ini_api_key_alone_is_the_inline_key_path():
    out = imported(proc(), ini_read(f"api-key = {KEY}\nctx-size = 4\n"))
    assert out.api_key.get_secret_value() == KEY and out.spec.require_api_key is True
    blob = "\n".join([*out.shown, repr(out), repr(out.spec), out.spec.model_dump_json()])
    assert KEY not in blob


def test_an_ini_key_beside_an_argv_key_is_more_than_one():
    rec = proc(("llama-server", "-m", "/m/a.gguf", "--api-key", "other"))
    assert attach(rec, ini_read(f"api-key = {KEY}\n")) == "holds more than one --api-key"


def test_an_ini_key_equal_to_the_argv_key_is_one_key():
    rec = proc(("llama-server", "-m", "/m/a.gguf", "--api-key", KEY))
    assert imported(rec, ini_read(f"api-key = {KEY}\n")).api_key.get_secret_value() == KEY


def test_an_ini_key_file_is_the_key_file_when_argv_and_environment_name_none(tmp_path):
    target = tmp_path / "k"
    target.write_text("topsecret\n")
    with record_opens() as opened:
        out = imported(proc(), ini_read(f"api-key-file = {target}\n"))
    assert str(target) not in opened
    assert out.spec.api_key_file == str(target) and out.spec.require_api_key is True
    assert f"key file: {target} (read at start, never by init)" in out.shown


def test_a_relative_ini_key_file_is_joined_to_the_process_cwd():
    out = imported(proc(cwd="/srv"), ini_read("api-key-file = keys/k\n"))
    assert out.spec.api_key_file == "/srv/keys/k"


def test_an_ini_tls_key_is_attach_only():
    assert attach(proc(), ini_read("ssl-key-file = /k.pem\n")) == "serves TLS (ssl-key-file)"
    assert attach(proc(), ini_read("LLAMA_ARG_SSL_CERT_FILE = /c\n")) == "serves TLS (LLAMA_ARG_SSL_CERT_FILE)"


def test_an_ini_download_key_is_the_download_path():
    rec = proc(("llama-server",))
    assert attach(rec, ini_read("hf-repo = org/x\n")) == \
        "a record needs a local path; none mapped under ~/.cache/huggingface/hub"


def test_an_ini_host_is_judged_by_the_matrix(monkeypatch):
    out = imported(proc(), ini_read("host = 0.0.0.0\n"))
    assert out.spec.bind_all is True and "every interface, no key" in out.shown
    assert out.spec.config_ini is None  # the bind lines left the kept text
    assert attach(proc(), ini_read("host = lan.example\n")) == "binds lan.example (a host name)"
    assert imported(proc(), ini_read("host = 127.0.0.1\n")).spec.bind_all is False


def test_argv_and_environment_beat_the_ini_bind_and_port():
    ini = ini_read("host = 0.0.0.0\nport = 9000\n")
    argv = proc(("llama-server", "-m", "/m/a.gguf", "--host", "127.0.0.1", "--port", "7000"))
    out = imported(argv, ini)
    assert out.spec.bind_all is False and out.spec.port == 7000
    env = proc(environ={"LLAMA_ARG_HOST": "127.0.0.1", "LLAMA_ARG_PORT": "7001"})
    out = imported(env, ini)
    assert out.spec.bind_all is False and out.spec.port == 7001
    out = imported(proc(), ini, port=5555)
    assert out.spec.port == 9000  # config.ini applies before the caller's listener port is consulted


def test_an_ini_host_in_a_unit_is_unit_grade():
    out = imported(proc(), ini_read("host = 0.0.0.0\n"), source="unit")
    assert out.spec.bind_all is False
    assert any("llm.service binds 0.0.0.0 with no key file" in line for line in out.shown)


def test_an_over_the_cap_line_is_shown_and_nothing_is_recorded():
    line = "config.ini (/cfg/llama.cpp) is over 65536 bytes — not carried"
    out = imported(proc(), IniRead("/cfg/llama.cpp", None, line))
    assert line in out.shown and out.spec.config_ini is None


def test_an_absent_ini_shows_and_records_nothing():
    out = imported(proc(), IniRead("/cfg/llama.cpp", None, None))
    assert out.spec.config_ini is None and not any(line.startswith("config.ini") for line in out.shown)
    assert imported(proc()).spec.config_ini is None


def test_a_file_with_only_comments_is_recorded_without_a_pairs_line():
    out = imported(proc(), ini_read("; just a note\n"))
    assert out.spec.config_ini == "; just a note\n"
    assert not any(line.startswith("config.ini (") for line in out.shown)


def test_the_ini_shown_line_comes_after_env_and_before_the_key_file_line():
    rec = proc(("llama-server", "-m", "/m/a.gguf", "--api-key-file", "/k", "-ngl", "9"),
               environ={"LLAMA_ARG_N_PARALLEL": "1"})
    out = imported(rec, ini_read("ctx-size = 4\n"))
    heads = [line.split(":")[0].split(" (")[0] for line in out.shown]
    assert heads[:4] == ["flags", "env", "config.ini", "key file"]
