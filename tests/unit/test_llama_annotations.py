"""D-27 / D-53: one annotation table marks the options that make a llama-server hand out a shell,
run executables from a directory, or serve a directory, across argv, env and config.ini spellings.
Display only: the exclusion table beside it keeps its 13 rows."""
from __future__ import annotations

import pytest

from localharness.provider import server

SHELL = "(serves a shell-exec tool on the port)"
EXEC_DIR = "(a directory the server runs executables from)"
SERVED = "(served to anyone who reaches the port)"

SHELL_NAMES = (
    "--tools", "-ag", "--agent", "--tools-runtime", "--mcp-servers-config", "--mcp-servers-json",
    "--ui-mcp-proxy", "--webui-mcp-proxy", "--ui_mcp_proxy",
    "LLAMA_ARG_TOOLS", "LLAMA_ARG_AGENT", "LLAMA_ARG_TOOLS_RUNTIME", "LLAMA_ARG_MCP_SERVERS_CONFIG",
    "LLAMA_ARG_MCP_SERVERS_JSON", "LLAMA_ARG_UI_MCP_PROXY",
    "tools", "agent", "tools-runtime", "mcp-servers-config", "mcp-servers-json", "ui-mcp-proxy",
)


@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize("name", SHELL_NAMES)
def test_the_tool_serving_options_carry_the_shell_exec_suffix(name, keyed):
    assert server.llama_annotation(name, keyed=keyed) == SHELL


@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize("name", ["LLAMA_ARG_VIDEO_FFMPEG_DIR", "--video-ffmpeg-dir", "video-ffmpeg-dir"])
def test_the_ffmpeg_directory_carries_the_executables_suffix(name, keyed):
    assert server.llama_annotation(name, keyed=keyed) == EXEC_DIR


@pytest.mark.parametrize("name", ["LLAMA_ARG_STATIC_PATH", "--path", "path"])
def test_the_static_path_is_annotated_on_an_unkeyed_record_only(name):
    assert server.llama_annotation(name, keyed=False) == SERVED
    assert server.llama_annotation(name, keyed=True) is None


@pytest.mark.parametrize("name", ["-ngl", "LLAMA_ARG_N_GPU_LAYERS", "HF_TOKEN", "", "--toolsx", "-"])
def test_other_options_carry_no_annotation(name):
    assert server.llama_annotation(name, keyed=False) is None


def test_letter_case_and_underscores_follow_llama_cpps_own_rule():
    assert server.llama_annotation("--Tools_Runtime", keyed=False) == SHELL
    assert server.llama_annotation("llama_arg_tools", keyed=False) == SHELL
    assert server.llama_annotation("  Mcp_Servers_Json ", keyed=False) == SHELL


def test_the_exclusion_table_is_untouched():
    assert len(server.LLAMA_EXCLUDED_OPTIONS) == 13
    for _, names, _ in server.LLAMA_ANNOTATIONS:
        assert not any(server.excluded_option(n) for n in names)


def test_the_suffix_constants_are_the_table_suffixes():
    assert {row[0] for row in server.LLAMA_ANNOTATIONS} == {
        server.SHELL_EXEC_NOTE, server.EXEC_DIR_NOTE, server.SERVED_DIR_NOTE}
    assert (server.SHELL_EXEC_NOTE, server.EXEC_DIR_NOTE, server.SERVED_DIR_NOTE) == (SHELL, EXEC_DIR, SERVED)
