"""Ingest-via-exec gate: an agent DENIED the web verbs may not fetch remote content through
bash_exec/python_exec instead (owner ruling 2026-09-17 — the live bypass was the root agent
running a ddgs search through bash after its delegation came back empty).

Covers the pure predicate (block / pass / the deliberate package-op carve-out / false-positive
pins), the designation bypass (has_ingest), the floor kill-switch, and the registry dispatch
chokepoint — where the blocked path proves the command NEVER RAN (marker file absent) and the
allowed path proves it really did (marker file present). The gate is a REDIRECT, not a sandbox
(obfuscation defeats it by design); these tests grade the redirect, not a containment claim.
"""
from __future__ import annotations

import pytest

from localharness.config.models import ToolConfig
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.capabilities import (
    EXEC_TOOLS,
    HOST_DANGEROUS,
    IngestViaExecError,
    apply_root_capability_floor,
    assert_no_ingest_via_exec,
    set_floor_enabled,
)
from localharness.tools.registry import ToolRegistry


# --- Membership pins ------------------------------------------------------

def test_exec_tools_is_the_fetch_capable_subset():
    # write/edit touch the host but FETCH nothing — gating them here would block authoring a
    # script that merely mentions curl. Execution surface only.
    assert EXEC_TOOLS == {"bash_exec", "python_exec"}
    assert EXEC_TOOLS < HOST_DANGEROUS


# --- Pure predicate: blocked ---------------------------------------------

@pytest.mark.parametrize(
    "tool,args",
    [
        ("bash_exec", {"command": "curl -s https://example.com/page"}),
        ("bash_exec", {"command": "wget https://example.com/f.tar.gz"}),
        ("bash_exec", {"command": "ddgs text -q 'best local llm'"}),  # the live 2026-09-17 vector
        ("bash_exec", {"command": "echo $(curl http://x)"}),          # command substitution
        ("bash_exec", {"command": "exec 3<>/dev/tcp/example.com/80"}),
        ("bash_exec", {"command": "openssl s_client -connect example.com:443"}),
        ("bash_exec", {"command": "CURL http://x"}),                  # case-insensitive
        ("bash_exec", {"command": "lynx -dump http://x"}),            # suffix guard keeps real invocations
        ("bash_exec", {"command": "curl http://x", "timeout": 30}),   # non-str values ignored
        ("python_exec", {"code": "import requests\nrequests.get('http://x')"}),
        ("python_exec", {"code": "from urllib.request import urlopen"}),
        ("python_exec", {"code": "s = aiohttp.ClientSession()"}),
    ],
)
def test_predicate_blocks_ingest_shapes(tool, args):
    with pytest.raises(IngestViaExecError) as exc_info:
        assert_no_ingest_via_exec(tool, args, agent_id="root", has_ingest=False)
    msg = str(exc_info.value)
    assert "DELEGATE" in msg, "the block must redirect to delegation, not just refuse"
    assert "web-researcher" in msg


def test_predicate_blocks_non_dict_arguments():
    with pytest.raises(IngestViaExecError):
        assert_no_ingest_via_exec("python_exec", "import requests", has_ingest=False)


# --- Pure predicate: passes ----------------------------------------------

@pytest.mark.parametrize(
    "tool,args",
    [
        # The deliberate carve-out: package/VCS/registry ops are not content ingestion.
        ("bash_exec", {"command": "pip install requests"}),
        ("bash_exec", {"command": "uv sync --extra dev"}),  # 'nc' inside 'sync' must not trip
        ("bash_exec", {"command": "git clone https://github.com/x/y.git"}),
        ("bash_exec", {"command": "apt-get install -y jq"}),
        # False-positive pins: word boundaries hold.
        ("bash_exec", {"command": "echo 'curling iron' && grep wget-log notes.txt"}),
        ("python_exec", {"code": "print('requests are welcome')"}),
        # A bare URL with no fetch verb is data, not ingestion.
        ("bash_exec", {"command": "echo https://example.com >> links.txt"}),
        # 'links' the browser is a NAMED residual (everyday English word — dropped from the
        # pattern); prose and filenames using it must never trip the gate.
        ("bash_exec", {"command": "echo 'useful links: see the docs'"}),
    ],
)
def test_predicate_passes_benign_shapes(tool, args):
    assert_no_ingest_via_exec(tool, args, agent_id="root", has_ingest=False)  # must not raise


def test_designated_ingester_is_untouched():
    # An agent that HOLDS a web verb is a designated ingester — the gate is not for it.
    assert_no_ingest_via_exec(
        "bash_exec", {"command": "curl https://example.com"}, has_ingest=True
    )


def test_non_exec_tools_are_out_of_scope():
    # Authoring a file that MENTIONS curl is not fetching; this gate covers execution only.
    assert_no_ingest_via_exec(
        "write", {"path": "fetch.sh", "content": "curl https://example.com"}, has_ingest=False
    )


def test_floor_disabled_disables_the_gate():
    with pytest.warns(UserWarning):
        set_floor_enabled(False)
    try:
        assert_no_ingest_via_exec(
            "bash_exec", {"command": "curl https://example.com"}, has_ingest=False
        )
    finally:
        set_floor_enabled(True)


# --- Chokepoint: registry.dispatch ---------------------------------------

async def _root_registry_and_cfg():
    """The real topology the gate exists for: a root agent floor-stripped of the web verbs,
    still holding bash — built with the exact call cli/start_cmd.py makes."""
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    cfg = ToolConfig()
    apply_root_capability_floor(cfg, enabled=True)
    return reg, cfg


@pytest.mark.asyncio
async def test_dispatch_blocks_before_the_command_runs(tmp_path):
    reg, cfg = await _root_registry_and_cfg()
    marker = tmp_path / "ran"
    result = await reg.dispatch(
        "bash_exec",
        {"command": f"touch {marker}; curl -s http://127.0.0.1:9/"},
        "root", "", cfg,
    )
    assert result.success is False
    assert result.error_type == "permission_denied"
    assert "holds no web-ingest tool" in (result.error or "")
    assert not marker.exists(), "BLOCKED must mean never executed — the gate fired after the shell ran"


@pytest.mark.asyncio
async def test_dispatch_carve_out_reaches_real_execution(tmp_path):
    reg, cfg = await _root_registry_and_cfg()
    marker = tmp_path / "ran"
    result = await reg.dispatch(
        "bash_exec",
        {"command": f"echo pip install requests > {marker}"},
        "root", "", cfg,
    )
    assert "web-ingest" not in (result.error or ""), "the package-op carve-out regressed"
    assert marker.exists(), "the allowed path must actually execute"
