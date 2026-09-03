"""The vLLM lifecycle family resolves GLOBAL, always (v0.13 Standing Invariant, phase 39).

``<global>/vllm/{server.pid,serve.log,venv/}`` is machine-wide: two workspaces must never each
believe they own the one GPU daemon. These sites are value-identical today (nothing feeds them a
workspace path yet), so a pure behavior test cannot discriminate a pinned site from an unpinned
one — same as ``tests/unit/test_bench_config_pinning.py``'s amendment-#3 half. Two kinds of proof:

1. BEHAVIORAL where a seam exists: ``OrchestratorREPL._server_config_dir`` is a real function of
   its input and is tested as one, including the None case; and the whole ``server.py`` path
   family is proven to derive from ``server_dir``, so pinning one input pins all four paths.
2. STRUCTURAL for the rest: the ``start_cmd`` and ``init_cmd`` call sites are asserted to name a
   pinned symbol. Mutation-check before trusting these (38-03's standing rule): revert one site,
   watch the test fail, restore. Done for every structural assertion below when written.

What this file does NOT claim: it fixes no live bug. It is the regression guard for the moment
``resolve_config_dir`` learns to discover a workspace ``.localharness/`` — if a later change makes
a lifecycle site follow that layer, it fails here rather than as two harnesses fighting over one
pidfile.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from localharness.config.paths import global_config_dir

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness" / "cli"


def _repl(config_dir):
    """The minimal OrchestratorREPL the property needs — it only reads self._config_dir."""
    from localharness.cli.repl import OrchestratorREPL

    return OrchestratorREPL(
        orchestrator=None,
        agent_loop=SimpleNamespace(),
        channel=SimpleNamespace(),
        bus=None,
        config_dir=config_dir,
        harness_config=SimpleNamespace(server=None),
    )


# --- Behavioral: the one real seam -------------------------------------------------------- #


def test_repl_server_config_dir_is_global(tmp_path):
    """The property answers with the GLOBAL layer for whatever config dir the session holds."""
    session = _repl(tmp_path)
    assert session._server_config_dir == global_config_dir(tmp_path)


def test_repl_server_config_dir_none_stays_none():
    """None must stay None. ``global_config_dir(None)`` would invent ``~/.localharness`` where
    today an unset config dir reaches the lifecycle call as None — that is a real state here (the
    /model guard checks for it), and the strategies handle it themselves."""
    assert _repl(None)._server_config_dir is None


def test_server_paths_all_derive_from_server_dir(tmp_path):
    """pid/log/venv all hang off ``server_dir``, so pinning its INPUT pins the whole family.

    This is what makes 16 call sites a single guarantee rather than 16 separate ones.
    """
    from localharness.provider import server as managed_server

    root = managed_server.server_dir(tmp_path)
    assert root == Path(tmp_path) / "vllm"
    for path in (
        managed_server.pid_path(tmp_path),
        managed_server.log_path(tmp_path),
        managed_server.venv_vllm_bin(tmp_path),
    ):
        assert root in path.parents, f"{path} escaped server_dir({tmp_path})"


# --- Structural: the sites name the pin (mutation-checked) --------------------------------- #


def test_start_cmd_lifecycle_sites_name_the_global_pin():
    """start's three lifecycle calls read a name derived from the RAW --config-dir."""
    text = (SRC / "start_cmd.py").read_text(encoding="utf-8")
    assert "server_cfg_path = global_config_dir(config_dir)" in text
    # Named in full: the bare substring "harness.server, server_cfg_path" is ALSO satisfied by the
    # activate line below it, so reverting liveness alone left this test green (caught by the
    # mutation check — the reason that check is mandatory before trusting a structural assertion).
    assert "_managed_server_running(strategy, harness.server, server_cfg_path)" in text
    assert "_managed_server_running(strategy, harness.server, cfg_path)" not in text
    assert "activate(harness.server, server_cfg_path, provider.base_url)" in text  # launch
    assert "wait_ready(provider.base_url, config_dir=server_cfg_path)" in text
    assert "wait_ready(provider.base_url, config_dir=cfg_path)" not in text
    # The helper states which layer it expects, so a caller cannot hand it a workspace by accident.
    assert "def _managed_server_running(strategy: Any, srv: Any, global_dir: Path)" in text


def test_repl_lifecycle_sites_name_the_global_pin():
    """Seven lifecycle calls go through the property; the non-lifecycle uses stay put.

    The counts are the assertion: moving an eighth site onto the property (or leaving a lifecycle
    call on the raw attribute) changes one of them.
    """
    text = (SRC / "repl.py").read_text(encoding="utf-8")
    assert text.count("self._server_config_dir") == 7
    # assignment + the property's own two reads on one line + 6 deliberately-untouched uses
    assert text.count("self._config_dir") == 9
    for raw in (
        "free_accelerator(self._active_heavy, ep.lifecycle, self._config_dir)",
        "strategy.stop(managed, self._config_dir)",
        "strategy_for(spec).activate(spec, self._config_dir, base_url)",
    ):
        assert raw not in text, f"lifecycle call still on the raw attribute: {raw}"
    # ...and the non-lifecycle writes are untouched: agent creation, endpoint + default-model
    # persistence keep following whatever layer the session reads (phase 41 owns that question).
    assert text.count("config_dir=self._config_dir") == 3


def test_init_guided_setup_names_the_global_pin():
    """The guided vLLM install/launch names the global layer at all six of its sites."""
    text = (SRC / "init_cmd.py").read_text(encoding="utf-8")
    assert "server_config_path = global_config_dir(config_path)" in text
    for pinned in (
        "managed_server.find_vllm(server_config_path)",
        "managed_server.server_dir(server_config_path)",
        "managed_server.install_vllm_venv(server_config_path",
        "managed_server.log_path(server_config_path)",
        "managed_server.start_server(server_config_path, cmd)",
        "wait_ready(base_url, config_dir=server_config_path)",
    ):
        assert pinned in text, f"guided setup lost its pin: {pinned}"
    assert "managed_server.start_server(config_path" not in text
    assert "managed_server.find_vllm(config_path)" not in text
